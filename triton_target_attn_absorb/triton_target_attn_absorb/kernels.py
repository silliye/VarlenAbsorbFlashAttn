"""
Triton kernels for packed-varlen Ragged Target Attention with matrix absorption (v1).

Scope:
  - packed varlen, decoupled cu_seqlens_q / cu_seqlens_k
  - BlockDiagonalMask (segment-wise full, no causal inside segment)
  - input/output dtype: fp32 (default) or bf16 (business path); same kernel
  - internal accumulators always fp32
  - W_V absorbed into the K-block streaming loop, αX = p @ X never lands in HBM
  - kernel does NOT touch W_Q / W_K — outer Python computes u = q_raw @ W_Q @ W_K^T
    as a plain GEMM (autograd handles dq_raw / dW_Q / dW_K automatically)

Layout convention (kernel-internal, all "last-dim contiguous"):
  u    : [T_q, H, D_KV]   strides (H*D_KV, D_KV, 1)
  X    : [T_k, D_KV]      strides (D_KV, 1)             ← no H dim, shared across heads
  W_V  : [D_KV, H, D_H]   strides (H*D_H, D_H, 1)
  O    : [T_q, H, D_H]    strides (H*D_H, D_H, 1)
  LSE  : [T_q, H]         strides (H, 1)                ← token-major, fp32 log2 domain

LSE: stored in log2 domain (lse = m + log2(sum exp2(s - m))). Bwd kernels
  consume the same convention so the natural-domain exponent cancels.

Math (per segment, per head; head superscript suppressed):
  fwd:
    s   = u @ X^T * (scale / ln 2)            log2 domain
    p   = softmax_natural(s)                  = exp2(s - lse)
    αX  = p @ X                               [N_q, D_KV]   (kernel-internal only)
    o   = αX @ W_V                            [N_q, D_H]    (final)

  bwd:
    delta = O · dO                            [N_q]         (fp32, predcomputed)
    dαX   = dO @ W_V^T                        [N_q, D_KV]
    dW_V += (αX_unnorm)^T @ dO  ... wait      see below
    dp    = dαX @ X^T                         [N_q, N_k]
    ds    = p * (dp - delta)                  softmax bwd
    dX   += scale * ds^T @ u   +   p^T @ dαX  (two contributions)
    du   += scale * ds @ X

  delta identity (matrix-absorption preserves the same delta as standard FA):
    delta_i = sum_j p_ij * dp_ij
            = sum_j p_ij * (dαX_i · X_j)
            = dαX_i · αX_i
            = (dO_i @ W_V^T) · αX_i
            = dO_i · (αX_i @ W_V)
            = dO_i · O_i                ← same as standard FA
"""

import torch
import triton
import triton.language as tl


# Documentation only; Triton 3.x kernels can't read Python globals so each
# kernel re-defines the constant inside its body.
_RCP_LN2_DOC = 1.4426950408889634


@triton.jit(do_not_specialize=["scale"])
def _fwd_kernel(
    U,                      # [T_q, H, D_KV]
    X,                      # [T_k, D_KV]
    WV,                     # [D_KV, H, D_H]
    O,                      # [T_q, H, D_H]
    LSE,                    # [T_q, H]            fp32
    cu_seqlens_q,           # [n_seg+1]           int32
    cu_seqlens_k,           # [n_seg+1]           int32
    q_chunk_indices,        # [NT_q, 2]           int32  (i_n, i_t)
    scale,                  # float, = 1/sqrt(D_H)
    H: tl.constexpr,
    D_KV: tl.constexpr,
    B_DKV: tl.constexpr,    # padded/power-of-two tile dim >= D_KV
    D_H: tl.constexpr,
    BT_Q: tl.constexpr,     # Q block size
    BT_K: tl.constexpr,     # K block size
):
    # grid = (NT_q, H)
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    # decode "which Q-chunk of which segment"
    i_n = tl.load(q_chunk_indices + i_chunk * 2).to(tl.int32)
    i_t = tl.load(q_chunk_indices + i_chunk * 2 + 1).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    q_start = i_t * BT_Q
    if q_start >= T_q_seg:
        return

    # ----- pointers ---------------------------------------------------------
    # u: head i_h slice of token range [bos_q, eos_q). Stride along token = H*D_KV.
    p_u = tl.make_block_ptr(
        U + bos_q * H * D_KV + i_h * D_KV,
        shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
    )
    # O: same layout as u but D_H last.
    p_o = tl.make_block_ptr(
        O + bos_q * H * D_H + i_h * D_H,
        shape=(T_q_seg, D_H), strides=(H * D_H, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, D_H), order=(1, 0),
    )
    # W_V at this head: W_V[:, i_h, :]. Shape (D_KV, D_H), stride (H*D_H, 1).
    p_wv = tl.make_block_ptr(
        WV + i_h * D_H,
        shape=(D_KV, D_H), strides=(H * D_H, 1),
        offsets=(0, 0), block_shape=(B_DKV, D_H), order=(1, 0),
    )

    b_u = tl.load(p_u, boundary_check=(0, 1), padding_option="zero")    # [BT_Q, B_DKV]
    b_wv = tl.load(p_wv, boundary_check=(0, 1), padding_option="zero")  # [B_DKV, D_H]

    # ----- online-softmax accumulators (fp32) -------------------------------
    b_o = tl.zeros([BT_Q, D_H], dtype=tl.float32)
    b_m = tl.full([BT_Q], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT_Q], dtype=tl.float32)

    # ----- single K loop over the entire K segment --------------------------
    # BlockDiagonal full attention: no causal, no window. Walk bos_k -> eos_k.
    for k_start in range(0, T_k_seg, BT_K):
        # X block: [BT_K, D_KV] from row bos_k+k_start, no H dim
        p_x = tl.make_block_ptr(
            X + bos_k * D_KV,
            shape=(T_k_seg, D_KV), strides=(D_KV, 1),
            offsets=(k_start, 0), block_shape=(BT_K, B_DKV), order=(1, 0),
        )
        b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero")  # [BT_K, B_DKV]

        # s = u @ X^T : [BT_Q, D_KV] @ [D_KV, BT_K] -> [BT_Q, BT_K]
        # We use tl.trans on b_x rather than fetching X^T directly because the
        # FA-style block_ptr trick (shape=(D, T), strides=(1, D_KV)) requires
        # a D_KV inner stride of 1, which we already have row-major.
        # tl.dot can take (M,K)@(K,N), so transposing b_x gives (D_KV, BT_K).
        b_s = tl.dot(b_u, tl.trans(b_x), input_precision="ieee") * scale * rcp_ln2

        # mask K-tail (positions >= T_k_seg become -inf -> p = 0)
        o_k = k_start + tl.arange(0, BT_K)
        m_k = o_k < T_k_seg
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        # online softmax update (log2 domain)
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        # guard fully-masked rows (only happens if T_k_seg == 0, which we
        # already early-returned on for Q side, but defensive)
        b_m_safe = tl.where(b_m_new == float("-inf"), 0.0, b_m_new)
        b_r = tl.exp2(b_m - b_m_safe)                   # [BT_Q]   rescale prev accum
        b_p = tl.exp2(b_s - b_m_safe[:, None])          # [BT_Q, BT_K]

        b_acc = b_acc * b_r + tl.sum(b_p, 1)

        # Matrix absorption: αX = p @ X is computed in registers and
        # IMMEDIATELY multiplied by W_V, so αX never lands in HBM.
        # Both matmuls run in fp32 accumulator; we cast operands to b_x.dtype
        # so the matmul itself uses the input dtype (fp32 or bf16).
        b_alphax = tl.dot(b_p.to(b_x.dtype), b_x, input_precision="ieee")
        b_o = (
            b_o * b_r[:, None]
            + tl.dot(b_alphax.to(b_wv.dtype), b_wv, input_precision="ieee")
        )
        b_m = b_m_new

    # ----- finalize ---------------------------------------------------------
    b_o = b_o / b_acc[:, None]
    # LSE in log2 domain — bwd uses the same convention.
    b_lse = b_m + tl.log2(b_acc)

    # store O
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    # store LSE: [T_q_total, H], stride H along T.
    o_q = q_start + tl.arange(0, BT_Q)
    m_q = o_q < T_q_seg
    lse_ptr = LSE + (bos_q + o_q) * H + i_h
    tl.store(lse_ptr, b_lse, mask=m_q)


@triton.jit(do_not_specialize=["scale"])
def _fwd_alpha_kernel(
    U,                      # [T_q, H, D_KV]
    X,                      # [T_k, D_KV]
    WV,                     # [D_KV, H, D_H]
    O,                      # [T_q, H, D_H]
    LSE,                    # [T_q, H]            fp32
    cu_seqlens_q,           # [n_seg+1]           int32
    cu_seqlens_k,           # [n_seg+1]           int32
    q_chunk_indices,        # [NT_q, 2]           int32  (i_n, i_t)
    scale,                  # float, = 1/sqrt(D_H)
    H: tl.constexpr,
    D_KV: tl.constexpr,
    B_DKV: tl.constexpr,
    D_H: tl.constexpr,
    BT_Q: tl.constexpr,
    BT_K: tl.constexpr,
):
    # Experimental forward variant:
    #   online accumulate alphaX = softmax(u @ X^T) @ X in fp32, then apply WV
    #   once at the end. This trades a much larger [BT_Q, B_DKV] accumulator for
    #   avoiding the per-K-block alphaX @ WV dot in _fwd_kernel.
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(q_chunk_indices + i_chunk * 2).to(tl.int32)
    i_t = tl.load(q_chunk_indices + i_chunk * 2 + 1).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    q_start = i_t * BT_Q
    if q_start >= T_q_seg:
        return

    p_u = tl.make_block_ptr(
        U + bos_q * H * D_KV + i_h * D_KV,
        shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
    )
    p_o = tl.make_block_ptr(
        O + bos_q * H * D_H + i_h * D_H,
        shape=(T_q_seg, D_H), strides=(H * D_H, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, D_H), order=(1, 0),
    )
    p_wv = tl.make_block_ptr(
        WV + i_h * D_H,
        shape=(D_KV, D_H), strides=(H * D_H, 1),
        offsets=(0, 0), block_shape=(B_DKV, D_H), order=(1, 0),
    )

    b_u = tl.load(p_u, boundary_check=(0, 1), padding_option="zero")
    b_wv = tl.load(p_wv, boundary_check=(0, 1), padding_option="zero")

    b_alphax_acc = tl.zeros([BT_Q, B_DKV], dtype=tl.float32)
    b_m = tl.full([BT_Q], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT_Q], dtype=tl.float32)

    for k_start in range(0, T_k_seg, BT_K):
        p_x = tl.make_block_ptr(
            X + bos_k * D_KV,
            shape=(T_k_seg, D_KV), strides=(D_KV, 1),
            offsets=(k_start, 0), block_shape=(BT_K, B_DKV), order=(1, 0),
        )
        b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero")

        b_s = tl.dot(b_u, tl.trans(b_x), input_precision="ieee") * scale * rcp_ln2

        o_k = k_start + tl.arange(0, BT_K)
        m_k = o_k < T_k_seg
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_m_safe = tl.where(b_m_new == float("-inf"), 0.0, b_m_new)
        b_r = tl.exp2(b_m - b_m_safe)
        b_p = tl.exp2(b_s - b_m_safe[:, None])

        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_alphax = tl.dot(b_p.to(b_x.dtype), b_x, input_precision="ieee")
        b_alphax_acc = b_alphax_acc * b_r[:, None] + b_alphax
        b_m = b_m_new

    b_alphax_acc = b_alphax_acc / b_acc[:, None]
    b_o = tl.dot(b_alphax_acc.to(b_wv.dtype), b_wv, input_precision="ieee")

    b_lse = b_m + tl.log2(b_acc)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    o_q = q_start + tl.arange(0, BT_Q)
    m_q = o_q < T_q_seg
    lse_ptr = LSE + (bos_q + o_q) * H + i_h
    tl.store(lse_ptr, b_lse, mask=m_q)


# ---------------------------------------------------------------------------
# Backward preprocess: delta[t, h] = sum_d O[t, h, d] * dO[t, h, d]   in fp32
#
# Same trick as standard FA bwd. As derived in the module docstring, even
# under matrix absorption the identity delta_i = O_i · dO_i holds:
#   sum_j p_ij * dp_ij = sum_j p_ij * (dαX_i · X_j)
#                     = dαX_i · αX_i
#                     = dO_i · (αX_i @ W_V)
#                     = dO_i · O_i
# so we can keep the same preprocess kernel as v2 FA.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T_q"])
def _bwd_preprocess_kernel(
    O,                      # [T_q, H, D_H]
    DO,                     # [T_q, H, D_H]
    Delta,                  # [T_q, H]    fp32
    T_q,                    # runtime int
    H: tl.constexpr,
    D_H: tl.constexpr,
    BT: tl.constexpr,
):
    # grid = (cdiv(T_q, BT), H)
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)

    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D_H)
    m_t = offs_t < T_q

    o_ptrs = O + offs_t[:, None] * (H * D_H) + i_h * D_H + offs_d[None, :]
    do_ptrs = DO + offs_t[:, None] * (H * D_H) + i_h * D_H + offs_d[None, :]

    b_o = tl.load(o_ptrs, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_do = tl.load(do_ptrs, mask=m_t[:, None], other=0.0).to(tl.float32)

    b_delta = tl.sum(b_o * b_do, axis=1)

    delta_ptrs = Delta + offs_t * H + i_h
    tl.store(delta_ptrs, b_delta, mask=m_t)


# ---------------------------------------------------------------------------
# Backward du kernel (split-kernel, grid over Q-chunks).
#
# Each program owns one Q-block of one head, walks the entire K-segment,
# and:
#   - writes dαX = dO @ W_V^T once for reuse by dX
#   - accumulates du = scale * sum_j ds_ij * X_j   (own write, no atomic)
#   - accumulates αX = sum_j p_ij * X_j, then writes
#     dW_V += αX^T @ dO once per Q-block via fp32 atomic_add
#
# We do NOT touch dX here — that's the _bwd_dx_kernel's job.
#
# Math (per Q-block, per head):
#   load u, dO, lse, delta for this Q-block
#   load W_V[:, i_h, :]                                    # resident
#   compute dαX = dO @ W_V^T  in fp32 once                 # [BT_Q, D_KV]
#   for each K-block:
#     load X_block                                         # [BT_K, D_KV]
#     s   = u @ X_block^T * scale * rcp_ln2                # log2 domain
#     p   = exp2(s - lse)                                  # natural-domain prob
#     dp  = dαX @ X_block^T                                # [BT_Q, BT_K]
#     ds  = p * (dp - delta)
#     du += ds @ X_block       (scaled by `scale` at the end, once)
#     αX += p @ X_block
#   dW_V += αX^T @ dO                                      # atomic_add fp32
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["scale"])
def _bwd_du_kernel(
    U,                      # [T_q, H, D_KV]
    X,                      # [T_k, D_KV]
    WV,                     # [D_KV, H, D_H]
    DO,                     # [T_q, H, D_H]
    DU,                     # [T_q, H, D_KV]      output
    DAlphaX,                # [T_q, H, D_KV]      fp32 output, reused by dX
    DWV_fp32,               # [D_KV, H, D_H]      fp32 atomic-add target
    LSE,                    # [T_q, H]            fp32  (log2 domain)
    Delta,                  # [T_q, H]            fp32
    cu_seqlens_q,           # [n_seg+1]           int32
    cu_seqlens_k,           # [n_seg+1]           int32
    q_chunk_indices,        # [NT_q, 2]           int32
    scale,                  # float
    H: tl.constexpr,
    D_KV: tl.constexpr,
    B_DKV: tl.constexpr,
    D_H: tl.constexpr,
    BT_Q: tl.constexpr,
    BT_K: tl.constexpr,
):
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(q_chunk_indices + i_chunk * 2).to(tl.int32)
    i_t = tl.load(q_chunk_indices + i_chunk * 2 + 1).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    q_start = i_t * BT_Q
    if q_start >= T_q_seg:
        return

    # ----- pointers ---------------------------------------------------------
    p_u = tl.make_block_ptr(
        U + bos_q * H * D_KV + i_h * D_KV,
        shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
    )
    p_do = tl.make_block_ptr(
        DO + bos_q * H * D_H + i_h * D_H,
        shape=(T_q_seg, D_H), strides=(H * D_H, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, D_H), order=(1, 0),
    )
    p_du = tl.make_block_ptr(
        DU + bos_q * H * D_KV + i_h * D_KV,
        shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
    )
    p_dalphax = tl.make_block_ptr(
        DAlphaX + bos_q * H * D_KV + i_h * D_KV,
        shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
    )
    p_wv = tl.make_block_ptr(
        WV + i_h * D_H,
        shape=(D_KV, D_H), strides=(H * D_H, 1),
        offsets=(0, 0), block_shape=(B_DKV, D_H), order=(1, 0),
    )
    p_lse = tl.make_block_ptr(
        LSE + bos_q * H + i_h,
        shape=(T_q_seg,), strides=(H,),
        offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
    )
    p_delta = tl.make_block_ptr(
        Delta + bos_q * H + i_h,
        shape=(T_q_seg,), strides=(H,),
        offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
    )

    b_u = tl.load(p_u, boundary_check=(0, 1), padding_option="zero")       # [BT_Q, B_DKV]
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")     # [BT_Q, D_H]
    b_wv = tl.load(p_wv, boundary_check=(0, 1), padding_option="zero")     # [B_DKV, D_H]
    b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")     # [BT_Q]
    b_delta = tl.load(p_delta, boundary_check=(0,), padding_option="zero") # [BT_Q]

    # dαX = dO @ W_V^T : [BT_Q, D_H] @ [D_H, D_KV] -> [BT_Q, D_KV]
    # Materialized in fp32 once, reused across the K loop.
    b_dalphax = tl.dot(b_do, tl.trans(b_wv), input_precision="ieee")
    tl.store(p_dalphax, b_dalphax, boundary_check=(0, 1))

    # du accumulator (fp32). Scale applied once at the end.
    b_du = tl.zeros([BT_Q, B_DKV], dtype=tl.float32)
    # αX accumulator for dW_V. This avoids a [D_KV, D_H] atomic_add per K block.
    b_alphax = tl.zeros([BT_Q, B_DKV], dtype=tl.float32)

    # ----- single K loop ----------------------------------------------------
    for k_start in range(0, T_k_seg, BT_K):
        p_x = tl.make_block_ptr(
            X + bos_k * D_KV,
            shape=(T_k_seg, D_KV), strides=(D_KV, 1),
            offsets=(k_start, 0), block_shape=(BT_K, B_DKV), order=(1, 0),
        )
        b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero")   # [BT_K, B_DKV]

        # recompute s in log2 domain (same recipe as fwd)
        b_s = tl.dot(b_u, tl.trans(b_x), input_precision="ieee") * scale * rcp_ln2

        o_k = k_start + tl.arange(0, BT_K)
        m_k = o_k < T_k_seg
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        # natural-domain softmax prob
        b_p = tl.exp2(b_s - b_lse[:, None])             # [BT_Q, BT_K]

        # dp = dαX @ X_block^T : [BT_Q, D_KV] @ [D_KV, BT_K] -> [BT_Q, BT_K]
        b_dp = tl.dot(b_dalphax.to(b_x.dtype), tl.trans(b_x), input_precision="ieee")
        b_ds = b_p * (b_dp - b_delta[:, None])          # [BT_Q, BT_K] fp32

        # du += ds @ X_block : [BT_Q, BT_K] @ [BT_K, D_KV] -> [BT_Q, D_KV]
        # (no scale yet; we multiply by scale once after the loop)
        b_du += tl.dot(b_ds.to(b_x.dtype), b_x, input_precision="ieee")

        # αX += p @ X_block. Since p is reconstructed from final LSE, it is
        # already normalized; no online-softmax rescale is needed here.
        b_alphax += tl.dot(b_p.to(b_x.dtype), b_x, input_precision="ieee")

    # dW_V[:, i_h, :] += αX^T @ dO. One atomic matrix write per Q-block instead
    # of one per K-block.
    b_dwv_contrib = tl.dot(tl.trans(b_alphax).to(b_do.dtype), b_do, input_precision="ieee")
    offs_d_kv = tl.arange(0, B_DKV)
    offs_d_h = tl.arange(0, D_H)
    m_d_kv = offs_d_kv < D_KV
    m_d_h = offs_d_h < D_H
    dwv_ptrs = (
        DWV_fp32
        + offs_d_kv[:, None] * (H * D_H)
        + i_h * D_H
        + offs_d_h[None, :]
    )
    tl.atomic_add(dwv_ptrs, b_dwv_contrib, mask=m_d_kv[:, None] & m_d_h[None, :])

    # final scale for du (chain rule from s = u·X*scale)
    b_du = b_du * scale

    tl.store(p_du, b_du.to(p_du.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Backward dX kernel (split-kernel, grid over K-chunks).
#
# Each program owns one K-block of one head. Walks every Q-block in the
# segment and accumulates that head's contribution to dX. X is shared across
# heads, so the final dX is a sum over H.
#
# Strategy chosen: this kernel writes a non-atomic per-head partial buffer:
# DX_partial_fp32 [H, T_k, D_KV]. A separate reduce kernel sums H partials into
# DX_fp32 [T_k, D_KV]. This trades one extra memory pass for removing fp32
# atomic contention across heads.
#
# Math (per K-block, per head):
#   load X_block, lse_seg ... actually no — lse is per-Q-token, not per-K
#   For each Q-block:
#     load u_q, dO_q, lse_q, delta_q
#     s = u_q @ X_block^T * scale * rcp_ln2
#     p = exp2(s - lse_q)
#     load dαX_q computed by _bwd_du_kernel
#     dp = dαX_q @ X_block^T
#     ds = p * (dp - delta_q)
#     dX_block += scale * ds^T @ u_q   +   p^T @ dαX_q
#
# Note: dαX used to be recomputed inside this K-chunk kernel. That repeated the
# same dO @ W_V^T work once per K chunk, which is expensive for long ragged
# batches. _bwd_du_kernel now writes dαX once per Q block/head, and dX loads it.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["scale", "T_k_total"])
def _bwd_dx_kernel(
    U,                      # [T_q, H, D_KV]
    X,                      # [T_k, D_KV]
    DAlphaX,                # [T_q, H, D_KV]      fp32, from _bwd_du_kernel
    DX_partial_fp32,        # [H, T_k, D_KV]      fp32 per-head partial target
    T_k_total,              # runtime int, stride for DX_partial head dim
    LSE,                    # [T_q, H]            fp32  (log2 domain)
    Delta,                  # [T_q, H]            fp32
    cu_seqlens_q,           # [n_seg+1]           int32
    cu_seqlens_k,           # [n_seg+1]           int32
    k_chunk_indices,        # [NT_k, 2]           int32
    scale,                  # float
    H: tl.constexpr,
    D_KV: tl.constexpr,
    B_DKV: tl.constexpr,
    D_H: tl.constexpr,
    BT_K: tl.constexpr,     # this kernel's outer block size = K block
    BT_Q: tl.constexpr,     # inner loop block size = Q block
):
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(k_chunk_indices + i_chunk * 2).to(tl.int32)
    i_t = tl.load(k_chunk_indices + i_chunk * 2 + 1).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    k_start = i_t * BT_K
    if k_start >= T_k_seg:
        return

    # ----- pointers ---------------------------------------------------------
    p_x = tl.make_block_ptr(
        X + bos_k * D_KV,
        shape=(T_k_seg, D_KV), strides=(D_KV, 1),
        offsets=(k_start, 0), block_shape=(BT_K, B_DKV), order=(1, 0),
    )
    b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero")       # [BT_K, B_DKV]

    # dX accumulator for this K-block and this head (fp32). At the end, store
    # into that head's partial slice. The reduce kernel sums heads later.
    b_dx = tl.zeros([BT_K, B_DKV], dtype=tl.float32)

    # ----- single Q loop over the entire Q segment --------------------------
    for q_start in range(0, T_q_seg, BT_Q):
        p_u = tl.make_block_ptr(
            U + bos_q * H * D_KV + i_h * D_KV,
            shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
            offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
        )
        p_dalphax = tl.make_block_ptr(
            DAlphaX + bos_q * H * D_KV + i_h * D_KV,
            shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
            offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
        )
        p_lse = tl.make_block_ptr(
            LSE + bos_q * H + i_h,
            shape=(T_q_seg,), strides=(H,),
            offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
        )
        p_delta = tl.make_block_ptr(
            Delta + bos_q * H + i_h,
            shape=(T_q_seg,), strides=(H,),
            offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
        )

        b_u = tl.load(p_u, boundary_check=(0, 1), padding_option="zero")       # [BT_Q, B_DKV]
        b_dalphax = tl.load(p_dalphax, boundary_check=(0, 1), padding_option="zero")
        b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")     # [BT_Q]
        b_delta = tl.load(p_delta, boundary_check=(0,), padding_option="zero") # [BT_Q]

        # s = u @ X^T : [BT_Q, D_KV] @ [D_KV, BT_K] -> [BT_Q, BT_K]
        b_s = tl.dot(b_u, tl.trans(b_x), input_precision="ieee") * scale * rcp_ln2

        # mask Q-tail
        o_q = q_start + tl.arange(0, BT_Q)
        m_q = o_q < T_q_seg
        b_s = tl.where(m_q[:, None], b_s, float("-inf"))
        # mask K-tail
        o_k = k_start + tl.arange(0, BT_K)
        m_k = o_k < T_k_seg
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_p = tl.exp2(b_s - b_lse[:, None])             # [BT_Q, BT_K]

        # dp = dαX @ X^T : [BT_Q, D_KV] @ [D_KV, BT_K] -> [BT_Q, BT_K]
        b_dp = tl.dot(b_dalphax.to(b_x.dtype), tl.trans(b_x), input_precision="ieee")
        b_ds = b_p * (b_dp - b_delta[:, None])          # [BT_Q, BT_K]

        # dX contribution 1: from s = u @ X^T (scale * ds^T @ u)
        # ds^T @ u : [BT_K, BT_Q] @ [BT_Q, D_KV] -> [BT_K, D_KV]
        b_dx += scale * tl.dot(tl.trans(b_ds).to(b_u.dtype), b_u, input_precision="ieee")

        # dX contribution 2: from αX = p @ X (p^T @ dαX)
        # p^T @ dαX : [BT_K, BT_Q] @ [BT_Q, D_KV] -> [BT_K, D_KV]
        b_dx += tl.dot(
            tl.trans(b_p).to(b_x.dtype),
            b_dalphax.to(b_x.dtype),
            input_precision="ieee",
        )

    # ----- store into DX_partial_fp32 --------------------------------------
    # Layout: DX_partial_fp32 [H, T_k, D_KV], stride (T_k*D_KV, D_KV, 1).
    offs_t = bos_k + k_start + tl.arange(0, BT_K)
    offs_d = tl.arange(0, B_DKV)
    m_t = (k_start + tl.arange(0, BT_K)) < T_k_seg
    m_d = offs_d < D_KV
    dx_ptrs = (
        DX_partial_fp32
        + i_h * T_k_total * D_KV
        + offs_t[:, None] * D_KV
        + offs_d[None, :]
    )
    tl.store(dx_ptrs, b_dx, mask=m_t[:, None] & m_d[None, :])


@triton.jit(do_not_specialize=["scale"])
def _bwd_dx_atomic_heads_kernel(
    U,                      # [T_q, H, D_KV]
    X,                      # [T_k, D_KV]
    DAlphaX,                # [T_q, H, D_KV]      fp32, from _bwd_du_kernel
    DX_fp32,                # [T_k, D_KV]         fp32 atomic-add target
    LSE,                    # [T_q, H]            fp32  (log2 domain)
    Delta,                  # [T_q, H]            fp32
    cu_seqlens_q,           # [n_seg+1]           int32
    cu_seqlens_k,           # [n_seg+1]           int32
    k_chunk_indices,        # [NT_k, 2]           int32
    scale,                  # float
    H: tl.constexpr,
    D_KV: tl.constexpr,
    B_DKV: tl.constexpr,
    D_H: tl.constexpr,
    BT_K: tl.constexpr,
    BT_Q: tl.constexpr,
):
    # grid = (NT_k, H). Each program owns one K-block/head and atomically adds
    # that head's contribution into final dX. This is an experimental path for
    # long batches: more parallelism than fused-heads, no giant partial buffer.
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(k_chunk_indices + i_chunk * 2).to(tl.int32)
    i_t = tl.load(k_chunk_indices + i_chunk * 2 + 1).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    k_start = i_t * BT_K
    if k_start >= T_k_seg:
        return

    p_x = tl.make_block_ptr(
        X + bos_k * D_KV,
        shape=(T_k_seg, D_KV), strides=(D_KV, 1),
        offsets=(k_start, 0), block_shape=(BT_K, B_DKV), order=(1, 0),
    )
    b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero")

    b_dx = tl.zeros([BT_K, B_DKV], dtype=tl.float32)

    for q_start in range(0, T_q_seg, BT_Q):
        p_u = tl.make_block_ptr(
            U + bos_q * H * D_KV + i_h * D_KV,
            shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
            offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
        )
        p_dalphax = tl.make_block_ptr(
            DAlphaX + bos_q * H * D_KV + i_h * D_KV,
            shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
            offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
        )
        p_lse = tl.make_block_ptr(
            LSE + bos_q * H + i_h,
            shape=(T_q_seg,), strides=(H,),
            offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
        )
        p_delta = tl.make_block_ptr(
            Delta + bos_q * H + i_h,
            shape=(T_q_seg,), strides=(H,),
            offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
        )

        b_u = tl.load(p_u, boundary_check=(0, 1), padding_option="zero")
        b_dalphax = tl.load(p_dalphax, boundary_check=(0, 1), padding_option="zero")
        b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
        b_delta = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

        b_s = tl.dot(b_u, tl.trans(b_x), input_precision="ieee") * scale * rcp_ln2

        o_q = q_start + tl.arange(0, BT_Q)
        m_q = o_q < T_q_seg
        b_s = tl.where(m_q[:, None], b_s, float("-inf"))
        o_k = k_start + tl.arange(0, BT_K)
        m_k = o_k < T_k_seg
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_p = tl.exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_dalphax.to(b_x.dtype), tl.trans(b_x), input_precision="ieee")
        b_ds = b_p * (b_dp - b_delta[:, None])

        b_dx += scale * tl.dot(tl.trans(b_ds).to(b_u.dtype), b_u, input_precision="ieee")
        b_dx += tl.dot(
            tl.trans(b_p).to(b_x.dtype),
            b_dalphax.to(b_x.dtype),
            input_precision="ieee",
        )

    offs_t = bos_k + k_start + tl.arange(0, BT_K)
    offs_d = tl.arange(0, B_DKV)
    m_t = (k_start + tl.arange(0, BT_K)) < T_k_seg
    m_d = offs_d < D_KV
    dx_ptrs = DX_fp32 + offs_t[:, None] * D_KV + offs_d[None, :]
    tl.atomic_add(dx_ptrs, b_dx, mask=m_t[:, None] & m_d[None, :])


@triton.jit(do_not_specialize=["T_k_total"])
def _bwd_dx_reduce_kernel(
    DX_partial_fp32,        # [H, T_k, D_KV]
    DX_fp32,                # [T_k, D_KV]
    T_k_total,              # runtime int
    H: tl.constexpr,
    D_KV: tl.constexpr,
    B_DKV: tl.constexpr,
    BT_K: tl.constexpr,
):
    # grid = (cdiv(T_k, BT_K),)
    i_t = tl.program_id(0)
    offs_t = i_t * BT_K + tl.arange(0, BT_K)
    offs_d = tl.arange(0, B_DKV)
    m_t = offs_t < T_k_total
    m_d = offs_d < D_KV

    acc = tl.zeros([BT_K, B_DKV], dtype=tl.float32)
    for i_h in tl.static_range(0, H):
        ptrs = (
            DX_partial_fp32
            + i_h * T_k_total * D_KV
            + offs_t[:, None] * D_KV
            + offs_d[None, :]
        )
        acc += tl.load(ptrs, mask=m_t[:, None] & m_d[None, :], other=0.0)

    out_ptrs = DX_fp32 + offs_t[:, None] * D_KV + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=m_t[:, None] & m_d[None, :])


# ---------------------------------------------------------------------------
# Backward dX kernel for large-K ragged batches (grid over K-chunks only).
#
# Each program owns one K-block and loops over all heads internally, producing
# the final cross-head dX for that K-block directly. This avoids the large
# DX_partial_fp32 [H, T_k, D_KV] buffer and the reduce pass, which become
# expensive when T_k is large and there are already enough K-blocks for
# parallelism. It also reuses the precomputed dαX buffer instead of recomputing
# dO @ W_V^T once per K chunk.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["scale"])
def _bwd_dx_fused_heads_kernel(
    U,                      # [T_q, H, D_KV]
    X,                      # [T_k, D_KV]
    DAlphaX,                # [T_q, H, D_KV]      fp32, from _bwd_du_kernel
    DX_fp32,                # [T_k, D_KV]         fp32 output
    LSE,                    # [T_q, H]            fp32  (log2 domain)
    Delta,                  # [T_q, H]            fp32
    cu_seqlens_q,           # [n_seg+1]           int32
    cu_seqlens_k,           # [n_seg+1]           int32
    k_chunk_indices,        # [NT_k, 2]           int32
    scale,                  # float
    H: tl.constexpr,
    D_KV: tl.constexpr,
    B_DKV: tl.constexpr,
    D_H: tl.constexpr,
    BT_K: tl.constexpr,
    BT_Q: tl.constexpr,
):
    i_chunk = tl.program_id(0)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(k_chunk_indices + i_chunk * 2).to(tl.int32)
    i_t = tl.load(k_chunk_indices + i_chunk * 2 + 1).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    k_start = i_t * BT_K
    if k_start >= T_k_seg:
        return

    p_x = tl.make_block_ptr(
        X + bos_k * D_KV,
        shape=(T_k_seg, D_KV), strides=(D_KV, 1),
        offsets=(k_start, 0), block_shape=(BT_K, B_DKV), order=(1, 0),
    )
    b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero")       # [BT_K, B_DKV]

    b_dx = tl.zeros([BT_K, B_DKV], dtype=tl.float32)

    for i_h in tl.static_range(0, H):
        for q_start in range(0, T_q_seg, BT_Q):
            p_u = tl.make_block_ptr(
                U + bos_q * H * D_KV + i_h * D_KV,
                shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
                offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
            )
            p_dalphax = tl.make_block_ptr(
                DAlphaX + bos_q * H * D_KV + i_h * D_KV,
                shape=(T_q_seg, D_KV), strides=(H * D_KV, 1),
                offsets=(q_start, 0), block_shape=(BT_Q, B_DKV), order=(1, 0),
            )
            p_lse = tl.make_block_ptr(
                LSE + bos_q * H + i_h,
                shape=(T_q_seg,), strides=(H,),
                offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
            )
            p_delta = tl.make_block_ptr(
                Delta + bos_q * H + i_h,
                shape=(T_q_seg,), strides=(H,),
                offsets=(q_start,), block_shape=(BT_Q,), order=(0,),
            )

            b_u = tl.load(p_u, boundary_check=(0, 1), padding_option="zero")
            b_dalphax = tl.load(p_dalphax, boundary_check=(0, 1), padding_option="zero")
            b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
            b_delta = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

            b_s = tl.dot(b_u, tl.trans(b_x), input_precision="ieee") * scale * rcp_ln2

            o_q = q_start + tl.arange(0, BT_Q)
            m_q = o_q < T_q_seg
            b_s = tl.where(m_q[:, None], b_s, float("-inf"))
            o_k = k_start + tl.arange(0, BT_K)
            m_k = o_k < T_k_seg
            b_s = tl.where(m_k[None, :], b_s, float("-inf"))

            b_p = tl.exp2(b_s - b_lse[:, None])
            b_dp = tl.dot(b_dalphax.to(b_x.dtype), tl.trans(b_x), input_precision="ieee")
            b_ds = b_p * (b_dp - b_delta[:, None])

            b_dx += scale * tl.dot(tl.trans(b_ds).to(b_u.dtype), b_u, input_precision="ieee")
            b_dx += tl.dot(
                tl.trans(b_p).to(b_x.dtype),
                b_dalphax.to(b_x.dtype),
                input_precision="ieee",
            )

    offs_t = bos_k + k_start + tl.arange(0, BT_K)
    offs_d = tl.arange(0, B_DKV)
    m_t = (k_start + tl.arange(0, BT_K)) < T_k_seg
    m_d = offs_d < D_KV
    dx_ptrs = DX_fp32 + offs_t[:, None] * D_KV + offs_d[None, :]
    tl.store(dx_ptrs, b_dx, mask=m_t[:, None] & m_d[None, :])
