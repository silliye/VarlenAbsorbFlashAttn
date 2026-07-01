"""
Triton kernels for packed varlen FlashAttention (v2 working branch).

Scope:
  - packed varlen, decoupled cu_seqlens_q / cu_seqlens_k
  - BlockDiagonalMask (segment-wise full, no causal inside segment)
  - bf16 IO, fp32 internal accumulators
  - MHA only (HQ == H), head_dim parametric (D constexpr)
  - no causal, no window, no GQA, no gate, no sink_bias, no bias, no dropout

v2 vs v1: adds split-K + combine path for long K segments. Short segments
(T_k_seg <= SPLIT_THRESHOLD) still take the v1 single-program path
(_fwd_kernel / _bwd_dq_kernel). Long segments go through:
  fwd:    _fwd_split_kernel       -> partial O / LSE buffer
          _fwd_combine_kernel     -> final O / LSE
  bwd_dq: _bwd_dq_split_kernel    -> partial dq buffer
          _bwd_dq_combine_kernel  -> final dq
bwd_dkv (split-kernel by K block) is unchanged from v1: Stage 1's Q segment
is only 500 tokens, so each dkv program already has a short Q loop.
Backward strategy: split-kernel (no atomic).
  - bwd_kernel_dq:  grid over Q-blocks, loops over K-blocks within the segment
  - bwd_kernel_dkv: grid over K-blocks, loops over Q-blocks within the segment

Layout convention (kernel-internal):
  Q: [T_q, H, D]  contiguous on last dim
  K: [T_k, H, D]
  V: [T_k, H, D]
  O: [T_q, H, D]
  LSE: [T_q, H]   fp32  (token-major, matches parallel.py; wrapper transposes
                         to FA2's [H, T_q] if needed)

  Wrapper handles [1, T, H, D] <-> [T, H, D] squeeze.

LSE convention: stored in **log2 domain** (i.e. lse = m + log2(sum exp2(s-m))).
  Bwd kernels consume the same log2-domain lse, so the conversion cancels.

Partial buffer layout (split-K path):
  partial_O:   [N_split_progs, H, BT, D]  fp32   (NORMALIZED partial output:
                                                  O_split / l_split, just like
                                                  what the v1 kernel stores)
  partial_LSE: [N_split_progs, H, BT]     fp32   (log2 domain, m + log2(l))
  partial buffers are indexed by program_id along the leading axis. The combine
  kernel reads partial[partial_start : partial_start + n_splits] for the chunk
  it owns and merges via log2-domain online softmax:
      m       = max_s LSE_s
      w_s     = exp2(LSE_s - m)
      O       = sum_s (O_s * w_s) / sum_s w_s
      LSE_out = m + log2(sum_s w_s)
"""

import torch
import triton
import triton.language as tl


# 1 / ln(2): used to convert natural-domain softmax to log2 domain so we can
# use tl.exp2 / tl.log2 (faster + more numerically stable on most GPUs).
# NOTE: Triton 3.x kernels can't read plain Python globals from inside @triton.jit,
# so we define this as a local constant in each kernel body. Keep this Python-side
# value only as documentation / for any host-side use.
_RCP_LN2_DOC = 1.4426950408889634


@triton.jit(do_not_specialize=["scale"])
def _fwd_kernel(
    Q,                      # [T_q, H, D]  bf16
    K,                      # [T_k, H, D]  bf16
    V,                      # [T_k, H, D]  bf16
    O,                      # [T_q, H, D]  bf16
    LSE,                    # [T_q, H]     fp32   (token-major)
    cu_seqlens_q,           # [n_seg+1]    int32
    cu_seqlens_k,           # [n_seg+1]    int32
    q_chunk_indices,        # [NT, 2]      int32  (i_n, i_t) per Q-chunk
    scale,                  # float
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,       # Q block size
    BS: tl.constexpr,       # K block size
):
    # grid = (NT, H)
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)

    # local constant — see note in module docstring; can't read Python globals here.
    rcp_ln2 = 1.4426950408889634

    # decode (segment_idx, block_in_segment) for this Q-chunk
    i_n = tl.load(q_chunk_indices + i_chunk * 2).to(tl.int32)
    i_t = tl.load(q_chunk_indices + i_chunk * 2 + 1).to(tl.int32)

    # segment boundaries: Q segment vs K segment (decoupled lengths)
    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    q_start = i_t * BT
    if q_start >= T_q_seg:
        return

    # --- pointers (Q / O / K / V) -----------------------------------------
    # Q[bos_q + q_start : ..., i_h, :]   shape view inside segment: (T_q_seg, D)
    p_q = tl.make_block_ptr(
        Q + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_o = tl.make_block_ptr(
        O + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )

    # load Q (stays resident the whole kernel)
    b_q = tl.load(p_q, boundary_check=(0, 1))

    # online softmax accumulators (fp32)
    b_o = tl.zeros([BT, D], dtype=tl.float32)
    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # --- single K loop over the entire K segment --------------------------
    # BlockDiagonal full attention: no causal, no window. Walk bos_k -> eos_k.
    for k_start in range(0, T_k_seg, BS):
        # K^T block: shape (D, BS) using strides (1, H*D)
        p_k = tl.make_block_ptr(
            K + (bos_k * H + i_h) * D,
            shape=(D, T_k_seg), strides=(1, H * D),
            offsets=(0, k_start), block_shape=(D, BS), order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            V + (bos_k * H + i_h) * D,
            shape=(T_k_seg, D), strides=(H * D, 1),
            offsets=(k_start, 0), block_shape=(BS, D), order=(1, 0),
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))   # [D, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))   # [BS, D]

        # [BT, BS] = [BT, D] @ [D, BS], scale * rcp_ln2 applied after the dot
        # so the multiply happens in fp32 and we get a single fma.
        b_s = tl.dot(b_q, b_k) * scale * rcp_ln2

        # mask the tail of the last K block (positions >= T_k_seg)
        o_k = k_start + tl.arange(0, BS)
        m_k = o_k < T_k_seg
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        # online softmax update (log2 domain)
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        # guard fully-masked rows: replace -inf pivot with 0 to avoid -inf - -inf = NaN
        b_m_safe = tl.where(b_m_new == float("-inf"), 0.0, b_m_new)
        b_r = tl.exp2(b_m - b_m_safe)              # rescale factor for prev accum
        b_p = tl.exp2(b_s - b_m_safe[:, None])     # [BT, BS]

        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_m = b_m_new

    # --- finalize ----------------------------------------------------------
    b_o = b_o / b_acc[:, None]
    # LSE in log2 domain. parallel.py-style; bwd uses the same convention.
    b_lse = b_m + tl.log2(b_acc)

    # store O
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    # store LSE: layout [T_q_total, H], stride H. Use flat indexing.
    o_q = q_start + tl.arange(0, BT)
    m_q = o_q < T_q_seg
    lse_ptr = LSE + (bos_q + o_q) * H + i_h
    tl.store(lse_ptr, b_lse, mask=m_q)


# ---------------------------------------------------------------------------
# Backward preprocess: delta[i, h] = sum_d O[i, h, d] * dO[i, h, d]
#
# Computing delta out-of-band lets the bwd kernels avoid recomputing a
# full O*dO row reduction inside their inner loop. Standard FA2 trick.
#
# Layout: O / dO are [T_q_total, H, D], delta is [T_q_total, H] (token-major).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T_q"])
def _bwd_preprocess_kernel(
    O,                      # [T_q, H, D]  bf16
    DO,                     # [T_q, H, D]  bf16
    Delta,                  # [T_q, H]     fp32
    T_q,                    # runtime
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    # grid = (cdiv(T_q, BT), H)
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)

    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    m_t = offs_t < T_q

    # gather O[t, i_h, :] and dO[t, i_h, :]: stride along T is H*D, along D is 1
    o_ptrs = O + offs_t[:, None] * (H * D) + i_h * D + offs_d[None, :]
    do_ptrs = DO + offs_t[:, None] * (H * D) + i_h * D + offs_d[None, :]

    b_o = tl.load(o_ptrs, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_do = tl.load(do_ptrs, mask=m_t[:, None], other=0.0).to(tl.float32)

    # row-wise dot: [BT]
    b_delta = tl.sum(b_o * b_do, axis=1)

    # store delta[t, i_h], stride along T is H
    delta_ptrs = Delta + offs_t * H + i_h
    tl.store(delta_ptrs, b_delta, mask=m_t)


# ---------------------------------------------------------------------------
# Backward dQ kernel (split-kernel, no atomic).
#
# Each program owns one Q-block of one head and accumulates dQ by looping
# over all K-blocks in the same segment.
#
# Math (recap, log2 domain LSE):
#   s_ij = (q_i · k_j) * scale * RCP_LN2          (log2 domain pre-softmax)
#   p_ij = exp2(s_ij - lse_i)                     (natural-domain softmax prob,
#                                                  since lse is log2 domain too)
#   dp_ij = dO_i · v_j                            (natural)
#   ds_ij = p_ij * (dp_ij - delta_i)              (natural,  delta_i = O_i · dO_i)
#   dq_i = scale * sum_j ds_ij * k_j
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["scale"])
def _bwd_dq_kernel(
    Q,                      # [T_q, H, D]  bf16
    K,                      # [T_k, H, D]  bf16
    V,                      # [T_k, H, D]  bf16
    DO,                     # [T_q, H, D]  bf16
    DQ,                     # [T_q, H, D]  bf16  (output)
    LSE,                    # [T_q, H]     fp32  (log2 domain)
    Delta,                  # [T_q, H]     fp32
    cu_seqlens_q,           # [n_seg+1]    int32
    cu_seqlens_k,           # [n_seg+1]    int32
    q_chunk_indices,        # [NT, 2]      int32
    scale,                  # float
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
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

    q_start = i_t * BT
    if q_start >= T_q_seg:
        return

    # --- pointers ----------------------------------------------------------
    p_q = tl.make_block_ptr(
        Q + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_do = tl.make_block_ptr(
        DO + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_dq = tl.make_block_ptr(
        DQ + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )

    # LSE / delta: [T_q_total, H] with stride H along T.
    p_lse = tl.make_block_ptr(
        LSE + bos_q * H + i_h,
        shape=(T_q_seg,), strides=(H,),
        offsets=(q_start,), block_shape=(BT,), order=(0,),
    )
    p_delta = tl.make_block_ptr(
        Delta + bos_q * H + i_h,
        shape=(T_q_seg,), strides=(H,),
        offsets=(q_start,), block_shape=(BT,), order=(0,),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    # dQ accumulator (fp32). Each program owns this Q-block uniquely, so
    # we just initialize, accumulate, and store at the end.
    b_dq = tl.zeros([BT, D], dtype=tl.float32)

    # --- single K loop over the entire K segment --------------------------
    for k_start in range(0, T_k_seg, BS):
        # K^T as (D, BS)
        p_k = tl.make_block_ptr(
            K + (bos_k * H + i_h) * D,
            shape=(D, T_k_seg), strides=(1, H * D),
            offsets=(0, k_start), block_shape=(D, BS), order=(0, 1),
        )
        # V^T as (D, BS): same trick as K so we get dp = dO @ V^T directly.
        p_v = tl.make_block_ptr(
            V + (bos_k * H + i_h) * D,
            shape=(D, T_k_seg), strides=(1, H * D),
            offsets=(0, k_start), block_shape=(D, BS), order=(0, 1),
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))   # [D, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))   # [D, BS]

        # recompute s in log2 domain
        b_s = tl.dot(b_q, b_k) * scale * rcp_ln2    # [BT, BS]

        o_k = k_start + tl.arange(0, BS)
        m_k = o_k < T_k_seg
        # mask: tail positions become -inf -> exp2(-inf) = 0 -> p = 0
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        # natural-domain softmax probability
        b_p = tl.exp2(b_s - b_lse[:, None])         # [BT, BS]

        # dp = dO @ V^T : [BT, D] @ [D, BS] -> [BT, BS]   (natural)
        b_dp = tl.dot(b_do, b_v).to(tl.float32)
        # ds = p * (dp - delta)   (natural)
        b_ds = b_p * (b_dp - b_delta[:, None])

        # dq += ds @ K  : [BT, BS] @ [BS, D] -> [BT, D]
        # we have b_k as (D, BS); transpose for the matmul.
        b_dq += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_k))

    # final scale (chain rule from u = q·k * scale)
    b_dq = b_dq * scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Backward dK / dV kernel (split-kernel, no atomic).
#
# Each program owns one K-block of one head and accumulates dK and dV by
# looping over all Q-blocks in the same segment. Uses k_chunk_indices to
# pick which (segment, k-block) it owns.
#
# Math:
#   p_ij  = exp2(s_ij - lse_i)
#   dp_ij = dO_i · v_j
#   ds_ij = p_ij * (dp_ij - delta_i)
#   dv_j  = sum_i p_ij * dO_i
#   dk_j  = scale * sum_i ds_ij * q_i
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["scale"])
def _bwd_dkv_kernel(
    Q,                      # [T_q, H, D]  bf16
    K,                      # [T_k, H, D]  bf16
    V,                      # [T_k, H, D]  bf16
    DO,                     # [T_q, H, D]  bf16
    DK,                     # [T_k, H, D]  bf16  (output)
    DV,                     # [T_k, H, D]  bf16  (output)
    LSE,                    # [T_q, H]     fp32  (log2 domain)
    Delta,                  # [T_q, H]     fp32
    cu_seqlens_q,           # [n_seg+1]    int32
    cu_seqlens_k,           # [n_seg+1]    int32
    k_chunk_indices,        # [NT_k, 2]    int32
    scale,                  # float
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,       # K block size (note: in this kernel BT means K-block)
    BS: tl.constexpr,       # Q block size
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

    k_start = i_t * BT
    if k_start >= T_k_seg:
        return

    # --- pointers ----------------------------------------------------------
    # K and V blocks owned by this program: shape (BT, D)
    p_k = tl.make_block_ptr(
        K + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_v = tl.make_block_ptr(
        V + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_dk = tl.make_block_ptr(
        DK + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_dv = tl.make_block_ptr(
        DV + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT, D), order=(1, 0),
    )

    b_k = tl.load(p_k, boundary_check=(0, 1))   # [BT, D]
    b_v = tl.load(p_v, boundary_check=(0, 1))   # [BT, D]

    # accumulators (fp32)
    b_dk = tl.zeros([BT, D], dtype=tl.float32)
    b_dv = tl.zeros([BT, D], dtype=tl.float32)

    # --- single Q loop over the entire Q segment -------------------------
    for q_start in range(0, T_q_seg, BS):
        # Q^T as (D, BS): same trick as bwd_dq used for K
        p_q = tl.make_block_ptr(
            Q + (bos_q * H + i_h) * D,
            shape=(D, T_q_seg), strides=(1, H * D),
            offsets=(0, q_start), block_shape=(D, BS), order=(0, 1),
        )
        # dO^T as (D, BS)
        p_do = tl.make_block_ptr(
            DO + (bos_q * H + i_h) * D,
            shape=(D, T_q_seg), strides=(1, H * D),
            offsets=(0, q_start), block_shape=(D, BS), order=(0, 1),
        )
        # LSE / delta along T with stride H
        p_lse = tl.make_block_ptr(
            LSE + bos_q * H + i_h,
            shape=(T_q_seg,), strides=(H,),
            offsets=(q_start,), block_shape=(BS,), order=(0,),
        )
        p_delta = tl.make_block_ptr(
            Delta + bos_q * H + i_h,
            shape=(T_q_seg,), strides=(H,),
            offsets=(q_start,), block_shape=(BS,), order=(0,),
        )

        b_q = tl.load(p_q, boundary_check=(0, 1))     # [D, BS]
        b_do = tl.load(p_do, boundary_check=(0, 1))   # [D, BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))   # [BS]
        b_delta = tl.load(p_delta, boundary_check=(0,))  # [BS]

        # s = (k @ q) * scale * rcp_ln2 : [BT, D] @ [D, BS] -> [BT, BS]
        b_s = tl.dot(b_k, b_q) * scale * rcp_ln2

        # mask Q tail (positions >= T_q_seg) -> -inf so p = 0
        o_q = q_start + tl.arange(0, BS)
        m_q = o_q < T_q_seg
        b_s = tl.where(m_q[None, :], b_s, float("-inf"))

        # natural-domain softmax probability
        b_p = tl.exp2(b_s - b_lse[None, :])           # [BT, BS]

        # dv += p @ dO : [BT, BS] @ [BS, D] -> [BT, D]
        # we have b_do as [D, BS]; transpose for the matmul
        b_dv += tl.dot(b_p.to(b_k.dtype), tl.trans(b_do))

        # dp = v @ dO : [BT, D] @ [D, BS] -> [BT, BS]   (natural)
        b_dp = tl.dot(b_v, b_do).to(tl.float32)
        # ds = p * (dp - delta)
        b_ds = b_p * (b_dp - b_delta[None, :])

        # dk += ds @ Q : [BT, BS] @ [BS, D] -> [BT, D]
        # we have b_q as [D, BS]; transpose for the matmul
        b_dk += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q))

    b_dk = b_dk * scale

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# Split-K forward path (long segments only)
# ===========================================================================
#
# Each program owns ONE Q-block of ONE head AND a contiguous slice of K
# of length SPLIT_K_BLOCK. It runs an online softmax over its K slice and
# writes:
#   partial_O[program_id, i_h, :, :] = O_partial / l_partial    (fp32)
#   partial_LSE[program_id, i_h, :]  = m_partial + log2(l_partial)  (log2)
#
# The leading axis of the partial buffers is program-major: the row a
# program writes is determined by tl.program_id(0). The host index
# `split_chunk_indices` therefore doubles as the slot allocator -- no
# atomic counters, no contention.
#
# A fully-masked row (k slice landed entirely past EOS, which can happen
# only for the very last split of the very last Q-block when T_q_seg
# isn't a multiple of BT) writes m=-inf, lse=-inf, O=0; the combine
# kernel filters those out via tl.where(lse == -inf, 0, ...).
@triton.jit(do_not_specialize=["scale"])
def _fwd_split_kernel(
    Q,                      # [T_q, H, D]   bf16
    K,                      # [T_k, H, D]   bf16
    V,                      # [T_k, H, D]   bf16
    PartialO,               # [N_split_progs, H, BT, D]   fp32
    PartialLSE,             # [N_split_progs, H, BT]      fp32
    cu_seqlens_q,           # [n_seg+1]     int32
    cu_seqlens_k,           # [n_seg+1]     int32
    split_chunk_indices,    # [N_split_progs, 3]  (i_n, i_t, i_s) int32
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    SPLIT_K_BLOCK: tl.constexpr,
):
    i_prog = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(split_chunk_indices + i_prog * 3).to(tl.int32)
    i_t = tl.load(split_chunk_indices + i_prog * 3 + 1).to(tl.int32)
    i_s = tl.load(split_chunk_indices + i_prog * 3 + 2).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    q_start = i_t * BT
    k_split_start = i_s * SPLIT_K_BLOCK
    k_split_end = tl.minimum(k_split_start + SPLIT_K_BLOCK, T_k_seg)

    # Q-block guard: still possible if the last Q-block was just-past EOS
    # (n_q_blocks already excludes that, but the kernel is defensive).
    if q_start >= T_q_seg:
        return

    # --- Q load --------------------------------------------------------------
    p_q = tl.make_block_ptr(
        Q + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))

    # online softmax accumulators (fp32) -- same as v1 fwd
    b_o = tl.zeros([BT, D], dtype=tl.float32)
    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # --- K loop over THIS SPLIT only ---------------------------------------
    for k_start in range(k_split_start, k_split_end, BS):
        p_k = tl.make_block_ptr(
            K + (bos_k * H + i_h) * D,
            shape=(D, T_k_seg), strides=(1, H * D),
            offsets=(0, k_start), block_shape=(D, BS), order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            V + (bos_k * H + i_h) * D,
            shape=(T_k_seg, D), strides=(H * D, 1),
            offsets=(k_start, 0), block_shape=(BS, D), order=(1, 0),
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k) * scale * rcp_ln2

        # mask both: tail of K segment AND tail of this split
        o_k = k_start + tl.arange(0, BS)
        m_k = (o_k < T_k_seg) & (o_k < k_split_end)
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_m_safe = tl.where(b_m_new == float("-inf"), 0.0, b_m_new)
        b_r = tl.exp2(b_m - b_m_safe)
        b_p = tl.exp2(b_s - b_m_safe[:, None])

        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_m = b_m_new

    # --- finalize partial: write O_partial = O_unnorm / l, LSE = m + log2(l)
    # Guard: a fully-masked split (all rows -inf) shouldn't divide by 0.
    # b_acc is 0 in that case; we'd produce NaN. Replace with 1 for the
    # division and rely on LSE = -inf to make combine drop this slot.
    b_acc_safe = tl.where(b_m == float("-inf"), 1.0, b_acc)
    b_o = b_o / b_acc_safe[:, None]
    b_lse = b_m + tl.log2(b_acc_safe)
    # Restore -inf LSE for fully-masked rows so combine can ignore them.
    b_lse = tl.where(b_m == float("-inf"), float("-inf"), b_lse)

    # --- store partial -----------------------------------------------------
    # PartialO row stride = H*BT*D, head stride = BT*D, BT stride = D, D=1
    o_row_base = (i_prog * H + i_h) * BT * D
    p_partial_o = tl.make_block_ptr(
        PartialO + o_row_base,
        shape=(BT, D), strides=(D, 1),
        offsets=(0, 0), block_shape=(BT, D), order=(1, 0),
    )
    tl.store(p_partial_o, b_o, boundary_check=(0, 1))

    # PartialLSE row stride = H*BT, head stride = BT
    lse_row_base = (i_prog * H + i_h) * BT
    o_t = tl.arange(0, BT)
    m_t = (q_start + o_t) < T_q_seg
    tl.store(PartialLSE + lse_row_base + o_t, b_lse, mask=m_t)


# ===========================================================================
# Combine kernel for split-K forward.
# ===========================================================================
#
# Each program owns one Q-block of one head. It reads N_SPLITS partial
# entries from PartialO / PartialLSE (rows partial_start ... partial_start
# + n_splits - 1) and merges them via log2-domain online softmax.
#
# Math:
#   m       = max_s LSE_s
#   w_s     = exp2(LSE_s - m)        (zero for fully-masked splits because
#                                     LSE_s = -inf -> w_s = 0)
#   l       = sum_s w_s
#   O_final = sum_s (O_s * w_s) / l
#   LSE_out = m + log2(l)
#
# We accumulate O and l in fp32, then store O cast to bf16 and LSE as fp32.
#
# We don't unroll the split loop (it's variable across chunks). N_splits
# in production is bounded by ceil(T_k_max / SPLIT_K_BLOCK) ~ 25 at
# T_k_max=100k, SPLIT_K_BLOCK=4096 -- a runtime loop is fine.
@triton.jit
def _fwd_combine_kernel(
    PartialO,               # [N_split_progs, H, BT, D]   fp32
    PartialLSE,             # [N_split_progs, H, BT]      fp32
    O,                      # [T_q, H, D]                 bf16
    LSE,                    # [T_q, H]                    fp32  (log2 domain)
    cu_seqlens_q,           # [n_seg+1]                   int32
    combine_chunk_indices,  # [N_combine_progs, 4]  (i_n, i_t, partial_start,
                            #                         n_splits) int32
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    i_prog = tl.program_id(0)
    i_h = tl.program_id(1)

    i_n = tl.load(combine_chunk_indices + i_prog * 4).to(tl.int32)
    i_t = tl.load(combine_chunk_indices + i_prog * 4 + 1).to(tl.int32)
    p_start = tl.load(combine_chunk_indices + i_prog * 4 + 2).to(tl.int32)
    n_splits = tl.load(combine_chunk_indices + i_prog * 4 + 3).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    T_q_seg = eos_q - bos_q

    q_start = i_t * BT
    if q_start >= T_q_seg:
        return

    o_t = tl.arange(0, BT)
    o_d = tl.arange(0, D)
    m_t = (q_start + o_t) < T_q_seg

    # --- pass 1: find row-wise max LSE across splits ---------------------
    m_global = tl.full([BT], float("-inf"), dtype=tl.float32)
    for s in range(0, n_splits):
        prog = p_start + s
        lse_ptrs = PartialLSE + (prog * H + i_h) * BT + o_t
        b_lse = tl.load(lse_ptrs, mask=m_t, other=float("-inf"))
        m_global = tl.maximum(m_global, b_lse)

    # If every split was fully masked (shouldn't happen for a real long
    # segment, but be safe): write zeros and -inf LSE.
    all_masked = m_global == float("-inf")
    m_safe = tl.where(all_masked, 0.0, m_global)

    # --- pass 2: accumulate O and l -------------------------------------
    b_o = tl.zeros([BT, D], dtype=tl.float32)
    b_l = tl.zeros([BT], dtype=tl.float32)
    for s in range(0, n_splits):
        prog = p_start + s
        lse_ptrs = PartialLSE + (prog * H + i_h) * BT + o_t
        b_lse = tl.load(lse_ptrs, mask=m_t, other=float("-inf"))
        b_w = tl.exp2(b_lse - m_safe)              # 0 for -inf splits
        b_l += b_w

        o_ptrs = (
            PartialO
            + (prog * H + i_h) * BT * D
            + o_t[:, None] * D
            + o_d[None, :]
        )
        b_o_s = tl.load(o_ptrs, mask=m_t[:, None], other=0.0)
        b_o += b_o_s * b_w[:, None]

    b_l_safe = tl.where(b_l == 0.0, 1.0, b_l)
    b_o = b_o / b_l_safe[:, None]
    b_lse_out = m_safe + tl.log2(b_l_safe)
    b_lse_out = tl.where(all_masked, float("-inf"), b_lse_out)

    # --- store final O / LSE --------------------------------------------
    p_o = tl.make_block_ptr(
        O + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    # LSE write: must include q_start so each Q-block of a long segment
    # writes to its own slice of LSE. Without q_start every combine program
    # in the same segment would clobber lse[bos_q + 0..BT, i_h].
    lse_global_ptrs = LSE + (bos_q + q_start + o_t) * H + i_h
    tl.store(lse_global_ptrs, b_lse_out, mask=m_t)


# ===========================================================================
# Split-K backward dQ path (long segments only)
# ===========================================================================
#
# Same partitioning as fwd: each program owns one Q-block of one head AND
# a K-slice. It computes the partial dQ contribution from that K-slice
# only, and writes it to a fp32 buffer at row=program_id.
#
# Math (no online softmax accumulation needed -- LSE is already finalized
# by fwd, so each split's p_ij is an independent number):
#   p_ij    = exp2(s_ij - lse_i)    (s_ij in log2 domain)
#   dp_ij   = dO_i . v_j             (natural)
#   ds_ij   = p_ij * (dp_ij - delta_i)
#   dq_partial_i = sum_{j in this split} ds_ij * k_j     (no scale yet)
#
# Final scale (* scale) is applied in the combine kernel after summing
# all splits.
@triton.jit(do_not_specialize=["scale"])
def _bwd_dq_split_kernel(
    Q,                      # [T_q, H, D]   bf16
    K,                      # [T_k, H, D]   bf16
    V,                      # [T_k, H, D]   bf16
    DO,                     # [T_q, H, D]   bf16
    PartialDQ,              # [N_split_progs, H, BT, D]   fp32
    LSE,                    # [T_q, H]      fp32  (log2 domain)
    Delta,                  # [T_q, H]      fp32
    cu_seqlens_q,           # [n_seg+1]     int32
    cu_seqlens_k,           # [n_seg+1]     int32
    split_chunk_indices,    # [N_split_progs, 3]  (i_n, i_t, i_s) int32
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    SPLIT_K_BLOCK: tl.constexpr,
):
    i_prog = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(split_chunk_indices + i_prog * 3).to(tl.int32)
    i_t = tl.load(split_chunk_indices + i_prog * 3 + 1).to(tl.int32)
    i_s = tl.load(split_chunk_indices + i_prog * 3 + 2).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    q_start = i_t * BT
    k_split_start = i_s * SPLIT_K_BLOCK
    k_split_end = tl.minimum(k_split_start + SPLIT_K_BLOCK, T_k_seg)

    if q_start >= T_q_seg:
        return

    p_q = tl.make_block_ptr(
        Q + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_do = tl.make_block_ptr(
        DO + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    p_lse = tl.make_block_ptr(
        LSE + bos_q * H + i_h,
        shape=(T_q_seg,), strides=(H,),
        offsets=(q_start,), block_shape=(BT,), order=(0,),
    )
    p_delta = tl.make_block_ptr(
        Delta + bos_q * H + i_h,
        shape=(T_q_seg,), strides=(H,),
        offsets=(q_start,), block_shape=(BT,), order=(0,),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([BT, D], dtype=tl.float32)

    for k_start in range(k_split_start, k_split_end, BS):
        p_k = tl.make_block_ptr(
            K + (bos_k * H + i_h) * D,
            shape=(D, T_k_seg), strides=(1, H * D),
            offsets=(0, k_start), block_shape=(D, BS), order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            V + (bos_k * H + i_h) * D,
            shape=(D, T_k_seg), strides=(1, H * D),
            offsets=(0, k_start), block_shape=(D, BS), order=(0, 1),
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k) * scale * rcp_ln2

        o_k = k_start + tl.arange(0, BS)
        m_k = (o_k < T_k_seg) & (o_k < k_split_end)
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_p = tl.exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v).to(tl.float32)
        b_ds = b_p * (b_dp - b_delta[:, None])

        b_dq += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_k))

    # NOTE: no `* scale` here. Combine kernel multiplies by scale once
    # after summing across splits, so the chain rule is applied exactly
    # once regardless of n_splits.

    p_partial_dq_base = PartialDQ + (i_prog * H + i_h) * BT * D
    p_partial_dq = tl.make_block_ptr(
        p_partial_dq_base,
        shape=(BT, D), strides=(D, 1),
        offsets=(0, 0), block_shape=(BT, D), order=(1, 0),
    )
    tl.store(p_partial_dq, b_dq, boundary_check=(0, 1))


# ===========================================================================
# Combine kernel for split-K bwd_dq.
# ===========================================================================
#
# Each program owns one Q-block of one head. It sums n_splits partial dQ
# entries (fp32) and writes the final dQ * scale (bf16).
@triton.jit(do_not_specialize=["scale"])
def _bwd_dq_combine_kernel(
    PartialDQ,              # [N_split_progs, H, BT, D]   fp32
    DQ,                     # [T_q, H, D]                 bf16
    cu_seqlens_q,           # [n_seg+1]                   int32
    combine_chunk_indices,  # [N_combine_progs, 4]        int32
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    i_prog = tl.program_id(0)
    i_h = tl.program_id(1)

    i_n = tl.load(combine_chunk_indices + i_prog * 4).to(tl.int32)
    i_t = tl.load(combine_chunk_indices + i_prog * 4 + 1).to(tl.int32)
    p_start = tl.load(combine_chunk_indices + i_prog * 4 + 2).to(tl.int32)
    n_splits = tl.load(combine_chunk_indices + i_prog * 4 + 3).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    T_q_seg = eos_q - bos_q

    q_start = i_t * BT
    if q_start >= T_q_seg:
        return

    o_t = tl.arange(0, BT)
    o_d = tl.arange(0, D)
    m_t = (q_start + o_t) < T_q_seg

    b_dq = tl.zeros([BT, D], dtype=tl.float32)
    for s in range(0, n_splits):
        prog = p_start + s
        ptrs = (
            PartialDQ
            + (prog * H + i_h) * BT * D
            + o_t[:, None] * D
            + o_d[None, :]
        )
        b_dq += tl.load(ptrs, mask=m_t[:, None], other=0.0)

    b_dq = b_dq * scale

    p_dq = tl.make_block_ptr(
        DQ + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT, D), order=(1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# Fused backward kernel for long segments (B3b).
# ===========================================================================
#
# One program per (segment, k_block, head). It scans every Q-block in
# the segment ONCE and:
#   - accumulates dk/dv inside the program. Each K-block is owned by
#     exactly one program, so we can store final dk/dv at the end (no
#     atomic, no partial buffer).
#   - emits a fp32 partial dq per (k_block, q_block) into PartialDQ.
#     A subsequent combine kernel sums those partials per (segment,
#     q_block) into the final dq.
#
# This halves HBM traffic vs the v1 split-kernel bwd: K is read once for
# both dq and dk/dv contributions, instead of once per pass.
#
# Math (recap, log2-domain LSE):
#   p_ij  = exp2(s_ij - lse_i)
#   dp_ij = dO_i . v_j
#   ds_ij = p_ij * (dp_ij - delta_i)
#   dv_j  += sum_i p_ij * dO_i
#   dk_j  += sum_i ds_ij * q_i
#   dq_partial[k_block, q_block, i] = sum_{j in k_block} ds_ij * k_j
#
# dq partial does NOT multiply by `scale` here; the combine kernel
# multiplies by scale once after summing across k_blocks (so the chain
# rule is applied exactly once regardless of n_k_progs).
@triton.jit(do_not_specialize=["scale"])
def _bwd_fused_kernel(
    Q,                      # [T_q, H, D]   bf16
    K,                      # [T_k, H, D]   bf16
    V,                      # [T_k, H, D]   bf16
    DO,                     # [T_q, H, D]   bf16
    DK,                     # [T_k, H, D]   bf16  (final, written here)
    DV,                     # [T_k, H, D]   bf16  (final, written here)
    PartialDQ,              # [N_kprog * MAX_Q_BLOCKS, H, BT_Q, D]  fp32
    LSE,                    # [T_q, H]      fp32  (log2 domain)
    Delta,                  # [T_q, H]      fp32
    cu_seqlens_q,           # [n_seg+1]     int32
    cu_seqlens_k,           # [n_seg+1]     int32
    kprog_indices,          # [N_kprog, 3]  (i_n, i_k_block, q_partial_base)
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT_Q: tl.constexpr,     # Q block size (matches fwd BT)
    BT_K: tl.constexpr,     # K block size (this program owns BT_K K tokens)
):
    i_kprog = tl.program_id(0)
    i_h = tl.program_id(1)

    rcp_ln2 = 1.4426950408889634

    i_n = tl.load(kprog_indices + i_kprog * 3).to(tl.int32)
    i_k_block = tl.load(kprog_indices + i_kprog * 3 + 1).to(tl.int32)
    q_partial_base = tl.load(kprog_indices + i_kprog * 3 + 2).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)

    T_q_seg = eos_q - bos_q
    T_k_seg = eos_k - bos_k

    k_start = i_k_block * BT_K
    if k_start >= T_k_seg:
        return

    # --- load K, V for this k-block (resident throughout) ----------------
    p_k = tl.make_block_ptr(
        K + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT_K, D), order=(1, 0),
    )
    p_v = tl.make_block_ptr(
        V + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT_K, D), order=(1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))   # [BT_K, D]
    b_v = tl.load(p_v, boundary_check=(0, 1))   # [BT_K, D]

    # dk/dv accumulators (fp32) -- this program owns this k-block exclusively
    b_dk = tl.zeros([BT_K, D], dtype=tl.float32)
    b_dv = tl.zeros([BT_K, D], dtype=tl.float32)

    # --- loop over every Q-block in this segment ------------------------
    n_q_blocks = (T_q_seg + BT_Q - 1) // BT_Q
    for q_block in range(0, n_q_blocks):
        q_start = q_block * BT_Q

        # Q^T as [D, BT_Q]
        p_q = tl.make_block_ptr(
            Q + (bos_q * H + i_h) * D,
            shape=(D, T_q_seg), strides=(1, H * D),
            offsets=(0, q_start), block_shape=(D, BT_Q), order=(0, 1),
        )
        p_do = tl.make_block_ptr(
            DO + (bos_q * H + i_h) * D,
            shape=(D, T_q_seg), strides=(1, H * D),
            offsets=(0, q_start), block_shape=(D, BT_Q), order=(0, 1),
        )
        # LSE / delta
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

        b_q = tl.load(p_q, boundary_check=(0, 1))     # [D, BT_Q]
        b_do = tl.load(p_do, boundary_check=(0, 1))   # [D, BT_Q]
        b_lse = tl.load(p_lse, boundary_check=(0,))   # [BT_Q]
        b_delta = tl.load(p_delta, boundary_check=(0,))  # [BT_Q]

        # s = (k @ q) * scale * rcp_ln2 : [BT_K, D] @ [D, BT_Q] -> [BT_K, BT_Q]
        b_s = tl.dot(b_k, b_q) * scale * rcp_ln2

        # mask Q tail (positions >= T_q_seg) -> -inf so p = 0
        o_q = q_start + tl.arange(0, BT_Q)
        m_q = o_q < T_q_seg
        b_s = tl.where(m_q[None, :], b_s, float("-inf"))

        # mask K tail (positions >= T_k_seg) so dk/dv don't pick up garbage
        o_k = k_start + tl.arange(0, BT_K)
        m_k = o_k < T_k_seg
        b_s = tl.where(m_k[:, None], b_s, float("-inf"))

        b_p = tl.exp2(b_s - b_lse[None, :])           # [BT_K, BT_Q]

        # dv += p @ dO : [BT_K, BT_Q] @ [BT_Q, D] -> [BT_K, D]
        b_dv += tl.dot(b_p.to(b_k.dtype), tl.trans(b_do))

        # dp = v @ dO : [BT_K, D] @ [D, BT_Q] -> [BT_K, BT_Q]
        b_dp = tl.dot(b_v, b_do).to(tl.float32)
        # ds = p * (dp - delta)
        b_ds = b_p * (b_dp - b_delta[None, :])

        # dk += ds @ Q : [BT_K, BT_Q] @ [BT_Q, D] -> [BT_K, D]
        b_dk += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q))

        # dq partial: contribution from THIS k-block to THIS q-block.
        #   shape: [BT_Q, D] = [BT_Q, BT_K] @ [BT_K, D]
        # we have b_ds as [BT_K, BT_Q]; transpose to [BT_Q, BT_K], then
        # multiply by b_k [BT_K, D].
        b_dq_partial = tl.dot(tl.trans(b_ds).to(b_k.dtype), b_k)

        # store partial at row (q_partial_base + q_block) -- one row per
        # (k_prog, q_block) pair. NO scale here; combine multiplies once.
        partial_row_base = (
            (q_partial_base + q_block) * H + i_h
        ) * BT_Q * D
        p_partial_dq = tl.make_block_ptr(
            PartialDQ + partial_row_base,
            shape=(BT_Q, D), strides=(D, 1),
            offsets=(0, 0), block_shape=(BT_Q, D), order=(1, 0),
        )
        tl.store(p_partial_dq, b_dq_partial, boundary_check=(0, 1))

    # --- store final dk/dv (this k-block has unique owner) -------------
    b_dk = b_dk * scale
    p_dk = tl.make_block_ptr(
        DK + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT_K, D), order=(1, 0),
    )
    p_dv = tl.make_block_ptr(
        DV + (bos_k * H + i_h) * D,
        shape=(T_k_seg, D), strides=(H * D, 1),
        offsets=(k_start, 0), block_shape=(BT_K, D), order=(1, 0),
    )
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# dq combine for fused bwd.
# ===========================================================================
#
# One program per (segment, q_block, head). It sums n_k_progs partial dQ
# rows from PartialDQ (stride MAX_Q_BLOCKS between consecutive k_progs)
# and writes the final dQ * scale (bf16).
@triton.jit(do_not_specialize=["scale"])
def _bwd_dq_combine_fused_kernel(
    PartialDQ,              # [N_kprog * MAX_Q_BLOCKS, H, BT_Q, D]  fp32
    DQ,                     # [T_q, H, D]                          bf16
    cu_seqlens_q,           # [n_seg+1]                            int32
    dq_combine_indices,     # [N_dq_combine, 4]    (i_n, i_q_block,
                            #                       partial_start, n_k_progs)
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT_Q: tl.constexpr,
    MAX_Q_BLOCKS: tl.constexpr,
):
    i_prog = tl.program_id(0)
    i_h = tl.program_id(1)

    i_n = tl.load(dq_combine_indices + i_prog * 4).to(tl.int32)
    i_q_block = tl.load(dq_combine_indices + i_prog * 4 + 1).to(tl.int32)
    p_start = tl.load(dq_combine_indices + i_prog * 4 + 2).to(tl.int32)
    n_k_progs = tl.load(dq_combine_indices + i_prog * 4 + 3).to(tl.int32)

    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    T_q_seg = eos_q - bos_q

    q_start = i_q_block * BT_Q
    if q_start >= T_q_seg:
        return

    o_t = tl.arange(0, BT_Q)
    o_d = tl.arange(0, D)
    m_t = (q_start + o_t) < T_q_seg

    b_dq = tl.zeros([BT_Q, D], dtype=tl.float32)
    for s in range(0, n_k_progs):
        # row index in PartialDQ for this k_prog and q_block:
        #   p_start (which already includes this q_block offset)
        #     + s * MAX_Q_BLOCKS                     (stride between k_progs)
        row = p_start + s * MAX_Q_BLOCKS
        ptrs = (
            PartialDQ
            + (row * H + i_h) * BT_Q * D
            + o_t[:, None] * D
            + o_d[None, :]
        )
        b_dq += tl.load(ptrs, mask=m_t[:, None], other=0.0)

    b_dq = b_dq * scale

    p_dq = tl.make_block_ptr(
        DQ + (bos_q * H + i_h) * D,
        shape=(T_q_seg, D), strides=(H * D, 1),
        offsets=(q_start, 0), block_shape=(BT_Q, D), order=(1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
