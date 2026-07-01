"""
Python autograd Function + user-facing wrapper for the Triton ragged target
attention (v1).

The wrapper layers responsibility this way:

  packed_target_attn(q_raw, X, W_Q, W_K, W_V, cu_seqlens_q, cu_seqlens_k, ...)
    │
    │ outer Python (PyTorch autograd handles dq_raw / dW_Q / dW_K automatically)
    │
    ├── u = q_raw @ W_Q @ W_K^T              shape [T_q, H, D_KV]
    │
    │   ┌─────────── _TargetAttnFunction.forward ───────────┐
    │   │  Triton _fwd_kernel(u, X, W_V) -> O, LSE          │
    │   └────────────────────────────────────────────────────┘
    │
    │   ┌─────────── _TargetAttnFunction.backward ──────────┐
    │   │  _bwd_preprocess_kernel:   O, dO -> delta          │
    │   │  _bwd_du_kernel:           u,X,W_V,dO,lse,delta    │
    │   │                              -> du, dαX, atomic dW_V│
    │   │  _bwd_dx_kernel:           u,X,dαX,lse,delta       │
    │   │                              -> per-head dX partial │
    │   │  _bwd_dx_reduce_kernel:    sum heads -> dX_fp32     │
    │   │  or _bwd_dx_fused_heads_kernel for large K batches  │
    │   │  cast: dW_V_fp32 -> dW_V,  dX_fp32 -> dX            │
    │   └─────────────────────────────────────────────────────┘
    │
    └── return O
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import triton

from triton_target_attn_absorb.kernels import (
    _fwd_kernel,
    _fwd_alpha_kernel,
    _bwd_preprocess_kernel,
    _bwd_du_kernel,
    _bwd_dx_kernel,
    _bwd_dx_reduce_kernel,
    _bwd_dx_fused_heads_kernel,
    _bwd_dx_atomic_heads_kernel,
)
from triton_target_attn_absorb.chunk_indices import (
    prepare_q_chunk_indices,
    prepare_k_chunk_indices,
)


# Default block sizes for v1. Tunable; no autotune yet.
# Keep these conservative: fp32 + D_KV=216 pads to B_DKV=256, and the fused
# W_V tile is large enough that 32x64 can exceed SM shared memory on common GPUs.
_BT_Q_FWD = 16      # Q block for fwd / bwd_du
_BT_K_FWD = 64      # K block for fwd / bwd_du
_BT_K_BWD = 64      # K block for bwd_dx (outer)
_BT_Q_BWD = 16      # Q block for bwd_dx (inner)
_DX_FUSED_HEADS_MIN_K_CHUNKS = 1024


def _select_fwd_mode() -> str:
    mode = os.environ.get("TRITON_TARGET_ATTN_FWD_MODE", "alpha").lower()
    if mode in ("direct", "alpha"):
        return mode
    raise ValueError("TRITON_TARGET_ATTN_FWD_MODE must be one of: direct, alpha")


def _select_dx_mode(n_k_chunks: int) -> str:
    mode = os.environ.get("TRITON_TARGET_ATTN_DX_MODE", "auto").lower()
    if mode == "partial":
        return "partial"
    if mode == "fused":
        return "fused"
    if mode == "atomic":
        return "atomic"
    if mode != "auto":
        raise ValueError(
            "TRITON_TARGET_ATTN_DX_MODE must be one of: auto, partial, fused, atomic"
        )
    return "fused" if n_k_chunks >= _DX_FUSED_HEADS_MIN_K_CHUNKS else "partial"


def _use_fused_heads_dx(n_k_chunks: int) -> bool:
    return _select_dx_mode(n_k_chunks) == "fused"


def _validate_cu_seqlens(
    cu: torch.Tensor,
    T: int,
    name: str,
) -> None:
    if cu.dim() != 1 or cu.numel() < 2:
        raise ValueError(f"{name}: must be 1-D with len>=2, got {tuple(cu.shape)}")
    cu_cpu = cu.detach().cpu()
    if int(cu_cpu[0]) != 0:
        raise ValueError(f"{name}[0] must be 0")
    if int(cu_cpu[-1]) != T:
        raise ValueError(f"{name}[-1] must equal total tokens {T}, got {int(cu_cpu[-1])}")
    diffs = cu_cpu[1:] - cu_cpu[:-1]
    if (diffs < 0).any().item():
        raise ValueError(f"{name} must be non-decreasing")


def _validate_active_segments_have_keys(
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
) -> None:
    q_lens = cu_q.detach().cpu()[1:] - cu_q.detach().cpu()[:-1]
    k_lens = cu_k.detach().cpu()[1:] - cu_k.detach().cpu()[:-1]
    if ((q_lens > 0) & (k_lens <= 0)).any().item():
        raise ValueError("segments with q_len > 0 must have k_len > 0")


def _next_power_of_2(x: int) -> int:
    if x < 1:
        raise ValueError(f"expected positive dimension, got {x}")
    return 1 << (x - 1).bit_length()


def prepare_target_attn_indices(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute chunk metadata for the current v1 block sizes.

    Build these once per packed batch and pass them into packed_target_attn with
    validate=False to keep CPU metadata work out of the hot path.
    """
    return (
        prepare_q_chunk_indices(cu_seqlens_q, _BT_Q_FWD),
        prepare_k_chunk_indices(cu_seqlens_k, _BT_K_BWD),
    )


class _TargetAttnFunction(torch.autograd.Function):
    """
    Inner autograd Function — operates on already-projected u (not q_raw).
    Inputs:
      u:   [T_q, H, D_KV]
      X:   [T_k, D_KV]
      W_V: [D_KV, H, D_H]
    Returns:
      O:   [T_q, H, D_H]
    """

    @staticmethod
    def forward(
        ctx,
        u,
        X,
        W_V,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        q_chunk_indices,
        k_chunk_indices,
        validate,
    ):
        T_q, H, D_KV = u.shape
        T_k, D_KV_x = X.shape
        D_KV_wv, H_wv, D_H = W_V.shape

        assert D_KV == D_KV_x == D_KV_wv, \
            f"D_KV mismatch: u={D_KV}, X={D_KV_x}, W_V={D_KV_wv}"
        assert H == H_wv, f"H mismatch: u={H}, W_V={H_wv}"
        assert u.dtype == X.dtype == W_V.dtype, \
            f"u/X/W_V must share dtype, got {u.dtype}/{X.dtype}/{W_V.dtype}"
        assert u.is_cuda and X.is_cuda and W_V.is_cuda, "all inputs must be CUDA"
        assert u.device == X.device == W_V.device, "u/X/W_V must be on same device"
        assert D_H in (16, 32, 64, 128, 256), f"D_H {D_H} not supported"

        cu_seqlens_q = cu_seqlens_q.to(device=u.device, dtype=torch.int32).contiguous()
        cu_seqlens_k = cu_seqlens_k.to(device=u.device, dtype=torch.int32).contiguous()
        if validate:
            n_seg = cu_seqlens_q.numel() - 1
            assert cu_seqlens_k.numel() - 1 == n_seg, "segment count mismatch q vs k"
            _validate_cu_seqlens(cu_seqlens_q, T_q, "cu_seqlens_q")
            _validate_cu_seqlens(cu_seqlens_k, T_k, "cu_seqlens_k")
            _validate_active_segments_have_keys(cu_seqlens_q, cu_seqlens_k)

        # Kernels use fixed contiguous strides, not arbitrary tensor strides.
        u = u.contiguous()
        X = X.contiguous()
        W_V = W_V.contiguous()
        if q_chunk_indices is None:
            q_chunk_indices = prepare_q_chunk_indices(cu_seqlens_q, _BT_Q_FWD)
        else:
            q_chunk_indices = q_chunk_indices.to(device=u.device, dtype=torch.int32).contiguous()
        if k_chunk_indices is None:
            k_chunk_indices = prepare_k_chunk_indices(cu_seqlens_k, _BT_K_BWD)
        else:
            k_chunk_indices = k_chunk_indices.to(device=u.device, dtype=torch.int32).contiguous()

        B_DKV = _next_power_of_2(D_KV)
        assert B_DKV <= 512, f"D_KV={D_KV} pads to {B_DKV}, which is too large for v1 tiles"

        O = torch.empty((T_q, H, D_H), dtype=u.dtype, device=u.device)
        LSE = torch.empty((T_q, H), dtype=torch.float32, device=u.device)

        if q_chunk_indices.shape[0] > 0:
            grid = (q_chunk_indices.shape[0], H)
            fwd_kernel = _fwd_alpha_kernel if _select_fwd_mode() == "alpha" else _fwd_kernel
            fwd_kernel[grid](
                u, X, W_V, O, LSE,
                cu_seqlens_q, cu_seqlens_k, q_chunk_indices,
                scale,
                H=H, D_KV=D_KV, B_DKV=B_DKV, D_H=D_H,
                BT_Q=_BT_Q_FWD, BT_K=_BT_K_FWD,
                num_warps=4, num_stages=1,
            )

        ctx.save_for_backward(
            u, X, W_V, O, LSE, cu_seqlens_q, cu_seqlens_k,
            q_chunk_indices, k_chunk_indices,
        )
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        (
            u, X, W_V, O, LSE, cu_seqlens_q, cu_seqlens_k,
            q_chunk_indices, k_chunk_indices,
        ) = ctx.saved_tensors
        scale = ctx.scale
        T_q, H, D_KV = u.shape
        T_k = X.shape[0]
        D_H = W_V.shape[2]
        B_DKV = _next_power_of_2(D_KV)

        if dO.stride(-1) != 1 or not dO.is_contiguous():
            dO = dO.contiguous()
        if dO.dtype != u.dtype:
            dO = dO.to(u.dtype)

        # delta[t, h] = sum_d O[t,h,d] * dO[t,h,d] in fp32
        delta = torch.empty((T_q, H), dtype=torch.float32, device=u.device)
        grid_pre = (triton.cdiv(T_q, _BT_Q_FWD), H)
        _bwd_preprocess_kernel[grid_pre](
            O, dO, delta,
            T_q,
            H=H, D_H=D_H, BT=_BT_Q_FWD,
            num_warps=4, num_stages=2,
        )

        du = torch.empty_like(u)
        # dW_V goes through an fp32 atomic-add buffer. dX is first written as
        # per-head partials, then reduced across heads in a separate kernel.
        dAlphaX_fp32 = torch.empty(
            (T_q, H, D_KV), dtype=torch.float32, device=u.device
        )
        dW_V_fp32 = torch.zeros(
            (D_KV, H, D_H), dtype=torch.float32, device=u.device
        )
        dX_fp32 = torch.empty(
            (T_k, D_KV), dtype=torch.float32, device=u.device
        )

        if q_chunk_indices.shape[0] > 0:
            grid_du = (q_chunk_indices.shape[0], H)
            _bwd_du_kernel[grid_du](
                u, X, W_V, dO, du, dAlphaX_fp32, dW_V_fp32, LSE, delta,
                cu_seqlens_q, cu_seqlens_k, q_chunk_indices,
                scale,
                H=H, D_KV=D_KV, B_DKV=B_DKV, D_H=D_H,
                BT_Q=_BT_Q_FWD, BT_K=_BT_K_FWD,
                num_warps=4, num_stages=1,
            )

        if k_chunk_indices.shape[0] > 0:
            dx_mode = _select_dx_mode(k_chunk_indices.shape[0])
            if dx_mode == "fused":
                _bwd_dx_fused_heads_kernel[(k_chunk_indices.shape[0],)](
                    u, X, dAlphaX_fp32, dX_fp32, LSE, delta,
                    cu_seqlens_q, cu_seqlens_k, k_chunk_indices,
                    scale,
                    H=H, D_KV=D_KV, B_DKV=B_DKV, D_H=D_H,
                    BT_K=_BT_K_BWD, BT_Q=_BT_Q_BWD,
                    num_warps=4, num_stages=1,
                )
            elif dx_mode == "atomic":
                dX_fp32.zero_()
                _bwd_dx_atomic_heads_kernel[(k_chunk_indices.shape[0], H)](
                    u, X, dAlphaX_fp32, dX_fp32, LSE, delta,
                    cu_seqlens_q, cu_seqlens_k, k_chunk_indices,
                    scale,
                    H=H, D_KV=D_KV, B_DKV=B_DKV, D_H=D_H,
                    BT_K=_BT_K_BWD, BT_Q=_BT_Q_BWD,
                    num_warps=4, num_stages=1,
                )
            else:
                dX_partial_fp32 = torch.empty(
                    (H, T_k, D_KV), dtype=torch.float32, device=u.device
                )
                grid_dx = (k_chunk_indices.shape[0], H)
                _bwd_dx_kernel[grid_dx](
                    u, X, dAlphaX_fp32, dX_partial_fp32, T_k, LSE, delta,
                    cu_seqlens_q, cu_seqlens_k, k_chunk_indices,
                    scale,
                    H=H, D_KV=D_KV, B_DKV=B_DKV, D_H=D_H,
                    BT_K=_BT_K_BWD, BT_Q=_BT_Q_BWD,
                    num_warps=4, num_stages=1,
                )
                _bwd_dx_reduce_kernel[(triton.cdiv(T_k, _BT_K_BWD),)](
                    dX_partial_fp32, dX_fp32, T_k,
                    H=H, D_KV=D_KV, B_DKV=B_DKV, BT_K=_BT_K_BWD,
                    num_warps=4, num_stages=1,
                )
        else:
            dX_fp32.zero_()

        dX = dX_fp32.to(u.dtype)
        dW_V = dW_V_fp32.to(u.dtype)

        return du, dX, dW_V, None, None, None, None, None, None


def target_attn_inner(
    u: torch.Tensor,
    X: torch.Tensor,
    W_V: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: Optional[float] = None,
    q_chunk_indices: Optional[torch.Tensor] = None,
    k_chunk_indices: Optional[torch.Tensor] = None,
    validate: bool = True,
) -> torch.Tensor:
    """
    Inner entry: takes already-projected u = q_raw @ W_Q @ W_K^T.

    Args:
        u:            [T_q, H, D_KV]
        X:            [T_k, D_KV]
        W_V:          [D_KV, H, D_H]
        cu_seqlens_q: [n_seg+1] int32
        cu_seqlens_k: [n_seg+1] int32
        scale:        defaults to 1/sqrt(D_H)

    Returns:
        O: [T_q, H, D_H]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(W_V.shape[2])
    return _TargetAttnFunction.apply(
        u, X, W_V, cu_seqlens_q, cu_seqlens_k, float(scale),
        q_chunk_indices, k_chunk_indices, validate,
    )


def packed_target_attn(
    q_raw: torch.Tensor,
    X: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_seqinfo: Optional[torch.Tensor] = None,
    k_seqinfo: Optional[torch.Tensor] = None,
    q_chunk_indices: Optional[torch.Tensor] = None,
    k_chunk_indices: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    bf16: bool = False,
    validate: bool = True,
) -> torch.Tensor:
    """
    Full ragged target attention with matrix absorption.

    Args:
        q_raw: [T_q, d_q] or [1, T_q, d_q]
        X:     [T_k, d_kv] or [1, T_k, d_kv]
        W_Q:   [d_q,  H, D_H]
        W_K:   [d_kv, H, D_H]
        W_V:   [d_kv, H, D_H]
        cu_seqlens_q / cu_seqlens_k: [n_seg+1] int32 (preferred)
        q_seqinfo / k_seqinfo: [n_seg] per-segment lengths (alternative)
        q_chunk_indices / k_chunk_indices: optional precomputed metadata from
            prepare_target_attn_indices; combine with validate=False to avoid
            CPU-side validation and metadata construction in hot paths.
        scale: defaults to 1/sqrt(D_H)
        bf16:  if True, cast q_raw / X / W_V to bf16 before calling the kernel;
               otherwise everything is fp32.

    Returns:
        O: [T_q, H, D_H] (same dtype as the kernel's I/O dtype — fp32 or bf16)
    """
    # squeeze batch=1 if present
    if q_raw.dim() == 3:
        assert q_raw.shape[0] == 1, "batch must be 1 for packed varlen"
        q_raw = q_raw.squeeze(0)
    if X.dim() == 3:
        assert X.shape[0] == 1
        X = X.squeeze(0)

    T_q, d_q = q_raw.shape
    T_k, d_kv = X.shape
    d_q_wq, H, D_H = W_Q.shape
    d_kv_wk, H_wk, D_H_wk = W_K.shape
    d_kv_wv, H_wv, D_H_wv = W_V.shape

    assert d_q_wq == d_q, f"W_Q[0]={d_q_wq} != d_q={d_q}"
    assert d_kv_wk == d_kv and d_kv_wv == d_kv, "W_K/W_V d_kv mismatch"
    assert H == H_wk == H_wv, "H mismatch across W_Q/W_K/W_V"
    assert D_H == D_H_wk == D_H_wv, "D_H mismatch across W_Q/W_K/W_V"

    # cu_seqlens
    if cu_seqlens_q is None:
        assert q_seqinfo is not None
        cu_seqlens_q = _seqinfo_to_cu(q_seqinfo)
    if cu_seqlens_k is None:
        assert k_seqinfo is not None
        cu_seqlens_k = _seqinfo_to_cu(k_seqinfo)

    if bf16:
        q_raw = q_raw.to(torch.bfloat16) if q_raw.dtype != torch.bfloat16 else q_raw
        X     = X.to(torch.bfloat16)     if X.dtype     != torch.bfloat16 else X
        W_Q   = W_Q.to(torch.bfloat16)   if W_Q.dtype   != torch.bfloat16 else W_Q
        W_K   = W_K.to(torch.bfloat16)   if W_K.dtype   != torch.bfloat16 else W_K
        W_V   = W_V.to(torch.bfloat16)   if W_V.dtype   != torch.bfloat16 else W_V

    # u = q_raw @ W_Q @ W_K^T, shape [T_q, H, d_kv]
    # Outer GEMM, PyTorch autograd handles dq_raw / dW_Q / dW_K automatically.
    # Compute as: q_proj = q_raw @ W_Q.view(d_q, H*D_H) -> [T_q, H*D_H]
    #             q_proj = q_proj.view(T_q, H, D_H)
    #             u = einsum('thd, khd -> thk', q_proj, W_K)  # [T_q, H, d_kv]
    q_proj = (q_raw @ W_Q.reshape(d_q, H * D_H)).view(T_q, H, D_H)
    u = torch.einsum('thd, khd -> thk', q_proj, W_K)  # [T_q, H, d_kv]

    if scale is None:
        scale = 1.0 / math.sqrt(D_H)

    return target_attn_inner(
        u, X, W_V, cu_seqlens_q, cu_seqlens_k, scale,
        q_chunk_indices=q_chunk_indices,
        k_chunk_indices=k_chunk_indices,
        validate=validate,
    )


def _seqinfo_to_cu(seqinfo: torch.Tensor) -> torch.Tensor:
    """[n_seg] per-segment lengths -> [n_seg+1] cu_seqlens int32."""
    assert seqinfo.dim() == 1 and seqinfo.numel() >= 1
    cs = seqinfo.to(torch.int64).cumsum(0)
    cu = torch.zeros(seqinfo.numel() + 1, dtype=torch.int64, device=seqinfo.device)
    cu[1:] = cs
    return cu.to(torch.int32)
