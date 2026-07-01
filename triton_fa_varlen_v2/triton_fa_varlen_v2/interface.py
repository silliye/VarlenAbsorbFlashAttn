"""
Python autograd Function + user-facing wrapper for the Triton packed-varlen
FlashAttention.

The wrapper:
  - accepts q/k/v as either [1, T, H, D] (matches business call site) or
    [T, H, D] (matches FA2 varlen API). Squeezes/unsqueezes as needed.
  - converts q_seqinfo / k_seqinfo (per-segment counts) to cu_seqlens (prefix
    sums). If the caller already passes cu_seqlens, that's also accepted.
  - launches forward and backward Triton kernels.

Only MHA (HQ == H) is supported.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton

from triton_fa_varlen_v2.kernels import (
    _fwd_kernel,
    _bwd_preprocess_kernel,
    _bwd_dq_kernel,
    _bwd_dkv_kernel,
    _fwd_split_kernel,
    _fwd_combine_kernel,
    _bwd_dq_split_kernel,
    _bwd_dq_combine_kernel,
)
from triton_fa_varlen_v2.chunk_indices import (
    prepare_q_chunk_indices,
    prepare_k_chunk_indices,
    prepare_split_indices,
)


# Block sizes for v2 (still using v1's defaults; tune in this branch).
_FWD_BT = 128   # Q block in fwd / bwd_dq
_FWD_BS = 32    # K block in fwd / bwd_dq
_BWD_DKV_BT = 128  # K block in bwd_dkv (short path) and fused (long path)
_BWD_DKV_BS = 32   # Q block in bwd_dkv (short path)

# Split-K parameters: any segment with T_k_seg > _SPLIT_THRESHOLD goes
# through the split + combine path. The threshold is set high enough that
# Stage 2's uniform 500-token segments stay on the v1 single-program
# fast path (no partial-buffer overhead, no extra launch). Stage 1's
# long history segments (T_k_seg up to ~80k) take the split path.
#
# _SPLIT_K_BLOCK is the number of K tokens each split program owns;
# 4096 = 8 launches per 32k segment, 16 per 64k -- both comfortable.
# Must be a multiple of _FWD_BS so the inner K loop ends on a clean
# boundary (otherwise the BS tail mask kicks in and we waste compute).
_SPLIT_THRESHOLD = 4096
_SPLIT_K_BLOCK = 4096
assert _SPLIT_K_BLOCK % _FWD_BS == 0, "SPLIT_K_BLOCK must be multiple of FWD_BS"

# Validate cu_seqlens by default. Set to False (e.g. via
# `set_validate_cu_seqlens(False)`) on hot training paths after the first few
# batches if profiling shows the host-side check is visible — most workloads
# build cu_seqlens once up-front and the check is a small cost relative to a
# full attn forward.
_VALIDATE_CU_SEQLENS = True


def set_validate_cu_seqlens(enabled: bool) -> None:
    """Toggle the host-side cu_seqlens validation. Off = trust the caller."""
    global _VALIDATE_CU_SEQLENS
    _VALIDATE_CU_SEQLENS = bool(enabled)


def _seqinfo_to_cu_seqlens(seqinfo: torch.Tensor) -> torch.Tensor:
    """[n_seg] (per-segment lengths) -> [n_seg+1] cu_seqlens int32."""
    if seqinfo.dim() != 1:
        raise ValueError(f"seqinfo must be 1-D [n_seg], got shape {tuple(seqinfo.shape)}")
    if seqinfo.numel() == 0:
        raise ValueError("seqinfo is empty (n_seg=0)")
    # negative segment lengths are nonsensical and would silently produce a
    # non-monotonic cu_seqlens that the kernel would dereference past EOS.
    if (seqinfo < 0).any().item():
        raise ValueError("seqinfo contains negative values")
    # cumsum on int32 can overflow with large totals. Compute in int64 then cast.
    cs = seqinfo.to(torch.int64).cumsum(0)
    cu = torch.zeros(seqinfo.numel() + 1, dtype=torch.int64, device=seqinfo.device)
    cu[1:] = cs
    # int32 head-room: total tokens must fit in int32 (~2.1B). Business worst
    # case is ~2.8M, so this is a safety net for misuse rather than a real limit.
    total = int(cs[-1].item())
    if total > 2_147_483_647:
        raise ValueError(f"total tokens {total} exceeds int32 range")
    return cu.to(torch.int32)


def _validate_cu_seqlens(cu: torch.Tensor, T: int, name: str) -> None:
    """Check cu_seqlens is a valid prefix-sum: starts at 0, non-decreasing, ends at T.

    Cost: one D2H copy of cu_seqlens (tiny, n_seg+1 elements) plus host-side
    arithmetic. Skipped entirely when _VALIDATE_CU_SEQLENS is False.
    """
    if not _VALIDATE_CU_SEQLENS:
        return
    if cu.dim() != 1:
        raise ValueError(f"{name}: must be 1-D, got shape {tuple(cu.shape)}")
    if cu.numel() < 2:
        raise ValueError(f"{name}: must have at least 2 elements (n_seg+1 with n_seg>=1)")
    # cu_seqlens is small (n_seg+1). One D2H copy here covers all checks.
    cu_cpu = cu.detach().cpu()
    first = int(cu_cpu[0].item())
    last = int(cu_cpu[-1].item())
    if first != 0:
        raise ValueError(f"{name}[0] must be 0, got {first}")
    if last != T:
        raise ValueError(f"{name}[-1] must equal total tokens {T}, got {last}")
    diffs = cu_cpu[1:] - cu_cpu[:-1]
    if (diffs < 0).any().item():
        raise ValueError(f"{name} must be non-decreasing (found negative segment length)")


def _normalize_qkv(x: torch.Tensor, name: str) -> Tuple[torch.Tensor, bool]:
    """
    Accept [1, T, H, D] (business) or [T, H, D] (FA2-style). Return [T, H, D]
    and a flag indicating whether the input was 4-D so we can unsqueeze the
    output back.

    Also forces contiguity along the last dim: business upstream may slice a
    larger tensor and pass us a non-contiguous view. The kernel assumes
    stride(-1) == 1 and stride(0) == H*D, so we materialize a contiguous
    copy here when the input violates either.
    """
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}: 4-D input must have batch=1 (packed varlen), "
                f"got shape {tuple(x.shape)}"
            )
        x = x.squeeze(0)
        was_4d = True
    elif x.dim() == 3:
        was_4d = False
    else:
        raise ValueError(
            f"{name}: expected 3-D [T,H,D] or 4-D [1,T,H,D], got {x.dim()}-D"
        )

    T, H, D = x.shape
    if x.stride(-1) != 1 or x.stride(1) != D or x.stride(0) != H * D:
        x = x.contiguous()
    return x, was_4d


class _PackedVarlenAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
        # All inputs here are already in [T, H, D] layout, contiguous (the
        # wrapper enforces both). We re-check shape/dtype/device defensively.
        T_q, H, D = q.shape
        T_k = k.shape[0]
        assert k.shape == (T_k, H, D), f"K shape mismatch: {k.shape} vs expected ({T_k},{H},{D})"
        assert v.shape == (T_k, H, D), f"V shape mismatch: {v.shape} vs expected ({T_k},{H},{D})"
        assert q.dtype == k.dtype == v.dtype, "q/k/v must share dtype"
        assert q.dtype.is_floating_point, f"q/k/v must be float, got {q.dtype}"
        assert q.is_cuda, "Triton FA requires CUDA tensors"
        assert k.device == q.device and v.device == q.device, "q/k/v must be on the same device"
        # head_dim is a constexpr in the kernel; powers of two are well-tested.
        assert D in (16, 32, 64, 128, 256), f"head_dim {D} not supported (try 64/128/256)"

        # cu_seqlens may live on a different device than the data tensors (e.g.
        # built on cpu). Move and cast in one step.
        cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
        cu_seqlens_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()
        n_seg = cu_seqlens_q.numel() - 1
        assert cu_seqlens_k.numel() - 1 == n_seg, \
            f"segment count mismatch: q has {n_seg}, k has {cu_seqlens_k.numel()-1}"
        _validate_cu_seqlens(cu_seqlens_q, T_q, "cu_seqlens_q")
        _validate_cu_seqlens(cu_seqlens_k, T_k, "cu_seqlens_k")

        # output buffers
        o = torch.empty_like(q)
        # LSE in [T_q, H] token-major fp32 (kernel-internal layout)
        lse = torch.empty((T_q, H), dtype=torch.float32, device=q.device)

        # Split-K dispatch: short segments go through the v1 single-program
        # path (fast path, no partial buffer); long segments go through the
        # split + combine path.
        split_idx = prepare_split_indices(
            cu_seqlens_q, cu_seqlens_k,
            BT=_FWD_BT,
            SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
            SPLIT_THRESHOLD=_SPLIT_THRESHOLD,
        )

        # --- short path (v1 _fwd_kernel) --------------------------------
        # Reuses the v1 kernel verbatim. We rebuild q_chunk_indices from
        # split_idx.short instead of going through prepare_q_chunk_indices,
        # because we want only the chunks that belong to short segments.
        if split_idx.short.shape[0] > 0:
            grid_short = (split_idx.short.shape[0], H)
            _fwd_kernel[grid_short](
                q, k, v, o, lse,
                cu_seqlens_q, cu_seqlens_k, split_idx.short,
                scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                num_warps=4, num_stages=2,
            )

        # --- long path (split + combine) -------------------------------
        partial_o = None
        partial_lse = None
        if split_idx.n_split_progs > 0:
            partial_o = torch.empty(
                (split_idx.n_split_progs, H, _FWD_BT, D),
                dtype=torch.float32, device=q.device,
            )
            partial_lse = torch.empty(
                (split_idx.n_split_progs, H, _FWD_BT),
                dtype=torch.float32, device=q.device,
            )
            grid_split = (split_idx.n_split_progs, H)
            _fwd_split_kernel[grid_split](
                q, k, v, partial_o, partial_lse,
                cu_seqlens_q, cu_seqlens_k, split_idx.split,
                scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                num_warps=4, num_stages=2,
            )

            grid_combine = (split_idx.n_combine_progs, H)
            _fwd_combine_kernel[grid_combine](
                partial_o, partial_lse, o, lse,
                cu_seqlens_q, split_idx.combine,
                H=H, D=D, BT=_FWD_BT,
                num_warps=4, num_stages=2,
            )

        # Stash everything bwd needs.
        # - q/k/v/o/lse/cu_seqlens: standard
        # - split_idx.short:   short-segment Q chunks  -> v1 _bwd_dq_kernel
        # - split_idx.split:   long-segment split rows -> _bwd_dq_split_kernel
        # - split_idx.combine: long-segment combine rows -> _bwd_dq_combine_kernel
        # dkv goes through v1 _bwd_dkv_kernel for all segments (no split-K
        # on dkv path); k_chunk_indices is rebuilt in bwd.
        ctx.save_for_backward(
            q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k,
            split_idx.short, split_idx.split, split_idx.combine,
        )
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        (q, k, v, o, lse,
         cu_seqlens_q, cu_seqlens_k,
         short_idx, split_idx_split, split_idx_combine) = ctx.saved_tensors
        scale = ctx.scale
        T_q, H, D = q.shape
        T_k = k.shape[0]

        # autograd may hand us a non-contiguous or wrong-dtype dO.
        if do.stride(-1) != 1 or not do.is_contiguous():
            do = do.contiguous()
        if do.dtype != q.dtype:
            do = do.to(q.dtype)

        # delta[t, h] = sum_d O[t,h,d] * dO[t,h,d]   in fp32
        delta = torch.empty((T_q, H), dtype=torch.float32, device=q.device)
        grid_pre = (triton.cdiv(T_q, _FWD_BT), H)
        _bwd_preprocess_kernel[grid_pre](
            o, do, delta,
            T_q,
            H=H, D=D, BT=_FWD_BT,
            num_warps=4, num_stages=2,
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # --- dq: short segments via v1 single-program path ---------------
        # Stage 2's uniform short segments avoid the partial-buffer overhead.
        if short_idx.shape[0] > 0:
            grid_short_dq = (short_idx.shape[0], H)
            _bwd_dq_kernel[grid_short_dq](
                q, k, v, do, dq, lse, delta,
                cu_seqlens_q, cu_seqlens_k, short_idx,
                scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                num_warps=4, num_stages=2,
            )

        # --- dq: long segments via split-K + combine --------------------
        # CRITICAL for T_q << T_k workloads: without split-K, dq grid for a
        # 2M-K / 500-Q segment is only ceil(500/128)*H = 16 programs, which
        # cannot fill the GPU. Split-K fans this out across K-slices so
        # ~thousands of programs land per long segment.
        n_split = int(split_idx_split.shape[0])
        n_combine = int(split_idx_combine.shape[0])
        if n_split > 0:
            partial_dq = torch.empty(
                (n_split, H, _FWD_BT, D),
                dtype=torch.float32, device=q.device,
            )
            grid_dq_split = (n_split, H)
            _bwd_dq_split_kernel[grid_dq_split](
                q, k, v, do, partial_dq, lse, delta,
                cu_seqlens_q, cu_seqlens_k, split_idx_split,
                scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                num_warps=4, num_stages=2,
            )

            grid_dq_combine = (n_combine, H)
            _bwd_dq_combine_kernel[grid_dq_combine](
                partial_dq, dq,
                cu_seqlens_q, split_idx_combine,
                scale,
                H=H, D=D, BT=_FWD_BT,
                num_warps=4, num_stages=2,
            )

        # --- dkv: v1 single-program path, all segments ------------------
        # No split-K on dkv yet; this is the v1 baseline behavior. In the
        # T_q << T_k extreme, the per-K-block program does little work and
        # the kernel under-utilizes SMs, but matching v1 is good enough
        # while dq carries the perf win.
        k_idx = prepare_k_chunk_indices(cu_seqlens_k, _BWD_DKV_BT)
        if k_idx.shape[0] > 0:
            grid_dkv = (k_idx.shape[0], H)
            _bwd_dkv_kernel[grid_dkv](
                q, k, v, do, dk, dv, lse, delta,
                cu_seqlens_q, cu_seqlens_k, k_idx,
                scale,
                H=H, D=D, BT=_BWD_DKV_BT, BS=_BWD_DKV_BS,
                num_warps=4, num_stages=2,
            )

        # forward signature was (q, k, v, cu_seqlens_q, cu_seqlens_k, scale)
        return dq, dk, dv, None, None, None


def packed_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_seqinfo: Optional[torch.Tensor] = None,
    k_seqinfo: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Packed-varlen FlashAttention with BlockDiagonalMask semantics.

    Args:
        q: [1, T_q, H, D] or [T_q, H, D]
        k: [1, T_k, H, D] or [T_k, H, D]
        v: [1, T_k, H, D] or [T_k, H, D]
        cu_seqlens_q / cu_seqlens_k: [n_seg+1] int32, prefix-sum offsets
            (preferred). Mutually exclusive with q_seqinfo/k_seqinfo.
        q_seqinfo / k_seqinfo: [n_seg] per-segment lengths (matches the
            business `q_seqinfo`/`k_seqinfo` arg). Will be converted to
            cu_seqlens internally.
        scale: softmax scale. Default 1/sqrt(D).

    Returns:
        Same layout as q (4-D in -> 4-D out, 3-D in -> 3-D out).
    """
    q3, q_was_4d = _normalize_qkv(q, "q")
    k3, k_was_4d = _normalize_qkv(k, "k")
    v3, v_was_4d = _normalize_qkv(v, "v")
    out_was_4d = q_was_4d  # output shape follows q

    # cu_seqlens vs seqinfo: prefer cu_seqlens if both given
    if cu_seqlens_q is None:
        if q_seqinfo is None:
            raise ValueError("must provide either cu_seqlens_q or q_seqinfo")
        cu_seqlens_q = _seqinfo_to_cu_seqlens(q_seqinfo)
    if cu_seqlens_k is None:
        if k_seqinfo is None:
            raise ValueError("must provide either cu_seqlens_k or k_seqinfo")
        cu_seqlens_k = _seqinfo_to_cu_seqlens(k_seqinfo)

    if scale is None:
        scale = 1.0 / math.sqrt(q3.shape[-1])

    o3 = _PackedVarlenAttnFunction.apply(q3, k3, v3, cu_seqlens_q, cu_seqlens_k, float(scale))

    return o3.unsqueeze(0) if out_was_4d else o3
