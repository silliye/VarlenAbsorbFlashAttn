"""
Host-side helpers for target-attn kernels.

Each kernel program owns a "chunk" — one BT-block of one segment. We precompute
(segment_idx, block_in_segment) pairs on CPU so each kernel program does a
single load to figure out where it is.

  - prepare_q_chunk_indices: used by _fwd_kernel and _bwd_du_kernel (Q-block grid)
  - prepare_k_chunk_indices: used by _bwd_dx_kernel (K-block grid)

Both return an int32 tensor of shape [NT, 2] on the same device as cu_seqlens.
"""

from __future__ import annotations

import torch


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, BT: int) -> torch.Tensor:
    """
    For each segment i with length L_i = cu_seqlens[i+1] - cu_seqlens[i],
    emit ceil(L_i / BT) rows: (i, 0), (i, 1), ..., (i, ceil(L_i/BT) - 1).

    Returns int32 tensor of shape [sum_i ceil(L_i / BT), 2] on cu_seqlens.device.
    """
    assert cu_seqlens.dtype in (torch.int32, torch.int64), \
        f"cu_seqlens must be int32/int64, got {cu_seqlens.dtype}"
    assert cu_seqlens.dim() == 1 and cu_seqlens.numel() >= 2, \
        f"cu_seqlens must be 1-D with len >= 2, got {tuple(cu_seqlens.shape)}"
    assert BT >= 1

    cs = cu_seqlens.to(torch.int64).cpu()
    seg_lens = cs[1:] - cs[:-1]                          # [n_seg]
    n_blocks_per_seg = (seg_lens + BT - 1) // BT         # [n_seg], ceil-div
    n_seg = seg_lens.numel()

    i_n = torch.repeat_interleave(
        torch.arange(n_seg, dtype=torch.int32),
        n_blocks_per_seg.to(torch.int32),
    )
    total = int(n_blocks_per_seg.sum().item())
    if total == 0:
        return torch.empty((0, 2), dtype=torch.int32, device=cu_seqlens.device)

    i_t = torch.arange(total, dtype=torch.int32)
    seg_starts = torch.cat([
        torch.zeros(1, dtype=torch.int64),
        n_blocks_per_seg.cumsum(0)[:-1],
    ])
    offsets_per_block = torch.repeat_interleave(
        seg_starts.to(torch.int32),
        n_blocks_per_seg.to(torch.int32),
    )
    i_t = i_t - offsets_per_block

    out = torch.stack([i_n, i_t], dim=1).contiguous()
    return out.to(cu_seqlens.device)


def prepare_q_chunk_indices(cu_seqlens_q: torch.Tensor, BT: int) -> torch.Tensor:
    """Used by fwd kernel and bwd_du kernel. BT here is the Q block size."""
    return _prepare_chunk_indices(cu_seqlens_q, BT)


def prepare_k_chunk_indices(cu_seqlens_k: torch.Tensor, BT: int) -> torch.Tensor:
    """Used by bwd_dx kernel. BT here is the K block size."""
    return _prepare_chunk_indices(cu_seqlens_k, BT)
