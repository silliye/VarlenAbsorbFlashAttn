"""
Host-side helpers for the Triton packed-varlen FA kernels.

The kernels are launched on a 1-D "chunk" grid axis where each program
handles one BT-block of one segment. We precompute (segment_idx,
block_in_segment) pairs on host so the kernel just does a single load
to figure out where it is.

Two flavors:
  - prepare_q_chunk_indices(cu_seqlens_q, BT) -> for fwd / bwd_dq
  - prepare_k_chunk_indices(cu_seqlens_k, BT) -> for bwd_dkv

Both return an int32 tensor of shape [NT, 2] on the same device.

For the v2 split-K path (segments with T_k_seg > threshold are processed
by multiple programs, each owning a contiguous K-slice), see
prepare_split_indices below.
"""

from typing import NamedTuple, Tuple

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
        f"cu_seqlens must be 1-D with len >= 2, got shape {tuple(cu_seqlens.shape)}"

    # work on CPU for the small index arithmetic, move to device at the end.
    # cu_seqlens is typically tiny (n_seg+1, with n_seg up to a few thousand).
    cs = cu_seqlens.to(torch.int64).cpu()
    seg_lens = cs[1:] - cs[:-1]                          # [n_seg]
    n_blocks_per_seg = (seg_lens + BT - 1) // BT         # [n_seg], ceil-div
    n_seg = seg_lens.numel()

    # i_n: segment index repeated n_blocks_per_seg[i] times
    i_n = torch.repeat_interleave(
        torch.arange(n_seg, dtype=torch.int32),
        n_blocks_per_seg.to(torch.int32),
    )
    # i_t: 0, 1, ..., n_blocks_per_seg[i]-1 for each segment, concatenated
    # built via cumulative offsets so we don't need a Python loop
    total = int(n_blocks_per_seg.sum().item())
    i_t = torch.arange(total, dtype=torch.int32)
    seg_starts = torch.cat([
        torch.zeros(1, dtype=torch.int64),
        n_blocks_per_seg.cumsum(0)[:-1],
    ])                                                   # [n_seg]
    # subtract the segment-start offset from each block's global index
    offsets_per_block = torch.repeat_interleave(
        seg_starts.to(torch.int32),
        n_blocks_per_seg.to(torch.int32),
    )
    i_t = i_t - offsets_per_block

    out = torch.stack([i_n, i_t], dim=1).contiguous()    # [total, 2] int32
    return out.to(cu_seqlens.device)


def prepare_q_chunk_indices(cu_seqlens_q: torch.Tensor, BT: int) -> torch.Tensor:
    """Used by fwd kernel and bwd_dq kernel. BT here is the Q block size."""
    return _prepare_chunk_indices(cu_seqlens_q, BT)


def prepare_k_chunk_indices(cu_seqlens_k: torch.Tensor, BT: int) -> torch.Tensor:
    """Used by bwd_dkv kernel. BT here is the K block size."""
    return _prepare_chunk_indices(cu_seqlens_k, BT)


def prepare_short_k_chunk_indices(
    cu_seqlens_k: torch.Tensor,
    BT: int,
    SPLIT_THRESHOLD: int,
) -> torch.Tensor:
    """
    Like prepare_k_chunk_indices, but only emits rows for SHORT segments
    (T_k_seg <= SPLIT_THRESHOLD). Long-segment dk/dv is handled by the
    fused kernel directly, so we must not let the short-path bwd_dkv kernel
    overwrite those.
    """
    assert cu_seqlens_k.dim() == 1 and cu_seqlens_k.numel() >= 2
    cs = cu_seqlens_k.to(torch.int64).cpu()
    seg_lens = cs[1:] - cs[:-1]
    is_short = seg_lens <= SPLIT_THRESHOLD
    n_blocks = (seg_lens + BT - 1) // BT
    # zero out long-segment block counts so repeat_interleave skips them
    n_blocks_short = torch.where(
        is_short,
        n_blocks,
        torch.zeros_like(n_blocks),
    )
    n_seg = seg_lens.numel()

    i_n = torch.repeat_interleave(
        torch.arange(n_seg, dtype=torch.int32),
        n_blocks_short.to(torch.int32),
    )
    total = int(n_blocks_short.sum().item())
    if total == 0:
        out = torch.empty((0, 2), dtype=torch.int32)
        return out.to(cu_seqlens_k.device)
    i_t_global = torch.arange(total, dtype=torch.int32)
    seg_starts = torch.cat([
        torch.zeros(1, dtype=torch.int64),
        n_blocks_short.cumsum(0)[:-1],
    ])
    offsets = torch.repeat_interleave(
        seg_starts.to(torch.int32),
        n_blocks_short.to(torch.int32),
    )
    i_t = i_t_global - offsets
    out = torch.stack([i_n, i_t], dim=1).contiguous()
    return out.to(cu_seqlens_k.device)


# ---------------------------------------------------------------------------
# Split-K indices (v2 long-segment path).
#
# Two segment "flavors" cohabit a single batch:
#   Short (T_k_seg <= SPLIT_THRESHOLD)  ->  v1 single-program path. Each
#       (i_n, i_t) chunk is processed by ONE program that walks the whole
#       segment K and writes the final O / LSE directly.
#   Long (T_k_seg > SPLIT_THRESHOLD)    ->  split-K + combine. Each
#       (i_n, i_t) chunk is processed by n_splits programs, each owning a
#       contiguous K-slice of length SPLIT_K_BLOCK. They write partial
#       O / LSE into a [n_split_progs, H, BT, D] fp32 buffer; a second
#       combine kernel merges them via log2-domain online softmax.
#
# We keep the two paths separate (rather than always going through the
# split path with n_splits=1) because Stage 2 has 30+ short segments per
# batch and forcing them all through partial-buffer + combine would inflate
# memory and add launch overhead for no benefit.
# ---------------------------------------------------------------------------


class SplitIndices(NamedTuple):
    """
    Container for v2 split-K dispatch.

    Fields:
      short:    int32 [N_short, 2]     (i_n, i_t) for short-segment chunks.
                                       Empty when every segment is long.
      split:    int32 [N_split, 3]     (i_n, i_t, i_s) for long-segment
                                       split-K programs. Each row's index
                                       in this tensor IS its slot in the
                                       partial buffer (so a program can
                                       just write partial[program_id]).
      combine:  int32 [N_combine, 4]   (i_n, i_t, partial_start, n_splits)
                                       per long-segment Q-chunk. The
                                       combine kernel reads
                                       partial[partial_start :
                                              partial_start + n_splits]
                                       and merges them.
      n_split_progs:   int    grid-x dim for the split kernel.
      n_combine_progs: int    grid-x dim for the combine kernel.
    """
    short: torch.Tensor
    split: torch.Tensor
    combine: torch.Tensor
    n_split_progs: int
    n_combine_progs: int


def prepare_split_indices(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    BT: int,
    SPLIT_K_BLOCK: int,
    SPLIT_THRESHOLD: int,
) -> SplitIndices:
    """
    Build dispatch indices for the v2 split-K + combine forward / bwd_dq
    pipeline. See module docstring for the partitioning rule.

    Args:
        cu_seqlens_q / cu_seqlens_k: [n_seg+1] int32/int64 prefix sums.
        BT: Q block size (must match the kernel's BT constexpr).
        SPLIT_K_BLOCK: K tokens processed per split program (long path).
        SPLIT_THRESHOLD: T_k_seg <= threshold takes the short path.

    Returns SplitIndices on cu_seqlens_q.device.
    """
    assert cu_seqlens_q.shape == cu_seqlens_k.shape, \
        f"cu_seqlens shape mismatch: q={cu_seqlens_q.shape} k={cu_seqlens_k.shape}"
    assert cu_seqlens_q.dim() == 1 and cu_seqlens_q.numel() >= 2
    assert SPLIT_K_BLOCK >= 1 and SPLIT_THRESHOLD >= 0
    assert BT >= 1

    # All arithmetic on CPU. Tensors are tiny (n_seg+1, n_seg <= a few
    # thousand in production) so we don't pay much for the .cpu() copy.
    cs_q = cu_seqlens_q.to(torch.int64).cpu()
    cs_k = cu_seqlens_k.to(torch.int64).cpu()
    seg_lens_q = cs_q[1:] - cs_q[:-1]                # [n_seg]
    seg_lens_k = cs_k[1:] - cs_k[:-1]                # [n_seg]
    n_seg = int(seg_lens_q.numel())

    n_q_blocks = (seg_lens_q + BT - 1) // BT         # [n_seg]
    n_splits   = (seg_lens_k + SPLIT_K_BLOCK - 1) // SPLIT_K_BLOCK  # [n_seg]
    is_long = seg_lens_k > SPLIT_THRESHOLD           # [n_seg] bool

    seg_idx = torch.arange(n_seg, dtype=torch.int64)

    # --------- short path: (i_n, i_t) for short-segment Q-chunks ----------
    short_mask = ~is_long
    short_segs = seg_idx[short_mask]
    n_blocks_short = n_q_blocks[short_mask]
    if int(n_blocks_short.sum().item()) > 0:
        i_n_short = torch.repeat_interleave(
            short_segs.to(torch.int32),
            n_blocks_short.to(torch.int32),
        )
        total_s = int(n_blocks_short.sum().item())
        all_t_s = torch.arange(total_s, dtype=torch.int32)
        seg_starts_s = torch.cat([
            torch.zeros(1, dtype=torch.int64),
            n_blocks_short.cumsum(0)[:-1],
        ])
        offsets_s = torch.repeat_interleave(
            seg_starts_s.to(torch.int32),
            n_blocks_short.to(torch.int32),
        )
        i_t_short = all_t_s - offsets_s
        short = torch.stack([i_n_short, i_t_short], dim=1).contiguous()
    else:
        short = torch.empty((0, 2), dtype=torch.int32)

    # --------- long path: split (i_n, i_t, i_s) and combine rows ----------
    long_mask = is_long
    long_segs = seg_idx[long_mask]
    n_blocks_long = n_q_blocks[long_mask]
    n_splits_long = n_splits[long_mask]
    n_long = int(long_segs.numel())

    if n_long == 0:
        split = torch.empty((0, 3), dtype=torch.int32)
        combine = torch.empty((0, 4), dtype=torch.int32)
    else:
        # Build per-segment chunks and concatenate. We Python-loop over
        # long segments only (typically <= a few hundred); if this ever
        # gets hot we can vectorize.
        split_pieces = []
        combine_rows = []
        partial_running = 0
        for li in range(n_long):
            seg_global = int(long_segs[li].item())
            nb = int(n_blocks_long[li].item())
            ns = int(n_splits_long[li].item())
            # split rows: nb * ns of them, ordered (q_block major, split minor)
            #   (seg, 0, 0), (seg, 0, 1), ..., (seg, 0, ns-1),
            #   (seg, 1, 0), ...
            i_t_slice = torch.arange(nb, dtype=torch.int32).repeat_interleave(ns)
            i_s_slice = torch.arange(ns, dtype=torch.int32).repeat(nb)
            i_n_slice = torch.full((nb * ns,), seg_global, dtype=torch.int32)
            split_pieces.append(torch.stack([i_n_slice, i_t_slice, i_s_slice], dim=1))

            # combine rows: nb of them. Each Q-block points to a contiguous
            # block of length ns inside partial buffer.
            for q_block in range(nb):
                combine_rows.append(
                    (seg_global, q_block, partial_running + q_block * ns, ns)
                )
            partial_running += nb * ns

        split = torch.cat(split_pieces, dim=0).contiguous()
        combine = torch.tensor(combine_rows, dtype=torch.int32)
        if combine.numel() == 0:
            combine = combine.reshape(0, 4)

    n_split_progs = int(split.shape[0])
    n_combine_progs = int(combine.shape[0])

    device = cu_seqlens_q.device
    return SplitIndices(
        short=short.to(device),
        split=split.to(device),
        combine=combine.to(device),
        n_split_progs=n_split_progs,
        n_combine_progs=n_combine_progs,
    )


# ---------------------------------------------------------------------------
# Fused backward indices (long-segment path, K-major grid).
#
# B3b plan: bwd_dq and bwd_dkv share one kernel for long segments. Each
# program owns ONE K-block of ONE long segment, scans every Q-block in that
# segment, and:
#   - accumulates dk/dv inside the program (writes final dk/dv at the end:
#     each K-block has a unique owner, no race)
#   - writes a fp32 dq partial PER (k_block, q_block) into a row-major
#     buffer at index (i_kprog * max_q_blocks + i_q_block).
#   A second combine kernel sums those partials per (segment, q_block) into
#   the final dq.
#
# Short segments stay on v1's split-kernel bwd (no partial buffer cost).
# ---------------------------------------------------------------------------


class FusedBwdIndices(NamedTuple):
    """
    Fields:
      kprog:     int32 [N_kprog, 3]      (i_n, i_k_block, q_block_base) per
                                         long-segment K-block program.
                                         q_block_base is the row offset into
                                         partial_dq for this k_prog (i.e.
                                         i_kprog * MAX_Q_BLOCKS_PER_SEG).
      dq_combine: int32 [N_dq_combine, 4] (i_n, i_q_block, partial_start,
                                          n_k_progs) per long-segment Q-chunk.
                                          partial_start points at the first
                                          (k_prog=0, q_block=i_q_block) row
                                          for this segment in partial_dq.
      n_kprog:    int     grid-x for fused kernel.
      n_dq_combine: int    grid-x for dq combine kernel.
      max_q_blocks: int    max Q-blocks per long segment; partial_dq stride.

      short_dq:  int32 [N_short_dq, 2]   (i_n, i_t) for short-segment dq
                                         chunks. Goes through v1 _bwd_dq_kernel.
      short_dkv_k_indices: int32 [N_short_dkv, 2] for short-segment dkv chunks
                                         (passed to v1 _bwd_dkv_kernel).
                                         Built independently because dkv uses
                                         a different K block size.
    """
    kprog: torch.Tensor
    dq_combine: torch.Tensor
    n_kprog: int
    n_dq_combine: int
    max_q_blocks: int
    short_dq: torch.Tensor


def prepare_fused_bwd_indices(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    BT_Q: int,
    BT_K: int,
    SPLIT_THRESHOLD: int,
) -> FusedBwdIndices:
    """
    Args:
      BT_Q: Q block size used inside the fused kernel (== fwd BT)
      BT_K: K block size for the fused grid (each program owns BT_K K tokens)
      SPLIT_THRESHOLD: T_k_seg > threshold -> long path; else short path.

    Returns FusedBwdIndices on cu_seqlens_q.device.
    """
    assert cu_seqlens_q.shape == cu_seqlens_k.shape
    assert cu_seqlens_q.dim() == 1 and cu_seqlens_q.numel() >= 2
    assert BT_Q >= 1 and BT_K >= 1

    cs_q = cu_seqlens_q.to(torch.int64).cpu()
    cs_k = cu_seqlens_k.to(torch.int64).cpu()
    seg_lens_q = cs_q[1:] - cs_q[:-1]
    seg_lens_k = cs_k[1:] - cs_k[:-1]
    n_seg = int(seg_lens_q.numel())

    n_q_blocks = (seg_lens_q + BT_Q - 1) // BT_Q          # [n_seg]
    n_k_blocks = (seg_lens_k + BT_K - 1) // BT_K          # [n_seg]
    is_long = seg_lens_k > SPLIT_THRESHOLD                # [n_seg] bool

    seg_idx = torch.arange(n_seg, dtype=torch.int64)

    # --------- short-segment dq indices ---------------------------------
    short_mask = ~is_long
    short_segs = seg_idx[short_mask]
    n_blocks_short = n_q_blocks[short_mask]
    if int(n_blocks_short.sum().item()) > 0:
        i_n_short = torch.repeat_interleave(
            short_segs.to(torch.int32),
            n_blocks_short.to(torch.int32),
        )
        total_s = int(n_blocks_short.sum().item())
        all_t_s = torch.arange(total_s, dtype=torch.int32)
        seg_starts_s = torch.cat([
            torch.zeros(1, dtype=torch.int64),
            n_blocks_short.cumsum(0)[:-1],
        ])
        offsets_s = torch.repeat_interleave(
            seg_starts_s.to(torch.int32),
            n_blocks_short.to(torch.int32),
        )
        i_t_short = all_t_s - offsets_s
        short_dq = torch.stack([i_n_short, i_t_short], dim=1).contiguous()
    else:
        short_dq = torch.empty((0, 2), dtype=torch.int32)

    # --------- long-segment fused indices --------------------------------
    long_mask = is_long
    long_segs = seg_idx[long_mask]
    n_blocks_long_q = n_q_blocks[long_mask]
    n_blocks_long_k = n_k_blocks[long_mask]
    n_long = int(long_segs.numel())

    if n_long == 0:
        kprog = torch.empty((0, 3), dtype=torch.int32)
        dq_combine = torch.empty((0, 4), dtype=torch.int32)
        max_q_blocks = 0
    else:
        # max_q_blocks = max number of Q-blocks across long segments.
        # partial_dq stride per k_prog is max_q_blocks (some k_progs in
        # smaller-Q segments waste a few rows but the buffer stays row-major).
        max_q_blocks = int(n_blocks_long_q.max().item())

        kprog_pieces = []
        dq_combine_rows = []
        kprog_running = 0
        for li in range(n_long):
            seg_global = int(long_segs[li].item())
            nq = int(n_blocks_long_q[li].item())
            nk = int(n_blocks_long_k[li].item())
            # kprog rows: nk per long segment, ordered by k_block_idx
            seg_col = torch.full((nk,), seg_global, dtype=torch.int32)
            k_col = torch.arange(nk, dtype=torch.int32)
            qbase_col = torch.arange(nk, dtype=torch.int32) * max_q_blocks
            qbase_col = qbase_col + (kprog_running * max_q_blocks)
            kprog_pieces.append(torch.stack([seg_col, k_col, qbase_col], dim=1))

            # dq_combine rows: nq per long segment.
            # partial_start for q_block q = kprog_running * max_q_blocks + q
            #   stride between consecutive k_progs = max_q_blocks
            # We pass this layout convention to combine kernel via
            # (partial_start, n_k_progs); combine reads
            #   partial_dq[partial_start, partial_start + max_q_blocks,
            #              partial_start + 2 * max_q_blocks, ...]
            # by stride = max_q_blocks (handed in as a separate constexpr).
            for q_block in range(nq):
                dq_combine_rows.append(
                    (
                        seg_global,
                        q_block,
                        kprog_running * max_q_blocks + q_block,
                        nk,
                    )
                )
            kprog_running += nk

        kprog = torch.cat(kprog_pieces, dim=0).contiguous()
        dq_combine = torch.tensor(dq_combine_rows, dtype=torch.int32)
        if dq_combine.numel() == 0:
            dq_combine = dq_combine.reshape(0, 4)

    n_kprog = int(kprog.shape[0])
    n_dq_combine = int(dq_combine.shape[0])

    device = cu_seqlens_q.device
    return FusedBwdIndices(
        kprog=kprog.to(device),
        dq_combine=dq_combine.to(device),
        n_kprog=n_kprog,
        n_dq_combine=n_dq_combine,
        max_q_blocks=max_q_blocks,
        short_dq=short_dq.to(device),
    )

