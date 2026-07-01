"""
Per-kernel profiling for triton_fa_varlen_v2.

Goal: identify which kernel dominates backward time. The benchmark only times
the user-facing wrapper, so dq_short / dq_split / dq_combine / dkv all blur
together. This script times each kernel in isolation via torch.cuda.Event so
we can confirm where to spend optimization effort.

Usage:
    python3 tools/profile_v2_kernels.py --shape-file fixtures/stage1_shapes.jsonl --limit 10

Output: per-case timings + an aggregate breakdown showing the % each kernel
contributes to the total fwd and bwd time.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import triton

from triton_fa_varlen_v2.kernels import (
    _fwd_kernel,
    _fwd_split_kernel,
    _fwd_combine_kernel,
    _bwd_preprocess_kernel,
    _bwd_dq_kernel,
    _bwd_dq_split_kernel,
    _bwd_dq_combine_kernel,
    _bwd_dkv_kernel,
)
from triton_fa_varlen_v2.chunk_indices import (
    prepare_k_chunk_indices,
    prepare_split_indices,
)

# match v2/interface.py
_FWD_BT = 128
_FWD_BS = 32
_BWD_DKV_BT = 128
_BWD_DKV_BS = 32
_SPLIT_THRESHOLD = 4096
_SPLIT_K_BLOCK = 4096


# ---------------------------------------------------------------------------
# CUDA event timing
# ---------------------------------------------------------------------------
def time_kernel(fn, repeats=20, warmup=3):
    """Return (avg_ms, min_ms) across `repeats` runs after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times) / len(times), min(times)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def build_cu_seqlens(lens, device):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int64, device=device)
    cu[1:] = torch.tensor(lens, dtype=torch.int64, device=device).cumsum(0)
    return cu.to(torch.int32)


def profile_case(case, device='cuda', repeats=20, dtype=torch.bfloat16):
    q_lens = case['q_lens']
    k_lens = case['k_lens']
    H, D = 4, 64
    T_q = sum(q_lens)
    T_k = sum(k_lens)

    torch.manual_seed(0)
    q = torch.randn(T_q, H, D, dtype=dtype, device=device)
    k = torch.randn(T_k, H, D, dtype=dtype, device=device)
    v = torch.randn(T_k, H, D, dtype=dtype, device=device)
    dout = torch.randn_like(q)

    cu_q = build_cu_seqlens(q_lens, device)
    cu_k = build_cu_seqlens(k_lens, device)
    scale = 1.0 / math.sqrt(D)

    split_idx = prepare_split_indices(
        cu_q, cu_k,
        BT=_FWD_BT,
        SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
        SPLIT_THRESHOLD=_SPLIT_THRESHOLD,
    )
    short_idx = split_idx.short
    split_arr = split_idx.split
    combine_arr = split_idx.combine
    n_split = split_idx.n_split_progs
    n_combine = split_idx.n_combine_progs

    o = torch.empty_like(q)
    lse = torch.empty((T_q, H), dtype=torch.float32, device=device)

    results = {}  # name -> (avg_ms, n_progs)

    # ------------------------------------------------------- forward short
    if short_idx.shape[0] > 0:
        grid = (short_idx.shape[0], H)
        def run():
            _fwd_kernel[grid](
                q, k, v, o, lse,
                cu_q, cu_k, short_idx, scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                num_warps=4, num_stages=2,
            )
        avg, _ = time_kernel(run, repeats=repeats)
        results['fwd_short'] = (avg, short_idx.shape[0])

    # ------------------------------------------------------- forward split + combine
    partial_o = None
    partial_lse = None
    if n_split > 0:
        partial_o = torch.empty((n_split, H, _FWD_BT, D),
                                dtype=torch.float32, device=device)
        partial_lse = torch.empty((n_split, H, _FWD_BT),
                                  dtype=torch.float32, device=device)
        grid_split = (n_split, H)
        def run_split():
            _fwd_split_kernel[grid_split](
                q, k, v, partial_o, partial_lse,
                cu_q, cu_k, split_arr, scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                num_warps=4, num_stages=2,
            )
        avg, _ = time_kernel(run_split, repeats=repeats)
        results['fwd_split'] = (avg, n_split)

        grid_combine = (n_combine, H)
        def run_combine():
            _fwd_combine_kernel[grid_combine](
                partial_o, partial_lse, o, lse,
                cu_q, combine_arr,
                H=H, D=D, BT=_FWD_BT,
                num_warps=4, num_stages=2,
            )
        avg, _ = time_kernel(run_combine, repeats=repeats)
        results['fwd_combine'] = (avg, n_combine)

    # ------------------------------------------------------- bwd preprocess
    delta = torch.empty((T_q, H), dtype=torch.float32, device=device)
    grid_pre = (triton.cdiv(T_q, _FWD_BT), H)
    def run_pre():
        _bwd_preprocess_kernel[grid_pre](
            o, dout, delta, T_q,
            H=H, D=D, BT=_FWD_BT,
            num_warps=4, num_stages=2,
        )
    avg, _ = time_kernel(run_pre, repeats=repeats)
    results['bwd_pre'] = (avg, grid_pre[0])

    # ------------------------------------------------------- bwd dq short
    dq = torch.empty_like(q)
    if short_idx.shape[0] > 0:
        grid = (short_idx.shape[0], H)
        def run_dq_short():
            _bwd_dq_kernel[grid](
                q, k, v, dout, dq, lse, delta,
                cu_q, cu_k, short_idx, scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                num_warps=4, num_stages=2,
            )
        avg, _ = time_kernel(run_dq_short, repeats=repeats)
        results['bwd_dq_short'] = (avg, short_idx.shape[0])

    # ------------------------------------------------------- bwd dq split + combine
    if n_split > 0:
        partial_dq = torch.empty((n_split, H, _FWD_BT, D),
                                 dtype=torch.float32, device=device)
        grid_dq_split = (n_split, H)
        def run_dq_split():
            _bwd_dq_split_kernel[grid_dq_split](
                q, k, v, dout, partial_dq, lse, delta,
                cu_q, cu_k, split_arr, scale,
                H=H, D=D, BT=_FWD_BT, BS=_FWD_BS,
                SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                num_warps=4, num_stages=2,
            )
        avg, _ = time_kernel(run_dq_split, repeats=repeats)
        results['bwd_dq_split'] = (avg, n_split)

        grid_dq_combine = (n_combine, H)
        def run_dq_combine():
            _bwd_dq_combine_kernel[grid_dq_combine](
                partial_dq, dq,
                cu_q, combine_arr, scale,
                H=H, D=D, BT=_FWD_BT,
                num_warps=4, num_stages=2,
            )
        avg, _ = time_kernel(run_dq_combine, repeats=repeats)
        results['bwd_dq_combine'] = (avg, n_combine)

    # ------------------------------------------------------- bwd dkv
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    k_idx = prepare_k_chunk_indices(cu_k, _BWD_DKV_BT)
    if k_idx.shape[0] > 0:
        grid_dkv = (k_idx.shape[0], H)
        def run_dkv():
            _bwd_dkv_kernel[grid_dkv](
                q, k, v, dout, dk, dv, lse, delta,
                cu_q, cu_k, k_idx, scale,
                H=H, D=D, BT=_BWD_DKV_BT, BS=_BWD_DKV_BS,
                num_warps=4, num_stages=2,
            )
        avg, _ = time_kernel(run_dkv, repeats=repeats)
        results['bwd_dkv'] = (avg, k_idx.shape[0])

    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
FWD_KERNELS = ['fwd_short', 'fwd_split', 'fwd_combine']
BWD_KERNELS = ['bwd_pre', 'bwd_dq_short', 'bwd_dq_split', 'bwd_dq_combine', 'bwd_dkv']
ALL_KERNELS = FWD_KERNELS + BWD_KERNELS


def main():
    parser = argparse.ArgumentParser()
    default_shape_file = Path(__file__).resolve().parents[1] / "fixtures" / "stage1_shapes.jsonl"
    parser.add_argument('--shape-file', type=str, default=str(default_shape_file))
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--per-case', action='store_true',
                        help='print per-case timings (default: aggregate only)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Run on the A800 machine.")

    cases = []
    with open(args.shape_file) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            cases.append(json.loads(line))

    print(f"Profiling {len(cases)} cases from {args.shape_file} "
          f"(repeats={args.repeats})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    totals = defaultdict(float)
    n_progs_total = defaultdict(int)
    case_breakdowns = []

    for i, case in enumerate(cases):
        T_q = case['tq']
        T_k = case['tk']
        Uv = case['users']
        if args.per_case:
            print(f"\n[{i+1:2d}/{len(cases)}] {case['name']} "
                  f"Uv={Uv} T_q={T_q} T_k={T_k}")

        t = profile_case(case, repeats=args.repeats)
        case_breakdowns.append((case['name'], t))

        for name, (avg, n_progs) in t.items():
            totals[name] += avg
            n_progs_total[name] += n_progs
            if args.per_case:
                print(f"    {name:20s} {avg:8.3f} ms   ({n_progs} progs)")

    fwd_total = sum(totals.get(k, 0.0) for k in FWD_KERNELS)
    bwd_total = sum(totals.get(k, 0.0) for k in BWD_KERNELS)

    print(f"\n{'='*70}")
    print(f"Aggregate (sum of per-case avg time, {len(cases)} cases)")
    print(f"{'='*70}")
    print(f"\n--- Forward total: {fwd_total:8.3f} ms ---")
    for k in FWD_KERNELS:
        if k in totals:
            pct = 100 * totals[k] / fwd_total if fwd_total > 0 else 0
            print(f"  {k:20s} {totals[k]:8.3f} ms  ({pct:5.1f}%)  "
                  f"progs/case={n_progs_total[k]//len(cases)}")

    print(f"\n--- Backward total: {bwd_total:8.3f} ms ---")
    for k in BWD_KERNELS:
        if k in totals:
            pct = 100 * totals[k] / bwd_total if bwd_total > 0 else 0
            print(f"  {k:20s} {totals[k]:8.3f} ms  ({pct:5.1f}%)  "
                  f"progs/case={n_progs_total[k]//len(cases)}")

    print(f"\n--- bwd/fwd ratio: {bwd_total / fwd_total:.2f}x  ---")

    # Highlight the prime suspect
    if bwd_total > 0:
        long_pole = max(BWD_KERNELS, key=lambda k: totals.get(k, 0))
        print(f"\nBackward long pole: {long_pole} "
              f"({100*totals[long_pole]/bwd_total:.1f}% of bwd time)")


if __name__ == '__main__':
    main()
