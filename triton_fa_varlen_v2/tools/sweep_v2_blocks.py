"""
Sweep BS / BT on the three long-pole kernels: fwd_split, bwd_dq_split, bwd_dkv.

Reuses the shape file and timing setup from profile_v2_kernels.py. For each
(BS, BT) combination, recompiles the kernel and times it. Reports the best
config per kernel.

Constraints:
  - fwd_split / bwd_dq_split: BS must divide SPLIT_K_BLOCK (4096). Try 32, 64, 128.
  - bwd_dkv: BT is K-block, BS is Q-block. Q-segs are 500, so BS=32->16 iters,
    BS=64->8 iters, BS=128->4 iters. BT=128 or 256.

Usage:
    python3 tools/sweep_v2_blocks.py --shape-file fixtures/stage1_shapes.jsonl --limit 5

A config that fails to compile (OOM regs, etc.) is reported and skipped.
"""

import argparse
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import triton

from triton_fa_varlen_v2.kernels import (
    _fwd_kernel,
    _fwd_split_kernel,
    _bwd_preprocess_kernel,
    _bwd_dq_split_kernel,
    _bwd_dkv_kernel,
)
from triton_fa_varlen_v2.chunk_indices import (
    prepare_k_chunk_indices,
    prepare_split_indices,
)

_SPLIT_THRESHOLD = 4096
_SPLIT_K_BLOCK = 4096


def time_kernel(fn, repeats=15, warmup=3):
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
    return sum(times) / len(times)


def build_cu_seqlens(lens, device):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int64, device=device)
    cu[1:] = torch.tensor(lens, dtype=torch.int64, device=device).cumsum(0)
    return cu.to(torch.int32)


def prep_case(case, device='cuda', dtype=torch.bfloat16):
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
    return q, k, v, dout, cu_q, cu_k, H, D, T_q, T_k


def sweep(args):
    cases = []
    with open(args.shape_file) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            cases.append(json.loads(line))
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Sweeping {len(cases)} cases x configs, repeats={args.repeats}\n")

    # FWD_BS, FWD_BT for fwd_split / bwd_dq_split
    fwd_configs = [
        # (BT, BS, warps, stages)
        (128, 32, 4, 2),   # baseline
        (128, 32, 4, 3),
        (128, 64, 4, 2),
        (128, 64, 4, 3),
        (128, 64, 8, 2),
        (128, 128, 8, 2),
        (128, 128, 8, 3),
        (64, 64, 4, 2),
    ]
    # DKV: (BT, BS, warps, stages) — BT is K, BS is Q
    dkv_configs = [
        (128, 32, 4, 2),   # baseline
        (128, 64, 4, 2),
        (128, 128, 4, 2),  # Q-seg is 500, so 4 iters
        (128, 32, 8, 2),
        (128, 64, 8, 2),
        (256, 32, 4, 2),
        (256, 32, 8, 2),
        (256, 64, 4, 2),
        (256, 64, 8, 2),
        (256, 128, 8, 2),
        (64, 32, 4, 2),
    ]

    # ---------------- bench fwd_split / bwd_dq_split ----------------
    print("=" * 80)
    print("fwd_split (the 96% of forward)")
    print("=" * 80)
    print(f"  {'BT':>4} {'BS':>4} {'W':>2} {'S':>2}  total_ms")
    fwd_results = []
    for (BT, BS, warps, stages) in fwd_configs:
        if _SPLIT_K_BLOCK % BS != 0:
            print(f"  skip BS={BS} (does not divide SPLIT_K_BLOCK={_SPLIT_K_BLOCK})")
            continue
        try:
            total = 0.0
            for case in cases:
                q, k, v, dout, cu_q, cu_k, H, D, T_q, T_k = prep_case(case)
                split_idx = prepare_split_indices(
                    cu_q, cu_k, BT=BT,
                    SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                    SPLIT_THRESHOLD=_SPLIT_THRESHOLD,
                )
                n_split = split_idx.n_split_progs
                if n_split == 0:
                    continue
                partial_o = torch.empty((n_split, H, BT, D), dtype=torch.float32, device='cuda')
                partial_lse = torch.empty((n_split, H, BT), dtype=torch.float32, device='cuda')
                grid = (n_split, H)
                scale = 1.0 / math.sqrt(D)

                def run():
                    _fwd_split_kernel[grid](
                        q, k, v, partial_o, partial_lse,
                        cu_q, cu_k, split_idx.split, scale,
                        H=H, D=D, BT=BT, BS=BS,
                        SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                        num_warps=warps, num_stages=stages,
                    )
                total += time_kernel(run, repeats=args.repeats)
            print(f"  {BT:>4} {BS:>4} {warps:>2} {stages:>2}  {total:8.3f}")
            fwd_results.append(((BT, BS, warps, stages), total))
            del partial_o, partial_lse, q, k, v, dout
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {BT:>4} {BS:>4} {warps:>2} {stages:>2}  FAILED: {e!r}")
            torch.cuda.empty_cache()

    # ---------------- bench bwd_dq_split ----------------
    print()
    print("=" * 80)
    print("bwd_dq_split (the 34% of backward)")
    print("=" * 80)
    print(f"  {'BT':>4} {'BS':>4} {'W':>2} {'S':>2}  total_ms")
    bwd_dq_results = []
    for (BT, BS, warps, stages) in fwd_configs:
        if _SPLIT_K_BLOCK % BS != 0:
            continue
        try:
            total = 0.0
            for case in cases:
                q, k, v, dout, cu_q, cu_k, H, D, T_q, T_k = prep_case(case)
                split_idx = prepare_split_indices(
                    cu_q, cu_k, BT=BT,
                    SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                    SPLIT_THRESHOLD=_SPLIT_THRESHOLD,
                )
                n_split = split_idx.n_split_progs
                if n_split == 0:
                    continue
                # need lse + delta from a fwd; cheap to fake them as random fp32
                lse = torch.randn(T_q, H, dtype=torch.float32, device='cuda')
                delta = torch.randn(T_q, H, dtype=torch.float32, device='cuda')
                partial_dq = torch.empty((n_split, H, BT, D), dtype=torch.float32, device='cuda')
                grid = (n_split, H)
                scale = 1.0 / math.sqrt(D)

                def run():
                    _bwd_dq_split_kernel[grid](
                        q, k, v, dout, partial_dq, lse, delta,
                        cu_q, cu_k, split_idx.split, scale,
                        H=H, D=D, BT=BT, BS=BS,
                        SPLIT_K_BLOCK=_SPLIT_K_BLOCK,
                        num_warps=warps, num_stages=stages,
                    )
                total += time_kernel(run, repeats=args.repeats)
            print(f"  {BT:>4} {BS:>4} {warps:>2} {stages:>2}  {total:8.3f}")
            bwd_dq_results.append(((BT, BS, warps, stages), total))
            del partial_dq, q, k, v, dout, lse, delta
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {BT:>4} {BS:>4} {warps:>2} {stages:>2}  FAILED: {e!r}")
            torch.cuda.empty_cache()

    # ---------------- bench bwd_dkv ----------------
    print()
    print("=" * 80)
    print("bwd_dkv (the 65% of backward)")
    print("=" * 80)
    print(f"  {'BT':>4} {'BS':>4} {'W':>2} {'S':>2}  total_ms")
    dkv_results = []
    for (BT, BS, warps, stages) in dkv_configs:
        try:
            total = 0.0
            for case in cases:
                q, k, v, dout, cu_q, cu_k, H, D, T_q, T_k = prep_case(case)
                lse = torch.randn(T_q, H, dtype=torch.float32, device='cuda')
                delta = torch.randn(T_q, H, dtype=torch.float32, device='cuda')
                dk = torch.empty_like(k)
                dv = torch.empty_like(v)
                k_idx = prepare_k_chunk_indices(cu_k, BT)
                if k_idx.shape[0] == 0:
                    continue
                grid = (k_idx.shape[0], H)
                scale = 1.0 / math.sqrt(D)

                def run():
                    _bwd_dkv_kernel[grid](
                        q, k, v, dout, dk, dv, lse, delta,
                        cu_q, cu_k, k_idx, scale,
                        H=H, D=D, BT=BT, BS=BS,
                        num_warps=warps, num_stages=stages,
                    )
                total += time_kernel(run, repeats=args.repeats)
            print(f"  {BT:>4} {BS:>4} {warps:>2} {stages:>2}  {total:8.3f}")
            dkv_results.append(((BT, BS, warps, stages), total))
            del q, k, v, dout, lse, delta, dk, dv
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {BT:>4} {BS:>4} {warps:>2} {stages:>2}  FAILED: {e!r}")
            torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("Best configs (lowest total_ms over cases)")
    print("=" * 80)
    for name, results in [
        ('fwd_split', fwd_results),
        ('bwd_dq_split', bwd_dq_results),
        ('bwd_dkv', dkv_results),
    ]:
        if not results:
            print(f"  {name}: no successful config")
            continue
        results.sort(key=lambda x: x[1])
        best = results[0]
        baseline = next((r for r in results if r[0][:2] == (128, 32)), None)
        msg = f"  {name}: best={best[0]} -> {best[1]:.3f} ms"
        if baseline is not None and baseline != best:
            speedup = baseline[1] / best[1]
            msg += f"   (baseline (BT=128,BS=32,W=4,S=2): {baseline[1]:.3f} ms, {speedup:.2f}x)"
        print(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_shape_file = Path(__file__).resolve().parents[1] / "fixtures" / "stage1_shapes.jsonl"
    parser.add_argument('--shape-file', type=str, default=str(default_shape_file))
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=15)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    sweep(args)
