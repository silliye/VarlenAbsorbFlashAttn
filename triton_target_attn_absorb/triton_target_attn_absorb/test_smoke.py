"""
Oracle reference + smoke tests for triton_target_attn_absorb.

Reference: pure PyTorch, per-segment loop, explicit K = X@W_K, V = X@W_V.
We deliberately keep this slow and obvious — no fusion, no clever vectorization.
The point is to give the Triton kernel a value to compare against.

Run:
  python -m triton_target_attn_absorb.test_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import triton

import triton_target_attn_absorb.interface as _ta_interface
from triton_target_attn_absorb.interface import (
    packed_target_attn,
    prepare_target_attn_indices,
    target_attn_inner,
    _BT_K_BWD,
    _BT_K_FWD,
    _BT_Q_BWD,
    _BT_Q_FWD,
    _select_dx_mode,
    _select_fwd_mode,
)
from triton_target_attn_absorb.kernels import (
    _bwd_dx_atomic_heads_kernel,
    _bwd_dx_kernel,
    _bwd_dx_fused_heads_kernel,
    _bwd_dx_reduce_kernel,
    _bwd_du_kernel,
    _bwd_preprocess_kernel,
    _fwd_alpha_kernel,
    _fwd_kernel,
)
from triton_target_attn_absorb.chunk_indices import (
    prepare_k_chunk_indices,
    prepare_q_chunk_indices,
)


_LOG_FILE_HANDLE = None

_DEFAULT_D_Q = 968
_DEFAULT_D_KV = 216
_DEFAULT_H = 8
_DEFAULT_D_H = 32   # current business shape: 8 heads, total attention dim 256


def _redirect_process_output(log_file: str | None) -> None:
    """
    Redirect process-level stdout/stderr to a log file.

    This uses dup2 instead of only replacing sys.stderr, so TensorFlow/C++ logs
    written directly to fd 2 are captured too.
    """
    if log_file is None:
        return

    path = os.path.abspath(log_file)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    global _LOG_FILE_HANDLE
    _LOG_FILE_HANDLE = open(path, "w", buffering=1)
    print(f"writing stdout/stderr to {path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_LOG_FILE_HANDLE.fileno(), 1)
    os.dup2(_LOG_FILE_HANDLE.fileno(), 2)
    sys.stdout = _LOG_FILE_HANDLE
    sys.stderr = _LOG_FILE_HANDLE
    print(f"# log file: {path}", flush=True)


def _override_runtime_constant(name: str, value: int | None) -> None:
    if value is None:
        return
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} override must be positive")
    setattr(_ta_interface, name, value)
    globals()[name] = value


def _apply_runtime_tuning(args) -> None:
    _override_runtime_constant("_BT_Q_FWD", args.bt_q_fwd)
    _override_runtime_constant("_BT_K_FWD", args.bt_k_fwd)
    _override_runtime_constant("_BT_Q_BWD", args.bt_q_bwd)
    _override_runtime_constant("_BT_K_BWD", args.bt_k_bwd)
    _override_runtime_constant(
        "_DX_FUSED_HEADS_MIN_K_CHUNKS",
        args.dx_fused_min_k_chunks,
    )


# -------------------------------------------------------------------------
# Reference (no matrix absorption — explicit K, V — for cross-checking)
# -------------------------------------------------------------------------
def reference_target_attn(
    q_raw: torch.Tensor,           # [T_q, d_q]
    X: torch.Tensor,               # [T_k, d_kv]
    W_Q: torch.Tensor,             # [d_q, H, D_H]
    W_K: torch.Tensor,             # [d_kv, H, D_H]
    W_V: torch.Tensor,             # [d_kv, H, D_H]
    cu_seqlens_q: torch.Tensor,    # [n_seg+1]
    cu_seqlens_k: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Returns O [T_q, H, D_H]. Pure PyTorch, per-segment per-head loop.
    """
    T_q, d_q = q_raw.shape
    T_k, d_kv = X.shape
    _, H, D_H = W_Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D_H)

    cu_q = cu_seqlens_q.cpu().tolist()
    cu_k = cu_seqlens_k.cpu().tolist()
    n_seg = len(cu_q) - 1

    O = torch.zeros(T_q, H, D_H, dtype=q_raw.dtype, device=q_raw.device)
    for i in range(n_seg):
        q_i = q_raw[cu_q[i]:cu_q[i+1]]               # [N_q_i, d_q]
        x_i = X[cu_k[i]:cu_k[i+1]]                   # [N_k_i, d_kv]
        if q_i.shape[0] == 0 or x_i.shape[0] == 0:
            continue
        for h in range(H):
            q_proj = q_i @ W_Q[:, h, :]              # [N_q_i, D_H]
            k_proj = x_i @ W_K[:, h, :]              # [N_k_i, D_H]
            v_proj = x_i @ W_V[:, h, :]              # [N_k_i, D_H]
            s = q_proj @ k_proj.T * scale            # [N_q_i, N_k_i]
            p = s.float().softmax(-1).to(q_raw.dtype)
            O[cu_q[i]:cu_q[i+1], h, :] = p @ v_proj
    return O


def reference_target_attn_vectorized(
    q_raw: torch.Tensor,           # [T_q, d_q]
    X: torch.Tensor,               # [T_k, d_kv]
    W_Q: torch.Tensor,             # [d_q, H, D_H]
    W_K: torch.Tensor,             # [d_kv, H, D_H]
    W_V: torch.Tensor,             # [d_kv, H, D_H]
    cu_seqlens_q: torch.Tensor,    # [n_seg+1]
    cu_seqlens_k: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Faster PyTorch explicit-K/V baseline for timing. Still materializes K and V,
    but computes all heads of one segment together.
    """
    T_q, _ = q_raw.shape
    _, H, D_H = W_Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D_H)

    cu_q = cu_seqlens_q.cpu().tolist()
    cu_k = cu_seqlens_k.cpu().tolist()
    O = torch.empty(T_q, H, D_H, dtype=q_raw.dtype, device=q_raw.device)

    for i in range(len(cu_q) - 1):
        q_i = q_raw[cu_q[i]:cu_q[i+1]]               # [N_q_i, d_q]
        x_i = X[cu_k[i]:cu_k[i+1]]                   # [N_k_i, d_kv]
        if q_i.shape[0] == 0:
            continue
        q_proj = torch.einsum("qa,ahd->qhd", q_i, W_Q)
        k_proj = torch.einsum("ka,ahd->khd", x_i, W_K)
        v_proj = torch.einsum("ka,ahd->khd", x_i, W_V)
        s = torch.einsum("qhd,khd->hqk", q_proj, k_proj) * scale
        p = s.float().softmax(-1).to(q_raw.dtype)
        O[cu_q[i]:cu_q[i+1]] = torch.einsum("hqk,khd->qhd", p, v_proj)

    return O


def reference_target_attn_absorbed_vectorized(
    q_raw: torch.Tensor,           # [T_q, d_q]
    X: torch.Tensor,               # [T_k, d_kv]
    W_Q: torch.Tensor,             # [d_q, H, D_H]
    W_K: torch.Tensor,             # [d_kv, H, D_H]
    W_V: torch.Tensor,             # [d_kv, H, D_H]
    cu_seqlens_q: torch.Tensor,    # [n_seg+1]
    cu_seqlens_k: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    PyTorch matrix-absorbed baseline for timing.

    Does not materialize K = X@W_K or V = X@W_V. It expresses the same absorbed
    math as the Triton path with framework ops:
      u = (q @ W_Q) @ W_K^T
      O = (softmax(u @ X^T) @ X) @ W_V
    It still materializes u / p / alphaX, so it is a framework baseline rather
    than a fused-kernel baseline.
    """
    T_q, _ = q_raw.shape
    _, H, D_H = W_Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D_H)

    cu_q = cu_seqlens_q.cpu().tolist()
    cu_k = cu_seqlens_k.cpu().tolist()
    O = torch.empty(T_q, H, D_H, dtype=q_raw.dtype, device=q_raw.device)

    for i in range(len(cu_q) - 1):
        q_i = q_raw[cu_q[i]:cu_q[i+1]]               # [N_q_i, d_q]
        x_i = X[cu_k[i]:cu_k[i+1]]                   # [N_k_i, d_kv]
        if q_i.shape[0] == 0:
            continue
        q_proj = torch.einsum("qa,ahd->qhd", q_i, W_Q)
        u = torch.einsum("qhd,ahd->qha", q_proj, W_K)
        s = torch.einsum("qha,ka->hqk", u, x_i) * scale
        p = s.float().softmax(-1).to(q_raw.dtype)
        alpha_x = torch.einsum("hqk,ka->qha", p, x_i)
        O[cu_q[i]:cu_q[i+1]] = torch.einsum("qha,ahd->qhd", alpha_x, W_V)

    return O


def _import_flash_attn_varlen_func():
    try:
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func
    except Exception:
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        return flash_attn_varlen_func
    except Exception:
        return None


def _flash_qkv_project(q_raw, X, W_Q, W_K, W_V):
    q_proj = torch.einsum("qa,ahd->qhd", q_raw, W_Q).contiguous()
    k_proj = torch.einsum("ka,ahd->khd", X, W_K).contiguous()
    v_proj = torch.einsum("ka,ahd->khd", X, W_V).contiguous()
    return q_proj, k_proj, v_proj


def _flash_absorbed_u_project(q_raw, W_Q, W_K):
    T_q, d_q = q_raw.shape
    _, H, D_H = W_Q.shape
    q_proj = (q_raw @ W_Q.reshape(d_q, H * D_H)).view(T_q, H, D_H)
    return torch.einsum("qhd,ahd->qha", q_proj, W_K).contiguous()


def _flash_attn_varlen_call(
    q_proj,
    k_proj,
    v_proj,
    cu_seqlens_q,
    cu_seqlens_k,
    scale,
):
    flash_attn_varlen_func = _import_flash_attn_varlen_func()
    if flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn is not installed")

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_seqlen_q = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
    max_seqlen_k = int(k_lens.max().item()) if k_lens.numel() > 0 else 0

    return flash_attn_varlen_func(
        q_proj,
        k_proj,
        v_proj,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
    )


def flash_attn_explicit_varlen(
    q_raw: torch.Tensor,
    X: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    FlashAttention varlen baseline with explicit K/V materialization.

    This is the strong CUDA baseline for the non-absorbed formulation:
      q = q_raw @ W_Q
      k = X @ W_K
      v = X @ W_V
      O = flash_attn_varlen(q, k, v)
    """
    flash_attn_varlen_func = _import_flash_attn_varlen_func()
    if flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn is not installed")

    _, H, D_H = W_Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D_H)

    q_proj, k_proj, v_proj = _flash_qkv_project(q_raw, X, W_Q, W_K, W_V)
    return _flash_attn_varlen_call(
        q_proj,
        k_proj,
        v_proj,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
    )


def flash_attn_absorbed_varlen(
    q_raw: torch.Tensor,
    X: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    FlashAttention backend for the matrix-absorbed formulation.

    This keeps K/V projection absorbed:
      u      = (q_raw @ W_Q) @ W_K^T       [T_q, H, D_KV]
      alphaX = flash_attn_varlen(u, X, X)  [T_q, H, D_KV]
      O      = alphaX @ W_V                [T_q, H, D_H]

    It materializes alphaX, so it is not as memory-frugal as the Triton kernel,
    but it uses the optimized CUDA FlashAttention backend for the attention
    part and is a useful performance ceiling for the absorbed math path.
    """
    if _import_flash_attn_varlen_func() is None:
        raise RuntimeError("flash-attn is not installed")

    _, _, D_H = W_Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D_H)

    u = _flash_absorbed_u_project(q_raw, W_Q, W_K)
    x_head = X[:, None, :].contiguous()
    alpha_x = _flash_attn_varlen_call(
        u,
        x_head,
        x_head,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
    )
    return torch.einsum("qha,ahd->qhd", alpha_x, W_V)


# -------------------------------------------------------------------------
# Test harness
# -------------------------------------------------------------------------
def _make_inputs(
    q_lens: list, k_lens: list,
    d_q: int = _DEFAULT_D_Q, d_kv: int = _DEFAULT_D_KV,
    H: int = _DEFAULT_H, D_H: int = _DEFAULT_D_H,
    dtype=torch.float32,
    device="cuda",
    seed: int = 0,
):
    torch.manual_seed(seed)
    T_q = sum(q_lens)
    T_k = sum(k_lens)
    cu_q = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=device)
    cu_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(0)
    cu_k = torch.zeros(len(k_lens) + 1, dtype=torch.int32, device=device)
    cu_k[1:] = torch.tensor(k_lens, dtype=torch.int32, device=device).cumsum(0)

    # small init range so softmax doesn't explode and bf16 is fine
    q_raw = torch.randn(T_q, d_q, dtype=dtype, device=device) * 0.1
    X     = torch.randn(T_k, d_kv, dtype=dtype, device=device) * 0.1
    W_Q   = torch.randn(d_q, H, D_H, dtype=dtype, device=device) * 0.1
    W_K   = torch.randn(d_kv, H, D_H, dtype=dtype, device=device) * 0.1
    W_V   = torch.randn(d_kv, H, D_H, dtype=dtype, device=device) * 0.1
    return q_raw, X, W_Q, W_K, W_V, cu_q, cu_k


def _make_many_shape_lens(
    n_seg: int = 1000,
    q_min: int = 1,
    q_max: int = 16,
    k_min: int = 64,
    k_max: int = 2048,
    seed: int = 17,
) -> tuple[list[int], list[int]]:
    """
    Deterministic ragged benchmark shape with many distinct K lengths.

    Defaults create 1000 segments, q_len in [1, 16], and 1000 unique k_len
    values sampled from [64, 2048].
    """
    if n_seg < 1:
        raise ValueError("n_seg must be positive")
    if q_min < 1 or q_max < q_min:
        raise ValueError("expected 1 <= q_min <= q_max")
    if k_min < 1 or k_max < k_min:
        raise ValueError("expected 1 <= k_min <= k_max")
    if n_seg > (k_max - k_min + 1):
        raise ValueError(
            f"cannot make {n_seg} unique k lengths from [{k_min}, {k_max}]"
        )

    # q varies but stays small, matching target-attn usage.
    q_span = q_max - q_min + 1
    q_lens = [
        q_min + ((i * 13 + seed) % q_span)
        for i in range(n_seg)
    ]

    # K lengths are unique and shuffled to stress ragged scheduling.
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_choices = torch.randperm(k_max - k_min + 1, generator=g)[:n_seg]
    k_lens = (k_choices + k_min).tolist()
    return q_lens, k_lens


def _make_shape_suite(start_seed: int = 17) -> list[tuple[str, list[int], list[int]]]:
    """
    A small portfolio of synthetic ragged distributions for benchmark coverage.

    The default long case is intentionally production-like; the others stress
    short-K, mid-K, long-K, tiny-Q, and q-heavy regimes without requiring any
    external data file.
    """
    specs = [
        ("prod-like many-shape", 1000, 1, 16, 64, 2048, start_seed),
        ("short-k many-shape", 384, 1, 16, 16, 512, start_seed + 101),
        ("mid-k many-shape", 768, 1, 16, 128, 1536, start_seed + 202),
        ("long-k fewer-seg", 512, 1, 16, 1024, 4096, start_seed + 303),
        ("tiny-q large-k", 1000, 1, 4, 256, 4096, start_seed + 404),
        ("q-heavy prod-k", 1000, 8, 32, 64, 2048, start_seed + 505),
    ]
    cases = []
    for name, n_seg, q_min, q_max, k_min, k_max, seed in specs:
        q_lens, k_lens = _make_many_shape_lens(
            n_seg=n_seg,
            q_min=q_min,
            q_max=q_max,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
        )
        cases.append((f"{name} seed={seed}", q_lens, k_lens))
    return cases


def _normalize_lens_case(raw, index: int) -> tuple[str, list[int], list[int]]:
    if not isinstance(raw, dict):
        raise ValueError(f"case #{index} must be an object")
    if raw.get("name") is not None:
        name = str(raw["name"])
    elif raw.get("tag") is not None:
        name = f"{raw['tag']} record {index}"
    else:
        name = f"lens-file case {index}"
    q_lens = raw.get("q_lens", raw.get("qlens", raw.get("q")))
    k_lens = raw.get("k_lens", raw.get("klens", raw.get("k")))
    if q_lens is None or k_lens is None:
        raise ValueError(
            f"{name}: expected q_lens/k_lens fields, or aliases q/k"
        )
    q_lens = [int(x) for x in q_lens]
    k_lens = [int(x) for x in k_lens]
    if len(q_lens) != len(k_lens):
        raise ValueError(f"{name}: q_lens and k_lens length mismatch")
    if not q_lens:
        raise ValueError(f"{name}: empty lens")
    if any(x < 0 for x in q_lens) or any(x < 0 for x in k_lens):
        raise ValueError(f"{name}: lengths must be non-negative")
    if sum(q_lens) <= 0:
        raise ValueError(f"{name}: total q length must be positive")
    if sum(k_lens) <= 0:
        raise ValueError(f"{name}: total k length must be positive")
    bad_segments = [i for i, (q, k) in enumerate(zip(q_lens, k_lens)) if q > 0 and k <= 0]
    if bad_segments:
        raise ValueError(
            f"{name}: segments with q_len > 0 need k_len > 0; first bad index {bad_segments[0]}"
        )
    return name, q_lens, k_lens


def _load_lens_cases(
    path: str,
    offset: int = 0,
    limit: int | None = None,
) -> list[tuple[str, list[int], list[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{path}: empty lens file")

    if path.endswith(".jsonl"):
        raw_cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict) and "cases" in payload:
            raw_cases = payload["cases"]
        elif isinstance(payload, dict) and "records" in payload:
            raw_cases = payload["records"]
        elif isinstance(payload, list):
            raw_cases = payload
        else:
            raw_cases = [payload]
    if offset < 0:
        raise ValueError("--bench-lens-offset must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("--bench-lens-limit must be positive")
    raw_cases = raw_cases[offset:]
    if limit is not None:
        raw_cases = raw_cases[:limit]
    return [
        _normalize_lens_case(raw, offset + i)
        for i, raw in enumerate(raw_cases)
    ]


def _lens_stats(lens: list[int]) -> str:
    total = sum(lens)
    return (
        f"n={len(lens)} total={total} min={min(lens)} "
        f"mean={total / len(lens):.1f} max={max(lens)}"
    )


def _bench_title(name: str, q_lens: list[int], k_lens: list[int], dtype) -> str:
    if len(q_lens) <= 16:
        return f"{name} | q_lens={q_lens} k_lens={k_lens} dtype={dtype}"
    return (
        f"{name} | segments={len(q_lens)} dtype={dtype} | "
        f"q[{_lens_stats(q_lens)}] | k[{_lens_stats(k_lens)}]"
    )


def _launch_fwd_kernel(grid, u, X, W_V, O, LSE, cu_q, cu_k, q_idx, scale, H, d_kv, B_DKV, D_H):
    fwd_kernel = _fwd_alpha_kernel if _select_fwd_mode() == "alpha" else _fwd_kernel
    fwd_kernel[grid](
        u, X, W_V, O, LSE,
        cu_q, cu_k, q_idx,
        scale,
        H=H, D_KV=d_kv, B_DKV=B_DKV, D_H=D_H,
        BT_Q=_BT_Q_FWD, BT_K=_BT_K_FWD,
        num_warps=4, num_stages=1,
    )


def _compare(name, a, b, atol, rtol):
    a32, b32 = a.float(), b.float()
    abs_diff = (a32 - b32).abs()
    rel_diff = abs_diff / (b32.abs() + 1e-6)
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    pct_close = ((abs_diff < atol) | (rel_diff < rtol)).float().mean().item()
    status = "PASS" if (max_abs < atol or max_rel < rtol or pct_close > 0.999) else "FAIL"
    print(f"  [{status}] {name:12s}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  close={pct_close*100:.2f}%")
    return status == "PASS"


def _run_case(
    name: str,
    q_lens: list, k_lens: list,
    d_q=_DEFAULT_D_Q, d_kv=_DEFAULT_D_KV, H=_DEFAULT_H, D_H=_DEFAULT_D_H,
    dtype=torch.float32,
    atol=1e-3, rtol=1e-3,
    test_bwd=True,
):
    print(f"\n=== {name} | q_lens={q_lens} k_lens={k_lens} dtype={dtype} ===")
    q_raw, X, W_Q, W_K, W_V, cu_q, cu_k = _make_inputs(
        q_lens, k_lens, d_q, d_kv, H, D_H, dtype
    )

    # --- forward ---
    q_raw_a = q_raw.clone().detach().requires_grad_(test_bwd)
    X_a     = X.clone().detach().requires_grad_(test_bwd)
    W_Q_a   = W_Q.clone().detach().requires_grad_(test_bwd)
    W_K_a   = W_K.clone().detach().requires_grad_(test_bwd)
    W_V_a   = W_V.clone().detach().requires_grad_(test_bwd)

    q_raw_r = q_raw.clone().detach().requires_grad_(test_bwd)
    X_r     = X.clone().detach().requires_grad_(test_bwd)
    W_Q_r   = W_Q.clone().detach().requires_grad_(test_bwd)
    W_K_r   = W_K.clone().detach().requires_grad_(test_bwd)
    W_V_r   = W_V.clone().detach().requires_grad_(test_bwd)

    O_tri = packed_target_attn(
        q_raw_a, X_a, W_Q_a, W_K_a, W_V_a,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    )
    O_ref = reference_target_attn(q_raw_r, X_r, W_Q_r, W_K_r, W_V_r, cu_q, cu_k)

    all_pass = True
    all_pass &= _compare("O", O_tri, O_ref, atol, rtol)

    if not test_bwd:
        return all_pass

    # --- backward ---
    dO = torch.randn_like(O_ref) * 0.1
    O_tri.backward(dO)
    O_ref.backward(dO)

    all_pass &= _compare("d q_raw", q_raw_a.grad, q_raw_r.grad, atol, rtol)
    all_pass &= _compare("dX",      X_a.grad,    X_r.grad,    atol, rtol)
    all_pass &= _compare("dW_Q",    W_Q_a.grad,  W_Q_r.grad,  atol, rtol)
    all_pass &= _compare("dW_K",    W_K_a.grad,  W_K_r.grad,  atol, rtol)
    all_pass &= _compare("dW_V",    W_V_a.grad,  W_V_r.grad,  atol, rtol)

    return all_pass


def _time_cuda(fn, *, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_wall(fn, sync, *, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        out = fn()
        sync(out)

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        sync(out)
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _project_u(q_raw, W_Q, W_K):
    T_q, d_q = q_raw.shape
    _, H, D_H = W_Q.shape
    q_proj = (q_raw @ W_Q.reshape(d_q, H * D_H)).view(T_q, H, D_H)
    return torch.einsum("thd,khd->thk", q_proj, W_K).contiguous()


def _run_benchmark_breakdown(
    name: str,
    q_lens: list,
    k_lens: list,
    d_q=_DEFAULT_D_Q,
    d_kv=_DEFAULT_D_KV,
    H=_DEFAULT_H,
    D_H=_DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
    bench_dx_all_modes: bool = False,
):
    print(f"\n=== breakdown {_bench_title(name, q_lens, k_lens, dtype)} ===")
    q_raw, X, W_Q, W_K, W_V, cu_q, cu_k = _make_inputs(
        q_lens, k_lens, d_q, d_kv, H, D_H, dtype
    )
    T_q = q_raw.shape[0]
    T_k = X.shape[0]
    scale = 1.0 / math.sqrt(D_H)
    B_DKV = _next_power_of_2(d_kv)
    q_idx = prepare_q_chunk_indices(cu_q, _BT_Q_FWD)
    k_idx = prepare_k_chunk_indices(cu_k, _BT_K_BWD)

    outer_ms = _time_cuda(
        lambda: _project_u(q_raw, W_Q, W_K),
        warmup=warmup,
        iters=iters,
    )
    print(f"  outer u projection    {outer_ms:.3f} ms")

    dU = torch.randn((T_q, H, d_kv), dtype=dtype, device=q_raw.device) * 0.1
    q_outer = q_raw.clone().detach().requires_grad_(True)
    W_Q_outer = W_Q.clone().detach().requires_grad_(True)
    W_K_outer = W_K.clone().detach().requires_grad_(True)

    def outer_fwd_bwd():
        q_outer.grad = None
        W_Q_outer.grad = None
        W_K_outer.grad = None
        _project_u(q_outer, W_Q_outer, W_K_outer).backward(dU)

    outer_fwd_bwd_ms = _time_cuda(
        outer_fwd_bwd,
        warmup=warmup,
        iters=iters,
    )
    print(f"  outer u fwd+bwd       {outer_fwd_bwd_ms:.3f} ms")

    u = _project_u(q_raw, W_Q, W_K)
    X_c = X.contiguous()
    W_V_c = W_V.contiguous()
    O = torch.empty((T_q, H, D_H), dtype=dtype, device=q_raw.device)
    LSE = torch.empty((T_q, H), dtype=torch.float32, device=q_raw.device)

    def inner_fwd():
        _launch_fwd_kernel(
            (q_idx.shape[0], H),
            u, X_c, W_V_c, O, LSE,
            cu_q, cu_k, q_idx,
            scale, H, d_kv, B_DKV, D_H,
        )

    inner_fwd_ms = _time_cuda(inner_fwd, warmup=warmup, iters=iters)
    print(f"  inner triton fwd      {inner_fwd_ms:.3f} ms")

    full_fwd_ms = _time_cuda(
        lambda: packed_target_attn(q_raw, X, W_Q, W_K, W_V, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k),
        warmup=warmup,
        iters=iters,
    )
    print(f"  full wrapper fwd      {full_fwd_ms:.3f} ms")
    full_fwd_fast_ms = _time_cuda(
        lambda: packed_target_attn(
            q_raw, X, W_Q, W_K, W_V,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            q_chunk_indices=q_idx, k_chunk_indices=k_idx,
            validate=False,
        ),
        warmup=warmup,
        iters=iters,
    )
    print(f"  full wrapper fwd fast {full_fwd_fast_ms:.3f} ms")

    dO = torch.randn((T_q, H, D_H), dtype=dtype, device=q_raw.device) * 0.1
    q_full = q_raw.clone().detach().requires_grad_(True)
    X_full = X.clone().detach().requires_grad_(True)
    W_Q_full = W_Q.clone().detach().requires_grad_(True)
    W_K_full = W_K.clone().detach().requires_grad_(True)
    W_V_full = W_V.clone().detach().requires_grad_(True)

    def full_fwd_bwd_fast():
        q_full.grad = None
        X_full.grad = None
        W_Q_full.grad = None
        W_K_full.grad = None
        W_V_full.grad = None
        packed_target_attn(
            q_full, X_full, W_Q_full, W_K_full, W_V_full,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            q_chunk_indices=q_idx, k_chunk_indices=k_idx,
            validate=False,
        ).backward(dO)

    full_fwd_bwd_fast_ms = _time_cuda(
        full_fwd_bwd_fast,
        warmup=warmup,
        iters=iters,
    )
    print(f"  full fwd+bwd fast     {full_fwd_bwd_fast_ms:.3f} ms")

    inner_fwd()
    delta = torch.empty((T_q, H), dtype=torch.float32, device=q_raw.device)
    du = torch.empty_like(u)
    dAlphaX_fp32 = torch.empty((T_q, H, d_kv), dtype=torch.float32, device=q_raw.device)
    dW_V_fp32 = torch.zeros((d_kv, H, D_H), dtype=torch.float32, device=q_raw.device)
    dX_fp32 = torch.empty((T_k, d_kv), dtype=torch.float32, device=q_raw.device)

    dx_mode = _select_dx_mode(k_idx.shape[0])
    measure_partial_dx = bench_dx_all_modes or dx_mode == "partial"
    measure_fused_dx = bench_dx_all_modes or dx_mode == "fused"
    measure_atomic_dx = bench_dx_all_modes or dx_mode == "atomic"
    dX_partial_fp32 = (
        torch.empty((H, T_k, d_kv), dtype=torch.float32, device=q_raw.device)
        if measure_partial_dx
        else None
    )
    dX_fused_fp32 = (
        torch.empty((T_k, d_kv), dtype=torch.float32, device=q_raw.device)
        if measure_fused_dx
        else None
    )
    dX_atomic_fp32 = (
        torch.empty((T_k, d_kv), dtype=torch.float32, device=q_raw.device)
        if measure_atomic_dx
        else None
    )

    u_inner = u.clone().detach().requires_grad_(True)
    X_inner = X_c.clone().detach().requires_grad_(True)
    W_V_inner = W_V_c.clone().detach().requires_grad_(True)

    def inner_fwd_bwd():
        u_inner.grad = None
        X_inner.grad = None
        W_V_inner.grad = None
        target_attn_inner(
            u_inner, X_inner, W_V_inner, cu_q, cu_k, scale,
            q_chunk_indices=q_idx, k_chunk_indices=k_idx,
            validate=False,
        ).backward(dO)

    inner_fwd_bwd_ms = _time_cuda(
        inner_fwd_bwd,
        warmup=warmup,
        iters=iters,
    )
    print(f"  inner fwd+bwd         {inner_fwd_bwd_ms:.3f} ms")

    def bwd_preprocess():
        _bwd_preprocess_kernel[(triton.cdiv(T_q, _BT_Q_FWD), H)](
            O, dO, delta,
            T_q,
            H=H, D_H=D_H, BT=_BT_Q_FWD,
            num_warps=4, num_stages=2,
        )

    preprocess_ms = _time_cuda(bwd_preprocess, warmup=warmup, iters=iters)
    print(f"  bwd preprocess        {preprocess_ms:.3f} ms")
    bwd_preprocess()

    def bwd_du():
        dW_V_fp32.zero_()
        _bwd_du_kernel[(q_idx.shape[0], H)](
            u, X_c, W_V_c, dO, du, dAlphaX_fp32, dW_V_fp32, LSE, delta,
            cu_q, cu_k, q_idx,
            scale,
            H=H, D_KV=d_kv, B_DKV=B_DKV, D_H=D_H,
            BT_Q=_BT_Q_FWD, BT_K=_BT_K_FWD,
            num_warps=4, num_stages=1,
        )

    bwd_du_ms = _time_cuda(bwd_du, warmup=warmup, iters=iters)
    print(f"  bwd du + dW_V         {bwd_du_ms:.3f} ms")
    bwd_du()

    bwd_dx_ms = None
    bwd_dx_fused_heads_ms = None
    bwd_dx_atomic_heads_ms = None

    if measure_partial_dx:
        def bwd_dx_partial():
            _bwd_dx_kernel[(k_idx.shape[0], H)](
                u, X_c, dAlphaX_fp32, dX_partial_fp32, T_k, LSE, delta,
                cu_q, cu_k, k_idx,
                scale,
                H=H, D_KV=d_kv, B_DKV=B_DKV, D_H=D_H,
                BT_K=_BT_K_BWD, BT_Q=_BT_Q_BWD,
                num_warps=4, num_stages=1,
            )

        def bwd_dx_reduce():
            _bwd_dx_reduce_kernel[(triton.cdiv(T_k, _BT_K_BWD),)](
                dX_partial_fp32, dX_fp32, T_k,
                H=H, D_KV=d_kv, B_DKV=B_DKV, BT_K=_BT_K_BWD,
                num_warps=4, num_stages=1,
            )

        def bwd_dx():
            bwd_dx_partial()
            bwd_dx_reduce()

        bwd_dx_partial_ms = _time_cuda(bwd_dx_partial, warmup=warmup, iters=iters)
        bwd_dx_partial()
        bwd_dx_reduce_ms = _time_cuda(bwd_dx_reduce, warmup=warmup, iters=iters)
        bwd_dx_ms = _time_cuda(bwd_dx, warmup=warmup, iters=iters)
        print(f"  bwd dX partial        {bwd_dx_partial_ms:.3f} ms")
        print(f"  bwd dX reduce         {bwd_dx_reduce_ms:.3f} ms")
        print(f"  bwd dX total          {bwd_dx_ms:.3f} ms")

    if measure_fused_dx:
        def bwd_dx_fused_heads():
            _bwd_dx_fused_heads_kernel[(k_idx.shape[0],)](
                u, X_c, dAlphaX_fp32, dX_fused_fp32, LSE, delta,
                cu_q, cu_k, k_idx,
                scale,
                H=H, D_KV=d_kv, B_DKV=B_DKV, D_H=D_H,
                BT_K=_BT_K_BWD, BT_Q=_BT_Q_BWD,
                num_warps=4, num_stages=1,
            )

        bwd_dx_fused_heads_ms = _time_cuda(
            bwd_dx_fused_heads,
            warmup=warmup,
            iters=iters,
        )
        print(f"  bwd dX fused-heads    {bwd_dx_fused_heads_ms:.3f} ms")

    if measure_atomic_dx:
        def bwd_dx_atomic_heads():
            dX_atomic_fp32.zero_()
            _bwd_dx_atomic_heads_kernel[(k_idx.shape[0], H)](
                u, X_c, dAlphaX_fp32, dX_atomic_fp32, LSE, delta,
                cu_q, cu_k, k_idx,
                scale,
                H=H, D_KV=d_kv, B_DKV=B_DKV, D_H=D_H,
                BT_K=_BT_K_BWD, BT_Q=_BT_Q_BWD,
                num_warps=4, num_stages=1,
            )

        bwd_dx_atomic_heads_ms = _time_cuda(
            bwd_dx_atomic_heads,
            warmup=warmup,
            iters=iters,
        )
        print(f"  bwd dX atomic-heads   {bwd_dx_atomic_heads_ms:.3f} ms")

    if dx_mode == "fused":
        chosen_dx_ms = bwd_dx_fused_heads_ms
        chosen_name = "fused-heads"
    elif dx_mode == "atomic":
        chosen_dx_ms = bwd_dx_atomic_heads_ms
        chosen_name = "atomic-heads"
    else:
        chosen_dx_ms = bwd_dx_ms
        chosen_name = "partial+reduce"
    print(f"  bwd dX chosen         {chosen_name} ({chosen_dx_ms:.3f} ms)")
    print(f"  manual inner bwd sum  {preprocess_ms + bwd_du_ms + chosen_dx_ms:.3f} ms")


def _run_flash_breakdown(
    name: str,
    q_lens: list,
    k_lens: list,
    d_q=_DEFAULT_D_Q,
    d_kv=_DEFAULT_D_KV,
    H=_DEFAULT_H,
    D_H=_DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
    ref_warmup: int = 3,
    ref_iters: int = 10,
):
    print(f"\n=== flash breakdown {_bench_title(name, q_lens, k_lens, dtype)} ===")
    if _import_flash_attn_varlen_func() is None:
        print("  SKIP: flash-attn is not installed")
        return

    q_raw, X, W_Q, W_K, W_V, cu_q, cu_k = _make_inputs(
        q_lens, k_lens, d_q, d_kv, H, D_H, dtype
    )
    scale = 1.0 / math.sqrt(D_H)
    dO = torch.randn((sum(q_lens), H, D_H), dtype=dtype, device=q_raw.device) * 0.1

    print("  -- explicit K/V backend --")
    qkv_fwd_ms = _time_cuda(
        lambda: _flash_qkv_project(q_raw, X, W_Q, W_K, W_V),
        warmup=warmup,
        iters=iters,
    )
    print(f"  flash qkv proj fwd        {qkv_fwd_ms:.3f} ms")

    q_proj, k_proj, v_proj = _flash_qkv_project(q_raw, X, W_Q, W_K, W_V)

    fa_only_fwd_ms = _time_cuda(
        lambda: _flash_attn_varlen_call(q_proj, k_proj, v_proj, cu_q, cu_k, scale),
        warmup=ref_warmup,
        iters=ref_iters,
    )
    print(f"  flash FA-only fwd         {fa_only_fwd_ms:.3f} ms")

    full_fwd_ms = _time_cuda(
        lambda: flash_attn_explicit_varlen(q_raw, X, W_Q, W_K, W_V, cu_q, cu_k, scale),
        warmup=ref_warmup,
        iters=ref_iters,
    )
    print(f"  flash full explicit fwd   {full_fwd_ms:.3f} ms")
    print(f"  flash fwd proj+FA approx  {qkv_fwd_ms + fa_only_fwd_ms:.3f} ms")

    q_p = q_raw.clone().detach().requires_grad_(True)
    X_p = X.clone().detach().requires_grad_(True)
    W_Q_p = W_Q.clone().detach().requires_grad_(True)
    W_K_p = W_K.clone().detach().requires_grad_(True)
    W_V_p = W_V.clone().detach().requires_grad_(True)
    dQ = torch.randn_like(q_proj)
    dK = torch.randn_like(k_proj)
    dV = torch.randn_like(v_proj)

    def qkv_fwd_bwd():
        q_p.grad = None
        X_p.grad = None
        W_Q_p.grad = None
        W_K_p.grad = None
        W_V_p.grad = None
        q_tmp, k_tmp, v_tmp = _flash_qkv_project(q_p, X_p, W_Q_p, W_K_p, W_V_p)
        torch.autograd.backward((q_tmp, k_tmp, v_tmp), (dQ, dK, dV))

    qkv_fwd_bwd_ms = _time_cuda(qkv_fwd_bwd, warmup=warmup, iters=iters)
    print(f"  flash qkv proj fwd+bwd    {qkv_fwd_bwd_ms:.3f} ms")

    q_leaf = q_proj.detach().clone().requires_grad_(True)
    k_leaf = k_proj.detach().clone().requires_grad_(True)
    v_leaf = v_proj.detach().clone().requires_grad_(True)

    def fa_only_fwd_bwd():
        q_leaf.grad = None
        k_leaf.grad = None
        v_leaf.grad = None
        _flash_attn_varlen_call(q_leaf, k_leaf, v_leaf, cu_q, cu_k, scale).backward(dO)

    fa_only_fwd_bwd_ms = _time_cuda(
        fa_only_fwd_bwd,
        warmup=ref_warmup,
        iters=ref_iters,
    )
    print(f"  flash FA-only fwd+bwd     {fa_only_fwd_bwd_ms:.3f} ms")

    q_f = q_raw.clone().detach().requires_grad_(True)
    X_f = X.clone().detach().requires_grad_(True)
    W_Q_f = W_Q.clone().detach().requires_grad_(True)
    W_K_f = W_K.clone().detach().requires_grad_(True)
    W_V_f = W_V.clone().detach().requires_grad_(True)

    def full_fwd_bwd():
        q_f.grad = None
        X_f.grad = None
        W_Q_f.grad = None
        W_K_f.grad = None
        W_V_f.grad = None
        flash_attn_explicit_varlen(q_f, X_f, W_Q_f, W_K_f, W_V_f, cu_q, cu_k, scale).backward(dO)

    full_fwd_bwd_ms = _time_cuda(
        full_fwd_bwd,
        warmup=ref_warmup,
        iters=ref_iters,
    )
    print(f"  flash full explicit fwd+bwd {full_fwd_bwd_ms:.3f} ms")
    print(f"  flash bwd proj+FA approx    {qkv_fwd_bwd_ms + fa_only_fwd_bwd_ms:.3f} ms")


def _import_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        import tensorflow as tf
    except ImportError:
        return None

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        # TensorFlow may already be initialized; this is just a best-effort guard.
        pass
    return tf


def _tf_sync(tf, out):
    if hasattr(tf.experimental, "async_wait"):
        tf.experimental.async_wait()
        return
    if isinstance(out, (list, tuple)):
        out = out[0]
    out.numpy()


def _to_tf(tf, x: torch.Tensor, dtype):
    # NumPy has spotty bfloat16 support, so move through fp32 and cast in TF.
    return tf.convert_to_tensor(x.detach().float().cpu().numpy(), dtype=dtype)


def _run_tf_benchmark(
    q_lens: list,
    k_lens: list,
    d_q=_DEFAULT_D_Q,
    d_kv=_DEFAULT_D_KV,
    H=_DEFAULT_H,
    D_H=_DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 5,
    iters: int = 20,
    triton_fwd_ms=None,
    triton_fwd_bwd_ms=None,
):
    def suffix(ms, base_ms):
        if base_ms is None:
            return ""
        return f"   speedup={ms / base_ms:.2f}x"

    tf = _import_tensorflow()
    if tf is None:
        print("\n=== tensorflow explicit/absorbed baselines ===")
        print("  SKIP: TensorFlow is not installed in this environment")
        return {}

    print(f"\n=== tensorflow explicit/absorbed baselines | dtype={dtype} ===")
    metrics = {}
    q_raw, X, W_Q, W_K, W_V, cu_q, cu_k = _make_inputs(
        q_lens, k_lens, d_q, d_kv, H, D_H, dtype
    )
    tf_dtype = tf.bfloat16 if dtype == torch.bfloat16 else tf.float32
    scale = tf.constant(1.0 / math.sqrt(D_H), dtype=tf.float32)
    q_bounds = cu_q.cpu().tolist()
    k_bounds = cu_k.cpu().tolist()

    q_tf = _to_tf(tf, q_raw, tf_dtype)
    X_tf = _to_tf(tf, X, tf_dtype)
    W_Q_tf = _to_tf(tf, W_Q, tf_dtype)
    W_K_tf = _to_tf(tf, W_K, tf_dtype)
    W_V_tf = _to_tf(tf, W_V, tf_dtype)
    dO_tf = tf.random.normal((sum(q_lens), H, D_H), dtype=tf.float32) * 0.1
    dO_tf = tf.cast(dO_tf, tf_dtype)

    @tf.function
    def tf_explicit_fwd(q_raw_t, X_t, W_Q_t, W_K_t, W_V_t):
        outs = []
        for i in range(len(q_bounds) - 1):
            q_i = q_raw_t[q_bounds[i]:q_bounds[i + 1]]
            x_i = X_t[k_bounds[i]:k_bounds[i + 1]]
            q_proj = tf.einsum("qa,ahd->qhd", q_i, W_Q_t)
            k_proj = tf.einsum("ka,ahd->khd", x_i, W_K_t)
            v_proj = tf.einsum("ka,ahd->khd", x_i, W_V_t)
            s = tf.einsum("qhd,khd->hqk", q_proj, k_proj)
            s = tf.cast(s, tf.float32) * scale
            p = tf.cast(tf.nn.softmax(s, axis=-1), tf_dtype)
            outs.append(tf.einsum("hqk,khd->qhd", p, v_proj))
        return tf.concat(outs, axis=0)

    @tf.function
    def tf_explicit_fwd_bwd(q_raw_t, X_t, W_Q_t, W_K_t, W_V_t, dO_t):
        with tf.GradientTape() as tape:
            tape.watch([q_raw_t, X_t, W_Q_t, W_K_t, W_V_t])
            O = tf_explicit_fwd(q_raw_t, X_t, W_Q_t, W_K_t, W_V_t)
            loss = tf.reduce_sum(tf.cast(O * dO_t, tf.float32))
        grads = tape.gradient(loss, [q_raw_t, X_t, W_Q_t, W_K_t, W_V_t])
        return tf.add_n([tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads if g is not None])

    @tf.function
    def tf_absorbed_fwd(q_raw_t, X_t, W_Q_t, W_K_t, W_V_t):
        outs = []
        for i in range(len(q_bounds) - 1):
            q_i = q_raw_t[q_bounds[i]:q_bounds[i + 1]]
            x_i = X_t[k_bounds[i]:k_bounds[i + 1]]
            q_proj = tf.einsum("qa,ahd->qhd", q_i, W_Q_t)
            u = tf.einsum("qhd,ahd->qha", q_proj, W_K_t)
            s = tf.einsum("qha,ka->hqk", u, x_i)
            s = tf.cast(s, tf.float32) * scale
            p = tf.cast(tf.nn.softmax(s, axis=-1), tf_dtype)
            alpha_x = tf.einsum("hqk,ka->qha", p, x_i)
            outs.append(tf.einsum("qha,ahd->qhd", alpha_x, W_V_t))
        return tf.concat(outs, axis=0)

    @tf.function
    def tf_absorbed_fwd_bwd(q_raw_t, X_t, W_Q_t, W_K_t, W_V_t, dO_t):
        with tf.GradientTape() as tape:
            tape.watch([q_raw_t, X_t, W_Q_t, W_K_t, W_V_t])
            O = tf_absorbed_fwd(q_raw_t, X_t, W_Q_t, W_K_t, W_V_t)
            loss = tf.reduce_sum(tf.cast(O * dO_t, tf.float32))
        grads = tape.gradient(loss, [q_raw_t, X_t, W_Q_t, W_K_t, W_V_t])
        return tf.add_n([tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads if g is not None])

    fwd_ms = _time_wall(
        lambda: tf_explicit_fwd(q_tf, X_tf, W_Q_tf, W_K_tf, W_V_tf),
        lambda out: _tf_sync(tf, out),
        warmup=warmup,
        iters=iters,
    )
    metrics["tf_explicit_fwd"] = fwd_ms
    print(f"  tensorflow explicit fwd     {fwd_ms:.3f} ms{suffix(fwd_ms, triton_fwd_ms)}")

    fwd_bwd_ms = _time_wall(
        lambda: tf_explicit_fwd_bwd(q_tf, X_tf, W_Q_tf, W_K_tf, W_V_tf, dO_tf),
        lambda out: _tf_sync(tf, out),
        warmup=warmup,
        iters=iters,
    )
    metrics["tf_explicit_fwd_bwd"] = fwd_bwd_ms
    print(
        f"  tensorflow explicit fwd+bwd {fwd_bwd_ms:.3f} ms"
        f"{suffix(fwd_bwd_ms, triton_fwd_bwd_ms)}"
    )

    absorbed_fwd_ms = _time_wall(
        lambda: tf_absorbed_fwd(q_tf, X_tf, W_Q_tf, W_K_tf, W_V_tf),
        lambda out: _tf_sync(tf, out),
        warmup=warmup,
        iters=iters,
    )
    metrics["tf_absorbed_fwd"] = absorbed_fwd_ms
    print(
        f"  tensorflow absorbed fwd     {absorbed_fwd_ms:.3f} ms"
        f"{suffix(absorbed_fwd_ms, triton_fwd_ms)}"
    )

    absorbed_fwd_bwd_ms = _time_wall(
        lambda: tf_absorbed_fwd_bwd(q_tf, X_tf, W_Q_tf, W_K_tf, W_V_tf, dO_tf),
        lambda out: _tf_sync(tf, out),
        warmup=warmup,
        iters=iters,
    )
    metrics["tf_absorbed_fwd_bwd"] = absorbed_fwd_bwd_ms
    print(
        f"  tensorflow absorbed fwd+bwd {absorbed_fwd_bwd_ms:.3f} ms"
        f"{suffix(absorbed_fwd_bwd_ms, triton_fwd_bwd_ms)}"
    )
    return metrics


def _run_benchmark(
    name: str,
    q_lens: list,
    k_lens: list,
    d_q=_DEFAULT_D_Q,
    d_kv=_DEFAULT_D_KV,
    H=_DEFAULT_H,
    D_H=_DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
    compare_ref: bool = True,
    ref_warmup: int = 3,
    ref_iters: int = 10,
    compare_flash: bool = False,
    compare_tf: bool = False,
    tf_warmup: int = 5,
    tf_iters: int = 20,
):
    print(f"\n=== bench {_bench_title(name, q_lens, k_lens, dtype)} ===")
    metrics = {}
    q_raw, X, W_Q, W_K, W_V, cu_q, cu_k = _make_inputs(
        q_lens, k_lens, d_q, d_kv, H, D_H, dtype
    )
    q_idx, k_idx = prepare_target_attn_indices(cu_q, cu_k)

    def fwd():
        return packed_target_attn(
            q_raw, X, W_Q, W_K, W_V,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            q_chunk_indices=q_idx, k_chunk_indices=k_idx,
            validate=False,
        )

    fwd_ms = _time_cuda(fwd, warmup=warmup, iters=iters)
    metrics["triton_fwd"] = fwd_ms
    print(f"  triton forward        {fwd_ms:.3f} ms")

    q_b = q_raw.clone().detach().requires_grad_(True)
    X_b = X.clone().detach().requires_grad_(True)
    W_Q_b = W_Q.clone().detach().requires_grad_(True)
    W_K_b = W_K.clone().detach().requires_grad_(True)
    W_V_b = W_V.clone().detach().requires_grad_(True)
    dO = torch.randn((sum(q_lens), H, D_H), dtype=dtype, device=q_raw.device) * 0.1

    def fwd_bwd():
        q_b.grad = None
        X_b.grad = None
        W_Q_b.grad = None
        W_K_b.grad = None
        W_V_b.grad = None
        O = packed_target_attn(
            q_b, X_b, W_Q_b, W_K_b, W_V_b,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            q_chunk_indices=q_idx, k_chunk_indices=k_idx,
            validate=False,
        )
        O.backward(dO)

    fwd_bwd_ms = _time_cuda(fwd_bwd, warmup=warmup, iters=iters)
    metrics["triton_fwd_bwd"] = fwd_bwd_ms
    print(f"  triton fwd+bwd        {fwd_bwd_ms:.3f} ms")

    if compare_flash:
        if _import_flash_attn_varlen_func() is None:
            print("  flash-attn explicit   SKIP: flash-attn is not installed")
        else:
            def flash_fwd():
                return flash_attn_explicit_varlen(q_raw, X, W_Q, W_K, W_V, cu_q, cu_k)

            flash_fwd_ms = _time_cuda(flash_fwd, warmup=ref_warmup, iters=ref_iters)
            metrics["flash_fwd"] = flash_fwd_ms
            metrics["flash_fwd_vs_triton"] = flash_fwd_ms / fwd_ms
            print(f"  flash-attn explicit fwd     {flash_fwd_ms:.3f} ms   speedup={flash_fwd_ms / fwd_ms:.2f}x")

            q_f = q_raw.clone().detach().requires_grad_(True)
            X_f = X.clone().detach().requires_grad_(True)
            W_Q_f = W_Q.clone().detach().requires_grad_(True)
            W_K_f = W_K.clone().detach().requires_grad_(True)
            W_V_f = W_V.clone().detach().requires_grad_(True)

            def flash_fwd_bwd():
                q_f.grad = None
                X_f.grad = None
                W_Q_f.grad = None
                W_K_f.grad = None
                W_V_f.grad = None
                O = flash_attn_explicit_varlen(q_f, X_f, W_Q_f, W_K_f, W_V_f, cu_q, cu_k)
                O.backward(dO)

            flash_fwd_bwd_ms = _time_cuda(
                flash_fwd_bwd,
                warmup=ref_warmup,
                iters=ref_iters,
            )
            metrics["flash_fwd_bwd"] = flash_fwd_bwd_ms
            metrics["flash_fwd_bwd_vs_triton"] = flash_fwd_bwd_ms / fwd_bwd_ms
            print(f"  flash-attn explicit fwd+bwd {flash_fwd_bwd_ms:.3f} ms   speedup={flash_fwd_bwd_ms / fwd_bwd_ms:.2f}x")

    if compare_tf:
        tf_metrics = _run_tf_benchmark(
            q_lens,
            k_lens,
            d_q=d_q,
            d_kv=d_kv,
            H=H,
            D_H=D_H,
            dtype=dtype,
            warmup=tf_warmup,
            iters=tf_iters,
            triton_fwd_ms=fwd_ms,
            triton_fwd_bwd_ms=fwd_bwd_ms,
        )
        metrics.update(tf_metrics)
        if "tf_explicit_fwd" in metrics:
            metrics["tf_explicit_fwd_vs_triton"] = metrics["tf_explicit_fwd"] / fwd_ms
        if "tf_explicit_fwd_bwd" in metrics:
            metrics["tf_explicit_fwd_bwd_vs_triton"] = (
                metrics["tf_explicit_fwd_bwd"] / fwd_bwd_ms
            )
        if "tf_absorbed_fwd" in metrics:
            metrics["tf_absorbed_fwd_vs_triton"] = metrics["tf_absorbed_fwd"] / fwd_ms
        if "tf_absorbed_fwd_bwd" in metrics:
            metrics["tf_absorbed_fwd_bwd_vs_triton"] = (
                metrics["tf_absorbed_fwd_bwd"] / fwd_bwd_ms
            )

    if not compare_ref:
        return metrics

    def ref_fwd():
        return reference_target_attn_vectorized(q_raw, X, W_Q, W_K, W_V, cu_q, cu_k)

    ref_fwd_ms = _time_cuda(ref_fwd, warmup=ref_warmup, iters=ref_iters)
    metrics["torch_explicit_fwd"] = ref_fwd_ms
    print(f"  torch explicit fwd    {ref_fwd_ms:.3f} ms   speedup={ref_fwd_ms / fwd_ms:.2f}x")

    def ref_absorbed_fwd():
        return reference_target_attn_absorbed_vectorized(q_raw, X, W_Q, W_K, W_V, cu_q, cu_k)

    ref_absorbed_fwd_ms = _time_cuda(ref_absorbed_fwd, warmup=ref_warmup, iters=ref_iters)
    metrics["torch_absorbed_fwd"] = ref_absorbed_fwd_ms
    print(f"  torch absorbed fwd    {ref_absorbed_fwd_ms:.3f} ms   speedup={ref_absorbed_fwd_ms / fwd_ms:.2f}x")

    q_r = q_raw.clone().detach().requires_grad_(True)
    X_r = X.clone().detach().requires_grad_(True)
    W_Q_r = W_Q.clone().detach().requires_grad_(True)
    W_K_r = W_K.clone().detach().requires_grad_(True)
    W_V_r = W_V.clone().detach().requires_grad_(True)

    def ref_fwd_bwd():
        q_r.grad = None
        X_r.grad = None
        W_Q_r.grad = None
        W_K_r.grad = None
        W_V_r.grad = None
        O = reference_target_attn_vectorized(q_r, X_r, W_Q_r, W_K_r, W_V_r, cu_q, cu_k)
        O.backward(dO)

    ref_fwd_bwd_ms = _time_cuda(ref_fwd_bwd, warmup=ref_warmup, iters=ref_iters)
    metrics["torch_explicit_fwd_bwd"] = ref_fwd_bwd_ms
    print(f"  torch explicit fwd+bwd {ref_fwd_bwd_ms:.3f} ms   speedup={ref_fwd_bwd_ms / fwd_bwd_ms:.2f}x")

    q_a = q_raw.clone().detach().requires_grad_(True)
    X_a = X.clone().detach().requires_grad_(True)
    W_Q_a = W_Q.clone().detach().requires_grad_(True)
    W_K_a = W_K.clone().detach().requires_grad_(True)
    W_V_a = W_V.clone().detach().requires_grad_(True)

    def ref_absorbed_fwd_bwd():
        q_a.grad = None
        X_a.grad = None
        W_Q_a.grad = None
        W_K_a.grad = None
        W_V_a.grad = None
        O = reference_target_attn_absorbed_vectorized(q_a, X_a, W_Q_a, W_K_a, W_V_a, cu_q, cu_k)
        O.backward(dO)

    ref_absorbed_fwd_bwd_ms = _time_cuda(
        ref_absorbed_fwd_bwd,
        warmup=ref_warmup,
        iters=ref_iters,
    )
    metrics["torch_absorbed_fwd_bwd"] = ref_absorbed_fwd_bwd_ms
    print(f"  torch absorbed fwd+bwd {ref_absorbed_fwd_bwd_ms:.3f} ms   speedup={ref_absorbed_fwd_bwd_ms / fwd_bwd_ms:.2f}x")
    return metrics


def _metric_stats(values: list[float]) -> str:
    mean = sum(values) / len(values)
    return f"mean={mean:.3f} ms  min={min(values):.3f}  max={max(values):.3f}"


def _ratio_stats(values: list[float]) -> str:
    mean = sum(values) / len(values)
    return f"mean={mean:.3f}x  min={min(values):.3f}  max={max(values):.3f}"


def _parse_int_list(text: str, name: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError(f"{name} entries must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def _parse_dx_mode_list(text: str) -> list[str]:
    values = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in ("auto", "partial", "fused", "atomic"):
            raise ValueError(
                f"sweep dx modes must be auto/partial/fused/atomic, got {part!r}"
            )
        values.append(part)
    if not values:
        raise ValueError("sweep dx mode list must not be empty")
    return values


def _parse_fwd_mode_list(text: str) -> list[str]:
    values = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in ("direct", "alpha"):
            raise ValueError(
                f"sweep fwd modes must be direct/alpha, got {part!r}"
            )
        values.append(part)
    if not values:
        raise ValueError("sweep fwd mode list must not be empty")
    return values


def _format_combo_result(result: dict) -> str:
    ratio = result.get("flash_fwd_bwd_vs_triton_mean")
    ratio_s = f"  flash/triton={ratio:.3f}x" if ratio is not None else ""
    return (
        f"  rank={result['rank']:02d} "
        f"bt_q_bwd={result['bt_q_bwd']:>2d} "
        f"bt_k_bwd={result['bt_k_bwd']:>2d} "
        f"dx={result['dx_mode']:<7s} "
        f"triton fwd+bwd mean={result['triton_fwd_bwd_mean']:.3f} ms "
        f"min={result['triton_fwd_bwd_min']:.3f} "
        f"max={result['triton_fwd_bwd_max']:.3f}"
        f"{ratio_s}"
    )


def _format_forward_combo_result(result: dict) -> str:
    ratio_fwd = result.get("flash_fwd_vs_triton_mean")
    ratio_bwd = result.get("flash_fwd_bwd_vs_triton_mean")
    ratio_parts = []
    if ratio_fwd is not None:
        ratio_parts.append(f"flash/triton fwd={ratio_fwd:.3f}x")
    if ratio_bwd is not None:
        ratio_parts.append(f"flash/triton fwd+bwd={ratio_bwd:.3f}x")
    ratio_s = "  " + "  ".join(ratio_parts) if ratio_parts else ""
    return (
        f"  rank={result['rank']:02d} "
        f"mode={result['fwd_mode']:<6s} "
        f"bt_q_fwd={result['bt_q_fwd']:>2d} "
        f"bt_k_fwd={result['bt_k_fwd']:>3d} "
        f"triton fwd mean={result['triton_fwd_mean']:.3f} ms "
        f"min={result['triton_fwd_min']:.3f} "
        f"max={result['triton_fwd_max']:.3f}  "
        f"fwd+bwd mean={result['triton_fwd_bwd_mean']:.3f} ms"
        f"{ratio_s}"
    )


def _brief_exception(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    first_line = text[0] if text else ""
    if len(first_line) > 220:
        first_line = first_line[:217] + "..."
    return f"{type(exc).__name__}: {first_line}"


def _time_triton_train_case(
    q_lens: list,
    k_lens: list,
    d_q=_DEFAULT_D_Q,
    d_kv=_DEFAULT_D_KV,
    H=_DEFAULT_H,
    D_H=_DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    q_raw, X, W_Q, W_K, W_V, cu_q, cu_k = _make_inputs(
        q_lens, k_lens, d_q, d_kv, H, D_H, dtype
    )
    q_idx, k_idx = prepare_target_attn_indices(cu_q, cu_k)

    def fwd():
        return packed_target_attn(
            q_raw, X, W_Q, W_K, W_V,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            q_chunk_indices=q_idx, k_chunk_indices=k_idx,
            validate=False,
        )

    fwd_ms = _time_cuda(fwd, warmup=warmup, iters=iters)

    q_b = q_raw.clone().detach().requires_grad_(True)
    X_b = X.clone().detach().requires_grad_(True)
    W_Q_b = W_Q.clone().detach().requires_grad_(True)
    W_K_b = W_K.clone().detach().requires_grad_(True)
    W_V_b = W_V.clone().detach().requires_grad_(True)
    dO = torch.randn((sum(q_lens), H, D_H), dtype=dtype, device=q_raw.device) * 0.1

    def fwd_bwd():
        q_b.grad = None
        X_b.grad = None
        W_Q_b.grad = None
        W_K_b.grad = None
        W_V_b.grad = None
        O = packed_target_attn(
            q_b, X_b, W_Q_b, W_K_b, W_V_b,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            q_chunk_indices=q_idx, k_chunk_indices=k_idx,
            validate=False,
        )
        O.backward(dO)

    fwd_bwd_ms = _time_cuda(fwd_bwd, warmup=warmup, iters=iters)
    return {"triton_fwd": fwd_ms, "triton_fwd_bwd": fwd_bwd_ms}


def _time_flash_explicit_train_case(
    q_lens: list,
    k_lens: list,
    d_q=_DEFAULT_D_Q,
    d_kv=_DEFAULT_D_KV,
    H=_DEFAULT_H,
    D_H=_DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 1,
    iters: int = 5,
) -> dict:
    if _import_flash_attn_varlen_func() is None:
        return {}

    q_raw, X, W_Q, W_K, W_V, cu_q, cu_k = _make_inputs(
        q_lens, k_lens, d_q, d_kv, H, D_H, dtype
    )

    def flash_fwd():
        return flash_attn_explicit_varlen(q_raw, X, W_Q, W_K, W_V, cu_q, cu_k)

    flash_fwd_ms = _time_cuda(
        flash_fwd,
        warmup=warmup,
        iters=iters,
    )

    q_f = q_raw.clone().detach().requires_grad_(True)
    X_f = X.clone().detach().requires_grad_(True)
    W_Q_f = W_Q.clone().detach().requires_grad_(True)
    W_K_f = W_K.clone().detach().requires_grad_(True)
    W_V_f = W_V.clone().detach().requires_grad_(True)
    dO = torch.randn((sum(q_lens), H, D_H), dtype=dtype, device=q_raw.device) * 0.1

    def flash_fwd_bwd():
        q_f.grad = None
        X_f.grad = None
        W_Q_f.grad = None
        W_K_f.grad = None
        W_V_f.grad = None
        O = flash_attn_explicit_varlen(q_f, X_f, W_Q_f, W_K_f, W_V_f, cu_q, cu_k)
        O.backward(dO)

    return {
        "flash_fwd": flash_fwd_ms,
        "flash_fwd_bwd": _time_cuda(
            flash_fwd_bwd,
            warmup=warmup,
            iters=iters,
        )
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _run_forward_block_sweep(
    count: int,
    start_seed: int,
    n_seg: int,
    q_min: int,
    q_max: int,
    k_min: int,
    k_max: int,
    bt_q_fwd_values: list[int],
    bt_k_fwd_values: list[int],
    fwd_modes: list[str],
    d_q: int = _DEFAULT_D_Q,
    d_kv: int = _DEFAULT_D_KV,
    H: int = _DEFAULT_H,
    D_H: int = _DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
    compare_flash: bool = False,
    ref_warmup: int = 1,
    ref_iters: int = 5,
) -> None:
    print(
        f"\n=== forward block sweep | cases={count} "
        f"seeds={start_seed}..{start_seed + count - 1} dtype={dtype} "
        f"| d_q={d_q} d_kv={d_kv} H={H} D_H={D_H} ==="
    )
    print(
        f"  bt_q_fwd={bt_q_fwd_values} "
        f"bt_k_fwd={bt_k_fwd_values} fwd_modes={fwd_modes}"
    )

    shapes = []
    for i in range(count):
        seed = start_seed + i
        q_lens, k_lens = _make_many_shape_lens(
            n_seg=n_seg,
            q_min=q_min,
            q_max=q_max,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
        )
        shapes.append((seed, q_lens, k_lens))

    flash_by_seed = {}
    if compare_flash:
        print("\n  computing flash-attn explicit fwd/fwd+bwd baseline once per seed...")
        for seed, q_lens, k_lens in shapes:
            metrics = _time_flash_explicit_train_case(
                q_lens,
                k_lens,
                d_q=d_q,
                d_kv=d_kv,
                H=H,
                D_H=D_H,
                dtype=dtype,
                warmup=ref_warmup,
                iters=ref_iters,
            )
            if metrics:
                flash_by_seed[seed] = metrics
            torch.cuda.empty_cache()
        if flash_by_seed:
            flash_fwd_values = [m["flash_fwd"] for m in flash_by_seed.values()]
            flash_bwd_values = [m["flash_fwd_bwd"] for m in flash_by_seed.values()]
            print(f"  flash explicit fwd     {_metric_stats(flash_fwd_values)}")
            print(f"  flash explicit fwd+bwd {_metric_stats(flash_bwd_values)}")
        else:
            print("  flash explicit SKIP: flash-attn is not installed")

    old_values = {
        "_BT_Q_FWD": _BT_Q_FWD,
        "_BT_K_FWD": _BT_K_FWD,
    }
    old_fwd_mode = os.environ.get("TRITON_TARGET_ATTN_FWD_MODE")
    results = []
    try:
        for fwd_mode in fwd_modes:
            os.environ["TRITON_TARGET_ATTN_FWD_MODE"] = fwd_mode
            for bt_q_fwd in bt_q_fwd_values:
                for bt_k_fwd in bt_k_fwd_values:
                    _override_runtime_constant("_BT_Q_FWD", bt_q_fwd)
                    _override_runtime_constant("_BT_K_FWD", bt_k_fwd)

                    combo_fwd_values = []
                    combo_bwd_values = []
                    fwd_ratios = []
                    bwd_ratios = []
                    status = "OK"
                    for seed, q_lens, k_lens in shapes:
                        try:
                            metrics = _time_triton_train_case(
                                q_lens,
                                k_lens,
                                d_q=d_q,
                                d_kv=d_kv,
                                H=H,
                                D_H=D_H,
                                dtype=dtype,
                                warmup=warmup,
                                iters=iters,
                            )
                        except Exception as exc:
                            status = f"SKIP {_brief_exception(exc)}"
                            break
                        combo_fwd_values.append(metrics["triton_fwd"])
                        combo_bwd_values.append(metrics["triton_fwd_bwd"])
                        if seed in flash_by_seed:
                            flash_metrics = flash_by_seed[seed]
                            fwd_ratios.append(
                                flash_metrics["flash_fwd"] / metrics["triton_fwd"]
                            )
                            bwd_ratios.append(
                                flash_metrics["flash_fwd_bwd"] / metrics["triton_fwd_bwd"]
                            )
                        torch.cuda.empty_cache()

                    if status != "OK":
                        print(
                            f"  mode={fwd_mode:<6s} "
                            f"bt_q_fwd={bt_q_fwd:>2d} "
                            f"bt_k_fwd={bt_k_fwd:>3d} {status}"
                        )
                        continue

                    result = {
                        "fwd_mode": fwd_mode,
                        "bt_q_fwd": bt_q_fwd,
                        "bt_k_fwd": bt_k_fwd,
                        "triton_fwd_mean": _mean(combo_fwd_values),
                        "triton_fwd_min": min(combo_fwd_values),
                        "triton_fwd_max": max(combo_fwd_values),
                        "triton_fwd_bwd_mean": _mean(combo_bwd_values),
                        "triton_fwd_bwd_min": min(combo_bwd_values),
                        "triton_fwd_bwd_max": max(combo_bwd_values),
                    }
                    if fwd_ratios:
                        result["flash_fwd_vs_triton_mean"] = _mean(fwd_ratios)
                    if bwd_ratios:
                        result["flash_fwd_bwd_vs_triton_mean"] = _mean(bwd_ratios)
                    results.append(result)
                    print(
                        f"  done mode={fwd_mode:<6s} "
                        f"bt_q_fwd={bt_q_fwd:>2d} "
                        f"bt_k_fwd={bt_k_fwd:>3d} "
                        f"triton fwd {_metric_stats(combo_fwd_values)}  "
                        f"fwd+bwd {_metric_stats(combo_bwd_values)}"
                    )
    finally:
        _override_runtime_constant("_BT_Q_FWD", old_values["_BT_Q_FWD"])
        _override_runtime_constant("_BT_K_FWD", old_values["_BT_K_FWD"])
        if old_fwd_mode is None:
            os.environ.pop("TRITON_TARGET_ATTN_FWD_MODE", None)
        else:
            os.environ["TRITON_TARGET_ATTN_FWD_MODE"] = old_fwd_mode

    results.sort(key=lambda item: item["triton_fwd_mean"])
    print("\n##### forward block sweep summary #####")
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        print(_format_forward_combo_result(result))


def _run_train_block_sweep(
    count: int,
    start_seed: int,
    n_seg: int,
    q_min: int,
    q_max: int,
    k_min: int,
    k_max: int,
    bt_q_bwd_values: list[int],
    bt_k_bwd_values: list[int],
    dx_modes: list[str],
    d_q: int = _DEFAULT_D_Q,
    d_kv: int = _DEFAULT_D_KV,
    H: int = _DEFAULT_H,
    D_H: int = _DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
    compare_flash: bool = False,
    ref_warmup: int = 1,
    ref_iters: int = 5,
) -> None:
    print(
        f"\n=== train block sweep | cases={count} "
        f"seeds={start_seed}..{start_seed + count - 1} dtype={dtype} "
        f"| d_q={d_q} d_kv={d_kv} H={H} D_H={D_H} ==="
    )
    print(
        f"  bt_q_bwd={bt_q_bwd_values} "
        f"bt_k_bwd={bt_k_bwd_values} dx_modes={dx_modes}"
    )

    shapes = []
    for i in range(count):
        seed = start_seed + i
        q_lens, k_lens = _make_many_shape_lens(
            n_seg=n_seg,
            q_min=q_min,
            q_max=q_max,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
        )
        shapes.append((seed, q_lens, k_lens))

    flash_by_seed = {}
    if compare_flash:
        print("\n  computing flash-attn explicit fwd+bwd baseline once per seed...")
        for seed, q_lens, k_lens in shapes:
            metrics = _time_flash_explicit_train_case(
                q_lens,
                k_lens,
                d_q=d_q,
                d_kv=d_kv,
                H=H,
                D_H=D_H,
                dtype=dtype,
                warmup=ref_warmup,
                iters=ref_iters,
            )
            if metrics:
                flash_by_seed[seed] = metrics["flash_fwd_bwd"]
            torch.cuda.empty_cache()
        if flash_by_seed:
            flash_values = list(flash_by_seed.values())
            print(f"  flash explicit fwd+bwd {_metric_stats(flash_values)}")
        else:
            print("  flash explicit fwd+bwd SKIP: flash-attn is not installed")

    old_values = {
        "_BT_Q_BWD": _BT_Q_BWD,
        "_BT_K_BWD": _BT_K_BWD,
    }
    old_dx_mode = os.environ.get("TRITON_TARGET_ATTN_DX_MODE")
    results = []
    try:
        for bt_q_bwd in bt_q_bwd_values:
            for bt_k_bwd in bt_k_bwd_values:
                for dx_mode in dx_modes:
                    _override_runtime_constant("_BT_Q_BWD", bt_q_bwd)
                    _override_runtime_constant("_BT_K_BWD", bt_k_bwd)
                    os.environ["TRITON_TARGET_ATTN_DX_MODE"] = dx_mode

                    combo_values = []
                    combo_fwd_values = []
                    ratios = []
                    status = "OK"
                    for seed, q_lens, k_lens in shapes:
                        try:
                            metrics = _time_triton_train_case(
                                q_lens,
                                k_lens,
                                d_q=d_q,
                                d_kv=d_kv,
                                H=H,
                                D_H=D_H,
                                dtype=dtype,
                                warmup=warmup,
                                iters=iters,
                            )
                        except Exception as exc:
                            status = f"SKIP {_brief_exception(exc)}"
                            break
                        combo_values.append(metrics["triton_fwd_bwd"])
                        combo_fwd_values.append(metrics["triton_fwd"])
                        if seed in flash_by_seed:
                            ratios.append(
                                flash_by_seed[seed] / metrics["triton_fwd_bwd"]
                            )
                        torch.cuda.empty_cache()

                    if status != "OK":
                        print(
                            f"  bt_q_bwd={bt_q_bwd:>2d} "
                            f"bt_k_bwd={bt_k_bwd:>2d} dx={dx_mode:<7s} {status}"
                        )
                        continue

                    result = {
                        "bt_q_bwd": bt_q_bwd,
                        "bt_k_bwd": bt_k_bwd,
                        "dx_mode": dx_mode,
                        "triton_fwd_mean": _mean(combo_fwd_values),
                        "triton_fwd_bwd_mean": _mean(combo_values),
                        "triton_fwd_bwd_min": min(combo_values),
                        "triton_fwd_bwd_max": max(combo_values),
                    }
                    if ratios:
                        result["flash_fwd_bwd_vs_triton_mean"] = _mean(ratios)
                    results.append(result)
                    print(
                        f"  done bt_q_bwd={bt_q_bwd:>2d} "
                        f"bt_k_bwd={bt_k_bwd:>2d} dx={dx_mode:<7s} "
                        f"triton fwd+bwd {_metric_stats(combo_values)}"
                    )
    finally:
        _override_runtime_constant("_BT_Q_BWD", old_values["_BT_Q_BWD"])
        _override_runtime_constant("_BT_K_BWD", old_values["_BT_K_BWD"])
        if old_dx_mode is None:
            os.environ.pop("TRITON_TARGET_ATTN_DX_MODE", None)
        else:
            os.environ["TRITON_TARGET_ATTN_DX_MODE"] = old_dx_mode

    results.sort(key=lambda item: item["triton_fwd_bwd_mean"])
    print("\n##### train block sweep summary #####")
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        print(_format_combo_result(result))


def _run_long_sweep(
    count: int,
    start_seed: int,
    n_seg: int,
    q_min: int,
    q_max: int,
    k_min: int,
    k_max: int,
    d_q: int = _DEFAULT_D_Q,
    d_kv: int = _DEFAULT_D_KV,
    H: int = _DEFAULT_H,
    D_H: int = _DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
    compare_flash: bool = False,
    compare_tf: bool = False,
    ref_warmup: int = 1,
    ref_iters: int = 5,
    tf_warmup: int = 5,
    tf_iters: int = 20,
):
    print(
        f"\n=== long many-shape sweep | cases={count} "
        f"seeds={start_seed}..{start_seed + count - 1} dtype={dtype} "
        f"| d_q={d_q} d_kv={d_kv} H={H} D_H={D_H} ==="
    )
    all_metrics = []
    for i in range(count):
        seed = start_seed + i
        q_lens, k_lens = _make_many_shape_lens(
            n_seg=n_seg,
            q_min=q_min,
            q_max=q_max,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
        )
        metrics = _run_benchmark(
            f"many-shape ragged seed={seed}",
            q_lens,
            k_lens,
            d_q=d_q,
            d_kv=d_kv,
            H=H,
            D_H=D_H,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            compare_ref=False,
            ref_warmup=ref_warmup,
            ref_iters=ref_iters,
            compare_flash=compare_flash,
            compare_tf=compare_tf,
            tf_warmup=tf_warmup,
            tf_iters=tf_iters,
        )
        metrics["seed"] = seed
        all_metrics.append(metrics)
        torch.cuda.empty_cache()

    print("\n##### long sweep summary #####")
    for key, label in [
        ("triton_fwd", "triton forward"),
        ("triton_fwd_bwd", "triton fwd+bwd"),
        ("flash_fwd", "flash-attn explicit fwd"),
        ("flash_fwd_bwd", "flash-attn explicit fwd+bwd"),
        ("tf_explicit_fwd", "tensorflow explicit fwd"),
        ("tf_explicit_fwd_bwd", "tensorflow explicit fwd+bwd"),
        ("tf_absorbed_fwd", "tensorflow absorbed fwd"),
        ("tf_absorbed_fwd_bwd", "tensorflow absorbed fwd+bwd"),
    ]:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            print(f"  {label:30s} {_metric_stats(values)}")

    for key, label in [
        ("flash_fwd_vs_triton", "flash/triton fwd"),
        ("flash_fwd_bwd_vs_triton", "flash/triton fwd+bwd"),
        ("tf_explicit_fwd_vs_triton", "tf explicit/triton fwd"),
        ("tf_explicit_fwd_bwd_vs_triton", "tf explicit/triton fwd+bwd"),
        ("tf_absorbed_fwd_vs_triton", "tf absorbed/triton fwd"),
        ("tf_absorbed_fwd_bwd_vs_triton", "tf absorbed/triton fwd+bwd"),
    ]:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            print(f"  {label:30s} {_ratio_stats(values)}")


def _run_named_benchmark_cases(
    title: str,
    cases: list[tuple[str, list[int], list[int]]],
    d_q: int = _DEFAULT_D_Q,
    d_kv: int = _DEFAULT_D_KV,
    H: int = _DEFAULT_H,
    D_H: int = _DEFAULT_D_H,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 50,
    compare_flash: bool = False,
    compare_tf: bool = False,
    ref_warmup: int = 1,
    ref_iters: int = 5,
    tf_warmup: int = 5,
    tf_iters: int = 20,
) -> None:
    print(
        f"\n=== {title} | cases={len(cases)} dtype={dtype} "
        f"| d_q={d_q} d_kv={d_kv} H={H} D_H={D_H} ==="
    )
    all_metrics = []
    for name, q_lens, k_lens in cases:
        metrics = _run_benchmark(
            name,
            q_lens,
            k_lens,
            d_q=d_q,
            d_kv=d_kv,
            H=H,
            D_H=D_H,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            compare_ref=False,
            ref_warmup=ref_warmup,
            ref_iters=ref_iters,
            compare_flash=compare_flash,
            compare_tf=compare_tf,
            tf_warmup=tf_warmup,
            tf_iters=tf_iters,
        )
        metrics["case_name"] = name
        all_metrics.append(metrics)
        torch.cuda.empty_cache()

    print(f"\n##### {title} summary #####")
    for metrics in all_metrics:
        parts = [
            f"triton fwd={metrics['triton_fwd']:.3f} ms",
            f"triton fwd+bwd={metrics['triton_fwd_bwd']:.3f} ms",
        ]
        if "flash_fwd" in metrics:
            parts.append(f"flash fwd={metrics['flash_fwd']:.3f} ms")
        if "flash_fwd_bwd" in metrics:
            parts.append(f"flash fwd+bwd={metrics['flash_fwd_bwd']:.3f} ms")
        print(f"  {metrics['case_name']}: " + "  ".join(parts))

    for key, label in [
        ("triton_fwd", "triton forward"),
        ("triton_fwd_bwd", "triton fwd+bwd"),
        ("flash_fwd", "flash-attn explicit fwd"),
        ("flash_fwd_bwd", "flash-attn explicit fwd+bwd"),
        ("tf_explicit_fwd", "tensorflow explicit fwd"),
        ("tf_explicit_fwd_bwd", "tensorflow explicit fwd+bwd"),
        ("tf_absorbed_fwd", "tensorflow absorbed fwd"),
        ("tf_absorbed_fwd_bwd", "tensorflow absorbed fwd+bwd"),
    ]:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            print(f"  {label:30s} {_metric_stats(values)}")

    for key, label in [
        ("flash_fwd_vs_triton", "flash/triton fwd"),
        ("flash_fwd_bwd_vs_triton", "flash/triton fwd+bwd"),
        ("tf_explicit_fwd_vs_triton", "tf explicit/triton fwd"),
        ("tf_explicit_fwd_bwd_vs_triton", "tf explicit/triton fwd+bwd"),
        ("tf_absorbed_fwd_vs_triton", "tf absorbed/triton fwd"),
        ("tf_absorbed_fwd_bwd_vs_triton", "tf absorbed/triton fwd+bwd"),
    ]:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            print(f"  {label:30s} {_ratio_stats(values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fp32", action="store_true")
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--d-q", type=int, default=_DEFAULT_D_Q)
    parser.add_argument("--d-kv", type=int, default=_DEFAULT_D_KV)
    parser.add_argument("--num-heads", type=int, default=_DEFAULT_H)
    parser.add_argument("--head-dim", type=int, default=_DEFAULT_D_H)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument(
        "--bench-flash",
        action="store_true",
        help="also benchmark explicit K/V + flash_attn_varlen_func if installed",
    )
    parser.add_argument(
        "--bench-flash-breakdown",
        action="store_true",
        help="break down explicit K/V + FlashAttention on the small real-ragged case",
    )
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--no-bench-ref", action="store_true")
    parser.add_argument("--bench-ref-iters", type=int, default=10)
    parser.add_argument("--bench-ref-warmup", type=int, default=3)
    parser.add_argument("--bench-breakdown", action="store_true")
    parser.add_argument(
        "--bench-dx-all-modes",
        action="store_true",
        help="in breakdown, time both dX implementations; may allocate a large partial buffer",
    )
    parser.add_argument(
        "--dx-mode",
        choices=("auto", "partial", "fused", "atomic"),
        default=None,
        help="override TRITON_TARGET_ATTN_DX_MODE for wrapper/breakdown dX path selection",
    )
    parser.add_argument(
        "--fwd-mode",
        choices=("direct", "alpha"),
        default=None,
        help="override TRITON_TARGET_ATTN_FWD_MODE",
    )
    parser.add_argument(
        "--bt-q-fwd",
        type=int,
        default=None,
        help="override Triton forward Q block size for tuning",
    )
    parser.add_argument(
        "--bt-k-fwd",
        type=int,
        default=None,
        help="override Triton forward K block size for tuning",
    )
    parser.add_argument(
        "--bt-q-bwd",
        type=int,
        default=None,
        help="override Triton backward Q block size for tuning",
    )
    parser.add_argument(
        "--bt-k-bwd",
        type=int,
        default=None,
        help="override Triton backward K block size for tuning",
    )
    parser.add_argument(
        "--dx-fused-min-k-chunks",
        type=int,
        default=None,
        help="override the auto threshold for choosing fused-heads dX",
    )
    parser.add_argument("--bench-tf", action="store_true")
    parser.add_argument("--bench-tf-iters", type=int, default=20)
    parser.add_argument("--bench-tf-warmup", type=int, default=5)
    parser.add_argument("--bench-long", action="store_true")
    parser.add_argument(
        "--bench-long-sweep",
        type=int,
        default=0,
        help="run this many long many-shape benchmark cases starting from --bench-long-seed",
    )
    parser.add_argument(
        "--bench-forward-block-sweep",
        type=int,
        default=0,
        help="run a long many-shape forward sweep over BT_Q_FWD/BT_K_FWD",
    )
    parser.add_argument(
        "--bench-train-block-sweep",
        type=int,
        default=0,
        help="run a long many-shape training sweep over backward block sizes",
    )
    parser.add_argument(
        "--sweep-bt-q-fwd",
        type=str,
        default="16,32",
        help="comma-separated BT_Q_FWD values for --bench-forward-block-sweep",
    )
    parser.add_argument(
        "--sweep-bt-k-fwd",
        type=str,
        default="16,32,64",
        help="comma-separated BT_K_FWD values for --bench-forward-block-sweep",
    )
    parser.add_argument(
        "--sweep-fwd-modes",
        type=str,
        default="alpha,direct",
        help="comma-separated forward modes for --bench-forward-block-sweep: direct,alpha",
    )
    parser.add_argument(
        "--sweep-bt-q-bwd",
        type=str,
        default="16,32",
        help="comma-separated BT_Q_BWD values for --bench-train-block-sweep",
    )
    parser.add_argument(
        "--sweep-bt-k-bwd",
        type=str,
        default="16,32,64",
        help="comma-separated BT_K_BWD values for --bench-train-block-sweep",
    )
    parser.add_argument(
        "--sweep-dx-modes",
        type=str,
        default="fused",
        help="comma-separated dx modes for --bench-train-block-sweep: auto,partial,fused,atomic",
    )
    parser.add_argument("--bench-long-breakdown", action="store_true")
    parser.add_argument(
        "--bench-long-flash-breakdown",
        action="store_true",
        help="break down explicit K/V + FlashAttention on the long many-shape case",
    )
    parser.add_argument("--bench-long-segments", type=int, default=1000)
    parser.add_argument("--bench-long-q-min", type=int, default=1)
    parser.add_argument("--bench-long-q-max", type=int, default=16)
    parser.add_argument("--bench-long-k-min", type=int, default=64)
    parser.add_argument("--bench-long-k-max", type=int, default=2048)
    parser.add_argument("--bench-long-seed", type=int, default=17)
    parser.add_argument(
        "--bench-shape-suite",
        action="store_true",
        help="run several synthetic ragged benchmark distributions",
    )
    parser.add_argument(
        "--bench-lens-file",
        type=str,
        default=None,
        help="JSON/JSONL file with benchmark cases or online records containing q_lens/k_lens",
    )
    parser.add_argument(
        "--bench-lens-offset",
        type=int,
        default=0,
        help="skip this many cases/records from --bench-lens-file",
    )
    parser.add_argument(
        "--bench-lens-limit",
        type=int,
        default=None,
        help="only run this many cases/records from --bench-lens-file",
    )
    parser.add_argument(
        "--bench-long-tf",
        action="store_true",
        help="also run TensorFlow baselines on the long many-shape case; this may compile slowly",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="write stdout/stderr, including TensorFlow C++ logs, to this file",
    )
    args = parser.parse_args()
    if args.fwd_mode is not None:
        os.environ["TRITON_TARGET_ATTN_FWD_MODE"] = args.fwd_mode
    if args.dx_mode is not None:
        os.environ["TRITON_TARGET_ATTN_DX_MODE"] = args.dx_mode
    _apply_runtime_tuning(args)
    _redirect_process_output(args.log_file)

    assert torch.cuda.is_available(), "needs CUDA"

    cases = [
        # (name, q_lens, k_lens, notes)
        ("smoke",             [4],           [64]),               # min size, Q all in tail
        ("single segment",    [7],           [256]),              # 1 seg, K整除
        ("4 uniform",         [8, 8, 8, 8],  [256]*4),            # 段间隔离
        ("4 ragged (real)",   [7, 8, 6, 9],  [2048,1024,1536,512]),  # 真实业务尺寸
        ("k-tail nontrivial", [5, 7, 9],     [123, 257, 999]),    # K 都不整除 BT_K → 测 K-mask
        ("q-tail nontrivial", [33, 65, 129], [256, 256, 256]),    # Q 都不整除 BT_Q → 测 Q-mask
        ("short-k ragged",    [1, 2, 4, 8],  [1, 3, 7, 15]),      # very short K, softmax edge
        ("mixed block tails", [15, 16, 17, 31], [31, 32, 33, 65]), # Q/K cross block boundaries
        ("Nq=1 mini",         [1],           [512]),              # 极端小 Q
    ]

    fp32_pass = True
    if not args.skip_fp32:
        print("\n##### fp32 path #####")
        for name, ql, kl in cases:
            fp32_pass &= _run_case(
                name, ql, kl,
                d_q=args.d_q, d_kv=args.d_kv, H=args.num_heads, D_H=args.head_dim,
                dtype=torch.float32,
                atol=1e-4, rtol=1e-4,
            )

    bf16_cases = [
        ("smoke",             [4],           [64]),
        ("4 ragged (real)",   [7, 8, 6, 9],  [2048,1024,1536,512]),
        ("k-tail nontrivial", [5, 7, 9],     [123, 257, 999]),
        ("q-tail nontrivial", [33, 65, 129], [256, 256, 256]),
        ("short-k ragged",    [1, 2, 4, 8],  [1, 3, 7, 15]),
        ("mixed block tails", [15, 16, 17, 31], [31, 32, 33, 65]),
        ("Nq=1 mini",         [1],           [512]),
    ]

    bf16_pass = True
    if not args.skip_bf16:
        print("\n##### bf16 path #####")
        for name, ql, kl in bf16_cases:
            bf16_pass &= _run_case(
                name, ql, kl,
                d_q=args.d_q, d_kv=args.d_kv, H=args.num_heads, D_H=args.head_dim,
                dtype=torch.bfloat16,
                atol=5e-2, rtol=5e-2,
            )

    if args.bench:
        print("\n##### benchmark #####")
        _run_benchmark(
            "real ragged",
            [7, 8, 6, 9],
            [2048, 1024, 1536, 512],
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            compare_ref=not args.no_bench_ref,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
            compare_flash=args.bench_flash,
            compare_tf=args.bench_tf,
            tf_warmup=args.bench_tf_warmup,
            tf_iters=args.bench_tf_iters,
        )

    long_q_lens = None
    long_k_lens = None
    if (
        args.bench_long
        or args.bench_long_breakdown
        or args.bench_long_flash_breakdown
        or args.bench_long_tf
    ):
        long_q_lens, long_k_lens = _make_many_shape_lens(
            n_seg=args.bench_long_segments,
            q_min=args.bench_long_q_min,
            q_max=args.bench_long_q_max,
            k_min=args.bench_long_k_min,
            k_max=args.bench_long_k_max,
            seed=args.bench_long_seed,
        )

    if args.bench_long:
        print("\n##### benchmark long many-shape #####")
        _run_benchmark(
            f"many-shape ragged seed={args.bench_long_seed}",
            long_q_lens,
            long_k_lens,
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            compare_ref=not args.no_bench_ref,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
            compare_flash=args.bench_flash,
            compare_tf=args.bench_long_tf,
            tf_warmup=args.bench_tf_warmup,
            tf_iters=args.bench_tf_iters,
        )

    if args.bench_long_sweep > 0:
        print("\n##### benchmark long many-shape sweep #####")
        _run_long_sweep(
            count=args.bench_long_sweep,
            start_seed=args.bench_long_seed,
            n_seg=args.bench_long_segments,
            q_min=args.bench_long_q_min,
            q_max=args.bench_long_q_max,
            k_min=args.bench_long_k_min,
            k_max=args.bench_long_k_max,
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            compare_flash=args.bench_flash,
            compare_tf=args.bench_long_tf,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
            tf_warmup=args.bench_tf_warmup,
            tf_iters=args.bench_tf_iters,
        )

    if args.bench_forward_block_sweep > 0:
        print("\n##### benchmark forward block sweep #####")
        _run_forward_block_sweep(
            count=args.bench_forward_block_sweep,
            start_seed=args.bench_long_seed,
            n_seg=args.bench_long_segments,
            q_min=args.bench_long_q_min,
            q_max=args.bench_long_q_max,
            k_min=args.bench_long_k_min,
            k_max=args.bench_long_k_max,
            bt_q_fwd_values=_parse_int_list(args.sweep_bt_q_fwd, "--sweep-bt-q-fwd"),
            bt_k_fwd_values=_parse_int_list(args.sweep_bt_k_fwd, "--sweep-bt-k-fwd"),
            fwd_modes=_parse_fwd_mode_list(args.sweep_fwd_modes),
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            compare_flash=args.bench_flash,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
        )

    if args.bench_train_block_sweep > 0:
        print("\n##### benchmark train block sweep #####")
        _run_train_block_sweep(
            count=args.bench_train_block_sweep,
            start_seed=args.bench_long_seed,
            n_seg=args.bench_long_segments,
            q_min=args.bench_long_q_min,
            q_max=args.bench_long_q_max,
            k_min=args.bench_long_k_min,
            k_max=args.bench_long_k_max,
            bt_q_bwd_values=_parse_int_list(args.sweep_bt_q_bwd, "--sweep-bt-q-bwd"),
            bt_k_bwd_values=_parse_int_list(args.sweep_bt_k_bwd, "--sweep-bt-k-bwd"),
            dx_modes=_parse_dx_mode_list(args.sweep_dx_modes),
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            compare_flash=args.bench_flash,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
        )

    if args.bench_shape_suite:
        print("\n##### benchmark shape suite #####")
        _run_named_benchmark_cases(
            "benchmark shape suite",
            _make_shape_suite(args.bench_long_seed),
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            compare_flash=args.bench_flash,
            compare_tf=args.bench_long_tf,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
            tf_warmup=args.bench_tf_warmup,
            tf_iters=args.bench_tf_iters,
        )

    if args.bench_lens_file is not None:
        print("\n##### benchmark lens file #####")
        _run_named_benchmark_cases(
            f"benchmark lens file {args.bench_lens_file}",
            _load_lens_cases(
                args.bench_lens_file,
                offset=args.bench_lens_offset,
                limit=args.bench_lens_limit,
            ),
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            compare_flash=args.bench_flash,
            compare_tf=args.bench_long_tf,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
            tf_warmup=args.bench_tf_warmup,
            tf_iters=args.bench_tf_iters,
        )

    if args.bench_breakdown:
        print("\n##### benchmark breakdown #####")
        _run_benchmark_breakdown(
            "real ragged",
            [7, 8, 6, 9],
            [2048, 1024, 1536, 512],
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            bench_dx_all_modes=args.bench_dx_all_modes,
        )

    if args.bench_flash_breakdown:
        print("\n##### flash benchmark breakdown #####")
        _run_flash_breakdown(
            "real ragged",
            [7, 8, 6, 9],
            [2048, 1024, 1536, 512],
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
        )

    if args.bench_long_breakdown:
        print("\n##### benchmark long many-shape breakdown #####")
        _run_benchmark_breakdown(
            f"many-shape ragged seed={args.bench_long_seed}",
            long_q_lens,
            long_k_lens,
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            bench_dx_all_modes=args.bench_dx_all_modes,
        )

    if args.bench_long_flash_breakdown:
        print("\n##### flash benchmark long many-shape breakdown #####")
        _run_flash_breakdown(
            f"many-shape ragged seed={args.bench_long_seed}",
            long_q_lens,
            long_k_lens,
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            ref_warmup=args.bench_ref_warmup,
            ref_iters=args.bench_ref_iters,
        )

    if args.bench_tf and not args.bench:
        print("\n##### tensorflow benchmark #####")
        _run_tf_benchmark(
            [7, 8, 6, 9],
            [2048, 1024, 1536, 512],
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_tf_warmup,
            iters=args.bench_tf_iters,
        )

    if args.bench_long_tf and not args.bench_long and args.bench_long_sweep <= 0:
        print("\n##### tensorflow benchmark long many-shape #####")
        _run_tf_benchmark(
            long_q_lens,
            long_k_lens,
            d_q=args.d_q,
            d_kv=args.d_kv,
            H=args.num_heads,
            D_H=args.head_dim,
            dtype=torch.bfloat16,
            warmup=args.bench_tf_warmup,
            iters=args.bench_tf_iters,
        )

    print("\n##### summary #####")
    if not args.skip_fp32:
        print(f"fp32: {'PASS' if fp32_pass else 'FAIL'}")
    if not args.skip_bf16:
        print(f"bf16: {'PASS' if bf16_pass else 'FAIL'}")
