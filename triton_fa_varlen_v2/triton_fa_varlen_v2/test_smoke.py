"""
Smoke test for triton_fa_varlen_v2.

Two layers of correctness:
  1) Tiny / medium cases: PyTorch fp32 BlockDiagonal attention as oracle,
     compares fwd output and dq/dk/dv against the Triton kernel.
  2) Business-scale case (200w K/V tokens): same fp32 oracle, but computed
     segment-by-segment so peak memory stays bounded by the largest single
     segment (a global graph over 200w K tokens would OOM).

Oracle implementation: explicit matmul + softmax in fp32 with tf32 disabled
(NOT F.scaled_dot_product_attention — that auto-selects backend and we want
a fully deterministic ground truth).

This is intentionally minimal — the user will write their own benchmark/full
test suite. Goal here: catch obvious correctness regressions during dev.

Run on a CUDA box:
    python -m triton_fa_varlen_v2.test_smoke
"""

import math
import time
import torch

from triton_fa_varlen_v2.interface import packed_varlen_attn


def reference_block_diagonal_attn(q, k, v, q_seqinfo, k_seqinfo, scale):
    """
    Pure-PyTorch BlockDiagonal attention reference.

    Inputs:
        q: [T_q, H, D]  any float dtype
        k: [T_k, H, D]
        v: [T_k, H, D]
        q_seqinfo / k_seqinfo: [n_seg] python ints or 1-D tensor
        scale: float

    Returns: o [T_q, H, D] same dtype as q.
    """
    if isinstance(q_seqinfo, torch.Tensor):
        q_seqinfo = q_seqinfo.tolist()
    if isinstance(k_seqinfo, torch.Tensor):
        k_seqinfo = k_seqinfo.tolist()
    assert len(q_seqinfo) == len(k_seqinfo)

    out = torch.empty_like(q)
    q_off = 0
    k_off = 0
    for Lq, Lk in zip(q_seqinfo, k_seqinfo):
        # cast to fp32 for the reference
        qi = q[q_off:q_off + Lq].float()      # [Lq, H, D]
        ki = k[k_off:k_off + Lk].float()      # [Lk, H, D]
        vi = v[k_off:k_off + Lk].float()      # [Lk, H, D]

        # per-head: [H, Lq, D] @ [H, D, Lk] -> [H, Lq, Lk]
        qi_h = qi.transpose(0, 1)             # [H, Lq, D]
        ki_h = ki.transpose(0, 1)             # [H, Lk, D]
        vi_h = vi.transpose(0, 1)             # [H, Lk, D]

        s = torch.matmul(qi_h, ki_h.transpose(-1, -2)) * scale     # [H, Lq, Lk]
        p = torch.softmax(s, dim=-1)                                # [H, Lq, Lk]
        oi_h = torch.matmul(p, vi_h)                                # [H, Lq, D]
        oi = oi_h.transpose(0, 1).contiguous()                      # [Lq, H, D]
        out[q_off:q_off + Lq] = oi.to(q.dtype)

        q_off += Lq
        k_off += Lk
    return out


def reference_block_diagonal_attn_segmented(
    q, k, v, do, q_seqinfo, k_seqinfo, scale,
):
    """
    Per-segment fp32 reference with separate per-segment autograd graphs.

    Uses **explicit matmul + softmax** (not F.scaled_dot_product_attention)
    so the oracle is deterministic regardless of PyTorch SDPA backend
    selection. We also disable tf32 for matmul during the reference to
    keep accumulators in true fp32.

    Returns (o, dq, dk, dv) — all in q.dtype, same shape as the corresponding
    inputs. The inputs are NOT required to require_grad: this function builds
    its own throwaway leaf tensors per segment so the autograd graph is
    bounded to a single segment's working set and gets released before
    moving to the next.

    Why segmented:
      Building a global graph over 200w K tokens would materialize per-segment
      attn matrices simultaneously and OOM. Per-segment + manual graph drop
      keeps peak memory bounded by the largest single segment.

    Inputs (no grad needed):
        q:  [T_q, H, D]  bf16/fp16/fp32
        k:  [T_k, H, D]
        v:  [T_k, H, D]
        do: [T_q, H, D]  upstream gradient
    """
    if isinstance(q_seqinfo, torch.Tensor):
        q_seqinfo = q_seqinfo.tolist()
    if isinstance(k_seqinfo, torch.Tensor):
        k_seqinfo = k_seqinfo.tolist()
    assert len(q_seqinfo) == len(k_seqinfo)

    out = torch.empty_like(q)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Force true fp32 accumulators inside this oracle so it's a stable
    # ground truth regardless of the global tf32 setting.
    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        q_off = 0
        k_off = 0
        for seg_idx, (Lq, Lk) in enumerate(zip(q_seqinfo, k_seqinfo)):
            if Lq == 0 or Lk == 0:
                q_off += Lq; k_off += Lk
                continue

            # fresh fp32 leaves per segment so autograd graph is local
            qi = q[q_off:q_off + Lq].detach().float().requires_grad_(True)
            ki = k[k_off:k_off + Lk].detach().float().requires_grad_(True)
            vi = v[k_off:k_off + Lk].detach().float().requires_grad_(True)

            # per-head: [H, Lq, D] x [H, Lk, D]
            qi_h = qi.transpose(0, 1).contiguous()        # [H, Lq, D]
            ki_h = ki.transpose(0, 1).contiguous()        # [H, Lk, D]
            vi_h = vi.transpose(0, 1).contiguous()        # [H, Lk, D]

            # explicit matmul + softmax (deterministic oracle)
            s = torch.matmul(qi_h, ki_h.transpose(-1, -2)) * scale  # [H, Lq, Lk]
            p = torch.softmax(s, dim=-1)                            # [H, Lq, Lk]
            oi_h = torch.matmul(p, vi_h)                            # [H, Lq, D]
            oi = oi_h.transpose(0, 1).contiguous()                  # [Lq, H, D]

            # backward: this segment's chunk of do
            doi = do[q_off:q_off + Lq].detach().float()
            oi.backward(doi)

            # write back, dtype matches the original q/k/v
            out[q_off:q_off + Lq] = oi.detach().to(q.dtype)
            dq[q_off:q_off + Lq] = qi.grad.detach().to(q.dtype)
            dk[k_off:k_off + Lk] = ki.grad.detach().to(k.dtype)
            dv[k_off:k_off + Lk] = vi.grad.detach().to(v.dtype)

            # explicit drop (segment graph also goes out of scope here)
            del qi, ki, vi, qi_h, ki_h, vi_h, s, p, oi_h, oi, doi
            if seg_idx % 16 == 0:
                torch.cuda.empty_cache()

            q_off += Lq
            k_off += Lk
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn

    return out, dq, dk, dv


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _rel_err(a, b):
    diff = (a.float() - b.float()).abs()
    base = b.float().abs().clamp_min(1e-3)
    return (diff / base).max().item()


def _run_one_case(name, q_lens, k_lens, H, D, dtype, tol_o, tol_grad):
    """Run a single shape case and assert correctness."""
    device = "cuda"
    q_seqinfo = torch.tensor(q_lens, dtype=torch.int32, device=device)
    k_seqinfo = torch.tensor(k_lens, dtype=torch.int32, device=device)
    T_q = int(q_seqinfo.sum().item())
    T_k = int(k_seqinfo.sum().item())

    q = torch.randn(T_q, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)

    scale = 1.0 / math.sqrt(D)

    o_t = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
    do = torch.randn_like(o_t)
    o_t.backward(do)
    dq_t, q.grad = q.grad.clone(), None
    dk_t, k.grad = k.grad.clone(), None
    dv_t, v.grad = v.grad.clone(), None

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    o_ref = reference_block_diagonal_attn(q_ref, k_ref, v_ref, q_seqinfo, k_seqinfo, scale)
    o_ref.backward(do)
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    eo = _max_abs(o_t, o_ref)
    edq = _max_abs(dq_t, dq_ref)
    edk = _max_abs(dk_t, dk_ref)
    edv = _max_abs(dv_t, dv_ref)
    print(f"[{name}] q={q_lens} k={k_lens}  T_q={T_q} T_k={T_k}")
    print(f"   o:{eo:.3e}  dq:{edq:.3e}  dk:{edk:.3e}  dv:{edv:.3e}")
    assert eo < tol_o,    f"[{name}] o mismatch: {eo:.3e} >= {tol_o:.0e}"
    assert edq < tol_grad, f"[{name}] dq mismatch: {edq:.3e} >= {tol_grad:.0e}"
    assert edk < tol_grad, f"[{name}] dk mismatch: {edk:.3e} >= {tol_grad:.0e}"
    assert edv < tol_grad, f"[{name}] dv mismatch: {edv:.3e} >= {tol_grad:.0e}"


def smoke_test():
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping smoke test.")
        return

    torch.manual_seed(0)
    H, D = 4, 64
    dtype = torch.bfloat16

    # Tolerances: bf16 elementwise quantum is ~1/128 ≈ 7.8e-3.
    # Accumulating across L K-positions roughly scales the abs error like sqrt(L)
    # under random inputs, but with normalized softmax weights the practical
    # ceiling stays in the low-1e-1 range up to L ~ a few thousand.
    # Output (post-softmax convex combo): tighter.
    # Gradients (sum across L K positions): looser.
    TOL_O_SMALL = 5e-2
    TOL_O_LARGE = 1e-1
    TOL_GRAD_SMALL = 5e-2
    TOL_GRAD_LARGE = 5e-1   # 800 K-positions -> grads of magnitude up to ~10 each

    # ---- Case 1: tiny boundary case (single block per segment) ----
    # Verifies boundary masking + cross-attn (decoupled cu_seqlens_q/k).
    _run_one_case(
        "tiny",
        q_lens=[7, 11], k_lens=[19, 5],
        H=H, D=D, dtype=dtype,
        tol_o=TOL_O_SMALL, tol_grad=TOL_GRAD_SMALL,
    )

    # ---- Case 2: cross-block ----
    # Q crosses BT=128 boundary; K inner loop iterates multiple times.
    # seg0: Q=129 (2 Q-blocks), K=33 (2 K-iters, last partial)
    # seg1: Q=257 (3 Q-blocks), K=130 (5 K-iters, last partial)
    _run_one_case(
        "cross_block",
        q_lens=[129, 257], k_lens=[33, 130],
        H=H, D=D, dtype=dtype,
        tol_o=TOL_O_SMALL, tol_grad=TOL_GRAD_SMALL,
    )

    # ---- Case 3: Stage1-style (uniform Q, highly imbalanced K) ----
    # Mirrors the business case where Q is uniform 500 per user but K
    # ranges from a handful to tens of thousands.
    # seg0: Q=500 (4 Q-blocks), K=19 (1 K-iter, all boundary)
    # seg1: Q=500 (4 Q-blocks), K=800 (25 K-iters)
    _run_one_case(
        "stage1_like",
        q_lens=[500, 500], k_lens=[19, 800],
        H=H, D=D, dtype=dtype,
        tol_o=TOL_O_LARGE, tol_grad=TOL_GRAD_LARGE,
    )

    # ---- Case 4: Stage2-style (both sides uniform, balanced) ----
    # seg0/1: Q=512 (4 blocks), K=500 (16 K-iters)
    _run_one_case(
        "stage2_like",
        q_lens=[512, 512], k_lens=[500, 500],
        H=H, D=D, dtype=dtype,
        tol_o=TOL_O_LARGE, tol_grad=TOL_GRAD_LARGE,
    )

    # ---- Case 5: single segment ----
    # n_seg=1: degenerate cu_seqlens path. Verifies the loop / chunk_indices
    # don't have an off-by-one when there's only one segment.
    _run_one_case(
        "single_seg",
        q_lens=[400], k_lens=[400],
        H=H, D=D, dtype=dtype,
        tol_o=TOL_O_LARGE, tol_grad=TOL_GRAD_LARGE,
    )

    # ---- Case 6: segments aligned to block sizes ----
    # All segment lengths divisible by BT=128 / BS=32 -> no partial blocks,
    # no boundary masking active. Catches the case where boundary_check
    # would otherwise hide a bug.
    _run_one_case(
        "block_aligned",
        q_lens=[256, 256], k_lens=[128, 128],
        H=H, D=D, dtype=dtype,
        tol_o=TOL_O_SMALL, tol_grad=TOL_GRAD_SMALL,
    )

    # ---- no-grad path ----
    # Verify the kernel runs under torch.no_grad() and produces the same
    # output as the grad-enabled path. (Note: the autograd Function's forward
    # always calls save_for_backward, but under no_grad the saved tensors
    # are unused — this test catches output divergence between the two
    # call paths, not graph state.)
    _check_no_grad_path(H=H, D=D, dtype=dtype)

    # ---- partial-grad path ----
    # Verify when only some inputs need grad (e.g. K is frozen embedding).
    # Compare against the deterministic fp32 reference to catch wrong
    # gradient values (not just NaN/Inf).
    _check_partial_grad_path(H=H, D=D, dtype=dtype)

    # ---- 4D layout ([1, T, H, D]) ----
    # The business call site passes 4-D; verify the wrapper's squeeze /
    # unsqueeze round-trip keeps shapes correct and backward flows through.
    _check_4d_layout(H=H, D=D, dtype=dtype)

    # ---- head_dim=128 ----
    # Kernel asserts D in {16,32,64,128,256}; verify a non-default D actually
    # compiles and runs correctly. Business uses 64 today but the kernel
    # contract says any of those should work.
    _run_one_case(
        "head_dim_128",
        q_lens=[200, 300], k_lens=[150, 250],
        H=H, D=128, dtype=dtype,
        tol_o=TOL_O_SMALL, tol_grad=TOL_GRAD_SMALL,
    )

    # ---- multi-seed sanity sweep ----
    # Same shape, 5 different seeds. Catches input-distribution-sensitive
    # bugs (e.g. softmax overflow at extreme magnitudes).
    _check_multi_seed(H=H, D=D, dtype=dtype)

    # ---- repeatability ----
    # Identical inputs -> identical outputs, bit-for-bit. Catches accidental
    # non-determinism (atomics, async race, uninit reads).
    _check_repeatability(H=H, D=D, dtype=dtype)

    # ---- business-scale self-consistency ----
    # ~200w K/V tokens (Q only ~25k); fp32 reference computed segment by
    # segment so peak memory is bounded by the largest single segment.
    _check_business_scale(H=H, D=D, dtype=dtype)

    # ---- business-scale variants ----
    # Same K_total=200w, different segment-count regimes:
    #   n_seg=10:  ~200k K per seg (mega-context)
    #   n_seg=200: ~10k K per seg  (long-tail / many small users)
    # Plus pareto-distributed K (real-world long-tail).
    _check_business_variants(H=H, D=D, dtype=dtype)

    print("smoke test PASS")


def _check_no_grad_path(H, D, dtype):
    device = "cuda"
    q_seqinfo = torch.tensor([300, 700], dtype=torch.int32, device=device)
    k_seqinfo = torch.tensor([200, 800], dtype=torch.int32, device=device)
    T_q = int(q_seqinfo.sum().item())
    T_k = int(k_seqinfo.sum().item())
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(T_q, H, D, device=device, dtype=dtype)
    k = torch.randn(T_k, H, D, device=device, dtype=dtype)
    v = torch.randn(T_k, H, D, device=device, dtype=dtype)

    # grad-enabled run
    qr, kr, vr = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    o_grad = packed_varlen_attn(qr, kr, vr, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)

    # no-grad run
    with torch.no_grad():
        o_nograd = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)

    diff = _max_abs(o_grad, o_nograd)
    print(f"[no_grad] T_q={T_q} T_k={T_k}  o_diff={diff:.3e} (expect 0)")
    assert diff == 0.0, f"no-grad path produced different output: {diff}"


def _check_partial_grad_path(H, D, dtype):
    """K and V are frozen (e.g. retrieval embeddings); only Q needs grad."""
    device = "cuda"
    q_seqinfo = torch.tensor([300, 200], dtype=torch.int32, device=device)
    k_seqinfo = torch.tensor([100, 400], dtype=torch.int32, device=device)
    T_q = int(q_seqinfo.sum().item())
    T_k = int(k_seqinfo.sum().item())
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(T_q, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(T_k, H, D, device=device, dtype=dtype)  # no grad
    v = torch.randn(T_k, H, D, device=device, dtype=dtype)  # no grad

    o = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
    do = torch.randn_like(o)
    o.backward(do)

    assert q.grad is not None and torch.isfinite(q.grad).all(), "dq has NaN/Inf"
    assert k.grad is None and v.grad is None, "K/V should have no grad"

    # Compare q.grad against fp32 reference.
    _o_ref, dq_ref, _dk_ref, _dv_ref = reference_block_diagonal_attn_segmented(
        q.detach(), k.detach(), v.detach(), do.detach(),
        q_seqinfo, k_seqinfo, scale,
    )
    edq = _max_abs(q.grad, dq_ref)
    print(f"[partial_grad] T_q={T_q} T_k={T_k}  dq max_abs={edq:.3e} "
          f"(K/V no-grad confirmed)")
    assert edq < 5e-2, f"partial_grad dq mismatch: {edq:.3e}"


def _check_4d_layout(H, D, dtype):
    """
    Business call site passes [1, T, H, D]. Verify:
      - wrapper accepts 4D input
      - wrapper returns 4D output with same leading 1
      - backward flows back to the original 4D tensors
      - numeric result matches the 3D-input path
    """
    device = "cuda"
    q_seqinfo = torch.tensor([200, 300], dtype=torch.int32, device=device)
    k_seqinfo = torch.tensor([150, 250], dtype=torch.int32, device=device)
    T_q = int(q_seqinfo.sum().item())
    T_k = int(k_seqinfo.sum().item())
    scale = 1.0 / math.sqrt(D)

    # 4D path
    q4 = torch.randn(1, T_q, H, D, device=device, dtype=dtype, requires_grad=True)
    k4 = torch.randn(1, T_k, H, D, device=device, dtype=dtype, requires_grad=True)
    v4 = torch.randn(1, T_k, H, D, device=device, dtype=dtype, requires_grad=True)

    o4 = packed_varlen_attn(q4, k4, v4, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
    assert o4.shape == (1, T_q, H, D), f"4D output shape wrong: {tuple(o4.shape)}"

    do4 = torch.randn_like(o4)
    o4.backward(do4)
    assert q4.grad is not None and q4.grad.shape == q4.shape
    assert k4.grad is not None and k4.grad.shape == k4.shape
    assert v4.grad is not None and v4.grad.shape == v4.shape

    # 3D path with the same underlying tensor data
    q3 = q4.detach().squeeze(0).clone().requires_grad_(True)
    k3 = k4.detach().squeeze(0).clone().requires_grad_(True)
    v3 = v4.detach().squeeze(0).clone().requires_grad_(True)
    o3 = packed_varlen_attn(q3, k3, v3, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
    assert o3.shape == (T_q, H, D), f"3D output shape wrong: {tuple(o3.shape)}"
    o3.backward(do4.squeeze(0))

    eo = _max_abs(o4.squeeze(0), o3)
    edq = _max_abs(q4.grad.squeeze(0), q3.grad)
    edk = _max_abs(k4.grad.squeeze(0), k3.grad)
    edv = _max_abs(v4.grad.squeeze(0), v3.grad)
    print(f"[4d_layout] T_q={T_q} T_k={T_k}  shape OK, "
          f"4d-vs-3d  o={eo:.3e} dq={edq:.3e} dk={edk:.3e} dv={edv:.3e}")
    assert eo == 0.0 and edq == 0.0 and edk == 0.0 and edv == 0.0, \
        "4D and 3D paths should be bit-identical"


def _check_multi_seed(H, D, dtype):
    """Same shape, multiple seeds. Catches input-distribution-sensitive bugs."""
    device = "cuda"
    q_seqinfo = torch.tensor([300, 400], dtype=torch.int32, device=device)
    k_seqinfo = torch.tensor([800, 600], dtype=torch.int32, device=device)
    T_q = int(q_seqinfo.sum().item())
    T_k = int(k_seqinfo.sum().item())
    scale = 1.0 / math.sqrt(D)

    worst_o = 0.0
    worst_grad = 0.0
    n_seeds = 5
    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        q = torch.randn(T_q, H, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)
        o = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
        do = torch.randn_like(o)
        o.backward(do)

        o_ref, dq_ref, dk_ref, dv_ref = reference_block_diagonal_attn_segmented(
            q.detach(), k.detach(), v.detach(), do.detach(),
            q_seqinfo, k_seqinfo, scale,
        )
        worst_o = max(worst_o, _max_abs(o, o_ref))
        worst_grad = max(
            worst_grad,
            _max_abs(q.grad, dq_ref),
            _max_abs(k.grad, dk_ref),
            _max_abs(v.grad, dv_ref),
        )

    print(f"[multi_seed] {n_seeds} seeds, T_q={T_q} T_k={T_k}  "
          f"worst o={worst_o:.3e} worst grad={worst_grad:.3e}")
    assert worst_o < 5e-2, f"multi-seed worst o err {worst_o}"
    assert worst_grad < 1e-1, f"multi-seed worst grad err {worst_grad}"


def _check_repeatability(H, D, dtype):
    """Identical inputs -> identical outputs, bit-for-bit (no atomics, no race)."""
    device = "cuda"
    q_seqinfo = torch.tensor([500, 500], dtype=torch.int32, device=device)
    k_seqinfo = torch.tensor([400, 600], dtype=torch.int32, device=device)
    T_q = int(q_seqinfo.sum().item())
    T_k = int(k_seqinfo.sum().item())
    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(7)
    q_data = torch.randn(T_q, H, D, device=device, dtype=dtype)
    k_data = torch.randn(T_k, H, D, device=device, dtype=dtype)
    v_data = torch.randn(T_k, H, D, device=device, dtype=dtype)

    def _run():
        q = q_data.clone().requires_grad_(True)
        k = k_data.clone().requires_grad_(True)
        v = v_data.clone().requires_grad_(True)
        o = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
        # use a fixed dO so the bwd path is fully deterministic too
        do = torch.ones_like(o)
        o.backward(do)
        return o.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

    o1, dq1, dk1, dv1 = _run()
    o2, dq2, dk2, dv2 = _run()
    deltas = [
        ("o", _max_abs(o1, o2)),
        ("dq", _max_abs(dq1, dq2)),
        ("dk", _max_abs(dk1, dk2)),
        ("dv", _max_abs(dv1, dv2)),
    ]
    print(f"[repeatability] T_q={T_q} T_k={T_k}  " +
          "  ".join(f"{n}={d:.0e}" for n, d in deltas))
    for name, d in deltas:
        assert d == 0.0, f"non-deterministic {name}: {d}"


def _check_business_variants(H, D, dtype):
    """
    Business-scale sweep: multiple K_total scales x multiple seeds each.

    For each scale, we run N_SEEDS independent seeds. The first
    N_REFERENCE_SEEDS of those also run the fp32 segmented reference
    for full numeric correctness; the remainder only do finite + timing
    so we get enough samples to report mean/std without paying for
    50 full references.

    Distribution: n_seg=50, Q=500/seg, K drawn from a coarse pareto-ish
    long-tail to mimic real user-history shape.
    """
    device = "cuda"
    Q_PER_SEG = 500
    N_SEG = 50

    SCALES_K_TOTAL = [800_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000]
    N_SEEDS = 10
    N_REFERENCE_SEEDS = 3   # full oracle check on first N out of N_SEEDS
    scale = 1.0 / math.sqrt(D)

    free, _ = torch.cuda.mem_get_info()
    print(f"[business_sweep] free mem {free/1e9:.1f}GB; scales={SCALES_K_TOTAL}; "
          f"{N_SEEDS} seeds each ({N_REFERENCE_SEEDS} with fp32 oracle)")

    for K_TOTAL in SCALES_K_TOTAL:
        # crude memory check: triton needs (Q + 2*K) * grads, oracle needs
        # max_seg attn matrix in fp32. Skip if not enough headroom.
        triton_bytes = (Q_PER_SEG * N_SEG + 2 * K_TOTAL) * H * D * 2 * 2  # bf16, with grads
        # max segment under our distribution can hit ~30% of K_TOTAL on tail
        max_seg_attn = H * Q_PER_SEG * int(K_TOTAL * 0.3) * 4 * 3   # s, p, ds in fp32
        need = triton_bytes + max_seg_attn
        if need > free * 0.6:
            print(f"  K_TOTAL={K_TOTAL:>8}  SKIP (need {need/1e9:.1f}GB > 60% free)")
            continue

        # generate one K-length distribution; reuse for all seeds at this scale
        # so timing variance reflects kernel behavior, not segmentation noise.
        torch.manual_seed(K_TOTAL)
        raw = (torch.rand(N_SEG) ** -0.667 - 1.0).clamp_min(0.01)
        raw = raw / raw.sum() * K_TOTAL
        k_lens_t = raw.round().long()
        k_lens_t[-1] = K_TOTAL - k_lens_t[:-1].sum()
        k_lens = k_lens_t.tolist()
        q_lens = [Q_PER_SEG] * N_SEG

        q_seqinfo = torch.tensor(q_lens, dtype=torch.int32, device=device)
        k_seqinfo = torch.tensor(k_lens, dtype=torch.int32, device=device)
        T_q = sum(q_lens)
        T_k = sum(k_lens)
        max_k = max(k_lens)
        min_k = min(k_lens)

        fwd_times = []
        bwd_times = []
        worst_o, worst_grad = 0.0, 0.0

        # warm up once at this scale (fresh shape -> some autotune may run)
        q = torch.randn(T_q, H, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)
        _o = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
        do = torch.randn_like(_o)
        _o.backward(do)
        del q, k, v, _o, do
        torch.cuda.synchronize()

        for seed in range(N_SEEDS):
            torch.manual_seed(2000 + seed)
            q = torch.randn(T_q, H, D, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
            torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1000)

            do = torch.randn_like(o)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o.backward(do)
            torch.cuda.synchronize()
            bwd_times.append((time.perf_counter() - t0) * 1000)

            # finite check on every seed (cheap)
            for nm, t in [("o", o), ("dq", q.grad), ("dk", k.grad), ("dv", v.grad)]:
                assert torch.isfinite(t).all(), \
                    f"[business_sweep K={K_TOTAL} seed={seed}] {nm} has NaN/Inf"

            # full oracle only on the first few seeds
            if seed < N_REFERENCE_SEEDS:
                o_ref, dq_ref, dk_ref, dv_ref = reference_block_diagonal_attn_segmented(
                    q.detach(), k.detach(), v.detach(), do.detach(),
                    q_seqinfo, k_seqinfo, scale,
                )
                worst_o = max(worst_o, _max_abs(o, o_ref))
                worst_grad = max(
                    worst_grad,
                    _max_abs(q.grad, dq_ref),
                    _max_abs(k.grad, dk_ref),
                    _max_abs(v.grad, dv_ref),
                )
                del o_ref, dq_ref, dk_ref, dv_ref

            del q, k, v, o, do
            torch.cuda.empty_cache()

        # stats
        def _mean_std(xs):
            m = sum(xs) / len(xs)
            v = sum((x - m) ** 2 for x in xs) / len(xs)
            return m, v ** 0.5

        fwd_m, fwd_s = _mean_std(fwd_times)
        bwd_m, bwd_s = _mean_std(bwd_times)
        print(f"  K_TOTAL={K_TOTAL:>8}  T_k={T_k}  k_range=[{min_k},{max_k}]")
        print(f"     fwd {fwd_m:6.2f} +/- {fwd_s:5.2f} ms     "
              f"bwd {bwd_m:6.2f} +/- {bwd_s:5.2f} ms     "
              f"({len(fwd_times)} seeds)")
        print(f"     oracle (first {N_REFERENCE_SEEDS}): "
              f"worst o={worst_o:.3e} worst grad={worst_grad:.3e}")

        # tolerance scales with sqrt(max_k) (bf16 reduction noise)
        tol_o = max(2e-1, 5e-3 * (max_k / 1000) ** 0.5)
        tol_grad = max(2.0, 0.1 * (max_k / 1000) ** 0.5)
        assert worst_o < tol_o, f"[K={K_TOTAL}] o err {worst_o:.3e} >= {tol_o:.3e}"
        assert worst_grad < tol_grad, f"[K={K_TOTAL}] grad err {worst_grad:.3e} >= {tol_grad:.3e}"


def _check_business_scale(H, D, dtype):
    """
    Stage1-shaped business scale:
      Q = 50 segments * 500 tokens each = 25k Q tokens total
      K/V = 200w (2_000_000) tokens total, distributed unevenly across 50 segs
    Per-segment fp32 reference, autograd done segment-by-segment so each
    segment's graph is released before the next runs (a global graph would
    OOM on the 200w-K dimension).
    """
    device = "cuda"
    n_seg = 50
    Q_PER_SEG = 500
    K_TOTAL = 2_000_000

    torch.manual_seed(42)
    q_lens = [Q_PER_SEG] * n_seg

    # K lens: random distribution summing to exactly K_TOTAL.
    # Range mimics business: from a few hundred to ~80k per segment.
    raw = torch.rand(n_seg)
    raw = raw / raw.sum() * K_TOTAL
    k_lens_t = raw.round().long()
    k_lens_t[-1] = K_TOTAL - k_lens_t[:-1].sum()    # fix rounding drift
    k_lens = k_lens_t.tolist()

    q_seqinfo = torch.tensor(q_lens, dtype=torch.int32, device=device)
    k_seqinfo = torch.tensor(k_lens, dtype=torch.int32, device=device)
    T_q = sum(q_lens)
    T_k = sum(k_lens)
    scale = 1.0 / math.sqrt(D)

    print(f"[business] n_seg={n_seg}  T_q={T_q}  T_k={T_k}  K-len range "
          f"[{min(k_lens)}, {max(k_lens)}]")

    # Memory check: Triton path needs Q + K + V + O + dQ + dK + dV in bf16
    # plus LSE + delta in fp32. Reference per-segment needs another ~4x the
    # max single segment in fp32. The 200w K is the dominant cost.
    free, total = torch.cuda.mem_get_info()
    bf16_bytes_total = (T_q * 2 + 2 * T_k * 2) * H * D * 2  # qkv + their grads
    fp32_lse_bytes = (T_q * H * 4) + (T_q * H * 4)
    max_seg_fp32 = (Q_PER_SEG + 2 * max(k_lens)) * H * D * 4 * 4  # qkv + grads
    need = bf16_bytes_total + fp32_lse_bytes + max_seg_fp32
    print(f"   est mem: {need/1e9:.2f}GB  free: {free/1e9:.2f}GB")
    if need > free * 0.7:
        print(f"   skipping business-scale: would use >70% of free memory")
        return

    q = torch.randn(T_q, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(T_k, H, D, device=device, dtype=dtype, requires_grad=True)

    # warm up
    _ = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
    torch.cuda.synchronize()

    # ---- Triton path ----
    t0 = time.perf_counter()
    o_t = packed_varlen_attn(q, k, v, q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, scale=scale)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) * 1000

    do = torch.randn_like(o_t)

    t0 = time.perf_counter()
    o_t.backward(do)
    torch.cuda.synchronize()
    t_bwd = (time.perf_counter() - t0) * 1000

    dq_t = q.grad.detach().clone()
    dk_t = k.grad.detach().clone()
    dv_t = v.grad.detach().clone()
    q.grad = None; k.grad = None; v.grad = None
    print(f"   triton fwd {t_fwd:6.1f} ms   bwd {t_bwd:6.1f} ms")

    # finite-ness sanity (cheap, catches NaN before we burn time on the ref)
    for name, t in [("o", o_t), ("dq", dq_t), ("dk", dk_t), ("dv", dv_t)]:
        assert torch.isfinite(t).all(), f"[business] triton {name} has NaN/Inf"

    # ---- per-segment fp32 reference ----
    t0 = time.perf_counter()
    o_ref, dq_ref, dk_ref, dv_ref = reference_block_diagonal_attn_segmented(
        q.detach(), k.detach(), v.detach(), do.detach(),
        q_seqinfo, k_seqinfo, scale,
    )
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) * 1000
    print(f"   fp32 segmented reference: {t_ref:6.0f} ms")

    eo = _max_abs(o_t, o_ref)
    edq = _max_abs(dq_t, dq_ref)
    edk = _max_abs(dk_t, dk_ref)
    edv = _max_abs(dv_t, dv_ref)
    # bf16 with K reductions of length up to ~80k: per-element noise ~ sqrt(L) * 2^-7,
    # which for L=80k is roughly sqrt(80000) * 0.0078 ≈ 2.2. In practice softmax
    # weights drop most of that, but be generous on grads.
    print(f"   o:{eo:.3e}   dq:{edq:.3e}   dk:{edk:.3e}   dv:{edv:.3e}")
    TOL_O = 2e-1
    TOL_GRAD = 2.0
    assert eo < TOL_O, f"o mismatch {eo:.3e} >= {TOL_O}"
    assert edq < TOL_GRAD, f"dq mismatch {edq:.3e} >= {TOL_GRAD}"
    assert edk < TOL_GRAD, f"dk mismatch {edk:.3e} >= {TOL_GRAD}"
    assert edv < TOL_GRAD, f"dv mismatch {edv:.3e} >= {TOL_GRAD}"


if __name__ == "__main__":
    smoke_test()
