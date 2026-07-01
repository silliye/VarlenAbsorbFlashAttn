"""
Professional benchmark harness for packed-varlen FlashAttention.

This script compares the project Triton implementation against:
  - FlashAttention-2 varlen CUDA
  - PyTorch scaled_dot_product_attention run segment-by-segment
  - An explicit fp32 matmul + softmax oracle for correctness

The benchmark is intentionally focused on the production subset:
  bf16/fp16, non-causal, dropout=0, no bias/window/GQA, packed varlen
  BlockDiagonalMask semantics.

Default candidate interface matches triton_fa_varlen_v2.interface.packed_varlen_attn:
    fn(q, k, v, *, cu_seqlens_q, cu_seqlens_k, scale=None) -> output

Examples:
    python tools/benchmark_packed_varlen_flash_attn.py --profile smoke

    python tools/benchmark_packed_varlen_flash_attn.py \
        --profile business \
        --candidate-varlen triton_fa_varlen_v2.interface:packed_varlen_attn \
        --correctness-only

    python tools/benchmark_packed_varlen_flash_attn.py \
        --profile business-sweep --repeat 10 --modes forward fwd_bwd

    python tools/benchmark_packed_varlen_flash_attn.py \
        --profile business-sweep --speed-only --repeat 20

    python tools/benchmark_packed_varlen_flash_attn.py \
        --profile custom --speed-only --modes forward backward \
        --n-seg 50 --q-per-seg 500 --k-total 2000000 --k-dist pareto

    # FA2 as the direct oracle (instead of torch_math fp32):
    python tools/benchmark_packed_varlen_flash_attn.py \
        --profile business-sweep --correctness-only --oracle fa2_varlen

    # Single mega-segment (n_seg=1, T_k=2M):
    python tools/benchmark_packed_varlen_flash_attn.py \
        --profile single-mega --speed-only

    # head_dim=128 sweep:
    python tools/benchmark_packed_varlen_flash_attn.py \
        --profile head-dim-128 --speed-only

Reading correctness results when --input-dist != "normal":

    The default tolerance schedule (effective_tol_o ~ 0.05, effective_tol_grad
    ~ 0.1) is calibrated for randn() inputs. With "large" / "exp" / "small"
    the softmax becomes much sharper or much flatter, and bf16 vs fp32 deltas
    can hit ~0.1+ on O and ~1+ on grads even when the kernel is correct -
    that's bf16 noise, not a bug.

    To make this less of a footgun, when --input-dist != "normal" and
    --oracle is left at the default (torch_math) and the user did NOT pin
    --tol-o / --tol-grad, the harness auto-relaxes both tols to 100. This is
    "stability-only" mode: it still catches NaN / Inf / gross corruption but
    won't false-positive on bf16 rounding noise.

    For a real cross-kernel bug hunt at extreme input dist, switch to FA2 as
    the oracle (same precision, same noise distribution):

        --input-dist large --oracle fa2_varlen --impls candidate \
            --tol-o 1 --tol-grad 10
"""

import argparse
import contextlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


_SCRIPT_DIR = Path(__file__).resolve().parent
for _import_root in (_SCRIPT_DIR.parent, _SCRIPT_DIR.parent.parent):
    _import_root_s = str(_import_root)
    if _import_root_s not in sys.path:
        sys.path.insert(0, _import_root_s)


DEFAULT_CANDIDATE = "triton_fa_varlen_v2.interface:packed_varlen_attn"


@dataclass(frozen=True)
class BenchCase:
    name: str
    q_lens: Tuple[int, ...]
    k_lens: Tuple[int, ...]

    @property
    def n_seg(self) -> int:
        return len(self.q_lens)

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def total_k(self) -> int:
        return sum(self.k_lens)

    @property
    def max_q(self) -> int:
        return max(self.q_lens)

    @property
    def max_k(self) -> int:
        return max(self.k_lens)


@dataclass(frozen=True)
class InputPack:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    do: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor


@dataclass(frozen=True)
class RunOutput:
    o: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


@dataclass(frozen=True)
class ErrorReport:
    impl_name: str
    o_max: float
    dq_max: float
    dk_max: float
    dv_max: float
    o_mean: float
    dq_mean: float
    dk_mean: float
    dv_mean: float

    @property
    def grad_max(self) -> float:
        return max(self.dq_max, self.dk_max, self.dv_max)

    @property
    def grad_mean(self) -> float:
        return max(self.dq_mean, self.dk_mean, self.dv_mean)


@dataclass(frozen=True)
class TimingResult:
    case_name: str
    impl_name: str
    mode: str
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    fwd_tflops_equiv: Optional[float]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_callable(spec: Optional[str]) -> Optional[Callable]:
    if spec is None or spec.lower() in ("", "none", "off"):
        return None
    if ":" not in spec:
        raise ValueError("--candidate-varlen must use module:function format")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _load_flash_attn_varlen() -> Optional[Callable]:
    try:
        module = importlib.import_module("flash_attn")
    except Exception as exc:
        print(f"[warn] flash_attn import failed; FA2 baseline disabled: {exc}")
        return None
    return getattr(module, "flash_attn_varlen_func")


def _first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _prefix_sums(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    values = [0]
    total = 0
    for length in lengths:
        total += int(length)
        values.append(total)
    return torch.tensor(values, device=device, dtype=torch.int32)


def _make_input_pack(
    case: BenchCase,
    H: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    input_dist: str = "normal",
    contig: str = "default",
) -> InputPack:
    """
    Build a fresh input pack for a case.

    input_dist:
      "normal":   randn() — the default, well-conditioned softmax inputs
      "large":    randn() * 8 — pushes softmax pivot far from zero (every
                  attention head has a sharp peak at one or two K positions)
      "exp":      randn().exp() — long-tailed positive values; dot products
                  span many decades, exposes log2-domain accumulation
      "small":    randn() * 0.05 — softmax becomes nearly uniform, tests
                  numerical stability when no K position dominates

    contig:
      "default":   plain torch.randn — standard packed [T,H,D] layout
      "non_contig": allocate a larger tensor and slice so stride(0)/stride(1)
                   are not minimal. Forces the wrapper to .contiguous() in
                   _normalize_qkv.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def _alloc(T):
        if contig == "default":
            x = torch.randn(T, H, D, device=device, dtype=dtype)
        elif contig == "non_contig":
            # Allocate a wider tensor and slice; resulting view has stride(0)
            # = (H*2)*D not H*D, breaking the kernel's stride assumption and
            # forcing the wrapper's .contiguous() path.
            big = torch.randn(T, H * 2, D, device=device, dtype=dtype)
            x = big[:, :H, :]
            assert not x.is_contiguous()
        else:
            raise ValueError(f"Unknown contig mode: {contig}")

        if input_dist == "normal":
            return x
        if input_dist == "large":
            return x * 8.0
        if input_dist == "exp":
            return x.exp()
        if input_dist == "small":
            return x * 0.05
        raise ValueError(f"Unknown input_dist: {input_dist}")

    q = _alloc(case.total_q)
    k = _alloc(case.total_k)
    v = _alloc(case.total_k)
    do = torch.randn(case.total_q, H, D, device=device, dtype=dtype)
    cu_q = _prefix_sums(case.q_lens, device)
    cu_k = _prefix_sums(case.k_lens, device)
    return InputPack(q=q, k=k, v=v, do=do, cu_q=cu_q, cu_k=cu_k)


def _clear_grads(tensors: Iterable[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def _attention_fwd_flops(case: BenchCase, H: int, D: int) -> int:
    # qk matmul and pv matmul, each roughly 2 FLOPs per multiply-add.
    cells = sum(q_len * k_len for q_len, k_len in zip(case.q_lens, case.k_lens))
    return 4 * H * D * cells


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


@contextlib.contextmanager
def _tf32_disabled():
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def _torch_math_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    case: BenchCase,
    scale: float,
) -> torch.Tensor:
    """Explicit fp32 BlockDiagonal attention oracle."""
    out = torch.empty_like(q)
    q_off = 0
    k_off = 0
    for Lq, Lk in zip(case.q_lens, case.k_lens):
        qi = q[q_off:q_off + Lq].float()
        ki = k[k_off:k_off + Lk].float()
        vi = v[k_off:k_off + Lk].float()

        qi_h = qi.transpose(0, 1).contiguous()  # [H, Lq, D]
        ki_h = ki.transpose(0, 1).contiguous()  # [H, Lk, D]
        vi_h = vi.transpose(0, 1).contiguous()  # [H, Lk, D]

        scores = torch.matmul(qi_h, ki_h.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        oi_h = torch.matmul(probs, vi_h)
        out[q_off:q_off + Lq] = oi_h.transpose(0, 1).contiguous().to(q.dtype)

        q_off += Lq
        k_off += Lk
    return out


def _run_torch_math_oracle_segmented(
    pack: InputPack,
    case: BenchCase,
    scale: float,
) -> RunOutput:
    """
    Explicit fp32 oracle with one autograd graph per segment.

    This keeps business-scale correctness bounded by the largest single
    segment instead of materializing every segment's attention graph at once.
    """
    out = torch.empty_like(pack.q)
    dq = torch.zeros_like(pack.q)
    dk = torch.zeros_like(pack.k)
    dv = torch.zeros_like(pack.v)

    with _tf32_disabled():
        q_off = 0
        k_off = 0
        for seg_idx, (Lq, Lk) in enumerate(zip(case.q_lens, case.k_lens)):
            qi = pack.q[q_off:q_off + Lq].detach().float().requires_grad_(True)
            ki = pack.k[k_off:k_off + Lk].detach().float().requires_grad_(True)
            vi = pack.v[k_off:k_off + Lk].detach().float().requires_grad_(True)

            qi_h = qi.transpose(0, 1).contiguous()
            ki_h = ki.transpose(0, 1).contiguous()
            vi_h = vi.transpose(0, 1).contiguous()

            scores = torch.matmul(qi_h, ki_h.transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)
            oi_h = torch.matmul(probs, vi_h)
            oi = oi_h.transpose(0, 1).contiguous()

            doi = pack.do[q_off:q_off + Lq].detach().float()
            oi.backward(doi)

            out[q_off:q_off + Lq] = oi.detach().to(pack.q.dtype)
            dq[q_off:q_off + Lq] = qi.grad.detach().to(pack.q.dtype)
            dk[k_off:k_off + Lk] = ki.grad.detach().to(pack.k.dtype)
            dv[k_off:k_off + Lk] = vi.grad.detach().to(pack.v.dtype)

            q_off += Lq
            k_off += Lk

            del qi, ki, vi, qi_h, ki_h, vi_h, scores, probs, oi_h, oi, doi
            if seg_idx % 16 == 0:
                torch.cuda.empty_cache()

    return RunOutput(o=out, dq=dq, dk=dk, dv=dv)


def _torch_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    case: BenchCase,
    scale: float,
) -> torch.Tensor:
    """PyTorch built-in SDPA, run per segment to preserve BlockDiagonalMask."""
    chunks = []
    q_off = 0
    k_off = 0
    for Lq, Lk in zip(case.q_lens, case.k_lens):
        qi = q[q_off:q_off + Lq].transpose(0, 1).unsqueeze(0)  # [1, H, Lq, D]
        ki = k[k_off:k_off + Lk].transpose(0, 1).unsqueeze(0)  # [1, H, Lk, D]
        vi = v[k_off:k_off + Lk].transpose(0, 1).unsqueeze(0)  # [1, H, Lk, D]
        oi = F.scaled_dot_product_attention(
            qi,
            ki,
            vi,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        chunks.append(oi.squeeze(0).transpose(0, 1).contiguous())
        q_off += Lq
        k_off += Lk
    return torch.cat(chunks, dim=0)


def _fa2_varlen_attention(
    flash_attn_varlen_func: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pack: InputPack,
    case: BenchCase,
    scale: float,
) -> torch.Tensor:
    return _first_tensor(
        flash_attn_varlen_func(
            q,
            k,
            v,
            pack.cu_q,
            pack.cu_k,
            case.max_q,
            case.max_k,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
        )
    )


def _candidate_attention(
    candidate: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pack: InputPack,
    scale: float,
    candidate_layout: str,
) -> torch.Tensor:
    if candidate_layout == "4d":
        out = candidate(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            cu_seqlens_q=pack.cu_q,
            cu_seqlens_k=pack.cu_k,
            scale=scale,
        )
        return _first_tensor(out).squeeze(0)

    out = candidate(
        q,
        k,
        v,
        cu_seqlens_q=pack.cu_q,
        cu_seqlens_k=pack.cu_k,
        scale=scale,
    )
    return _first_tensor(out)


def _call_impl(
    impl_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pack: InputPack,
    case: BenchCase,
    scale: float,
    candidate: Optional[Callable],
    flash_attn_varlen_func: Optional[Callable],
    candidate_layout: str,
) -> torch.Tensor:
    if impl_name == "torch_math":
        with _tf32_disabled():
            return _torch_math_attention(q, k, v, case, scale)
    if impl_name == "torch_sdpa":
        return _torch_sdpa_attention(q, k, v, case, scale)
    if impl_name == "fa2_varlen":
        if flash_attn_varlen_func is None:
            raise RuntimeError("FA2 varlen baseline is unavailable")
        return _fa2_varlen_attention(flash_attn_varlen_func, q, k, v, pack, case, scale)
    if impl_name == "candidate":
        if candidate is None:
            raise RuntimeError("Candidate varlen implementation is unavailable")
        return _candidate_attention(candidate, q, k, v, pack, scale, candidate_layout)
    raise ValueError(f"Unknown implementation: {impl_name}")


def _run_impl_once(
    impl_name: str,
    pack: InputPack,
    case: BenchCase,
    scale: float,
    candidate: Optional[Callable],
    flash_attn_varlen_func: Optional[Callable],
    candidate_layout: str,
) -> RunOutput:
    if impl_name == "torch_math":
        return _run_torch_math_oracle_segmented(pack, case, scale)

    q = pack.q.detach().clone().requires_grad_(True)
    k = pack.k.detach().clone().requires_grad_(True)
    v = pack.v.detach().clone().requires_grad_(True)
    out = _call_impl(
        impl_name,
        q,
        k,
        v,
        pack,
        case,
        scale,
        candidate,
        flash_attn_varlen_func,
        candidate_layout,
    )
    out.backward(pack.do)
    return RunOutput(
        o=out.detach(),
        dq=q.grad.detach(),
        dk=k.grad.detach(),
        dv=v.grad.detach(),
    )


def _check_case_correctness(
    case: BenchCase,
    pack: InputPack,
    impls: Sequence[str],
    scale: float,
    candidate: Optional[Callable],
    flash_attn_varlen_func: Optional[Callable],
    candidate_layout: str,
    tol_o: Optional[float],
    tol_grad: Optional[float],
    oracle: str = "torch_math",
) -> List[ErrorReport]:
    """
    Compute reference once, then diff every requested impl against it.

    oracle:
      "torch_math": fp32 explicit matmul+softmax (default — strongest oracle,
                    requires that we trust the math).
      "fa2_varlen": run FA2 in bf16 and use its outputs as the reference.
                    This catches the "candidate and FA2 both deviate from fp32
                    but in different directions" class of bug that the
                    indirect (vs torch_math) check would miss.
    """
    if oracle == "torch_math":
        ref = _run_impl_once(
            "torch_math",
            pack,
            case,
            scale,
            candidate,
            flash_attn_varlen_func,
            candidate_layout,
        )
    elif oracle == "fa2_varlen":
        if flash_attn_varlen_func is None:
            raise RuntimeError("FA2 oracle requested but flash_attn is unavailable")
        ref = _run_impl_once(
            "fa2_varlen",
            pack,
            case,
            scale,
            candidate,
            flash_attn_varlen_func,
            candidate_layout,
        )
    else:
        raise ValueError(f"Unknown oracle: {oracle}")

    effective_tol_o = (
        tol_o if tol_o is not None
        else max(5e-2, 5e-3 * math.sqrt(case.max_k / 1000.0))
    )
    effective_tol_grad = (
        tol_grad if tol_grad is not None
        else max(1e-1, 2e-2 * math.sqrt(case.max_k / 1000.0))
    )

    reports = []
    for impl_name in impls:
        if impl_name == oracle:
            # don't diff oracle against itself; that's always zero
            continue
        if impl_name == "torch_math":
            # if oracle is fa2_varlen and torch_math is in impls, still skip;
            # torch_math is too expensive to time and diffing it as a "user"
            # would just duplicate the inverse comparison.
            continue
        out = _run_impl_once(
            impl_name,
            pack,
            case,
            scale,
            candidate,
            flash_attn_varlen_func,
            candidate_layout,
        )
        report = ErrorReport(
            impl_name=impl_name,
            o_max=_max_abs(out.o, ref.o),
            dq_max=_max_abs(out.dq, ref.dq),
            dk_max=_max_abs(out.dk, ref.dk),
            dv_max=_max_abs(out.dv, ref.dv),
            o_mean=_mean_abs(out.o, ref.o),
            dq_mean=_mean_abs(out.dq, ref.dq),
            dk_mean=_mean_abs(out.dk, ref.dk),
            dv_mean=_mean_abs(out.dv, ref.dv),
        )
        reports.append(report)
        assert report.o_max < effective_tol_o, (
            f"[{case.name} {impl_name} vs {oracle}] output mismatch "
            f"{report.o_max:.3e} >= {effective_tol_o:.3e}"
        )
        assert report.grad_max < effective_tol_grad, (
            f"[{case.name} {impl_name} vs {oracle}] grad mismatch "
            f"{report.grad_max:.3e} >= {effective_tol_grad:.3e}"
        )
        del out

    del ref
    torch.cuda.empty_cache()
    return reports


def _time_cuda(fn: Callable[[], None], warmup: int, repeat: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _benchmark_impl(
    case: BenchCase,
    impl_name: str,
    mode: str,
    warmup: int,
    repeat: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    scale: float,
    candidate: Optional[Callable],
    flash_attn_varlen_func: Optional[Callable],
    candidate_layout: str,
    input_dist: str = "normal",
    contig: str = "default",
) -> TimingResult:
    pack = _make_input_pack(
        case, H, D, dtype, device, seed,
        input_dist=input_dist, contig=contig,
    )

    if mode == "forward":

        def run():
            with torch.no_grad():
                q = pack.q
                k = pack.k
                v = pack.v
                _call_impl(
                    impl_name,
                    q,
                    k,
                    v,
                    pack,
                    case,
                    scale,
                    candidate,
                    flash_attn_varlen_func,
                    candidate_layout,
                )

    elif mode == "backward":
        q = pack.q.detach().clone().requires_grad_(True)
        k = pack.k.detach().clone().requires_grad_(True)
        v = pack.v.detach().clone().requires_grad_(True)
        out = _call_impl(
            impl_name,
            q,
            k,
            v,
            pack,
            case,
            scale,
            candidate,
            flash_attn_varlen_func,
            candidate_layout,
        )

        def run():
            _clear_grads((q, k, v))
            out.backward(pack.do, retain_graph=True)

    elif mode == "fwd_bwd":
        q = pack.q.detach().clone().requires_grad_(True)
        k = pack.k.detach().clone().requires_grad_(True)
        v = pack.v.detach().clone().requires_grad_(True)

        def run():
            _clear_grads((q, k, v))
            out = _call_impl(
                impl_name,
                q,
                k,
                v,
                pack,
                case,
                scale,
                candidate,
                flash_attn_varlen_func,
                candidate_layout,
            )
            out.backward(pack.do)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    times = _time_cuda(run, warmup=warmup, repeat=repeat)
    mean_ms, std_ms = _mean_std(times)
    p50_ms = _percentile(times, 0.50)
    p90_ms = _percentile(times, 0.90)
    tflops = None
    if mode == "forward":
        tflops = _attention_fwd_flops(case, H, D) / (mean_ms * 1e-3) / 1e12
    return TimingResult(
        case_name=case.name,
        impl_name=impl_name,
        mode=mode,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        min_ms=min(times),
        max_ms=max(times),
        fwd_tflops_equiv=tflops,
    )


def _pareto_lens(n_seg: int, total: int, seed: int) -> Tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = (torch.rand(n_seg, generator=generator) ** -0.667 - 1.0).clamp_min(0.01)
    raw = raw / raw.sum() * total
    lens = raw.round().long().clamp_min(1)
    lens[-1] = total - lens[:-1].sum()
    if lens[-1] <= 0:
        # Rounding can theoretically over-allocate the first n-1 buckets.
        # Pull the deficit back from the largest earlier buckets.
        deficit = int(1 - lens[-1].item())
        lens[-1] = 1
        for idx in torch.argsort(lens[:-1], descending=True).tolist():
            take = min(deficit, int(lens[idx].item()) - 1)
            lens[idx] -= take
            deficit -= take
            if deficit == 0:
                break
        if deficit != 0:
            raise RuntimeError("Unable to construct positive pareto lengths")
    assert int(lens.sum().item()) == total
    return tuple(int(x) for x in lens.tolist())


def _balanced_lens(n_seg: int, total: int) -> Tuple[int, ...]:
    base = total // n_seg
    rem = total - base * n_seg
    return tuple(base + (1 if idx < rem else 0) for idx in range(n_seg))


def _random_lens(n_seg: int, total: int, seed: int) -> Tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.rand(n_seg, generator=generator).clamp_min(1e-6)
    raw = raw / raw.sum() * total
    lens = raw.round().long().clamp_min(1)
    lens[-1] = total - lens[:-1].sum()
    if lens[-1] <= 0:
        lens = torch.tensor(_balanced_lens(n_seg, total), dtype=torch.long)
    assert int(lens.sum().item()) == total
    return tuple(int(x) for x in lens.tolist())


def _extreme_tail_lens(n_seg: int, total: int) -> Tuple[int, ...]:
    if total < n_seg:
        raise ValueError("--k-total must be >= --n-seg so every segment has K >= 1")
    return tuple([1] * (n_seg - 1) + [total - (n_seg - 1)])


def _truncnorm_lens(
    n_seg: int,
    mean: int,
    std: int,
    min_k: int,
    max_k: int,
    seed: int,
) -> Tuple[int, ...]:
    """Per-segment K lengths drawn from a truncated normal(mean, std), clipped
    to [min_k, max_k]. Each segment is sampled independently. The total is
    whatever the samples produce (NOT rescaled to a target), so the realized
    distribution stays faithful to the parameters.

    Used by the prod-train profile to model real online K-length distribution:
    mild variation around a typical user history length, capped at 100k.
    """
    if n_seg < 1:
        raise ValueError("n_seg must be >= 1")
    if not (min_k <= mean <= max_k):
        raise ValueError(f"mean {mean} must be in [{min_k}, {max_k}]")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # rejection sampling: redraw any sample outside [min_k, max_k]. For the
    # parameters we use (std = 0.2*mean, range ~[1w, 10w] around mean ~6.7w),
    # the truncation rate is small so one pass usually suffices.
    out = torch.empty(n_seg, dtype=torch.long)
    remaining = n_seg
    filled = 0
    while remaining > 0:
        # over-sample to reduce loop iterations
        draw = (
            torch.randn(remaining * 2, generator=g) * std + mean
        ).round().long()
        ok = draw[(draw >= min_k) & (draw <= max_k)]
        take = min(ok.numel(), remaining)
        if take == 0:
            # truncation too aggressive; fall back to uniform fill of remainder
            fill = torch.randint(min_k, max_k + 1, (remaining,), generator=g)
            out[filled:filled + remaining] = fill
            break
        out[filled:filled + take] = ok[:take]
        filled += take
        remaining -= take
    return tuple(int(v) for v in out.tolist())


def _custom_k_lens(n_seg: int, total: int, dist: str, seed: int) -> Tuple[int, ...]:
    if n_seg < 1:
        raise ValueError("--n-seg must be >= 1")
    if total < n_seg:
        raise ValueError("--k-total must be >= --n-seg so every segment has K >= 1")
    if dist == "pareto":
        return _pareto_lens(n_seg=n_seg, total=total, seed=seed)
    if dist == "uniform":
        return _balanced_lens(n_seg=n_seg, total=total)
    if dist == "random":
        return _random_lens(n_seg=n_seg, total=total, seed=seed)
    if dist == "extreme":
        return _extreme_tail_lens(n_seg=n_seg, total=total)
    raise ValueError(f"Unknown K distribution: {dist}")


def _load_shape_file(
    shape_file: Optional[str],
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[BenchCase]:
    if not shape_file:
        raise ValueError("--shape-file is required when --profile shape-file")
    path = Path(shape_file)
    cases: List[BenchCase] = []
    seen = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q_lens = tuple(int(x) for x in row["q_lens"])
            k_lens = tuple(int(x) for x in row["k_lens"])

            if len(q_lens) != len(k_lens):
                raise ValueError(
                    f"{path}:{line_no}: q_lens/k_lens length mismatch "
                    f"{len(q_lens)} vs {len(k_lens)}"
                )
            if any(x <= 0 for x in q_lens) or any(x <= 0 for x in k_lens):
                raise ValueError(f"{path}:{line_no}: q_lens/k_lens must be positive")
            if "users" in row and int(row["users"]) != len(q_lens):
                raise ValueError(
                    f"{path}:{line_no}: users={row['users']} but "
                    f"len(q_lens)={len(q_lens)}"
                )
            if "tq" in row and int(row["tq"]) != sum(q_lens):
                raise ValueError(
                    f"{path}:{line_no}: tq={row['tq']} but sum(q_lens)={sum(q_lens)}"
                )
            if "tk" in row and int(row["tk"]) != sum(k_lens):
                raise ValueError(
                    f"{path}:{line_no}: tk={row['tk']} but sum(k_lens)={sum(k_lens)}"
                )

            if seen < offset:
                seen += 1
                continue
            if limit is not None and len(cases) >= limit:
                break

            name = str(row.get("name") or f"shape_line{line_no}")
            cases.append(BenchCase(name, q_lens, k_lens))
            seen += 1

    if not cases:
        raise ValueError(f"No shapes loaded from {path}")
    return cases


def _build_cases(
    profile: str,
    n_seg: int,
    q_per_seg: int,
    k_total: int,
    k_dist: str,
    k_seed: int,
    shape_file: Optional[str] = None,
    shape_limit: Optional[int] = None,
    shape_offset: int = 0,
) -> List[BenchCase]:
    if profile == "shape-file":
        return _load_shape_file(
            shape_file,
            limit=shape_limit,
            offset=shape_offset,
        )

    if profile == "custom":
        return [
            BenchCase(
                f"custom_{k_dist}_b{n_seg}_q{q_per_seg}_k{k_total // 1000}k",
                (q_per_seg,) * n_seg,
                _custom_k_lens(n_seg=n_seg, total=k_total, dist=k_dist, seed=k_seed),
            )
        ]

    if profile == "smoke":
        return [
            BenchCase("tiny", (7, 11), (19, 5)),
            BenchCase("cross_block", (129, 257), (33, 130)),
            BenchCase("single_seg", (400,), (400,)),
            BenchCase("block_aligned", (256, 256), (128, 128)),
            BenchCase("stage1_like", (500, 500), (19, 800)),
            BenchCase("stage2_uniform_b2", (512, 512), (500, 500)),
        ]

    if profile == "business-lite":
        return [
            BenchCase("stage2_uniform_b8", (512,) * 8, (500,) * 8),
            BenchCase("stage2_uniform_b32", (512,) * 32, (500,) * 32),
            BenchCase(
                "stage1_imbalanced_b8",
                (500,) * 8,
                (19, 800, 4096, 8192, 16384, 32768, 4096, 20000),
            ),
            BenchCase(
                "stage1_extreme_tail_b16",
                (500,) * 16,
                (1,) * 15 + (199_985,),
            ),
            BenchCase(
                "long_tail_b16_200k",
                (500,) * 16,
                _pareto_lens(n_seg=16, total=200_000, seed=200_000),
            ),
        ]

    if profile == "business":
        return [
            BenchCase("stage2_uniform_b32", (512,) * 32, (500,) * 32),
            BenchCase(
                "stage1_long_b50_2m",
                (500,) * 50,
                _pareto_lens(n_seg=50, total=2_000_000, seed=2_000_000),
            ),
        ]

    if profile == "business-sweep":
        return [
            BenchCase(
                f"stage1_long_b50_{total // 1000}k",
                (500,) * 50,
                _pareto_lens(n_seg=50, total=total, seed=total),
            )
            for total in (800_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000)
        ]

    if profile == "single-mega":
        # n_seg=1 stresses the kernel's single-program K loop limit. The
        # 2_000_000 case sends ONE program through ~62500 K-iters at BS=32.
        return [
            BenchCase("single_seg_q500_k200k", (500,), (200_000,)),
            BenchCase("single_seg_q500_k500k", (500,), (500_000,)),
            BenchCase("single_seg_q500_k1m", (500,), (1_000_000,)),
            BenchCase("single_seg_q500_k2m", (500,), (2_000_000,)),
        ]

    if profile == "head-dim-128":
        # Same shapes as business-lite but driven by D=128 callers (the
        # benchmark's --head-dim flag carries D into the pack). Needs the
        # caller to pass --head-dim 128 alongside this profile.
        return [
            BenchCase("d128_uniform_b8", (512,) * 8, (500,) * 8),
            BenchCase("d128_uniform_b32", (512,) * 32, (500,) * 32),
            BenchCase(
                "d128_long_tail_b16_200k",
                (500,) * 16,
                _pareto_lens(n_seg=16, total=200_000, seed=200_000),
            ),
        ]

    if profile == "qk-ratio":
        # Different Q-vs-K asymmetries to exercise both grid arms (fwd / dq
        # iterate over K, dkv iterates over Q). Each ~200k total to keep
        # the matrix small enough that the fp32 oracle runs fast.
        return [
            # K >> Q (business stage1 mega-user shape: many history tokens
            # attended to by a small interest-token block).
            BenchCase("k_dominant_q500_k200k", (500,) * 4, (50_000,) * 4),
            # Q >> K (rare in production but the kernel handles it).
            BenchCase("q_dominant_q50k_k500", (50_000,) * 4, (500,) * 4),
            # Q ~= K (self-attn-equivalent shape; the symmetric case).
            BenchCase("balanced_q5k_k5k", (5_000,) * 8, (5_000,) * 8),
        ]

    if profile == "prod-train":
        # Realistic online training distribution from production ranking job:
        #   - bs (item count) = 2000, dedup'd unique users per batch <= 50
        #   - q_per_seg = 500 (interest tokens per user, business-fixed)
        #   - per-user history length ~6w-8w mean, capped at ~10w
        # K is sampled from a truncated normal (mean ~6.7w, std=0.2*mean),
        # clipped to [10000, 100000]. No extreme tail -- mild variation only.
        # Each case fixes a different (n_seg, mean_k) point along the curve
        # from "small batch" to "max realistic batch (~2.6M K)".
        #
        # min_k=10000 is a placeholder; user will refine with real distribution.
        Q_PER_SEG = 500
        MIN_K = 10_000
        MAX_K = 100_000
        SEED = 2_000_000
        cases_spec = [
            ("prod_b12_k800k",  12, 67_000),
            ("prod_b15_k1000k", 15, 67_000),
            ("prod_b20_k1400k", 20, 70_000),
            ("prod_b30_k2000k", 30, 67_000),
            ("prod_b40_k2600k", 40, 65_000),
        ]
        cases = []
        for name, n_seg_i, mean_k_i in cases_spec:
            std_k = int(mean_k_i * 0.2)
            k_lens = _truncnorm_lens(
                n_seg=n_seg_i,
                mean=mean_k_i,
                std=std_k,
                min_k=MIN_K,
                max_k=MAX_K,
                seed=SEED + n_seg_i,  # vary per case so cases aren't identical draws
            )
            cases.append(BenchCase(name, (Q_PER_SEG,) * n_seg_i, k_lens))
        return cases

    raise ValueError(f"Unknown profile: {profile}")


def _estimate_case_memory_bytes(
    case: BenchCase,
    H: int,
    D: int,
    dtype: torch.dtype,
    needs_fp32_oracle: bool,
) -> int:
    """
    Conservative upper bound on peak memory for one case.

    Two components:
      packed_bytes:    q/k/v + cloned q/k/v + grads (always present).
      max_attn_bytes:  fp32 [max_q, max_k, H] attention matrix that the
                       torch_math oracle materializes per segment. Only
                       added when we'll actually run that oracle -- for
                       --speed-only or --oracle fa2_varlen this is dead
                       weight that wrongly skips legitimately runnable
                       cases (a 2M K-segment estimates at ~64GB but the
                       Triton kernel itself only needs ~256MB of partial
                       buffer + the bf16 inputs).
    """
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    packed_bytes = (case.total_q + 2 * case.total_k) * H * D * elem_bytes * 4
    if not needs_fp32_oracle:
        return packed_bytes
    max_attn_bytes = case.max_q * case.max_k * H * 4 * 4
    return packed_bytes + max_attn_bytes


def _print_case_header(case: BenchCase) -> None:
    print(
        f"\n[{case.name}] n_seg={case.n_seg} "
        f"T_q={case.total_q} T_k={case.total_k} "
        f"q_range=[{min(case.q_lens)},{max(case.q_lens)}] "
        f"k_range=[{min(case.k_lens)},{max(case.k_lens)}]"
    )


def _print_progress(index: int, total: int, case: BenchCase) -> None:
    width = 30
    done = int(width * index / max(total, 1))
    bar = "#" * done + "-" * (width - done)
    print(f"\n[progress] [{bar}] {index}/{total}  {case.name}")


def _print_correctness(reports: Sequence[ErrorReport], oracle: str = "torch_math") -> None:
    if not reports:
        return
    label = (
        "correctness vs torch_math fp32 oracle"
        if oracle == "torch_math"
        else f"correctness vs {oracle} (bf16 direct compare)"
    )
    print(f"  {label}:")
    print(
        f"    {'impl':<14} {'stat':<5} "
        f"{'o':>10} {'dq':>10} {'dk':>10} {'dv':>10} {'grad':>10}"
    )
    for report in reports:
        print(
            f"    {report.impl_name:<14} {'max':<5} "
            f"{report.o_max:10.3e} {report.dq_max:10.3e} "
            f"{report.dk_max:10.3e} {report.dv_max:10.3e} "
            f"{report.grad_max:10.3e}"
        )
        print(
            f"    {report.impl_name:<14} {'mean':<5} "
            f"{report.o_mean:10.3e} {report.dq_mean:10.3e} "
            f"{report.dk_mean:10.3e} {report.dv_mean:10.3e} "
            f"{report.grad_mean:10.3e}"
        )


def _merge_error_reports(
    worst: Dict[str, ErrorReport],
    reports: Sequence[ErrorReport],
) -> None:
    for report in reports:
        prev = worst.get(report.impl_name)
        if prev is None:
            worst[report.impl_name] = report
            continue
        worst[report.impl_name] = ErrorReport(
            impl_name=report.impl_name,
            o_max=max(prev.o_max, report.o_max),
            dq_max=max(prev.dq_max, report.dq_max),
            dk_max=max(prev.dk_max, report.dk_max),
            dv_max=max(prev.dv_max, report.dv_max),
            o_mean=max(prev.o_mean, report.o_mean),
            dq_mean=max(prev.dq_mean, report.dq_mean),
            dk_mean=max(prev.dk_mean, report.dk_mean),
            dv_mean=max(prev.dv_mean, report.dv_mean),
        )


def _print_timing(results: Sequence[TimingResult]) -> None:
    if not results:
        return

    by_key: Dict[Tuple[str, str], float] = {}
    for result in results:
        by_key[(result.case_name, result.mode)] = min(
            by_key.get((result.case_name, result.mode), float("inf")),
            result.mean_ms if result.impl_name == "fa2_varlen" else float("inf"),
        )

    print("  timing:")
    print(
        f"    {'impl':<14} {'mode':<8} {'mean':>9} {'std':>8} "
        f"{'p50':>9} {'p90':>9} {'min':>9} {'max':>9} "
        f"{'ratio':>8} {'fwd TFLOPS':>11}"
    )
    for result in results:
        baseline = by_key.get((result.case_name, result.mode), float("inf"))
        ratio = "-"
        if math.isfinite(baseline) and baseline > 0:
            ratio = f"{result.mean_ms / baseline:7.2f}x"
        tflops = "-"
        if result.fwd_tflops_equiv is not None:
            tflops = f"{result.fwd_tflops_equiv:10.2f}"
        print(
            f"    {result.impl_name:<14} {result.mode:<8} "
            f"{result.mean_ms:9.3f} {result.std_ms:8.3f} "
            f"{result.p50_ms:9.3f} {result.p90_ms:9.3f} "
            f"{result.min_ms:9.3f} {result.max_ms:9.3f} "
            f"{ratio:>8} {tflops:>11}"
        )


TimingPair = Tuple[BenchCase, TimingResult, TimingResult, float]


def _case_k_stats(case: BenchCase) -> Tuple[int, int, float, float]:
    k_min = min(case.k_lens)
    k_max = max(case.k_lens)
    k_mean = case.total_k / max(case.n_seg, 1)
    tail_ratio = k_max / max(k_mean, 1.0)
    return k_min, k_max, k_mean, tail_ratio


def _timing_pairs(
    results: Sequence[TimingResult],
    cases_by_name: Dict[str, BenchCase],
    mode: str,
) -> List[TimingPair]:
    by_key: Dict[Tuple[str, str, str], TimingResult] = {
        (r.case_name, r.impl_name, r.mode): r
        for r in results
    }
    pairs: List[TimingPair] = []
    for case_name, case in cases_by_name.items():
        candidate = by_key.get((case_name, "candidate", mode))
        fa2 = by_key.get((case_name, "fa2_varlen", mode))
        if candidate is None or fa2 is None or fa2.mean_ms <= 0:
            continue
        pairs.append((case, candidate, fa2, candidate.mean_ms / fa2.mean_ms))
    return pairs


def _print_pair_summary(mode: str, pairs: Sequence[TimingPair]) -> None:
    if not pairs:
        return
    candidate_ms = [candidate.mean_ms for _, candidate, _, _ in pairs]
    fa2_ms = [fa2.mean_ms for _, _, fa2, _ in pairs]
    ratios = [ratio for _, _, _, ratio in pairs]
    total_ratio = sum(candidate_ms) / max(sum(fa2_ms), 1e-12)
    win_rate = 100.0 * sum(1 for ratio in ratios if ratio < 1.0) / len(ratios)
    print(
        f"  {mode:<8} {len(pairs):6d} "
        f"{sum(candidate_ms) / len(candidate_ms):10.3f} "
        f"{sum(fa2_ms) / len(fa2_ms):10.3f} "
        f"{total_ratio:11.3f}x "
        f"{sum(ratios) / len(ratios):10.3f}x "
        f"{_percentile(ratios, 0.50):10.3f}x "
        f"{_percentile(ratios, 0.90):10.3f}x "
        f"{win_rate:8.1f}%"
    )


def _print_bucket_rows(
    title: str,
    pairs: Sequence[TimingPair],
    bucket_fn: Callable[[BenchCase], str],
) -> None:
    buckets: Dict[str, List[TimingPair]] = {}
    for pair in pairs:
        buckets.setdefault(bucket_fn(pair[0]), []).append(pair)
    if not buckets:
        return

    print(f"\n[bucket summary: {title}]")
    print(
        f"  {'bucket':<14} {'cases':>6} {'cand_ms':>10} {'fa2_ms':>10} "
        f"{'total_ratio':>11} {'ratio_p50':>10} {'ratio_p90':>10} {'win%':>8}"
    )
    for bucket, rows in buckets.items():
        candidate_ms = [candidate.mean_ms for _, candidate, _, _ in rows]
        fa2_ms = [fa2.mean_ms for _, _, fa2, _ in rows]
        ratios = [ratio for _, _, _, ratio in rows]
        total_ratio = sum(candidate_ms) / max(sum(fa2_ms), 1e-12)
        win_rate = 100.0 * sum(1 for ratio in ratios if ratio < 1.0) / len(ratios)
        print(
            f"  {bucket:<14} {len(rows):6d} "
            f"{sum(candidate_ms) / len(candidate_ms):10.3f} "
            f"{sum(fa2_ms) / len(fa2_ms):10.3f} "
            f"{total_ratio:11.3f}x "
            f"{_percentile(ratios, 0.50):10.3f}x "
            f"{_percentile(ratios, 0.90):10.3f}x "
            f"{win_rate:8.1f}%"
        )


def _print_extreme_cases(title: str, pairs: Sequence[TimingPair], *, reverse: bool) -> None:
    if not pairs:
        return
    rows = sorted(pairs, key=lambda item: item[3], reverse=reverse)[:10]
    print(f"\n[{title}]")
    print(
        f"  {'ratio':>8} {'cand_ms':>9} {'fa2_ms':>9} {'users':>5} "
        f"{'Tq':>8} {'Tk':>10} {'k_min':>8} {'k_max':>8} "
        f"{'k_mean':>10} {'tail':>6}  name"
    )
    for case, candidate, fa2, ratio in rows:
        k_min, k_max, k_mean, tail_ratio = _case_k_stats(case)
        print(
            f"  {ratio:7.3f}x {candidate.mean_ms:9.3f} {fa2.mean_ms:9.3f} "
            f"{case.n_seg:5d} {case.total_q:8d} {case.total_k:10d} "
            f"{k_min:8d} {k_max:8d} {k_mean:10.1f} {tail_ratio:6.2f}  "
            f"{case.name}"
        )


def _print_aggregate_timing_summary(
    results: Sequence[TimingResult],
    cases_by_name: Dict[str, BenchCase],
) -> None:
    if not results:
        return

    print("\n[overall timing summary: candidate vs fa2_varlen]")
    print(
        f"  {'mode':<8} {'cases':>6} {'cand_ms':>10} {'fa2_ms':>10} "
        f"{'total_ratio':>11} {'ratio_avg':>10} {'ratio_p50':>10} "
        f"{'ratio_p90':>10} {'win%':>8}"
    )
    mode_pairs: Dict[str, List[TimingPair]] = {}
    for mode in ("forward", "backward", "fwd_bwd"):
        pairs = _timing_pairs(results, cases_by_name, mode)
        if not pairs:
            continue
        mode_pairs[mode] = pairs
        _print_pair_summary(mode, pairs)

    fwd_bwd_pairs = mode_pairs.get("fwd_bwd")
    if not fwd_bwd_pairs:
        return

    _print_bucket_rows(
        "fwd_bwd by users",
        fwd_bwd_pairs,
        lambda case: (
            "users<=15" if case.n_seg <= 15
            else "users16-25" if case.n_seg <= 25
            else "users26-35" if case.n_seg <= 35
            else "users>35"
        ),
    )
    _print_bucket_rows(
        "fwd_bwd by total K",
        fwd_bwd_pairs,
        lambda case: (
            "Tk<1M" if case.total_k < 1_000_000
            else "Tk1-1.5M" if case.total_k < 1_500_000
            else "Tk1.5-2M" if case.total_k < 2_000_000
            else "Tk>=2M"
        ),
    )
    _print_bucket_rows(
        "fwd_bwd by max K",
        fwd_bwd_pairs,
        lambda case: (
            "maxK<64k" if case.max_k < 64_000
            else "maxK64-90k" if case.max_k < 90_000
            else "maxK>=90k"
        ),
    )
    _print_bucket_rows(
        "fwd_bwd by tail ratio",
        fwd_bwd_pairs,
        lambda case: (
            "tail<1.3" if _case_k_stats(case)[3] < 1.3
            else "tail1.3-1.8" if _case_k_stats(case)[3] < 1.8
            else "tail>=1.8"
        ),
    )
    _print_extreme_cases("best fwd_bwd cases for candidate", fwd_bwd_pairs, reverse=False)
    _print_extreme_cases("worst fwd_bwd cases for candidate", fwd_bwd_pairs, reverse=True)


def _resolve_impls(
    requested: Sequence[str],
    candidate: Optional[Callable],
    flash_attn_varlen_func: Optional[Callable],
) -> List[str]:
    impls = []
    for name in requested:
        if name == "candidate" and candidate is None:
            print("[warn] candidate requested but unavailable; skipping candidate")
            continue
        if name == "fa2_varlen" and flash_attn_varlen_func is None:
            print("[warn] fa2_varlen requested but unavailable; skipping FA2")
            continue
        impls.append(name)
    return impls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "smoke",
            "business-lite",
            "business",
            "business-sweep",
            "single-mega",
            "head-dim-128",
            "qk-ratio",
            "prod-train",
            "shape-file",
            "custom",
        ),
        default="smoke",
        help="Shape profile to benchmark.",
    )
    parser.add_argument(
        "--n-seg",
        type=int,
        default=50,
        help="Custom profile: number of packed segments.",
    )
    parser.add_argument(
        "--q-per-seg",
        type=int,
        default=500,
        help="Custom profile: Q tokens per segment.",
    )
    parser.add_argument(
        "--k-total",
        type=int,
        default=2_000_000,
        help="Custom profile: total K/V tokens across all segments.",
    )
    parser.add_argument(
        "--k-dist",
        choices=("pareto", "uniform", "random", "extreme"),
        default="pareto",
        help="Custom profile: K/V length distribution across segments.",
    )
    parser.add_argument(
        "--k-seed",
        type=int,
        default=2_000_000,
        help="Custom profile: seed for pareto/random K/V length distribution.",
    )
    parser.add_argument(
        "--shape-file",
        default=None,
        help=(
            "JSONL file for --profile shape-file. Each row must contain "
            "q_lens and k_lens arrays; optional name/users/tq/tk fields are validated."
        ),
    )
    parser.add_argument(
        "--shape-limit",
        type=int,
        default=None,
        help="Only load this many rows from --shape-file after --shape-offset.",
    )
    parser.add_argument(
        "--shape-offset",
        type=int,
        default=0,
        help="Skip this many valid rows from --shape-file before loading cases.",
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--candidate-varlen",
        default=DEFAULT_CANDIDATE,
        help="Candidate module:function. Use 'none' to disable.",
    )
    parser.add_argument(
        "--candidate-layout",
        choices=("3d", "4d"),
        default="3d",
        help="Call candidate with [T,H,D] or [1,T,H,D] q/k/v.",
    )
    parser.add_argument(
        "--impls",
        nargs="+",
        choices=("candidate", "fa2_varlen", "torch_sdpa", "torch_math"),
        default=None,
        help=(
            "Implementations to time and check against torch_math. "
            "Defaults to candidate/fa2/torch_sdpa for correctness, "
            "and candidate/fa2 for --speed-only."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("forward", "backward", "fwd_bwd"),
        default=None,
        help=(
            "Benchmark modes. Defaults to forward/backward for --speed-only, "
            "otherwise forward/backward/fwd_bwd."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timing-seeds",
        type=int,
        default=1,
        help="Number of independent input seeds to include in timing statistics.",
    )
    parser.add_argument("--softmax-scale", type=float, default=None)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Only run timing. Not recommended for new kernels.",
    )
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help=(
            "Shortcut for performance benchmarking: skip correctness and, "
            "unless --impls is provided, time only candidate and FA2 varlen."
        ),
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run fwd/bwd correctness checks and skip timing.",
    )
    parser.add_argument(
        "--correctness-seeds",
        type=int,
        default=1,
        help="Number of random input seeds per case for correctness checks.",
    )
    parser.add_argument("--tol-o", type=float, default=None)
    parser.add_argument("--tol-grad", type=float, default=None)
    parser.add_argument(
        "--input-dist",
        choices=("normal", "large", "exp", "small"),
        default="normal",
        help=(
            "Input distribution for q/k/v. "
            "'large' (randn*8) and 'exp' (randn().exp()) push softmax to "
            "extreme dynamic range; useful for stability bug hunts."
        ),
    )
    parser.add_argument(
        "--contig",
        choices=("default", "non_contig"),
        default="default",
        help=(
            "Input layout for q/k/v. 'non_contig' allocates a wider tensor "
            "and slices, exercising the wrapper's auto-contiguous path."
        ),
    )
    parser.add_argument(
        "--oracle",
        choices=("torch_math", "fa2_varlen"),
        default="torch_math",
        help=(
            "Reference for correctness diff. 'torch_math' is the strongest "
            "oracle (fp32 ground truth). 'fa2_varlen' compares directly to "
            "FA2's bf16 output and catches the class of bugs where both "
            "kernels deviate from fp32 in different directions."
        ),
    )
    parser.add_argument(
        "--skip-if-need-ratio",
        type=float,
        default=0.75,
        help="Skip a case when estimated memory exceeds this fraction of free memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.speed_only:
        args.skip_correctness = True
    if args.skip_correctness and args.correctness_only:
        raise ValueError(
            "--skip-correctness/--speed-only and --correctness-only are mutually exclusive"
        )
    if args.correctness_seeds < 1:
        raise ValueError("--correctness-seeds must be >= 1")
    if args.timing_seeds < 1:
        raise ValueError("--timing-seeds must be >= 1")
    if args.q_per_seg < 1:
        raise ValueError("--q-per-seg must be >= 1")
    if args.shape_limit is not None and args.shape_limit < 1:
        raise ValueError("--shape-limit must be >= 1")
    if args.shape_offset < 0:
        raise ValueError("--shape-offset must be >= 0")
    if args.modes is None:
        args.modes = (
            ("forward", "backward")
            if args.speed_only
            else ("forward", "backward", "fwd_bwd")
        )

    device = torch.device("cuda")
    dtype = _dtype_from_name(args.dtype)
    scale = args.softmax_scale
    if scale is None:
        scale = 1.0 / math.sqrt(args.head_dim)

    try:
        candidate = _load_callable(args.candidate_varlen)
    except Exception as exc:
        print(f"[warn] candidate import failed; candidate disabled: {exc}")
        candidate = None

    flash_attn_varlen_func = _load_flash_attn_varlen()
    if args.impls is None:
        args.impls = (
            ("candidate", "fa2_varlen")
            if args.speed_only
            else ("candidate", "fa2_varlen", "torch_sdpa")
        )
    impls = _resolve_impls(args.impls, candidate, flash_attn_varlen_func)
    if not impls:
        raise RuntimeError("No benchmark implementations are available")
    if args.oracle == "fa2_varlen" and flash_attn_varlen_func is None:
        raise RuntimeError(
            "--oracle fa2_varlen requested but flash_attn is unavailable. "
            "Either install flash-attn or use --oracle torch_math."
        )

    cases = _build_cases(
        args.profile,
        n_seg=args.n_seg,
        q_per_seg=args.q_per_seg,
        k_total=args.k_total,
        k_dist=args.k_dist,
        k_seed=args.k_seed,
        shape_file=args.shape_file,
        shape_limit=args.shape_limit,
        shape_offset=args.shape_offset,
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    print(
        f"Profile={args.profile}, dtype={args.dtype}, H={args.heads}, "
        f"D={args.head_dim}, scale={scale:.8f}"
    )
    print(f"Candidate={args.candidate_varlen} layout={args.candidate_layout}")
    if args.profile == "custom":
        print(
            f"Custom shape: n_seg={args.n_seg}, q_per_seg={args.q_per_seg}, "
            f"k_total={args.k_total}, k_dist={args.k_dist}, k_seed={args.k_seed}"
        )
    if args.profile == "shape-file":
        print(
            f"Shape file: path={args.shape_file}, loaded={len(cases)}, "
            f"offset={args.shape_offset}, limit={args.shape_limit}"
        )
    print(
        f"Impls={impls}; modes={list(args.modes)}; warmup={args.warmup}; "
        f"repeat={args.repeat}; correctness_seeds={args.correctness_seeds}; "
        f"timing_seeds={args.timing_seeds}"
    )
    print(
        f"Inputs: dist={args.input_dist}, contig={args.contig}; "
        f"oracle={args.oracle}"
    )
    # Extreme input distributions push bf16 vs fp32 noise above the default
    # 0.05 / 0.1 thresholds — that's the precision floor, not a kernel bug.
    # When the user picked an extreme dist + fp32 oracle but didn't pin tols,
    # auto-relax tols to a "stability-only" envelope (catches NaN/Inf and
    # gross corruption) and tell them how to do real cross-kernel alignment.
    if (
        not args.skip_correctness
        and args.input_dist != "normal"
        and args.oracle == "torch_math"
        and args.tol_o is None
        and args.tol_grad is None
    ):
        args.tol_o = 100.0
        args.tol_grad = 100.0
        print(
            f"[note] --input-dist {args.input_dist} produces bf16 vs fp32 noise "
            f"that saturates the default tol. Auto-relaxing to "
            f"--tol-o 100 --tol-grad 100 (stability-only mode: catches NaN / "
            f"Inf / gross corruption, not bf16 rounding noise)."
        )
        print(
            f"  For real cross-kernel bug hunting, rerun with "
            f"--oracle fa2_varlen --impls candidate --tol-o 1 --tol-grad 10."
        )
    if args.speed_only:
        print("Mode=speed-only; correctness is skipped")
    if args.correctness_only:
        print("Mode=correctness-only; timing is skipped")
    print(f"GPU memory free={free_bytes / 1e9:.2f}GB total={total_bytes / 1e9:.2f}GB")

    # Whether we'll actually run the fp32 torch_math oracle: only when
    # correctness is on AND the oracle is torch_math AND impls includes
    # something other than the candidate (otherwise nothing materializes
    # the per-segment fp32 attention matrix).
    needs_fp32_oracle = (
        not args.skip_correctness
        and args.oracle == "torch_math"
    )

    total_cases = len(cases)
    cases_by_name = {case.name: case for case in cases}
    all_timing_results: List[TimingResult] = []
    for case_index, case in enumerate(cases):
        _print_progress(case_index + 1, total_cases, case)
        _print_case_header(case)
        need = _estimate_case_memory_bytes(
            case, args.heads, args.head_dim, dtype,
            needs_fp32_oracle=needs_fp32_oracle,
        )
        if need > free_bytes * args.skip_if_need_ratio:
            print(
                f"  SKIP: estimated need {need / 1e9:.2f}GB exceeds "
                f"{args.skip_if_need_ratio:.0%} of free memory"
            )
            continue

        if not args.skip_correctness:
            worst_reports: Dict[str, ErrorReport] = {}
            for correctness_seed in range(args.correctness_seeds):
                pack = _make_input_pack(
                    case,
                    args.heads,
                    args.head_dim,
                    dtype,
                    device,
                    seed=args.seed + case_index * 1009 + correctness_seed,
                    input_dist=args.input_dist,
                    contig=args.contig,
                )
                reports = _check_case_correctness(
                    case,
                    pack,
                    impls,
                    scale,
                    candidate,
                    flash_attn_varlen_func,
                    args.candidate_layout,
                    args.tol_o,
                    args.tol_grad,
                    oracle=args.oracle,
                )
                _merge_error_reports(worst_reports, reports)
                del pack
                torch.cuda.empty_cache()

            ordered_reports = [
                worst_reports[impl_name]
                for impl_name in impls
                if impl_name in worst_reports
            ]
            if args.correctness_seeds > 1:
                print(f"  worst over {args.correctness_seeds} correctness seeds")
            _print_correctness(ordered_reports, oracle=args.oracle)

        if args.correctness_only:
            continue

        timing_results: List[TimingResult] = []
        for mode in args.modes:
            for impl_name in impls:
                per_seed = []
                for timing_seed in range(args.timing_seeds):
                    result = _benchmark_impl(
                        case,
                        impl_name,
                        mode,
                        args.warmup,
                        args.repeat,
                        args.heads,
                        args.head_dim,
                        dtype,
                        device,
                        seed=args.seed + case_index * 1009 + 17 + timing_seed,
                        scale=scale,
                        candidate=candidate,
                        flash_attn_varlen_func=flash_attn_varlen_func,
                        candidate_layout=args.candidate_layout,
                        input_dist=args.input_dist,
                        contig=args.contig,
                    )
                    per_seed.append(result)
                    torch.cuda.empty_cache()

                if len(per_seed) == 1:
                    timing_results.append(per_seed[0])
                else:
                    means = [x.mean_ms for x in per_seed]
                    mins = [x.min_ms for x in per_seed]
                    maxs = [x.max_ms for x in per_seed]
                    p50s = [x.p50_ms for x in per_seed]
                    p90s = [x.p90_ms for x in per_seed]
                    mean_ms, std_ms = _mean_std(means)
                    timing_results.append(
                        TimingResult(
                            case_name=case.name,
                            impl_name=impl_name,
                            mode=mode,
                            mean_ms=mean_ms,
                            std_ms=std_ms,
                            p50_ms=_percentile(p50s, 0.50),
                            p90_ms=_percentile(p90s, 0.90),
                            min_ms=min(mins),
                            max_ms=max(maxs),
                            fwd_tflops_equiv=(
                                _attention_fwd_flops(case, args.heads, args.head_dim)
                                / (mean_ms * 1e-3)
                                / 1e12
                                if mode == "forward"
                                else None
                            ),
                        )
                    )

        _print_timing(timing_results)
        all_timing_results.extend(timing_results)

    if not args.correctness_only:
        _print_aggregate_timing_summary(all_timing_results, cases_by_name)
    print("\nbenchmark PASS")


if __name__ == "__main__":
    main()
