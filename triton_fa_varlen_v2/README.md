# triton_fa_varlen_v2

Packed variable-length FlashAttention implemented in Triton.

Scope:

- non-causal attention
- no dropout / no bias
- MHA only, `HQ == H`
- input layout: `[T, H, D]` or `[1, T, H, D]`
- sequence metadata: `cu_seqlens_q / cu_seqlens_k` or `q_seqinfo / k_seqinfo`

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA
- PyTorch with CUDA support
- Triton
- flash-attn
- TensorFlow

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install runtime dependencies:

```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from triton_fa_varlen_v2 import packed_varlen_attn

out = packed_varlen_attn(
    q,
    k,
    v,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
)
```

With per-segment lengths:

```python
out = packed_varlen_attn(
    q,
    k,
    v,
    q_seqinfo=q_lens,
    k_seqinfo=k_lens,
)
```

## Tests

```bash
python -m triton_fa_varlen_v2.test_smoke
```

The smoke test checks forward and backward against a PyTorch fp32 reference.

## Benchmarks

```bash
python tools/benchmark_packed_varlen_flash_attn.py --profile smoke
python tools/benchmark_packed_varlen_flash_attn.py \
  --profile shape-file \
  --shape-file fixtures/stage1_shapes.jsonl \
  --shape-limit 10
python tools/profile_v2_kernels.py --limit 10
python tools/sweep_v2_blocks.py --limit 5
```

## Layout

```text
triton_fa_varlen_v2/
├── triton_fa_varlen_v2/
│   ├── interface.py
│   ├── kernels.py
│   ├── chunk_indices.py
│   └── test_smoke.py
├── tools/
├── fixtures/
├── pyproject.toml
└── requirements.txt
```

## License

Add a `LICENSE` file before publishing.
