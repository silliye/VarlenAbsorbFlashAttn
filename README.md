# VarlenAbsorbFlashAttn

An efficient fusion matrix absorption operator supporting variable-length FlashAttention.

This repository contains two Triton kernel packages:

- `triton_fa_varlen_v2`: packed variable-length FlashAttention
- `triton_target_attn_absorb`: ragged absorb target-attention, `H=8, D_H=32`

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA
- PyTorch with CUDA support
- Triton
- flash-attn
- TensorFlow

## Installation

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install `flash-attn`:

```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation
```

Install each package:

```bash
cd triton_fa_varlen_v2
pip install -r requirements.txt
pip install -e .

cd ../triton_target_attn_absorb
pip install -r requirements.txt
pip install -e .
```

If PyTorch is installed from PyPI as a CPU build, reinstall it from the official CUDA wheel index before running the kernels.

## Usage

Varlen FlashAttention:

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

Absorb target-attention:

```python
from triton_target_attn_absorb import packed_target_attn

out = packed_target_attn(
    q_raw,
    X,
    W_Q,
    W_K,
    W_V,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
)
```

## Tests

```bash
cd triton_fa_varlen_v2
python -m triton_fa_varlen_v2.test_smoke

cd ../triton_target_attn_absorb
python -m triton_target_attn_absorb.test_smoke
```

## Benchmarks

Varlen FlashAttention:

```bash
cd triton_fa_varlen_v2
python tools/benchmark_packed_varlen_flash_attn.py --profile smoke
python tools/benchmark_packed_varlen_flash_attn.py \
  --profile shape-file \
  --shape-file fixtures/stage1_shapes.jsonl \
  --shape-limit 10
```

Absorb target-attention:

```bash
cd triton_target_attn_absorb
python -m triton_target_attn_absorb.test_smoke \
  --bench-lens-file fixtures/real_shape_all.json \
  --bench-lens-limit 5 \
  --bench-flash
```

## Layout

```text
.
├── triton_fa_varlen_v2/
│   ├── triton_fa_varlen_v2/
│   ├── tools/
│   └── fixtures/
└── triton_target_attn_absorb/
    ├── triton_target_attn_absorb/
    ├── tools/
    └── fixtures/
```

## License

Add a `LICENSE` file before publishing.
