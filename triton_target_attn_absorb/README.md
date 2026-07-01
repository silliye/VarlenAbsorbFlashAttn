# triton_target_attn_absorb

Ragged absorb target-attention implemented in Triton.

This snapshot targets `H=8, D_H=32`.

Core computation:

```text
u = q_raw @ W_Q @ W_K.T
O = softmax(u @ X.T) @ (X @ W_V)
```

PyTorch autograd handles the outer projections. Triton kernels compute the inner ragged attention and gradients for `u`, `X`, and `W_V`.

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

Precompute chunk metadata on hot paths:

```python
from triton_target_attn_absorb import (
    packed_target_attn,
    prepare_target_attn_indices,
)

q_idx, k_idx = prepare_target_attn_indices(cu_q, cu_k)

out = packed_target_attn(
    q_raw,
    X,
    W_Q,
    W_K,
    W_V,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    q_chunk_indices=q_idx,
    k_chunk_indices=k_idx,
    validate=False,
)
```

## Tests

```bash
python -m triton_target_attn_absorb.test_smoke
```

## Benchmarks

```bash
python -m triton_target_attn_absorb.test_smoke --bench
python -m triton_target_attn_absorb.test_smoke --bench-shape-suite
python -m triton_target_attn_absorb.test_smoke \
  --bench-lens-file fixtures/real_shape_all.json \
  --bench-lens-limit 5 \
  --bench-flash
```

## Tools

- `tools/parse_ta_shape_log.py`
- `tools/parse_ta_pack_stats_log.py`
- `tools/summarize_ta_shape_records.py`
- `tools/summarize_ta_lens_records.py`

## Layout

```text
triton_target_attn_absorb/
├── triton_target_attn_absorb/
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
