# Task 1 report: general directed attention reference

## RED

Remote host: `gb10-direct`, directory: `~/topoformer-pilot`, virtual environment: `.venv`.

Command (after syncing `pyproject.toml` and `tests/` without `--delete`):

```text
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_attention.py -q
```

Result: pytest exited 2 during collection with the expected
`ModuleNotFoundError: No module named 'topoformer'` because the implementation
did not exist.

A later precision regression check against the pre-fix contraction produced one
expected failure: low-precision output was `-6.98046875` when float32
accumulation called for `-6.984375`.

## GREEN

Targeted command:

```text
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_attention.py -q
```

Result: `18 passed in 0.75s`.

Full-suite command:

```text
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m pytest -q
```

Result: `18 passed in 0.70s`.

Remote versions: Python 3.12, torch 2.14.0+cpu, numpy 2.5.3, pytest 9.1.1.

## Changes

- Added the reusable `structural_attention` function with directed additive
  bias, strict broadcast validation, boolean masking, finite-input checks,
  all-masked-row rejection, rectangular query/key support, float32/float64
  score computation, and value-dtype outputs.
- Added `graph_structure` for `none`, numeric-adjacency `soft`, and
  adjacency-or-self `hard` modes.
- Added the package exports and minimal package/test metadata.
- Added tests for every behavior named in the brief, including orientation,
  isolated nodes, gradients, masking, and low-precision accumulation.

## Review and concerns

Self-review found that casting weights down before the value contraction lost
low-precision accuracy. The added regression test failed against that version;
the contraction now accumulates in the score dtype before returning the value
dtype.

The isolated environment uses the official CPU torch wheel. CPU behavior is
fully verified. GPU support was not exercised because this environment has no
CUDA-enabled torch build; the implementation uses device-preserving PyTorch
operations and contains no CPU-specific transfers.
