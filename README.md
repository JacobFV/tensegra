# Topoformer structure pilot

This repository contains a controlled pilot for testing graph-biased attention on
synthetic sparse and robot-shaped dynamical systems. The first generators use a
shared, nonnegative, uniform-neighbor mechanism. That makes topology identifiable
to a shared permutation-equivariant predictor, but it does not test arbitrary
edge coefficients or physical rigid-body dynamics.

Create a reproducible environment and install the project:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

For a CPU-only environment, install PyTorch from its CPU wheel index first, then
install the project:

```sh
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test]'
```

Run the tests in the prepared environment:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src python -m pytest
```

Run the paired pilot:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  python -m topoformer.experiment --config configs/pilot.json --output results/pilot
```

The runner validates configuration before allocating data, uses training-only
scalar normalization, clones one initial model state and one batch schedule across
graph ablations, and writes `metrics.jsonl` plus `summary.json`. It reports raw and
normalized one-step error, recursive rollout error, and a transient initial-state
shift using fresh trajectories at twice the initial scale with no burn-in. The
zero and persistence references expose noise-dominated results. The conditional
mean oracle receives privileged generator weights and is explicitly diagnostic,
not a fair learned baseline. Soft-bias strength is selected from validation error;
test errors are not used for selection. A wall-time limit retains completed rows
and marks the summary incomplete rather than implying a matched comparison.

## Core attention API

The public API is `topoformer.structural_attention` and
`topoformer.graph_structure`:

```python
import torch
from topoformer import graph_structure, structural_attention

# read_graph[b, i, j] means query row i may read key column j.
read_graph = torch.tensor([[[1, 1], [0, 1]]], dtype=torch.bool)
q = k = v = torch.randn(1, 1, 2, 8)  # [batch, heads, tokens, features]
bias, allowed = graph_structure(read_graph, mode="soft")
output, weights = structural_attention(q, k, v, bias=bias, strength=1.0)

# Strict local masking permits graph edges plus self reads.
_, allowed = graph_structure(read_graph, mode="hard")
output, weights = structural_attention(q, k, v, allowed=allowed)
```

Soft mode adds a finite bonus and still permits every otherwise legal read. Hard
mode masks nonedges while preserving self reads. `attention.py` contains these
primitives, `graphs.py` and `data.py` build synthetic systems, `model.py` defines
the shared graph predictor and token MLP, and `training.py`, `evaluation.py`, and
`experiment.py` provide the paired training, metrics, and CLI path.

See the [pilot report](.development/pilot-report.md) for results, limitations, and
the recommended next experiments.
