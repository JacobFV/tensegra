# Topoformer: programmable attention geometry

This repository contains a controlled pilot for testing graph-biased attention on
synthetic sparse and robot-shaped dynamical systems. The first generators use a
shared, nonnegative, uniform-neighbor mechanism. That makes topology identifiable
to a shared permutation-equivariant predictor, but it does not test arbitrary
edge coefficients or physical rigid-body dynamics.

Create a reproducible environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

For a CPU-only environment, install PyTorch from its CPU wheel index, then install
the project:

```sh
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test]'
```

Alternatively, let pip select the PyTorch build available from its default index:

```sh
.venv/bin/python -m pip install -e '.[test]'
```

Run the tests in the prepared environment:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m pytest
```

Run the paired pilot:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m topoformer.experiment \
  --config configs/pilot.json --output results/pilot
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

## Controlled follow-up studies

Stage 2 tests sample and optimization efficiency, incomplete topology, unseen
graphs and larger node counts, a graph-conditioned input baseline, signed edge
weights, and zero-initialized learned layer/head biases. A supplementary efficiency
suite gives every compared model the same learnable variable-identity table;
this separates fixed-graph learning from the identity-free baseline's inability
to directly memorize arbitrary adjacency. Transfer models remain identity-free.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m topoformer.study \
  --config configs/study-smoke.json --output results/study-smoke

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m topoformer.study \
  --config configs/study.json --output results/study

PYTHONPATH=src .venv/bin/python -m topoformer.study_analysis \
  results/study/metrics.jsonl results/study-analysis
```

Use `--suite efficiency`, `corruption`, `transfer`, `heterogeneous`, `learned`, or
`efficiency_identity` to run one study. Outputs refuse overwriting. Full runs use
two CPU threads, bounded evaluation batches and a configured wall-time limit.
For optional exported scientific figures, install `matplotlib` in the environment
and add `--plots` to the analysis command with a fresh output directory.

`study_data.py` defines size-stable process families and reproducible graph
corruptions; `study_model.py` composes fixed, learned and typed graph biases with
the shared predictor; `study.py` owns paired datasets, training and provenance;
`study_analysis.py` reads artifacts without importing Torch. Graph-input models
receive neighbor-history aggregates through an input projection while attention
stays unbiased. Typed models receive privileged signed coefficient information,
which is disclosed separately from adjacency-only comparisons.

Efficiency thresholds use validation oracle-to-zero gaps, with unreached targets
reported as censored. Curves include initialization and all declared optimizer
checkpoints. Deterministic rollouts iterate the known noise-free transition from
observed histories; they are not exact multi-step conditional expectations of the
stochastic nonlinear process. The robot generator remains a morphology-shaped
synthetic proxy, not a physics simulator or VLA.

The preregistered design and implementation notes are in
[stage2-design.md](.development/stage2-design.md) and
[stage2-journal.md](.development/stage2-journal.md).

The completed [Stage 2 report](.development/stage2-report.md) covers all 570 runs,
including negative findings. [Methods](.development/stage2-methods.md),
[raw metrics](.development/results/stage2/metrics.jsonl),
[run configuration and provenance](.development/results/stage2/summary.json), and
[artifact audit](.development/results/stage2/audit.json) are committed alongside it.

## Latent grounding into runtime graphs

Stage 3 decouples token positions from graph entities. `RuntimeGraph` stores
batched stable IDs, identity embeddings, directed relation matrices and padding
masks. `SoftGrounding` has independent query/key projections and an edge-free null
binding. `induce_bias` computes `Pq @ A_r @ Pk.transpose(-1, -2)`; gradients flow
through both grounding roles and relation strengths.

```python
from topoformer.runtime_graph import RuntimeGraph
from topoformer.grounding import SoftGrounding, induce_bias

# IDs [B,N], entity embeddings [B,N,D], directed adjacency [B,R,N,N].
graph = RuntimeGraph(node_ids, entity_embeddings, adjacency)
grounder = SoftGrounding(latent_dim=32, entity_dim=16, temperature=0.05)
pq, pk = grounder(query_residuals, key_residuals, graph)  # final slot is null
bias = induce_bias(pq, graph.adjacency, pk)  # [B,R,T_query,T_key]
# Call grounder again on the updated residuals at the next layer.
```

The small traversal transformer recomputes grounding from an evolving query
residual at every recurrent step over shuffled immutable entity memory. Relation
instructions externally schedule those steps. Identity-aligned initialization is
an explicit prior, compared against random initialization. This is a keyed-memory
architectural experiment, not a programming-language interpreter or a language
model experiment. The original explicit-graph predictors and studies are unchanged.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m topoformer.grounding_study \
  --config configs/stage3-smoke.json --output results/grounding-smoke

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m topoformer.grounding_study \
  --config configs/stage3.json --output results/grounding
```

`traversal_data.py` generates fresh graphs, opaque IDs, continuous identity keys,
independent categorical values and token permutations. Gold paths are excluded
from model inputs. `traversal_oracle.py` separately verifies exact one-hot bindings
through the actual attention contraction and retrieval operation. Diagnostics
separate answer accuracy, binding trajectories and attention trajectories.

See the [Stage 3 design](.development/stage3-design.md),
[model assumptions](.development/stage3-model-notes.md) and
[leakage review](.development/stage3-leakage-review.md).

The [Stage 3 report](.development/stage3-report.md) separates the main 39-run study
from a nine-run stronger content-prior control. [Methods](.development/stage3-methods.md),
[main raw metrics](.development/results/stage3/main/metrics.jsonl),
[supplement raw metrics](.development/results/stage3/keyed/metrics.jsonl), and
[figures and diagnostic trajectories](.development/results/stage3/main/analysis/report.md)
are included. Identity-initialized grounding supports some depth transfer;
random initialization and joint depth/size transfer remain weak. A stronger
graph-as-data control removes the evidence for metric-specific superiority.

Reproduce the supplemental control and analyze either family's metrics separately:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m topoformer.grounding_study \
  --config configs/stage3-keyed.json --output results/grounding-keyed

PYTHONPATH=src .venv/bin/python -m topoformer.grounding_analysis \
  results/grounding/metrics.jsonl results/grounding-analysis
```

Figures use optional `matplotlib`; add `--no-plots` for tables and JSON only.
Without Torch installed, invoke `python3 src/topoformer/grounding_analysis.py`
directly with the same arguments to avoid importing the Torch-backed package.
