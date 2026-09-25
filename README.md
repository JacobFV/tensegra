# Tensegra: programmable attention geometry

*A [Tensaco](https://tensaco.ai) research model. Formerly **Topoformer**: the old `topoformer` import path
still works as an alias, and the research archive under `research/` keeps the old name
so that its recorded paths and hashes stay valid.*


Research reports, audits, and raw experiment artifacts are indexed in [research/](research/README.md).

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
  .venv/bin/python -m tensegra.experiment \
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

The public API is `tensegra.structural_attention` and
`tensegra.graph_structure`:

```python
import torch
from tensegra import graph_structure, structural_attention

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

See the [pilot report](research/stages/stage-01/pilot-report.md) for results, limitations, and
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
  .venv/bin/python -m tensegra.study \
  --config configs/study-smoke.json --output results/study-smoke

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.study \
  --config configs/study.json --output results/study

PYTHONPATH=src .venv/bin/python -m tensegra.study_analysis \
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
[stage2-design.md](research/stages/stage-02/stage2-design.md) and
[stage2-journal.md](research/stages/stage-02/stage2-journal.md).

The completed [Stage 2 report](research/stages/stage-02/stage2-report.md) covers all 570 runs,
including negative findings. [Methods](research/stages/stage-02/stage2-methods.md),
[raw metrics](research/results/stage2/metrics.jsonl),
[run configuration and provenance](research/results/stage2/summary.json), and
[artifact audit](research/results/stage2/audit.json) are committed alongside it.

## Latent grounding into runtime graphs

Stage 3 decouples token positions from graph entities. `RuntimeGraph` stores
batched stable IDs, identity embeddings, directed relation matrices and padding
masks. `SoftGrounding` has independent query/key projections and an edge-free null
binding. `induce_bias` computes `Pq @ A_r @ Pk.transpose(-1, -2)`; gradients flow
through both grounding roles and relation strengths.

```python
from tensegra.runtime_graph import RuntimeGraph
from tensegra.grounding import SoftGrounding, induce_bias

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
  .venv/bin/python -m tensegra.grounding_study \
  --config configs/stage3-smoke.json --output results/grounding-smoke

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.grounding_study \
  --config configs/stage3.json --output results/grounding
```

`traversal_data.py` generates fresh graphs, opaque IDs, continuous identity keys,
independent categorical values and token permutations. Gold paths are excluded
from model inputs. `traversal_oracle.py` separately verifies exact one-hot bindings
through the actual attention contraction and retrieval operation. Diagnostics
separate answer accuracy, binding trajectories and attention trajectories.

See the [Stage 3 design](research/stages/stage-03/stage3-design.md),
[model assumptions](research/stages/stage-03/stage3-model-notes.md) and
[leakage review](research/stages/stage-03/stage3-leakage-review.md).

The [Stage 3 report](research/stages/stage-03/stage3-report.md) separates the main 39-run study
from a nine-run stronger content-prior control. [Methods](research/stages/stage-03/stage3-methods.md),
[main raw metrics](research/results/stage3/main/metrics.jsonl),
[supplement raw metrics](research/results/stage3/keyed/metrics.jsonl), and
[figures and diagnostic trajectories](research/results/stage3/main/analysis/report.md)
are included. Identity-initialized grounding supports some depth transfer;
random initialization and joint depth/size transfer remain weak. A stronger
graph-as-data control removes the evidence for metric-specific superiority.

Reproduce the supplemental control and analyze either family's metrics separately:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.grounding_study \
  --config configs/stage3-keyed.json --output results/grounding-keyed

PYTHONPATH=src .venv/bin/python -m tensegra.grounding_analysis \
  results/grounding/metrics.jsonl results/grounding-analysis
```

Figures use optional `matplotlib`; add `--no-plots` for tables and JSON only.
Without Torch installed, invoke `python3 src/tensegra/grounding_analysis.py`
directly with the same arguments to avoid importing the Torch-backed package.

## Stable binding substrate

Stage 4 separates learned matching from identity-state updates. `BindingTransformer`
adds cosine-normalized learned projections, identity-only grounding inputs,
attention-weighted immutable-key writes, and a separately labeled explicit
`Pq @ A` pointer write. Content computation remains neural. Pointer writes program
more of the transition directly and retain graph dependence even at zero logit
strength; they are not an ordinary-attention equivalence control.

`binding_metrics.py` keeps auxiliary ground/null/transition-consistency losses
outside model inference. The study distinguishes answer-only training from these
supervised losses. Canonical state sequences record initial binding and one binding
per transition, supporting first-error, persistence, recovery and reconvergence
analysis without counting a state twice.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.binding_study \
  --config configs/stage4-smoke.json --output results/binding-smoke

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.binding_study \
  --config configs/stage4.json --output results/binding

python3 src/tensegra/binding_analysis.py \
  results/binding/metrics.jsonl results/binding-analysis
```

The full study evaluates all 20 combinations of 16–128 nodes and 4–64 transitions,
with initialization anchors and three paired seeds. The analyzer also accepts
losslessly compressed `.jsonl.gz` metrics. See the
[Stage 4 design](research/stages/stage-04/stage4-design.md) and
[methods](research/stages/stage-04/stage4-methods.md) for the architectural priors, supervision
boundaries and preregistered stability gate before adaptive-strength experiments.

The [Stage 4 report](research/stages/stage-04/stage4-report.md) includes the completed 69-run
matrix, raw metrics, heatmaps, and independent audits. At depth 64 on 128 nodes,
identity-initialized explicit pointer writes achieve 99.74% task accuracy and 100%
complete paths; protected attention writes achieve 92.45% and 90.89%. The pointer
paths were already perfect at initialization. Randomly initialized pointer models
with grounding supervision reach 83.85% task accuracy and 86.20% complete paths.
These results distinguish preserving supplied binding from learning it from scratch.

The separately gated `configs/stage4-adaptive.json` compares two fresh pointer
models with base strength frozen at 4: fixed versus null-aware entropy modulation.
It changes content attention, while the explicit pointer transition remains intact.
Run it with the same `tensegra.binding_study` command and a separate output directory.

## Minimal protected runtime

Stage 5 adds an exact immutable language and a separate learned lowering/lifting
study. The runtime keeps bindings, values, slots, functions, call instances and
frames distinct, validates typed primitives, and rejects invalid operations
without partial state changes. It never evaluates generated host-language code.

```python
from tensegra.tiny_language import execute

assert execute('let x = 5; let y = add(x, 3); mul(y, 2)').value == 16
assert execute('let car = {wheels: [{radius: 10}, {radius: 13}]}; '
               'car.wheels[1].radius').value == 13
```

The learned benchmark supplies segmented operation clauses and noisy selector
features. A small transformer lowers them into protected runtime operations;
a learned head lifts scalar results into numeric, comparison or sign reports.
Scheduling and lexical-scope resolution remain supplied runtime semantics.
Matched neural controls receive the same initial graph as typed edge records.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.runtime_study \
  --config configs/stage5-smoke.json --output results/runtime-smoke

python3 src/tensegra/runtime_analysis.py \
  results/runtime/metrics.jsonl.gz results/runtime/analysis \
  --config results/runtime/config.json
```

See the [Stage 5 methods](research/stages/stage-05/stage5-methods.md),
[implementation plan](research/stages/stage-05/plan-05.md), and
[report](research/stages/stage-05/stage5-report.md) for supervision boundaries, controls,
confidence accounting and the distinction between interpreter support and
what the neural benchmark measures.

## Recurrent workspace and local runtime execution

Stage 6 adds a four-phase recurrent workspace, candidate-specific typed lowering
and readiness, adaptive emission, and protected persistent arithmetic registers.
Typed runtime returns re-enter the workspace before further computation. Candidate
routes may overlap; workspace rows are features, not discrete thought identities.

Two experiments isolate different interfaces: a controlled progressive-evidence
runtime task, and semantic decoding from multiple surfaces of a pinned TCN language
generator. Hidden graphs and traces are supervision targets, not actor inputs.
The language decoder does not yet drive the runtime end to end.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.thinking_study \
  --config configs/stage6-smoke.json --output results/thinking-smoke

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m tensegra.thinking_language \
  --config configs/stage6-language-pilot.json
```

The full paired budgets are in `configs/stage6.json` and
`configs/stage6-language.json`. Read the [Stage 6 report](research/stages/stage-06/stage6-report.md)
and [methods](research/stages/stage-06/stage6-methods.md) for experiment status, supervision
boundaries, negative findings, and the distinction between learned behavior and
supplied exact semantics. Earlier stages remain independently reproducible.

The completed 57-run study is a bounded negative result: learned policies did not
complete correct runtime trajectories, adaptive emission collapsed to the minimum
depth, and the language decoder remained near chance. Oracle execution stayed
exact, but learned output after symbolic return remained weak. The implementation
does not yet demonstrate a reliable learned latent–symbolic–latent interface.
