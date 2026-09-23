# Stage 2 implementation plan

> For agentic workers: use subagent-driven-development. User explicitly requested full implementation and execution with subagents.

**Goal:** Test efficiency, graph corruption, runtime transfer, input-graph controls, heterogeneous edges, and learned structural strength.
**Architecture:** Three independent modules (data, model, runner) plus artifact-only analysis. Preserve the pilot modules.
**Tech stack:** Python >=3.10, torch>=2.0, pytest; CPU remote execution.
**Spec:** `research/stages/stage-02/stage2-design.md`.

## Global constraints
- No local training; remote CPU with two threads.
- Preserve pilot public API and original results.
- Paired initial shared parameters/data/schedules; report exceptions and extra parameters.
- All model selection uses validation only; no test-derived thresholds.
- Directed convention: row i reads column j. Self edges survive corruption.
- Finite metrics, disjoint graph and trajectory splits, no target leakage.

## Review focus
- Corruption rounding and empty edge sets; preserve input tensors.
- Graph-batch indexing must follow sampled example indices.
- Size transfer must not silently change expected degree or fit test normalization.
- Unreached efficiency thresholds must remain censored, never infinite fabricated speedups.
- Deadlines must save completed results without labeling partial training complete.

## Task 1 — study data
- [x] Write tests for exact seeded corruption, signed contraction, graph disjointness and deterministic targets; observe failure remotely.
- [x] Implement `study_data.py`, `tests/test_study_data.py`.
- [x] Test remotely and commit.
Interfaces: `make_system(kind,n,seed,mechanism='uniform',degree=3) -> Dynamics`; `supply_graph(graph, corruption='clean', fraction=0.0, seed=0) -> bool tensor`; `graph_quality(truth,supplied) -> dict`; `deterministic_future(system,history,horizon) -> [B,H,N]`.

## Task 2 — study models
- [x] Write tests for zero-alpha equivalence, alpha gradients, permutation equivariance, batched graphs, graph-input influence, typed relation coefficients; observe failure remotely.
- [x] Implement `StudyPredictor(history,width,heads,layers,variant='none',strength=4.0)` in `study_model.py`; `forward(x,graph,weights=None)`; `structure_coefficients() -> list`.
Variants: none, soft, hard, graph_input, learned, typed. Graphs [N,N] or [B,N,N], optional weights same shape. Reuse core layers, no global pooling or node-specific embeddings. Copy matching state keys between variants for paired initialization.
- [x] Run covering tests and commit.

## Task 3 — study runner
- [x] Test smoke execution, paired schedule identity, graph/data split isolation, corruption semantics, censoring and incremental artifact behavior.
- [x] Implement `study.py` and configs for efficiency, corruption, transfer, heterogeneous, learned suites, using above interfaces.
- [x] Run remote smoke and full suite tests; commit and push source before experiments.

## Task 4 — execute, analyze, report
- [x] Run bounded complete remote suites with 3 seeds; retain raw per-run and per-evaluation results.
- [x] Implement artifact-only analysis, efficiency/paired summaries and exportable figures; record exact thresholds and selection rules.
- [x] Write detailed `research/stages/stage-02/stage2-report.md` including negative results, limits and next milestone decision; update README and journal.
- [x] Independent code/scientific review, address findings, run covering checks; commit/push artifacts and reports.

### Supplementary control — identity-enabled fixed-graph efficiency
- [x] Model optional `node_count` creates learnable node embeddings with consistent state keys; default behavior unchanged. Test exact paired zero-alpha behavior, embedding gradients, and size validation.
- [x] Runner `efficiency_identity` repeats counts/seeds/domains with none/soft4/hard; explicit `node_identity` metadata. No identities in transfer.
- [x] Analysis treats both efficiency suites separately and explains representational limitation of the identity-free no-graph baseline.
