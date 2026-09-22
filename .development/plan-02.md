# Stage 2 implementation plan

> For agentic workers: use subagent-driven-development. User explicitly requested full implementation and execution with subagents.

**Goal:** Test efficiency, graph corruption, runtime transfer, input-graph controls, heterogeneous edges, and learned structural strength.
**Architecture:** Three independent modules (data, model, runner) plus artifact-only analysis. Preserve the pilot modules.
**Tech stack:** Python >=3.10, torch>=2.0, pytest; CPU remote execution.
**Spec:** `.development/stage2-design.md`.

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
- [ ] Write tests for exact seeded corruption, signed contraction, graph disjointness and deterministic targets; observe failure remotely.
- [ ] Implement `study_data.py`, `tests/test_study_data.py`.
- [ ] Test remotely and commit.
Interfaces: `make_system(kind,n,seed,mechanism='uniform',degree=3) -> Dynamics`; `supply_graph(graph, corruption='clean', fraction=0.0, seed=0) -> bool tensor`; `graph_quality(truth,supplied) -> dict`; `deterministic_future(system,history,horizon) -> [B,H,N]`.

## Task 2 — study models
- [ ] Write tests for zero-alpha equivalence, alpha gradients, permutation equivariance, batched graphs, graph-input influence, typed relation coefficients; observe failure remotely.
- [ ] Implement `StudyPredictor(history,width,heads,layers,variant='none',strength=4.0)` in `study_model.py`; `forward(x,graph,weights=None)`; `structure_coefficients() -> list`.
Variants: none, soft, hard, graph_input, learned, typed. Graphs [N,N] or [B,N,N], optional weights same shape. Reuse core layers, no global pooling or node-specific embeddings. Copy matching state keys between variants for paired initialization.
- [ ] Run covering tests and commit.

## Task 3 — study runner
- [ ] Test smoke execution, paired schedule identity, graph/data split isolation, corruption semantics, censoring and incremental artifact behavior.
- [ ] Implement `study.py` and configs for efficiency, corruption, transfer, heterogeneous, learned suites, using above interfaces.
- [ ] Run remote smoke and full suite tests; commit and push source before experiments.

## Task 4 — execute, analyze, report
- [ ] Run bounded complete remote suites with 3 seeds; retain raw per-run and per-evaluation results.
- [ ] Implement artifact-only analysis, efficiency/paired summaries and exportable figures; record exact thresholds and selection rules.
- [ ] Write detailed `.development/stage2-report.md` including negative results, limits and next milestone decision; update README and journal.
- [ ] Independent code/scientific review, address findings, run covering checks; commit/push artifacts and reports.
