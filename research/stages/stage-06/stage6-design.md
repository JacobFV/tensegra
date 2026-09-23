# Stage 6: recurrent latent workspace with local typed execution

The user specification is authoritative. Build and measure phases 1–7; no pretrained model integration. Preserve every Stage 1–5 source, configuration, test and artifact. Use the linked CPU machine for Torch tests/training with two threads and bounded batches.

## Architecture

A current workspace of distributed feature rows is updated by four distinct attention/MLP sublayers, reused at every latent microstep. Rows are not named thoughts. Candidate queries softly pool overlapping workspace components; each predicts a primitive, ordered operand bindings, readiness and a soft routing factor. Free heads remain unregularized. Selected structural heads use role-specific normalized grounding and induced Pq A Pk-transpose bias; graph supervision never replaces the actor's predicted graph with a hidden target.

A protected session owns typed literal/value registers and executed results independently of the workspace. Arithmetic and comparison form the first exact primitive subset. Multiple valid independent proposals may execute in a microstep; overlapping proposals are audited and duplicate transitions are idempotent. Formal validity checks types/current availability, never gold correctness. Typed return events contain the primitive, ordered argument identities, value/type and provenance. Their learned encoding is added through soft candidate routing, never by deleting residual components. A subsequent recurrent update is mandatory before emission. Output heads read the workspace only, not an extra exact-result argument. Symbolic results remain in persistent memory.

Emission uses a learned probe with hard minimum/maximum, smooth minimum/maximum biases and ponder cost. Token time and microstep time are distinct. Initial benchmarks emit a bounded answer sequence or answer label; this is not a general autoregressive language model. No past latent Q/K/V cache is retained.

## Data and privilege

Pin and vendor a minimal TCN language-engine dependency closure with MIT license and file hashes. Compile its terms into our typed graph with explicit ordered argument slots, bindings/scopes and provenance. Render one fixed semantic construction into multiple surfaces; never assume same seed across language-dependent generators means same semantics. TCN construction, future rewrites, instance IDs and solution semantics are auxiliary targets/audit only.

Use two independently reported measurements: actual TCN surface-to-semantic grounding/generalization, and a controlled progressively disclosed expression/rewrite workload exercising candidate competition, exact execution and reintegration. Share the semantic graph schema and recurrent architecture; do not describe a controlled supplied interface as full language induction. The execution workload must have a downstream decision requiring a returned value plus independent context. Event-drop/shuffle/wrong-value controls test causal use of reintegration.

Training may use privileged trace-aligned proposals/events, explicitly labeled teacher forcing. Free-running evaluation must use predicted actions/readiness and actual persistent state. Every hidden-target mutation must leave frozen-parameter actor inference unchanged. Independent schedules anneal grounding, topology, transition and readiness losses. Task-only from cold and warm initialization are distinct controls.

## Measurement and interpretation

Run paired seeds, step-zero/curves, shallow training and depth8/16/32 evaluation where feasible. Report architectural supplied priors, learned behavior, exact semantics and interpretation separately. Record input/source/data/compiler/renderer hashes. Metrics cover task/result/trajectory, per-loss values, grounding/null/entropy/margin, head edge mass, local readiness calibration, premature/refused/conflicting execution, projection overlap/interference, event round trips, microstep/compute distributions and halting failures. Undefined conditional denominators remain null. Unsupported capabilities are explicit; failure is a result, not grounds to silently narrow the scientific claim.

## Module contracts and ownership

- `semantic_graph.py`, `tcn_data.py`, vendor closure: canonical immutable node/edge graphs and multi-surface records (graph worker).
- `thinking.py`: reusable cell, current workspace, candidate heads/routing, emit policy and event encoder (model worker).
- `thinking_runtime.py`: protected values, typed primitive schemas, incremental exact proposals, persistent typed events (runtime worker).
- `thinking_tasks.py`: controlled ambiguity/rewrite episodes and physically separate public/privileged records (data worker).
- `thinking_study.py`, configs: teacher-forced training, free-running evaluation, baselines/annealing/provenance (runner worker after interfaces).
- `thinking_analysis.py`, reports: artifact-only aggregation, audits and plots (analysis worker after schema).

Root coordinates public contracts and independent reviews, runs frozen remote experiments and integrates/pushes verified work. Workers own disjoint files and must not revert peers.
