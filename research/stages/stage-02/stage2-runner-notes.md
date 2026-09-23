# Stage 2 runner implementation notes

## Interfaces and scope

`python -m topoformer.study --config configs/study.json --output RESULTS` runs all six studies; `--suite NAME` filters one suite. Default full grid has 570 trained runs across three seeds: efficiency 216, identity-controlled efficiency 108, corruption 150, transfer 30, heterogeneous 42, learned coefficients 24. Checkpoints at 0/25/50/100/200/300/600 supply optimizer-budget curves without repeated restarts. All test evaluation happens after final or deadline-limited training, never at intermediate checkpoints. Rows mark partial training/evaluation explicitly.

The runner preserves the pilot modules and reuses the separately implemented study data/model APIs. CPU threads are limited to two; evaluation chunks are capped at 64 (default 32). No local torch installation or training was used. Temporary development tests ran in `~/topoformer-stage2-runner` via the existing remote CPU venv. End-to-end smoke artifacts are development verification, not reported experimental evidence.

## Scientific accounting

- Fixed-process studies split trajectory seeds while keeping the generating system fixed. Transfer uses separate graph seeds and rejects any exact graph identity overlap between train/validation/test.
- Efficiency trajectory subsets are nested. Scalar normalization fits only the smallest training subset (8 by default), reused across counts, so small-count runs do not use larger-count statistics. Fixed-process companion suites use those same training-only statistics. Transfer fits training graphs only.
- Training uses graph-uniform batches, each wholly belonging to the sampled graph; mixed node counts do not require padding. The exact graph/index schedule and shared initial parameters are hashed. Pairing is within the same trajectory-count case across modes; different counts share the schedule RNG seed but necessarily have different sampled index ranges. All common parameter tensors are copied from the same unbiased initialization, while variant-specific parameters and their costs remain explicit.
- Validation and one-step test use at most 256 deterministic equally spaced trajectory-local windows per graph. The row records actual window counts. Recursive tests use the first observed history of every test trajectory. Graphs are averaged equally during validation, not weighted by node count. Per-graph test values remain raw in artifacts for matching analysis.
- Oracle-gap thresholds are declared before full results: oracle validation error plus 10%, 25%, or 50% of zero-minus-oracle validation error. Nonpositive reference gaps produce null thresholds. First observed crossings are recorded; unreached thresholds remain null. Counts are trajectories, separately from correlated window counts and optimizer examples.
- Deterministic rollout iterates the noise-free generator from the same observed history. It is a deterministic transition reference, not the exact multistep stochastic conditional mean. The stochastic oracle rollout applies those same deterministic transitions against stochastic future observations.
- Corruption retrains soft1/soft4/hard under each supplied graph condition. The clean condition additionally trains an unbiased reference. Clean-trained rows also receive all seven runtime corruptions. Generating weights never change. Self edges are preserved; realized graph quality is recorded.
- Graph-input receives adjacency through neighbor-history aggregation with unbiased attention, explicitly a graph-conditioned input/message-passing control. Typed signed channels receive actual edge weights and are labeled as additional information. Typed models appear only in the clean signed suite by default.

## Artifacts and resources

`metrics.jsonl` is flushed after each model; `summary.json` is atomically replaced after each row and at the end. Both retain per-graph evaluations, validation curves, coefficients, thresholds, schedule and parameter hashes, graph/trajectory seeds, training-only normalization, sample counts, parameter counts, step counts and timings. Summary records config, source revision, git status, environment and process-lifetime peak RSS. Nonempty output directories are rejected. A global monotonic deadline saves completed rows and the final partial row rather than presenting partial runs as completed experiments.

## Verification

Tests were written first; the initial remote test collection failed with `ModuleNotFoundError: topoformer.study` before implementation. Ten runner tests then passed remotely, covering smoke artifacts, overwrite protection, deadline summary, pairing, smallest-subset normalization, trajectory isolation, graph split isolation, mixed-size transfer, corruption conditions and runtime evaluation, typed information labels, and schedule indexing. A complete five-suite smoke produced 104 rows before the six-seed-independent clean corruption-reference addition (the revised six-suite one-seed smoke completed 118 rows in 2.12 seconds). Representative 600-step 12-node runs took approximately 2.1–2.6 seconds each. These timings only estimate full execution budget; report actual full-suite elapsed time from final artifacts.

## Pre-result review refinements

The identity-controlled efficiency suite adds learned fixed node identities identically to none/soft4/hard, with matched identity-table initialization and parameter counts. This distinguishes some representational limitations of the permutation-equivariant unbiased baseline from topology learning efficiency. Transfer remains identity-free. Every row labels `node_identity`. Checkpoints and final trained model states are hashed for auditability. Typed overrides are restricted to the signed suite; booleans masquerading as numerical config values and duplicate/unsorted sweep values are rejected. Corruption mixed coverage is specifically 25% drop plus 25% add, alongside individual drop/add at 10/25/50%.

Final runner verification after pre-result refinements: 22 tests passed, including identity initialization/parameter matching, final/checkpoint state hashes, and strict configuration regression cases.
