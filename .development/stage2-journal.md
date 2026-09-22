# Stage 2 development journal

## 2026-09-22 — kickoff

The user supplied a detailed interpretation of pilot 1 and explicitly requested full implementation, empirical tests and detailed reports. Plan 02 implements the requested order: efficiency, imperfect structure, unseen graphs/sizes, graph-as-input control, signed edges, and learned per-head/layer strengths. Language models remain deferred until the transfer/robustness milestone is demonstrated.

Baseline: source 7c3dda3, 67 tests passed remotely in 1.25 seconds. Worktree `.worktrees/stage2`, branch `feat/stage2`. CPU venv on gb10-direct retained; remote has ~115 GiB available and 504 GiB disk. No local training or Torch install.

Three separate implementation owners: study_data, study_model, study runner. Existing pilot modules and results remain unchanged. Parent coordinates scientific design, artifact analysis, independent review and integration.

Decisions:
- Iterate deterministic transitions from observed test histories to measure noise-free rollout; do not mislabel this as the exact multistep conditional expectation of the stochastic nonlinear process.
- Graph-as-input uses permutation-equivariant neighbor-history aggregation, a disclosed input/message-passing baseline with unbiased attention. This is a strong relevant control, but does not exhaust graph serialization designs.
- Signed typed bias receives sign and magnitude; distinguish its information advantage from topology-only comparisons.
- Size-transfer sparse graphs have fixed expected indegree 3. This avoids the original fixed-density generator's increasing neighbor count as N grows.
- Efficiency thresholds are defined using validation oracle and zero-predictor errors before inspecting test outcomes. Unreached thresholds are censored.
- Preserve full paired seed effects; three seeds are exploratory evidence, not a significance claim.

Pre-implementation refinement: shared efficiency normalization is fitted on the smallest (8-trajectory) training subset, so larger training pools do not leak distribution statistics into the smallest sample budget. The same statistics are reused at every count.

Before any stage-2 result, added the tighter 10% remaining oracle-to-zero gap threshold alongside 25% and 50%; the pilot suggests the looser targets could saturate early. Report every threshold without selecting one for the strongest result.

Before full experiments: identified a scientific limitation of identity-free fixed-graph baselines. Added preregistered `efficiency_identity` supplementary suite with matched learned node embeddings for none/soft4/hard across all sample budgets and seeds. This permits fixed variable identities to help learn adjacency. It costs 108 additional runs; it is separate from node-count-agnostic transfer. No stage-2 result motivated this addition.

## Full execution launch

Source `ee49edc` pushed before execution. Exact-source remote project tests: **126 passed in 1.55 seconds**. Runner independent review: spec PASS, quality PASS; final/checkpoint state hashes and strict configuration checks added after review. The complete six-suite smoke produced 118 completed reduced runs.

Full preset: 570 runs (216 efficiency, 108 identity-enabled efficiency, 150 corruption, 30 transfer, 42 heterogeneous, 24 learned), three seeds, 600 optimizer steps, checkpoint0/25/50/100/200/300/600. Remote output `~/topoformer-stage2-run/results/stage2-ee49edc`, log `~/topoformer-stage2-full.log`. One process, OMP/OpenBLAS2, cooperative7200-second deadline, external7500-second timeout. Output is incremental and preserved if interrupted.

Review clarifications: mixed corruption is one25% simultaneous drop/add stress condition; separate drop and add each cover10/25/50%. Pairing of exact batch indices is within each count/case across variants, not across different sample budgets. These limits are explicit in the design.
