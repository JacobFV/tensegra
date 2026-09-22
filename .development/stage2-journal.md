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
