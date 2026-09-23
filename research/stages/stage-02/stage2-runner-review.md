# Stage 2 runner review

Scope: `research/stages/stage-02/stage2-design.md`, `research/stages/stage-02/plan-02.md`, `src/topoformer/study.py`, `configs/study*.json`, and `tests/test_study.py`. This is a static scientific/code review; the parent agent owns the full smoke and suite runs.

## Spec verdict

**Pass after the scoped follow-up.** The runner gets the central controls right: the count sweep uses genuinely nested prefixes; normalization is fit to the smallest training prefix and reused; validation/test trajectories have separate seeds; transfer graph identities are split and checked; sparse transfer keeps expected nonself indegree fixed; modes within each case share base parameters and the exact minibatch schedule; graph corruption changes only supplied topology; clean-trained runtime corruption is separately labeled; typed models receive signed weights explicitly; and deterministic rollout begins at the same observed history as the stochastic target.

The remaining experiment-facing findings are:

1. **Resolved — the efficiency interpretation now has a preregistered node-identity control.** The identity-free `none` model is permutation equivariant even on the fixed asymmetric graph and cannot learn arbitrary adjacency as a table keyed by variable identity. The new `efficiency_identity` suite gives none/soft4/hard the same learned node table, initialization, schedule, and parameter count, and it remains excluded from transfer. The design accurately says the original suite measures graph information/representability together with learning efficiency and requires both suites to be reported.

2. **Resolved by clarification — pairing is explicitly scoped within count/case.** Modes at one trajectory count receive identical sampled indices. Across nested counts, the same seed is passed to `torch.randint` with different upper bounds, so realized window indices differ. The design now states this exactly and does not claim cross-count schedule identity.

3. **Resolved — trained-state provenance is present.** Every validation checkpoint now records `model_state_hash`, and each row records `final_state_hash`; the test pins the final hash to the last curve state.

4. **Resolved — strict validation covers the relevant malformed and unsupported inputs.** Numeric types, booleans, duplicate/inverted sweep lists, checkpoint element types, and typed-suite compatibility are checked before execution. Custom modes remain an intentional experiment override; the committed configurations use the preregistered matched comparisons, and rows expose parameter counts and information labels needed to audit an override.

## Quality verdict

**Pass.** The module boundaries are clear, overwrite protection is sound, rows are flushed incrementally, partial runs are not labeled complete, evaluation is chunked, CPU threads are capped, and test evaluation is not used for selection or thresholds. The code records graph/data seeds and hashes, checkpoint/final model hashes, graph quality, parameter counts, optimizer exposure, wall time, and peak process RSS.

One operational limitation should be explicit: `wall_seconds` is one process-wide deadline and is checked between training steps/evaluation records, not during dataset construction or a checkpoint validation pass. Thus it bounds the training loop approximately rather than imposing a hard per-suite/process kill time. Given the preregistered small synthetic setup, this is acceptable if described accurately; external job timeout should enforce the hard resource bound.

No leakage issue was found in one-step or rollout evaluation. The normalized zero baseline is algebraically correct, the one-step oracle uses the known noise-free transition against noisy targets, and deterministic recursive rollout starts at the final state of the same observed history. The output correctly labels that metric as deterministic reference error rather than a stochastic conditional expectation.
