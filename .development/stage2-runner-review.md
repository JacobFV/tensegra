# Stage 2 runner review

Scope: `.development/stage2-design.md`, `.development/plan-02.md`, `src/topoformer/study.py`, `configs/study*.json`, and `tests/test_study.py`. This is a static scientific/code review; the parent agent owns the full smoke and suite runs.

## Spec verdict

**Provisionally needs changes before experiments.** The runner gets the central controls right: the count sweep uses genuinely nested prefixes; normalization is fit to the smallest training prefix and reused; validation/test trajectories have separate seeds; transfer graph identities are split and checked; sparse transfer keeps expected nonself indegree fixed; modes within each case share base parameters and the exact minibatch schedule; graph corruption changes only supplied topology; clean-trained runtime corruption is separately labeled; typed models receive signed weights explicitly; and deterministic rollout begins at the same observed history as the stochastic target.

The remaining experiment-facing findings are:

1. **High — the efficiency interpretation needs the node-identity control now preregistered by the parent.** `StudyPredictor` has no node identity parameters, so the `none` model is permutation equivariant even on the fixed asymmetric graph and cannot learn an arbitrary adjacency as a table keyed by variable identity. The default none-versus-topology result therefore measures the value of supplied graph information/representability together with data efficiency; it does not isolate optimizer sample savings after both models can represent the fixed graph. The new fixed-node-ID `efficiency_identity` comparison should be complete, paired, and reported separately before making a learning-efficiency claim.

2. **Medium — cross-count schedules are deterministic but not shared exactly** (`study.py:_schedule`, `_train`). Modes at one trajectory count receive identical sampled indices, which is the important treatment pairing. Across nested counts, the same seed is passed to `torch.randint` with different upper bounds, so the realized window indices and schedule hashes differ. Reports should say schedules are paired across modes within a count. If the preregistration intends exact cross-count coupling, derive each schedule from common random variates/trajectory-local coordinates and record that coupling.

3. **Medium — trained-state provenance is absent** (`study.py:_train`). Rows contain initialization and schedule hashes but no final or checkpoint model-state hash, despite the design requiring state hashes. Record at least the final state hash; checkpoint hashes alongside each validation point would make curves auditable.

4. **Medium — supported configuration input is not yet strict enough** (`StudyConfig.__post_init__`). Boolean seeds, `burn_in`, `noise`, and `wall_seconds` are accepted as integers/numbers. Non-numeric values for `noise`, `learning_rate`, and `wall_seconds` can escape as `TypeError` from `math.isfinite`; checkpoints can similarly fail while sorting before element validation. Counts and size lists allow duplicates/unsorted values, weakening the meaning of a nested sweep. Arbitrary custom modes are accepted for every suite, so a `typed` mode can run on a uniform mechanism while being labeled `signed_edge_weights`. Validate types before arithmetic/sorting, require unique increasing sweep lists where order has scientific meaning, and enforce or accurately label suite/mode combinations.

## Quality verdict

**Good structure, with provenance/validation hardening required.** The module boundaries are clear, overwrite protection is sound, rows are flushed incrementally, partial runs are not labeled complete, evaluation is chunked, CPU threads are capped, and test evaluation is not used for selection or thresholds. The code records graph/data seeds and hashes, graph quality, parameter counts, optimizer exposure, wall time, and peak process RSS.

One operational limitation should be explicit: `wall_seconds` is one process-wide deadline and is checked between training steps/evaluation records, not during dataset construction or a checkpoint validation pass. Thus it bounds the training loop approximately rather than imposing a hard per-suite/process kill time. Given the preregistered small synthetic setup, this is acceptable if described accurately; external job timeout should enforce the hard resource bound.

No leakage issue was found in one-step or rollout evaluation. The normalized zero baseline is algebraically correct, the one-step oracle uses the known noise-free transition against noisy targets, and deterministic recursive rollout starts at the final state of the same observed history. The output correctly labels that metric as deterministic reference error rather than a stochastic conditional expectation.
