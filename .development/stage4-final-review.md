# Stage 4 independent artifact audit

Status: main artifact audit passed; gated adaptive follow-up remains separately reviewable.

The main training source is clean commit `3b63b88fee75b45887f11837164af659dc882a2d`. The audit executes separately in `/tmp` on gb10-direct and never mutates the running checkout or experiment records. `scripts/audit_binding_results.py` loads one JSONL run at a time, checks finite numbers, complete grid/budgets, raw canonical trajectory counts, task counts, paired data/schedules/ordinary initialization, initialization/final pairing, source/config identity and the per-seed corner gate. Local full-artifact loading is unnecessary.

A partial check of 40/69 completed runs passed all applicable assertions. Incomplete gate rows are not evidence of failure or success. Final coverage must be rerun without `--partial`.

## Independent exact-routing control

At frozen source, `exact_attention_traverse` was evaluated on the actual D64/N128 corner seeds and batches: **384/384 answers and complete 65-state paths correct**, 128 examples in each of three seeds. Runtime was 2.607 seconds with two CPU threads. Public model inputs alone were passed into the oracle; targets and paths were scored afterward. Exact identity matching and hard graph routing are privileged, nonlearned controls, so this result establishes only that the attention/rebinding substrate supports the requested computation. Raw counts and seed rule are in `stage4-oracle-audit.json`.

## Remaining audit

- Finish 69-run coverage, per-seed stability gate and aggregate/report arithmetic.
- Coordinator's runner reviewer owns checkpoint source/state/scalar replay audits, avoiding duplicate execution.
- Confirm original Stage 1–3 tracked source/config/test files remain unchanged at integration.
- Review gated adaptive results separately if the fixed-strength gate passes.

## Completed main audit

The final streaming audit passed **69/69 runs**, all 400 steps, 23 variants × three seeds, each with the full 20-cell size/depth matrix. Source/config fingerprints agree; all numeric fields are finite; raw task/canonical path correctness and pooled persistence/recovery counts agree with reported metrics. Ordinary initialization, training schedules, evaluation examples and initialization/final comparisons are paired as declared. The audit JSON records raw-metrics, config-file and audit-script SHA256 hashes.

Exactly one configuration passes the preregistered D64/N128 gate: `cosine_pointer_identity`, with task accuracy **1.0000, 1.0000, 0.9921875** and complete canonical paths **1.0000 in all three seeds**. This configuration contains the explicit graph pushforward identity update. Its pass is not evidence that attention bias alone learns stable traversal.

`cosine_attention_identity` falls short: task accuracy **0.9921875, 0.8515625, 0.9296875** and path completion **1.0000, 0.8046875, 0.921875**. Averaging seeds cannot turn this into a gate pass. The supervised random pointer variants improve substantially but none passes the per-seed gate.

All **41** previously tracked files under src/tests/configs at baseline 8243cba remain unchanged. Main artifact checks are independently recorded in `results/stage4/main/audit.json`. No full raw dataset was loaded locally, and no running source/checkpoint was modified.

## Adaptive implementation and main report review

Reviewed follow-up changes 23b45dd and 11b921a. Confidence is `(1 - p_null) * clamp(1 - H(P)/log(N+1), 0, 1)`: confident real identity yields one; uniform or fully null grounding yields zero. It scales attention strengths only, not the explicit pointer transition. Both arms freeze the same base strength at four and initialize identical parameters with no added parameters. The main registry still defaults to its original 23 variants. Tests cover endpoints, finite gradients, parameter/RNG matching, fixed base strengths, unchanged pointer writes and zero-strength equivalence. Coordinator reports the clean follow-up checkout passes **268 tests in 2.87 seconds**. No implementation blocker found; adaptive empirical review awaits complete rows.

The main narrative report was checked against the artifact-only aggregate: **72 architecture-table values, 60 auxiliary-table values and 20 initialization-table values** agree with reported rounding. Independently verified the two highlighted persistence/recovery counts, marginal products, six/21 recovered-final trajectories, named feature correlations, 1,062.4-second total wall time, 297.8-second optimizer time and 961.2-MiB process high-water mark. Scope statements correctly separate initialized preservation, supervised acquisition, attention writes and programmed pointer writes. Adaptive findings must remain distinct from the main study.
