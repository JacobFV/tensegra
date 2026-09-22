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
