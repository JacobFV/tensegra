# Stage 4 independent artifact audit

Status: main run in progress; final disposition pending complete coverage and any gated follow-up.

The main training source is clean commit `3b63b88fee75b45887f11837164af659dc882a2d`. The audit executes separately in `/tmp` on gb10-direct and never mutates the running checkout or experiment records. `scripts/audit_binding_results.py` loads one JSONL run at a time, checks finite numbers, complete grid/budgets, raw canonical trajectory counts, task counts, paired data/schedules/ordinary initialization, initialization/final pairing, source/config identity and the per-seed corner gate. Local full-artifact loading is unnecessary.

A partial check of 40/69 completed runs passed all applicable assertions. Incomplete gate rows are not evidence of failure or success. Final coverage must be rerun without `--partial`.

## Independent exact-routing control

At frozen source, `exact_attention_traverse` was evaluated on the actual D64/N128 corner seeds and batches: **384/384 answers and complete 65-state paths correct**, 128 examples in each of three seeds. Runtime was 2.607 seconds with two CPU threads. Public model inputs alone were passed into the oracle; targets and paths were scored afterward. Exact identity matching and hard graph routing are privileged, nonlearned controls, so this result establishes only that the attention/rebinding substrate supports the requested computation. Raw counts and seed rule are in `stage4-oracle-audit.json`.

## Remaining audit

- Finish 69-run coverage, per-seed stability gate and aggregate/report arithmetic.
- Coordinator's runner reviewer owns checkpoint source/state/scalar replay audits, avoiding duplicate execution.
- Confirm original Stage 1–3 tracked source/config/test files remain unchanged at integration.
- Review gated adaptive results separately if the fixed-strength gate passes.
