# Prospective S19 analysis

`S19-analysis.py` consumes only a complete, successful frozen main: config `07ccff2ddb6d345355ea0b443dd93000d5296e7afeebd6fe2998ab826b645e9b`, all four curves, eight complete evaluation artifacts, 4,096 TRAIN constructions visited eight times, and the fixed construction stream. It checks source and wrapper hashes, final-state linkage, raw prediction/component consistency, invalid-output zero credit, teacher-forced field arithmetic, cell denominators, and identical event/target populations across curves. Checkpoint model/AdamW bytes and full codec replay remain independently audited; this script does not instantiate a model.

Output schema:

- `curves`: every 0/1,024/2,048/4,096 TRAIN128 and DEV2048 panel, eight teacher-forced loss/accuracy/support records, per-cell valid/complete/component counts, invalid reasons, macro graph F1 with invalid outputs assigned zero, and canonical-edge-order diagnostic.
- `endpoint_paired`: final event-matched transitions against original/context/10pass, each raw/historical/matched policy. Reference archive artifacts are SHA-bound through the independently audited S18 analysis. Exact semantic IDs, target tensors and graph identities must agree.
- `decisions`: final TRAIN≥64/128 and each trained fresh cell≥52/512 determine promotion; heldout3×4≥52 is reported separately. Extension eligibility requires TRAIN≥64, weighted trained accuracy≥5%, and ≥3 percentage-point gain from 2,048 to4,096; weights .5/.25/.25. Integer numerator arithmetic preserves threshold boundaries. Eligibility never launches an extension.
- `artifact_inventory`, immutable config/manifest/receipt hashes, phase timings and exposure counts make consumed evidence reviewable. Raw/canonical generated records remain in the input archive rather than being duplicated.

Report prose must lead with fixed endpoint gates, then TRAIN-versus-DEV acquisition, teacher-forced versus free-running discrepancy, invalid/error components, all baseline policies, and cost/history limitations. One scratch initialization gives no seed-level uncertainty estimate. Earlier actors inherit196,608 presentations; S19 inherits none. Parameters, objective, sequential interface and computation differ, so this is not an isolated architecture comparison. No confirmation is read and no threshold or checkpoint is selected.

The six synthetic fixtures in `S19-analysis-check.py` test promotion/heldout boundaries, weighted extension boundaries, paired transitions, count rejection, duplicate IDs, and invalid-zero aggregation without opening main results. Execute analysis only after the full main archive and receipt are closed.
