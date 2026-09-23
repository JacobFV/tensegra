# R07: nonlinear accessibility of frozen return features

R06 balanced fitting improved the worst development grid cell from 41/64 to 55/64, while the original-mixture minimum improved from 4076/4096 to 4088/4096. Its independent audit is complete (09cb46d). Neither result establishes uniform exact reconstruction. The next question is whether a nonlinear consumer can access the remaining distinctions in the same frozen workspace.

## One intervention

Reuse the immutable R06 balanced feature cache, backbone 12, and all existing development populations. Compare the replayed linear 1024→33 head (33,825 parameters) with a residual MLP:

`linear(z) + output(GELU(hidden(z)))`

The hidden layer has width 1024; the output projection starts at zero. Both arms therefore start with exactly the same original-wide-head predictions. The residual consumer has 1,117,250 parameters. All readout parameters train; the 1024-dimensional workspace, memory, encoder, and backbone do not change. This tests additional decoder capacity, with correspondingly greater readout computation, not an equal-parameter or equal-FLOP comparison.

## Inputs, supervision, and exposure

Both arms receive the same train-standardized scalar workspace. Both receive the same 33-class exact-value targets for 16,384 balanced public return events at delays 0/1/2/4/8/16. Both run AdamW at .003, batch 256, for exactly 900 updates (230,400 presentations), using the same sampled row stream and original linear warm start. R06's supplied legal-type/value sampling prior remains explicit. The nonlinear initialization seed is 34000012; the shared batch generator retains 33000012.

The linear endpoint must reproduce R06's balanced weights within 1e-6 and its visited-row bitset exactly. A replay mismatch blocks interpretation. Record parameters, fit support, visited rows/events, losses, calibration curves, full logits, all non-value predictions, and per-value/type/operation error groups.

## Selection, evaluation, and stopping

The primary endpoint is fixed at 900 updates. Calibration checkpoints every 100 updates are diagnostic; no best-checkpoint selection, threshold tuning, or automatic extension to 1800 updates is authorized. Advance toward fresh confirmation only if the nonlinear head improves the minimum calibration-grid type/value cell across delays 0/1/16 and every original-mixture calibration cell remains at least 98%. Report full validation results regardless. Reused R06 calibration and validation are development data, not untouched confirmation.

Report fitting accuracy and fresh-context accuracy separately, by delay and type/value stratum. Preserve zero-step ingestion and all non-value/full-joint metrics. A failed nonlinear probe does not prove missing information. Delay 32 is absent from this experiment. Do not replace the audited R05 consumer or change composition dependencies from this diagnostic.

## Compute and provenance

Mechanical profile: load the immutable cache and run 10 matched updates per arm, capped at 60 seconds. Its outputs are timing/mechanical diagnostics, not a model-selection opportunity. Development is initially estimated below two GPU minutes, to be revised from the profile before release. No GPU execution without coordinator authorization. Freeze source/configuration before execution; retain both model states, compressed raw predictions/logits, original cache/checkpoint hashes, exact replay evidence, and process-occupancy receipt.
