# Stage 10 return readout: cross-delay contracts

Status: design before new inference. The workspace remains 1024-dimensional. Use the three immutable Stage 9 factorized-persistent backbones (initialization seeds 10, 11, 12), with no encoder, recurrent-cell, backbone, or runtime training.

## Question

Stage 9 showed that fresh readouts recover values that the original classifier misses. It did not establish whether one fixed readout works across recurrent delays. This study measures the full transfer matrix from decoder-training delay to evaluation delay on the **same underlying held-out events**.

The observation supplied to every decoder is exactly the normalized 1024-dimensional scalar workspace row used by the original classifier. The privileged target is its known 33-class half-unit scalar label. Exact value reconstruction is primary; numerical distance is descriptive. Frozen backbone training covered delays 0/1/2/4. New readout training at a longer delay does not make that delay in-distribution for the backbone, but does make it covered for that readout.

## Frozen comparisons

1. Seven equal-capacity categorical ridge decoders: train at source delay `d_s` in `{0,1,2,4,8,16,32}`, then evaluate at every target delay in that set. Each has 33,825 coefficients including intercept. Train-only standardization and the regularization grid `{.01,.1,1,10}` are fixed. Select regularization by exact accuracy on calibration examples at the **source delay only**, with smaller grid index breaking ties. A decoder trained at 32 is a diagnostic reference, not a 32-step extrapolation result.
2. Original frozen scalar classifier at every target delay.
3. Fresh short-pool ridge reference trained on 0/1/2/4.
4. One shared ridge decoder trained on 0/1/2/4/8/16, without any delay identifier or changed workspace. Its 32-step evaluation is genuine readout-length extrapolation. Calibration uses only its covered delays.

For the shared/short comparison, use the same 2,048 distinct training events and **8,192 training feature rows**. The short pool uses all four states per event. The shared pool uses a deterministic balanced, cyclic subsample of six states, retaining every event and assigning per-delay row counts differing by at most one. The selection is fixed by event indices before labels or features are inspected. This matches total row exposure and distinct events while changing delay coverage; it cannot match the full distribution of training states. Report all per-delay counts and selection hashes. Per-source decoders use 2,048 rows each and are not exposure-matched to the pooled heads.

All heads are deterministic fits to fixed backbone features. The three independent initialization replicas are the inherited backbone seeds, not invented random seeds for a closed-form solve. These experiments do not test new neural acquisition from random initialization.

## Data and isolation

Generate fresh partitions with the existing exact return generator: train seed 10,100,001 (2,048 events), calibration seed 10,200,001 (512 events), test seed 10,300,001 (512 events). These do not reuse Stage 9 probe, head-fitting, or inspected test data. Training uses two distractor rows; calibration/test evaluate two and eight distractors. The event generator creates identity keys and semantic fields before distractors, so equal seed/count pairs identify the same events across delays and distractor conditions; verify event hashes and all target arrays rather than assuming this.

Every delay cell retains all 512 events, each initialization separately. All 33 scalar labels belong to the training domain. Fresh-context/nonce-identity generalization is distinct from unseen numerical labels. No held-out test state, label, or delay-32 calibration enters the shared or short-pool fit/selection. The source-32 diagnostic fit is isolated and cannot select any other head.

No Stage 9 feature cache is reused: fresh data separation is more valuable than avoiding this small capture. Cache new float32 features once per backbone, losslessly, with source/config/checkpoint/data/feature hashes and event identities. Store compact predictions/targets and fit coefficients; retain cache blobs remotely if their size makes repository inclusion inappropriate. Report nominal and effective feature dimensions, parameters, memory allocation, row counts, unique events, calibration exposure, GPU allocation, process RSS, and runtime. Reuse current-state-only frozen recurrence; no historical q/k/v memory is introduced.

## Metrics and advancement boundary

Report all 7×7 source/target exact-accuracy matrices for every seed and distractor condition, with diagonal versus off-diagonal differences computed on paired events. Also report each pooled/reference curve, confusion matrices, signed/absolute errors, and whether failures move to neighboring values. A matrix cell cannot infer information absence from a linear probe failure.

A restricted scalar-coverage gate requires exact accuracy **above 98% in every prescribed seed and both distractor conditions at every covered delay 0/1/2/4/8/16**, at least 512 examples per cell. A separate scalar-OOD gate requires the same criterion at 32 for the shared decoder. These are scalar-only readout gates, not complete return-interface passes. Preserve all inherited type, primitive, identity, joint and autonomous-composition blocks. No non-value competence claim follows from a scalar matrix.

No hyperparameter change follows test inspection. Stop after this matrix and one matched-row shared comparison. If diagonal fits fail, report a readout-family/acquisition limitation. If diagonal fits work but transfer fails, report a delay-dependent interface rather than asserting information destruction. If shared coverage helps but 32 fails, distinguish readout coverage from extrapolation. Do not introduce new encodings, larger probes, alignment modules or recurrent training to rescue the outcome.

## Profile and budget

First profile two batches and a representative ridge fit on the configured GB10 environment. Historical capture timings suggest a few GPU minutes; the track ceiling is 35 GPU-minutes. The root coordinator must release the GPU and approve the profiled final budget before full inference. Freeze source/configuration before test evaluation. An independent raw-metric and provenance audit follows the completed matrix.

The compact feature cache also preserves each original non-value prediction and all six targets at every delay. New scalar predictions replace only the value field, so full-joint and inherited field-gate counts remain directly reconstructible; they are never inferred by multiplying marginals. A version-local hook records the complete state at the historical normalization boundary during the same forward pass, then applies the unchanged decoder. Mechanical tests compare these predictions with independent original forwards. Individual event hashes prove partition disjointness and exact event ordering across delay/distractor conditions, in addition to batch/public-input hashes. No historical module is modified.
