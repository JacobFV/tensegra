# R11: optimizer localization on the fixed R10 feature population

Prospective development only. No GPU release. This is a single bounded optimizer comparison, not an automatic retry of the failedR10diversity screen. Its populations are reused development data, including previously inspected validation outcomes; a promoted recipe requires fresh confirmation.

## Why this diagnostic, and what it cannot establish yet

R10's16k→64kdiversity change reduced late balanced validation errors88→63, but the64kconsumer still misclassified479of65,536fitting examples at16updates.477of those are float-valued. Float+7.5 has110/1,261fitting errors and11/128validation errors. Both ingestion grids are perfect. This motivates checking optimization of the existing linear consumer, not replacing its phase, encoding or memory.

The64kcalibration minimum fluctuated1,023→1,010→1,021of1,024atsteps2,700/3,000/3,600; late minibatch-loss spikes are visible. These observations do not prove excessive learning rate or information loss. Minibatch composition, optimization conditioning, representation accessibility and finite exposure remain alternatives. R08 varied learning rates for a nonlinear residual consumer, so it does not settle this linear-head question.

## Fixed controlled comparison

Use exactly the immutableR10-64kfeatures, train normalization, scalar targets and original33,825-parameter linear readout. The1024-wide backbone and all non-value outputs remain frozen. No new capture, horizon, event population, loss, regularizer, phase feature or architecture is introduced.

First replay the original3,600AdamW updates atLR.003from the original Stage11scalar head, using the same batch size256and CUDA sampling RNG. Require **bit-exact** final weights/bias, mean/scale, sampled-row bitset and complete loss sequence against the archivedR10endpoint. If any differs, stop before either fork. The historical optimizer state was not saved; its state is reconstructed through this exact deterministic replay, not falsely described as loaded from an old checkpoint.

Save the replayed model, full AdamW state, RNG state and normalization. Fork two arms for exactly3,600additional updates:

- constant: LR.003;
- decay: LR.0003.

Restore identical AdamW moments/step counters and RNG for both. The sole between-fork intervention is learning rate. Additional sampled indices must match exactly; archive the SHA256of the full ordered little-endianint64index stream and final RNGstate for each fork, and require both hashes equal. The new replay index stream is also hashed, but R10did not archive an index-order hash: historical equivalence is verified through exact losses/parameters/normalization/visited membership plus the frozen deterministic sampling recipe, not an independently archived historical order hash. Both receive921,600extra presentations, reaching7,200total updates and1,843,200presentations; the frozen3,600endpoint remains an explicit reference. Save full optimizer/RNG state at every endpoint going forward. The fresh forward output still reconstructs the scalar from workspace features; it is not exact copying.

## Evaluation and fixed decisions

Reuse the fullR10development matrix: unchanged-mixture calibration1,024and validation4,096with2/8distractors, delays0/1/2/4/8/16; separate balanced calibration/validation grids128per52legal strata at0/1/16with8distractors. Delay32is absent and may not enter this experiment. Original-mixture and grid populations remain separate.

Record calibration curves every300extra steps, but never select an intermediate checkpoint. Only the fixed7,200endpoint may advance. Require all of:

1. every mixture calibration cell≥98%;
2. worst balanced calibration stratum≥122/128;
3. at16updates, fitting scalar accuracy≥99.8% (≥65,405/65,536).

These establish a development acquisition/calibration screen, not uniform98%reliability or a historical gate pass. Report both outcomes even if neither advances. If both advance, the candidate for a separately registered confirmation is the better calibration-grid minimum, then mixture minimum, with constant winning exact ties. To specifically credit lowerLRrather than additional exposure, require decay's calibration-grid minimum exceed constant by at least4/128; otherwise report common continuation effects or a mixed result. Validation is reused descriptive evidence and cannot change selection.

Report per-delay/value/type/primitive errors, distances and signs, fitting-versus-reused-validation gaps, full/non-value joint accuracy and paired fixes/breaks. Confirm every non-value prediction remains byte-identical. Frozen reference predictions must replay archivedR10exactly. No broader composition consumer is replaced automatically.

## Budget, provenance and review

Mechanical profile replaysR10's2,048-event/100-updateprofile endpoint exactly, then adds100updates per arm. It uses real1024-dimensional frozen features; small CPUtensor tests are labeled mathematical fixtures. Measure cache verification/decompression/preparation, exact replay, paired fitting, endpoint evaluation and export separately; full outer occupancy includes all phases. Proposed profile cap60seconds. Main allocation follows measured profiling; preliminary expectation60–120seconds with a conservative180-second cap subject to root scheduling.

All old recipes remain recorded. Stop after this fixed diagnostic rather than automatically extending optimization again. Independent review should challenge the inference from fluctuating losses, exact-replay guarantees, calibration-only decision, complete matrix, immutable inputs and optimizer/RNG pairing. Independently replay endpoint predictions from frozen features and compare raw counts. A failed replay or timeout remains a recorded result and is not silently restarted.
