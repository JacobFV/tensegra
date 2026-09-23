# S11: one late learning-rate reduction materially improves complete graph recovery

The fixed196,608-presentation endpoint recovers122/512 development graphs under TRAIN-calibrated decoding, versus0/512 for the matched constant-LR continuation. Raw decoding recovers13/512 versus0/512. This is a single development-seed optimization result, not an untouched confirmation or broad semantic transfer result.

| At196,608 presentations | Constant1e-4 S04 | Reduced1e-5 S11 |
|---|---:|---:|
| Complete DEV, raw |0/512|13/512|
| Complete DEV, TRAIN-calibrated |0/512|122/512|
| Copy accuracy (graph macro) |.988349|.996019|
| Typed-edge F1, calibrated |.919315|.979922|
| Ordered-edge F1, calibrated |.937973|.986786|

S11 starts from the exact S03 parent at131,072 presentations. Its intermediate163,840 checkpoint recovers77/512 calibrated graphs; the declared endpoint improves to122/512. The final128-example TRAIN diagnostic recovers6 raw and29 calibrated complete graphs, with.999421 copy accuracy. The prespecified development promotion criterion passes, while most full graphs remain imperfect. No checkpoint was substituted for the endpoint.

The sole training change is overriding restored AdamW group LR from1e-4 to1e-5. Weights, optimizer moments/steps, sample/pair RNG, ordering, curriculum, public inputs and all objectives are preserved. Initial raw/calibrated outputs and thresholds exactly replay; the first sampled batch/pairs agree with the constant-LR comparator. LR also scales AdamW's decoupled decay update, so this is not a pure gradient-noise intervention. Global CPU/CUDA RNG was not archived; the active sampling generators were, and dropout is zero.

On the same TRAIN predicted-present pair support (9,970 positives,1,930,683 negatives), natural edge BCE falls from.016588 at the parent to.002074 at the S11 endpoint; the constant-LR endpoint is.023445. Balanced BCE falls from.015155 to.001252, versus.013788 constant. These archived-score diagnostics differ from the sampled optimization objective; support and per-relation losses are retained. They do not establish a unique causal mechanism beyond the paired LR intervention.

All actor inputs remain public English text and copy inventory. No programmed reference/schema decoder is included in these primary outcomes. Separate frozen S10 bookkeeping diagnostics are reported by their owner and cannot replace these scores. The corpus contains8,192 equality/reference patterns but only two ordered tree motifs at depth3. No new renderer, structural-depth or lexical transfer is established. Confirmation data remain uninspected.

Width1,024;8 workspace rows;57,853,781 parameters. Added65,536 presentations,196,608 cumulative,8,192 distinct constructions. Full process occupancy649.4482 seconds, optimizer495.5331 seconds. Peak allocated CUDA1,224,941,568 bytes, maximum process RSS3,837,948KiB; these are different quantities. Final checkpoint SHA3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e is retained in the immutable remote checkpoint and archived manifest.

Raw predictions, targets, calibration float arrays, losses, source/config/data/checkpoint hashes and receipt: `research/results/campaign-01/semantics/s11-lr-decay-n8192-196k-dev201`. Source freeze14f0293; protocol8e3316a. Independent audit1cd6d22 reproduces3,840 graph decisions,39 TRAIN thresholds, checkpoint bytes and inherited optimizer/model/sampling state except the declared LR. S09 remains profiled and unrun; this result supports prioritizing confirmation of a minimal optimization repair before an added architectural path.
