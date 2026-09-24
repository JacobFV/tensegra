# A12: mixing at both reads, with substantial head disagreement

Frozen model-score hardening improves moderate-depth task accuracy but does not repair deep execution. Removing destination-read mixing has a larger observed deep-task effect than removing record-key mixing alone. Even both interventions solve only124/512 deep events and4/512 complete suffix trajectories. These are supplied-kernel interventions, not an acquired-soft competence pass or evidence for a preferred architecture.

| Policy | IID task | Moderate task | Joint task | Joint complete suffix |
|---|---:|---:|---:|---:|
| Unchanged | 508/512 (99.22%) | 479/512 (93.55%) | 79/512 (15.43%) | 0/512 |
| Record hard only | 503/512 (98.24%) | 494/512 (96.48%) | 82/512 (16.02%) | 0/512 |
| Destination hard only | 505/512 (98.63%) | 493/512 (96.29%) | 115/512 (22.46%) | 0/512 |
| Both hard | 504/512 (98.44%) | 497/512 (97.07%) | 124/512 (24.22%) | 4/512 |

All four policies share512 fresh events at each shape. Relative to unchanged, record/destination/both hardening fix17/21/20 moderate errors while breaking2/7/2 prior correct answers. On joint events they fix9/48/52 and break6/12/7. On IID they fix zero and break5/3/4. Moderate complete-suffix counts are341/355/340/372; improved final task does not imply consistently repaired intermediate values.

## What the two reads reveal

Unchanged joint mean-head argmax path accuracy is99.02%, but individual head-node-step argmax correctness is73.87% at the record read and73.47% at the destination read. Only7.67% of joint node-steps have all eight heads select the same record destination;7.53% have all heads select the same destination-read node. Per-head record correctness ranges65.05–79.17%, so this is not one isolated anomalous head. These metrics contradict interpreting a correct mean-head path as every head executing an exact pointer. They do not prove each head was intended to implement that pointer rather than a distributed computation.

Joint record target mass is.4245 before hardening and.7387 after hardening. Record hardening makes returned-key MSE against the selected record's key exactly zero, and raises destination soft target mass from.5509 to.6639 (maximum mass.6596→.8985). Thus mixing in the first read measurably affects concentration in the second read. It does not improve record-score argmax correctness; hard choices can preserve the wrong destination and disagreement across heads. Joint destination argmax correctness after record hardening equals record correctness73.87%.

Destination hardening makes payload MSE against each head's own selected current payload exactly zero. On moderate events, unchanged relative payload mixing MSE is.02404; record hardening reduces it to.001880, while destination/both hardening remove it by construction. Conditional on correct destination argmax, local payload MSE falls from.27780 unchanged to.04016 record-hard and zero destination/both-hard. This local removal helps some final predictions, but cannot remove incorrect head selections, cross-head mixing through the learned update, or prior errors already present in the current payload state.

On joint events the global relative mixing MSE is already small (.0006021 unchanged), despite task failure. Across reverse execution steps1/8/16/32, unchanged selected-payload energy rises.983/16.90/79.98/476.89 while own-argmax local MSE falls.161/.138/.00275/.00000217. The reference is the model's evolving payload, not a gold hidden state. Small late-step local distortion therefore does not demonstrate information preservation or correct decoded values. No causal attribution to hidden-state collapse is established by these summaries alone.

The observations localize finite mixing at both reads and expose a separate per-head selection/disagreement boundary. Their total intervention effects are not additive, and no single-stage sharpening repair explains deep failure. Keep the unchanged A11 gate failure and all prior endpoints; no training or follow-up search follows automatically.

## Integrity, cost and audit

Frozen source71f1971; input checkpoint SHA256 `f78aa380ddb2bb7e18234f73c5de751f5ebea2d457211ccc35591c62dd48609f`; initial/final tensor hash identical. Zero optimizer updates. Main config SHA256 `b62eae8650a9f8d1e813edf3c09ad2c73d96ef7bd30b512f93d60334a8aa0518`. Four prior mechanical tests verified unchanged exact parity including width1024, frozen tensors, own-argmax mixing identities, and unchanged outputs under posthoc path-label perturbation. Hardening uses model scores only.

Full-process occupancy114.503450415seconds, atomic remote receipt; internal113.072818358seconds; CUDA peak1,479,944,192bytes. Timings include instrumentation and are not bare inference comparisons. GPU verified free before analysis. Cumulative attention accounting1530.085226674seconds includes A08 profile's explicitly uncertain60-second upper bound.

Raw twelve cells and shared public arrays: `research/results/campaign-01/attention/a12-main/results`. Reproducible paired and reverse-step summaries: `research/tools/campaign_a12_analysis.py` and `a12-main/paired-analysis.json`. Independent raw audit requested; no additional GPU run launched.
