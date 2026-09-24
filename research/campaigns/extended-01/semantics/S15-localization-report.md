# S15 posthoc localization: trained motif allocation succeeds, edge decisions dominate

The new trained3×3 motif has a much narrower residual failure than the zero-shot S14 diagnostic. Mixed prediction has exact presence512/512, exact scalar values512, all ordered slot labels507, node kinds478, and identity copy477; only1 has exact typed edges. Exactly455 examples fail solely on typed edges. Gold-edge replacement alone yields456 complete graphs, edges+slots457, while replacing any single presence/kind/value/copy/slot field leaves1. Gold `refers_to` alone yields113 raw or76 calibrated complete graphs, so reference edges are important but not the sole relation problem. These privileged replacements localize archived errors; they are not learned inference or a performance claim.

Control on3×3 fails all six component-exact tests in all512 examples and systematically overpredicts node count. Mixed has exact node count and presence in every3×3 example; its predicted non-entity/non-scope kind sequence matches the3×3 sequence observed in DEV in480/512. Consequently node-count or old-shape allocation does not explain the mixed trained-cell complete-graph failure. Sequence matching alone ignores edges/slots and is not proof of a copied graph template or causal mechanism.

| Mixed3×3 relation | Raw false positive / negative | Historical-TRAIN calibrated false positive / negative | Gold edges |
|---|---:|---:|---:|
| refers_to |5745 /34|228 /1844|6656|
| argument |1000 /1|262 /435|6656|
| contains |79 /10|34 /22|10240|
| declares |80 /11|22 /44|2866|
| item |13 /0|2 /11|1536|
| field:fact |0 /0|1182 /0|0|
| field:substitution |0 /0|376 /0|0|
| binding_scope |0 /0|102 /0|0|
| binds |0 /0|50 /0|0|
| field:scene |0 /0|31 /0|0|

Raw and historical-calibrated complete counts both remain1. Calibration removes many positive reference/argument errors but introduces missing true edges and false predictions in relations absent from these targets. This motivates testing calibration-population mismatch before changing the architecture; stored binary decisions do not establish that another threshold can separate the scores. No alternative cutoff or margin is inferred from them. Slot labels on gold pairs are already exact for507graphs; calibrated fourth-argument-slot errors occur in zero3×3 graphs versus11 raw graphs.

The historical English TRAIN128 calibration panel itself scores control12raw/42calibrated versus mixed0raw/14calibrated complete graphs. These are old arity-four training examples, not a new-shape TRAIN-fit panel. Their improvement under calibration does not establish suitability of their threshold policy for the newly trained motif. All old policy results and S15 failed criteria remain unchanged.

The held-out3×4 combination has a different and broader boundary: mixed exact presence506/value511 but zero exact kinds, copy, slots or typed edges. No single-field or tested partial joint replacement restores a complete graph. The4×3 DEV kind sequence appears in96 predicted sequences, and the3×4 sequence in zero; this is descriptive sequence evidence, not a complete-template reconstruction. Therefore successful allocation on the trained3×3 motif does not establish allocation transfer to the held-out combination.

`S15-localize.py` reproduces factual complete metrics for all4096 endpoint examples under both policies, then records component conjunctions, all single-field and declared joint interventions, relation replacements/FP/FN/slot errors, count pairs and kind sequences in `s15-localization.json`. Unlike S14 compact records, S15 retains unmasked raw/calibrated edge decisions and every pair's slot label, making presence-only/edge-only/slot-only replacements exactly recoverable. Logits and alternative thresholds are absent; no new model call or fitting was used.

The smallest next experiment is S17, an explicitly S15-informed frozen-checkpoint calibration-population diagnostic: identical threshold algorithm and128actual TRAIN examples per arm, deterministic mixture matching, one global relation policy applied to all four existing DEV cells. It tests output calibration before the more costly S09 contextual-read intervention. It cannot retroactively turn S15 into a pass or establish a newly learned capability. If edge failures persist under the declared alternative calibration, the existing contextual-read path remains a separately profiled learned-edge acquisition candidate; no supplied schema or exact equality decoder is folded into that claim.
