# Stage 11 prospective frozen-actor fresh construction evaluation

Declared before fresh model inference. This is conditional on all three prescribed public-text acquisition seeds reaching 8/8 complete canonical recovery at the final 4000-update checkpoint. No new training is authorized by this document.

Hypothesis: the interface acquired from eight English unification examples transfers to fresh alpha-distinct constructions from the same unchanged generator and known renderer. This evaluates transfer from a tiny training set, not adequacy of broad public semantic supervision.

## Data and observability

Generate 512 unique alpha-normalized unification constructions at difficulty .5, starting seed 11100000, maximum 10000 candidates. Exclude exact alpha keys of the eight training constructions and previous accepted candidates. Keep the original frozen finite value vocabulary, node capacity 128, ordered-slot capacity 32, and public-copy rules. Reject only unrepresentable finite non-copy values, capacity violations, contradictory identical public strings, lexical hash collisions, or noncopyable required identities; report all attempted/eligible/rejected denominators and reasons. Do not reject novel copyable names or tokens. Report lexical novelty explicitly. Labels-only audit precedes prediction; no prediction-based selection. Alpha-disjointness is relative to Stage 11 training, not an assertion that no historical Stage 8 artifact ever contained that construction.

Only public English text and its derived token copy inventory enter the frozen actor. Gold node counts, edge identities, canonical order, and target values remain evaluation labels. The canonical compiler convention is supplied.

## Frozen policy and gate

Use final-update-4000 checkpoints for seeds 30/31/32. Keep each seed's final thresholds fitted on its eight TRAIN graphs. Never call the training calibration evaluator with the fresh examples. Never regenerate or extend the finite vocabulary from fresh labels. Presence threshold remains zero. Report raw zero-logit edges and frozen calibrated edges separately on identical examples.

Primary restricted known-renderer transfer gate: **complete canonical graph accuracy >95% in every seed**, over all 512 eligible unique examples (at least 487/512). Report typed and ordered edge precision/recall/F1, conditional slot accuracy, node/type/copy metrics, support by graph size, lexical novelty, and component-replacement diagnostic ceilings. Missing coverage or a failed seed fails advancement. No renderer or lexicon claim follows from this gate.

## Budget and stopping

Inference only, three frozen actors x512 examples. Estimate under two GPU minutes from fixed evaluation timings; hard cap five GPU minutes, subject to parent GPU release after fixed acquisition and review. No optimizer, parameter changes, threshold calibration, selection, or additional training. Save compressed predictions/targets with stable graph identities, frozen threshold/checkpoint/source/config hashes, elapsed time and resource counters. Stop after this declared matrix; failure does not authorize training expansion. Further heldout renderer or alpha-renaming experiments require a separately frozen registry and authorization.
