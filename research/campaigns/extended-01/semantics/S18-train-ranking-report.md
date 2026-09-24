# S18 actual-TRAIN residual: strong pair ranking, imperfect graph conjunction

CPU-only analysis of the archived4096actualTRAIN128 calibration scores/labels and full packed predictions. Inputs are SHA-checked against the pinned full analysis. The existing thresholds remain fixed: no new threshold search, refit, model forward, DEV diagnostic selection or confirmation inference. These labels fitted the calibration policy, so this is an optimistic in-sample diagnostic. Scores cover predicted-present pairs; full packed graph conjunction is checked separately.

| TRAIN3×3,64graphs | Original | Context |10pass |
|---|---:|---:|---:|
| Complete / exact-edge graphs |5/5|6/6|0/0|
| Fixed relation errors |241|190|557|
| Median errors per graph |3|2|7|
| Graphs with0 /1 /2–5 /6+errors |5 /6 /40 /13|6 /13 /38 /7|0 /1 /17 /46|
| Reference-edge AUROC |.999242|.999488|.997900|
| Reference-edge average precision |.957046|.971089|.890182|
| Reference FP / FN |52 /136|61 /68|56 /306|
| Argument average precision |.986773|.990666|.951840|
| Argument FP / FN |41 /11|41 /18|86 /85|

Each arm has582,777active pair×relation decisions in this cell, overwhelmingly negative. Context's190errors are only0.0326%of these decisions, yet58/64graphs fail;57fail on edges alone and1on kind+edges. All6error-free active-pair graphs also pass the full graph check. The original has58edge-only failures and1copy+edge failure. Thus the low exact-graph count coexists with strong but imperfect relation discrimination; it is not evidence that the learner obtained no useful mapping. High AUROC is partly a class-imbalance-friendly summary; AP and explicit FP/FN counts expose remaining overlap. AUROC below1means rankings are not perfectly separated, so another scalar threshold cannot make every positive/negative pair correct. No independence assumption is used to explain how errors combine across graphs.

Context improves reference ranking and removes68reference false negatives while adding9false positives on these actualTRAIN3×3 examples. Argument errors rise52→59. The remaining few errors per graph explain why that improvement only changes full TRAIN success5→6. On known-arity TRAIN cells, context has83errors/32graphs(4×3)and150/32(4×4), with complete3and1; original60and175errors with complete7and2. Better aggregate pair accuracy does not guarantee better error concentration or whole-graph acquisition.10pass is substantially worse across these panels.

This sharpens the previously stated poor-TRAIN branch: the unresolved question is reliable **joint** node/reference/ordered-edge construction, not merely improving already high independent relation ranking or choosing another calibration population. It supports retaining the negative S18 gates and considering the separately authorized variable-length graph-decoder development comparison. It does not establish that autoregression will solve the problem, that the current edge head is intrinsically incapable, or that output allocation alone caused these failures.

`S18-train-ranking.py` computes tie-aware AUROC and score-bucket average precision without fitting, with perfect/tied synthetic rank checks. `s18-train-ranking.json` retains every role, cell, fixed cutoff, errors, per-graph histograms, full failure patterns and input hashes for independent review. Absent-positive/negative roles have undefined ranking statistics rather than fabricated scores.
