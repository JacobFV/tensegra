# S14: unseen arity-three motifs

All six frozen S12 endpoints score **0/1024 complete graphs**, with both raw and original TRAIN-calibrated policies. The matched frequency baseline also scores zero. The diagnostic changes ordered motifs, not depth: 525 four-fact and 499 three-fact constructions, all depth three, using the existing n-ary English renderer and supported argument slots. No thresholds were fitted or selected on this population. Full wrapper occupancy was 97.600552094 seconds, exit 0, within the 420-second allocation.

CPU reconstruction from archived compact predictions reproduces every row's complete-graph metric. `S14-analyze.py` produces `s14-motif-main/analysis.json`; `S14-localize.py` produces `localization.json`, including per-relation, canonical-position, predicted-arity, and motif/node-size summaries. No model calls or new examples were used.

| Seed / arm | Exact presence | Exact node kinds | Exact copy | Mean copy accuracy | Typed edge micro F1 | Ordered edge micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| 701 constant | 519 | 0 | 0 | .647 | .536 | .270 |
| 701 decay | 521 | 0 | 0 | .653 | .557 | .276 |
| 702 constant | 479 | 0 | 0 | .593 | .517 | .256 |
| 702 decay | 520 | 0 | 0 | .557 | .528 | .294 |
| 703 constant | 494 | 0 | 0 | .584 | .553 | .290 |
| 703 decay | 516 | 0 | 0 | .652 | .583 | .280 |

Counts have denominator 1024; edge F1 columns use the calibrated policy, and node/copy predictions are shared by both policies. These component summaries show distributed failure: replacing copy alone or all node attributes with gold still yields zero complete graphs; replacing structure jointly (presence, typed edges, and ordered slots) also yields zero. Even gold structure plus gold copy leaves zero, and gold structure plus gold kind/value leaves zero. Replacing everything except presence recovers exactly the presence counts above; the all-gold check recovers 1024/1024. These are privileged posthoc intervention results, not learned inference or achievable performance estimates.

Predictions retain evidence consistent with the old shape prior. An argument edge labeled slot 3, which cannot occur in the arity-three targets, occurs in all 1024 raw predictions from every model and in 1018–1024 calibrated predictions. (List item slot 3 remains valid and is excluded from this count.) The non-entity/non-scope predicted kind sequence matches one of the two old arity-four sequences in 191, 175, 102, 149, 140, and 352 cases respectively, and matches the new arity-three sequences in zero cases. Kind sequences ignore edges and slots: this supports template bias but does not establish that every output is an exact old template graph. Early canonical positions retain high type/copy accuracy; later position errors mix alignment changes with recognition errors and do not isolate a causal module.

Compact archival limits matter. Edges incident to predicted-absent nodes were discarded, and slot labels for pairs absent from both edge policies were not retained. Therefore a presence-only intervention that activates nodes, or an edge-only replacement requiring missing slots, cannot in general be reconstructed exactly. We report joint structure replacements supported by the archive, not invented finer oracle scores.

The positive S12 calibrated advantage on known motifs does not establish shape transfer. S14 locates failures in both structure and node attributes/copy, with evidence of the historical shape prior. The fixed S13 language acquisition comparison remains separate and unchanged. Any subsequent shape-exposure experiment needs new construction namespaces and must distinguish acquisition of a trained motif from evaluation on an untrained combination.
