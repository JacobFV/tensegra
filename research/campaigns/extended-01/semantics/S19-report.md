# S19 development result

S19 passes its fixed known-motif acquisition gate: **128/128 complete TRAIN graphs**, with **508/512, 507/512 and463/512** complete fresh DEV graphs on trained3×3,4×3 and4×4 motifs. It fails the separate heldout-combination criterion: **0/512 on3×4**, at every curve. This is one scratch initialization on inspected development populations, not confirmation or evidence of broad structural generalization.

| Added updates | TRAIN complete /128 | DEV3×3 /512 | DEV3×4 /512 | DEV4×3 /512 | DEV4×4 /512 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1,024 | 56 | 329 | 0 | 221 | 29 |
| 2,048 | 88 | 428 | 0 | 357 | 176 |
| 4,096 | 128 | 508 | 0 | 507 | 463 |

The fixed final endpoint exceeds TRAIN64/128 and52/512 in each trained cell. Weighted trained accuracy is96.972656%, increasing29.150391 percentage points since2,048. Thus the separately registered extension *eligibility* rule also passes. No extension is automatic; fresh confirmation is the more discriminating next use of compute.

All eight final teacher-forced TRAIN field accuracies are100%, accompanied by100% free-running TRAIN complete graphs. Aggregate DEV teacher-forced accuracies are type99.9924%, kind97.0735%, value99.9540%, copy92.4424%, source98.2094%, target96.2924%, role96.1125%, slot96.2000%. Gold-prefix field accuracy is a separate diagnostic and is not substituted for generated-graph accuracy.

Final valid-output counts are511/391/511/512 in3×3/3×4/4×3/4×4. On heldout3×4,294 graphs have exact presence but none has exact kind, value, copy, edge, or slot components. Its121 invalid outputs comprise108 invalid generated node references, five duplicate edges, and eight multiple pair-slot label conflicts unsupported by historical scoring. Valid outputs still fail exact graph recovery; malformed generation alone cannot explain the heldout failure. On4×4, exact presence/kind/value are508 each, copy493, edges463, slots506: remaining known-motif errors are concentrated in relations and copying. Full per-cell components, invalid reasons, macro graph F1 with invalid-zero credit, and all earlier curves remain in the JSON.

| Frozen comparison endpoint/policy |3×3|3×4|4×3|4×4|
|---|---:|---:|---:|---:|
| S19 categorical greedy |508|0|507|463|
| S18 original raw |1|0|7|0|
| S18 original historical |1|0|80|10|
| S18 original matched |22|0|109|10|
| S18 contextual raw |0|0|0|0|
| S18 contextual historical |7|0|79|15|
| S18 contextual matched |43|0|68|16|
| S18 ten-pass raw |0|0|0|0|
| S18 ten-pass historical |0|0|2|0|
| S18 ten-pass matched |1|0|7|0|

Every comparison uses the same512 event IDs and exact targets per cell, with all paired gains/losses retained. Against the strongest matched contextual reference, S19 gains465/439/449 complete graphs and loses0/0/2 on the three trained cells. Against matched original it gains486/398/454 and loses0/0/1. All methods score zero heldout. S19 has no edge calibration; the gain is present against raw and both frozen calibrated reference policies.

The alternative learner changes parameters (62,677,315), objective, contextual encoder, sequential output interface and supplied typed serialization together. It starts from scratch, whereas historical actors have196,608 inherited presentations. All share32,768 incremental mixed presentations and the construction stream; S19 has no sampled negative-edge queries. Its3,229,392 training records,1,720,320 tokens,988,008 nodes and2,208,616 edges are explicit exposure counts, not matched FLOPs. These results establish substantially better known-motif acquisition for this learner, not an isolated causal claim about decoder architecture or the absence of information in the old actor.

The successful main charged321.10 seconds (outer321.09; optimizer185.947973; child319.667298), below the1,000-second cap. Together with profile14.18, S19 development has charged335.28 seconds. This actual timing replaces prospective cost estimates; unused cap is not consumed compute. No extension or confirmation has run.

Analysis source4b8901fc consumed the complete main archive220df6f4 with frozen config07ccff2d and SHA-bound S18 references. It checks all source/receipt/artifact links, exact exposures, complete curves, teacher-forced arithmetic, tensor components and event/target identity. Independent strict generated-record codec replay and checkpoint/AdamW auditing are separately required. The initial analysis commands encountered missing worktree paths; correcting source/reference CLI locations changed no data, source, formula or policy. `s19-analysis.json` preserves the exact inventory and all outcomes.

Recommend a separately registered S20 confirmation: three scratch S19 seeds paired with three genuinely independent fixed S12 parent lineages701/702/703, each furnishing both original and contextual continuation arms. Keep all32,768 incremental presentations, fixed endpoint, TRAIN128 global calibration rules and public inputs matched; disclose inherited histories. Use the reserved fresh512/cell confirmation only after freezing the complete protocol, with all cells and seeds reported regardless of outcome. Do not reinterpret known-motif gains as3×4 transfer. No additional same-seed exposure or calibration search is needed before this discriminating replication.
