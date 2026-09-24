# S17: matched-TRAIN calibration helps, but leaves graph construction failures

The frozen alternative global policy raises mixed-arm trained3x3 complete graphs from1to22/512, while heldout3x4 remains0/512. This exploratory, S15-informed diagnostic supports a calibration-population contribution to the residual. It does not remove the residual, demonstrate broad structural transfer, or change the failed S15 gates. No training, checkpoint selection, DEV fitting, confirmation or additional threshold search occurred.

Each endpoint used its preselected actualTRAIN128 population, the same inherited threshold-fitting algorithm, and one global policy across all2048DEV. Every raw prediction and target exactly replayed the original S15 archive; model state hashes stayed unchanged. CPU analysis guards and synthetic fixtures passed. Independent full-main audit is pending.

| Endpoint | Cell (512 each) | Raw | Historical calibration | Matched TRAIN calibration |
|---|---|---:|---:|---:|
| Control | 3x3 | 0 | 0 | 0 |
| Control | 3x4 | 0 | 0 | 0 |
| Control | 4x3 | 109 | 231 | 249 |
| Control | 4x4 | 3 | 45 | 43 |
| Mixed | 3x3 | 1 | 1 | 22 |
| Mixed | 3x4 | 0 | 0 | 0 |
| Mixed | 4x3 | 7 | 80 | 109 |
| Mixed | 4x4 | 0 | 10 | 10 |

Historical→matched transitions: mixed3x3 gains21and loses0; mixed4x3 gains47and loses18; mixed4x4 gains5and loses5. Control4x3 gains30and loses12; control4x4 gains8and loses10. Full transitions, exact component counts, macro graph-F1 averages and every role's signed FP/FN/slot-error changes are in `s17-paired-analysis.json`.

Mixed3x3 still has exact presence512, kinds478, values512, copy477, slots507, and edges22. Threshold changes affect edges only. `refers_to` false negatives fall1844→965 while false positives rise228→468; argument false negatives fall435→169 while false positives rise262→318. Spurious field:fact edges fall1182→11 and field:substitution376→4. Thus improved suppression and recall coexist with substantial remaining relation errors; these results do not establish that a different architecture is necessary or sufficient.

Both selected calibration populations lack positive binds, binding_scope, field:substitution, field:fact and field:scene edges. The unchanged algorithm chooses the largest observed predicted-present calibration score for a role without positives. This is fitted finite-sample behavior, not a supplied schema prohibition: DEV false positives can remain. Gold prevalence and absent-role metadata, original/new global thresholds, scores/labels and active-pair masks are retained for audit. Roles absent in one DEV cell need not be absent from the entire calibration mixture.

The full paired main completed exit0 within300seconds: outer100.73s (charge100.74), inner100.665215025s. Paired calibration7.235087s, DEV49.707825s and export24.194096s were measured separately. The220.291s predeclared forecast scaled all export and verification32× conservatively; no recipe changed after the mechanical profile. GPU is free. No successor has been launched.
