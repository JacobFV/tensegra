# Stage 5 development journal

## Design and baseline

Baseline `3c5be1d` passed all 269 tests on the linked machine (2.41 seconds, two CPU threads). Work proceeds in `.worktrees/runtime` on `feat/runtime`; existing stages remain untouched. The detailed user specification supplies scope and authorization, including subagents and incremental commits/pushes.

The exact language and the learned benchmark are distinct deliverables. The language implements the requested immutable semantic constructs. The benchmark initially supplies segmented instructions and learns operation/reference lowering. Scheduling is observable to every matched model, not learned planning. Lexical resolution remains exact runtime semantics after the model supplies a name; this does not establish neural discovery of scope algorithms.

Independent design review identified five critical checks: no future return values in input graphs; no gold-state resets or gold-selected candidates; full observable world information for neural controls; explicit gradients/estimators at the discrete boundary; and abstentions measured separately from correct answers. Workers are implementing these as interface and test requirements.

Ruling: use a declared score-function estimator for task-only discrete lowering, and report warm-to-weak curricula separately from cold acquisition. A loss on the learned lifting head alone would not train semantic binding. Confidence evaluation includes uncertain/noisy and ambiguous references, fixed thresholds, risk/coverage and explicit deferral; no claim of spontaneous crystallization.

The provisional experiment has ten training conditions, three paired seeds, a 12-cell depth/object matrix pooled across six families, and anchor family breakdowns plus diagnostic interventions. A bounded pilot will determine a feasible fixed step/batch budget before main training. No OOD-based retuning.

## Pre-main verification and lifting pilot

The integrated source at `8aa678e` passed all 340 tests. The exact language and generated benchmark have separate checks: benchmark scalar labels now come from independently sampled leaves/manual arithmetic, and the executor must agree; no runtime output is silently reused as its own ground truth.

A shallow-only oracle-binding pilot (400 updates, one seed) isolated a lifting optimization failure: exact results and trajectories were perfect but output validation stayed at .375. One pre-main representation change supplies the learned lifting MLP with smooth RBF features centered on the bounded integer vocabulary, alongside scalar/comparison/style inputs. The identical pilot then reached .6875 validation and .703125 separate shallow evaluation. Both raw runs/configs/source snapshots are preserved; their config and data schedule match and only the model source hash differs. No OOD scores were used to select this change. Further lifting residuals will be reported rather than tuned away.

Fairness review also made all neural controls consume directed typed edge-record tokens, gave them the same semantic-lowering supervision, and added direct numeric-result supervision in their favor. Their controller is a clause transformer plus recurrent attention/GRU, not a pure decoder-only transformer. Graph-as-data and soft-bias models add their respective interventions over that shared observable input. Numeric results, actual symbolic execution, and counterfactual lowering audits remain separate metrics.

Confidence uses observable semantic equivalence groups, not gold masks. Missing-reference and type-invalid cases have no defined answer; rejection is reported separately and cannot count as successful task prediction. Training uses valid cases only, so null recognition remains an OOD stress test. No additional inference pass follows deferral in this implementation.

## Matched-root shortcut control

Before main training, inspection identified that a single object chain with globally unique field names allowed direct final-field retrieval. Lookup, alias and mixed families now include one matched decoy chain with the same shape and field/index labels but different scalar payloads. Swapping only the starting binding must change the result; remaining selector semantics are unchanged. Independent checks verify both leaf results at depths 4/16/32. This makes starting-object identity relevant without creating an unbounded number of deep distractor trees. The earlier lifting pilots remain a paired representation diagnostic on their explicitly archived earlier generator.

## Main study launch

Frozen main source: `2e95c9ff9403e9a2226a66bb0e875b762cb8736d`, clean remote checkout `~/topoformer-runtime-run`. The full suite passed 353 tests in 3.30 seconds; all ten variants passed the six-family smoke including every diagnostic intervention, and the artifact analyzer accepted those records. Main configuration is 400 updates, warm boundary 200, batch 16, three seeds, 64 examples per condition, 12 depth/distractor cells plus 8 stress conditions. It runs sequentially with two CPU threads. No model/data/training edits will alter this checkout while the study runs.

## Disclosed stronger-control supplement during main execution

While first-seed main results were becoming visible, interface inspection identified a remaining mechanism comparison: graph-data and protected-learned controls share supervised clause features but do not directly consume the learned selector/op distributions in their neural transitions. The Stage3 keyed-control result makes this distinction scientifically important. The frozen main runs remain unchanged.

A separate three-variant supplement will directly feed predicted semantic reads to graph-data and protected-learned controllers, plus a privileged known-lowering protected-neural control. This adds `P_selector @ observable_node_memory` to content retrieval and an operation-distribution read of the existing operation weights to clause state; it does not add exact arithmetic, scope resolution or interpreter transitions. Oracle lowering uses explicitly supplied semantic distributions only in its labeled condition. Default behavior, parameter shapes, initial RNG and common data schedules must match the main models before their results are reused as comparisons. The extra connection is an architectural control, not a claim that metric attention is superior.

The supplement is fixed to the same 400-update/three-seed budget and main evaluation conditions. It is disclosed post-main-launch work, not retroactive preregistration. No OOD checkpoint selection or result-dependent budget tuning is permitted.

## Completed experiments and release verification

All 30 main and nine supplemental runs completed with separate immutable training sources. Independent audits verified 1,131 recorded cells, 39 saved checkpoints and 90 cross-study pairing comparisons. The final source suite passes 359 tests. Old Stage 1–4 code/config/tests/results are unchanged. Compressed raw metrics, resolved configurations, source/checkpoint hashes, class frequencies, failure traces, heatmaps and conditional-count summaries are committed.

Supervised lowering reaches 186/192 exact results and complete trajectories at depth 32 with 64 distractor bindings; execution conditional on fully correct lowering is 186/186. Learned output accuracy is 139/192, with numeric reporting a substantial bottleneck. Stronger semantic-read neural controls remain weak. Held-out surface templates, noisy references, null recognition and reduced-supervision maintenance expose distinct failures; no pretrained-model integration follows. The report distinguishes supplied interpreter semantics from learned selector behavior.

Two analysis corrections preserve the raw experiments: undefined-answer confidence risk is null, and the nominal renamed condition is disclosed as a duplicate shallow cell. Main and supplementary analyzer hashes are separately recorded. The work establishes a useful controlled interface, not completion of robust language-to-runtime-to-latent generalization.
