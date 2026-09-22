# Stage 5 development journal

## Design and baseline

Baseline `3c5be1d` passed all 269 tests on the linked machine (2.41 seconds, two CPU threads). Work proceeds in `.worktrees/runtime` on `feat/runtime`; existing stages remain untouched. The detailed user specification supplies scope and authorization, including subagents and incremental commits/pushes.

The exact language and the learned benchmark are distinct deliverables. The language implements the requested immutable semantic constructs. The benchmark initially supplies segmented instructions and learns operation/reference lowering. Scheduling is observable to every matched model, not learned planning. Lexical resolution remains exact runtime semantics after the model supplies a name; this does not establish neural discovery of scope algorithms.

Independent design review identified five critical checks: no future return values in input graphs; no gold-state resets or gold-selected candidates; full observable world information for neural controls; explicit gradients/estimators at the discrete boundary; and abstentions measured separately from correct answers. Workers are implementing these as interface and test requirements.

Ruling: use a declared score-function estimator for task-only discrete lowering, and report warm-to-weak curricula separately from cold acquisition. A loss on the learned lifting head alone would not train semantic binding. Confidence evaluation includes uncertain/noisy and ambiguous references, fixed thresholds, risk/coverage and explicit deferral; no claim of spontaneous crystallization.

The provisional experiment has ten training conditions, three paired seeds, a 12-cell depth/object matrix pooled across six families, and anchor family breakdowns plus diagnostic interventions. A bounded pilot will determine a feasible fixed step/batch budget before main training. No OOD-based retuning.
