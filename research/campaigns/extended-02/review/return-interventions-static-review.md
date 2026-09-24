# Return-fault layer static scope review

Source `afdee357`, reviewed without execution. Canonical protected records remain unchanged; actor-visible and actually retrieved/used views are faulted. Target selection uses public chronological ordinals and optional primitive kind, not hidden successful bindings. Independent world commit/path checks remain active. This is a deliberate memory-contract violation, not a solver correctness test.

Interpretation depends on the consumed field. Default wrong-value path(2,) changes a subset's reported weight, but `use_return` consumes assignment indices and the simulator recomputes physical weight. No outcome change is required. Label this summary sensitivity, or separately register an assignment-index corruption when testing dependence on bindings. Default stale first-return targets a subset whose immutable semantics intentionally do not require current route version; target a route for stale-route rejection tests.

Swapping two same-primitive payloads can be unchanged or type-valid yet semantically incompatible with the original draft's index mapping. Keep eligible, applied, actually changed, original-goal-correct and supplied-record-valid populations separately. The scripted two-return preparation is supplied orchestration and must not be counted as a learned sequence. Wrong payloads intentionally retain a prior certificate flag: label this post-validation corruption rather than claiming independent validation accepted the corrupted instance.

No blocking leakage issue identified by static inspection; no execution or GPU claim.
