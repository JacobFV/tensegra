# Supplied / learned / generalized ledger (extended-04, maintained per experiment)

| Mechanism | Supplied | Learned | Generalization shown |
|---|---|---|---|
| depworld world, solvers, validators, applicability relations (d1) | yes (extended-03) | - | - |
| X1 bootstrap policies | dep_reuse teacher supervision (public-only) | action choice | IID sealed (extended-03) |
| progress diagnostic v1 (signature, classes) | yes (hand-specified, public-only) | - | - |
| recovery rules R-mask / R-sample | yes (hand-specified) | - | - |
| A2 rehearsal (a2-reh, a2-imit) | dep_reuse teacher actions on the policy's own sampled states (public-only) | policy update | (A2 screen) |
| A2 entropy regulation | entropy target = bootstrap entropy (measured) | policy update under the constraint | (A2 screen) |
| probeworld env, prior table, outcome model, event invalidation subset | yes | - | - |
| exact DP labels Q*, V*, optimal sets, case/switch/stage/dependency (oracle, offline) | yes (training targets only; L1–L4) | - | - |
| belief tracking, dependency validity, when to switch, strategy vs ρ_k (probeworld) | - | learned (L0 from utility only; L1–L4 with supplied supervision) | B1 + F2 confirmed (fresh seeds and worlds): held-out price and k yes; held-out condition combinations no |
| A1/F1b: recovery rule R-mask over frozen policies | rule and diagnostic supplied | - (no learning) | A1: collapsed P1-RL recovered to .87–.92 success; F1b: fresh lineages r3–r5 on sealed 180M worlds, stagnant steps −74–86% at no cost |
| A2 RL variants (anchor/rehearsal/critic/entropy/masked) | recipes supplied | RL policy updates | none deployable: sampled-mode gains only (screening, one lineage) |
| Track C metacontroller | telemetry (89 features), recovery rule, branch labels (oracle simulation, training only) | appraisal GRU (30k parameters) and intervention choice | calibrated on fresh worlds (C-H4); no utility gain over rules (C-H1 fail) |
