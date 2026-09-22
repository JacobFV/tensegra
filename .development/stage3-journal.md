# Stage 3 journal

## 2026-09-22 — scope and implementation start

Baseline main aed7404 passes all 143 tests on gb10-direct (two CPU threads, 1.96 s). Work isolated on feat/grounding. No stage 1/2 source is being modified.

The important modeling limitation is explicit: relation instructions externally schedule one recurrent block per hop. We test whether a learned state can keep its binding while traversing, not whether it discovers a program counter. A fixed shallow stack could not support an honest depth-32 comparison without an increased computation budget.

The first architecture uses raw random entity identity keys as a coordinate prior in query/memory residuals. Independent trainable query/key grounding projections can start aligned to that coordinate system; a random-initialization ablation is necessary to expose how much this initialization contributes. Thus even successful identity-initialized traversal will not establish unsupervised discovery of arbitrary symbolic identities.

The graph-as-data baseline is intentionally strong: it uses fixed cosine matching of supplied identity keys and place graph-neighbor messages into token values, while retrieval uses ordinary attention. Its fixed identity matching is a disclosed alignment prior. We must not weaken this control to obtain metric-specific superiority.

Raw metrics, diagnostic trajectories and resource measurements will be committed. No result has yet been observed for Stage 3.

## Pilot and controlled-study launch

Source 48d3ca3 passed 197 tests remotely. The end-to-end two-step smoke completed all 13 variants. An IID-only seed-0 pilot used 100 training steps, then stopped: validation answer accuracy was soft .53125, random initialization .1953125, none .203125, graph-input .296875, known .8203125, hard .3046875. These are feasibility observations, not final evidence. No OOD conditions were included in the pilot. The preregistered main budget remains 400 steps across three seeds and 13 variants; no hyperparameters were selected against OOD results.

Remote GitHub DNS briefly failed; an exact Git bundle transferred over the authorized direct SSH link preserved committed source provenance. No local training was performed.
