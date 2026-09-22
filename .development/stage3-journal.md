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

## Main observations and supplementary control

All 39 main runs completed at source 2adc87c with 400 steps per run (167.75 s total measured run time, process peak RSS 430.79 MiB). Main raw metrics are preserved separately. Identity-initialized learned binding transfers to longer instruction sequences on 16-node graphs, while simultaneous depth/size transfer is weak and random-initialized binding fails. These are provisional observations pending final report audit.

A comparison limitation was identified from architecture review and shallow validation, before selecting any supplementary OOD result: the soft path begins with initialized entity retrieval, while graph-input ordinary Q/K begins random. A separately labeled keyed-content family adds graph-independent cosine identity logits with fixed beta=8 to none, soft and graph-input. Soft starts at lambda=16 so its next-entity structural preference can compete with the added current-entity content preference. These constants are explicit architectural priors, not OOD-tuned hyperparameters. Original main results remain unchanged. This control tests whether a stronger data-channel model erases apparent metric-specific superiority.

## Final experimental outcome

The keyed supplementary family completed nine runs in 34.76 measured seconds. Its graph-input control reaches .97135 task accuracy at depth 32 on 16-node graphs, versus .36198 for soft-keyed in that family. Both fail substantially on the joint depth-32/node-64 condition. This rules out using the main weak graph-input comparison as evidence that attention-logit geometry is uniquely superior. The added content prior is also an architectural intervention that competes with destination routing, so this is not a universal architecture ranking.

The exact one-hot attention oracle reaches 100% task and path accuracy on every clean condition. Learned soft grounding with initial lambda 4 reaches .84375 task accuracy and .671875 complete grounding trajectories at depth 32 on 16-node graphs, but only .19010 task accuracy and zero complete trajectories at depth 32 on 64 nodes. Random-initialized soft grounding does not solve binding/traversal under this answer-only budget. The primary scientific target is therefore not achieved; the implemented primitive and failure localization are useful intermediate results.
