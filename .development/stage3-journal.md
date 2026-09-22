# Stage 3 journal

## 2026-09-22 — scope and implementation start

Baseline main aed7404 passes all 143 tests on gb10-direct (two CPU threads, 1.96 s). Work isolated on feat/grounding. No stage 1/2 source is being modified.

The important modeling limitation is explicit: relation instructions externally schedule one recurrent block per hop. We test whether a learned state can keep its binding while traversing, not whether it discovers a program counter. A fixed shallow stack could not support an honest depth-32 comparison without an increased computation budget.

The first architecture uses raw random entity identity keys as a coordinate prior in query/memory residuals. Independent trainable query/key grounding projections can start aligned to that coordinate system; a random-initialization ablation is necessary to expose how much this initialization contributes. Thus even successful identity-initialized traversal will not establish unsupervised discovery of arbitrary symbolic identities.

The graph-as-data baseline is intentionally strong: it can match identical supplied identity keys and place graph-neighbor messages into token values, while retrieval uses ordinary attention. Its exact identity matching is a disclosed alignment prior. We must not weaken this control to obtain metric-specific superiority.

Raw metrics, diagnostic trajectories and resource measurements will be committed. No result has yet been observed for Stage 3.
