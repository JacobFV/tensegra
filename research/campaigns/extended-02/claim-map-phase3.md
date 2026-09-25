# Extended-02 phase-3 claim map

It extends [claim-map-phase2.md](claim-map-phase2.md). All results come from sealed evaluations of frozen endpoints. Each experiment has three lineages unless stated otherwise.

| Category | Claim | Evidence | Limit |
|---|---|---|---|
| f. Selection | Depth-allocating selection (successive halving) beats copy-based PBT and matched multistart at equal RL updates | E15: +0.0149 vs PBT and +0.0052 vs multistart pooled IID; 3/3 replicates in point estimates (r0 vs PBT CI includes 0) | Below the pre-tuned single learner (−0.006); registered robust-mode rule not supported (1/3); banks shared with E08 |
| f. Diversity | Supplied behavioral-niche protection does not improve the selected finalist | E15 niche − plain −0.0024 | Niche descriptor supplied, not discovered |
| e. Composition | Learned action sequencing within a supplied, public, environment-enforced stage order transfers to unseen stage orders and 3-stage sequences (not functional composition) | E16: 1.000 IID; registered transfer rule passes 2/3 on aggregate reading (1/3 per condition) | Stage-typed features supplied; one environment family; distinct-stage sequences only |
| e/c. Failure boundary | Novel composition exposes a return-binding shortcut (apply the existing return of the wrong type) | E16 A→S 0.00 in one lineage; 12,195 type-mismatch uses | Localized; see E17 |
| c. Return binding | Wrong-typed distractor returns derail untrained controllers; training with them repairs both the distractors and the held-out order | E16 on distractor worlds 0.32–0.38; E17 A→S 0.57 → 0.94 (means); rule passes on mean reading, fails per lineage (r2 IID .941) | Type-only distractors; provenance: see next row |
| d. Resource allocation | Budget-bound hard-assign region: learned controllers equal the teacher | E16/E17 hard assign = teacher (0.746 / 0.824) | Not superior |
| e. New primitive | Prior two-primitive training gives a large few-shot head start on a never-seen primitive; not reliable after 120 updates | E18: adapted@20 0.67–0.98 vs scratch 0.02–0.19 (unregistered); @120 +0.02/+0.46/−0.19 → partial | Few-shot retraining with teacher supervision; zero-shot not identifiable; one new primitive |
| c. Provenance binding | Same-type (wrong-provenance) prior results collapse controllers without provenance inputs; public provenance features + exposure make them fully robust | E19 E17 endpoints 0.35–0.37; E20 1.000 in 3/3 with no retention loss | Features supplied; input and data changes not separated; one lineage keeps a direct-commit perseveration on select-after-assign triples |
| f. Selection + robustness | Depth-allocating selection with robustness-visible fitness does not select the robust combined policy | E21: 0/3 combined; IID below plain halving in 3/3 | Banks shared with E08; one selection-panel composition |
| d. Failure memory | Direct-commit perseveration is removed by public per-stage failure counters | E22: perseveration failures 126 → 0 over 22,272 sealed episodes per arm; no retention loss | Evidence concentrated in one lineage (r2); supplied counters, not learned memory |
