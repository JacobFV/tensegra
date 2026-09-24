# RL02: pointer node addressing does not repair omitted composition

**The pre-registered promotion rule fails.** The rule was: omitted 3×4 ≥52/512 complete graphs, at least +26 over the comparator, and known retention within 26. The pointer arm has **0/512** omitted complete graphs at u0, u256 and u1024. So does the re-run all-record comparator. No confirmation is authorized, and this recipe is not extended (see [RL02-design.md](RL02-design.md)).

## Outcomes (final fixed endpoint u1024, correct NODE prefix supplied)

| Arm | Known complete /512 | Omitted complete /512 | Omitted teacher-forced edge tuples /35,396 | Omitted strict-valid /512 |
|---|---:|---:|---:|---:|
| all_records (RL01 recipe re-run) u0 / u256 / u1024 | 496 / 490 / 489 | 0 / 0 / 0 | 27,597 / 27,581 / 27,546 | 352 / 354 / 376 |
| pointer u0 / u256 / u1024 | 0 / 497 / **501** | 0 / 0 / **0** | 34 / 29,047 / **29,188** | 3 / 5 / 12 |

The comparator re-run reproduces RL01 closely: RL01 all-record had 484 known, 0 omitted and 27,535 local tuples at u1024. The small differences are GPU nondeterminism under a new source snapshot and are expected by the design.

## Where the invalid outputs come from

| Arm | Valid | Invalid generated node reference | Duplicate edge | Pair-slot multiplicity unsupported |
|---|---:|---:|---:|---:|
| all_records | 376 | 121 | 13 | 2 |
| pointer | 12 | 0 (impossible by construction) | **351** | 149 |

The pointer mask removes out-of-range references. That is a **supplied structural constraint**, not a learned result. On omitted compositions, free decoding then fails a different way: it repeatedly addresses the same node pair (duplicate edges), or it assigns multiple slot labels to one pair. Teacher-forced local edge accuracy on omitted graphs improves by 1,642 tuples (4.6 percentage points). Known exact retention also improves, 501 vs 489. Neither changes complete-graph generalization.

## Interpretation

Content-addressable target selection improves local relation choices and in-distribution retention. It does **not** repair the omitted-combination boundary under this parent, data and budget. The failure moves from invalid addresses to repeated addresses. That suggests the missing capability is tracking *which* admissible nodes have already been related (coverage/state over the emitted edge set), not only reaching the right address space. This is a localized hypothesis for future work, not a demonstrated cause. Per-relation localization of the RL02 predictions is part of the phase-2 independent audit.

Cost: main job wall 808.3 s, CPU 961.9 core-s, during heavy GPU sharing. Profile: 36.9 s. Receipts are in `research/results/campaign-02/rl02-{main,profile}-process/`. Predictions and checkpoints are in `~/topoformer-campaign02/results/rl02-main-v1` on gb10-direct.
