# S10 complementary frequency-plus-grammar control

All four fixed policies recover **zero exact graphs** on both TRAIN128 and DEV512 from the inherited all-8,192-TRAIN frequency prediction. The archived 512 original DEV metrics replay exactly before policy application. No refitting or input conditioning was added.

| Split | Policy | Exact | Typed F1 | Ordered F1 | Removed active edge errors | Introduced active edge errors |
|---|---|---:|---:|---:|---:|---:|
| development | baseline | 0/512 | 0.61760 | 0.43441 | 0 | 0 |
| development | bookkeeping | 0/512 | 0.61038 | 0.43441 | 1336 | 3784 |
| development | combined | 0/512 | 0.61038 | 0.43441 | 1336 | 3784 |
| development | schema | 0/512 | 0.61760 | 0.43441 | 0 | 0 |
| train | baseline | 0/128 | 0.61983 | 0.44255 | 0 | 0 |
| train | bookkeeping | 0/128 | 0.61105 | 0.44255 | 322 | 958 |
| train | combined | 0/128 | 0.61105 | 0.44255 | 322 | 958 |
| train | schema | 0/128 | 0.61983 | 0.44255 | 0 | 0 |

The generic output also contains identically duplicated calibrated labels; frequency scores are not separately calibrated. These aliases are not independent conditions. Schema masking does not change this already admissible constant graph. Bookkeeping worsens typed F1 because its constant copied-index identities and predicted attributes are not correct for each event.

This control receives no per-event text, token features, copy inventory, current length, or renderer information. Fixed canonical output coordinates and TRAIN copy-index/type/value/slot frequencies remain, with public compiler rules supplied. It is an input-independent prior control, not a competitive equal-capacity learned comparator. It does not replace S01 or S03.

The result rules out the narrow explanation that this particular inherited frequency prior plus these same rules reproduces the learned-actor diagnostic recovery. It does not establish broad semantic competence, a unique architecture benefit, or successful unseen-renderer transfer. All learned S10 results and historical gates remain unchanged.

CPU wall33.17seconds, RSS760812KiB, zeroGPU. Protocol/source were frozen and independently preflighted before outcomes. Baseline prediction and source hashes, transformed target archive, all5120 per-graph records, exact output hashes, and receipts are retained. Independent raw reconstruction pending.
