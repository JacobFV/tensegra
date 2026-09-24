# S19 measured main cost proposal

The completed profile supports a **1,000-second whole-process main cap**. The planning estimate is **777.65 seconds**, including a 200-second preparation/process reserve; the remaining margin is 222.35 seconds. This is an estimate, not a runtime guarantee. No profile quality, confirmation data, or outcome-based recipe selection enters this proposal.

The profile charged **14.18 seconds** (outer 14.17; inner 14.140743). Its immutable manifest is `668c4ed217dd654638ae9a7a02180de26a70a43a5ae1a56add868d501413ebc2`, frozen config `d9f853f7e9d4943ced1dbe2d378648608d2117ef99b8a08710ded2abb253d492`. Archive: `research/results/campaign-01/semantics/s19-profile/s19-profile-v1/manifest.json.gz`, integrated profile commit `831e13a7`. Exact values and arithmetic are in `S19-measured-main-cost.json`.

| Main cost term | Scaling | Seconds |
|---|---|---:|
| Optimizer training | 1.148895 seconds / 20 updates × 4,096 | 235.294 |
| TRAIN teacher forcing | Sum of two profile panels × 2 | 0.556 |
| DEV teacher forcing | Sum of two profile panels × 64 | 7.079 |
| Public greedy decoding | 272 batches × 0.791760 seconds for 160 steps | 215.359 |
| Evaluation export | Population-scaled profile export × 2 allowance | 17.807 |
| Evaluation residual | Population-scaled packing/scoring residual × 2 allowance | 88.589 |
| Checkpoint export | Initial empty-AdamW checkpoint + three full-state checkpoints | 5.411 |
| Preflight | Twice measured preflight | 3.655 |
| Setup | Measured fixed setup | 0.824 |
| Other outer-process overhead | Profile charge minus separately measured phases | 3.077 |
| Additional preparation/process reserve | Explicit engineering reserve | 200.000 |
| **Planning total** | | **777.650** |

All four main curves evaluate TRAIN128 and DEV2048: `4 × (128 + 2048) / 32 = 272` greedy batches. The stress timing uses the longest supported public inputs (64 tokens), cached own-prediction decoding, and host argmax synchronization for all 160 steps; it ignores EOS only for timing. It replaces extrapolation from naturally short or invalid profile outputs. Teacher forcing remains a separate cost. TRAIN sums scale by two because there are four rather than two curves; DEV sums scale by 64 because both curve count and per-curve population increase.

Natural profile outputs do not measure full valid-graph scoring/export payloads. The doubled export and evaluation-residual terms explicitly allow for this difference; they are engineering allowances, not measured full-main bounds. The extra 200 seconds also allows per-update preparation outside the optimizer timer (about 48.8 milliseconds per update if entirely consumed there), process variance, and unmodeled overhead. Checkpoint scaling distinguishes the initial empty optimizer from subsequent AdamW states. Main does not repeat the separate stress benchmark.

The fixed main remains 4,096 updates, batch eight, seed 1901, construction order seed 15115, and curves at 0/1,024/2,048/4,096 with the complete registered TRAIN and DEV panels. No truncation or automatic extension is authorized by this estimate. A timeout would be reported as incomplete under the cap.

Profile charge plus proposed main cap is **1,014.18 seconds**, leaving **2,585.82 seconds** in the 3,600-second S19 development envelope. This is unallocated headroom, not permission for another run or evidence that an extension is warranted. Root allocation, independent review, immutable main freeze, and explicit GPU release remain separate.
