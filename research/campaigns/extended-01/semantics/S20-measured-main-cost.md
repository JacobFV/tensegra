# S20 measured full-main cost

Propose a **4,300-second whole-process cap** for all nine fixed main arms. The timing-only planning estimate is **3,891.06 seconds**, including600 seconds of preparation/process reserve and conservative export/decoding allowances. Margin is408.94 seconds. Profile charge33.83 plus this proposed cap totals4,333.83, within the5,000-second branch envelope with666.17 unallocated. This is an estimate, not a bound or launch authorization.

The successful three-arm profile archive is7e99ad3a; manifest SHA54533444357774858d69b8b86b91b6f6be3864a4163e78bf448094c6ca122fec. Only phase timings were used. No model settings, populations, seed choices or decisions derive from profile efficacy.

| Per-lineage term | Original | Contextual | Record |
|---|---:|---:|---:|
| Optimizer:20-update timer ×204.8 |389.224|319.823|182.213|
| Fixed TRAIN128 calibration |2.998|2.719|none|
| Confirmation forward:DEV64 timer ×32 |19.961|25.164|see decoder term|
| Setup + final checkpoint |4.364|2.369|1.998|
| Workspace export:profile ×2176/192 ×2 |34.948|33.200|see record term|
| Record teacher forcing TRAIN +confirmation |—|—|1.272|
| Record160-step decoding:68 batches |—|—|53.840|
| Record export and packing, scaled then doubled |—|—|15.204|
| State/evaluation wrapper residual |0.638|0.703|0.791|

Multiply each complete arm estimate by three independent lineages: **3,274.29 seconds**. Add three-times preflight1.82, three-times measured outer residual14.95, and600 seconds reserve: **3,891.06 seconds**.

Workspace calibration retains the fixed128 population once per endpoint; only64→2,048 evaluation forward time scales32-fold. Export scales by total archived graph count,192→2,176, then doubles for payload/CPU variability. This scaling is deliberately conservative because fixed calibration arrays do not grow with confirmation. Record decoding covers TRAIN128 plus confirmation2,048:68 batches of32 at all160 steps. Use the larger of the current S20 stress0.396373 and earlier identical-size S19 stress0.791760 seconds; this timing-only margin avoids extrapolating from early natural-stop output. Teacher forcing and packing/export are accounted separately. No main stress replay is needed.

The600-second reserve covers data preparation/query sampling outside timed optimizer steps, source/parent hashing, allocations, output and runtime variability across seeds. The profile's five-second residual also includes one-time imports and unclassified work; charging it three times is conservative. Final checkpoints have full optimizer states in both profile and main. Uncertainty remains in long-run load and valid prediction packing; these allowances are engineering estimates, not measured upper bounds.

Observed S19 full-main optimizer185.95 seconds is close to this record forecast182.21; S18 contextual optimizer335.17 is close to319.82. Neither observation licenses narrowing the unchanged scientific recipe. All nine main arms remain4,096 updates with final TRAIN128 and confirmation2,048, no interim DEV evaluations, no per-seed outcome adaptation. A timeout is incomplete evidence, not permission to silently shrink support or extend compute. Root allocation, immutable freeze, independent review and explicit release remain separate.
