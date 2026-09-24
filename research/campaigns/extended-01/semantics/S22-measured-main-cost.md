# S22 measured complete-main budget proposal

Propose a **1,160-second whole-process main cap**. The timing-only planning estimate is **1,121.764 seconds**, including controller, payload and preparation reserves; margin beyond those reserves is38.236 seconds. Completed profile charge34.29 plus cap totals**1,194.29**, within the1,200 branch envelope by5.71 seconds. This is an engineering estimate, not a deterministic runtime bound or launch authorization.

Profile archive9c413d4a, manifest SHA5abc9b3c959709cc406c936c1b638efc502955b9aed14f052b7c044234a06fa8. Only timings and public-input lengths enter the forecast. Both backbones and all three diagnostic policies remain fixed; no profile efficacy or sealed confirmation is inspected for this decision.

| Main term | Seconds |
|---|---:|
|672 full batch32 decodes × worst measured160-step time|735.842|
|Explicit controller Python forcing/counter allowance,10%|73.584|
|All six exports ×32 population scaling ×2 payload allowance|85.405|
|Remaining evaluation packing/metrics work ×32|156.538|
|Both checkpoint/model setups, fixed|0.906|
|Preflight, twice measured|2.233|
|Full public-reference loading/checkpoint rehash/outer residual, fixed|17.255|
|Additional preparation/process reserve|50.000|
|**Planning total**|**1,121.764**|

Main is six conditions ×3,584 events /32 =672 full batches. Profile stress measures1.095004 seconds original and1.069520 broad for160 cached own-prediction steps; both batches demonstrably reach74 public tokens. Use the larger1.095004 for every main batch, then add10% specifically for policy forcing and counter work absent from the public-only stress helper. This decode allowance808.427 seconds is substantially above the alternative natural policy-decode extrapolation211.587 seconds (all six profile decode timers ×32). The natural timings already include actual controller work, but do not establish worst-case output-length coverage; they do not replace the larger allowance.

Profile112→main3,584 scales by32. The profile's fourth batch is half full, making this simple per-population timing scale conservative relative to full main batches. Export doubles after scaling to allow differing valid graph payloads; codec/metrics/metadata residual scales32-fold and remains separate. Main has no teacher forcing, optimizer, training preparation or newly written model checkpoints. Its fixed reference files are already full3,584-example S21 artifacts in the profile, so their load/re-hash cost belongs in fixed overhead rather than a32-fold term. The50-second extra reserve covers larger DEV target allocation, unclassified setup and process variability; it is additional to the73.584 controller and42.703 export allowances.

All six full conditions, seven DEV512 cells,160-record bound, original strict codec, privileged boundary definitions and unchanged checkpoints are retained. A timeout would be incomplete diagnostic evidence, not authority to truncate populations, suppress a policy or extend automatically. Root allocation, independent profile/cost review, immutable main freeze and explicit GPU release remain separate. Main source/config pin will be set only after that freeze.
