# Resume here

Campaign ACTIVE, not a single-stage stop. Baseline123299a7c5312e1f75019fa491618943f54d0de2; started2026-09-23T19:50:55Z, deadline2026-09-24T19:50:55Z. Ceiling43,200 GPU-process seconds, reserve8,640 for confirmation. Root is sole queue owner; budget.json and queue.json are authoritative. Do not restart completed jobs or launch around an active release.

Coordinator: `.worktrees/campaign`, branch `campaign/extended-01`. Worker branches/worktrees `campaign/{returns,semantics,attention,review}` under `.worktrees/campaign-*`; workers commit, root cherry-picks. Active agents `/root/stage9_returns`, `/root/stage9_semantics`, `/root/campaign_attention`, `/root/stage9_review`. Historical main remains123299a; progress branch is pushed at useful verified milestones. Existing old agents are completed, not competing coordinators.

GB10 SSH `gb10-direct`; CUDApython `~/topoformer-stage8-cuda/bin/python`, CPUpython `~/topoformer-pilot/.venv/bin/python`. No installs/upgrades. Local systemPython lacks torch/pytest; use existing remoteCPUenv for tests. Ten combined campaign tests passed earlier; new worker-specific tests also pass. Final integration tests still required. All3953 baseline tracked blobs preserved by independent initial audit; new work is additive.

## Current evidence and queue

Read campaign.md, experiment_registry.json, queue.json, budget.json and track handoffs. Root has integrated R04 balanced-grid source a4bd692 (root31422b7), including e368633 contract test, plus2b2f34e witness test. No unresolved conflict.

S01 ladder completed: N128603.831s, N1024646.9737s, N8192665.95281s. DEVcopy46.97/72.44/95.77%; orderedF1.500/.663/.854; exact0/512 each. N8192 lastinterval gains justify planned131072presentation optimizer-preserving extension. Worker preparing provenance/registration, not yet released.

A03 seed201completed78.11s and independently audited: primaryN64D16soft4=123/1024;soft8/context/message/hard1024/1024;frozen4→8also1024. Seeds202/203 completed76.17/75.20s; full3seed independent audit passes, with one intermediate permutation argmax discrepancy (zero final task/routes) disclosed. Source27fc238 unchanged. Do not duplicate live GPU work. Worker must report FREE immediately, then root records actual occupancy and releases next.

R04confirmation is RUNNING cap600, thenS02maincap180 andS03exposurecap900; order may adapt at safe boundaries. R04fresh3backbone 4kvs16k CE900,4096val/test perbackbone,32testonly; separate3328balanced-grid diagnostics. R01failedgate remains. R03same16kcache900/1800/3600 had no consistent fresh gain;900 retained by calibration. R02/R03 raw/cache audits pass.

S02 exact cached/public BF16 logitparity passed; profile7.29668s. FrozenN128 publictext-derived node features; existingedgehead+optimizer only, otherheads frozen. Main1200/4800updates,128TRAIN/512DEV. Edgeonlyclip differs from historicalglobalclip; disclose. Wholegraph ceiling126/128 due frozen2sloterrors.

A04 next attention design is content-dependent multi-neighbor selection with fresh attribute codes, strong dynamic neighbor-attention baseline. Design only pending allA03results. GAT/GATv2 methods consulted; no implementation/GPU yet.

CPU audit ledger review/cpu-audit-time.json updated independently; budget CPUtotal copied at boundaries. Charged GPUbefore runningR042456.10419seconds. Plenty of authorized budget; continue informative branches, no stage-stop. Push integrated verified milestones, final fulltests/audit/preservation/jobs check later.
