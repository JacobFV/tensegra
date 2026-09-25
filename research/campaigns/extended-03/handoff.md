# Handoff: extended-03 PAUSED (2026-09-25)

**Paused by the user**, who needs the GB10 to train a robot policy. No extended-03 job was ever launched on the GB10: no training, no evaluation, and zero GPU or remote CPU charged. The machine was verified idle at pause (0 campaign processes, GPU 0%). **Before resuming, check that the GB10 is free again. Do not interrupt the user's robot-policy training.**

## Done and validated (merged to main)
- **Campaign root and design.** [campaign.md](campaign.md), [world-spec.md](world-spec.md), [protocol-P1.md](protocol-P1.md) including its freeze amendment, [decisions.md](decisions.md).
- **Stage A, depworld-v1** (`src/topoformer/campaign03_depworld.py`) and the builder's notes ([depworld-notes.md](depworld-notes.md)):
  - world, d1 encoders, five references, and the progress-triggered events;
  - 32 depworld tests;
  - the full campaign0 suite passes: 173 on the GB10 env, CPU.
  - **Leverage gate (progress trigger, dev seeds 2e9+).** Reuse .976 = recompute .976 at −11% cost (−15% to −29% with 4 foreign records). Greedy .228. Reuse without revision .866. Naive reuse .809 with foreign records; no loss from event staleness alone (commits are revalidated, which is a disclosed safety property).
- **P1 configs.** `research/tools/campaign03_p1_configs.py`; bootstrap configs are in `configs/campaign03/p1-boot-*.json`. RL and sealed configs are generated after the bootstraps and freeze.

## Not done / not validated
- **Stage B preflight audit (training gate): interrupted.** Its partial work is on branch **`campaign/e03-preflight` (caf1a785)**: an unfinished `research/tools/campaign03_preflight.py` plus draft `d1-noapp`/`d1-noattempt` mask edits. It is **untested and not merged.** Resume it, or restart it, with an independent auditor (not the encoder's author). The task brief is the "Stage B preflight" instructions in decisions.md: distinctions 1–11, ablation masks erasing exactly the intended distinctions.
- **No model has been trained.** P1 rules R1–R5 are untested.

## Resume checklist
1. Confirm the GB10 is idle and the user has released it.
2. Finish Stage B on `campaign/e03-preflight`; the independent audit must pass every required d1 distinction. If a distinction fails, register an encoder fix before training.
3. Merge the masks and run the tests on the GB10 env.
4. Launch the 12 P1 bootstraps (`configs/campaign03/p1-boot-*.json`), then the RL configs (`campaign03_p1_configs.py rl --banks ...`), then the sealed evaluation (`... sealed --checkpoints ... --references dep_greedy,dep_recompute,dep_reuse,dep_naive_reuse,dep_reuse_norevise`), and an independent audit.
5. The budget clock was declared from 05:31Z on 2026-09-25. On resumption, re-declare the window: the pause should not count against the 24 h wall, and the compute ceilings remain unspent.

Infrastructure: see the extended-02 handoff (gb10-direct, `campaign02_job.py` receipts, `source-<sha>` snapshots).
