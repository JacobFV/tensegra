# Decisions (extended-06)

- 2026-09-27T05:52Z: **Brief adopted** (the user's extended-06 brief, sent in reply to the extended-05 offer). Window 05:51:06Z → 17:51:06Z.
  - **State reconciled:** main 58b231d6. The stale extended-05 Track B audit worktree was locked by pid 590201, identified as **this Claude session** (not a stray process). Its branch db1522c3 was already merged and the worktree was clean, so it was unlocked and removed; no process was killed.
  - **pro6000:** idle (load 0), 60 GB RAM free, 188 GB disk free, keep-alive Running, no campaign units.
  - **New root** ~/structured-latent-dynamics-campaign06 with job.py and metered.sh (self-test OK: torch 2.14.0+cu130, numpy 2.5.3).
  - **Tooling:** campaign06_remote.py adds a conflict-marker refusal and an e06- job-name guard; campaign06_ledger.py.
- 2026-09-27T05:53Z: Design v1 written (portworld for Track A with headroom gate GA; probeworld-v3 screening, balanced counterfactuals, matched-volume arms B0–B3 and factorized arms for Track B; Track C conditional). Internal review and two Phase-1 builders launched in parallel. The builders start with the model-independent headroom and screening work, which the review cannot invalidate cheaply.
