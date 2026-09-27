# Extended-07 decisions (timestamps from date/git)

- 2026-09-27T16:04Z: **Campaign adopted** from the user's brief; the window opened at 16:02:54Z.
  - **pro6000 state:** idle, 61 GiB free, 24 cores, no e0x units running, keepalive running. The GPU shows 10% / 5 GB used by an unrelated process, which is not touched.
  - **New root:** ~/structured-latent-dynamics-campaign07, with job.py copied (sha256 identical to campaign06) and metered.sh retargeted.
  - Historical campaign06 artifacts (checkpoints, labels) are **read-only** inputs.
  - **Tools:** campaign07_remote.py (new guards: --needs, --mkdir, concurrency/memory checks, output-path restriction) and campaign07_ledger.py (GPU charged only for '-gpu-' jobs; legacy occupancy reported separately).
  - **Plan:** Phase 1 (≤ 3 core-h) is two parallel builders:
    - (a) factor-contract and gradient-flow audit (static, plus mechanical tests);
    - (b) a frozen-checkpoint diagnostic runner, read-only with versioned outputs.
- 2026-09-27T16:07Z: **Launch guards smoke-tested:**
  - refusal works for a bad name, an output under campaign06, and a missing --needs input;
  - a real job ran after a quoting fix (shlex.join; the extended-06 helper had the same latent bug for commands containing shell metacharacters);
  - e07-smoke-launch3 exited 0.
- 2026-09-27T16:07Z: **Phase 0/1 builders started in parallel:**
  - campaign/e07-contracts: factor contract + gradient-flow audit, mechanical tests;
  - campaign/e07-diag: frozen-checkpoint diagnostic runner (common-history and free-running protocols, octet cf-only path, interventions, two-level scoring).

  Builder dev caps: 1.5k and 2.0k core-s. The Phase-1 tranche must total ≤ 10.8k core-s.
- Heartbeat cron be7769c4 (hourly at :43).
