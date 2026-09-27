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
- 2026-09-27T16:25Z: **P0 contracts merged** (campaign/e07-contracts cc620c64; 17 tests pass; 72.6 metered + ~25 local core-s).
  - **Factor contract:** 23 coordinates in 4 groups. LRN was a single 23-d regression, not "four quantities". **No hidden-state leak** (3 mechanical checks).
  - **Strategy-cost coordinates are far from Q\*:** errors of 3–27 units against a 0.5 tolerance; the formula-cheapest strategy is optimal at only 36–38% of decisions.
  - **Defects:**
    - candidate_trust double-counts under U (irrelevant to S+C+E);
    - cost_probe_first omits the false-solved loss;
    - b1/use costs omit the event re-run;
    - p_probe_resolves ≡ belief_H;
    - exp_hard_cost and steps_left are constant on b6/b6c;
    - clip saturation of cost_use (0.45%) and build_value (1.1%);
    - build_value carries 32% of the target variance;
    - only 11/23 coordinates are functionally independent given the config.
  - **Gradient flow verified:**
    - the policy reads tanh(W[z; stopgrad(aux(z))]) through a fusion layer shared with the value head;
    - the aux loss updates the encoder, GRU, trunk and aux head;
    - LRN's fusion init differs from RAWF/SUP (RNG order);
    - RAWF's 2,944 factor-column weights are dead;
    - global clip and Adam are shared (aux ~1.1% of the squared gradient norm; clip effect < 1%);
    - action-sampling RNG drift.

    **LRN − RAWF therefore bundled** shaping, reading, init and dead-vs-live columns.
  - A 2×2 spec with copied init, separate optimizers/clips and per-decision sampling RNG is ready for Phase 2.
  - Seed block 6.870–6.880e9 registered (dev_smoke).
