# Decisions (extended-04)

- 2026-09-26T18:43Z: The user's campaign brief was adopted. The new metered window runs to 2026-09-27T18:43Z (48 CPU core-h, 12 GPU-h, 20% reserve).
  - **State reconciled:** main = 7e9d44f9; no other active coordinator (no worktrees or new branches); the pro6000 is idle.
  - **WSL keep-alive:** the task had stopped (status Ready; WSL uptime 0 min at check). It was restarted and the 63 extended-03 result directories are intact. The keep-alive status is now checked before long jobs.
  - **New remote root:** ~/tensegra-campaign04. `bin/metered.sh` meters all remote dev/test/audit work, closing the extended-03 unmetered gap. The env is reused from ~/tensegra-campaign03/env (torch 2.14.0+cu130).
  - Branch campaign/extended-04 from main.
- 2026-09-26T18:55Z: Design v1 written ([design.md](design.md)). An internal review was commissioned. The Phase A throughput optimizer starts in parallel, since it is design-independent and gated by bit-identity tests. The Track B/C builders and the progress diagnostic start after the review.
- 2026-09-26T19:15Z: **Internal design review** (review/design-review.md: 5 BLOCKER, 11 MAJOR, 3 MINOR). All adopted as **design v2** (design.md, v2 revisions 1–12):
  - promotion compares against the bootstrap under the same mode and under its best mode;
  - an imitation-only control is added for rehearsal;
  - the A2 arms run on the C1 base, except entropy (on C0);
  - A2-dep uses on-policy masked sampling;
  - the environment gets a clone() with a shared solver and an exact solver cache;
  - Track C's default is greedy + R-mask, equal to the label continuation;
  - branch labels are used only as regression targets, with a stop-gradient;
  - Track C criteria are split by base type, with random-rate and automatic-threshold controls;
  - diagnostic fixes: content-keyed records, think counted, class precedence, "triggered revision";
  - probeworld splits are defined by generator parameters;
  - Tracks B/C run CPU-only;
  - the Phase F launch deadline is ~2026-09-27T09:45Z;
  - a fallback "localized reason" deliverable and a supplied/learned ledger are added.
- 2026-09-26T19:20Z: Builders launched in isolated worktrees:
  - **e04-infra:** diagnostic v1, deployment procedures (R-mask, R-sample, masked sampling), clone and cache, A1 tooling;
  - **e04-a2arms:** rehearsal and imitation-only, entropy Lagrangian, critic warm-up, masked-rollout hook, A2 configs;
  - **e04-probeworld:** Track B.

  The Phase A throughput work continues (e04-throughput). All remote dev work is metered.
- 2026-09-26T19:11Z: A2 options merged (6cb0ecce; 268 tests on snapshot 71109534, metered). **Four A2 screens launched:** a2-reh, a2-imit, a2-crit, a2-ent. a2-dep waits for the infra progress tracker. [protocol-A2.md](protocol-A2.md) registered at 19:15Z, before any A2 output: registered modes, the promotion rule against same-mode and best-mode bootstraps, the imitation control, and screening seeds from 140M.
  - Metering note: metered.sh runs commands without a shell, so globs must go through `bash -c`.
- 2026-09-26T19:20Z: **Phase A throughput merged** (449176be; [throughput.md](throughput.md)).
  - GPU tranche 1.66–1.69×, sealed-style evaluation 1.8–2.0×; the Python side is 2.05× faster.
  - The fast path is default. Encoded tensors, parameters, optimizer state, RNG and evaluation rows are bit-identical to the reference (`TENSEGRA_REFERENCE_PATH=1` or `campaign04_fast.set_default(False)`).
  - One additive merge conflict with the A2 options was resolved. The post-merge suite passes 308/308 on **both** fast and reference paths (metered).
  - The four running A2 screens finish on their launch source (71109534, reference path). Later launches use the fast path, which is result-identical by the equivalence tests, so A2 comparisons are unaffected.
- 2026-09-26T19:40Z: Track B probeworld merged (a2e6a56b).
  - Exact DP matches brute force on 7 cases. Labels are a function of visible history only. The 21-vs-100 worked example and its violation variants reproduce. The ρ_k/k build switch is shown from exact labels.
  - Dev CPU: 467 core-s metered, plus ~300 core-s local (estimate).
  - **[protocol-B1.md](protocol-B1.md)** registered before any full-ladder run: B-H1…B-H6, best rung fixed as L4, transfer floor .8 × π* success.
  - The evaluation got first-action probe-rate metrics for B-H4 (added before any ladder output).
- 2026-09-26T19:27Z: B1 labels launched (source-28152106). The first attempt failed in 0 s because launch-cmd's REMAINDER parsing captured the cap options. The helper is fixed; the failed receipt is kept as b-labels-launchfail-process (charged).
- 2026-09-26T19:45Z: **Infra merged** (9559016b).
  - Diagnostic v1 reproduces 52/52 registered P2a cells and agrees 99.2% per step; the documented differences are repeated identical calls and same-content retrievals now counted. The references score 0.
  - Clone and cache are result-identical (0/49,346 mismatches).
  - Root added `campaign04_progress.rollout_mask_fn`, the adapter matching A2's `progress_v1` contract; the infra tracker API differs from the stub contract. Two A2 tests that assumed the module was absent were fixed.
  - The full suite is 363/363 (metered). An A2-dep CPU smoke masks actions (33 masked decisions in a batch).
  - Diagnostic cost policy (F16) is registered in [protocol-A1.md](protocol-A1.md).
- 2026-09-26T19:46Z: **B1 ladder trainings done:** 15/15 exit 0, 7,830 core-s total (above the 6k estimate). Evaluations and references launched next.
- 2026-09-26T19:55Z: **B1 ladder results**, scored as registered by `research/tools/campaign04_b1_score.py` (research/results/campaign-04/b1/b1-score.json). The independent audit is pending.
  - **Held-out regret per episode** (mean of heldout_price/k/comp; V* ≈ 230–340), per seed:

    | Rung | s0 | s1 | s2 |
    |---|---:|---:|---:|
    | L0 | 25.0 | 24.8 | 39.0 |
    | L1 | 2.6 | 3.0 | 3.5 |
    | L2 | 2.5 | 2.3 | 2.1 |
    | L3 | 2.1 | 1.8 | 2.9 |
    | L4 | 2.1 | 2.6 | 4.1 |

    For comparison: fixed rules 104–127, π* ≈ 0. All runs pass the transfer floor.
  - **B-H1 SUPPORTED.**
  - **B-H2:** L0→L1 adds value, as do L1→L2 and L2→L3 (≥ 10% in 2/3 seeds); L3→L4 does not.
  - **B-H3 SUPPORTED:** L4's build-rate deviation from π* across effective-ρ bins is .02–.05.
  - **B-H4 SUPPORTED:** L4's first-probe rate is .03–.04 when probing is not ε-optimal and .88–.91 when it is uniquely optimal. L0 never probes (0/0), so outcome-only learning found no probing at all.
  - **B-H5 SUPPORTED:** L4 unjustified switches are .044–.050 of all switches.
  - **B-H6 NOT SUPPORTED:** value-head reliability error is .13–.29 R for the supervised rungs and .09–.13 R for L0, against a .05 R threshold. The likely cause, *not* validated: L1–L4 value heads are trained toward V* (the optimal continuation) while calibration is registered against the model's own realized return. This is a continuation mismatch of exactly the kind the brief warns about.
  - L0 seeds 0 and 1 have different weights (model.pt hashes differ) but identical greedy behaviour on the evaluation episodes: convergence to the same deterministic strategy, not a seeding fault.
- 2026-09-26T20:30Z: **Track C merged** (36648c0f; 377/377 tests).
  - The default runner is bit-identical to deploy_episodes. Labels are regression-only (paired advantages, K = 4). The appraisal GRU has 30,150 parameters.
  - Dev smoke (descriptive): little one-step headroom over greedy + R-mask for sample and mask-top; "stop" beats the default at 22% of loop-prone label points, where the default then fails.
  - Integrity: a pre-registration smoke touched the first 16 registered evaluation worlds (160M), so the evaluation range was moved to **161M**. **[protocol-C1.md](protocol-C1.md)** registered: C-H1…C-H5, split by base type, with random and threshold controls, calibration and causal checks.
- 2026-09-26T20:35Z: 12 Track C label launches went out with an **empty snapshot SHA** (the snapshot was refused because the ledger had dirtied budget.json, and the shell continued). They failed immediately; receipts renamed *-emptysha-process and charged. The helper now refuses a malformed or missing snapshot.
- 2026-09-26T20:40Z: **A1 deployment matrix done** (8 jobs, exit 0, 4,844 core-s). Analysis: research/results/campaign-04/a1/. IID-group readings (descriptive, per protocol-A1):
  - **(i) Public recovery rules rescue the collapsed P1-RL policies.**

    | Policy | greedy | R-mask | R-sample | sampled |
    |---|---:|---:|---:|---:|
    | r0 success | .078 | .865 | .881 | .838 |
    | r2 success | .000 | .918 | .920 | .885 |

    Their utility is .77–.83, which is still **below the bootstrap greedy (.857–.871)**. P1-RL r1 (stable): greedy .825 → R-mask .836.
  - **(ii) R-mask does no harm to the competent policies:** bootstrap utility within ±.003 of greedy, with the no-progress rate cut 5–7× (.055–.115 → .008–.016). C1: .854 → .856. Sampled is the worst mode for every competent policy (−.06 to −.07 utility). No learned policy in any mode beats dep_reuse (.908) or dep_recompute (.904).
  - **(iii) Track C headroom:** on the loop-prone bases, default (R-mask) utility sits .04–.09 below the competent bootstrap, which is the space a metacontroller could recover. On the competent bases, the only room is the residual gap to dep_reuse (~.04).
- 2026-09-26T20:36Z: Track C label chunks 0–15 done for all 6 bases (12 jobs, exit 0, ~5.3k core-s < 8k), so the third chunk (16–23) was launched per protocol-C1.
