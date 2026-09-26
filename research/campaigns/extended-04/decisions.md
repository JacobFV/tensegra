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
- 2026-09-26T21:00Z: **Correction to the 19:55Z B-H6 note.** The claim that the "L1–L4 value heads are trained toward V*" is **wrong**. No loss uses V*. The value head is trained on the Monte-Carlo return-to-go of the **sampled** training rollouts; only the Q head uses Q* (in L4). The B1 numbers also show value-vs-own-return and value-vs-V* errors to be nearly equal, and the in-distribution error is not small. The live hypotheses are a sampled-training vs greedy-evaluation continuation mismatch, generalization, or fit/capacity.
  - **B2** ([protocol-B2.md](protocol-B2.md), registered before any run by the builder) adds a stop-gradient `v_own` head trained on own-greedy returns. B1 parameters and evaluation are bit-identical when it is on. Merged (268c21ae); launched with 6 runs (L4 and L1 × 3 seeds, ~4k core-s).
- 2026-09-26T21:05Z: **Independent B1 audit merged** (60785946).
  - The rescoring matches b1-score.json exactly. Labels are verified by brute force (8 configs, 2e-10) and Monte Carlo (178 tests). There are 0 label conflicts across 9,786 histories, splits are disjoint by generator parameters, inputs contain 0 leaks (2,525 steps rebuilt from public information), and streams and initializations match across rungs.
  - **Corrections adopted for the report:**
    - **B-H1** is worded as "L1-level action-set supervision and above beats outcome-only actor-critic, which is stuck at a non-probing local optimum (prop → exact_b1 → commit)".
    - **B-H2:** only L0→L1 is beyond noise; L1–L4 are all 1.75–4.06 with SE ≈ 1.
    - **B-H4 and B-H5 are "supported when the held-out splits are pooled"**. On heldout_comp alone both fail in 3/3 seeds: uniquely-optimal probe rate .64–.77, justified-switch ratio .59–.67.
    - **B-H6's cause is held-out generalization.** The value head over-predicts by +.03 to +.23 R, worst at k = 4, while being calibrated in-distribution (L4 dev .046) and on heldout_price. It is not a V* target, and not a sampled-vs-greedy mismatch (within ±.03). B2 still runs as registered; its "not continuation" outcome is now the expected one.
    - The L0 identical behaviour holds on 3 of 5 splits.
  - Audit CPU: 86 core-s metered, plus ~400 core-s local (estimate).
- 2026-09-26T20:50Z: Track C labels complete: 18 jobs, all exit 0, ~7.4k core-s. Training complete: 6 jobs, ~1.0k core-s.
  - Development readings (label development split; not an evaluation): ECE .011–.06.
  - The registered one-step gain over greedy + R-mask is ≈ 0 on the bootstraps and on P1-RL r0/r1 (several seeds registered m = ∞, i.e. never intervene). It is .017–.020 on P1-RL r2.
  - **Track C evaluation launched:** 24 tier-1 jobs (6 bases × 4 conditions; branch evaluation on the IID conditions) plus 6 causal jobs, on evaluation worlds from 161M. B2 evaluations launched: 6 jobs.
- 2026-09-26T20:55Z: **A2 training done** (5 runs, exit 0, ~4.5–5k core-s each). Deployment rule (P2a rule): reh/imit/crit/dep deploy their finals (30/30 qualify). **a2-ent deploys attempt 24** (24/30 qualify). Screening configs (configs/campaign04/a2s-*.json): 9 policies + dep_reuse; seeds 140M per protocol-A2; modes greedy/sampled/r_mask. Launched.
- 2026-09-26T21:00Z: **Phase F preparation:** three fresh X1 bootstraps (lineages r3–r5, the P1 recipe via campaign03_p1_configs.boot; training 3.06e9/3.08e9/3.10e9, development 3.903e9–3.905e9, initialization 300300+; disjoint from r0–r2 and all RL streams) launched now, so any promoted arm or controller can be confirmed on fresh lineages. The bootstrap recipe is unchanged; the fast path is result-identical.
- 2026-09-26T21:35Z: **A2 screening done** (10 jobs, exit 0, 8.65k core-s). The registered promotion rule (`campaign04_a2_promote.py`; research/results/campaign-04/a2s/) **promotes no arm**. IID group, lineage X1-r2, screening seeds 140M:

  | Policy | greedy | R-mask | sampled | work/success (greedy) |
  |---|---:|---:|---:|---:|
  | bootstrap | .857 | **.861** (best) | .794 | 143 |
  | reh | .849 | .848 | .844 | 155 |
  | imit | .851 | .850 | .842 | 145 |
  | crit | .845 | .856 | .816 | 164 |
  | ent (final; deployed attempt 24 similar) | .806 | .817 | .829 | 334 |
  | dep | .859 | .858 | .824 | 164 |
  | C1 | .851 | .851 | .828 | 179 |
  | C0 | −.147 | .830 | .800 | – |
  | dep_reuse | .900 | | | |

  - **Localized reason (the Track A fallback deliverable, v2 rev. 12):**
    - (1) Every stabilized RL variant (anchor, rehearsal, critic warm-up, entropy control, deployment-aligned masked sampling) **prevents the greedy collapse on this lineage**, but none exceeds the bootstrap in any deployment mode.
    - (2) RL reliably improves the objective it optimizes, the **sampled** policy (.794 → .816–.844), yet the sampled policy's utility stays below the greedy/R-mask bootstrap (.857/.861). The available sampled-mode gain lies below what argmax deployment of the imitation policy already achieves.
    - (3) Rehearsal ≈ imitation-only (.849 vs .851), so RL adds nothing on top of teacher supervision here.
    - (4) Entropy control prevents collapse but **more than doubles work per success** (334 vs 143). The P2a anchor effect is therefore not "just entropy control": the anchor keeps work at ~180.
    - (5) Training through the deployed recovery rule (dep) gives the best greedy/R-mask RL utility (.859/.858) but is still not above the bootstrap.
    - (6) Every RL arm raises work per success over the bootstrap.
  - Conclusion: with this reward, recipe and horizon, on-policy actor-critic finds no deployable improvement over imitation. Improvement would require an objective that credits argmax/deployed behaviour, or a better teacher. dep_reuse is .04 above the bootstrap, and imitation leaves that gap open.
  - **No A2 Phase F.** Single lineage, exploratory; stated as screening evidence.
- 2026-09-26T21:45Z: **Track C (protocol-C1) scored** (`campaign04_c1_score.py`; research/results/campaign-04/c1/). Evaluation CPU was 45.5k core-s, **~3× the 14k estimate**, the main budget overrun of the campaign (total 109.4k at 21:37Z).
  - **C-H1 NOT SUPPORTED** (loop-prone):
    - r0: learned .754 vs best fixed (R-sample) .781;
    - r1: learned .798 vs .822;
    - r2: learned .831 vs best fixed (R-mask) .813, i.e. **+.018 (< .02)**, cost .075 vs .099 (−24%), success .906 vs .912.
  - **C-H2 passes by rule but trivially:** the learned controllers on bootstraps r1/r2 intervene at .006/.0004 of decisions, and r0 fails (.840 vs .851).
  - **C-H3:** loop-prone NOT SUPPORTED (1/3). Competent passes only because matched-rate random intervention is harmful.
  - **C-H4 SUPPORTED:** ECE .009–.026 on evaluation worlds; pooled predicted-vs-actual Spearman .55–.75 (CIs > 0). **Per intervention:** stop and mask-top effects are predicted (ρ .24–.57), while the sample-step effect is **not** (ρ ≈ −.18 to .23, mostly with CIs including 0).
  - **C-H5:** mostly unevaluable (the learned gains are ≤ 0). On r2 the shuffled-target controller collapses (.374), so the only positive gain depends on trained appraisal. Appraisal-only equals the default (±1e-4, sanity).
  - **Localized reading (registered):** calibrated appraisal of outcome and of stop/mask-top effects, but **no actionable headroom over simple public rules**, except a sub-threshold, single-lineage cost reduction on P1-RL r2.
- 2026-09-26T21:50Z: **[protocol-F.md](protocol-F.md) registered.**
  - F1: R-mask vs greedy on fresh bootstraps r3–r5, on sealed fresh worlds from 170M, 512 per condition.
  - F2: probeworld L0/L1/L4 on fresh seeds 3–5 with fresh world draws (offset 700).
  - Added `--world-offset` to the probeworld evaluation (default 500, so B1 is unchanged).
- 2026-09-26T21:55Z: Fresh bootstraps r3–r5 done (exit 0). F1 configs built (latest checkpoints, hashes recorded; sealed seeds 170M, 512 per condition, greedy + r_mask, plus a dep_reuse reference). F1 launched; F2 (9 trainings) launched.
- 2026-09-26T21:42Z: **B2 scored** as registered (`campaign04_b2_score.py`; research/results/campaign-04/b2/). **Primary NOT SUPPORTED:** the own-greedy-return value head `v_own` is no better calibrated on held-out splits than V.
  - L4 v_own reliability error, per seed:

    | Split | s0 | s1 | s2 |
    |---|---:|---:|---:|
    | heldout_k | .13 | .16 | .14 |
    | heldout_comp | .10 | .08 | .09 |
    | heldout_price | .03 | .03 | .06 |

    It is calibrated in-distribution (test_iid .02–.04). The V* noise floor is ≤ .06 everywhere, so the metric is not noise-limited.
  - The policy is unchanged, as required: B1 parameters and evaluation keys are bit-identical, with 0 regret difference.
  - **Failure prediction:** commit-level AUROC .45–.65, CIs include .5 (9–20 wrong commits per seed); not supported.
  - **Registered reading:** "not continuation". The value estimates fail to generalize to held-out generator regions (k = 4 worst), matching the B1 auditor's diagnosis. **Deliverable 4 status:** calibration holds in-distribution in both tracks (probeworld test_iid; depworld Track C ECE ≤ .026 on fresh worlds of the training distribution), but not under generator-parameter shift in probeworld.
- 2026-09-26T21:47Z: **F2 confirmation scored** (`campaign04_f2_score.py`; research/results/campaign-04/f2/). Fresh seeds 3–5, fresh world draws (offset 700), 512 per held-out split. All 9 trainings and 10 evaluations exited 0 (training 3.75k core-s).
  - **F2-H1 SUPPORTED (3/3 pairs):** held-out regret L0 39.0/82.3/72.5 vs L1 1.9/2.6/2.0 and L4 2.6/1.7/1.5. π* ≈ 0; fixed rules 99–126. All pass the transfer floor.
  - **F2-H2 SUPPORTED (3/3):** ρ-bin deviation .016–.042.
  - **F2-H3 SUPPORTED (pooled, 3/3):** probe when not optimal ≤ .01, uniquely-optimal probe .83–.86.
  - **F2-H4 SUPPORTED (pooled, 3/3):** unjustified switches .041–.050, justified-after-failed-probe ratio .82–.86.
  - **The composition limit replicates exactly.** On heldout_comp alone:
    - the uniquely-optimal probe rate is .545 in all 3 seeds (threshold .80);
    - the justified-after-failed-probe ratio is .51–.58.

    It passes per split on heldout_price and heldout_k. Rational probing and switching therefore transfer to new price regions and reuse horizons, **but not to held-out combinations of conditions** (unreliable+events, side-effect+correlated).
- 2026-09-26T21:50Z: **F1 scored** (4 jobs, exit 0, 1.32k core-s; research/results/campaign-04/f1/). **F1-H NOT SUPPORTED as registered.**
  - The utility and success clauses pass in 3/3 fresh lineages: IID-group r_mask − greedy utility is +.0007, −.0014 and +.0030, and success is +.001, .000 and +.004.
  - The registered **episode-level** no-progress clause fails in 3/3: the episode rate is identical under both modes (.033, .058, .060).
  - **Mechanism (a specification error in protocol-F, not a rule failure):** R-mask masks an action only *after* the diagnostic flags its first stagnant repeat. An episode that stalls therefore always records at least that first no-progress step, and the episode rate cannot fall by construction. The A1 reading F1 was meant to confirm was the **per-step** rate. Protocol-F transcribed it as the episode rate.
  - **Descriptive, not registered:** the per-step no-progress rate falls 4–7× in every fresh lineage (.013 → .003, .052 → .013, .077 → .011), and on the events and foreign4 conditions (.022–.085 → .004–.015). All conditions meet the absolute floor (success ≥ .8 × dep_reuse).
  - The report states the registered failure and the per-step reading side by side; no pass is claimed.
- 2026-09-26T21:52Z: **Housekeeping.**
  - Three decision entries (F2, F1, B2) carried timestamps ahead of wall-clock time (22:00–22:55Z). They are corrected to the actual times (21:42–21:50Z).
  - The ledger's 13 'running' entries are the failed launches (12 empty-SHA label jobs and b-labels-launchfail). They exited before the wrapper wrote occupancy receipts, and their CPU is negligible (< 1 core-s each).
