# Decisions (extended-05)

- 2026-09-27T01:26Z: **Brief adopted** (the user chose "adopt and start now").
  - New window 01:26:57Z → 2026-09-28T01:26:57Z; new root ~/structured-latent-dynamics-campaign05 with bin/job.py and metered.sh.
  - The pro6000 was checked: WSL up 6.5 h, keep-alive Running, GPU desktop-only. An unrelated user `uv pip install` (inpaint360gs env) is running and will not be disturbed.
  - The pro6000 directories were renamed by the user (tensegra-campaign0X → structured-latent-dynamics-campaign0X); tooling was repointed (7851b490).
  - Branch campaign/extended-05 from main 3a566812 (which includes the extended-04 report correction on the composition probe rates).
- 2026-09-27T01:29Z: Design v1, registry (7 entries) and machine-readable job plan (~88k core-s with 25% contingency; ~42k ungated) written. Internal review commissioned next. Timestamps for later entries come from git commits.
- 2026-09-27T01:44Z: Internal design review (3 BLOCKER, 11 MAJOR) adopted as **design v2**. Key changes: G1 counts no abstain gains and is a necessary condition only, plus a deployable cross-fitted single-deviation gate G1b ≥ .01; O(I) anchored to D's action kind; r3–r5 reserved for confirmation; the new probeworld split table (holds removed from train and dev, new seeds); exact-solver pool sizing (≥ 20 supporting configurations per cell); a B-HR headroom gate; BX3 modular arm; BO trained on the extended-04 split; learned-head evaluation priced at .35 core-s/episode. Both builders notified.
- 2026-09-27T02:06Z: **Track B tooling merged** (df55e56c; 39 tests; extended-04 paths bit-identical).
  - Split v2 audit: 18/18 pass.
  - B-LOC on 12 existing models: ~70% of L4 first errors on the challenge pairs are **value-estimate** errors (the Q-head argmax is also wrong); 57–65% occur at the first decision; the probing shortfall concentrates in U+E.
  - **The new holds are weak compositions:** ignoring one flag costs > ε in only 16% (U+C) and 12% (S+E) of configurations, vs 38% for U+E. The B-HR gate may therefore show no failure; that would be reported, not repaired.
  - BX2 is non-trivial but null at the first decision (belief = prior).
  - **The B-X non-inferiority regret tolerance is registered at 1.0 per episode** before any B-X run.
  - Launched: bx-labels and BO seeds 0–2.
- 2026-09-27T02:09Z: **Track A headroom tooling merged** (7028e2c5; 394 tests).
  - D matches extended-04 r_mask on 32 dev worlds for each of r0–r5.
  - O(I) is anchored to D's action kind, with 'not_call' added per review F3 (accepted).
  - The sampling rule is public and online, so the single-deviation policy is deployable. Abstain is its own class.
  - The G1a hindsight bound and the G1b cross-fitted ridge estimate are implemented. r3–r5 are refused.
  - Dev smoke (non-protocol): single-deviation H excl. abstain = .0013 per episode (90% CI .0004–.0024); G1b −.0005. Both suggest little headroom.
  - Registered A-HR launched: r0–r2 × iid_f0/iid_f2 × 256 worlds on seeds from 220M, plus the full-catalog subsample (~3.9k core-s).
- 2026-09-27T02:24Z: **A-HR gate G1 FAILS** (16 jobs, exit 0, 5.2k core-s; research/results/campaign-05/a-hr/analysis.json; 1,536 episodes, 512 world clusters).
  - Hindsight single-deviation H excl. abstain = **.0077 per episode** (90% CI .0041–.0122), below .02; incl. abstain .0134.
  - Per-state H excl. abstain .0014: median 0, p99 .027, 1.8% of states ≥ .01.
  - **G1b** (cross-fitted same-information single deviation) = **−.0021** (CI −.0042 to −.0004). Deviating from D on public estimates is harmful; the harmful rate is .11.
  - Full catalog vs O(I) per state: .0008 vs .0003.
  - **Reading:** one-step deviations from greedy + R-mask have negligible headroom.
  - Per design: register and test one bounded multi-step option class (**A-HR2**): (T) delegate to the public dep_reuse teacher until the next commit; (B) next-call budget override. Otherwise stop Track A.
- 2026-09-27T02:27Z: **B-LOC final** (12 existing models plus 3 BO oracle-belief models on the ext-04 split; challenge pairs U+E and S+C; bloc-final exit 0; one earlier launch failed with a missing output dir, charged).
  - First consequential error class in L4 (seeds 0–5): **value-estimate 57–66%**, ranking 23–38%, belief-dependent (fixed by oracle belief) 6–17%, deployment 0%.
  - L1: belief-dependent 14–37%; the rest cannot be classified without a Q head.
  - Realized regret .7–3.0, gap regret 1.8–3.0 per episode.
  - **Reading:** the historical combination failure is mainly **wrong value estimates for computations** under unseen condition combinations, not belief tracking or deployment. This is consistent with the extended-04 B-H6 finding that the value head generalizes poorly under shift.
- 2026-09-27T02:43Z: **B-HR gate** (B0 = L1 recipe on split v2, seeds 10–12; training ~375 core-s each; all exit 0):
  - **b5_hold_uc (U+C) is EVALUABLE:** uniquely-optimal first-probe rate .633/.650/.517 (60 supporting configurations each), below .80, so a failure exists to repair.
  - **b5_hold_se (S+E) shows NO FAILURE:** .983/.950/.967, reported as 'no failure to repair'.
  - U+C flag-sensitive subset: .727/.727/.576 (33 configurations), ambiguous.
  - B-X arms BX1/BX2/BX3 × seeds 10–12 launched. The primary endpoint is on U+C.
- 2026-09-27T02:50Z: **A-HR2 tooling merged** (e6fa7e99; 20 hr tests).
  - Option T (delegate to the public dep_reuse teacher until the next commit or 12 steps, then D). Option B is identical to the existing budget/call_now options, so it is read from the A-HR labels.
  - Dev smoke (non-protocol): T's hindsight gain comes entirely from rare episodes D fails; G1b-multi is ≈ 0.
  - The 5 pre-existing failures elsewhere in the full test suite (binding_study, main_freeze, s21_cpu, grounding_study, bf16) also fail on the base and are unrelated.
  - Registered A-HR2 launched: 12 jobs on the A-HR worlds.
- 2026-09-27T02:56Z: **B-X scored** (9 trainings and 9 evaluations, exit 0, 5.3k core-s; research/results/campaign-05/b-x/bx-score.json). **Primary NOT SUPPORTED for any arm on U+C** (the registered hold; S+E showed no failure). Per seed (10/11/12):

  | Arm | Uniquely-optimal probe | Regret | Gap regret | Not-optimal probe |
  |---|---|---|---|---|
  | B0 | .633/.650/.517 | 2.90/2.97/2.23 | 3.18/3.95/3.17 | |
  | **BX1 (exposure incl. historical pairs)** | .733/.800/.650 (+.10/+.15/+.13; the ≥ .15 bar passes 1/3) | **.32/1.46/.93** | **2.43/3.12/2.65** | .02–.06 |
  | BX2 (supplied belief) | .45/.60/.45 (lower) | 1.3–1.7 | | .008–.03 |
  | BX3 (modular per-flag) | .58–.67 | 3.7–5.2 (worse) | | |

  - **Reading:** broader exposure to other condition combinations partly improves probing and lowers regret on the unseen U+C combination, but below the registered threshold. Supplied belief does not fix probing (as predicted: belief = prior at the first decision); it makes the policy more conservative. The modular encoder hurts.
  - Together with B-LOC (value-estimate errors dominate), the failure looks like **acquisition of computation values under unseen interactions**, partly helped by combination diversity. It is not belief tracking and not a lack of factorized inputs.
  - Realized regret is noisy (it goes negative on the flag-sensitive subset); gap regret is the steadier measure.
- 2026-09-27T02:57Z: **A-HR2 PASSES both gates** (12 jobs, exit 0; research/results/campaign-05/a-hr2/analysis.json; 1,536 episodes, 512 clusters).
  - **G1a-multi** (hindsight, delegate option T): **.0261 per episode** (90% CI .0159–.0375), ≥ .02. T + O(I): .0283. The budget option B read from A-HR: .0061.
  - **G1b-multi** (cross-fitted same-information single deviation to T): **+.0214** (CI .0105–.0332). ≥ .01 and the CI excludes 0.
  - The gain comes from D-failure episodes (+.535, n = 70), with −.003 on D successes (n = 1,466). By firing anchor: call +.12, commit_revise +.19, reuse_recompute +.04 (CI includes 0).
  - Per design, proceed to A-PI with the passing option class: **learn when to delegate to the supplied teacher sub-policy**.
  - **Registered before A-PI (A-PI-T):**
    - the teacher alone (always-teacher, ~.90 vs D ~.87) plus simple trigger rules and random-matched delegation are comparators;
    - PI-T-1 = improvement over D;
    - PI-T-2 = value beyond simple rules (it must beat always-teacher and the rules while delegating less than always-teacher).
  - Scope: supplied sub-policy; the learned component is only when to invoke it.
- 2026-09-27T02:57Z: **B-Q registered** (diagnostic on inspected U+C): L4 Q-head models B0-L4 vs BX1-L4, seeds 10–12, to test whether exposure improves value estimates of computations (Q ranking). Launched.
- 2026-09-27T03:07Z: **B-Q (diagnostic; U+C already inspected)** (6 L4 trainings and 6 evaluations, exit 0).
  - BX1-L4 vs B0-L4 on U+C, per seed:
    - uniquely-optimal probe .717/.750/.733 vs .600/.467/.617 (+.12/+.28/+.12);
    - gap regret 2.36/2.76/2.42 vs 3.42/3.84/3.35.
  - **All-pairs Q-ranking accuracy does not improve** (.915–.932 vs .927–.938; ~345k pairs each).
  - **Reading:** exposure's benefit shows up in the decisions that matter (first-probe choice, gap regret), not in aggregate pairwise Q-ranking accuracy. That aggregate is dominated by easy action pairs and is not decision-relevant.
  - Consistent with the BX1-L1 result, combination exposure partially helps across two model families. The effect is sub-threshold per the registered B-X rule; there is no transfer claim.
- 2026-09-27T03:08Z: **B-XC registered (adaptive, labelled):** BX1 vs B0 L1 on fresh seeds 20–22, evaluated on **fresh U+C configuration draws** (new split b5c_hold_uc, seed base 5.9e9, same sizing rule). Primary: mean paired first-probe gain ≥ +.10 with all pairs > 0, and gap regret lower in 3/3.
- 2026-09-27T03:09Z: B-XC trainings launched (B0 and BX1 L1, seeds 20–22, existing bx-labels). The b5c evaluation split is being added by a builder; b5c labels will be built only after merge.
- 2026-09-27T03:22Z: B-XC tooling merged (f22ece8d). The b5c split (seed base 5.9e9) passes its audit 11/11; all b1/b5 paths are bit-identical; the scorer's --bxc implements the registered primary. B-XC labels and references launched.
- 2026-09-27T03:34Z: **A-PI-T controller frozen before evaluation** (81d1ebee): per-anchor ridge λ = 1, m = .075, 372 parameters, trained on the A-HR2 option-T labels (220M worlds, r0–r2); controller.json sha256 dea16e83…
  - Model and m were chosen by the G1b single-deviation criterion (.0247 per episode, CI .014–.036; all candidates within ~.003).
  - The builder changed the selection rule before any evaluation: the initial sum-over-firing-points rule over-credited delegation and always chose m = 0. The change is recorded in the artifact's rule_history.
  - Dev smoke (non-protocol): π_T is about +.05 over D and ≈ always-teacher while delegating ~7% of steps.
- 2026-09-27T03:34Z: **Incident.** Merging e05-apit conflicted in seed-ranges.json (both branches added ranges). A later `git commit -am` in the same command chain committed and pushed the merge **with conflict markers** (d558ad58). The empty-snapshot guard then refused all 12 A-PI-T launches, so nothing ran on broken code.
  - Fix: the union of both range sets was resolved by hand (28 ranges, verified pairwise disjoint), with no other markers in the tree.
  - Lesson recorded: never chain `commit -am` after a merge in the same command.
- 2026-09-27T03:39Z: A-PI-T screen launched (12 jobs, source-55a66cec; 42 tests pass on the snapshot, metered). The two entries above were first hand-stamped 03:42/03:43Z, ahead of wall-clock; corrected to their commit time 03:34Z. Timestamps from here on come from `date`/git only.
- 2026-09-27T03:40Z: **B-XC CONFIRMED** (adaptive registration, labelled). Fresh U+C configuration draws: b5c_hold_uc, seed base 5.9e9, 1,289 configurations, 60 eligible, 36 flag-sensitive eligible. Fresh training seeds 20–22 (B0 vs BX1, L1). Validity checks all pass. research/results/campaign-05/b-xc/bxc-score.json.
  - Mean paired uniquely-optimal first-probe gain **+.206** (per pair +.167/+.233/+.217), all > 0.
  - Gap regret lower in 3/3 (−.81/−1.09/−1.62).
  - **Claim:** training exposure to more condition combinations improves value-of-computation decisions (whether to probe first) and gap regret on an unseen combination, replicated on fresh configurations and fresh initializations.
  - **Caveats:** the U+C family had been inspected (B-X/B-Q); one held-out pair; a weak composition (the second flag matters in ~16% of configurations); the L1 recipe; the effect comes from **data exposure** (acquisition), not from supplied factorization (BX2) or modular structure (BX3).
- 2026-09-27T03:40Z: **Incident:** the 12 A-PI-T jobs first failed at startup because the controller had never been staged (the staging step was in the command chain broken by the merge conflict). Their receipts were renamed *-nocontroller-process (charged, ~0 s). The controller was staged with its hash verified (dea16e83…), and the 12 jobs were relaunched on source-55a66cec.
- 2026-09-27T03:50Z: **A-PI-T screen scored** (12 jobs, exit 0; research/results/campaign-05/a-pi-t/score.json; r0–r2 × 512 worlds per condition, seeds from 240M).
  - **PI-T-1 PASSES 3/3:** utility over D is +.035 (r0), +.044 (r1), +.036 (r2); success +.029/+.041/+.033.
  - **PI-T-2 FAILS 3/3:** π_T is slightly *below* the best simple rule R1 ('delegate at every call anchor') by −.0011/−.0012/−.0017 (pooled CI90 −.0037 to −.0004). It matches always-teacher (.893) while delegating only **6.4–7.7% of steps** (teacher 100%).
  - **Reading:** delegating to the better supplied sub-policy improves the deployed policy. The learned *timing* recovers the gain with little delegation but adds no utility beyond a simple trigger rule.
  - **A-CF-T registered (amends A-CF before any run):** the frozen controller is applied unchanged to fresh lineages r3–r5 on sealed worlds from 260M; primary PI-T-1 in 3/3. Launched.
- 2026-09-27T03:54Z: **A-CF-T launched** (12 jobs, source-63d52c02). The first launch (acf61800) failed at startup on a missing 'acf' address namespace; receipts renamed *-nons-process (charged, seconds). Fixed and smoke-verified on one job before launching the rest.
- 2026-09-27T04:04Z: **A-CF-T scored** (12 jobs, exit 0; research/results/campaign-05/a-cf-t/score.json). Frozen r0–r2 controller on fresh lineages r3–r5, sealed worlds from 260M, 512 per condition.
  - **PI-T-1 CONFIRMED 3/3:** utility over D +.0247/+.0215/+.0277; success +.021/+.017/+.023.
  - **PI-T-2 fails 3/3:** below the best simple rule (R2 'delegate at every commit_revise', .906) by −.0020/−.0018/−.0011. Always-teacher .904; delegated steps 5.5–7.4%.
  - **Confirmed claim:** a learned public-information controller that decides when to delegate to a supplied better sub-policy improves the imitation deployment on fresh lineages and sealed worlds. It adds no utility over simple trigger rules while delegation is free.
  - **A-PI-C registered** (a discriminating follow-up): a per-step delegation cost c ∈ {.001, .003} makes *when* to invoke matter. The controller is retrained on cost-adjusted existing labels; screen on 270M worlds, confirmation on 280M. Primary: beat max(D, teacher, R1, R2, random) + .005 in 3/3.
- 2026-09-27T04:18Z: **A-PI-C tooling and frozen controllers merged** (43f57727). c = .001: ridge λ = 1, m = .075 (dev G1b .0245). c = .003: MLP h16, m = .1 (dev G1b .0234). File sha256 7b7928c6…/e251ff5f…. At c = 0 A-PI-T is reproduced exactly. The screen evaluates both costs on the same worlds (270M, r0–r2).
- 2026-09-27T04:22Z: **Independent Track A audit merged** (3fa978e9). Every exactly defined number reproduces (A-HR G1a/G1b, A-HR2 G1a/G1b-multi, A-PI-T and A-CF-T PI-T-1/PI-T-2), and the gate and pass/fail outcomes stand.
  - **Integrity is clean:**
    - D branches reproduce the main line, with costs charged once (0 violations in 11,013 points).
    - Controller dea16e83 was committed (03:15:49Z) before the first protocol launch (03:39:01Z) and is used in all 384 evaluation metas.
    - D ≡ extended-04 r_mask (80/80); teacher ≡ dep_reuse (32/32).
    - The worlds are fresh and disjoint.
  - **Corrections adopted:**
    1. **"Improves over imitation" = "recovers the supplied teacher".** π_T's per-world success equals always-teacher's on 100% of worlds in 5 of 6 lineages (r5: 99.9%). It rescues the same D-failure worlds, and its gain over D matches the teacher's gap within .0009.
    2. **R2 dominates:** it matches or beats π_T in 6/6 lineages while delegating fewer steps (3.0–4.0% vs 5.5–7.7%). "Little delegation" is not a learned advantage, and the "delegates less than the teacher" clause of PI-T-2 is trivially met when delegation is free.
    3. PI-T-2 is worded "not better than the best rule" (several CIs include 0). The pooled π_T − R1 is −.0013 (CI90 −.0033 to −.0002); the earlier "pooled" CI was r2's.
    4. A-HR2 passes G1a on its point estimate only (CI lower bound .016), and all its value comes from 70/1,536 D-failure episodes.
    5. "Fresh lineages" means new imitation actors only; the teacher and controller are shared.
  - **Ledger refreshed** (the stale committed ledger lacked the A-CF-T receipts).
- 2026-09-27T04:22Z: **A-PI-C screen launched** (12 jobs, both costs on the same 270M worlds, source-da722185).
  - The first smoke job failed because snapshots exclude research/results, so the controllers were not staged. The receipt is kept (apic-r0-iid_f0-nocontroller). The controllers were staged via tar with hashes verified (7b7928c6…, e251ff5f…).
  - Smoke-then-launch discipline was used: one job was verified running before the other 11.
- 2026-09-27T04:33Z: **Independent Track B audit merged** (db1522c3). 750 point values reproduce within 1e-14; B-HR, B-X (not supported) and B-XC (confirmed; mean +.206, 95% CI .113–.307) stand.
  - **Integrity:** the split pools equal the registered generator; U+C and S+E appear 0 times in training/selection; b5c shares 0 of 10,688 earlier configurations; labels depend only on visible history (65 re-simulated episodes, own DP, 0 mismatches); the hold sizing reproduces; B-XC registration (03:08:34Z) precedes all B-XC trainings, labels and evaluations.
  - **Corrections adopted:**
    1. **B-LOC's 'belief-dependent' class is withdrawn as named.** Other seeds of the same public-input model fix first errors as often as the belief-supplied BO models (27% vs 21%), and 36% of the belief-classed errors sit at the first decision, where belief = prior. **No belief effect is detectable.**
    2. The B-LOC value-estimate share is **52–66%**, not 57–66%.
    3. "Not a lack of factorized inputs" is too strong. BX2 (supplied belief) lowers gap regret in 3/3 pairs (CI excludes 0), about as much as BX1, through **later** decisions; it does not fix the first-probe decision.
    4. B-Q's registered mechanism (exposure works through better value estimates) is **not supported**: the first-decision Q choice improves clearly in 1/3 seeds.
    5. B-XC caveats:
       - BX1 roughly doubles the overall first-probe rate, and its not-optimal probing rises in 3/3 pairs (still ≤ .10);
       - most of the gap-regret gain comes from later decisions;
       - the adaptive rule would also have passed on the inspected B-X data, so the evidence rests on the fresh replication;
       - B-XC also passes the stricter original B-X rule.
- 2026-09-27T04:35Z: **A-PI-C screen scored** (12 jobs, both costs on the same 270M worlds, r0–r2; research/results/campaign-05/a-pi-c/score.json).
  - **PI-C-1 passes 3/3 at both costs:** vs D +.030/+.044/+.039 at c = .001 and +.028/+.041/+.037 at c = .003.
  - **PI-C-2 FAILS 0/3 at both costs.** The cost removes always-teacher's advantage (teacher .872 at c = .001, .815 at c = .003, vs D .855–.869). But the simple rule **R2 ('delegate at every commit_revise anchor') remains best** (.899–.901), exceeding π_T by .0018–.0024 in every lineage. π_T delegates 4.4–6.3% of steps.
  - **No confirmation run** (PI-C-2 failed).
  - **Track A evidential boundary:**
    - (i) one-step deviations from D have no deployable headroom (A-HR);
    - (ii) delegating to a better supplied sub-policy improves the imitation deployment, confirmed on fresh actors and sealed worlds (A-CF-T; equivalent to recovering the teacher);
    - (iii) a learned public-information regulator of **when** to delegate adds no value beyond a simple trigger rule, under free and under costly delegation.
- 2026-09-27T04:35Z: **Both primary questions have reached an evidential boundary.** The campaign closes: final report, handoff, merge. The remaining budget (~125k core-s) is intentionally unspent.
- 2026-09-27T04:35Z: **Final report and handoff written; campaign complete.**
- 2026-09-27T05:35Z: **Report wording corrected after external review** (no numbers changed):
  - the 'no fixed rule can express' phrasing is replaced by 'beat strong, simple, equally informed heuristics and tuned portfolios';
  - B-XC is stated as conditional first-decision accuracy on the 60 eligible configurations, with per-seed rates;
  - exposure is confounded with data volume (384 configurations/7 combinations vs 768/9);
  - a possible generic probe-boundary shift is noted;
  - the B-LOC value-estimate finding is labelled an auxiliary-head observation, not a demonstrated cause.
