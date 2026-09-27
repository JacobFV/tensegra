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
