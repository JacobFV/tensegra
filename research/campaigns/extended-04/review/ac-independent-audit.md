# Independent audit: extended-04 Tracks A2 and C (plus the A1 matrix they rely on)

Auditor: independent (did not build the A2 options, the progress diagnostic, the deployment procedures, the Track C metacontroller or any scorer). Date 2026-09-26. Base `campaign/extended-04` @ b63aafd0, branch `campaign/e04-ac-audit`.

**Method.** Raw rows were pulled read-only from the pro6000 (`a1-*`, `a2s-*`, `a2-*-x1-r2/{state.json,development}`, `c-eval-*`, `c-train-*/*.json`, `c-labels-*/*.json`, every `*-process` receipt) and re-scored in pure Python by the auditor's scripts in `research/tools/acaudit/`. `campaign04_a2_promote.py`, `campaign04_c1_score.py`, `campaign04_a1_analysis.py` and `campaign04_c_analysis.py` were not opened until the reconstruction below was done. They were read afterwards only to explain two discrepancies (ECE averaging; the C-H5 comparison). The remote work was two torch-free checks: checkpoint hashes and a telemetry counterfactual replay. No result was modified.

Detail files are in this directory: `ac-audit-*.json` and the summary `ac-independent-audit.json`.

## Verdict summary

| Item | Root | Audit |
|---|---|---|
| A2 promotion | no arm promoted | **Reproduced exactly.** The utilities in a2-promotion.json match to 1e-15. Every criterion flag matches when "no-progress episodes" is read as the share of episodes with at least one no-progress step. |
| A2 integrity | seeds 140M, same worlds, deployment by P2a rule, one change per arm, exit 0 | **All confirmed** (details below). |
| A2 "localized reason" | 6 statements (21:35Z) | Statements (1), (5) and (6) need small wording fixes. **(2) is confounded and must be rewritten.** (3) holds. (4) holds with a weaker comparator. |
| C-H1 | NOT SUPPORTED | **Reproduced.** Learned is also significantly **worse** than the default on P1-RL r0 and r1. |
| C-H2 | passes, trivially | **Reproduced; triviality understated.** The pass rests on a handful of early stops in one seed per base. |
| C-H3 | loop-prone no; competent yes | **Reproduced.** The competent pass is as trivial as C-H2. |
| C-H4 | SUPPORTED | **Reproduced by rule**, but the pooled Spearman mostly measures that "stop" is predicted to be bad. The within-intervention ρ is only .18–.27. |
| C-H5 | "mostly unevaluable (learned gains ≤ 0)" | **Incorrect as stated.** P1-boot r1 has a positive learned gain (+.0021 ± .0022), and there the shuffled-target controller keeps all of it, so C-H5 fails on that base. The root comparison is also unpaired (128 vs 256 worlds). It passes only on P1-RL r2. |
| C integrity | 161M worlds, disjoint labels, default = r_mask, charge once | **All confirmed**, including a telemetry hidden-state counterfactual (0 violations). |
| A1 readings (i)/(ii) | as in decisions 20:40Z | **Reproduced** from raw rows. |
| Accounting | Track C evaluation 45.5k core-s vs 14k estimate | **Confirmed: 45,509 core-s (3.25×).** No job was capped or failed among 69 receipted jobs. The 12 empty-SHA launches left **no occupancy receipt**. |

## 1. A2 promotion (protocol-A2)

**IID-group table** (mean of iid_f0 and iid_f2, 256 + 256 worlds, seeds 140,000,000 + 100,000·i). Auditor numbers are from raw rows. "NP ep" is the share of episodes with at least one no-progress step. Root's a2s/a1-analysis.md table agrees on every entry at its printed precision; the utilities in a2-promotion.json agree to 1e-15.

| policy | greedy U | R-mask U | sampled U | greedy succ | greedy cost | NP ep (g / rm / s) | work/succ (g) |
|---|---:|---:|---:|---:|---:|---|---:|
| bootstrap X1-r2 | .8572 | **.8607** | .7945 | .936 | .0784 | .064 / .064 / .104 | 143.5 |
| reh | .8488 | .8483 | .8438 | .932 | .0829 | .082 / .082 / .117 | 155.2 |
| imit | .8507 | .8503 | .8416 | .932 | .0810 | .088 / .088 / .111 | 145.3 |
| crit | .8446 | .8558 | .8164 | .932 | .0871 | .076 / .076 / .186 | 163.9 |
| ent (final) | .8065 | .8168 | .8295 | .924 | .1173 | .086 / .086 / .148 | 333.9 |
| ent (deployed, attempt 24) | .8070 | .8144 | .8292 | .924 | .1168 | .088 / .088 / .186 | 330.3 |
| dep | .8588 | .8582 | .8243 | .943 | .0846 | .082 / .082 / .129 | 163.7 |
| C1 (P2a) | .8506 | .8511 | .8276 | .939 | .0889 | .064 / .064 / .174 | 178.5 |
| C0 (P2a) | −.1471 | .8305 | .8002 | .000 | .1471 | 1.0 / 1.0 / .873 | n/a |
| dep_reuse | .8999 | | | .977 | .0766 | 0 | 135.6 |

**Promotion rule applied by the auditor.** The bootstrap's best mode is R-mask (.8607).

| arm (mode) | c1 same-mode | c2 best-mode + .005 | c3 NP episodes | c4 (reh vs imit) | promoted | root |
|---|---|---|---|---|---|---|
| reh (greedy) | F | F | T | F | no | identical |
| imit (greedy) | F | F | F (.088 > .064 + .02) | n/a | no | identical |
| crit (greedy) | F | F | T | n/a | no | identical |
| ent (greedy) | F | F | F | n/a | no | identical |
| dep (R-mask) | F | F | T | n/a | no | identical |

- **Metric note.** The root flags match only under the episode-share reading of "no-progress episodes". The per-step rate would flip crit's c3 and imit's c3; the outcome is unchanged either way.
- **Construction note.** NP episodes are identical under greedy and R-mask for **every** policy (9/9). R-mask acts only after the first no-progress step, so criterion 3 cannot distinguish the two modes. Its constant value for the dep arm is an artifact of the metric.

**Integrity (all pass).**
- Screening seeds are exactly 140,000,000–255, 140,100,000–255, 140,200,000–255 and 140,300,000–255.
- Worlds (seed plus spec hash) are identical across all 9 policies × 3 modes plus dep_reuse: 1 distinct world list per condition. The row `deployment.mode` equals the directory's mode in every file.
- **Deployment rule re-derived from the raw development rows.** 30 files per run were recomputed and match state.json and its sha256, with 0 mismatches. The bootstrap development values (.953125 / .8808375) come from extended-03 p1-boot-x1-r2 state.json.
  - reh, imit, crit and dep: 30/30 qualify, so the final is deployed.
  - ent: 24/30 qualify, so attempt 24 is deployed (dev U .8613 ≥ .8608; attempts 25–29 fall below).
  - This matches the a2s-deployment-*.json files, including the checkpoint sha256.
- **Configs vs P2a base, as a flattened diff.**
  - reh = C1 + `rehearsal_weight .5`;
  - imit = C1 + `rehearsal_weight .5, rl_loss_weight 0`;
  - crit = C1 + `critic_warmup_updates 300, critic_warmup_shared false`;
  - dep = C1 + `rollout_mask progress_v1`;
  - ent = **C0** + the `entropy_*` fields (no `anchor_*`).
  - No other field differs, and the run state.json `a2.options` agree.
- **Hashes.** All 14 checkpoints referenced by the a2s and A1 configs were re-hashed on the pro6000: 0 mismatches.
- **Receipts.** A2 training: 5 jobs, exit 0, 23,112 core-s. A2 screening: 10 jobs, exit 0, 8,653 core-s. No job exceeded 90% of its CPU cap.

### Assessment of the "localized reason" (decisions 21:35Z)

Paired differences are over the same 512 IID worlds; SE is the paired SE. All are single lineage and single seed, with no multiplicity correction.

| # | Root statement | Data | Verdict |
|---|---|---|---|
| (1) | "none exceeds the bootstrap in any deployment mode" | True against the bootstrap's best mode (.861). Against the same mode, dep greedy is +.0016 ± .0033 above the bootstrap greedy, which is noise. | Reword to "none exceeds the bootstrap's best mode; same-mode differences are within noise (dep greedy +.002 ± .003)". |
| (2) | "RL reliably improves the objective it optimizes, the **sampled** policy (.794 → .816–.844)" | **Confounded.** The top of that range is reh (.844) and **imit (.842), which has the RL loss off**. Imitation-only gives the *largest* sampled gain (+.047 ± .012). The pure-RL arms gain less: crit +.022 ± .011, dep +.030 ± .012, C1 +.033 ± .012, ent +.035 ± .013. C0 gains +.006 ± .013. | **Correct it.** "Every stabilized training variant raises sampled utility, and imitation-only raises it most. The RL-only arms gain +.02–.035 (z 2.0–2.7, one lineage), and C0 gains nothing. The sampled gains are not attributable to RL." The second half, that sampled stays below the greedy/R-mask bootstrap, holds: reh sampled is −.013 ± .006 vs bootstrap greedy, imit −.016 ± .006. |
| (3) | "Rehearsal ≈ imitation-only (.849 vs .851)" | Greedy −.0019 ± .0005 (z −4), R-mask −.0020 ± .0005, sampled +.002 ± .006. reh has 7% more work per success than imit (CI 1.04–1.10). | Holds. It could be sharpened: adding RL to rehearsal is slightly but detectably **worse** under argmax deployment, not neutral. |
| (4) | "Entropy control more than doubles work/success (334 vs 143); the anchor effect is not just entropy control (anchor ~180)" | ent/bootstrap work per success is 2.32× (bootstrap CI 1.82–2.86). **ent/C1 is 1.87× (1.47–2.28)**. ent − C1 utility is −.044 ± .011 greedy and −.034 ± .010 R-mask. The training-batch entropy of ent (mean .81 over 30 tranche-final updates) is similar to the anchored C1-base arms (.87–.89), so the difference is not a matter of mean entropy level. | Holds, with two caveats. The fair comparator is the anchor: 1.9×, not "more than double". And the Lagrangian is **two-sided**: the dual goes negative (−.10 to +.09), so it also *penalises* entropy above target. Word it as "a mean-entropy equality constraint at the bootstrap's level does not reproduce the anchor's effect (one lineage, one seed)". |
| (5) | dep has the best greedy/R-mask RL utility (.859/.858), not above the bootstrap | Confirmed. dep R-mask − bootstrap R-mask = −.0025 ± .0027. | Holds. |
| (6) | "Every RL arm raises work per success" | reh 155, crit 164, ent 334, dep 164, C1 179 vs 143.5. imit (145) is not an RL arm. | Holds if restricted to arms with RL on. |

The concluding sentence ("on-policy actor-critic finds no deployable improvement over imitation") is supported as single-lineage screening evidence.

## 2. Track C (protocol-C1)

**IID group per base.** Learned is the mean over 3 meta seeds. Per-seed learned values, random, threshold and appraisal-only agree with c1-score.json to ≤ 1e-15.

| base | greedy | sampled | R-mask (D0) | R-sample | appraisal_only | learned (s0/s1/s2) | learned − best fixed (paired SE) | random_matched | threshold | learned interv. rate |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| P1-RL r0 | −.044 | .719 | .771 | **.781** | .771 | .754 (.776/.742/.743) | −.027 | .602 | .787 | .117 |
| P1-RL r1 | .819 | .758 | **.822** | .817 | .822 | .798 (.790/.789/.816) | −.023 | .648 | .822 | .142 |
| P1-RL r2 | −.148 | .766 | **.813** | .811 | .813 | .831 (.832/.835/.826) | **+.018 ± .005** | .726 | .823 | .030 |
| boot r0 | .850 | .782 | **.851** | .851 | .851 | .840 (.822/.849/.850) | −.011 | .748 | .844 | .120 |
| boot r1 | .841 | .792 | .849 | **.852** | .849 | .851 (.850/.855/.849) | −.001 | .818 | .849 | .006 |
| boot r2 | **.852** | .790 | .851 | .849 | .851 | .851 (.851/.851/.850) | −.001 | .844 | .847 | .0004 |

Learned vs the default (D0), paired over 512 worlds:

| base | learned − D0 | SE |
|---|---:|---:|
| RL r0 | −.017 | .005 |
| RL r1 | −.023 | .006 |
| RL r2 | +.018 | .005 |
| boot r0 | −.011 | .004 |
| boot r1 | +.002 | .002 |
| boot r2 | −.0002 | .0006 |

**Decision rules applied by the auditor:**

- **C-H1: NOT SUPPORTED (0/3).** This agrees with root.
  - r2 misses by .002: +.0179 against .02. Seed 1 alone reaches +.022, but the registered unit is the seed-mean.
  - r2's gain is mostly early **stops**. There are about 45 stops per 512 episodes per seed, cost falls from .099 to .075 (−24%) and success falls from .912 to .906.
  - **Missing from root's statement:** on r0 and r1 the learned controller is significantly *worse* than the default (−.017 and −.023, z ≈ −3.5). It is not merely "not better than the best fixed rule".
- **C-H2: passes by rule (2/3: boot r1 and r2).** This agrees with root.
  - The pass is **degenerate**. On boot r2, seeds 0 and 1 never intervene (m = ∞); they cost *more* than D0 by the meta charge. The cost reduction (−.00045) comes entirely from seed 2's 14 stops and 4 mask-tops in 1,024 episodes. On boot r1, it comes from 25 stops (seed 0) and 232 interventions (seed 1); seed 2 never intervenes.
  - "Cost … below the default's" has no margin, so any single early stop of a failing episode satisfies it.
  - protocol-C1 says "≥ default − .01" while trackc.md says "≥ best fixed − .01". Both give the same outcome here.
- **C-H3: loop-prone 1/3 (NOT SUPPORTED); competent 2/3 (passes).** This agrees with root.
  - The competent pass needs learned > threshold. On boot r1 the registered τ = 0 in all seeds, so "threshold" *is* D0, and learned wins by +.002 ± .002. On boot r2, learned (≈ D0) beats a threshold rule that intervenes at 6.9% of decisions.
  - "Deliberate beats automatic" is therefore not shown. What is shown is that doing almost nothing beats random or threshold intervention on competent bases.
- **C-H4: SUPPORTED by rule (3/3 per type).** This agrees with root.
  - **ECE of P(success).** The auditor pools both conditions; root averages the per-condition ECEs, and this explains the small differences. All are ≤ .05, so the verdict does not change.

    | base | auditor step-level | first decision | branch default | root |
    |---|---:|---:|---:|---:|
    | RL r0 | .025 | .043 | .031 | .026 |
    | RL r1 | .020 | .003 | .050 | .019 |
    | RL r2 | .015 | .009 | .015 | .016 |
    | boot r0 | .009 | .009 | .003 | .009 |
    | boot r1 | .010 | .008 | .015 | .014 |
    | boot r2 | .015 | .007 | .016 | .017 |

  - **Spearman (appraisal-only branch points).** The auditor reproduces root's ρ exactly: .551, .631, .658, .725, .644 and .746, and the lower CI bound is > 0 everywhere.
  - **Caveat (flag):** the pooled correlation mixes three intervention types. "Stop" has large negative predicted and actual effects, so the pooled ρ is mostly between-type separation. **Within-type ρ averages only .18–.27.**
  - **Per-type ρ (appraisal-only; auditor episode-cluster bootstrap CIs):**
    - stop: .24–.50, all CIs > 0;
    - mask_top: .33–.57 on 5 bases, but **.07 (CI −.08 to .20) on RL r0**;
    - sample: −.18 to .23. RL r0 is +.23 (CI excludes 0), and boot r2 is −.18 (CI below 0, i.e. anti-calibrated).
  - Root's "stop and mask-top effects are predicted (ρ .24–.57)" should exclude RL r0's mask_top. Root's CIs resample branch points rather than episodes, but the CIs are similar.
- **C-H5.** Root says "mostly unevaluable (the learned gains are ≤ 0)". **That is inaccurate.**
  - Boot r1 has a positive learned-vs-default gain (+.0021 ± .0022, lineage level; +.0028 for seed 0 on the 256 causal worlds). There, the shuffled-target controller keeps **115%** of the gain (+.0032 on the same worlds), so **C-H5 fails on boot r1** by the literal rule. The gain is itself noise.
  - On RL r2, the shuffled-target controller collapses: .374 vs learned-s0 .849 and D0 .832 on the same 256 worlds, i.e. it loses 28× the gain. So the rule passes.
  - The comparison is weak. The shuffled-target controller intervenes at almost every step (e.g. 93 mask_top in a 96-step episode), so *any* broken controller would lose the gain. The retrained drop-group models and the zero-group ablations on RL r2 mostly keep the gain: zero-diagnostic, zero-budget, zero-control and drop-info_events each give +.017 to +.021 vs D0. No single telemetry group is necessary for it.
  - RL r0 learned-s0 also has a positive gain on the causal worlds (+.0033), and shuffled loses it. But the lineage-level gain is negative, so this is unevaluable at lineage level.
  - **Method flaw in c1-score.json:** `shuffled128_minus_default256` compares shuffled on the first 128 worlds with D0 on all 256. The paired D0 and learned-s0 rows for those same 128 worlds exist in the tier-1 files. trackc.md §11 specified "compared with the tier-1 learned rows on the common worlds". The unpaired value misstates the effect: boot r1 +.011 unpaired vs +.003 paired; RL r1 +.011 vs −.00005.
  - C-H5 also has no lineage-count rule. The honest reading is "passes on 1 base (RL r2), fails on 1 (boot r1, noise-level gain), unevaluable on 4".
  - "Appraisal-only equals the default" holds exactly (see integrity).
- **Transfer (secondary).** All bases meet the .8 × D0 success floor on events and foreign4.
  - On RL r2, learned beats D0 on both transfer conditions: events +.028, foreign4 +.023.
  - Elsewhere, learned is ≤ D0 on transfer.

**Track C integrity (all pass):**

- **Evaluation worlds.** Every fixed, learned, random, threshold and appraisal row uses seeds 161,000,000 + 100,000·c + 0…255 in order. No 160M (smoke-touched) world appears. Causal jobs use the first 128 per IID condition.
- **Label worlds.** 24 chunks × 32 per base give contiguous 150,000,000–150,000,767, the same for all 6 bases. They are disjoint from the 161M evaluation range, A1 (130M) and A2 (140M). All 144 chunk files carry one identical source-hash set, which equals the evaluation jobs' (apart from the evaluation tool).
- **Default path.** appraisal_only against fixed:r_mask over 6 bases × 4 conditions × 256 worlds:
  - identical step-class strings;
  - `utility_no_meta` equal to the fixed row's utility (maximum difference 2.2e-16);
  - 0 interventions.
- **Meta charge applied once**, over 86,016 rows:
  - utility = success − cost in every row;
  - `utility_no_meta − utility` = meta_forwards × charge_per_forward (≈ 1.28e-6), with 0 mismatches;
  - compute_units = decisions + meta units, with 0 mismatches;
  - fixed arms carry no charge.
  - Disclosure: random_matched rows have **0 meta forwards and no charge**, since the random arm needs no forward. This is correct but asymmetric, at about 5e-5 utility, which is immaterial.
- **Label and branch compute is excluded from utility.** Worlds carrying branch-evaluation points (the first 64 per IID condition) have appraisal_only utilities identical to fixed:r_mask. Label jobs are separate receipts.
- **Registration.** m and τ were re-derived from each training record's dev tables (argmax gain; ties go to larger m and smaller τ) for all 18 main models: 18/18 agree.
- **Telemetry hidden-state spot check** (auditor's `telemetry_spotcheck.py`, run in source-b7d5d8e3, metered).
  - Method: for 40 recorded worlds, the auditor built counterfactual worlds that differ only in hidden content (the event argument; one item's price and weight), replayed the identical recorded actions with the real ProgressTracker and TelemetryRecorder, and compared telemetry step by step.
  - Result: 60 pairs, 796 identical-prefix steps, **0 cases of telemetry diverging before the public observation diverged**.
  - Static read of `campaign04_telemetry.py`: inputs are DepObservation, the tracker and policy statistics only.
  - Limitation: policy statistics were held fixed. The actor sees only the observation encoding.

## 3. A1 spot checks (raw rows)

IID group; seeds 130,000,000 + 100,000·i; one world list per condition shared by all policies and modes.

| policy | greedy S / U | R-mask S / U | R-sample S / U | sampled S / U |
|---|---|---|---|---|
| P1-RL r0 | .078 / −.038 | .865 / .772 | .881 / .787 | .838 / .715 |
| P1-RL r2 | .000 / −.146 | .918 / .824 | .920 / .826 | .885 / .776 |
| P1-RL r1 | .904 / .825 | .916 / .836 | .914 / .832 | .865 / .766 |
| boot r0 | .949 / .8706 | .949 / .8707 | .949 / .8708 | .885 / .799 |
| boot r1 | .934 / .8572 | .938 / .8603 | .938 / .8603 | .875 / .793 |
| boot r2 | .943 / .8656 | .943 / .8651 | .943 / .8652 | .885 / .801 |
| P2a C1 | .939 / .854 | .943 / .856 | .941 / .853 | .910 / .822 |

References: dep_reuse .9085, dep_recompute .9035.

Readings (i) and (ii) in decisions 20:40Z reproduce:
- recovery of r0 and r2;
- the R-mask − greedy difference on the bootstraps is +.0001, +.0031 and −.0005 ("±.003"; r1 is at the edge);
- the no-progress step rate falls 6.1–7.1×;
- sampled is the worst mode for competent policies (−.065 to −.072);
- dep_reuse and dep_recompute are .908 and .904.

## 4. Accounting (job receipts, occupancy.json)

| group | jobs | core-s | exit ≠ 0 | capped |
|---|---:|---:|---:|---:|
| A1 | 8 | 4,844 | 0 | 0 |
| A2 training | 5 | 23,112 | 0 | 0 |
| A2 screening | 10 | 8,653 | 0 | 0 |
| C labels | 18 | 7,352 | 0 | 0 |
| C labels, empty-SHA launches | 12 | (no occupancy.json) | n/a | n/a |
| C train | 6 | 1,006 | 0 | 0 |
| C evaluation tier 1 | 24 | 36,452 | 0 | 0 |
| C evaluation causal | 6 | 9,057 | 0 | 0 |
| **total** | 89 | **90,477** | 0 | 0 |

- **Track C evaluation is 45,509 core-s**: 3.25× the 14k estimate. Tier 1 is 3.5× its 10.4k estimate and causal is 2.5× its 3.6k estimate. The largest single job was 2,050 core-s against a 4,000 cap.
- Track C total (labels, training and evaluation) is 53.9k against the ~24k plan, 2.2×.
- **The 12 "*-emptysha" launches have only launch.json and an empty process.log, and no occupancy.json.** Their charge cannot be verified from a receipt; it is ~0. Their cwd was `source-env`, and argv[0] was the literal `CUDA_VISIBLE_DEVICES=`. So they would have failed on exec regardless of the snapshot. Nothing was written to the label directories: all 144 chunk files carry one source-hash set.

## 5. Claims to weaken or correct

1. **A2 (2):** rewrite as described in §1. The sampled-mode improvement is largest with RL *off*, so it cannot be credited to RL.
2. **A2 (1):** use "not above the bootstrap's best mode" rather than "in any mode".
3. **A2 (4):** use the anchored C1 as the comparator (1.9× work, −.044 utility greedy). Disclose that the entropy dual is two-sided. Frame the finding as "a mean-entropy constraint does not reproduce the anchor".
4. **A2 c3:** disclose that "no-progress episodes" is identical under greedy and R-mask by construction.
5. **C-H1:** add that learned control **harms** RL r0 and r1 relative to D0 (−.017, −.023, z ≈ −3.5). Add that the r2 gain comes mainly from early stops (cost −24%, success −.006).
6. **C-H2 and C-H3 (competent):** report them as degenerate passes. They come from about 25 stops in one seed per base, cost criteria without a margin, and a threshold rule whose registered τ is 0 (identical to D0) on boot r1. Do not read them as "trims waste" or "deliberate > automatic".
7. **C-H4:** keep SUPPORTED as registered. Report the within-type ρ (.18–.27), note that pooling is dominated by stop, drop RL r0 mask_top from "predicted", and note the negative sample-step ρ on boot r2.
8. **C-H5:** replace "mostly unevaluable (learned gains ≤ 0)" with "passes on RL r2; fails on boot r1 (shuffled keeps the +.002 noise-level gain); unevaluable elsewhere". Recompute paired on the common 128 + 128 worlds. Note that the shuffled control is a weak test, and that no single telemetry group is necessary for the r2 gain.
9. **Budget:** keep the 45.5k (3.25×) disclosure. Add that the empty-SHA receipts lack occupancy files.

## 6. Auditor resource use

- **Metered (pro6000), 575.1 core-s:**
  - pull tar: 0.5;
  - hashes: 0.4;
  - telemetry spot check: 270.5;
  - a first telemetry attempt killed at its 300 core-s cap by a spawn-guard bug in the auditor script: 303.7.
  - Receipts are under `results/dev/acaudit-*`.
- **Local (auditor workstation, no torch), ≈ 250 core-s (estimate):**
  - ≈ 160 core-s of Python analysis, from `time`;
  - plus tar extraction and transfer.
- **Scripts** (research/tools/acaudit/): a2_reconstruct.py, a2_config_diff.py, a2_deploy_rederive.py, a2_paired.py, a1_spot.py, c_reconstruct.py, c_supplement.py, c_causal.py, accounting.py, hash_check.py and telemetry_spotcheck.py.
