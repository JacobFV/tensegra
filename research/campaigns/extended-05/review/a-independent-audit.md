# Extended-05 Track A: independent audit (A-HR, A-HR2, A-PI-T, A-CF-T)

**Auditor:** independent; did not build the A-HR, A-HR2 or A-PI-T tools. Base: `campaign/extended-05` @ 88bd3b0c. Branch `campaign/e05-a-audit`.

**Method.**
- I pulled a read-only copy of the raw rows, via one metered `tar`:
  - `e05-hr/branch` + `full`;
  - `e05-hr2/branch`;
  - `e05-apit/{controller.json, eval}`;
  - `e05-acf/eval`;
  - all `hr-*`, `hr2-*`, `apit-*` and `acf-*` receipts, including the failed ones.
- I reconstructed every gate and claim from the rows with my own code (`research/tools/audit_e05a/`).
- I did **not** read the `campaign05_hr.py` analyze code or the `campaign05_apit.py` score code. I read only the world-construction and loader plumbing, `run_chunk`, and the charge docstring, after the reconstruction was done.
- One small remote re-run used the **unchanged extended-04** `deploy_episodes(mode="r_mask")` and the `dep_reuse` reference.

Machine-readable results: [a-independent-audit.json](a-independent-audit.json).

## Verdicts

| Item | Root | Auditor | Verdict |
|---|---|---|---|
| A-HR G1a: hindsight single-deviation H, excl. abstain, per episode | .00770 (CI90 .0041–.0122) | **.00770** (identical to 1e-12; CI identical) | **Reproduces; G1 FAILS** |
| A-HR G1b: cross-fitted same-information estimate | −.0021 | −.0022 (the root's margin grid m ≤ .05); −.0003 (a wider grid to .10). CI90 < 0 in both | **Reproduces (fail)**; the digits depend on the margin grid |
| A-HR2 G1a-multi (T) | .0260 (CI90 .0159–.0375) | **.0260** (identical) | **Reproduces; passes the point-estimate gate ≥ .02** (the CI lower bound .016 is below .02) |
| T + O(I) / option B | .0283 / .0061 | .0283 / .0061 | Reproduces |
| A-HR2 G1b-multi (T) | .0214 (CI90 .0105–.0332) | .0252 (single ridge; CI90 .0152–.0367). Per-anchor .0251; trigger anchors only .0225; other fold seed .0244; λ = 100 .0253 | **Reproduces (passes ≥ .01, CI excludes 0)**, robust to the estimator |
| A-PI-T PI-T-1 (r0/r1/r2) | +.0351/+.0439/+.0363 | identical | **PASS 3/3** |
| A-PI-T PI-T-2 | −.0011/−.0012/−.0017 vs R1 | identical | **FAIL 3/3** |
| A-CF-T PI-T-1 (r3/r4/r5) | +.0247/+.0215/+.0277 | identical | **PASS 3/3** |
| A-CF-T PI-T-2 | −.0020/−.0018/−.0011 vs R2 | identical | **FAIL 3/3** |
| A-CF-T per-condition floor (success ≥ .8 × dep_reuse) | pass | 12/12 lineage × condition cells pass; π_T success .967–.982 | Pass |

**Numeric discrepancies.** Zero on every exactly-defined quantity: G1a, G1a-multi, T + O(I), B, and all PI-T-1/PI-T-2 per-lineage values, success diffs, delegated-step fractions and comparator utilities.

The G1b estimators are independent re-implementations, with my own standardized ridge, fold assignment and margin grid. They agree on sign, magnitude and gate outcome.

The committed files are byte-identical to the remote results: `a-hr/analysis.json`, `a-hr2/analysis.json`, `a-pi-t/score.json`, `a-cf-t/score.json` and `controller.json`.

**Two documentation discrepancies (minor):**
- **decisions.md 03:50Z** says PI-T-2 is "pooled CI90 −.0037 to −.0004". That is **r2's per-lineage CI** vs R1, not a pooled value.
  - The actual world-clustered pooled value (3 lineages, IID group) of π_T − R1 is **−.0013 (CI90 −.0033 to −.0002)**.
  - In r0 the per-lineage CI includes 0 (upper +.00003).
- **`a-cf-t/score.json` notes** say always-teacher comes "from the x1-r0 jobs". In A-CF-T the teacher rows come from the **x1-r3** jobs (the r0 jobs were not run on 260M). The values are correct; the note text is stale.

## 1. Reconstruction details

**A-HR (G1a).**
- Per episode: max(0, max over the *publicly sampled* points and the non-abstain, non-D options of dU − q_d).
- 1,536 episodes; 512 world clusters (condition × seed; the same world across r0–r2 is one cluster).
- D-failure episodes (n = 70) carry the mass: .133 vs .0017 on D successes.
- Using every branched point on the all-states worlds changes it only to .0079.
- Per base × condition: .0030/.0110 (r0), .0021/.0172 (r1), .0061/.0067 (r2) for iid_f0/iid_f2.

**A-HR2.**
- **G1a-multi:** by D outcome, .559 on D failures (n = 70) and .0006 on D successes. So the gate passes only because of about 4.6% of episodes.
- **G1b-multi:** deviation rate .22–.29; harmful rate about .05. I did not re-derive decisions.md's "+.535 on D failures" breakdown of G1b-multi; the G1a-multi split above has the same shape.
- Restricting to the A-PI-T trigger anchors ({call, reuse_recompute, commit_revise}) gives .0225, so the anchor restriction used later does not change the gate.

**A-PI-T / A-CF-T.**
- The IID group (iid_f0 + iid_f2, 1,024 paired worlds per lineage) is recomputed from the row-level `rows`.
- The delegated-step fraction is pooled Σteacher_steps / Σsteps, which matches the root. The per-episode mean is also reported in the JSON.

## 2. Label validity
All counts come from `hr_audit.py`, over 11,013 points and 45,749 A-HR plus 22,026 A-HR2 options.

- **The D-branch reproduces the main line.**
  - For every point: q_d = U_final − U_t, the D option's dU = q_d, D's dsteps = T − step, D's index equals the main-line action at that step, and d_success equals the main-line success. **0 mismatches.**
  - The builders' embedded default checks: 1093/1093 (A-HR), 1093/1093 (A-HR2), 28/28 (full).
- **Costs are charged once.**
  - Every option satisfies dU = 1[success] − cost_rem.
  - Every episode satisfies utility = 1[success] − cost.
  - No final utility exceeds 1.
  - **0 violations.**
  - Every evaluation row (all six policies) also satisfies utility = success − cost.
- **Option-T termination.**
  - 11,013 delegate branches end by commit 7,447, by episode end 2,344, and by the 12-step cap 1,222.
  - Every cap-ended branch has exactly 12 teacher steps. None has more than 12.
  - Delegates whose teacher agreed with D on every step (non-commit end) reproduce q_d exactly.
  - Two branches end "commit" with no D continuation and `truncated=False`. Both are commits exactly at the 96-step cap (step 88 + 8), which is benign. The `truncated` flag is False at the cap in those rows, a cosmetic inconsistency.
- **The A-HR and A-HR2 main lines are identical**: 1,536/1,536 episodes, with the same actions, sampled steps, branched steps, q_d and telemetry at every shared state.
- **Sampling bookkeeping.**
  - The sampled flags equal `sampled_steps` in the decision log for every episode.
  - No episode has more than 8 sampled points.
  - Non-all-states episodes contain only sampled points.
  - There are 102 all-states episodes.
- **Hindsight vs same-information.**
  - G1a is labelled and used only as a necessary condition.
  - G1b is cross-fitted by world in both implementations. The same world across lineages is one fold unit, so no world is in both train and test.
  - The features are the pre-decision public telemetry (identical across the A-HR/A-HR2 branch files for the same state), the option's public descriptors and the teacher's *proposed* first action type.
  - No outcome key (success, dU, cost_rem, dsteps, teacher_agree, delegate_end) appears in any feature dict.
  - The one name-screen hit, `telemetry:feedback:status_success`, is the public feedback status of the last action, not the episode outcome.

## 3. Controller integrity
- **The hash matches everywhere.** controller.json sha256 `dea16e83…` matches in the repo, the remote staged file, and the `controller.file_sha256` field of **all 192 A-PI-T and all 192 A-CF-T** evaluation metas.
  - Every meta has margin .075 with `margin_override = null`.
  - The controller's 48 training files are the A-HR2 branch files on r0–r2 (hashes match 48/48).
- **Committed before evaluation.**
  - The controller was first committed in 03fe1b22 at 2026-09-27T03:15:49Z.
  - The first protocol evaluation launch (the failed *nocontroller* attempt) started at 03:39:01Z. The successful A-PI-T launches ran 03:39:42–03:48:46Z.
  - The staged controller file's mtime is 03:39:41Z, with its hash verified.
- **The selection-rule change came before any evaluation.**
  - The draft IPW controller (`agent-apit/controller-draft-ipw.json`, 03:13:21Z) selected ridge_l100 with m = 0.
  - The first-firing criterion check (`apit-ffcrit`, 03:14:17Z) and the retrain (`apit-train2`, 03:15:29Z) preceded the commit at 03:15:49Z.
  - All π_T smokes (03:16–03:19Z) used the final controller on development worlds 2.2505e9+ (not 240M).
  - So no protocol or dev evaluation of π_T preceded the change.
  - The change was made after seeing the draft's *training-label objective* (m = 0 selected), and it is disclosed in `rule_history`.
- **Features are public-only.** The 124 features are: c-telemetry (89), option descriptors, the teacher's proposed first action type, anchor one-hots and a bias. See §2 for the leakage screen.
- **A caveat to disclose.** The selection objective scores a single deviation at the first sampled trigger point, but π_T deploys the same m at *every* eligible decision, with multiple delegations and after its own delegations. This is disclosed in the tool docstring. It is not a flaw, but the offline objective is not the deployed policy's value.

## 4. Comparator correctness
- **D is the extended-04 r_mask deployment.**
  - `campaign04_deploy.py` sha256 c7a36b47… is identical at 02d54404, at main 3a566812, and in every meta's `sources`.
  - My re-run of the unchanged `deploy_episodes(mode="r_mask")` reproduces the stored D rows exactly (utility and steps) on **80/80 episodes**:
    - 240M r0/iid_f0 and r2/iid_f2;
    - 260M r3/iid_f0 and r5/iid_f2;
    - 220M r1/iid_f0 (the A-HR main line).
- **Always-teacher is the dep_reuse evaluation paired to the same worlds.**
  - The re-run of `run_episode(DepReference('reuse'), tariff 1)` reproduces the teacher rows on **32/32** episodes.
  - Teacher rows exist once per experiment (r0 jobs for A-PI-T, r3 jobs for A-CF-T) and are shared across lineages, which is valid because the teacher does not use the actor.
  - The seed sets are identical across all six policies and all lineages in every condition.
- **R1 and R2.** At row level:
  - R1 delegates at every eligible call anchor and nowhere else;
  - R2 delegates at every eligible commit_revise anchor and nowhere else;
  - D never delegates;
  - teacher_steps = steps for the teacher;
  - π_T never delegates outside {call, reuse_recompute, commit_revise}.
  - 0 violations.
- **Random matched rate.**
  - In all 384 chunk metas, `matched_random_rate` equals π_T's realized per-eligible delegation rate on the same chunk.
  - Realized pooled rates: π_T .035/.033/.044 vs random .035/.033/.046 (A-PI-T).
  - The rate is matched in-sample to π_T's realized rate, not a registered constant. That is acceptable for a matched-rate control.

## 5. World integrity
- **A-HR / A-HR2:** 220,000,000 + 100,000·i + n (n < 256), with i ∈ {iid_f0, iid_f2}.
  - The worlds are identical across r0–r2 and across A-HR and A-HR2.
  - Bases r0–r2 only (hashes 6733870c/786fed13/74ff10a6).
- **A-PI-T:** 240M + 100k·i, 512 per condition, 4 conditions. Identical across policies and r0–r2.
- **A-CF-T:** 260M + 100k·i, 512 per condition. Identical across policies and r3–r5 (5a25a83d/c7cd0d5b/e60b325b).
- The 240M and 260M sets are disjoint. The seed-range registry shows no overlap of 220M/240M/260M with any extended-03/04 range.
- **r3–r5 were used only in A-CF-T** in these experiments.
  - The only other r3–r5 use is the builders' dev-world D check, which is permitted.
  - The failed A-CF-T launch (`nons`) raised KeyError in the namespace lookup *before* any world was constructed, so the sealed worlds were not touched before the real run.
- **A-CF-T used the frozen r0–r2 controller unchanged**: same hash and margin, no override, no new labels.

## 6. Accounting

| Group | Receipts | Exit codes | CPU core-s | Launch window (UTC) | Source |
|---|---|---|---|---|---|
| A-HR branch | 12 | 0 ×12 | 3,913.8 | 02:09:34–02:23:19 | source-4debc95f |
| A-HR full catalog | 4 | 0 ×4 | 1,292.5 | 02:09:55–02:17:13 | source-4debc95f |
| A-HR2 | 12 | 0 ×12 | 1,539.2 | 02:51:07–02:55:26 | source-b3a64bbc |
| A-PI-T nocontroller (failed) | 12 | 1 ×12 | 13.2 | 03:39:01–03:39:16 | source-55a66cec |
| A-PI-T | 12 | 0 ×12 | 5,242.9 | 03:39:42–03:48:46 | source-55a66cec |
| A-CF-T nons (failed) | 12 | 1 ×12 | 13.1 | 03:51:52–03:52:07 | source-acf61800 |
| A-CF-T | 12 | 0 ×12 | 5,123.0 | 03:53:00–04:03:15 | source-63d52c02 |
| Root analyze/score/pretest (dev) | 6 | 0 ×6 | 300.3 | | |

- A-HR = 5,206 core-s, which matches "5.2k".
- The registrations precede their runs:
  - A-HR2 registered e29f06ba at 02:24:48Z, before 02:51Z;
  - A-PI-T registered f04720a0 at 02:57:10Z, before the controller;
  - A-CF-T registered e8ae3271 at 03:50:20Z, before 03:51:52Z.
- **Accounting gap (MAJOR, bookkeeping only).** The committed `research/results/campaign-05/receipts.json` / `budget.json` are last updated at 22d326cc (03:40Z).
  - They still list the 12 A-PI-T jobs as `running`.
  - They contain **none** of the 24 A-CF-T receipts (12 ok plus 12 failed `nons`) or the root score and pretest receipts.
  - The failed launches are documented in decisions.md and exist as receipts, but they are **not yet in the committed ledger**. The ledger must be refreshed before the report; about 10.4k core-s are missing.

## 7. Claims to weaken or rephrase

1. **"Improves over imitation" means reaching the supplied teacher, not beyond it.**
   - π_T's per-world success equals always-teacher's on **100%** of IID worlds in 5 of 6 lineages and on 99.9% in r5.
   - π_T rescues **exactly the same number of D-failure worlds as the teacher**: 33/43/36 (r0–r2) and 22/18/24 (r3–r5).
   - π_T − D equals teacher − D to within ≤ .0009 in every lineage (e.g. r1 .0439 vs .0442; r4 .0215 vs .0221). π_T utility is ≤ teacher in 5/6 lineages; A-CF-T pooled is +.00003 (CI90 −.0003 to +.0007).
   - The confirmed claim should read: *"A public-information controller that decides when to hand control to the supplied dep_reuse sub-policy recovers the supplied sub-policy's utility (+.021 to +.044 over the imitation deployment D); it does not exceed the sub-policy itself."*
   - The gain is the teacher-vs-clone gap, not a gain produced by learned improvement of the deployed policy.
   - It is **not** evidence for "deployment-aware policy improvement over competent imitation" in the brief's sense, because the teacher is the supplied competence the imitation came from.
2. **PI-T-2: simple rules dominate.** R2 ("delegate at every commit_revise anchor") matches or beats π_T in all 6 lineages (+.0001 to +.0020) while delegating **fewer** steps (3.0–4.0% vs π_T's 5.5–7.7%).
   - So "the learned timing recovers the gain with little delegation" is not a distinguishing property of the learned controller: a one-line public rule does it with less delegation.
   - Also, a delegated step costs the same as a D step, so the delegated-step fraction has no utility meaning. The PI-T-2 "less delegation than the teacher" clause is trivially satisfied and carries no evidential weight.
3. **PI-T-2 "below the best rule" wording.**
   - A-PI-T: vs R1, pooled −.0013 (CI90 −.0033 to −.0002). r0's per-lineage CI includes 0.
   - A-CF-T: vs R2, per-lineage CI90 upper bounds are +.00004/+.0002/+.000003; pooled −.0017 (CI90 −.0040 to +.0001).
   - Say "not better than the best simple rule (differences −.001 to −.002; CIs at or including 0)", not "below". Correct the mislabelled "pooled CI90 −.0037 to −.0004".
4. **The A-HR2 gate is marginal.** G1a-multi passed on its point estimate (.026 vs .02; CI90 lower bound .016). The whole hindsight value comes from 70 of 1,536 episodes (D failures). The design uses point estimates, so the pass stands, but the report should state that the pass is point-estimate-only and comes from D-failure episodes.
5. **Scope of A-CF-T "fresh lineages".** Only D differs across r3–r5; the teacher, the controller and the option are shared. The confirmation shows the delegation *trigger* transfers to new imitation actors. It does not show that a learned policy generalizes.
6. **Headline answer.** For the brief's question "Did the learned regulator add value beyond simple rules?", Track A's evidence is **no**, under free delegation. A-PI-C (a delegation cost) is the right discriminating follow-up.
7. **Minor.**
   - Fix the stale note in `a-cf-t/score.json` (the teacher source is x1-r3).
   - The controller's input includes a teacher consult at every eligible decision. It is priced inside the controller forward (disclosed); an A-PI-C cost analysis should consider re-pricing consults (`consults` × price) as the docstring suggests.

## Auditor compute
- **Remote (metered, pro6000, ≤ 1 core): 19.3 core-s.**
  - `aaudit-pack` 0.23.
  - `aaudit-dspot` ×4: 0.01 (exit 127, wrong python), 3.09 (exit 1, multiprocessing spawn needed a `__main__` guard), 3.49 (exit 1, result key), 12.47 (exit 0).
  - Unmetered: directory listings and two `ssh cat` transfers, under 1 core-s (estimated).
- **Local: about 640 core-s (estimated).**
  - The first `hr_audit.py` run used unrestricted multithreaded BLAS: 587 core-s user for 37 s wall. This exceeded the intended 2 cores briefly on the local machine; the remote host was unaffected.
  - Later runs were pinned to 2 threads, at about 16–21 core-s each.
- **Total about 660 core-s**, within 2,000. No results were modified, and the user's other workloads were untouched (load average about 1 during the re-run).

## Scripts (`research/tools/audit_e05a/`)

| Script | What it does |
|---|---|
| `common.py` | Loaders and the world-clustered bootstrap |
| `hr_audit.py` | Label validity, G1a/G1b, G1a-multi/G1b-multi |
| `eval_audit.py` | PI-T-1/2, comparators, pairing, worlds |
| `receipts_audit.py` | Receipts and timeline |
| `integrity.py` | Controller and output hashes, feature screen |
| `d_spotcheck.py` (remote) + `d_compare.py` | D and teacher re-run against stored rows |
| `compose.py` | Assembles the JSON |
