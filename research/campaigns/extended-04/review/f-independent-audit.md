# Independent audit: extended-04 Phase F (F1, F1b, F2) and Track B2

- **Auditor:** an independent agent. I did not build these experiments or their scorers.
- **Base:** `campaign/extended-04` at 138da719. Branch: `campaign/e04-f-audit`.
- **Method:** I rebuilt every reading from raw data before opening `campaign04_f2_score.py`, `campaign04_b2_score.py`, `f2-score.json` or `b2-score.json`, then compared.
  - F1/F1b: raw depworld rows (`*.jsonl.gz`) and `worlds.jsonl.gz`.
  - F2: `eval.json` and `f-brefs.json`.
  - B2: `model.pt`, `eval.json` and `failure_records.json`.
- **Scripts:** `research/tools/audit_f/` (`f1_audit.py`, `f2_audit.py`, `fb_remote.py`, `extra.py`, `compose.py`; `peek.py` inspected the schema).
- **Machine-readable output:** [f-independent-audit.json](f-independent-audit.json).
- **Results were not modified.**

## Verdicts

| Test | Root | Audit |
|---|---|---|
| F1-H (registered, episode clause) | NOT SUPPORTED | **Reproduced: NOT SUPPORTED.** Utility and success pass 3/3; the episode clause fails 3/3 with identical rates. |
| F1b-H (adaptive, per-step clause) | SUPPORTED | **Reproduced: SUPPORTED 3/3.** Correctly labelled adaptive; the headline wording needs qualifying (below). |
| F1 mechanism ("episode rate cannot fall") | spec error | **Confirmed at row level:** 48/48 cells. |
| F2-H1 to H4 | all SUPPORTED | **Reproduced exactly.** Every number matches `f2-score.json`. |
| F2 heldout_comp per-split failures | .545; .51–.58 | **Reproduced.** The switching shortfall is mostly downstream of the probing shortfall (below). |
| B2-P (primary) | NOT SUPPORTED | **Reproduced: 0/3 seeds.** |
| B2-S1 (policy unchanged) | holds | **Confirmed.** 22/22 B1 tensors are bit-identical in 6/6 runs, and there are 0 eval-key mismatches. |
| B2 reading | "not continuation" (reading 2) | **Reproduced by rule, but fragile.** It rests on a .051 vs .05 dev miss and on the declared linear-probe confound. |

## 1. F1 / F1b (depworld, fresh bootstraps r3–r5)

### 1.1 Reproduction, IID group (iid_f0 + iid_f2, 1,024 episodes per lineage)

The per-step rate is Σ no_progress / Σ steps. The episode rate is the share of episodes with no_progress > 0. Both are read from the rows' `progress` diagnostic fields.

| run | lineage | ΔU (r_mask − greedy) | ΔS | NP-episode rate g / m | NP per-step rate g → m | ratio | episode clause | step clause |
|---|---|---:|---:|---|---|---:|:-:|:-:|
| F1 (170M) | r3 | +.0007 | +.0010 | .0332 / .0332 | .0132 → .0033 | 3.97× | fail | pass |
| F1 (170M) | r4 | −.0015 | .0000 | .0576 / .0576 | .0522 → .0133 | 3.94× | fail | pass |
| F1 (170M) | r5 | +.0030 | +.0039 | .0596 / .0596 | .0771 → .0104 | 7.38× | fail | pass |
| F1b (180M) | r3 | +.0002 | +.0010 | .0508 / .0508 | .0283 → .0059 | 4.78× | fail | pass |
| F1b (180M) | r4 | +.0010 | +.0020 | .0684 / .0684 | .0469 → .0124 | 3.77× | fail | pass |
| F1b (180M) | r5 | +.0019 | +.0029 | .0625 / .0625 | .0763 → .0111 | 6.91× | fail | pass |

- **All values match the root's `a1-analysis.md` tables** to the printed precision. That covers success, utility, cost, NP rate, NP episodes and interventions per episode, in every cell including events and foreign4.
- **The floor holds everywhere:** r_mask success ≥ .8 × dep_reuse in all 24 lineage × condition cells, in both runs. dep_reuse's IID score is .900 (F1) and .901 (F1b).
- **Verdicts:**
  - F1-H (episode clause) is NOT SUPPORTED.
  - F1b-H (per-step clause) is SUPPORTED in 3/3 lineages.
  - The per-step clause would also have passed on the F1 worlds in 3/3 lineages.
- **Rounding slips in the prose, not in the tables:**
  - decisions.md 21:50Z says r5 ".077 → .011"; it is .0104.
  - decisions.md 21:58Z says r4 ".047 → .013"; it is .0124.
  - The report headline says "74–86%" of stagnant steps removed; F1b gives 73.5 / 79.1 / 85.5%, so it should read **73–86%**.

### 1.2 Mechanism: why the episode rate cannot fall

I checked this in every lineage × condition cell of both runs: 24 cells, 48 with the two runs counted separately. In every cell:
- **The set of no-progress episodes is identical** under greedy and r_mask. No episode is flagged only under one mode.
- **In every flagged episode, the two action sequences are identical up to and including the first no-progress step.** The first-flag index is also the same.
- **Every episode without a flag has a byte-identical action trajectory** under both modes.

So R-mask equals greedy until the diagnostic first flags a step. The episode-level rate is therefore identical by construction, as the root states. Protocol-F's episode clause could only fail, and pre-registration review should have caught that.

### 1.3 Integrity

- **Worlds:**
  - Each condition has seeds `start + 100,000·i + [0, 512)`: F1 from 170,000,000, F1b from 180,000,000.
  - Each condition has 512 distinct spec hashes.
  - The world file matches every row's spec_hash.
  - The spec is identical per seed across r3/r4/r5 × greedy/r_mask and the dep_reuse reference.
  - **The F1 and F1b world sets are disjoint:** 0 of 2,048 spec hashes are shared.
  - **No overlap with other extended-04 depworld evaluation worlds** on disk: A1, 29,696 worlds; A2 screening, 28,672 worlds.
  - The 170M and 180M ranges also lie outside every registered range:
    - extended-03: 110M, 120M, 1.99e9, 2.0–2.1e9, 3.0–4.0e9;
    - extended-04: A1 130M, A2 140M, Track C labels 150M and eval 161M, probeworld 4.1–4.7e9 and 8e9+.
    - Legacy configs `campaign-a07/a08` use 171M–184M seed starts. These are a different generation, and none of them falls in [170.0M, 170.3M+512) or [180.0M, 180.3M+512).
- **Bootstraps:**
  - Configs r3–r5 are recipe-identical to the extended-03 P1 bootstraps r0–r2: `diff` of everything except the seed fields is empty.
  - The seed fields are fresh and disjoint:
    - training streams 3.06e9, 3.08e9 and 3.10e9, against 3.00/3.02/3.04e9 and the RL streams from 3.40e9;
    - development streams 3.903e9–3.905e9, against 3.900–3.902e9;
    - initialization seeds 300300+, 300400+ and 300500+;
    - population seeds 30003–30005.
  - Each F1/F1b config names `round-0-slot-5-attempt-5.pt`, the newest of 6 checkpoints by mtime. Its recomputed SHA-256 matches the config (5a25a83d…, c7cd0d5b…, e60b325b…). No selection was made.
- **Receipts:** all have exit code 0.
  - f-boot r3–r5: 627, 661 and 656 core-s.
  - f1-* (3 runs plus references): 1,317 core-s.
  - f1b-* (3 runs plus references): 1,079 core-s.
  - F1 and F1b used `--device cuda`, which the protocol does not forbid.
- **Pre-registration order,** from git commit times and the source snapshot each job ran from:

  | Step | Time (UTC) |
  |---|---|
  | protocol-F committed (7655c0b0) | 21:40:05 |
  | F2 trainings launched from `source-7655c0b0` | 21:40:18 |
  | F1 launched from `source-4cdece88` | 21:40:57 |
  | F1 scored (ed632338) | 21:50:45 |
  | F1b registered (08e85c0e) | 21:51:18 |
  | F1b launched from `source-08e85c0e` | 21:51:27 |
  | F1b scored (138da719) | 21:58:16 |

  - The order is sound. Each job ran from the snapshot of the commit that registered it.
- **Decision-log timestamps are still wrong after the 21:52Z "housekeeping" fix:**
  - "21:50Z protocol-F registered": it was committed at 21:40Z.
  - "21:55Z … F1 launched; F2 launched": the launches were at 21:40–21:41Z.
  - "21:47Z F2 scored": the F2 evaluations ended at 21:49:23Z, and the commit was at 21:49:57Z.
  - The entries are also out of chronological order.
  - None of this changes an ordering conclusion, but the log should be corrected from git and receipts.

### 1.4 Is F1b's adaptive registration adequately labelled?

- **Mostly yes.** The F1b addendum, the decisions entries and the report's §5 all say "adaptive", keep F1 recorded as NOT SUPPORTED, and use new sealed worlds. The worlds are verifiably fresh (§1.3).
- **The headline row in `report.md` ("Supplied rule, confirmed (F1b)")** should also carry the word *adaptively registered*. The per-step effect had already been observed on the F1 worlds before F1b was registered, so F1b is a fresh-world replication of an observed effect, not a blind confirmation.

### 1.5 What F1b does and does not show (claims to qualify)

**Per-step reduction:**
- The per-step reduction is close to mechanical. R-mask masks exactly the actions that its own diagnostic flags, and the effect is then measured with that same diagnostic.
- The informative content of F1b is **non-inferiority**: utility and success do not fall.
- It is **not** evidence that stagnation is resolved at the episode level. Stalled episodes are exactly as frequent as before (§1.2), and the to-cap rate barely moves (.048 → .046, .041 → .038, .053 → .049).

**Paired effects,** IID group, 1,024 episodes, differences only on the ~50–70 flagged episodes:

| run / lineage | ΔU ± 95% CI | successes rescued / lost | Δcost per episode |
|---|---|---|---:|
| F1 r3 | +.0007 [−.0011, +.0025] | 1 / 0 | +.0003 |
| F1 r4 | **−.0015 [−.0025, −.0005]** | 0 / 0 | +.0015 |
| F1 r5 | +.0030 [−.0008, +.0068] | 4 / 0 | +.0009 |
| F1b r3 | +.0002 [−.0017, +.0021] | 1 / 0 | +.0007 |
| F1b r4 | +.0010 [−.0017, +.0037] | 2 / 0 | +.0009 |
| F1b r5 | +.0019 [−.0015, +.0053] | 3 / 0 | +.0010 |

- **Cost rises in 6/6 runs.** Work per success rises by 3–5% (root tables).
- In F1 r4 the utility difference is **significantly negative**, although it stays inside the .005 margin.
- **Suggested wording:** "R-mask is utility-neutral (non-inferior at the registered .005 margin) on fresh competent lineages; it removes 73–86% of the steps its diagnostic flags as stagnant, rescues 0–4 successes per 1,024 episodes, never loses one, and slightly raises cost; it does not reduce the share of episodes that stall." The phrase "at no cost" should become "at no utility or success cost".

## 2. F2 (probeworld, fresh seeds 3–5, world offset 700)

### 2.1 Reproduction

- **Inputs:** my recomputation reads the committed `research/results/campaign-04/f2/**/eval.json` and `f-brefs.json`.
  - They are byte-identical to the remote files (SHA-256; 9 eval.json + f-brefs).
  - `train_meta` gives seeds 3/4/5, rungs L0/L1/L4, 4,000 × 64 updates and 129,189 parameters.
- **Every number below equals `f2-score.json` exactly.**

**H1: held-out mean regret, and whether it is ≤ .8 × L0.**

| seed | L0 | L1 | L4 | pass |
|---|---:|---:|---:|---|
| 3 | 38.98 | 1.86 | 2.57 | ✓ ✓ |
| 4 | 82.34 | 2.57 | 1.70 | ✓ ✓ |
| 5 | 72.46 | 2.00 | 1.51 | ✓ ✓ |

- L1 and L4 are also far below L0 on every split separately.
- The transfer floor (success ≥ .8 × π*) passes everywhere, but it does not bind: even L0's success is ≥ .98.
- References: π* regret is −1.4 to .4; the fixed rules score 99–126.

**H2: L4 effective-ρ build deviation, mean |Δ| over bins.**
- heldout_k .032 / .027 / .023; heldout_price .016 / .040 / .042. It passes 3/3.
- The n-weighted variant also passes.
- **The test discriminates:** L0 scores .51 here.
- One heldout_price bin has only n = 4 and still gets equal weight.

**H3: first-probe rate, pooled and n-weighted.**
- When probing is not optimal: .005 / .010 / .005.
- When probing is uniquely optimal: .833 / .843 / .861. It passes 3/3.
- **Per split:** price and k pass. heldout_comp is **.545 in all three seeds** (n = 88; 48/88). It fails.

**H4: switching.**
- Pooled unjustified-switch fraction: .044 / .041 / .050.
- Justified case-(b) switches per episode, as a ratio to π* on the same worlds: .833 / .859 / .817. It passes 3/3.
- heldout_comp: unjustified .065–.079, ratio .56 / .58 / .51. It fails.

### 2.2 Fresh worlds, same configurations

- **Seeds:** world seed = split_base + 50M + 1000·idx + rep, where rep is 700–703 for F2 and 500–503 for B1. There are 0 overlapping seeds.
- **Every held-out episode is a different world.** I rebuilt all 3 × 512 episodes at both offsets with `pw.Episode`. **0/1,536 are identical worlds:** the outcome RNG streams differ in every case. The hidden (θ, z) coincides in 71, 19 and 84 of 512 episodes, only because the support is small and discrete.
- **The configurations are B1's 128 per split** (same labels directory; pickle SHA-256s in the JSON). That is what the protocol intends.
- So F2 is fresh in training seeds and in outcome draws, **not in held-out configurations**. The researcher had already seen B1's results on these configurations, although nothing was tuned on them. The report should say "fresh seeds and fresh world draws on the same held-out configurations".
- **Code:** the F2 trainings ran from `source-7655c0b0`, B1's from `source-28152106`. Between the two, the tool diff only adds the B2 `--own-value` path (off by default; bit-identity separately established) and `--world-offset`.

### 2.3 The heldout_comp limit is mainly a probing limit

- **The model encounters fewer failed probes, but reacts to them almost like π*.** Normalize justified case-(b) switches by the number of case-(b) failed probes the policy actually experienced. On heldout_comp the model/π* ratio is then **.90 / .98 / .93**, against .51–.58 per episode.
- **The per-episode shortfall follows from the probing shortfall.** The model reaches only 55–63% of π*'s failed probes because it skips probing in 40/88 uniquely-optimal situations (H3).
- **Suggested wording:** "rational first-probing does not transfer to held-out condition compositions; conditional on a failed probe, switching remains near-optimal". The current wording, that probing and switching both fail to transfer, overstates the switching part.
- **.545 in all three seeds (48/88 each time) points to a deterministic, configuration-level blind spot, not seed noise.** A per-configuration breakdown would localize it.

## 3. B2 (own-greedy-return value head)

### 3.1 Primary (L4, v_own reliability ≤ .05 R on each held-out split in ≥ 2/3 seeds)

| L4 v_own reliability error | s0 | s1 | s2 |
|---|---:|---:|---:|
| heldout_price | .032 | .032 | .059 |
| heldout_k | .129 | .161 | .137 |
| heldout_comp | .102 | .084 | .089 |
| dev | .042 | .029 | **.051** |
| test_iid | .039 | .021 | .029 |
| V head (same eval), heldout_k | .148 | .155 | .128 |
| V* noise floor, max over splits | .033 | .037 | .060 |

- **The primary fails in 0/3 seeds,** because heldout_k and heldout_comp fail in every seed.
- **L1 is worse everywhere,** including in-distribution: v_own on test_iid is .15 / .09 / .09.
- All values match `b2-score.json`.

### 3.2 Policy unchanged (B2-S1)

- **Parameters:** I loaded each `b2-train-{L1,L4}-s{0,1,2}/run/model.pt` and the matching `b-train-*/run/model.pt`. All 22 B1 tensors are present and `torch.equal` in 6/6 runs. The only extra tensors are `v_own.weight` and `v_own.bias` (129,318 − 129,189 = 129 parameters).
- **Evaluation:** a recursive comparison of every key of the B1 `eval.json` against the B2 `eval.json` finds **0 mismatches** in 6/6 runs, so the regret difference is exactly 0 on all 5 splits.
- **Confirmed.**

### 3.3 Reading and failure prediction

- **Reading:**
  - The registered rule selects reading 2 ("not continuation; fails to generalize"), because v_own passes .05 R on dev and test_iid in 2/3 L4 seeds.
  - That 2/3 depends on s2 dev being .051. Seed s0's dev value is .042. **Reading 2 versus reading 3 (fit/capacity) turns on a .001 margin.**
  - The registered confound also stands: v_own is a stop-gradient linear probe on the B1 trunk.
- **Supported claim:** "a linear own-return readout of the B1 trunk is no better calibrated held-out than V; the held-out miscalibration is concentrated in heldout_k (k = 4) and heldout_comp, while heldout_price is calibrated in 2/3 seeds." This is narrower than "B2 shows it is generalization, not a continuation mismatch".
- **v_own and V differ by ≤ .035 on every split (L4: ≤ .02).** As protocol-B2 anticipated, the sampled-vs-greedy continuation gap was small, so B2 had little power to show a continuation effect.
- **Failure prediction (B2-S2):**
  - Pooled held-out, commit level, L4: AUROC .56 / .57 / .45, with 9 / 10 / 20 wrong commits. All bootstrap CIs include .5.
  - It is NOT SUPPORTED, and s0 is "insufficient" (< 10 wrong commits).
  - **The oracle predictors are equally uninformative:** Q*(taken) scores .59 / .62 / .50 and V* .57 / .62 / .49 (from `b2-score.json`). Wrong commits are close to unpredictable from state value, so this null says little about v_own.

## 4. Claims to weaken or correct

1. **report.md headline, deliverable 3:** change "74–86%" to **73–86%**. Add "adaptively registered". Add that R-mask does not reduce the share of stalled episodes, and that it slightly raises cost (+.0003 to +.0015 per episode in 6/6 runs). "At no cost" becomes "at no utility or success cost".
2. **report.md headline, deliverable 4, and §2 "Calibration":**
   - "Probeworld value heads are calibrated in-distribution" holds for **L4 only.** L1's V and v_own reach .055–.16 in-distribution (dev/test_iid).
   - "Value heads fail held-out calibration (.08–.26 R)" omits heldout_price, where L4 passes in 2/3 seeds (.03 / .03 / .06). The failure is on heldout_k and heldout_comp.
   - "B2 shows it is generalization, not a continuation mismatch" is too strong; use the narrower form in §3.3.
3. **F2 composition limit** (decisions 21:47Z, report §2): the switching shortfall on heldout_comp is mainly a consequence of under-probing; per failed probe the ratio is .90–.98. Phrase the limit as probing-first.
4. **F2 freshness:** say "fresh training seeds and fresh outcome draws on B1's held-out configurations", not unqualified "fresh worlds".
5. **decisions.md:**
   - Correct the remaining timestamps from git and receipts (§1.3).
   - Correct two rounding slips (r5 F1 .010, not .011; r4 F1b .012, not .013).
6. **Protocol lesson:** the F1 episode clause was unfalsifiable-to-pass by construction. This is a pre-registration review miss, and the report already discloses it.

## 5. Compute

- **Remote (pro6000), metered, CPU only:** every job ran under `metered.sh` with `CUDA_VISIBLE_DEVICES=` and 1 thread. Receipts are in `results/dev/faudit-*-process`.

  | Job | core-s |
  |---:|---:|
  | faudit-ls | 0.01 |
  | faudit-peek | 0.02 |
  | faudit-f1 | 23.5 |
  | faudit-fb | 5.7 |
  | faudit-extra | 16.4 |
  | **Total** | **≈ 45.6** |

  - This is well under the ~1,500 core-s allowance.
- **Remote, unmetered:** the first directory listing and `nproc`, 4 tar syncs of the scripts, and 3 `cat` calls of output files. All were trivial I/O, estimated < 1 core-s in total.
- **Local (desktop CPU, no torch):** git and jq inspection, the F2 recomputation on the committed eval.json files, and JSON composition. Estimated ≈ 5 core-s.
