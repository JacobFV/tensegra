# Extended-06 Track B: independent audit (B-ARMS primary and the descriptive B-FACT results)

**Auditor:** independent agent, 2026-09-27. **Base:** `campaign/extended-06` @ ddc0da0c. **Branch:** `campaign/e06-b-audit`.

**Scope.** B-SCREEN (context only), B-ARMS (registered primary plus the descriptive arms) and the descriptive B-FACT results on `b6_hold_SCE` and the 320 S+C+E octets.
- **B-FACT-C is out of scope and was not touched.** No b6c file (bases 6.42e9 / 6.65e9) was read or evaluated. The B-FACT-C training process directories that came along in the tar pull (`*-s35..39-process`) were deleted unread.

**Inputs read:**
- campaign.md, design.md (v1 and v2 revisions 1–6), registry.json (B-SCREEN, B-ARMS, B-FACT, B-FACT-C), decisions.md, trackb-screen.md, seed-ranges.json;
- `src/tensegra/campaign06_probeworld.py`;
- `research/tools/campaign04_probeworld_train.py` (the b6 label/train/eval paths), `campaign06_bscore.py`, `campaign06_bprimary.py`, `campaign06_bscreen.py`;
- the staged `research/results/campaign-06/b-arms/{score,primary}.{json,md}`.

**Raw data pulled (read-only tar pipe) from pro6000** `~/structured-latent-dynamics-campaign06/results/`:
- for all 34 runs (B0/B2 × seeds 30–34; B1, B3, dose1–3, SUP, LRN, RAWF × seeds 30–32): `eval_episodes.jsonl.gz`, `cf_eval.json`, `eval.json`, `train_meta.json`, `train_log.json`;
- `e06-tb-labels/labels/*.json` (labels_meta.train/eval, the `*_s0.json` facts, `cf_SCE.json`, sha256 `c365186f…6558`);
- every `e06-tb-*-process/{launch,occupancy}.json` and `OOM_KILLED.txt`.

**Method.**
- All endpoints were recomputed with independently written code. My loader parses the episode rows and octet records directly. It does not import `campaign06_bscore`. Each context is rebuilt from `rec`/`step_in_query`, and correctness is taken as `delta <= .5`.
- Bootstraps: 2,000 draws, numpy percentile. Two versions are reported:
  - **config-only**, as registered: configurations and octets resampled, pairs fixed;
  - **two-level**: seed pairs resampled with replacement, then configurations and octets.
- The generator module was used only to rebuild the training pools, classify first actions and regenerate the hold configurations for the disjointness checks.

## Summary of verdicts

| # | Check | Verdict |
|---|---|---|
| 1a | B2−B0 pooled primary (P1, P2, P3) recomputed from raw rows | **CONFIRMED**: exact to 1e-15; NOT SUPPORTED |
| 1b | The P2 definition is consistent with the registration and trackb-screen.md | **CONFIRMED** |
| 1c | Alternative P2 definitions | **CONFIRMED**: none changes the verdict (one post-hoc sub-context note) |
| 2a | B0/B2 matched on configuration, updates and examples; frequency, share and first-action mix as claimed | **CONFIRMED**, with one interpretive note (single-type mix) |
| 2b | S+C+E (and U+C+E) never in any training split; hold pool disjoint from training | **CONFIRMED** |
| 3a | The "B0 seed 32 outlier drives the B0 comparisons" caveat | **CORRECTION NEEDED** |
| 3b | Which descriptive passes survive seed variance | **CORRECTION NEEDED**: only LRN survives every reference choice |
| 3c | Registry: "RAWF passes P1 only" | **CORRECTION NEEDED**: RAWF passes P1 and P3 |
| 3d | "LRN is the only arm with a consistent sign" | **CONFIRMED** (fair), with an attribution caveat |
| 4 | "B2's first-decision gain is a shift, not systematic use of the interacting factors" | **CONFIRMED in substance; wording CORRECTION NEEDED** (it is not a *global* shift, and the near-miss evidence is not seed-robust) |
| 5 | Integrity: OOM rerun, cf label cap, score rerun, paired worlds | **CONFIRMED**: no risk to results (one provenance note) |

**Net.** The registered verdict, **B-ARMS primary NOT SUPPORTED**, reproduces exactly. It is robust to every alternative later-decision definition I tried and to a two-level bootstrap. The corrections concern how the seed-variance caveat, the descriptive arm table and the "shift" interpretation are worded. None of them changes a registered outcome.

---

## 1. B2−B0 primary

### 1a. Recomputation

My point estimates equal `primary.json` to within 1.3e-15 on every endpoint. My per-seed values equal `score.json` to within 1.4e-14 on every run (gap, first accuracy, near-miss, false change).

B2 − B0, 5 pairs, 400 configurations, 320 octets:

| endpoint | mean | config-only 95% CI (mine) | staged CI | two-level 95% CI | pairs −/+ | per pair (s30…s34) |
|---|---|---|---|---|---|---|
| P1 gap regret | −1.648 | [−6.26, +3.20] | [−6.10, +3.08] | [−17.5, +19.3] | 4/1 | −0.1 / +32.4 / −25.3 / −9.5 / −5.7 |
| P2 later-decision acc | +.0104 | [−.003, +.023] | [−.002, +.022] | [−.009, +.030] | 1/4 | .000 / +.025 / +.029 / −.011 / +.009 |
| P3 near-miss acc | −.085 | [−.157, −.016] | [−.155, −.005] | [−.221, +.055] | 4/1 | +.120 / −.107 / −.267 / −.133 / −.040 |
| P3 false change | +.0065 | [−.001, +.014] | [−.001, +.014] | [−.018, +.030] | 3/2 | |
| first-decision acc (sec.) | +.075 | [+.044, +.109] | [+.041, +.106] | [+.010, +.135] | 1/4 | |
| flip full-member acc | −.025 | [−.084, +.032] | | [−.113, +.063] | 3/2 | |
| balanced acc | −.075 | [−.124, −.028] | | [−.146, −.017] | **5/0** | |

- **Verdict: P1 fail, P2 fail, P3 fail (near-miss), NOT SUPPORTED. CONFIRMED.** The CI differences from the staged file are bootstrap RNG noise only.
- **Leave-one-pair-out (means only).** The mean gap ranges from −10.2 (dropping pair 31) to +4.3 (dropping pair 32). Later-decision accuracy ranges from +.006 to +.016. Near-miss ranges from −.137 to −.040, so P3 fails in every 4-pair subset.

### 1b. P2 definition

- **Registration** (B-ARMS.primary_registered.P2): "decisions after the first, context classes as defined in trackb-screen.md".
- **trackb-screen §7** defines the contexts as query_first (later queries), after_probe_solved, after_probe_failed, after_b1_timeout and other, "in the model's own trajectory". Its proposed primary is "later-decision accuracy (pooled contexts)".
- **bprimary** takes every decision at index ≥ 1, pools the five classes (a micro-average over decisions), and scores a decision correct when delta ≤ .5.
- In all 34 runs, `delta <= .5` coincides exactly with membership of the label's eps-optimal set `opt` (0 mismatches). eps .5 is the label eps (`labels_meta.eps`).
- **The definition is consistent. CONFIRMED.**

### 1c. Alternative definitions (B2 − B0; config-only CI, then two-level CI)

| definition | mean | config-only CI | two-level CI |
|---|---|---|---|
| registered (pooled micro, eps .5) | +.010 | [−.003, +.023] | [−.009, +.030] |
| macro-average over the 5 contexts | +.008 | [−.021, +.038] | [−.032, +.045] |
| excluding `other` | +.003 | [−.022, +.026] | [−.027, +.031] |
| per-episode mean, then mean over configurations | +.005 | [−.007, +.016] | [−.010, +.021] |
| eps 0 / 1 / 2 | +.010 / +.005 / +.008 | all include 0 | all include 0 |
| within-query decisions only | +.002 | [−.011, +.014] | [−.014, +.019] |
| **query_first only** (post hoc) | +.047 | [+.002, +.087] | [−.015, +.105] |
| **state-matched**: octet full-member accuracy on the 5 non-first decision types | −.004 | [−.020, +.013] | [−.049, +.050] |

- **No reasonable alternative turns P2 into a pass under the registered rule's form.**
- The only config-only exclusion of 0 is query_first. That is a post-hoc sub-context, and it does not survive the two-level bootstrap.
- The state-matched octet measure removes the on-policy confound: B2 takes 0.49 fewer later decisions per episode than B0, CI [−.69, −.28]. On that measure the difference is −.004.
- **Recommendation (reporting only):** report the state-matched octet later-type accuracy alongside the on-policy pooled measure. The on-policy measure averages over different state distributions per arm.

## 2. Matched arms and held-out exclusion

### 2a. Matching

**Training configuration** (`train_meta.json`, all 34 runs):
- every run: rung L1, 4,000 updates × 64 = 256,000 episodes, lr 1e-3, hidden 128; each train_log ends at update 3,999;
- B0 and B2: identical 129,701 parameters, in_dim 72, train_n 384;
- the fuse arms: RAWF and SUP 149,157 parameters, LRN 168,636, as stated;
- every job's launch cwd: `source-f5ff2d64`;
- each run's `train_combo_counts` equals its split's composition in `labels_meta.train.json`.

**Composition** (from the recorded streams):

| check | B0 | B2 |
|---|---|---|
| none / single / pair shares | 64 / 256 / 64 | 64 / 256 / 64 |
| U / S / C / E frequency | 1/3, 1/3, 1/6, 1/6 | identical |
| single-type mix | U 64, S 64, C 64, E 64 | U 64, **S 106**, C 43, E 43 |
| pair types | US 64 | US 22, UC 21, UE 21 |

- **Per-factor frequency and the none/single/pair shares match exactly** in every arm, including dose1–3 and B1/B3.
- B2 keeps 300 of B0's configurations and adds 84. B0 is the first half of B1 type by type.

**First-action class mix.** I re-solved all B0 and B2 configurations with my own loop (96 core-s local):
- B0: probe 177 / exact 9 / gather 131 / structure 67;
- B2: 164 / 7 / 140 / 73;
- both equal the recorded report. The largest class difference is 13 of 384.
- The class multiset difference between the 84 dropped and the 84 added configurations is 15. That is consistent with the reported 17 unmatched fallbacks: 2 fallbacks happened to hit the right class.
- **CONFIRMED.**

**Interpretive note (not a defect: disclosed in trackb-screen §5, absent from registry/decisions).** Matching per-factor frequency by replacement forces a different single-type mix:
- B2 has 42 more S-single configurations, 21 fewer C and 21 fewer E singles, and 42 fewer U+S pairs;
- the dropped configurations are mostly C-/E-single "probe" configurations (34) and U+S "gather/structure" ones.

So B2 − B0 contrasts combination variety **plus** an S-heavier single mix. Section 4 shows B2's behavioural change concentrates on S-containing inputs. The mix change is therefore a live alternative explanation for the direction of B2's shift. B1, which has no mix change, shows the same build-rate rise, so the mix is not the only candidate.

### 2b. Exclusion and disjointness

- **Held-out combinations.** No training stream of any arm (B0–B3, dose1–3) contains a configuration with S, C and E all on, or with U, C and E all on (0 of 1,011 distinct training configurations). `type_seed` asserts `not excluded_from_training`.
- **Dose arms** use only S+E, C+E and S+C (dose3), never the triple.
- **Seed blocks:**
  - training configuration seeds lie in [6.200e9, 6.210e9];
  - hold S+C+E configurations 6.400e9 + [0, 400), octets 6.600e9 + [0, 320);
  - hold eval worlds 6.450e9 + 1000 i (verified equal to `world_seed6`);
  - training worlds 8e9 + 1e8 × seed + counter, i.e. ≥ 1.1e10.
- **Configuration-level check.** None of the 400 regenerated hold configurations equals any training configuration (public-vector comparison). All hold configurations have exactly flags (S, C, E) on, with k ∈ {2: 204, 8: 196}.
- Checkpoints are final (no dev-based selection).
- **CONFIRMED.**

## 3. Seed variance and the descriptive arms

### 3a. The "B0 seed 32 outlier" caveat

Per-seed gap regret on b6_hold_SCE:

| arm | per seed | SD |
|---|---|---|
| B0 (s30–34) | 20.2 / 29.9 / 40.9 / **41.2** / 29.0 | 8.9 |
| B2 (s30–34) | 20.0 / **62.2** / 15.6 / 31.8 / 23.3 | 18.7 |

- **B0 seed 32 is not an outlier.** B0 seed 33 is equally high (41.2).
- **The largest single run is B2 seed 31.** It produces the +32.4 pair. The −25.3 pair comes from B0 s32.
- **For the descriptive arms** (vs B0 s30–32, mean 30.3), the B0 reference is *lower* than the 5-seed B0 mean (32.2). Removing B0 s32 leaves every gap-regret "win" essentially unchanged:

| arm | gap vs B0 s30–32 (paired) | vs pooled B0 s30–34 | vs B0 without s32 |
|---|---|---|---|
| dose1 | −13.0 | −15.0 | −12.8 |
| SUP | −17.3 | −19.2 | −17.1 |
| LRN | −14.9 | −16.9 | −14.7 |
| RAWF | −9.4 | −11.4 | −9.2 |

- 14 of the 15 dose1–3 / SUP / LRN runs have gap below the best B0 seed (20.2); the exception is dose3 s32 at 21.1. What *is* seed-fragile is near-miss accuracy: B0 near-miss per seed is .47 / .59 / .71 / .80 / .64.

### 3b. Which descriptive passes survive

Registered rule applied descriptively:
- P1: gap mean < 0 and upper bound < 0;
- P2: later mean > 0 and lower bound > 0;
- P3: near-miss mean ≥ −.02 and false change ≤ +.02.

| arm | paired 3 (config) | paired 3 (two-level) | vs pooled B0 ×5 (config) | vs pooled B0 ×5 (two-level: arm seeds and B0 seeds resampled independently, then configs) | vs B0 without s32 (two-level) |
|---|---|---|---|---|---|
| B1 | P2 | P2 | P2 | P2 | P2 |
| B3 | P1 P2 | — (P1 ub +7.4, P2 lb −.002) | P1 P2 | P2 (P1 ub +0.01) | P2 |
| dose1 | **P1 P2 P3** | **P1 P2 P3** | P1 P2 (nm −.067) | P1 | P1 |
| dose2 | P1 P2 | P1 P2 | P1 P2 | P1 P2 | P1 P2 |
| dose3 | P1 P2 | P1 | P1 P2 | P1 P2 | P1 |
| SUP | **P1 P2 P3** | **P1 P2 P3** | P1 P2 (nm −.067) | P1 P2 | P1 P2 |
| LRN | **P1 P2 P3** | **P1 P2 P3** | **P1 P2 P3** | **P1 P2 P3** | **P1 P2 P3** |
| RAWF | P1 P3 | P1 P3 | P1 P3 | P1 P3 | P1 P3 |

- **Only LRN passes all three under every reference and bootstrap choice.**
- dose1 and SUP pass only against B0 s30–32. Against all five B0 seeds, their near-miss difference is −.067, so P3 fails. dose1's P2 also fails under the two-level bootstrap.
- B3's P1 does not survive the two-level bootstrap.
- P3 is a point-estimate rule. Under the two-level bootstrap, the only near-miss differences vs B0 whose CI excludes 0 are:
  - dose2 and dose3 (worse, every reference);
  - B3 (worse, pooled-B0 reference only);
  - LRN (better, paired-3 reference only: +.098 [+.009, +.192]).
- Every other arm's CI includes 0.

### 3c. RAWF

The registry B-ARMS result says "RAWF passes P1 only". This is wrong. RAWF's verdict in `primary.json` is `{"P1": true, "P2": false, "P3": true}`: near-miss +.120, false change +.002. It passes P1 and P3 and fails P2 in every variant above.

### 3d. "LRN is the only arm with a consistent sign"

**This is literally true.** Across its 3 pairs LRN has gap −7.6 / −11.1 / −26.1, later +.032 / +.025 / +.034 and near-miss +.147 / +.133 / +.013. No other arm has three same-sign pairs on all of gap, later and near-miss. **It is a fair reading**, and 3b shows it is also the only reference-robust pass.

**Attribution caveat.**
- **Near-miss.** LRN's near-miss advantage over B0 is *shared by RAWF*, which has the same fuse architecture and no auxiliary heads (+.120 vs B0). **LRN − RAWF near-miss is −.022** (per pair −.027 / −.093 / +.053; two-level CI [−.106, +.066]). So the near-miss component of LRN's "consistency" comes from the fuse architecture, not from the derived-quantity heads.
- **What the heads add** (LRN − RAWF, paired, two-level CIs):
  - gap −5.5 [−8.7, −2.6], 3/3;
  - later-decision +.030 [+.014, +.044], 3/3;
  - false change −.019 [−.028, −.010], 3/3.
- **Where the later-decision gain sits.** It concentrates in the first decision of later queries: query_first +.126 [+.077, +.173], 3/3; within-query +.002; octet q2_after_notH +.137, 3/3. It also holds on the state-matched octet later types: +.041 [+.020, +.062], 3/3.
- These are dev-split (b6_hold_SCE) observations only. They say nothing about B-FACT-C's outcome. Note, though, that the dev near-miss difference (−.022) already sits just outside B-FACT-C's Q3 non-inferiority margin (−.02).

## 4. Interpretation: "a shift, not systematic use of the interacting factors"

**Evidence for (reproduced):**
- **The first-decision gain is on non-flip units.**
  - Octet first-decision flip-unit accuracy: B2 .594 vs B0 .677 (31 flip units per seed).
  - Non-flip: B2 .729 vs B0 .624.
  - Pooled flip accuracy −.025; balanced accuracy −.075, 5/5 pairs negative, two-level CI [−.146, −.017].
- **The first-decision gain is an action-rate change toward build.**
  - hold_SCE first actions (pooled seeds): build 9.6% → 27.7%, inspect 66% → 52%, probe 20.7% → 13.9%.
  - Accuracy where build is uniquely optimal: .40 → .87. Where probe is: .68 → .54. Where inspect is: .89 → .85.
- **Not specific to variety.** B1, which adds configurations but no new combination types, has a larger first-decision gain (+.111) and the same build rate (27.8%). So the first-decision gain is not specific to combination variety.
- **The same preference hurts elsewhere.** After a solved probe, commit is uniquely optimal in all 320 octets. B2's commit accuracy drops from .81 to .52 (−.29, 4/5 pairs), because it inspects, props or builds instead.
- **Near-miss** drops by −.085 (4/5 pairs). The drop is concentrated in first-decision near-misses (72 → 52 correct of 125 seed-units) and after_probe_failed (15 → 7 of 20).

**Evidence that the wording overstates:**
- **It is not a *global* shift.**
  - False change is unchanged (+.006).
  - Accuracy on the factor-free member barely moves (first −.012; after_probe_solved −.016).
  - B2's changes concentrate on members where **S and C are both on**, a combination neither arm trained on. At first: SC +.126, SCE +.087. After a solved probe: SC −.20, SCE −.29. At q2_after_notH: SC +.245, SCE +.196; C alone +.03, S alone −.04.
- **B2 does gain on q2_after_notH** (+.196, 5/5 pairs), the context where C's relevance arrives. That gain is offset by the after_probe_solved loss. It is not a composition success, because flip-unit accuracy does not improve, but it is also not "no use of C".
- **The near-miss decrease** is 4/5 pairs, but its two-level CI includes 0 ([−.22, +.05]). The seed-robust evidence is the balanced accuracy (5/5, two-level CI below 0) plus the flip vs non-flip split.

**Verdict.** The conclusion that B2's gains are not composition is supported. The phrase "a shift" should be qualified as an input-region-specific action-preference change (toward build/prop on S∧C inputs), not a global action-rate shift, and should cite the seed-robust evidence.

## 5. Integrity

**OOM (e06-tb-dose2-eval-s30).**
- The killed process left only launch.json, `OOM_KILLED.txt` and process.log (no occupancy receipt). It was renamed `*-oomkilled-process`.
- The rerun used the identical command (same labels dir, `--world-offset 0 --worlds 1`, same source-f5ff2d64 cwd) and exited 0 (170.8 core-s, peak RSS 3.4 GB).
- The dose2-s30 rows are complete: 400 hold configurations and 320 octets, identical keys and worlds.
- **No risk.**

**cf label cap (e06-tb-labels-cf, exit −9 at 8,086 core-s).**
- `cf_SCE.json` holds 320 complete sets (sha256 `c365186f11cd8963f7dc5389dc57893670d28a95fcd2d0a44b178a984bc76558`).
- Every model eval started after the kill, so none could read a file being written.
- All 34 `cf_eval.json` carry identical octet indices, flip flags and per-member `opt` sets.
- **Provenance note.** The killed job never wrote a `labels_meta.cf*.json` for the b6 cf part, so the octet provenance rests on the file itself. Recommend recording the sha above in decisions or receipts.

**Score rerun.**
- e06-tb-score (exit 1 at the final write) and e06-tb-score2 have identical 64-argument commands.
- The staged score.json equals my raw-row recomputation to 1.4e-14.
- The primary ran from `source-4cd69b1c`, the snapshot that adds bprimary. All training, eval, labels and score jobs ran from `source-f5ff2d64`.
- **No risk.**

**Paired evaluation worlds.**
- All 34 runs evaluate the same 400 b6_hold_SCE configurations with identical world seeds (6.450e9 + 1000 i, one world each) and identical V*.
- They also share the identical 320 octets and labels.
- The seed-paired contrasts are genuinely paired on worlds.
- Evaluation has `--world-offset 0 --worlds 1` in every eval launch.
- **CONFIRMED.**

## Required corrections (exact replacement wording)

**C1. decisions.md 10:47Z, caveat bullet.** Replace
> "**Caveat:** the configuration-only CIs omit seed variance. B0 seed 32 is an outlier (gap 40.9), and it drives the B0 comparisons."

with
> "**Caveat:** the configuration-only CIs omit seed variance, which is large: B0 gap regret per seed 20.2/29.9/40.9/41.2/29.0 (SD 8.9), B2 20.0/62.2/15.6/31.8/23.3 (SD 18.7). The B2−B0 per-pair swing (+32.4 / −25.3) comes from B2 seed 31 and B0 seed 32. A two-level (seed pair + configuration) bootstrap gives gap −1.6 [−17.5, +19.3] and later-decision +.010 [−.009, +.030]; the verdict is unchanged. The descriptive gap-regret wins vs B0 do not depend on B0 seed 32 (they are unchanged vs B0 without s32 or vs all five B0 seeds). What is seed-fragile is near-miss accuracy (B0 per seed .47–.80)."

**C2. decisions.md 10:47Z, the "shift" sentence.** Replace
> "B2's first-decision gain (+.075) came with worse near-miss accuracy, i.e. a shift."

with
> "B2's first-decision gain (+.075) is on non-flip units (first-decision flip accuracy .59 vs .68) and comes from a higher build rate (first-action build 28% vs 10%; B1 shows the same rise with no added variety). The same preference costs after a solved probe (commit accuracy .52 vs .81). Balanced octet accuracy is lower in 5/5 pairs (−.075), and near-miss accuracy is lower by .085 (4/5 pairs). This is an input-specific action-preference change concentrated on S∧C inputs, not a global action-rate shift (false change +.006) and not composition."

**C3. registry.json B-ARMS.result.** Replace
> "its first-decision gain comes with worse near-miss accuracy (a shift, not systematic use of the interacting factors)"

with
> "its first-decision gain is on non-flip units and comes from a higher build rate, with worse balanced (5/5 pairs) and near-miss (4/5) octet accuracy and a large loss after a solved probe: an action-preference change on S∧C inputs, not systematic use of the interacting factors"

**C4. registry.json B-ARMS.result, descriptive clause.** Replace
> "Descriptive arms (3 pairs vs B0 seeds 30-32): B3, dose2, dose3 pass P1/P2 but fail near-miss (P3); dose1, SUP, LRN pass all three descriptively; RAWF passes P1 only. CAVEAT: the CI resamples configurations only; per-pair gap-regret differences swing by +-30 (B0 seed 32 gap 40.9 drives most 'wins' vs B0), so seed variance is not in the CI and the descriptive passes are fragile."

with
> "Descriptive arms (3 pairs vs B0 seeds 30-32): B3, dose2, dose3 pass P1/P2 but fail near-miss (P3); dose1, SUP, LRN pass all three; RAWF passes P1 and P3 (fails P2). CAVEAT: the CI resamples configurations only. Under the independent audit (two-level bootstrap; B0 reference = all five seeds or seeds without s32), only LRN still passes all three: dose1 and SUP fail P3 (near-miss −.067 vs pooled B0), and B3 loses P1. The gap-regret wins of dose1-3/SUP/LRN/RAWF are robust to the B0 reference; near-miss is the seed-fragile endpoint."

**C5. registry.json B-FACT.result, final sentence.** Replace
> "LRN-vs-RAWF isolates the auxiliary derived-quantity heads: later-decision accuracy +~.03."

with
> "LRN-vs-RAWF isolates the auxiliary derived-quantity heads: later-decision accuracy +.030 (3/3 pairs; concentrated in the first decision of later queries, +.126), gap −5.5 (3/3), false change −.019 (3/3), near-miss −.022 (2/3 pairs negative). LRN's near-miss advantage over B0 is shared by RAWF (+.120) and is therefore a fuse-architecture effect."

**Optional (reporting, not corrections):**
- Add the state-matched octet later-type accuracy as a companion to the on-policy later-decision endpoint (§1c).
- State in the registry that B2 − B0 also changes the single-type mix (S 106 vs 64; C and E 43 vs 64), as trackb-screen §5 already does.
- Record the `cf_SCE.json` sha256.

## Compute used

**Local** (this machine, pure Python/numpy, process time): **about 190 core-s**, within the ~800 budget. The main items:
- 18 s loading the 34 runs;
- 2 × ~20 s bootstraps;
- ~5 s shift analyses;
- 96 s re-solving and classifying the 468 B0/B2 training configurations;
- <1 s disjointness checks.

**Remote:** **no compute jobs; nothing metered was launched.** Only read-only commands were run (`free`, `ls`, `du`, and one tar | base64 read of ~160 MB). No process was killed, no detached job was started, and GB10s were not used.
