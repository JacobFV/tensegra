# Extended-07 Phase 1 diagnosis: independent adversarial audit

**Audited:** `p1-diagnosis.md` and the latest `decisions.md` entry (2026-09-27T17:31Z), at `campaign/extended-07` db5b8723. Analysis code `campaign07_p1analysis.py` v1 (97cc728e) and v1.1 (ae189416). Staged outputs are in `research/results/campaign-07/p1/`.

**Method:**
- I wrote my own re-derivation (`review/p1_audit_rederive.py`) without importing `campaign07_p1analysis`. It reads the raw `diag-v1-cf-{SCE,UCE}-{LRN,SUP,RAWF}` logs of b6c s35–39 and dev s30–32, plus the frozen `model.pt` fusion and policy weights (read-only).
- It recomputes the following from the raw per-decision records:
  - the flip, near-miss and invariance membership, from logged Q\*;
  - the arm accuracies;
  - the Q2 interventions;
  - the Q3 policy recomputation and perturbations, adding new within-support perturbations;
  - nMAE by level;
  - not-H staleness;
  - the Q5 counts, adding new flip-restricted extrapolation rates.
- Output: `research/results/campaign-07/p1-audit/audit-v1-all.json` (sha256 `d6709593…`).
- Summary printer: `review/p1_audit_summarize.py`.
- **Not independently re-derived:** the Q1 interaction-reproduction slopes, the nearest-member rates, the Q4 errscale cross-tab cells and the bootstrap CIs. For the CIs I used the analyst's JSON where needed.

## Verdict summary

| # | item | verdict |
|---|---|---|
| 1a | Arm accuracies (flip .405/.562/.453; near-miss, invariance; dev) | **CONFIRMED** (exact) |
| 1b | Consumer reliance: φ→0 change LRN .045 vs SUP .242; fusion variance share 2.4–3.0% vs 18–21% | **CONFIRMED**. SUP's upper value is .215, so "18–22%" is the precise range. |
| 1c | Exact replacement on flips: 9 changed, 7 rescued (5 in support), 0 harmed; +.011 | **CONFIRMED** (exact) |
| 1d | SUP consumer fed LRN predictions: flip .511 vs LRN .405 vs SUP-exact .562 | **Numbers CONFIRMED; interpretation CORRECTION NEEDED** (§3.1) |
| 1e | Error growth none .316 → single .356 → pair .435 → full .531; flip-unit full − single +.223 | **CONFIRMED** (exact) |
| 1f | Not-H belief staleness (single-C .95, full .91) | **CONFIRMED** qualitatively. The values move by ≤ .03 with the inclusion rule (see §1). |
| 1g | Q5: 9/128 third-order, 119 pair-inherited, pairwise-Q 1838/1920 (96%) and 899/960 (94%) | **Counts CONFIRMED; the "94–96%" framing needs CORRECTION** (§4) |
| 2a | Code frozen before any b6c per-decision read by the analysis | **CONFIRMED as far as receipts allow** (§2) |
| 2b | v1 → v1.1 fix limited as claimed | **CONFIRMED**: diff of 4 lines; v1 and v1.1 JSONs are identical outside `q4` and `q4_summary` |
| 2c | Registered tolerances used | **CONFIRMED**. Caveat: Q3, Q4 and Q5 are not registered readouts, so they are exploratory (§2). |
| 3 | "Consumption primary, acquisition secondary" | **CORRECTION NEEDED**: overstated. The largest part of the SUP − LRN flip gap lies outside the factor channel (§3). |
| 3' | "LRN gains come from auxiliary shaping (D)" | **CORRECTION NEEDED**: this is suggested, not justified. LRN − RAWF bundles shaping with init, RNG, optimizer coupling and dead-vs-live columns (P0b). |
| 5 | Specific wording in `p1-diagnosis.md` / `decisions.md` | see §5 (9 replacements) |

## 1. Reproduction from raw logs (b6c SCE, s35–39, unless noted)

**Integrity checks** (all pass):
- The key `(cfg_id, decision type)` is unique in every file.
- The LRN, SUP and RAWF logs hold the same key set.
- `hist_id` is identical across arms at every key (0 mismatches in 8 × 2 files).
- Exact targets are identical across arms.
- SUP's logged φ equals the exact target at every decision.
- The recomputed flip flag agrees with the campaign06 flag 1920/1920 (SCE per seed), 960/960 (UCE) and 1920/1920 (dev).

**Policy recomputation fidelity** is 1.0 in most cells. It is 0.99994 in one to two cells per block, i.e. one decision out of 15,419; the analyst's own JSON shows the same, e.g. SUP s35 "all" = .999935. "Matched … for 100% of decisions in every cell" should read ">99.99% (at most 1 mismatch per 15,419-decision cell)".

**Accuracies** (pooled over seeds): n and values reproduce exactly.

| stratum | LRN | SUP | RAWF |
|---|---|---|---|
| flip (640) | .4047 | .5625 | .4531 |
| near-miss (295) | .7492 | .5356 | .6441 |
| invariance (8960) | .7403 | .7629 | .7036 |
| all octet decisions (77095) | .8172 | .8386 | .7976 |

Dev reproduces too: LRN .326, SUP .527, RAWF .393 on 402 flips. UCE flip reproduces: .624, .614, .603.

**Q2** (logged interventions):

| stratum | intervention | changed | rescue | rescue in support | harm | Δacc |
|---|---|---|---|---|---|---|
| flip | exact | 9 | 7 | 5 | 0 | +.0109 |
| flip | mirror | 23 | 10 | 2 | 4 | |
| flip | gauss_pred:1 | 1 | 0 | | 0 | |
| invariance | exact | 223 | 100 | 48 | 60 | |
| all | exact | 1177 | 581 | 289 | 225 | |
| dev flip | exact | 14 | 7 | | 3 | |
| UCE flip | exact | 8 | 1 | | 4 | |

All match the report.

**Q3.** These are my recomputations from z and φ with the checkpoint weights, as the mean of per-seed rates.

| perturbation | LRN all | SUP all | LRN flip | SUP flip |
|---|---|---|---|---|
| φ → 0 | .045 | .242 | .044 | .216 |
| φ → mean | .046 | .200 | | |
| φ → permuted | .074 | .337 | | |
| +1σ | .006 | .023 | | |
| +4σ | .025 | .092 | | |
| z → permuted | .658 | .532 | | |

- **Fusion contribution-variance share:** LRN .026/.026/.024/.030/.024; SUP .184/.209/.196/.215/.203.
- **UCE:** φ → 0 changes .041 (LRN) vs .215 (SUP).
- **SUP fed LRN's predictions:** flip .511, per seed .500/.484/.531/.547/.492.
- **SUP exact:** .562.

All confirmed.

**How "SUP fed LRN predictions" is computed**, verified in code and re-implemented:
- At every cf key, SUP's own logged pre-fusion z is used. It comes from SUP's own recurrent state, rebuilt from the identical visible history; `hist_id` equality was verified.
- LRN's logged prediction at the same key replaces SUP's supplied φ in `π(tanh(W[z; φ]))`.

**Leakage: none found.**
- SUP's trunk takes only the base inputs (`inp(x[:, :base_dim])` in fuse mode). The supplied factors enter only at the fusion layer, so z_SUP carries no exact-factor input from this or earlier steps.
- The LRN twin must have identical exact targets; this is asserted.
- It is a same-seed pairing (LRN-s35 with SUP-s35).

**Q1 nMAE (DEC):**

| stratum | nMAE |
|---|---|
| none | .316 |
| single | .356 |
| pair | .435 |
| full | .531 |
| flip | .569 |
| invariance | .528 |
| near-miss | .442 |

- Flip-unit full − single is +.223, per seed .238/.229/.181/.221/.245.
- UCE: none .317, single .358, pair .454, full .567; full − single +.244.

Exact match.

**Not-H staleness.** My inclusion rule: members containing C, the no-C member at the same octet and type, the same context, and the exact belief_H differing by more than 1e-6.

| context | level | stale (mine) | stale (reported) | update captured, median (mine) | captured (reported) |
|---|---|---|---|---|---|
| query2_after_notH | single | .956 | .952 | −.13 | −.135 |
| query2_after_notH | pair | .926 | | | |
| query2_after_notH | full | .908 | .909 | | |
| query2_after_H | single | .388 | .359 | | |
| query2_after_H | pair | .518 | .505 | | |
| query2_after_H | full | .655 | .650 | .33 | .389 |

The small differences come from the inclusion rule. The conclusion, "the not-H update is not learned, even for trained single C; the H update degrades with composition", is confirmed.

**Q5:** counts reproduce exactly:
- 1920 units, 1404 with a unique full optimum, 128 flips vs singles, 9 also vs every pair;
- inherited SE 70, SC 65, CE 10;
- flips by type: 28/8/18/20/54;
- pairwise-Q 1838/1920;
- UCE: 960, 583, 59, 9, UE 46, 899/960;
- dev: 134 flips, 15 also vs every pair.

## 2. Protocol

**Freeze before b6c.**
- Commit 97cc728e is at 17:12:30Z.
- The first analysis job touching b6c logs is receipt `e07-dev-p1an-b6c-20260927T171513`, with `started_unix` = 17:15:13Z.
  - It ran with `--code-sha 97cc728e`.
  - The script sha256 recorded in its output, `2e177a40…`, equals the git blob of `campaign07_p1analysis.py` at 97cc728e, which I re-hashed.
  - Every earlier p1an receipt (rt1, rt2, dev, devfinal) reads only the `e07-diag-dev-s3x` dirs.
- v1.1:
  - commit ae189416 is at 17:21:06Z; the run started at 17:21:19Z;
  - the script sha `7a1f53f9…` equals the ae189416 blob and the staged MANIFEST;
  - `diagscore` sha `aa0a8d2a…` equals the committed file, last changed at efe4c2e9, before the P1-DIAG merge 1aa9c388.
- **Limit:** file access times cannot show whether b6c logs were opened interactively before 17:12:30Z. The orchestrator's b6c scoring job read them at about 17:00, so atime carries no information. No receipt or output indicates a b6c read by the analyst before the freeze.

**Fix scope.** `git diff 97cc728e ae189416` touches only `campaign07_p1analysis.py`, in 4 lines:
- VERSION;
- the Q4 `want` lambda now prefixes strata with the population;
- the markdown filter parses the new prefix.

Comparing the full v1 and v1.1 JSONs on pro6000 (b6c and dev), every differing entry is under `q4` and `q4_summary`: 320 entries on b6c, 224 on dev, the latter being key renames only. The claim "every other v1 number re-derived identically" is **CONFIRMED**.

**Tolerances.** `semantic_tol` is exactly as registered in P1-DIAG `tolerances_frozen`: cost .005, probabilities .05, remaining_queries 1/16, steps_left .25, flags .5.
- The errscale tolerance is correctly labelled a sensitivity.
- However, the Q4 table and the IX statements in the report are errscale-only, since semantic DEC is always "out". They should carry "(sensitivity tolerance)".

**Registration scope.** P1-DIAG's registered primary readouts are three: per-group error, immediate rescue/harm, and common-history accuracy. The Q3 reliance analysis (including SUP+LRN-pred), the Q4 cross-tab and Q5 are **not registered readouts**. The report marks only the per-seed Q3 differences as post hoc. All of Q3, Q4 and Q5 should be labelled exploratory.

**Minor:** the registry says dev uses "160 b6 SCE octets", but 320 were used. The report discloses this.

## 3. Interpretation

### 3.1 The key omission: SUP's flip advantage survives with its factor channel blanked

The analyst's own v1.1 JSON contains, but the report does not show, SUP's flip accuracy with its φ replaced by zero or by the population mean. I re-derived these from raw logs; values are the analyst's JSON, per-seed paired, not bootstrapped.

**b6c SCE flips:**

| SUP's network reads … | flip acc | per seed (s35–39) |
|---|---|---|
| exact φ (its training input) | .562 | .547/.539/.578/.609/.539 |
| population-mean φ (constant) | .539 [.440,.633] | .547/.453/.578/.586/.531 |
| φ = 0 | .514 [.416,.609] | .539/.414/.523/.586/.508 |
| **LRN's predicted φ** | **.511** [.416,.599] | .500/.484/.531/.547/.492 |
| exact φ of a single constituent member, same octet/type (in-support; audit) | .508 | .484/.469/.523/.562/.500 |
| exact φ of the none member, same octet/type (in-support; audit) | .475 | .461/.398/.508/.539/.469 |
| **LRN (own network)** | .405 | .445/.336/.398/.430/.414 |

**Seed-paired contrasts:**

| contrast | b6c | per seed | dev (3 seeds) |
|---|---|---|---|
| SUP(φ=0) − LRN | **+.109** | +.094/+.078/+.125/+.156/+.094; 5/5 > 0 | +.104 |
| SUP(φ=mean) − LRN | **+.134** | 5/5 > 0 | +.201 |
| SUP(LRN-pred) − LRN (the report's +.106) | +.106 | | +.152 |
| SUP(LRN-pred) − SUP(φ=0) | −.003 | −.039/+.070/+.008/−.039/−.016 | +.047 |
| SUP(LRN-pred) − SUP(φ=mean) | −.028 | 4/5 < 0 | −.050, 3/3 < 0 |
| SUP(exact) − SUP(φ=0) | +.048 | | |
| SUP(exact) − SUP(φ=mean) | +.023 | | .000 |

**Consequences:**
1. **Most of the SUP − LRN flip gap (+.158) does not depend on reading decision-specific factor values.** About +.11 to +.13 of it remains when SUP's network receives no decision-specific factor information: .69–.85 of the gap on b6c, and .52–1.0 on dev. It is therefore located in SUP's learned network (trunk representation z + fusion/policy weights), which was trained with informative factor inputs. It is not located in decision-time reading of φ.
2. **LRN's predictions deliver none of the reading-attributable part to SUP's consumer.**
   - SUP+LRN-pred ≈ SUP+zero on b6c, and is worse than a constant mean vector on b6c (−.028) and dev (−.050).
   - LRN's predictions are worth about as much as one constituent's exact factors: .511 vs .508.
   - The report's sentence "The predictions carry usable information when a consumer that does read factors is given them" is **not supported on flips**.
   - Its "SUP's network reading LRN's predictions recovers about 2/3 of SUP's flip advantage" is arithmetically right, but the same network recovers as much with zeros and more with a constant. The recovery is a property of SUP's network, not of LRN's predictions.
3. **LRN's flip deficit vs RAWF (−.048) is also outside the channel.** LRN with φ = 0 scores .403, which is .405 unchanged. LRN − RAWF is therefore a trunk/head difference. The report's "D explains LRN's gains but not the flip deficit" is inconsistent with its own logic: whatever trunk-level difference produces LRN's invariance gains also carries its flip deficit.
4. **SUP's near-miss cost is not an over-reaction to the supplied values.** SUP's near-miss accuracy with φ = 0 is .505 and with φ = mean .519, against .536 with exact and LRN's .749 (audit recomputation, all-decision mean). The cost persists without informative factor inputs, so "consumer over-reaction" should read "a property of SUP's learned network".

### 3.2 Is LRN's insensitivity an off-support artifact? No (confirmed)

Zeroing is off-support for both arms. I added two in-support substitutions: the same arm's own φ taken from the none member, or from a random single member, of the same octet at the same decision type.

| substitution | LRN all | SUP all | LRN flip | SUP flip |
|---|---|---|---|---|
| none-member φ | .013 | .061 | .028 | .220 |
| single-member φ | .013 | .061 | .027 | .155 |

- The 5–8× LRN/SUP sensitivity ratio holds within support.
- Q2 on flips puts 5 of the 7 rescues in support, and gauss_pred at 1σ changes .002.
- **"LRN's frozen policy does not read its factor channel" is CONFIRMED and robust.**

### 3.3 Other interpretation points

**Fusion variance share vs functional reliance.** The two agree in direction: share 2–3% vs 18–21%, and change rates 4–5× lower. A variance share alone would not establish reliance, but the perturbation results do. Confirmed as supporting evidence only.

**Redundancy by construction.** "φ = sg(aux(z)) is a deterministic function of z, so the channel adds no information" is correct information-theoretically. "Gives the fusion layer no pressure to route through it" is too strong: aux is a 2-layer tanh MLP, so φ can supply nonlinear features that the single linear+tanh fusion cannot compute from z directly. Suggested wording: "adds no information beyond z (it can only add nonlinear features of z), and empirically the fusion layer barely uses it".

**Exact vs mirror.** "The mirror moves actions more than exact does, so the consumer does not treat the correct values specially" is a flawed comparison. The mirror (2·target − pred) is displaced from the current φ by twice as much as exact, so larger movement is expected from displacement size alone. The comparison supports only "the response scales with displacement". It does not show indifference to correctness.

**"Consumption primary, acquisition secondary."** Given §3.1, this ranking is not supported. What the data establish:
- (a) LRN's policy does not read its factor channel. This is robust, so no factor-channel fix, including exact values, helps the frozen LRN.
- (b) Most of SUP's flip advantage is carried by SUP's network even with the channel blanked. This is a training-dynamics/representation effect of having trained with informative factor inputs.
- (c) The part of SUP's advantage that does depend on factor values (≈ .02–.05) is not delivered by LRN's predictions. This is an acquisition limit, consistent with the Q1 findings: additive predictions, interaction terms absent, not-H not learned.
- Neither consumption nor acquisition is shown to be primary. The single largest component is (b).

**"LRN gains from auxiliary shaping (D)."** Only suggested. The gains live outside the channel: LRN with φ = 0 keeps invariance .730 vs RAWF .704, against .740 unzeroed, so at most ≈ .01 of the +.037 depends on the channel. But per P0b, LRN − RAWF also bundles copied-vs-independent init, action-sampling RNG drift, shared Adam/clip coupling and dead-vs-live fusion columns. "Points to D" should be "is consistent with D, not separated from the bundled init/RNG/optimizer differences; the S×R 2×2 tests it".

**Phase-2 recommendation.** The S×R 2×2 remains the right vehicle, and a consumer trained with informative φ (R1) is exactly the manipulation that §3.1(b) implicates. The motivation should change, though: SUP's advantage is mostly not decision-time reading. Every Phase-2 arm should therefore be evaluated with φ ∈ {own, exact, zero, population-mean} on flips, near-misses and invariances. That separates the effect of channel reading from the training-induced trunk/head effect. Without it, an R1 gain would be mis-attributed to "reading".

## 4. Q5: interaction order

- **Definition** (verified in code and re-implemented): flip = unique full ε-optimum not containing the lowest-index exact argmax of any single member. "Also vs every pair" applies the same test to the three pairs. "Inherited" means a pair's argmax lies in the full optimum.
- **Counts:** confirmed exactly (§1).
- **Correction to the "94–96%" pairwise-Q statistic:** it is computed over all 1920 (960) units, most of which are non-flip units where every extrapolation is trivially right. A first-order extrapolation (Σ singles − 2·none) already gets **1737/1920 (90%)** and **745/960 (78%)**. Restricted to flip units:

| family | pairwise-Q, flip units | first-order, flip units | pairwise-Q on the 9 "third-order" units |
|---|---|---|---|
| SCE | **117/128 (91%)** | 89/128 (70%) | **8/9** |
| UCE | **49/59 (83%)** | 26/59 (44%) | **4/9** |

So even most of the 9 "irreducible third-order" SCE units are recoverable from pairwise-additive Q. The claim "not irreducible third-order" is **CONFIRMED and, if anything, understated**. The supporting statistic should be the flip-restricted one.

## 5. Exact replacement wording

1. **p1-diagnosis.md, Summary §1:**
   - Replace "The flip deficit of LRN relative to SUP is mainly **consumption**. LRN's frozen policy barely reads its factor channel: correcting the channel with exact values changes 1.4% of flip actions. The predictions carry usable information when a consumer that does read factors is given them."
   - with: "LRN's frozen policy barely reads its factor channel: correcting the channel with exact values changes 1.4% of flip actions, and in-support substitutions confirm the insensitivity. Most of the LRN − SUP flip gap (+.158), however, is not decision-time reading. SUP's own network keeps +.109 (φ = 0) to +.134 (φ = population mean) over LRN when its factor inputs carry no decision-specific information (5/5 seeds). That gap is therefore carried by SUP's learned trunk and heads. LRN's predictions give SUP's consumer no flip benefit over zero (−.003) or mean (−.028) inputs."

2. **Summary, acquisition paragraph:** replace "There is a genuine but secondary **acquisition** problem" with "There is a genuine **acquisition** problem. It accounts for the whole factor-value-dependent part of SUP's advantage (≈ .02–.05), but that part is the smaller component".

3. **Q3 "Accuracy on flips…" bullets:** after "SUP's own network reading LRN's predictions instead of exact inputs: .562 → .511 [.416,.599]", add: "The same network reading φ = 0 scores .514 [.416,.609], and reading the flip-population mean φ scores .539 [.440,.633]; on dev these are .430 and .527 vs .478 with LRN predictions. LRN's predictions are therefore no better than an uninformative input for SUP's consumer on flips."

4. **Q3 recomputation fidelity:** "Recomputed actions matched the logged ones for 100% of decisions in every cell" → "Recomputed actions matched the logged ones for > 99.99% of decisions (at most 1 mismatch per 15,419-decision cell)."

5. **Q3 interpretation, last bullet:** "so the channel adds no information and gives the fusion layer no pressure to route through it" → "so the channel adds no information beyond z (only nonlinear features of it), and empirically the fusion layer barely uses it."

6. **Q2 readings:** "The mirror moves actions more than exact does, so the consumer does not treat the correct values specially" → "The mirror is displaced twice as far from the prediction as exact and moves more actions; the response scales with displacement, and this comparison cannot show whether correctness matters."

7. **Q4 near-miss reading:** "SUP's near-miss cost is a consumer over-reaction, not LRN's problem" → "SUP's near-miss cost persists with its factor inputs zeroed (.505) or mean-filled (.519), so it is a property of SUP's learned network, not a reaction to the supplied values."

8. **Q5 "What the benchmark supports":** "and the full optimum is recoverable from pairwise-additive Q in 94–96% of units" → "and pairwise-additive Q recovers the full optimum in 117/128 (91%) SCE and 49/59 (83%) UCE flip units, including 8/9 SCE 'third-order' units. The first-order single-sum baseline is 89/128 and 26/59; the all-unit rates of 96%/94% are dominated by trivial non-flip units, where the first-order baseline already reaches 90%/78%."

9. **Localization table and "What is established":**
   - **consumption row:**
     - verdict → "Established that the frozen LRN does not read its factor channel. Not established as the main source of the flip gap."
     - replace the last evidence clause with: "SUP's network scores +.106 over LRN with LRN's predictions but +.109 with φ = 0 and +.134 with φ = mean (5/5 seeds): the recovery is a property of SUP's network."
   - **add a row "training-induced trunk/head representation"**:
     - verdict: "Largest component of the LRN − SUP flip gap (≈ .11–.13 of .158); also carries LRN − RAWF (LRN with φ = 0: .403)"
     - evidence: "SUP flip .514/.539 with φ zero/mean; LRN flip unchanged by φ = 0".
   - **acquisition row:** "real but secondary" → "real; it limits the factor-value-dependent part of SUP's advantage entirely (SUP+LRN-pred ≤ SUP+mean)".
   - **"What is established", first bullet:** "LRN's flip deficit is a consumer that ignores a redundant channel, with prediction error as a second-order contributor" → "LRN's policy ignores its (redundant) factor channel. The LRN − SUP flip gap is mostly carried by SUP's learned network independently of its factor values, and LRN's predictions supply none of the factor-value-dependent remainder."
   - **"What is established", second bullet:** "points to the **auxiliary shaping of the trunk (D) as the source of LRN's gains** and to **non-reading (B) as the source of its flip deficit**" → "is consistent with trunk-level effects (auxiliary shaping D, not separated from the init/RNG/optimizer differences bundled in LRN − RAWF) as the source of both LRN's gains and its flip deficit vs RAWF; non-reading (B) explains why exact factors do not help the frozen LRN."
   - **Recommended Phase 2:** keep "branch B through the S×R 2×2" and add a required readout: "evaluate every arm with φ ∈ {own, exact, zero, population-mean} so that channel reading is separated from training-induced trunk/head effects".

10. **decisions.md, 2026-09-27T17:31Z entry:**
    - "**Localization:** **consumption is the primary flip failure.**" → "**Localization (audit-corrected):** the frozen LRN does not read its factor channel, but most of the SUP − LRN flip gap is carried by SUP's learned network even with its factor inputs zeroed or mean-filled."
    - "SUP's consumer fed LRN's *predicted* factors reaches flip .511 vs LRN .405 (SUP exact .562; 5/5 seeds, post hoc)" → "SUP's consumer fed LRN's *predicted* factors reaches flip .511 vs LRN .405 (SUP exact .562), but reaches .514 with φ = 0 and .539 with φ = mean. LRN's predictions add nothing for SUP's consumer (post hoc, 5/5 seeds)."
    - "factor columns carry 2–3% of fusion variance vs 18–21%" → "… vs 18–22%".
    - "**Acquisition is a secondary limit:**" → "**Acquisition limits the factor-value-dependent part (≈ .02–.05) entirely:**".
    - "LRN's gains over RAWF (invariance, pool) point to auxiliary shaping (D), not reading" → "LRN's gains and its flip deficit vs RAWF are both outside the factor channel; this is consistent with shaping (D) but not separated from the bundled init/RNG/optimizer differences".
    - "pairwise Q sums recover the full optimum in 94–96%" → "pairwise Q sums recover the full optimum in 91% (SCE) / 83% (UCE) of flip units (first-order baseline 70% / 44%)".

## Compute

| where | what | core-s |
|---|---|---|
| pro6000, metered | test run on s35 | 7.5 |
| pro6000, metered | full re-derivation over 8 lineages | 47.7 |
| pro6000, metered | v1/v1.1 JSON comparison | 0.2 |
| pro6000, unmetered | read-only peeks (record format, receipts, `ls`) | ≈ 3 |
| local | summarising, git, JSON reads | ≈ 5 |
| **total** | | **≈ 63** |

- Remote runs used 1 thread, peak RSS 1.5 GB, and no detached jobs.
- Outputs went only to `~/structured-latent-dynamics-campaign07/results/dev/e07-audit-p1/` (plus the job receipts under `results/dev/`).
- Nothing was written under campaign06.
