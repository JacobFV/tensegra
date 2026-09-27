# Extended-07 Phase 1: mechanism diagnosis (P1-DIAG readouts)

**Status:** diagnostic, not confirmatory. b6c was already scored end to end by extended-06 (registry P1-DIAG).

**Inputs:** read only.
- per-decision logs `diag-v1-*.jsonl.gz` from `campaign07_diag.py`;
- LRN support references;
- frozen checkpoints and `train_meta.json`.

**Code:** `research/tools/campaign07_p1analysis.py`.
- **Development:** on dev only (lineages s30–32; b6_hold_SCE + 320 b6 SCE octets). The brief says 160; the dev `cf_SCE.json` holds 320.
- **Freeze:** frozen at commit `97cc728e` before any b6c per-decision output was read.
- **v1.1 fix (`ae189416`):** reading the v1 b6c output exposed one mechanical bug. The Q4 cross-tabs pooled cf:SCE and cf:UCE decisions into a single table. v1.1 keys them per population and changes nothing else. Every other v1 number was re-derived identically; for example, the SCE flip accuracy is 0.404687 in both versions.
- **Reported numbers:** all come from v1.1, whose script sha256 is `7a1f53f9…`.

**Statistics:**
- two-level bootstrap: seeds with replacement × whole octets (cf) or configurations (pool);
- n_boot 20000, seed 7, percentile 95% CI;
- the half-split MC check is in the JSON.

**Notation:**
- Cells read `mean [CI] (per-seed values)`.
- Contrasts are seed-paired.
- **Tolerances:**
  - **semantic:** the registered values, e.g. cost coordinates 0.005 (= ε);
  - **errscale:** LRN's own training-state error RMS. This is a registered-as-sensitivity tolerance.
- **nMAE** = |err| / the target std of the training support. `exp_hard_cost_rel` and `steps_left_rel` are constant on b6 and are excluded.

**Machine-readable results:** `research/results/campaign-07/p1/`.
- `e07-p1an-{dev,b6c}-ae189416/p1analysis-v1.1-*.{json,md}`: every endpoint, CI and per-seed value; the full 4-way cross-tab cells; per-coordinate tables.
- `MANIFEST.json`: sha256 of every input consumed (85 files for b6c, 42 for dev) and of every output.

**Membership:** "flip" = full member of a unit (octet × decision type) whose unique optimum differs from every single-factor member's optimum. The campaign06 flag agrees 1920/1920 (SCE) and 960/960 (UCE) with the recomputation.

Unless marked *dev*, the numbers below are **b6c SCE, lineages s35–39** (640 flip, 295 near-miss and 8960 invariance decisions).

## Summary

**Localization.** LRN's frozen policy barely reads its factor channel: correcting the channel with exact values changes 1.4% of flip actions, and in-support substitutions confirm the insensitivity. Most of the LRN − SUP flip gap (+.158), however, is not decision-time reading. SUP's own network keeps +.109 (φ = 0) to +.134 (φ = population mean) over LRN when its factor inputs carry no decision-specific information (5/5 seeds). That gap is therefore carried by SUP's learned trunk and heads. LRN's predictions give SUP's consumer no flip benefit over zero (−.003) or mean (−.028) inputs. *(audit-corrected)*

There is a genuine **acquisition** problem. It accounts for the whole factor-value-dependent part of SUP's advantage (≈ .02–.05), but that part is the smaller component *(audit-corrected)*:
- prediction error grows with the number of combined factors;
- the 2nd/3rd-order interaction content of the cost coordinates is essentially absent from the predictions;
- the not-H belief update is never learned.

On **targets**, the benchmark supports mainly full-vs-single (inherited-from-a-pair, i.e. second-order) flips, not irreducible third-order ones.

On **semantics**, drift of meaning through the trunk is not separately identified by these data: it is not ruled out, and no test isolates it. **Recommended Phase 2: branch B** (consumer ineffective), run through the registered S×R 2×2 so that D (shaping) is tested at the same time.

| arm (b6c SCE, identical octet histories) | flip | near-miss | invariance | all octet decisions |
|---|---|---|---|---|
| LRN | .405 [.306,.502] | .749 [.636,.854] | .740 [.712,.766] | .817 |
| SUP | .562 [.475,.648] | .536 [.412,.659] | .763 [.731,.792] | .839 |
| RAWF | .453 [.350,.551] | .644 [.506,.774] | .704 [.673,.736] | .798 |
| LRN − RAWF | −.048 [−.093,−.006] (−.031/−.070/−.055/−.055/−.031) | +.105 [+.013,+.211] | +.037 [+.009,+.064] | +.020 [+.007,+.032] |
| SUP − RAWF | +.109 [+.053,+.172] (+.070/+.133/+.125/+.125/+.094) | −.108 [−.204,−.021] | +.059 [+.029,+.085] | +.041 |
| LRN − SUP | −.158 [−.226,−.092] (5/5 seeds < 0) | +.214 [+.112,+.315] | −.023 [−.056,+.015] | −.021 |

**The dev pattern replicates on b6c.**
- **Dev flips:** LRN − RAWF −.067 [−.145,+.003], SUP − RAWF +.134 [+.057,+.218].
- **Dev near-miss:** SUP − RAWF −.133.
- **Dev invariance:** LRN − RAWF +.062.
- **On b6c:** the LRN flip deficit is now resolved below zero, and SUP's near-miss cost is resolved.

## Q1: Are LRN's predicted factors accurate where interactions matter?

**No, but they are not specifically worse on flips.** Error is governed by how many held-out factors are combined, and it is equally bad on flips and invariances.

**Error by membership and level.** nMAE (lower is better) and chitE (share of coordinates within the error-scale tolerance).

| stratum | nMAE DEC (G1–G3) | nMAE G1 probs | nMAE G2 side/event | nMAE G3 strategy costs | nMAE G4 | chitE DEC |
|---|---|---|---|---|---|---|
| training states (support ref, per seed) | .22/.22/.22/.23/.22 | .20–.27 | .36–.38 | .17–.18 | .05–.07 | — |
| none member (combo 0, trained) | .316 [.301,.331] | .348 | .433 | .217 | .098 | .687 |
| single (S, C, E; trained) | .356 [.345,.369] | .367 | .568 | .254 | .114 | .635 |
| pair (SC, SE, CE; **never trained**) | .435 [.420,.451] | .404 | .853 | .317 | .133 | .554 |
| full SCE | .531 [.506,.552] | .445 | 1.226 | .389 | .152 | .475 |
| – flip (full) | .569 [.523,.616] | .515 | 1.214 | .398 | .136 | .416 |
| – invariance (full) | .528 [.504,.550] | .440 | 1.226 | .388 | .153 | .479 |
| – near-miss | .442 [.398,.489] | .441 | .610 | .377 | .144 | .472 |
| pool π\* histories (natural SCE states) | .483 | .445 | 1.043 | .320 | .104 | .498 |

**Tolerance hits.** At the registered semantic tolerances, LRN is essentially never within tolerance on the cost groups: the DEC all-coordinates hit rate is 0 in every stratum. Its training-state error is already 10–30× ε on the G3 costs, so the semantic-tolerance hit rate cannot discriminate. The errscale tolerance is the informative one.

**Paired within-octet contrasts: combined-effect errors are real.**
- **Full − single, flip units:**
  - DEC nMAE +.223 [+.193,+.251], positive in 5/5 seeds;
  - G2 +.692;
  - G1 +.150.
- **Full − pair, flip units:** DEC +.121 [+.103,+.138], 5/5 seeds.
- **All units:** full − single +.174, full − pair +.096.
- **Flip − invariance:** DEC nMAE +.041 [−.001,+.085], not resolved. G1 is +.074 [+.022,+.131], a small G1 excess on flips.
- **Flip − near-miss:** DEC +.127 [+.085,+.170], G2 +.605. The near-miss moves factors toward their weakest level, so the combination is milder.

**Interaction content.** The interaction-carrying coordinates (IX) are those whose full-member value differs by more than tolerance from every single member's value. Flips have 3.4 such coordinates on average and invariances 2.2.
- **Hit rate on IX at errscale:** flip .175 [.110,.247], invariance .209.
- **Reproduction of the pairwise (SC, SE, CE) and three-way interaction contrasts:** the slope of the predicted contrast on the exact contrast is .01–.21 and R² is < 0 for every cost coordinate with ≥ 50 non-trivial units (`q1_desc.interaction_reproduction`). **LRN's predictions do not carry the interaction terms**; they are roughly additive.
- **Nearest member:** a full member's prediction is closest to its own exact vector only on flips .223 [.133,.317] and invariances .337. The equivalent rates are .764 for singles and .769 for the none member. On flips it is nearer a pair's (.383) or a single's (.394) exact vector.

**Stale belief after the verifier reveal (members containing C).** "Captured" is the fraction of the correlation update to belief_H that the prediction reproduces.

| context | level | stale (pred nearer the no-C value) | update captured |
|---|---|---|---|
| query2_after_H | single (C, trained) | .359 | .689 |
| query2_after_H | pair | .505 | .550 |
| query2_after_H | full | .650 [.456,.786] | .389 |
| query2_after_notH | single (C, trained) | .952 [.932,.970] | −.135 |
| query2_after_notH | full | .909 | −.093 |

- The downward belief update after a not-H reveal is **never learned, even in the trained single-C combination.** This is an acquisition defect independent of composition.
- The update after an H reveal is learned for singles and degrades with composition.
- **Context:** errors are largest after a b1 timeout (G1 nMAE .98, G2 1.9 on flips; the same on invariances) and at query2_after_notH (G2 1.29).

**Saturation and clipping:**
- Probability coordinates outside [0, 1] occur on 91–95% of decisions in every stratum. This is small overshoot of an unconstrained linear head; it is equally frequent at the trained none member (.914), so it is not specific to the held-out combination.
- Predictions outside the model's own training-prediction range: flip .152, invariance .192, single .105, none .075.
- Targets at the ±5 clip: ≤ 6%.
- There is no sign that clipping drives the flip errors.

## Q2: Does correcting the factors improve the same frozen LRN?

**Almost not, because the consumer barely responds.** Immediate replacement at the aux output, same state.

**b6c SCE:**

| population | intervention | n | actions changed | rescue (in/out of support) | harm (in/out) | Δacc | change rate |
|---|---|---|---|---|---|---|---|
| flip | exact | 640 | 9 | 7 (5/2) | 0 | +.011 [.000,+.029] | .014 [.002,.034] |
| flip | dep:G1 | 640 | 9 | 6 (5/1) | 0 | +.009 | .014 |
| flip | dep:G3 | 640 | 8 | 5 (4/1) | 1 | +.006 | .013 |
| flip | iso:G1 | 640 | 3 | 1 | 1 | .000 | .005 |
| flip | gauss_pred:1 | 640 | 1 | 0 | 0 | .000 | .002 |
| flip | mirror | 640 | 23 | 10 (2/8) | 4 (0/4) | +.009 | .036 |
| near-miss | exact | 295 | 5 | 2 (2/0) | 3 (3/0) | −.003 | .017 |
| invariance | exact | 8960 | 223 | 100 (48/52) | 60 (43/17) | +.004 [−.001,+.010] | .025 |
| all octet decisions | exact | 77095 | 1177 | 581 (289/292) | 225 (154/71) | +.005 [+.002,+.008] | .015 |
| pool π\* | exact | 31105 | 384 | 169 | 98 | +.002 [.000,+.004] | .012 |

**Readings:**
- The replaced vectors are in LRN's training support most of the time. On flips, rescue is 5 in-support against 2 out. Out-of-support replacement is therefore not what makes correction ineffective.
- **Action-change rate, exact vs noise at LRN's own error scale:**
  - exact − gauss_pred: flip +.013 [.000,+.032], all +.009 [+.007,+.012];
  - exact − mirror (same error magnitude, opposite sign): flip −.022 [−.043,−.006].
  - The mirror is displaced twice as far from the prediction as exact and moves more actions; the response scales with displacement, and this comparison cannot show whether correctness matters. *(audit-corrected)*
- **Dev agrees:** exact on flips changed 14/402 actions, rescued 7 and harmed 3, Δacc +.010 [−.008,+.034].
- **Rollout-level replacement** (R-exact etc., named separately) was not re-analysed here. Dev score: R-exact .820 vs B .818.

## Q3: Consumer reliance, LRN vs SUP

The policy was recomputed from the logged pre-fusion latent z (float16) and φ, reading the checkpoint weights. Recomputed actions matched the logged ones for > 99.99% of decisions (at most 1 mismatch per 15,419-decision cell). *(audit-corrected)* The noise scale σ is the same per-seed LRN training-error RMS for both arms.

**Action-change rate on b6c SCE** (LRN − SUP contrasts in brackets):

| perturbation of φ | LRN all | SUP all | LRN − SUP | LRN flip | SUP flip |
|---|---|---|---|---|---|
| +1σ noise | .006 | .024 | −.018 [−.020,−.016] | .005 | .039 |
| +4σ noise | .025 | .095 | −.070 [−.076,−.064] | .025 | .125 |
| φ → 0 | .045 | .242 | −.197 [−.220,−.174] | .044 | .216 |
| φ → population mean | .046 | .200 | | .036 | .170 |
| φ → another decision's φ | .073 | .338 | −.265 [−.282,−.248] | .045 | .270 |
| **reference:** z → another decision's z | .663 | .530 | +.132 [+.102,+.164] | .603 | .475 |

**Accuracy on flips when the φ channel is swapped:**
- LRN: .405 → .416 with exact φ, .403 with φ = 0.
- **SUP's own network reading LRN's predictions instead of exact inputs:** .562 → **.511 [.416,.599]**. The same network reading φ = 0 scores .514 [.416,.609], and reading the flip-population mean φ scores .539 [.440,.633]; on dev these are .430 and .527 vs .478 with LRN predictions. LRN's predictions are therefore no better than an uninformative input for SUP's consumer on flips. *(audit-corrected)*
- **Per-seed paired differences** (not bootstrapped; post hoc; all 5 seeds share the sign):

  | comparison | per-seed differences | mean |
  |---|---|---|
  | SUP+LRN-pred − LRN | +.055/+.148/+.133/+.117/+.078 | +.106 |
  | SUP exact − SUP+LRN-pred | +.047/+.055/+.047/+.062/+.047 | +.051 |

- On dev the same values are .527 → .478 against LRN .326.
- **Natural pool states:** SUP+LRN-pred .791 is below LRN .809 and SUP .829. On ordinary states LRN's prediction errors do hurt a factor-reading consumer.

**Fusion weights, b6c SCE, 5 seeds.** The share of the fusion pre-activation variance contributed by the 23 factor columns:

| arm | contribution-variance share | input-scaled Frobenius share |
|---|---|---|
| LRN | .024–.030 | .13–.15 |
| SUP | .18–.21 | .43–.47 |

RAWF's factor columns are dead (P0b).

**Interpretation:**
- LRN's policy is carried by the trunk: permuting z changes 66% of actions.
- It is 4–5× less sensitive to its factor channel than SUP is to the same-scale perturbations.
- This is expected mechanically. φ = sg(aux(z)) is a deterministic function of the same z the fusion layer already sees, so the channel adds no information beyond z (only nonlinear features of it), and empirically the fusion layer barely uses it. *(audit-corrected)*

## Q4: 4-way cross-tab on flips (b6c SCE, identical histories; counts summed over seeds)

"Pred in" = within tolerance on the relevant set at errscale. Semantic-tolerance tables are in the JSON; there, DEC is always "out".

| relevant set | pred | LRN | n | exact fixes | exact breaks | SUP ok | SUP ok & exact changes nothing | RAWF ok |
|---|---|---|---|---|---|---|---|---|
| DEC (G1–G3) | out | ok | 259 | 0 | 0 | 245 | 245 | 239 |
| DEC (G1–G3) | out | wrong | 381 | **7** | 0 | **115** | **108** | 51 |
| IX coords | in | ok | 50 | 0 | 0 | 47 | 47 | 44 |
| IX coords | in | wrong | 62 | 0 | 0 | 11 | 11 | 7 |
| IX coords | out | ok | 209 | 0 | 0 | 198 | 198 | 195 |
| IX coords | out | wrong | 319 | 7 | 0 | 104 | 97 | 44 |
| G1 | in | wrong | 5 | 0 | 0 | 3 | 3 | 2 |

**Readings:**
- Of the 381 flips LRN gets wrong, SUP gets 115 right at the identical history. Exact replacement fixes 7 of them.
- **In 108 cases SUP is right and exact factors do not change LRN's action**: the dominant cell.
- When LRN's interaction-carrying coordinates are already within error scale, LRN is still wrong on 62 of 112 (55%). When they are out, it is wrong on 319 of 528 (60%). Prediction accuracy on the relevant coordinates barely predicts LRN's correctness.
- On near-misses, LRN is right on 221 of 295 while SUP is right on only 155 of those. SUP's near-miss cost persists with its factor inputs zeroed (.505) or mean-filled (.519), so it is a property of SUP's learned network, not a reaction to the supplied values. *(audit-corrected)*
- Per-seed counts and the flip_vs_pairs / flip_pair_inherited splits are in `q4` / `q4_summary`.

## Q5: Is the interaction content third-order? (octet labels, b6c)

| family | units with all members | unique full optimum | flip vs singles | … also vs every pair | … inherited from a pair | pair-level flips (a pair's optimum differs from both of its singles) | pairwise-Q extrapolation picks a full optimum |
|---|---|---|---|---|---|---|---|
| SCE (320 octets × 6 types) | 1920 | 1404 | 128 | **9** | 119 (SE 70, SC 65, CE 10; overlapping) | 209 | 1838/1920 (96%) |
| UCE (160 × 6) | 960 | 583 | 59 | **9** | 50 (UE 46) | 87 | 899/960 (94%) |

Flips by type (SCE): first 28, after_probe_failed 8, after_b1_timeout 18, query2_after_H 20, query2_after_notH 54, after_probe_solved 0. Under the stricter disjoint-ε-set definition, 73 flips remain vs singles and 3 vs singles and all pairs. Dev (b6 SCE): 134 flips, of which 15 are also vs all pairs.

**What the benchmark supports:** the SCE/UCE flips test **full-vs-single composition, whose optimum is overwhelmingly already present in one constituent pair (SCE 119/128 = 93%, UCE 50/59 = 85%)**, and pairwise-additive Q recovers the full optimum in 117/128 (91%) SCE and 49/59 (83%) UCE flip units, including 8/9 SCE 'third-order' units. The first-order single-sum baseline is 89/128 and 26/59; the all-unit rates of 96%/94% are dominated by trivial non-flip units, where the first-order baseline already reaches 90%/78%. *(audit-corrected)*
- Third-order-only flips are 9 units per family, i.e. 45 decisions over 5 seeds. That is too few to resolve anything: the LRN − SUP flip_vs_pairs contrast on SCE is −.133 [−.422,+.060].
- The benchmark does **not** support claims about irreducible third-order interaction.
- Because **no arm ever saw any constituent pair** (Q6), what it does test is extrapolation from singles to pairwise interactions.

## Q6: UCE octets (descriptive) and exposure

**Exposure:** all four arms, every lineage, `train_split` b6_B0. The combinations are 0, U, S, C, E and US, 64 configurations each.
- SCE constituent pairs SC, SE, CE: **0** in training. Singles S, C and E: 64 each.
- UCE pairs UC, UE, CE: **0**. Singles U, C and E: 64 each.
- The only trained pair, US, is not a constituent of either family.

| UCE (b6c) | LRN | SUP | RAWF | LRN − RAWF | SUP − RAWF |
|---|---|---|---|---|---|
| flip (n = 295) | .624 | .614 | .603 | +.020 [−.055,+.098] | +.010 [−.087,+.104] |
| flip vs all pairs (45) | .267 | .289 | .311 | −.044 | −.022 |
| near-miss (95) | .200 | .147 | .179 | +.021 | −.032 |
| invariance (4505) | .729 | .729 | .694 | +.035 [+.006,+.065] | +.035 [−.005,+.077] |

**UCE follows the same error pattern:**
- nMAE DEC rises from none .317 to single .358, pair .454 and full .567. The flip-unit full − single contrast is +.244 [+.189,+.300].
- Belief after a not-H reveal is stale in .95–.99 of cases.
- Exact replacement on flips changes 2.7% of actions (rescue 1, harm 4).
- LRN's sensitivity to φ → 0 is .041 (SUP .215).
- There is no SUP advantage on UCE flips. SUP+LRN-pred scores .603 against SUP .614 and LRN .624.
- 30 of the 37 flips at after_probe_solved are inherited from UE.
- UCE therefore does not show the SUP > RAWF flip effect that the diagnosis explains for SCE.

## Localization

| candidate locus | verdict | evidence |
|---|---|---|
| **consumption** | **Established that the frozen LRN does not read its factor channel. Not established as the main source of the flip gap.** *(audit-corrected)* | Q2: exact factors move 1.4% of flip actions and fix 7 of 381 errors. Q3: LRN's φ-sensitivity is ¼–⅕ of SUP's and its factor share of fusion variance is 2–3%. Q4: 108 of 381 flips where SUP is right and exact factors change nothing. SUP's network scores +.106 over LRN with LRN's predictions but +.109 with φ = 0 and +.134 with φ = mean (5/5 seeds): the recovery is a property of SUP's network. *(audit-corrected)* |
| **training-induced trunk/head representation** | **Largest component of the LRN − SUP flip gap (≈ .11–.13 of .158); also carries LRN − RAWF (LRN with φ = 0: .403)** *(audit-corrected)* | SUP flip .514/.539 with φ zero/mean; LRN flip unchanged by φ = 0. |
| **acquisition** | **Established as real; it limits the factor-value-dependent part of SUP's advantage entirely (SUP+LRN-pred ≤ SUP+mean).** *(audit-corrected)* | Monotone error growth with the number of combined held-out factors (full − single +.22 nMAE). Interaction contrasts not reproduced (slope ≤ .2). Not-H belief update never learned. SUP's consumer loses .051 on flips (5/5 seeds) and falls below LRN on pool states when fed LRN predictions. Flip errors ≈ invariance errors, so acquisition does not explain what is specific to flips. |
| **semantics** (meaning drift through the shared trunk) | Not identified. | These data cannot separate it from acquisition. The predictions are measured against their targets, which covers acquisition, but whether the trunk's own code for the factors drifts is untested here. The 2×2 (S0 vs S1 trunks) addresses it. |
| **learning of the downstream interaction** | Partly implied, not separately established. | RAWF and LRN both reach ~.4–.45 on flips. SUP's consumer, which was trained with exact factors on the same singles-only data, reaches .56. The downstream mapping is learnable from supplied factors without pair exposure, but only partially. |
| **targets / ontology** | Established as a scope limit, not a failure cause. | Flips are 85–93% pair-inherited (second-order) and pairwise-additive Q recovers the full optimum in 91% (SCE) / 83% (UCE) of flip units (all-unit 94–96% is dominated by trivial non-flip units) *(audit-corrected)*. G3 strategy costs are myopic (P0a). Third-order support is 9 units per family. |

**What is established:**
- LRN's policy ignores its (redundant) factor channel. The LRN − SUP flip gap is mostly carried by SUP's learned network independently of its factor values, and LRN's predictions supply none of the factor-value-dependent remainder. *(audit-corrected)*
- Together with the LRN − RAWF gains on invariance (+.037) and pool (+.034), which cannot come through a channel the policy ignores, this is consistent with trunk-level effects (auxiliary shaping D, not separated from the init/RNG/optimizer differences bundled in LRN − RAWF) as the source of both LRN's gains and its flip deficit vs RAWF; non-reading (B) explains why exact factors do not help the frozen LRN. *(audit-corrected)*

**What is not established:**
- that fixing consumption would reach SUP's level: SUP+LRN-pred .511 < SUP .562;
- anything about third-order interaction;
- semantic drift;
- anything confirmatory (b6c is diagnostic).

## Recommended Phase 2 branch: B

**Branch B: predictions are adequate enough, but the consumer is ineffective.** Run it through the registered S×R 2×2 with copied initialization.

The minimal intervention the evidence justifies is **to make the factor channel non-redundant for the consumer**:
- train the consumer on out-of-fold (OOF) predictions, or on an exact/noisy mixture, so that reading φ pays off;
- or decouple R from S: a policy that reads φ from a separately trained predictor.

In both cases, evaluate with predicted and exact inputs on SCE flips, near-misses and invariances, plus the pool.

**Required readout:** evaluate every arm with φ ∈ {own, exact, zero, population-mean} so that channel reading is separated from training-induced trunk/head effects. *(audit-corrected)*

**Why B and not the alternatives:**
- **A (acquisition):** A alone would not help, because the current consumer ignores even exact values.
- **C (semantics):** C is not identified.
- **D (shaping):** D explains LRN's gains but not the flip deficit. The 2×2 tests it at no extra cost: S1R0 against S0R0.
- **E (seed-unstable or negligible):** E is ruled out for the flip deficit, which is 5/5 seeds for both LRN − SUP and SUP+LRN-pred − LRN.

**Guardrails for Phase 2:**
- Report near-miss non-inferiority. SUP's consumer pays −.108 there.
- Include acquisition improvements for G2 and for the not-H belief update as a secondary arm only after B.
- Scope every claim to pair-inherited (second-order) flips.

## Compute

- **Metered on pro6000:** 971 core-s.
  - smoke tests rt1 and rt2: 13 + 14;
  - dev v1 development run: 144;
  - dev v1 frozen: 148;
  - b6c v1: 246;
  - b6c v1.1: 254;
  - dev v1.1: 152.
- **Local:** about 150 core-s, for development test runs on a copied s30 subset.
- **Resources:** 1 core per job, run sequentially; peak RSS 2.5 GB. No detached jobs.
- **Outputs:** only under `results/dev/` and `agent-p1an/`.
