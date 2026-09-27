# Extended-07 P2 frontier: discrimination vs bias in the P2 screen

**Question.** The screen table suggested every model sits on a flip/near-miss trade-off ((flip + near-miss)/2 ≈ .55–.63). Do the P2 interventions move *action-change propensity* (bias) rather than the ability to tell apart the situations in which the optimum changes (discrimination)?

**Short answer.** It depends on which contrast you ask about, and the two answers are different:

1. **Magnitude / near-miss discrimination is flat.** This asks: can the model tell the full combination apart from a milder full-family configuration whose optimum is the constituent's? Every model, every φ condition and every intervention lies on one trade-off line.
   - Across the 36 distinct model@φ points, flip and near-miss accuracy correlate at r = −.82, with slope −1.8 near-miss per unit flip.
   - Balanced accuracy BA_nm stays in .51–.60 (SD .018).
   - Where the optimum always changes (full → near-miss), models change their action only 35–44% of the time, against a 19–35% change rate where it does not change. J_nm is .07–.21 for every model.
   - No contrast against S1R1 or S0R0 reaches the .03 margin with a CI above 0, except one: S0R1 − S1R1 BA_nm. That one reflects S1R1's shaping-induced near-miss loss, not a gain in S0R1.
   - **On this axis the screen's differences are bias: flip gains are paid for in near-miss accuracy.**
2. **Composition discrimination is real and varies.** This asks: does the model change its action between the full member and its pairs/singles exactly when the optimum changes? Youden J_sub ranges from .25 to .43, far below π* (1) and below even a first-order additive oracle (.77).
   - Shaping (S1) raises it: S1R1 − S0R0 = +.074 [+.010, +.143]. The rise comes from hits, with no rise in false alarms.
   - Factor inputs raise it within a model by *suppressing false alarms*. Zero/mean φ raise FA_sub by +.04 to +.13 with hits unchanged.
   - Every deployable P2 candidate is *below* S1R1: ΔJ_sub −.04 to −.08.
3. **SUP-h.** It has the highest J_sub (.433; +.043 [−.011, +.104] vs S1R1, 5/5 seeds positive, the only model above the .03 margin). Its bias is not higher (+.011). But its composition gain is:
   - concentrated in q2_after_notH (+.209 [+.089, +.358]);
   - partly cancelled after_b1_timeout (−.280);
   - absent on the near-miss axis (ΔBA_nm +.007 [−.068, +.076], ΔJ_nm +.020 [−.139, +.167]).
   - Its +.094 flip gain is fully repaid by −.159 on near-misses.
   - With φ = zero/mean its J_sub collapses to .30 and its bias rises by +.08 to +.12. The "blank-input" flip advantage recorded in P1 is therefore bias.

**Verdict.**
- The P2 interventions (shaping, reading, consumers, separate predictors, φ contracts) do **not** change interaction discrimination in the near-miss sense. They move models along a fixed flip/near-miss trade-off, so flip-accuracy gains there are bias.
- They do change *composition* discrimination (full vs constituents), mostly through training-induced trunk effects (S1) and through factor inputs acting as a "do not change here" signal.
- Nothing deployable beats S1R1 on any discrimination index.

All numbers are from b6c SCE octets, seeds 40–44 (evalX: historical lineages 35–39, paired by index), 320 whole octets, and two-level bootstrap (seed × whole octet, 20,000 draws, seed 7). UCE is descriptive. Full tables: `research/results/campaign-07/p2-frontier/tables.md`; machine-readable: `frontier.json`.

## Definitions (fixed in the code before any output was seen)

These are in `research/tools/campaign07_frontier.py` (module docstring).

- **Unit** = (octet, decision type). All members of an octet are decided after the *same* visible history. An action difference between two members can therefore only come from the configuration (public vector / factor channel).
- **Flip / near-miss / invariance accuracy:** identical to `campaign07_diagscore.membership`, the registered screen endpoints. The values reproduce score.md (for example, S1R1 flip .469, near-miss .695).
- **Pairs** are those with a *unique* optimum at both ends. optchg = the two optima differ; chg = the model's greedy actions differ.
  - **H** = P(chg | optchg), **FA** = P(chg | ¬optchg), **J = H − FA** (Youden). π* has J = 1. A policy whose change rate ignores whether the optimum changes has J = 0 at any rate.
  - **bias** = P(chg) over the same pairs.
  - **sub** (primary): full member vs each of its 3 pairs and 3 singles.
  - **edge** (secondary): all 12 one-factor edges of the cube.
- **BA_nm** = (flip accuracy on flip units that have a near-miss + near-miss accuracy)/2. This is the balanced accuracy of "act as the combination requires" vs "act as the constituent does". A policy that cannot distinguish the full member from its near-miss, and always picks one of the two optima, scores .5.
- **Post hoc** (added after the first tables were seen, and labelled †):
  - H_nm = P(chg between the full member and its near-miss). The optimum always changes there.
  - J_nm = H_nm − FA_sub.
- **Margin:** .03 absolute, the registered P2-SCREEN minimum flip gain. It is also about the size of the smallest J_sub contrast the screen resolves: contrast CI half-widths are ≈ .03–.06 for J_sub and ≈ .05–.07 for BA_nm.
- **Reference policies** (no seed dimension; CI over octets):
  - π*.
  - **add1** (first-order): every multi-factor member is decided by argmax Σ_f Q_f − (|m|−1) Q_0.
  - **add2** (pairwise, P1 Q5): the full member is decided by argmax Σ_pairs Q_p − Σ_singles Q_f + Q_0; lower members exact.
  - Both use *exact* constituent labels, so they are oracles on the constituents, not learnable policies. Neither has a near-miss decision.

## 1. Pooled SCE table (distinct rows; own inputs unless stated)

| model@φ | flip | near-miss | inv | BA_nm | H_sub | FA_sub | **J_sub** | **bias_sub** | H_nm† | J_nm† |
|---|---|---|---|---|---|---|---|---|---|---|
| π* | 1 | 1 | 1 | 1 | 1 | 0 | 1 | .175 | 1 | 1 |
| add2 (pairwise oracle) | .914 | – | .960 | – | .955 | .036 | **.919** [.893, .943] | .197 | – | – |
| add1 (first-order oracle) | .695 | – | .920 | – | .842 | .068 | **.774** [.738, .807] | .204 | – | – |
| S0R0 | .372 | .847 | .767 | .580 [.527, .636] | .509 | .193 | .316 [.255, .376] | .248 | .359 | .167 |
| S1R0 | .453 | .708 | .751 | .549 | .597 | .220 | .377 [.330, .423] | .286 | .410 | .190 |
| S0R1 | .447 | .817 | .754 | .593 | .566 | .216 | .349 [.290, .410] | .278 | .417 | .201 |
| **S1R1 (reference)** | .469 | .695 | .760 | .542 [.475, .612] | .607 | .217 | .390 [.345, .433] | .286 | .380 | .162 |
| SEP | .416 | .814 | .730 | .575 | .535 | .229 | .307 | .282 | .359 | .131 |
| CONS-exact (exact φ, privileged) | .456 | .780 | .750 | .585 | .593 | .225 | .368 | .290 | .424 | .198 |
| CONS-exact-pred | .453 | .776 | .743 | .580 | .583 | .229 | .353 | .291 | .376 | .147 |
| CONS-mix-pred | .445 | .786 | .741 | .580 | .562 | .228 | .334 | .286 | .373 | .145 |
| CONS-oof-pred | .414 | .814 | .759 | .578 | .548 | .206 | .342 | .266 | .353 | .146 |
| CONS-exact-predLRN | .439 | .797 | .737 | .581 | .578 | .230 | .348 | .291 | .380 | .150 |
| LRN-h | .405 | .749 | .740 | .563 | .528 | .221 | .307 | .275 | .403 | .183 |
| **SUP-h** | **.562** | **.536** | .763 | .549 [.479, .618] | .654 | .221 | **.433** [.389, .475] | .297 | .403 | .182 |
| SUP-h @zero | .514 | .505 | .626 | .508 | .651 | .349 | .302 | .402 | .427 | .078 |
| SUP-h @mean | .497 | .637 | .655 | .563 | .613 | .314 | .299 | .366 | .444 | .130 |

Base rate of optimum change on sub pairs: .175. Across the 36 distinct model@φ points:

| | range | SD |
|---|---|---|
| flip | .37–.56 | .036 |
| near-miss | .51–.85 | .076 |
| BA_nm | .51–.60 | .018 |
| J_sub | .25–.43 | .040 |
| bias_sub | .25–.40 | .031 |

Correlations across the same points:
- flip–near-miss r = −.82 (slope −1.77);
- flip–H_sub r = +.92;
- near-miss–J_sub r = −.41.

**What the table shows:**
- Flip accuracy tracks hit rate on composition pairs.
- Near-miss accuracy moves *against* composition discrimination. A model that changes its action more reliably between full and constituents also changes it on the milder full-family configuration, where it should not.

## 2. Does anything exceed the reference by the .03 margin?

Paired contrasts, pooled SCE: mean [95% CI], seeds positive/5.

| model@own | ΔJ_sub vs S1R1 | Δbias_sub vs S1R1 | ΔBA_nm vs S1R1 | ΔJ_nm† vs S1R1 | ΔJ_sub vs S0R0 | ΔBA_nm vs S0R0 |
|---|---|---|---|---|---|---|
| S0R0 | −.074 [−.143, −.010] 1/5 | −.038 [−.067, −.005] | +.037 [−.016, +.095] | +.004 [−.121, +.123] | – | – |
| S1R0 | −.013 [−.040, +.014] | +.001 | +.007 | +.028 | +.060 [−.000, +.125] 3/5 | −.031 [−.088, +.020] |
| S0R1 | −.041 [−.092, +.008] | −.008 | **+.051 [+.007, +.098] 5/5** | +.038 [−.063, +.140] | +.033 [−.016, +.076] | +.014 [−.030, +.048] |
| S1R1 | – | – | – | – | **+.074 [+.010, +.143] 4/5** | −.037 [−.095, +.016] |
| SEP | −.083 [−.131, −.042] 0/5 | −.003 | +.032 [−.031, +.097] | −.032 | −.010 | −.005 |
| CONS-exact-pred | −.037 [−.068, −.006] 0/5 | +.006 | +.037 [−.021, +.096] | −.015 | +.037 [−.030, +.104] | +.000 |
| CONS-mix-pred | −.056 [−.093, −.021] 0/5 | +.001 | +.037 | −.017 | +.018 | +.000 |
| CONS-oof-pred | −.048 [−.092, −.007] 0/5 | −.020 | +.036 | −.016 | +.026 | −.002 |
| CONS-exact-predLRN | −.042 [−.075, −.011] 0/5 | +.005 | +.039 | −.013 | +.032 | +.002 |
| CONS-exact (exact φ) | −.022 [−.055, +.013] | +.004 | +.042 [−.018, +.104] | +.036 | +.052 [−.014, +.114] 4/5 | +.005 |
| LRN-h | −.083 [−.137, −.023] 1/5 | −.011 | +.020 | +.020 | −.009 | −.017 |
| **SUP-h** | **+.043 [−.011, +.104] 5/5** | +.011 [−.031, +.055] | +.007 [−.068, +.076] | +.020 [−.139, +.167] | **+.116 [+.055, +.180] 5/5** | −.031 [−.104, +.043] |

**Composition (J_sub):**
- No deployable candidate reaches S1R1. Each is lower, with CIs excluding 0 for SEP and all predicted-input consumers.
- Only SUP-h exceeds S1R1 by more than the margin in point estimate. Its CI includes 0. On the secondary edge set it is resolved: J_edge +.047 [+.016, +.082], 5/5 seeds.

**Near-miss axis (BA_nm, J_nm†):**
- Nothing exceeds S0R0 (all ΔBA_nm in [−.037, +.014]).
- The only resolved contrast is S0R1 − S1R1 (+.051), which is S1R1's shaping-induced near-miss loss (S1R1 − S0R0 = −.037) seen from the other side.
- ΔJ_nm CIs are ±.10–.15. The near-miss axis is weakly powered: about 59 near-miss units per seed.

**SUP-h: higher discrimination or higher bias?** Both, depending on the axis and the context.

On composition pairs SUP-h gains *hits*, not changes in general:

| | Δ vs S1R1 |
|---|---|
| H_sub | +.047 [−.005, +.097] |
| FA_sub | +.004 |
| bias_sub | +.011 |

Per decision type (ΔJ_sub vs S1R1):

| decision type | ΔJ_sub | Δbias_sub |
|---|---|---|
| q2_after_notH | **+.209 [+.089, +.358] 5/5** | +.074 |
| first | +.015 | +.064 [+.024, +.112] |
| q2_after_H | +.012 | +.068 |
| after_probe_failed | −.011 | −.054 |
| after_b1_timeout | **−.280 [−.410, −.152] 0/5** | −.139 |

- The gain sits in the one context where the exact not-H belief update matters. P1 found that update is never learned by the predictors. It is a privileged-input effect, not general interaction reasoning.
- On the near-miss axis SUP-h's flip gain (+.094 [+.013, +.174]) is repaid by near-miss (−.159 [−.303, −.029]). ΔBA_nm is +.007 and H_nm is .403, the same as the other models. SUP-h cannot tell the full combination from a weaker one any better than S1R1.
- UCE (descriptive): SUP-h ΔJ_sub is +.008 [−.073, +.081], but Δbias_sub is +.050 [+.021, +.080] (5/5), so there its advantage is bias.

## 3. Do factor inputs change discrimination or only bias?

Within-model φ contrasts, pooled SCE (own − zero / own − mean):

| model | ΔJ_sub own−zero | ΔFA_sub own−zero | ΔH_sub own−zero | ΔJ_sub own−mean | ΔBA_nm own−zero / own−mean |
|---|---|---|---|---|---|
| CONS-exact (exact φ) | +.056 [+.019, +.095] 5/5 | −.069 [−.093, −.050] | −.013 | +.105 [+.072, +.135] | +.005 / +.002 |
| CONS-mix (exact φ) | +.058 [+.036, +.080] 5/5 | −.065 | −.007 | +.093 | +.005 / −.005 |
| CONS-oof (exact φ) | +.039 [+.013, +.066] 5/5 | −.058 | −.019 | +.069 | −.003 / −.000 |
| SUP-h | +.131 [+.094, +.167] 5/5 | −.128 [−.156, −.103] | +.002 | +.133 [+.089, +.176] | +.041 [−.009, +.096] / −.014 |
| SEP | +.021 [+.004, +.040] 5/5 | −.045 | −.024 | +.054 | +.000 / −.002 |
| S1R1 | +.002 [−.012, +.016] | −.014 | −.012 | +.015 [−.001, +.031] | +.007 / −.002 |
| S0R1 | −.002 | −.005 | −.006 | +.004 | +.007 / +.003 |
| LRN-h | +.003 | −.015 | −.012 | +.022 [+.001, +.043] | −.008 / −.002 |

- **For models that actually read the channel (consumers, SUP-h, SEP):** factor inputs change *composition discrimination* (J_sub +.02 to +.13, 5/5 seeds), but only by lowering false alarms. Removing the inputs makes the model change its action where the optimum does not change (bias_sub +.03 to +.11). Hits do not rise with the factors.
- This is the same effect as the invariance/pool gains in the screen, measured pairwise.
- **Near-miss balanced accuracy is untouched by the factor channel in every model** (|ΔBA_nm| ≤ .041, all CIs span 0).
- **For the 2×2 R1 arms and LRN-h:** φ barely matters (|ΔJ| ≤ .02), consistent with non-reading.
- **Answer:** factor inputs mostly act as bias control, suppressing spurious changes. They are not magnitude/interaction discrimination.

## 4. Upper references: is discrimination achievable from pairwise structure?

- **π\*:** J = 1 on every pair set by construction; bias_sub .175 is the true base rate.
- **add2** (pairwise-additive Q from the octet's exact pair, single and none labels):
  - SCE: J_sub **.919** [.893, .943], flip .914, invariance .960.
  - UCE: J_sub .873, flip .831.
- **add1** (first-order):
  - SCE: J_sub .774, flip .695.
  - UCE: J_sub .585, flip .441.
- **So yes.** The composition discrimination these octets test is almost entirely recoverable from pairwise structure, and mostly from first-order structure. Every trained model (J_sub ≤ .43) sits far below even the first-order oracle.
- **Caveats:**
  - The oracles use *exact* constituent Q*, which no arm ever observed for pairs (P1 Q6).
  - Neither oracle has a near-miss decision, because near-miss constituents are unlabelled. The near-miss axis therefore has no additive-oracle bound.
  - The shortfall is in extracting and composing constituent values, not in any need for third-order structure.

## 5. Per decision type (J_sub, SCE, point estimates)

CIs are in frontier.json. after_probe_solved has no optimum-changing pairs.

| model | first | after_probe_failed | after_b1_timeout | q2_after_H | q2_after_notH |
|---|---|---|---|---|---|
| add2 | .968 | .833 | .956 | .880 | .854 |
| S0R0 | .300 | .371 | .363 | .262 | .022 |
| S1R1 | .342 | .414 | .388 | .277 | .130 |
| CONS-exact-pred | .319 | .459 | .416 | .288 | .047 |
| SEP | .294 | .372 | .338 | .303 | −.035 |
| LRN-h | .334 | .319 | .068 | .311 | .036 |
| SUP-h | .357 | .403 | .108 | .289 | .339 |

- q2_after_notH is near-zero discrimination for every model except SUP-h: the belief-update gap.
- after_b1_timeout is poor for both historical on-policy lineages (LRN-h, SUP-h), but fine in the bank-mode arms.

## 6. Recommended minimal next test

The registered P2 selection rule was flip gain ≥ .03 with near-miss ≥ −.05. It is **not bias-proof**. Along the observed trade-off (slope −1.8), a pure criterion shift that buys +.03 flip costs only ≈ −.054 near-miss, which is at the non-inferiority edge. Nearly all flip variance across models is hit rate (r = .92).

For the planned explicit-interaction consumer test (deeper MLP vs bilinear/interaction consumer, same information):

1. **Judge on discrimination, not on flip.**
   - Primary: ΔJ_sub vs the matched MLP consumer, with a margin of .03.
   - Co-primary: near-miss discrimination, ΔBA_nm ≥ 0 (non-inferiority −.02) and ΔJ_nm reported.
   - Guard: ΔFA_sub ≤ +.02 (no bias increase) and near-miss accuracy non-inferior at −.05.
   - Flip accuracy becomes secondary.
2. **Same information, same φ:** evaluate each consumer at own / exact / zero / mean. An interaction consumer that only raises J_sub through FA suppression under exact φ is doing what current consumers already do.
3. **Power:** BA_nm/J_nm contrasts at 5 seeds × 320 octets have CI half-widths of .05–.15. The near-miss axis needs more near-miss units before it can confirm a .03 effect: pool b6d's 320 octets in the confirmation, or add near-miss-rich octets. J_sub at the current size resolves about .04.
4. **Pre-register q2_after_notH as a stratum.** It is where the only composition gain in the screen (SUP-h) lives, and where every deployable model is at J ≈ 0. It tests belief acquisition, not the consumer, so report it separately so it cannot masquerade as an interaction effect.

## Provenance and compute

- **Inputs** (read only): 580 files `diag-v1-cf-{SCE,UCE}-*.jsonl.gz` from `~/structured-latent-dynamics-campaign07/results/e07-p2-eval-s4{0..4}` and `e07-p2-evalX-s4{0..4}`. The per-file sha256 is in `research/results/campaign-07/p2-frontier/inputs.sha256` (digest in manifest.json).
- **Code:**
  - `research/tools/campaign07_frontier.py` does the analysis and bootstrap, reusing `campaign07_diagscore.draws/two_level/_ci`.
  - `research/tools/campaign07_frontier_tables.py` does formatting only.
- **Outputs:** `research/results/campaign-07/p2-frontier/{frontier.json, tables.md, inputs.sha256, manifest.json, receipt-v2-occupancy.json}`. frontier.json sha256 442e0418…a08, verified against the remote copy.
- **Compute:** metered on pro6000, 1 thread, max RSS ≈ 1 GB. Smoke 3.4 + v1 310.8 + v2 363.8 = **678 core-s** (cap 2,000). v1 is superseded: v2 is the same code plus the post hoc H_nm/J_nm endpoints, and every number here is from v2.
- **Local compute:** table formatting only (< 5 core-s). No GB10 analysis compute. Outputs remotely only under results/dev; nothing under campaign06 was written.
