# Extended-06 B-FACT-C (adaptive confirmation): independent audit

**Auditor:** independent agent, 2026-09-27. **Base:** `campaign/extended-06` @ 71b0221c. **Branch:** `campaign/e06-bfc-audit`.

**Scope.** Registry entry B-FACT-C: the primary contrast LRN − RAWF (Q1–Q3) and the secondary contrasts SUP − RAWF, LRN − B0, RAWF − B0 and SUP − B0. All were evaluated on the fresh `b6c_hold_SCE` pool (400 configurations) and 320 fresh S+C+E octets, with seeds 35–39.

**Inputs read.**
- In the repository: registry.json (B-ARMS, B-FACT, B-FACT-C), decisions.md, seed-ranges.json and review/b-independent-audit.md; `git show 228b2f41`, 9795f9a2, 096b6ff5 and 71b0221c.
- Code: `src/tensegra/campaign06_probeworld.py` (SPLITS6C, CF_BASE_C, audit_b6c, counterfactual_set); `research/tools/campaign04_probeworld_train.py` (ProbeNet fuse arm, losses, cmd_train, the b6c label and eval paths, and the diff f5ff2d64 → 096b6ff5); `campaign06_bprimary.py` (`--two-level`); `campaign06_bscore.py`.
- Staged results: `research/results/campaign-06/b-factc/{primary_rawf,secondary_b0}.{json,md}`.

**Raw data.** Pulled read-only by tar pipe from pro6000 `~/structured-latent-dynamics-campaign06/results/`:
- for all 20 runs `e06-tb-{lrn,rawf,sup,b0}-s{35..39}/run/`: `eval_b6c_episodes.jsonl.gz`, `cf_eval_b6c.json`, `eval_b6c.json`, `train_meta.json`, `train_log.json`;
- `e06-tbc-labels/labels/*.json` (labels_meta.b6c.{eval,cf}, `cf_SCE.json`, `b6c_hold_SCE_s0.json`);
- every `e06-tb-*-s3[5-9]-process` and `e06-tbc-*-process` `{launch,occupancy}.json` and `process.log`.

**Method.**
- **Own code.** All endpoints were recomputed with independently written code. It parses the episode rows and octet records directly and does not import `campaign06_bscore`. Contexts are rebuilt from `rec`/`step_in_query`, correctness is `delta <= .5`, and the one-factor pairs are rebuilt from the octet members.
- **Own bootstrap.** Numpy, vectorised with multinomial configuration weights. It has two variants:
  1. **Shared draw** (as the tool does): seed pairs are resampled with replacement, then one configuration/octet draw is shared by all drawn pairs.
  2. **Independent draw:** each drawn pair slot gets its own configuration draw.
- **The campaign's own `contrast_two_level`** was also re-run, with other bootstrap seeds and n_boot, to separate Monte Carlo error from any implementation difference.

## Summary of verdicts

| # | Check | Verdict |
|---|---|---|
| 1a | Point estimates, per-pair values, signs and staged CIs for all 5 contrasts × 7 endpoints, recomputed from raw rows | **CONFIRMED** (exact; the tool at seed 1000 / n_boot 1000 reproduces every CI to 4 d.p.) |
| 1b | Q1 (gap regret upper bound < 0), robustness to the bootstrap seed and n_boot | **CORRECTION NEEDED.** Q1's CI condition is a Monte Carlo coin flip. The converged two-level upper bound is **> 0** (≈ +0.02 to +0.06). |
| 1c | Q2 and Q3 robustness | **CONFIRMED**: robust to the bootstrap seed, n_boot and the bootstrap variant |
| 2 | Protocol fidelity: pre-registration, freshness and disjointness, training as registered, no selection | **CONFIRMED**, with 3 disclosures (D1–D3) |
| 2b | "LRN and RAWF differ only in the auxiliary heads" | **CONFIRMED in substance, CLARIFY.** The heads' loss also trains the shared trunk. The heads add 19,479 parameters, and the fuse-layer initialisation differs. |
| 3a | Flip-set accuracy caveat −.048 | **CONFIRMED** (−.0484; 5/5 pairs; CI upper bound between −.002 and −.008 across bootstrap seeds) |
| 3b | Where the flip loss and the Q2 gain sit, and what the behaviour is | **CORRECTION NEEDED (interpretation).** Both sit in the later-query decisions. "Fewer spurious changes" is overstated: LRN changes less in both directions and discriminates no better. The gain does not come from fewer builds or probes. |
| 4 | Learned-head prediction quality | Descriptive only: training-distribution auxiliary MSE ≈ .005–.006 in all 5 seeds. Held-out prediction quality was not logged, so nothing can be concluded. |
| 5 | Disclosure of the adaptive labelling and the development evidence | **Mostly adequate. CORRECTION (minor):** add the selection and multiplicity sentence and the timing facts |

**Net.** Under the procedure as executed (tool default n_boot = 1,000, bootstrap seed 1000, both committed before any b6c data), the registered rule returns SUPPORTED. That verdict is not robust: at n_boot ≥ 5,000 the same tool gives Q1 FAIL in 6 of 7 bootstrap seeds (1/5 pass at 5,000; 0/2 at 20,000). Q2 and Q3 hold robustly. The claim should be restated as: **the later-decision gain (Q2) and not-a-global-shift condition (Q3) confirm; the gap-regret improvement (Q1) is borderline, not established at 95%.** Replacement wording follows in section 6.

## 1. Recomputation and bootstrap sensitivity

### 1a. Reproduction

Own code, all 400 configurations, 320 octets. World seeds are identical across all 20 runs.

LRN − RAWF per pair (s35..s39):

| endpoint | mean | per pair |
|---|---|---|
| gap_regret | −2.649 | −2.031 / −0.658 / −1.370 / −2.629 / −6.558 |
| later_acc | +.0178 | +.008 / +.024 / −.003 / +.038 / +.021 |
| first_acc | +.0010 | +.017 / +.050 / −.088 / +.030 / −.005 |
| cf_near_miss_acc | +.1051 | +.034 / +.068 / +.051 / +.271 / +.102 |
| iv_false_change | −.0319 | −.012 / −.063 / −.026 / −.061 / +.002 |
| cf_flip_full_acc | −.0484 | −.031 / −.070 / −.055 / −.055 / −.031 |
| cf_balanced_acc | +.0254 | +.008 / −.008 / .000 / +.085 / +.042 |

These are identical to `primary_rawf.md`. All secondary contrasts (SUP − RAWF, LRN − B0, RAWF − B0, SUP − B0) are also identical to the staged tables. The tool re-run with `seed=1000, n_boot=1000` reproduces every staged CI exactly (gap [−5.8728, −0.0285]) and verdict SUPPORTED.

### 1b. Q1 is within Monte Carlo error of the threshold

Two-level 95% upper bound on the LRN − RAWF gap regret, using **the campaign's own `contrast_two_level`**:

| n_boot | bootstrap seeds | upper-bound range | Q1 CI condition passes |
|---|---|---|---|
| 1,000 | 1000–1019 | −0.142 … **+0.353** | **8/20** |
| 5,000 | 1000–1004 | −0.030 … +0.136 | **1/5** |
| 20,000 | 1000–1001 | **+0.018 … +0.057** | **0/2** |

My independent numpy implementation (shared draw) agrees:
- n_boot 1,000 over 10 seeds: upper bound −0.19 … +0.16;
- n_boot 5,000: −0.02 … +0.14;
- n_boot 20,000: +0.06 … +0.09;
- at n_boot 100,000, P(bootstrap mean ≥ 0) = **2.7%**, and the 97.5th percentile is **+0.040**.

The converged upper bound is therefore slightly **above** zero. The staged −0.028 comes from one favourable 1,000-draw realisation.

Other views of the same five pairs:
- Across-pair t(4) 95% CI: [−5.51, +0.21].
- Sign test 5/5 negative: one-sided p = .031.
- **Independent-draw bootstrap** (each drawn pair slot gets its own configuration draw): [−5.46, −0.41]. This variant is narrower because it breaks the configuration correlation that the design actually has. All pairs are evaluated on the same 400 configurations, so the tool's shared draw is the appropriate one.

**Concentration** (descriptive):
- In each pair the median per-configuration difference is 0, and 22–36% of configurations are exactly tied.
- Pooled over pairs, the 20 configurations (5%) with the largest LRN advantage account for −1,197 of the total −1,060. Over the other 380, LRN is slightly worse (mean +0.36 per configuration).

Heavy-tailed regret makes this pattern common, but it is why the upper bound sits at zero.

**Verdict 1b: CORRECTION NEEDED.** n_boot was not fixed in the registration. The tool default (1,000) was committed at 9795f9a2 (11:01:53Z), before any b6c evaluation, so the executed procedure is formally the registered one and its verdict stands as executed. The report must nevertheless state that Q1 does not survive a converged bootstrap. Section 6 gives the wording.

### 1c. Q2 and Q3 are robust

LRN − RAWF, two-level 95% CI across bootstrap seeds and n_boot (1k / 5k / 20k), plus the independent-draw variant:

| endpoint | lower bound range | upper bound range | independent-draw CI | verdict |
|---|---|---|---|---|
| later_acc | +.0010 … +.0033 | +.032 … +.035 | [+.004, +.032] | Q2 passes robustly (4/5 pairs positive) |
| near-miss | +.004 … +.018 | — | — | Q3 passes (mean +.105 ≥ −.02) |
| false-change | — | −.011 … −.007 | — | Q3 passes (mean −.032 ≤ +.02) |
| flip-set acc | — | −.008 … −.002 | — | the caveat holds (entirely below 0) |

- The across-pair t(4) CI for later_acc is [−.0015, +.037], so Q2 also rests on the configuration level.
- Q3 is a point-estimate condition, so it has no Monte Carlo sensitivity.

## 2. Protocol fidelity

**Registration predates data: CONFIRMED.**
- Commit 228b2f41 (10:47:48Z) adds the complete B-FACT-C entry: hypothesis, design, primary Q1–Q3 with the ≥ 4/5 sign rule, and secondary contrasts. Its status reads "registered (ADAPTIVE: after inspecting B-FACT descriptive results on b6_hold_SCE)".
- The first B-FACT-C job (`e06-tb-lrn-s35`) started 10:48:04Z, per `launch.json` `started_unix` 1790506084.
- The b6c tooling (9795f9a2, 11:01:53Z) and the bases (096b6ff5, 11:02:35Z) were committed before the label jobs (11:02:42Z / 11:02:43Z, run from `source-096b6ff5`).
- All 20 evals started at 11:53–11:56Z; primary and secondary scoring ran at 11:58Z.
- All 44 receipts (20 trainings, 20 evals, 2 label jobs, primary, secondary) have `exit_code` 0.

**Fresh and disjoint pool and octets: CONFIRMED.**
- `audit_b6c()` rerun locally: pass. Disjoint from every b6 pool, world block, training stream and cf stream; inside the hold and cf sub-ranges.
- The labels_meta of both label jobs embed the same passing audit.
- `check_b6c_labels` asserts base 6.42e9 and that every octet seed = 6.65e9 + index.
- Checked against every range in seed-ranges.json, the new blocks overlap only their own entries and their parents. The blocks are configurations 6.42e9 + [0, 4000), worlds 6.47e9 + [0, 4e6), octets 6.65e9 + [0, 4000), and training worlds 11.5–12.0e9.
- 6.42e9, 6.47e9 and 6.65e9 are the next unused slots of the b6 patterns (hold bases 6.40/6.41; worlds 6.45/6.46; cf bases 6.60–6.64e9).
- Octet construction uses only seed base + index (exact labels, no worlds).
- Training worlds (8e9 + 1e8·seed + counter) never meet the eval blocks.
- S+C+E is a family-level hold, excluded from the b6_B0 training pool by construction (B-ARMS audit).

**Training as registered: CONFIRMED.**
- All 20 runs use `source-f5ff2d64`, `--rung L1 --train-split b6_B0` and the b6 labels dir `e06-tb-labels/labels`.
- train_meta for every run: updates 4000, batch 64, lr 1e-3, hidden 128, train_n 384 with identical combo counts.
- LRN: `--arch fuse --factor-mode learned --aux-weight 1` (weights include aux 1.0). RAWF: `--arch fuse` (factor_mode none). SUP: `--inputs factors6 --arch fuse` (in_dim 95). B0: default.
- `model.pt` is written once, after the final update (`cmd_train`, line 781). There is no checkpoint selection, and b6_dev is monitoring only.
- The eval snapshot 096b6ff5 differs from f5ff2d64 only by additive b6c code paths: new split set, own output names, label checks. There is no change to the model or eval semantics.

**Disclosures (not errors):**
- **D1.** The fresh bases (6.42e9 / 6.65e9) and the seed-range entry for training worlds 35–39 were committed at 11:02Z, 14 min after training started. Training does not touch the b6c blocks, and the bases were fixed before any b6c label or eval existed, so there is no leakage.
- **D2.** n_boot was not in the registration text. The tool default of 1,000 was fixed before data (see 1b).
- **D3.** The b6c labels were built after the trainings started but before any evaluation. That matches the registered wording "labels built before any confirm evaluation".

**2b. "Differ only in the auxiliary heads": CONFIRMED in substance, with clarifications.**
- **Parameter counts.** LRN 168,636 vs RAWF 149,157. The difference, 19,479, equals the auxiliary MLP exactly: 128·128 + 128 + 128·23 + 23.
- **RAWF's fuse input.** RAWF's fuse layer has the same 23 φ-input columns, but it is fed zeros.
- **Common training streams.** Both arms share `torch.manual_seed(1000 + seed)` and the same data stream (`data_rng` = 7000 + seed, identical training configurations and world seeds). They also share the action-sampling RNG seed.

The heads nonetheless make three further differences:
1. **The auxiliary loss also trains the shared trunk.** `fuse_step` detaches only the prediction fed to the fuse layer (`phi = aux(z).detach()`). The factor-regression loss back-propagates through `aux(z)` into the trunk and GRU, and it enters the global grad-norm clip (1.0). LRN − RAWF therefore bundles (i) representation shaping by an auxiliary prediction task and (ii) the policy reading stop-gradient predictions. This design does not separate them. The registry wording "downstream policy consumes predictions" (B-FACT arms) describes only (ii).
2. **Fuse-layer initialisation.** The aux module is created before `fuse`, so the fuse-layer initialisation differs between arms. Every other parameter is identically initialised.
3. **Compute.** LRN uses ~7% more training CPU (~466 vs ~438 core-s).

None of these is a protocol violation. They belong in the interpretation.

## 3. The flip caveat, where the gains sit, and behaviour

**Octet endpoints by decision type** (LRN − RAWF, mean over 5 pairs; n = units per run):

| decision type | flip-set acc (n) | near-miss acc (n) | P(change \| opt same) | P(change \| opt changes) |
|---|---|---|---|---|
| first | **+.029** (28) | +.120 (15) | −.021 | −.030 |
| after_probe_solved | — (0 flips) | — | −.035 | — |
| after_probe_failed | .000 (8) | +.067 (3) | −.029 | **−.112** |
| after_b1_timeout | +.011 (18) | .000 (1) | −.025 | −.056 |
| **q2_after_H** | **−.220 (20; 5/5 pairs)** | **+.233** (12) | −.037 | −.045 |
| q2_after_notH | −.052 (54) | +.050 (28) | −.045 | −.023 |

- **The flip loss sits in the later-query decisions**, mainly the first decision of query 2 after the verifier reveals type H: RAWF .730 → LRN ≈ .51. On the first decision, LRN is slightly better on flips.
- **Near-miss gains come from the same place.** In those decision types LRN more often takes the action that the one-factor ablations would take. That is right on near-miss configurations and wrong on true flips. This is lower sensitivity to the full interaction, not composition.

**One-factor interventions** (pooled over decision types, mean of 5 seeds):

| arm | P(change \| opt changes) | P(change \| opt same) | difference (discrimination) | flip acc | near-miss acc |
|---|---|---|---|---|---|
| LRN | .505 | .140 | .364 | .405 | .749 |
| RAWF | .551 | .172 | .378 | .453 | .644 |
| SUP | .602 | .133 | **.469** | .562 | .536 |
| B0 | .547 | .169 | .377 | .475 | .647 |

**LRN changes its action less often under one-factor interventions in both directions.** It changes less both when it should (−.046; 5/5 pairs negative) and when it should not (−.032). Its discrimination is not better than RAWF's (.364 vs .378). "Fewer spurious one-factor changes" is therefore true as a rate but overstated as an improvement: it comes with proportionally fewer correct changes. SUP, by contrast, genuinely discriminates better.

**Q2 by context** (LRN − RAWF):

| context | RAWF acc | LRN − RAWF acc (per pair) | Δ errors per episode |
|---|---|---|---|
| **query_first** (first decision of queries 2..k) | .607 | **+.079** (+.069 / +.090 / −.005 / +.169 / +.072) | **−0.304** |
| after_probe_solved | .996 | −.002 | +0.003 |
| after_probe_failed | .845 | −.010 | +0.008 |
| after_b1_timeout | .878 | −.054 (n ≈ 45) | +0.008 |
| other | .856 | +.002 | −0.070 |

- The Q2 gain is almost entirely the first decision of later queries, as in development (+.126 there).
- In that context RAWF's dominant error is taking `inspect` when the optimum is `use` (after a build), `probe` or `prop`. Such errors fall from ≈ 1.28 to ≈ 0.79 per episode under LRN.
- LRN's accuracy there is flat across query index (.71 → .67), so this is not a learning-within-episode effect.

**Behaviour: this is not conservatism about builds or probes.**

| arm | build rate | probes per episode | steps | first-action mix |
|---|---|---|---|---|
| LRN | .350 | 3.00 | 16.5 | similar |
| RAWF | .364 | 3.04 | 16.9 | similar |
| π* | .333 | — | — | — |

- The gain is fewer redundant `inspect` actions at query starts, i.e. less re-gathering once the cheaper continuation is determined. LRN takes ~0.4 fewer later decisions per episode.
- Q2 is defined on the model's own trajectory, so its denominators differ slightly by arm (registered; noted).
- Gap regret: the gain appears both where π* does not build (−2.83; 267 configurations) and where it builds (−2.28; 133 configurations, 2 pairs positive), and at k = 2 (−1.88; 5/5 pairs) and k = 8 (−3.50).

**Verdict 3a: CONFIRMED** (−.048). **Verdict 3b: CORRECTION NEEDED** on the interpretation (section 6).

## 4. Learned-head prediction quality (descriptive)

- `train_log` records the auxiliary MSE on the training batches only. It falls from .33–.45 to .0053–.0064 (mean of the last 5 logs) in all 5 LRN seeds; the heads fit the derived quantities on the training distribution.
- The rank correlation with per-pair effects is −0.7 for gap, +0.7 for later and +0.5 for flip. With n = 5 and a narrow range, this is noise. **Nothing can be concluded.**
- **Prediction quality on S+C+E (held-out) was not measured.** The registry claim that the quantities carry interaction-relevant information "that the learned heads do not exploit" therefore cannot distinguish two explanations: the heads mispredict on S+C+E, or the policy does not use good predictions.

## 5. Adaptive labelling and development evidence

**Adequately disclosed:**
- The adaptive status appears in the registration commit, the current status ("SUPPORTED (adaptive; …)") and decisions.md (10:47Z and 11:59Z entries).
- B-FACT.result gives the development numbers: LRN − RAWF later +.030 (3/3), concentrated in query_first (+.126); gap −5.5; near-miss −.022.
- decisions.md (11:17Z) records that development near-miss was just outside Q3's margin and that Q3 was not changed.
- Fresh seeds, configurations and octets mean the selection cannot bias the confirmation estimates themselves.

**Not stated:** the contrast and endpoint were chosen as the best-looking result among many descriptive comparisons. On b6_hold_SCE, seeds 30–32, eight arms were compared with B0 (B1, B3, dose1–3, SUP, LRN, RAWF) plus LRN − RAWF, each on several endpoints. The winner's development effect (later +.030, gap −5.5) was about 2× the confirmed effect (+.018, −2.65), the usual winner's-curse shrinkage. The report should say so (section 6).

## 6. Corrections: exact replacement wording

**C1. registry B-FACT-C.status.** Replace with:
> complete: SUPPORTED by the registered procedure as executed (adaptive; n_boot 1,000, bootstrap seed 1000). Q1 is borderline: at a converged bootstrap the gap-regret upper bound is > 0 (independent audit, review/bfc-independent-audit.md)

**C2. registry B-FACT-C.result.** Append:
> AUDIT (review/bfc-independent-audit.md): all numbers reproduce exactly from raw rows. Q1's CI condition is within Monte Carlo error. With the same tool the gap upper bound ranges from −0.14 to +0.35 across 20 bootstrap seeds at n_boot 1,000 (passes in 8/20), and is +0.02 to +0.06 at n_boot 20,000 (0/2). P(bootstrap ≥ 0) = 2.7%. The across-pair t(4) CI is [−5.5, +0.2]; 5/5 pairs are negative (sign test p = .031). The gap gain is concentrated: 5% of configurations carry more than all of it. Q2 (lower bound +.001 to +.003) and Q3 are robust. The flip-set loss (−.048) sits in the query-2 decisions (q2_after_H −.22, 5/5 pairs; q2_after_notH −.05); first-decision flips are +.03. LRN changes its action less under one-factor interventions both when the optimum changes (−.046, 5/5) and when it does not (−.032); discrimination .364 vs RAWF .378. Build and probe rates are unchanged (.350 vs .364; 3.00 vs 3.04 probes). The Q2 gain is the first decision of later queries (+.079; RAWF's redundant 'inspect' errors 1.28 → 0.79 per episode).

**C3. registry B-FACT-C.interpretation.** Replace with:
> On a fresh S+C+E pool and fresh seeds, auxiliary heads trained to predict derived decision quantities (LRN) improve later-decision accuracy over the identical fuse architecture without them (RAWF). The improvement is small (+.018; 4/5 pairs) and robust, and comes almost entirely from the first decision of later queries, where LRN makes fewer redundant 'inspect' actions. Held-out gap regret is lower in all 5 pairs (−2.65), but the 95% two-level CI only just touches zero and does not exclude it at a converged bootstrap, so the regret gain is suggestive, not established. LRN does NOT use the interacting factors better. It is less responsive to one-factor changes overall (fewer false and fewer correct changes; discrimination unchanged). It is more ablation-like exactly in the query-2 decisions: near-miss accuracy is higher, and accuracy is lower on octet flips that require the full interaction (−.048 overall; −.22 after the verifier reveals H). The heads' loss also trains the shared trunk, so representation shaping and reading the predictions are not separated. Supplying the exact quantities (SUP) improves flip accuracy (+.109) and interaction discrimination (.469 vs .378) at a near-miss cost. The quantities therefore carry interaction-relevant information that the learned route did not deliver on the held-out triple. Whether that is because the heads mispredict on S+C+E or because the policy does not use them was not measured.

**C4. registry B-FACT-C.notes.** Append:
> Adaptive selection: LRN − RAWF and the later-decision endpoint were chosen after inspecting ~9 descriptive contrasts × several endpoints on b6_hold_SCE (seeds 30–32). The development effects (later +.030, gap −5.5) were about twice the confirmed ones. n_boot was not registered; the tool default of 1,000 was fixed at 9795f9a2 before any b6c data. The b6c bases and the seed-range entry for training worlds 35–39 were committed at 11:02Z, after training started (10:48Z) and before any b6c labels or eval.

**C5. decisions.md, 11:59Z entry.** Change the heading to:
> **B-FACT-C (adaptive) SUPPORTED by its registered rule as executed; Q1 borderline (not robust to bootstrap Monte Carlo error, independent audit).**

Also replace the caveat line's "improve value-sensitive later decisions and reduce spurious changes" with:
> improve the first decision of later queries (fewer redundant inspects) and change actions less often under one-factor interventions (both false and correct changes; no better discrimination)

## 7. Compute

**Local: ~260 core-s**, all on this workstation (no GB10, no GPU):
- own-code analysis: 54 core-s;
- re-running the campaign's `contrast_two_level` across 27 seed/n_boot settings: 194 core-s;
- misc: ~10 core-s.

**Remote (pro6000): read-only tar pipes of ~21 MB compressed, a few core-s.** No remote job was launched, so `metered.sh` was not needed. Nothing was killed, detached or deleted.

**Scripts:** `review/bfc-audit/{audit.py,their.py}`. They expect the pulled files under `./bfc/` next to the script.
