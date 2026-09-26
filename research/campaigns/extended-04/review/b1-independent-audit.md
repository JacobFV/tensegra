# B1 independent audit (probeworld factorization-supervision ladder)

**Auditor:** independent. I did not build probeworld, the ladder trainer or the scorer. **Base:** `campaign/extended-04` @ 48286d97, on branch `campaign/e04-b1-audit`. **Date:** 2026-09-26.

**Inputs:**
- protocol-B1.md, probeworld.md, decisions.md (19:40Z, 19:55Z) and the campaign.md guardrails;
- research/results/campaign-04/b1/;
- remote results on pro6000, read-only.

I reconstructed the scorer (`research/tools/campaign04_b1_audit_score.py`) before opening `campaign04_b1_score.py`.

**Scripts:**
- `research/tools/campaign04_b1_audit_score.py`: verdicts, run locally;
- `research/tools/campaign04_b1_audit_labels.py`: labels and splits, pure Python, run locally;
- `research/tools/campaign04_b1_audit_remote.py`: leakage, matching, L0 and B-H6; torch, run remotely and metered.

**Outputs:** `b1-audit-score.json`, `b1-audit-labels.json`, `b1-audit-remote.json` and the summary `b1-independent-audit.json`, all in this directory.

## Bottom line

- **Numbers.** All 15 eval.json files, b-refs.json and b1-score.json are internally consistent. My reconstruction gives **0 numeric discrepancies** against b1-score.json.
- **Verdicts.** They match the root verdicts under the root's readings of the protocol.
- **Integrity.**
  - Labels are exact: brute force agrees to ≤ 2e-10, and a Monte Carlo check over 178 Q* tests is consistent.
  - Labels depend only on the visible history.
  - Splits are disjoint by generator parameters.
  - Model inputs are public only: 2,525 steps rebuilt bit-exactly from public information.
  - Rungs share the world stream and initialization, and evaluation used the final checkpoint only.
- **Four reported claims need weakening.**
  1. B-H4 and B-H5 hold only when the held-out splits are pooled. Both fail on heldout_comp in 3/3 seeds.
  2. B-H2's L1→L2 and L2→L3 "adds value" results are within noise and depend on the metric.
  3. The stated likely cause of the B-H6 failure is **refuted**: no rung trains the value head toward V*.
  4. "L0 seeds 0/1 identical behaviour" holds on 3 of 5 splits only.

## 1. Verdicts (mine vs root)

| ID | Root (b1-score.json / decisions 19:55Z) | Auditor | Reading-dependent? |
|---|---|---|---|
| B-H1 | SUPPORTED | **SUPPORTED** | No. Holds on realized and on gap regret. L4/L0 held-out ratio: .082/.104/.104 realized, .083/.082/.067 gap. Every split has 3/3 seeds with L4 ≤ L0. Floor passes. |
| B-H2 | L0→L1, L1→L2, L2→L3 add value; L3→L4 does not | Same under realized regret. **Under gap regret, L1→L2 does not add value** (1/3 seeds). | Yes (metric) |
| B-H3 | SUPPORTED (L4 MAD .02–.05) | **SUPPORTED**. MAD .021–.045 for L4. | No. Holds unweighted or n-weighted, per split or split mean. |
| B-H4 | SUPPORTED (pooled: not-opt .03–.04, unique .88–.91) | **SUPPORTED only pooled.** Per split, heldout_comp unique-opt probe rate is **.636 / .636 / .773** (n = 88), below .80 in 3/3 seeds. | **Yes** |
| B-H5 | SUPPORTED (unjustified .044–.050) | **SUPPORTED only pooled.** Unjustified fraction ≤ .07 on every split. The justified-after-(b) ratio to π* on heldout_comp is **.59 / .67 / .59**, below .8 in 3/3 seeds. | **Yes** |
| B-H6 | NOT SUPPORTED | **NOT SUPPORTED** under every reading: L4 only or L1–L4, each split or split mean. | No |

### Ambiguous protocol wording, and how the root scorer resolved it

All the root's resolutions below were verified in `campaign04_b1_score.py` after my reconstruction.

1. **Regret metric.** "Utility regret against V*" could mean realized V*(I₀) − U, or the gap regret the evaluation also reports. The root uses **realized**. B-H1 is unaffected. For B-H2 the choice flips L1→L2.
   - Realized regret has per-split SE ≈ 1.0–1.4 per episode. The L1–L4 held-out means are all 1.75–4.06, so every rung-to-rung step after L0→L1 sits within about 1–2 SE.
2. **B-H3 bin weighting and π*'s P(build).** The root takes an **unweighted** mean over the populated registered ρ_eff bins, applied to each of heldout_k and heldout_price, and requires both splits to pass per seed.
   - π*'s P(build) is π*'s empirical build rate on the same worlds, not the exact P(build).
   - The verdict is unchanged under every variant I tried.
3. **B-H4 and B-H5 split scope.** The protocol does not name the splits. The root **pools the three held-out splits**, n-weighted.
   - B-H5's "π*'s rate" is read as justified switches after case (b) per episode, pooled, against π* on the same worlds.
   - B-H5's "after case-(b) failed probes" is implemented as after **any** case-(b) step, which includes b1 timeouts, not only failed probes.
   - Reading the ratio conditionally (per case-(b) step) also passes pooled: .93–.95.
   - **Per split, both B-H4 and B-H5 fail on heldout_comp.** That split holds the side_effect+correlated and unreliable+events combinations, so it is exactly where the probing decision is new.
4. **B-H6 field and rungs.** The root uses `value_calibration_own_return.reliability_error`, the n-weighted |mean V̂ − mean return| over 10 quantile bins of **per-step** pairs, in units of R.
   - It reads "rungs with Q/V supervision" as **L4 only**, which is correct: only L4 has Q supervision and no rung has V supervision (§5).
   - It requires every held-out split ≤ .05 in ≥ 2/3 seeds. The protocol states no seed rule for B-H6.
   - The decisions text's ".13–.29 R for supervised rungs, .09–.13 R for L0" is the per-seed **worst held-out split** over L1–L4. L4 alone is .128–.155 on its worst split, and on heldout_price it is .027/.032/.052, which passes in 2/3 seeds.
5. **Transfer floor.** The root applies the floor only in B-H1. Every run passes it on every held-out split (success ≥ .993 vs π* ≥ .996), so this has no effect.
6. **Minor spec deviation.** Evaluation world seeds are `base + 5e7 + 1000·i + 500 + r`; probeworld.md says `+ r`. The ranges are still disjoint from every other range (checked), so this is harmless but undocumented.

## 2. Label integrity (local CPU, pure Python)

- **Splits by generator parameters.** I checked train 0–383 and each eval split 0–127, re-deriving each value from the generated Config itself (cell from prices, `k`, flags). All 12 checks pass with 0 problems:
  - train has no centre cell, no k = 4 and no held-out flag pair;
  - heldout_price uses only cell (1, 1), heldout_k only k = 4, and heldout_comp only the two held-out pairs;
  - apart from its held-out parameter, each held-out split uses only training parameters;
  - every training flag is seen singly;
  - world-seed and config-seed ranges are disjoint;
  - no configuration is duplicated across splits.

  Remotely, `train.pkl` (384 configs) contains k ∈ {1, 2, 8} and 8 cells, with no held-out parameter.
- **Brute force.** This is my own expectimax over **raw visible histories**:
  - the posterior comes from likelihood enumeration over θ;
  - the action, outcome and candidate rules are re-coded from probeworld.md;
  - event configurations are memoized on my own replay summary, because raw event histories explode.

  It covers 8 k = 1 held-out configurations (heldout_price and heldout_comp), with every flag type represented. V*, all root Q* and depth-1 V* agree with `ExactSolver` to **max abs err 1.9e-10**.
- **Monte Carlo.**
  - **Setup:** I rejection-sampled 400 simulator worlds consistent with each visible history, forced action a, then continued with π*.
  - **Coverage:** 25 histories from π* rollouts on held-out evaluation worlds: heldout_k (k = 4, correlated, including mid-episode queries 2–3), heldout_comp (events, side effect, correlated) and heldout_price.
  - **Result:** 178 Q* tests, mean z −.09, mean z² .54. One |z| > 3 (3.54, commit_infeasible); at 20k and 40k samples it is |z| ≤ .7.
  - **Belief follow-up:** a direct check of the posterior against the empirical frequency of the hidden θ. One cell reached z = 3.2 at 20k samples; at 40k with two fresh seeds, all |z| ≤ 1.5.
- **Labels are a function of the visible history.** Random-policy rollouts over 24 held-out configurations give 9,786 distinct histories, 0 label conflicts, and 0 cases where the simulator state differs from `public_state_from_history`.
- **Pickled labels equal a fresh DP.** Remotely, the configs in all three held-out pickles equal `split_config`. For 9 configurations, a fresh DP gives the same state set and max |ΔV*| = 0.0.
- **Code identity.** The sha256 of probeworld and the trainer is identical in the labels and training snapshot (28152106), the evaluation snapshot (871a9a9d) and the audited tree.

## 3. Leakage and matching

- **Inputs are public only.**
  - *Code:* `encode()` uses `cfg.public_vector()` (prices/R, q, D_side, corr, p_event, η, p_conflict, k, ρ_k, flags, declared prior), the previous visible record (a, o, event, revealed θ after a terminal), the available-action mask and queries done/k. Labels (Q, opt, stage, dep, sw, vstar) are computed after the forward pass and are used only in losses and metrics.
  - *Empirical:* I captured the actual input `x` at every step of greedy L4-s0 batches on heldout_comp and heldout_k and rebuilt it independently from the Config fields plus `history[:t]` through the public transition. Result: **2,525 steps, 0 mismatches, max |Δ| = 0.0**, in_dim 68.
- **Same world stream and initialization across rungs.** For each seed, the logged batch V* is identical across all five rungs at all 41 logged updates, so the configuration stream is the same. The update-0 batch utility U is identical across all five rungs, which requires the same initialization and the same sampling RNG. The world-seed formula depends only on seed and counter.
- **Final checkpoint only.** Each run directory has a single `model.pt`, written once at the end of `cmd_train`; `eval.json` is newer. The 15 evaluation receipts run `eval --run …/run`, which loads `run/model.pt`. Results are the final checkpoints, with no selection.

## 4. The L0 "identical seeds" observation, and L0's strategy

- **Weights differ.** The max |Δ| over parameters is 1.23 for s0 vs s1, 1.15 for s0 vs s2 and 0.98 for s1 vs s2.
- **Greedy behaviour coincides only partly.** Fraction of identical per-episode action sequences:

  | pair | dev | test_iid | heldout_price | heldout_k | heldout_comp |
  |---|---:|---:|---:|---:|---:|
  | s0 vs s1 | 1.000 | 1.000 | 1.000 | **.984** | **.992** |
  | s0 vs s2 | .781 | .797 | .742 | **1.000** | .828 |

  The statement "identical greedy behaviour on the evaluation episodes" is therefore true for dev, test_iid and heldout_price only. On heldout_k, s0 matches **s2** rather than s1.
- **Strategy.** L0 converged to a near-deterministic, never-wrong rule:
  - never probe, never inspect, essentially never exact_b2;
  - every query runs **prop → exact_b1 → commit**, or commit_infeasible after a propagation conflict or a timeout after reduction; a repeated prop/b1 appears after events;
  - s0 and s1 also **build → use** on some high-ρ configs (build rate .26 on heldout_price, 0 on heldout_k); s2 never builds.

  This route always resolves the type: reduction makes b1 solve F, and X is exposed by a conflict or a timeout. So success is 1.0 with 0 wrong commits, but it pays prop + b1 on every query, including the many H-type queries where a probe would do.

  This makes the identical behaviour plausible: argmax behaviour saturates onto one discrete rule, and different weights give the same argmax except near decision boundaries. Those boundaries are the build decision (whether to build or not), and they are what differs on heldout_k and heldout_comp. It is not a seeding fault, since the update-0 streams differ by seed.
- **Consequence for B-H1.** The baseline is a local optimum that outcome-only RL did not escape in 256k episodes; its training regret was still 19–37 at the end. B-H1 therefore measures supervision against a stuck outcome-only learner at this budget, not against a well-optimized RL baseline.

## 5. B-H6 failure: checking the suspected cause

**The hypothesis in decisions.md is refuted by the code.** It says: "L1–L4 value heads are trained toward V* while calibration is against own return."

- In `losses()`, the value loss is `(v − G_t)²` with G_t the **own sampled** Monte Carlo return-to-go, in **every** rung.
- No rung regresses V on V*. L4 adds Q* regression on the **Q head** only.

Empirical check, re-run remotely (it reproduces eval.json exactly):

| | L4 held-out reliability vs own return | vs V* | vs own return of the **sampled** policy the head was trained on |
|---|---|---|---|
| heldout_price | .032 / .027 / .052 | .021 / .027 / .040 | .020 / .025 / .048 |
| heldout_k | .148 / .155 / .128 | .143 / .145 / .136 | .149 / .160 / .132 |
| heldout_comp | .114 / .081 / .095 | .086 / .054 / .050 | .141 / .110 / .089 |

- **The target is not the problem.** For the supervised rungs, own greedy return ≈ V* (bias −.01 to −.06 R), so calibration against own return and against V* differ by ≤ .06 R. The failure is present under either target.
- **The continuation is not the problem.** Calibration against the sampled-policy return is within ±.03 of the greedy figure, so this is not a greedy-vs-sampled mismatch.
- **What is actually happening:**
  - the V head **over-predicts** on heldout_k and heldout_comp: V̂ − G bias is +.03 to +.13 R for L4 and up to +.23 R for L1;
  - it is roughly calibrated in-distribution (L4 dev .046, test_iid .046) and on heldout_price;
  - the failure is a generalization failure of the value head, worst on k = 4.
- **Why L0 looks better calibrated (.01–.13).** Its fixed prop → b1 rule makes the return-to-go highly predictable. Against V*, L0 is badly off (.07–.27), because its return is .10–.34 R below V*.
- **The L4 Q head.** Its max_a Q̂, as a V estimate, fails the same way (heldout_k .127–.168). For L1–L3 the Q head is untrained and meaningless.

**Correct statement:** B-H6 is not supported. The value head, trained on its own Monte Carlo return, generalizes poorly to k = 4 and to the held-out condition pairs, with over-prediction. This is not a V*-vs-own-return continuation mismatch.

## 6. Claims to weaken

1. **B-H4 / B-H5.** Report them as "SUPPORTED when the held-out splits are pooled (root reading). On heldout_comp alone, the unique-optimal probe rate (.64–.77) and the justified-switch-after-(b) ratio (.59–.67) fail in 3/3 seeds." The pooling choice was not registered.
2. **B-H2.** Only L0→L1 is a reliable step. The L1→L2 and L2→L3 "adds value" results are within noise (realized-regret SE ≈ 1 per split) and flip under gap regret (L1→L2 at 1/3 seeds). Say "L1 (A*-set imitation) accounts for essentially all the gain; L2–L4 differences are not resolved at 3 seeds × 512 episodes."
3. **B-H1 wording.** "Factorized supervision helps held-out" should read "**optimal-action-set supervision (L1) and above** beats outcome-only actor-critic, whose policy stayed at a non-probing local optimum at this budget." Factorization beyond A* (L2–L4) is not shown to add anything.
4. **B-H6 cause.** Replace "likely cause: V head trained toward V*" (factually wrong about the trainer) with the §5 finding: held-out generalization of an own-return value head, with over-prediction, worst on k = 4. Also note that L4 passes .05 on heldout_price in 2/3 seeds and fails on heldout_k and heldout_comp.
5. **L0 identical behaviour.** "Identical on dev, test_iid and heldout_price; 98–99% identical on heldout_k and heldout_comp; weights differ." Also state L0's strategy (prop → b1 → commit, never probe).
6. **Spec text.** Document the `+500` evaluation world-seed offset. B-H5's "failed probes" is implemented as all case-(b) steps.

## Compute

- **Remote (metered):** 86.1 core-s, one job, 80 s wall, 1 thread, CPU only. Receipt: `~/tensegra-campaign04/results/dev/b1audit-remote-20260926T203508-533086-process`. A few `ls`/`sha256sum`/`cat` inspection commands ran unmetered (negligible, < 5 core-s).
- **Local CPU (Dell GB10, pure Python):** ≈ 0.4k core-s. This includes ≈ 0.3k core-s lost to an aborted non-memoized brute-force attempt on an event configuration and a timed-out probe. The final label run took 38.5 core-s, plus ≈ 45 core-s of re-runs.
- **Total:** ≈ 0.5k core-s, under the ~1.5k cap. No results were modified.
