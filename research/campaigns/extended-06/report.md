# Extended-06 report: computational strategy selection, structural composition, decision-relevant value learning

**Status: FINAL** (all three independent audits merged and their corrections applied). The campaign ran on the pro6000 (WSL2, CPU only) from 2026-09-27T05:51:06Z, with the window closing at 17:51:06Z.

**Audits** (corrections applied throughout):
- Track A: [review/a-independent-audit.md](review/a-independent-audit.md).
- Track B B-ARMS: [review/b-independent-audit.md](review/b-independent-audit.md).
- B-FACT-C: [review/bfc-independent-audit.md](review/bfc-independent-audit.md).
- A design review preceded implementation: [review/design-review.md](review/design-review.md).

**Sources:**
- Design: [design.md](design.md) (v1 plus v2 revisions 1–13).
- Registry: [registry.json](registry.json). Every entry is labelled prospective or adaptive.
- Decisions: [decisions.md](decisions.md), with timestamps from `date` or git.
- Job plan and seed ranges: [jobplan.json](jobplan.json), [seed-ranges.json](seed-ranges.json).
- Task specifications: [portworld.md](portworld.md) (Track A) and [trackb-screen.md](trackb-screen.md) (Track B).

## Headline

| # | Question (brief §33) | Answer |
|---|---|---|
| 1 | Did we discover a task with genuine computational-strategy headroom? | **Yes over single methods, no over simple portfolios.** In portworld, the per-instance best method beats the best single method by .064 utility (hidden-state oracle .9835 vs .920). A tuned depth-2/3 decision tree over public features already captures ~54% of that. The rest (.029 above the best simple portfolio) is mostly hidden-state information that no public selector reaches. The registered practical-headroom gate (GA-1: learned public-information selector ≥ best simple + .02) **failed**: +.0057 [.0030, .0087]. |
| 2 | Did a learned selector beat the strongest individual and simple portfolio baselines? | **It beat the best single method (+.035 to +.040), but not a tuned simple portfolio in any practical or robust sense.** The fresh-instance confirmation A-CF-SMALL passed its registered test 3/3 at +.0033 [.0016, .0055]. The audit showed this margin: fails at 2× selector compute charge (1/3); reverses under per-output inference charging (0/3); falls to 2/3 against a select-tuned tree; and is zero against a depth-3 tree fit in-sample. Reading: **no practically or robustly detectable learned-selector advantage.** |
| 3 | Did structural-combination diversity transfer beyond mere extra-data effects? | **No.** At matched data volume (B2 vs B0, 5 seed pairs), broader combination variety did not improve decisions on the untouched S+C+E triple. Gap regret −1.65 [−6.1, +3.1]; later-decision accuracy +.010 [−.002, +.022]; near-miss −.085. The registered primary is **not supported**. B2's first-decision gain (+.075) is an input-specific action-preference change (more builds on S∧C inputs), not composition. |
| 4 | Did any factorized/value representation causally improve held-out decisions? | **A small later-decision improvement, adaptively confirmed; the regret gain is not established; there was no interaction composition.** Auxiliary heads trained to predict derived decision quantities (LRN) were compared with the identical architecture without them (RAWF), on fresh configurations and seeds. Later-decision accuracy is +.018 [+.002, +.034] (4/5 pairs, robust). It comes almost entirely from the first decision of later queries: fewer redundant inspects. Gap regret is −2.65 with all 5 pairs lower. The registered two-level CI excluded 0 as executed ([−5.87, −0.03]), but **not at a converged bootstrap** (audit: upper bound +.02 to +.06 at 20k draws), so the regret gain is suggestive. LRN is **less** accurate on the counterfactual flips that need the full interaction (−.048; −.22 after the verifier reveals H), and less responsive to one-factor changes in both directions. Supplying the exact quantities (SUP) raises flip accuracy (+.109) and interaction discrimination at a near-miss cost. The information is interaction-relevant, but the learned route did not deliver it. |
| 5 | Did learned regulation add value where a simple rule could not already capture the opportunity? | **No.** Track A is the regulation test, and a depth-3 tree captures the public-information opportunity. Track C (value/metacontrol arms) was **not triggered**: its precondition, GA-1 headroom, failed. |
| 6 | What was supplied, what was learned, what generalized? | See [below](#supplied-learned-generalized). |
| 7 | What remains unresolved? | See [below](#unresolved). |

**Stopping condition.** The brief lists "a strong portfolio eliminates all supposed metacontrol headroom" as a legitimate stopping point, and Track A reached it. Track B reached its registered answers (B-ARMS negative; B-FACT-C: a small robust later-decision gain, a borderline regret gain, no composition), with the window and budget to spare.

## Track A: portworld algorithm selection

**Task** ([portworld.md](portworld.md), generator pw-v3):
- Constrained packing with conflicts, requires-links, hidden constraint parameters (explicit posteriors; `inspect` reveals them), and episodes of related instances with a reusable-or-stale cached solution.
- Utility = verified quality (value / exact optimum) − priced compute − priced observation − failure loss.
- **Methods (supplied, with contracts):** G, GR, BM (beam), RV (verify-and-commit cache), RU (repair cache), PD (propagate + decompose + B&B), RUPD. That gives a menu of 45 method × budget arms, with budgets read off a single full-budget trace per method.
- **Exact oracle:** hand-written B&B with brute-force cross-checks. It certified 100% of gate instances and 99.86% on confirm; uncertified instances are scored against the upper bound and never dropped.

**A-HS: registered headroom gate** (prospective). Dev_gate range 305M; 640 episodes; 2,891 instances; episode-clustered cross-fitting; best simple family chosen on inner folds.

| policy | utility |
|---|---|
| best single method (A0) | .920 |
| tuned cascade | .924 |
| hand portfolio | .939 |
| strong v2 baseline (design item 8) | .946 |
| threshold / logistic | .946 / .947 |
| trees depth 2 / 3 | .954 / .954 |
| **best simple (inner-selected)** | **.9544** |
| learned one-shot (GBT, cross-fitted) | .9601 |
| learned sequential (telemetry) | .9602 |
| hidden-state oracle (non-deployable) | .9835 |

- GA-1: one-shot +.0057 [.0030, .0087], sequential +.0058: **fail**. GA-2 (vs best single): +.040: pass.
- The development estimate (+.020) came from the data used while revising the generator (v1 → v3), and it did not replicate.
- **No generator revision.** The design permits one revision after a GA-1 failure but does not require it. The hidden-state ceiling above best simple was only .029, so another revision would amount to manufacturing headroom.

**A-CF-SMALL: small-margin confirmation** (registered after the gate; labelled). Setup:
- 3 lineages with disjoint training populations (310M, 311M, 312M).
- Shared selection population 320M; families and hyperparameters were chosen there only.
- Fits frozen with sha256 before any confirm data existed.
- Shared confirm population 330M (640 episodes, 2,852 instances).

| lineage | best simple (chosen on select) | one-shot − best simple | sequential − best simple | one-shot − best single |
|---|---|---|---|---|
| 0 | tree_d3 | +.0032 [+.0006, +.0067] | +.0034 [+.0004, +.0069] | +.0351 |
| 1 | tree_d3 | +.0025 [+.0005, +.0046] | +.0032 [+.0012, +.0053] | +.0349 |
| 2 | tree_d3 | +.0043 [+.0014, +.0080] | +.0052 [+.0022, +.0087] | +.0352 |
| paired mean | | **+.0033 [+.0016, +.0055]** (3/3 registered pass) | +.0039 [+.0021, +.0062] | |

**Audit sensitivity** (§3 and §5b of the audit; not registered):

| scenario | lineages passing |
|---|---|
| zero selector charge | 3/3 |
| 2× selector charge | 1/3 |
| GBT inference charged per output, like the logistic | 0/3; −.010 to −.012 |
| select-tuned tree (min_leaf, bins) | 2/3; +.0014 / +.0030 / +.0023 |
| depth-3 tree fit in-sample on confirm | learned − tree = −.0002 [−.0012, +.0008] |

~5% of episodes carry the whole net margin.

**Strong baselines.**
- The design-v2 item-8 strong baseline is implemented as specified, with one disclosed deviation: its inspect rule compares θ·hidden_frac·L with the total observation cost.
- It lacks a commit-verified-cache shortcut, so the trees, not the strong baseline, are the binding simple baseline.

## Track B: clean compositional generalization (probeworld-v3)

**Structural screen (B-SCREEN, prospective, exact DP, registered 06:58Z before any model):**
- The generator was revised to v3: C only at k ≥ 2 with corr ∈ [.25, .45]; p_event ∈ [.3, .6].
- Among untouched triples, **S+C+E** had the largest minimum per-factor decision relevance, .66 [.55, .76], with global-shift closable fraction .26. It is the **primary** held-out family.
- **U+C+E** (.55; GS .27) is secondary.
- Composition content is mostly in later decisions (first-decision relevance of C and E is .07), so later-decision accuracy and gap regret are primary and first-decision endpoints are secondary.

**Exact splits:**
- **Training pools (base 6.2e9, common random numbers across arms):**
  - B0 = 384 configurations over the base combination set.
  - B1 = 768 (2N) over the base set.
  - B2 = 384, broader variety (includes U+C, U+E), built by replacement. It matches B0's per-factor frequency, per-combination share and first-action mix; 17/84 replacements are unmatched and disclosed. The single-type mix also differs: S 106 vs 64; C and E 43 vs 64.
  - B3 = 768, broader variety.
  - dose1–3: 1–3 of S+C+E's constituent pairs present.
  - Every arm uses 4,000 × 64 on-policy episodes, so updates and examples are equal.
  - **Neither S+C+E nor U+C+E nor the 4-factor superset appears in any training, dev or selection split**, checked on generator parameters by `audit_split_table` and re-verified by the audit.
- **Evaluation:**
  - `b6_hold_SCE`: 400 configurations, base 6.4e9, worlds 6.45e9.
  - `b6_hold_UCE`: 400.
  - Historical challenge sets `b6_hist_{UE, SC, UC}`: 200 each.
  - Test pools `b6_test_{base, pairs}`: 128 each.
  - Balanced counterfactual octets: 320 S+C+E octets (all 2³ ablations plus near-miss variants; CF base 6.6e9; `cf_SCE.json` sha256 prefix c365186f).
- **Fresh confirm split (B-FACT-C):**
  - `b6c_hold_SCE`: 400 configurations, base 6.42e9, worlds 6.47e9.
  - 320 fresh S+C+E octets at 6.65e9.
  - Disjoint from every prior pool and world block (`audit_b6c`).
- **Training worlds:** 8e9 + 1e8·seed (seeds 30–34 and 35–39 registered).
- **Recipe:** trainer L1, final checkpoint, no model selection.

**B-ARMS: matched data-volume arms** (prospective; 5 seed pairs for B2 vs B0). Setup:
- Held-out S+C+E: 400 configurations and 320 octets.
- Pooled over pairs with a joint configuration/octet bootstrap (`campaign06_bprimary.py`).
- Registered rule: P1 gap regret < 0 with upper bound < 0; P2 later-decision accuracy > 0 with lower bound > 0; P3 near-miss accuracy and false-change rate not worse by > .02.

| B2 − B0 | mean | 95% CI | per pair |
|---|---|---|---|
| gap regret (B0 32.2) | −1.65 | [−6.10, +3.08] | −0.1 / +32.4 / −25.3 / −9.5 / −5.7 |
| later-decision accuracy (B0 .795) | +.010 | [−.002, +.022] | +.000 / +.025 / +.029 / −.011 / +.009 |
| first-decision accuracy (secondary) | +.075 | [+.041, +.106] | |
| octet near-miss accuracy | −.085 | [−.155, −.005] | |
| one-factor false-change rate | +.006 | [−.001, +.014] | |
| octet balanced accuracy | −.075 | [−.125, −.029] | 5/5 pairs lower |

- **Verdict: NOT SUPPORTED** (P1, P2 and P3 all fail).
- Audit: exact reproduction; robust to alternative later-decision definitions; two-level (seed + configuration) bootstrap gap −1.6 [−17.5, +19.3].
- Seed variance is large (B0 gap per seed 20.2/29.9/40.9/41.2/29.0; B2 20.0/62.2/15.6/31.8/23.3).
- B2's first-decision gain:
  - is on non-flip units;
  - comes from building first more often (28% vs 10%; B1 shows the same rise with no added variety);
  - costs accuracy after a solved probe (.52 vs .81).

**Descriptive arms** (3 pairs vs B0 seeds 30–32; not registered tests):
- B3, dose2 and dose3 improve gap regret and later decisions but lose near-miss accuracy.
- Under the audit's seed-robust checks, only **LRN** passes all three conditions. dose1 and SUP fail near-miss against the 5-seed B0, B3 loses P1, and RAWF passes P1 and P3.
- The gap-regret improvements of the dose arms and the fuse-architecture arms are robust to the B0 reference. Near-miss is the seed-fragile endpoint.
- **U+C+E** (secondary holdout; B2/B3 train on U+C and U+E, a pair-exposure confound): B0 gap 34.9–55.2 vs B2 21.0–25.6 and B1 ~19.4. More exposure lowers U+C+E regret about equally with (B2) or without (B1) new combinations.
- UCE octets were built (160) but not evaluated; the reason is in decisions.md.

**B-FACT-C: factorized representations** (ADAPTIVE: registered after inspecting B-FACT development results; registration commit 228b2f41 predates all b6c data). Setup:
- Fresh seeds 35–39 (5 pairs).
- Fresh S+C+E pool (400) and octets (320).
- Two-level bootstrap (resample seed pairs, then configurations and octets).
- Registered rule: Q1 gap regret upper bound < 0 and ≥ 4/5 pairs negative; Q2 later-decision accuracy lower bound > 0 and ≥ 4/5 pairs positive; Q3 near-miss not worse by > .02 and false-change not higher by > .02.

| contrast | gap regret | later-decision accuracy | near-miss | false change | flip-set accuracy | verdict |
|---|---|---|---|---|---|---|
| **LRN − RAWF** (primary; RAWF gap 17.09, later .815) | **−2.65 [−5.87, −0.03]**, 5/5 | **+.018 [+.002, +.034]**, 4/5 | +.105 [+.017, +.212] | −.032 [−.056, −.009] | **−.048 [−.091, −.003]**, 5/5 lower | **SUPPORTED as executed; Q1 borderline (audit)** |
| SUP − RAWF | −4.71 [−12.06, +2.38] | +.025 [−.001, +.050] | −.108 | −.039 | +.109 [+.050, +.178] | not supported |
| LRN − B0 (B0 gap 21.18) | −6.73 [−12.18, −2.40], 5/5 | +.020 [+.008, +.035], 5/5 | +.102 | −.029 | −.070 | supported |
| RAWF − B0 | −4.09 [−10.04, +1.80] | +.002 | −.003 | +.003 | −.022 | not supported |
| SUP − B0 | −8.80 [−16.50, −0.06] | +.028 | −.112 | −.036 | +.087 | fails Q3 |

- **Arm definitions:**
  - RAWF: fuse architecture, raw public inputs.
  - LRN: the same model plus auxiliary heads predicting the four derived quantities (posterior over hidden type, expected remaining cost per strategy, P(probe resolves), amortized build value), with the policy consuming the predictions.
  - SUP: the exact derived quantities as inputs, a localization ceiling.
- **Independent audit** ([review/bfc-independent-audit.md](review/bfc-independent-audit.md)):
  - **Numbers:** every number reproduces exactly from raw rows. Protocol fidelity is confirmed: the registration commit predates the first training by 16 s; the pool, worlds and octets are disjoint from everything else; all 44 receipts exited 0.
  - **Q1 is borderline:** at n_boot 1,000 the gap upper bound depends on the bootstrap seed (−0.14 to +0.35; passes 8/20 seeds). At 20,000 draws it is +.02 to +.06 (0/2); P(bootstrap ≥ 0) = 2.7%. The across-pair t(4) CI is [−5.5, +0.2]; the sign test gives p = .031 (5/5). 5% of configurations carry more than the whole gap gain.
  - **Q2 and Q3 are robust:** Q2's lower bound is +.001 to +.003 in every variant.
  - **Where the effects sit:**
    - The Q2 gain is the first decision of later queries (+.079): redundant 'inspect' errors fall from 1.28 to 0.79 per episode. Build and probe rates are unchanged.
    - The flip-set loss sits in the query-2 decisions (−.22 after H, 5/5; −.05 after not-H). First-decision flips are +.03.
    - LRN changes its action less under one-factor interventions both when the optimum changes (−.046) and when it does not (−.032). Discrimination .364 vs RAWF .378, i.e. no better.
  - **Bundling:** the auxiliary loss also trains the shared trunk. The contrast bundles representation shaping with the policy reading the predictions.
  - **Head accuracy on S+C+E was not logged,** so "heads mispredict" cannot be told apart from "the policy ignores good predictions".
  - **Adaptive-selection disclosures:**
    - LRN − RAWF and the later-decision endpoint were chosen from ~9 descriptive contrasts × several endpoints on the development split (seeds 30–32).
    - The development effects were about twice the confirmed ones.
    - n_boot was not registered; the tool default of 1,000 was fixed before any b6c data.
    - The b6c bases and the seed-range entry for seeds 35–39 were committed after training started but before any b6c labels or evals.
- **Reading:**
  - The learned derived-quantity heads give a **small, robust later-decision improvement** (fewer redundant information-gathering actions at the start of later queries).
  - The regret gain is suggestive (5/5 pairs) but not established.
  - They do **not** compose the interacting factors: they are less responsive to single-factor changes overall and worse on full-interaction flips.
  - The supplied quantities do carry interaction-relevant information (SUP flip +.109, discrimination .469), but SUP over-changes near-misses.

## Track C and optional branches

- **Track C** (value arms C0–C3) was **not run**. Its trigger (design v2 item 13) required GA-1 headroom and ≥ 50% of learned-selector regret localized to value misestimates. GA-1 failed.
- The optional attention and microstep branches were not started. Neither had a clean prerequisite from A or B.
- Population evolution was deferred per the brief.

## Supplied, learned, generalized

- **Supplied:**
  - portworld, its seven methods and their contracts, budgets from traces, and the exact oracle (evaluation only);
  - probeworld-v3 exact labels (training targets), the split table, the octet construction;
  - the definitions of the four derived quantities (and their exact values for SUP);
  - the fuse architecture.
- **Learned:**
  - Track A: GBT and tree selectors mapping public instance features (and telemetry) to a method/budget arm.
  - Track B: the probeworld policy under action-set supervision.
  - LRN: auxiliary predictors of the derived quantities from raw public inputs.
- **Generalized:**
  - The learned public selector generalizes to fresh instances (+.035 over best single on 330M). So do the simple trees; the two are practically equal.
  - LRN's later-decision gain over RAWF held on a fresh S+C+E pool with fresh initializations. Its regret gain replicated in sign (5/5) but not with a robust CI.
  - No arm generalized **composition** of S, C and E: flip-set accuracy did not improve for any learned arm.

## Unresolved

1. **Is there public-information headroom beyond trees in any natural strategy-selection family?** Portworld's remaining oracle gap is mostly hidden-state information. A task whose best strategy depends on *inferable but non-obvious* structure (e.g. long-range constraint interactions only revealed by computation) is still needed.
2. **Why do learned derived-quantity heads not help flips while supplied quantities do?** Candidates: head prediction error concentrated on S+C+E, or the policy under-weighting its predictions. Held-out head accuracy was not logged, so a head-accuracy × flip analysis is the first next step. A variant that separates auxiliary representation shaping from reading the predictions (e.g. stop-gradient heads) would also help.
6. **Regret effects near the resolution limit.** Future registrations must fix n_boot (≥ 10,000) and prefer across-seed tests when there are few seed pairs.
3. **Composition via exposure** (the extended-05 B-XC effect) did not reappear at matched volume on S+C+E. Whether it was a data-volume effect (the ext-05 confound) is now more likely but untested at the pair level.
4. **U+C+E octets** were built but not evaluated.
5. **Seed variance in probeworld training is large** (gap regret SD 9–19 across seeds). Future Track B tests need ≥ 5 seeds and two-level CIs by default.

## Costs, integrity and machine state

**Compute** (ledger from job receipts; [budget.json](budget.json)):
- **CPU:** 79.1k / 172.8k core-s metered (dev 8.8k).
  - Plus ~11.0k core-s local or unmetered, disclosed: the design reviewer's local exact-DP screen (~7.0k); a portworld-builder analysis run killed without a receipt (~3.0k); builders ~0.3k and audits ~0.7k local.
  - About 84k core-s was left unspent, above the 20% reserve.
- **GPU:** the ledger's "GPU occupancy" (17.4k s) is a conservative wall-clock union of all main jobs. **Actual GPU use was zero**: every training and evaluation ran with `CUDA_VISIBLE_DEVICES` empty, and the scoring jobs are pure Python.
- **Unit costs:**

  | job type | core-s |
  |---|---|
  | Track B training (L1) | ~400–550 |
  | b6 eval | ~150 |
  | b6c eval | ~40 |
  | SCE octet labels (320) | 3,020 |
  | UCE octet labels (160) | 4,630 |
  | portworld evaluate | ~.44 per episode |

**Incidents** (all in decisions.md; each failed job is receipted and charged):
1. `e06-tb-labels-cf` hit its 8,000 core-s cap during the secondary UCE family. The primary SCE file was complete; UCE was relaunched at 160 octets.
2. **OOM:** I launched 33 evals concurrently against the plan's 4–8. One (`dose2-eval-s30`) was OOM-killed; it was relabelled and rerun alone. No non-campaign process was affected. A ≤ 8 rule is now in place.
3. `e06-tb-score` failed at its final write because its output directory was missing (1,068 core-s charged). It was rerun with identical inputs.
4. A portworld builder subagent killed an analysis wrapper, losing a receipt (~3.0k core-s, disclosed).
5. A seed-ranges merge conflict (from extended-05) recurred once and was resolved by a named union. The launch guard refuses conflict markers.
6. A mis-named commit (e8537d2d) contains only ledger updates.
7. A misplaced `cd` in one read-only remote command listed the Windows home directory. Nothing was modified.

**Prospective vs adaptive:**
- Prospective: A-HS, B-SCREEN, B-ARMS.
- Adaptive (labelled): A-CF-SMALL (after the gate), B-FACT-C (after B-FACT development results; chosen from ~9 descriptive contrasts).
- Descriptive: B-FACT (3 seeds), the other B-ARMS arms, U+C+E.

**Machine state:** see the final section of [handoff.md](handoff.md).
