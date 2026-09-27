# Extended-05 design review (internal, pre-registration)

**Reviewer:** internal design-review subagent. It reviewed; it did not implement.
**Scope:** design.md v1, registry.json, jobplan.json and `research/tools/campaign05_jobplan.py`, checked against the brief's guardrails (campaign.md). Also read: the extended-04 report, audits, probeworld.md and trackc.md; `campaign03_depworld.action_catalog`, `campaign04_deploy`, `campaign04_meta` (branching), `campaign04_probeworld`; and the extended-04 receipts.

**Checks run** (non-protocol, dev worlds only; nothing registered touched):
- **Remote, metered, ≈ 65 core-s:** `review-oi-probe` 54.1, `review-oi-smoke` 7.6 plus a 2.5 failed start, `review-maxsteps` 1.2. The probe ran D (p1-boot-x1-r0, r_mask) on 80 dev worlds (2.17e9 + 350,000 range, iid_f0/f2), enumerated O(I) at every decision, and branched every O(I) option at 48 sampled states.
- **Local, pure Python, ≈ 182 core-s (estimated):** `ExactSolver` at s0 for 128 generator configurations per pair (train cells, k ∈ {1, 2, 8}).
- **Probe code:** scratch only, in the remote workdir `~/structured-latent-dynamics-campaign05/review-e05-design`.

**Severity:** BLOCKER = fix before registering or launching the stage; MAJOR = fix before the protocol is registered; MINOR = when convenient.

**Overall.**
- **Keep:** the D contract; the naming of the hidden-state bound as non-deployable; no REINFORCE through argmax; the teacher-supervision comparator; config-clustered statistics; the reclassification of the historical pairs.
- **The two headline gates are not well-posed:** G1 passes by construction from hindsight stopping (F1, F2), and the B-X primary has almost no support on U+C (F11).
- **BX2 is predicted null** for the measured failure (F14).
- **The job plan** underprices evaluation with a learned head, the extended-04 overrun mechanism, and omits several lines (F17).

## Dev-probe facts used below
p1-boot-x1-r0, 80 dev worlds, 2,683 decisions, success 74/80.
- **What D chooses.** An O(I)-type action at 38% of decisions: use_return 17%, call 11%, start_*/build_route 8%, uncommit 1.6%, abstain 0%. 89% of D's calls use budget 128 (1024 is used 25 times, 16 eight times).
- **O(I) size.** Mean **17.1** options, tail beyond 30; the plan assumes 12. Calls exist at **82%** of decisions, because problems persist once opened, so O(I) is dominated by "insert a call now" deviations rather than by the budget choice at D's own calls.
- **Branch cost** (wall-timed, serial, including the solver child): 0.073–0.087 core-s for call/uncommit/recompute/use_return/default continuations; ≈ 0.001 for abstain. D rollout: 0.121 core-s per episode.
- **One-step hindsight headroom** (max over O(I) of Q_true^D minus Q_true^D(D), 48 states): 42 states are 0; the other six are .044, .116, .145, .256, .390, .407.
  - **Abstain drives it:** it beats D by more than .01 at 10.4% of states, max .41.
  - **Excluding abstain,** the maxima are uncommit .106 (5.2% of options > .01), recompute .088 (1.4%), call .050 (3.3%), use_return 0.
- **Probeworld support at s0** (probe uniquely optimal, per 128 configurations):

  | pair | U+C | S+E | U+E (historical) | S+C (historical) |
  |---|---:|---:|---:|---:|
  | configs | **5** | 23 | 25 | 25 |

  The historical figure matches extended-04's n = 88, which is 22 configurations × 4 worlds. **Correlated flag at k = 1:** it changes V* in only 5/38 U+C and 4/40 S+C configurations (clipping cases).

## Track A

**F1 [BLOCKER]: G1 gates on the hindsight bound H, which passes by construction.**
- **Problem.** Q_true^D is deterministic given the world (cloned environment RNG, cached solver), so E[max_a Q_true] ≥ max_a E[Q | I] (Jensen).
- **Evidence.** H is carried by **hindsight abstention in episodes D goes on to fail**: per-state mean ≈ .028, and 5 of 6 positive states are abstain-driven. At episode level that is roughly P(fail) × gain(stop) ≈ .06 × .4 ≈ .025 ≥ .02, with no deployable headroom behind it.
- **Precedent.** Extended-04 showed public appraisal cannot find these episodes: stop was predicted harmful, and r2's gain came from early stopping.
- **Fix.**
  - Keep H only as a **necessary** condition (G1a: fail means stop). Report it per option class, **with and without abstain**.
  - Add **G1b** as the go/no-go: the same-information estimate (F2) of a deployable single-deviation policy ≥ .01 utility per episode (lower 90% CI > 0). It uses cross-fitted Q̂^D fitted and evaluated on disjoint *worlds*, with abstain gains reported separately.
  - Register **before A-HR** whether stop-driven gain counts. The recommendation: it counts, but is reported by type and never pooled.

**F2 [BLOCKER]: Summing per-state headroom along trajectories is invalid, and the planned oracle run is underpowered.**
- **Why the sum is invalid.** By the performance-difference identity, J(π′) − J(D) = E over **π′'s** states of A^D(s, π′(s)). Advantages summed over *D's* states are the gain of no policy.
- **Why the oracle run cannot resolve .02.** Two lineages × 32 worlds is n = 64; with per-episode utility SD ≈ .25, SE ≈ .03. Powering it (n ≈ 1,000) costs about 30 steps × 17 options × 0.1 ≈ 50 core-s per episode, ≈ 50k core-s.
- **Fix: a single-deviation policy.** Follow D. At decision points chosen by a registered public Bernoulli sampler (Track C style, inclusion probabilities recorded), deviate *at most once* to the rule's option if its predicted gain > m. Then follow D.
  - Its episode gain is exactly the branched advantage at the first firing point, and zero otherwise.
  - So G1a (oracle rule) and G1b (cross-fitted rule) are both episode-level, cost nothing beyond the planned branches, and have world-clustered CIs.
- **Plan changes.** Drop "summed along trajectories" from the design and the registry. Keep the online multi-step oracle only as a descriptive subsample, or remove it (−4.75k core-s).

**F3 [MAJOR]: O(I) is well-defined but mis-aimed at the named decisions.**
- **Problem.** At almost every state O(I) is "D, abstain, and about 15 call-now options". The brief's decisions are tied to D's own action kind: budget at D's call (11% of decisions), reuse vs recompute at D's use_return/start (about 25%), and revision scope at D's uncommit (1.6%).
- **Fix: register a decision-anchored O(I).**
  - At D-call states: the same problem at every other allowed budget, plus "not calling" (D's second-best non-call).
  - At use_return/start/build states: the matching reuse or recompute alternatives.
  - At uncommit: the other scope.
  - Abstain: its own class, everywhere.
- **Sampling.** Stratify states by D's action kind, with inclusion probabilities. Keep the full-catalog subsample to measure what is omitted.

**F4 [MAJOR]: The ranking labels can become hindsight-best labels.**
- **Problem.** Branch returns are single-world realizations, and "K repeats where stochastic" is moot because branches are deterministic per world. A pairwise loss whose *sign* comes from the realized ΔQ is a hindsight class target unless it is cost-linear.
- **Fix, one of:**
  - make the primary a Q̂^D regression head (extended-04 F8: squared error on realized ΔU, argmax at deployment); or
  - register weighted-all-pairs, loss = Σ ΔQ⁺·h(s_a − s_b) + ΔQ⁻·h(s_b − s_a). It is linear in realized costs, so its minimizer has sign(E[ΔQ | I]).
- **Near-ties** on realized ΔQ are then only a reporting convention, and realized-ΔQ ranking accuracy has a noise ceiling below 1.
- **Fit m and the fallback threshold** on world-level cross-fit folds.

**F5 [MAJOR]: The iteration-2 continuation is unnamed.**
- **Problem.** RSPI iteration 2 needs Q^{π₁} targets. The design keeps D continuations and only adds π₁-visited states, which is coverage expansion, not a second policy-iteration step.
- **Fix.** Name one choice in the registry and budget its labels (neither is in the job plan):
  - (a) Q^D targets on π₁ states; or
  - (b) Q^{π₁} targets, with the continuation renamed in every label.

**F6 [MAJOR]: The teacher comparator and "matched label cost" are underspecified.**
- **Comparator.** (ii) should use the same head, features and states. Its target is dep_reuse's action where that action is in O(I), else D's action (cross-entropy).
- **Matching.** Teacher labels are almost free, so the primary is matched **states**; a matched-**CPU** variant (more teacher states) is secondary.
- **Seeds.** One seed per arm cannot resolve "beats (ii) by .01". Use 3 head seeds per arm (the heads are tiny) and paired world-clustered CIs.
- **Intervention type.** Report gains by intervention type. An all-stop π₁ is a stopping rule and must be labelled as such.

**F7 [MINOR]: D's reproducibility has operational gaps.**
- **Stale paths.** `configs/campaign03/p1-sealed-checkpoints.json` and the campaign04 configs still point to the non-existent `/home/brand/tensegra-campaign03/`. Add a path-remap layer; do not edit hashed configs.
- **Device.** Register it (Track C used CPU, 1 thread, fp32). The job plan says "GPU actor", and a CUDA argmax on near-ties can differ from CPU.
- **Pre-flight.** Show bit-identity to `deploy_episodes(mode="r_mask")` on 32 dev worlds per lineage r0–r5 before A-HR.
- **max_steps** is verified at 96.

**F8 [MINOR]: Lineage roles.** A-HR samples r3–r5, which A-CF later "confirms".
- Keep A-HR on development world ranges only, disjoint from the sealed A-CF ranges.
- Name the single A-PI screen lineage.
- Disclose in the report that A-CF lineages were seen at the headroom stage.

**F9 [MINOR]: Decision-level endpoints need evaluation branching.** Ranking accuracy, chosen-option regret and false-positive/missed interventions all need it, and none is in the job plan. Add e.g. 128 worlds × 2 points × 17 options per policy.

## Track B

**F10 [MAJOR]: The C-containing holds are partly inert at k = 1.**
- **Problem.** corr acts only across queries; at k = 1 V* is unchanged in about 87–90% of configurations. A third of U+C (and of S+C) is effectively a single-flag task.
- **Fix.** Draw C-containing holds from k ∈ {2, 8}, or report the k = 1 stratum separately, outside the primary.

**F11 [BLOCKER]: The B-X primary has almost no support on U+C.**
- **The endpoint is config-level.** `first_probe_rate_when_unique_opt` is taken at the first decision, which is a function of the configuration. A deterministic greedy model gives one binary per configuration, so 4 worlds add nothing (extended-04's 48/88 is 12/22 configurations).
- **Consequence.** At 64 configurations per pair, U+C yields about 2–3 eligible configurations and S+E about 11. A +.15 lift on U+C is unmeasurable.
- **Fix.**
  - **(a)** Size each hold pool from π* s0 support before any model run. Register ≥ 60 eligible configurations per pair: about 1,500 U+C and about 350 S+E configurations, 1 world each for the s0 endpoint. DP is ≈ 0.3 core-s per configuration locally.
  - **(b)** Add a per-query-first-decision variant (more support when k > 1).
  - **(c)** Use configuration-level CIs, and mark a hold "insufficient" if it has < 20 eligible configurations, declared before any model run. Never swap pairs after results.

**F12 [MAJOR]: The split change touches every split.**
- **Leak risk.** `SPLITS` are hard-coded. The current `TRAIN_COMBOS`, which dev, test_iid, heldout_price and heldout_k also use, include U+C and S+E. Dev is used for selection, so the new holds would leak into it.
- **Silent redraw.** `generator_params` uses `rng.choice(combos)`, so a changed tuple silently changes which configuration each seed produces.
- **Fix.**
  - Add a versioned split table (probeworld-split-v2) with **new config seed bases** for every split and for the holds (not 4.6e9).
  - Keep the holds on train cells and train k.
  - Update `split_of_params` so the audit proves U+C/S+E appear in no training or dev configuration.
- **Wording.** C+E is supported (all 16 combinations are generated), so drop "where the generator supports it". B0's training set is {none, U, S, C, E, U+S, C+E}; BX1 adds U+E and S+C.

**F13 [MAJOR]: No headroom gate before B-X.**
- **Problem.** If B0 already probes at ≥ .80 on a hold, the +.15 criterion is unreachable, and "composition transfers here" is the result.
- **Fix.**
  - Register per hold: the primary is evaluable only if B0's rate is ≤ .70 in at least 2/3 seeds.
  - Replace "regret ≤ B0" with a registered non-inferiority tolerance (configuration-level CI).

**F14 [MAJOR]: BX2 is predicted null for the measured failure.**
- **Why.** At s0 the exact belief equals the declared prior. `public_vector()` already holds the prior, q, D_side, corr, p_event and the flags. So where the historical failure lives, BX2 only duplicates existing inputs.
- **Not trivial, but not a fair factorization test.** BX2 is not a lookup (the value of computation still needs lookahead over priced actions).
- **Fix.**
  - Keep BX2 as the **supplied-belief control**, labelled supplied and predicted null at s0.
  - Add a **factorized-architecture arm:** per-flag modular encoders over the *same* public inputs, summed into the trunk, with parameters matched to B0 and disclosed.
  - Report every arm by decision depth (s0 vs later).

**F15 [MAJOR]: BO and B-LOC do not line up, and BO is not trained in the plan.**
- **Problems.**
  - B-LOC diagnoses extended-04 models trained on the *old* split. A belief model trained on the new split cannot localize their failure.
  - The job plan evaluates 12 models but trains only 9.
- **Fix.**
  - BO = the L1 recipe plus a belief input, trained on the **extended-04 split** (3 seeds, ≈ 2.4k core-s) and evaluated on the historical pairs.
  - Expect BO ≈ L1 at s0 (F14). That would localize the failure to value or ranking, not belief.
- **Class definitions.** Class (4), deployment, is vacuous: there are no masks and greedy ties have measure zero. Q-head classes (1)/(2) apply to L4 only, since L1's Q head is untrained.

**F16 [MINOR]: BX1 is undefined.** Define BX1's pool size. Extended-04 used 384 configurations at ≈ 2.5 GB RSS; 768 would be about 5 GB per process. Budget its labels.

## Budget and job plan

**F17 [MAJOR]: Unit costs against the receipts.**

| unit | plan | measured | factor |
|---|---:|---:|---|
| D episode (competent) | 0.10 | A1 0.13 (536/4,096); F1b 0.085; A3s 0.079; probe 0.12 | OK |
| branch | 0.165 | Track C 0.136–0.168 *incl.* base episodes (e.g. 911/5,800); probe 0.08 | conservative |
| options per state | 12 | 17.1 (tail > 30) | ×1.4 |
| episode with a learned head | 0.13 | **Track C evaluation 0.35** (41.4k / 118,272) | **×2.7** |
| A-PI screen policies | 5 | D, teacher-sup, 3 heuristics, dep_reuse, π₁ (+ iteration 2) = 7–8 | ×1.6 |

- **Missing lines:** F2 cross-fit (≈ 0); multi-step option test (≈ 3k); evaluation branching (≈ 2k per screen); BO (≈ 2.4k); Track B pools (≈ 1.5k); iteration-2 labels. State that the structural branch is excluded.
- **Re-estimate with contingency** (A-HR at 17 × 0.10):

  | item | plan | re-estimate |
  |---|---:|---:|
  | A-HR | 10.1k | 8.7k |
  | oracle rollout | 4.75k | dropped |
  | A-PI screen eval | 0.83k | 5.7k |
  | A-CF evaluation | 3.0k | 13.4k |
  | A-CF labels | 22.8k | 19.6k |
  | other changes | – | ≈ +7k |
  | **total** | ≈ 88k | **≈ 100k** |

  This fits the 138k usable, but evaluation is again the item most likely to overrun.
- **Fix.** Regenerate `campaign05_jobplan.py` with per-policy × per-condition dims, `eval_learned = 0.35` and options = 17.

## Also missing

**F18 [MINOR]:**
- **Multi-step fallback.** Needs a concrete spec, a threshold and a cost line **registered before any A-HR output**; otherwise it is adaptive.
- **Ledger.** Add the Track A supplied items: the O(I) enumerator and catalog, the R-mask, the branch labels (privileged, offline) and dep_reuse.
- **§5 trigger.** Make it checkable, e.g. the head's inputs include the record/problem dependency graph and ablating it costs ≥ .005 utility.
- **Registry.** A-HR drops "per episode … summed" (F2) and adds G1b (F1).
