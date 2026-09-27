# Extended-06 design review (internal, pre-registration)

**Reviewer:** internal design-review subagent (read-only; nothing implemented, nothing committed).
**Scope:** campaign.md, design.md v1, decisions.md, seed-ranges.json, budget.json; the ext-05 report and reviews; ext-04 probeworld.md; `campaign04_probeworld.py`, `campaign05_probeworld.py`, `campaign04_probeworld_train.py`.
**Severity:** BLOCKER = fix before registering/launching the stage; MAJOR = before the protocol is frozen; MINOR = when convenient.

**Checks run (disclosed).**
- **Local, pure Python, ≈ 7.0k core-s** (user time via `/usr/bin/time`; 20-core workstation).
  - An exact-DP screen of every probeworld pair, triple and the quadruple, using the unmodified ext-05 code (`ExactSolver`, `ablate`, `evaluate_foreign_policy`).
  - Seeds 2,300,000,000+i and 2,300,500,000+i (ext06 dev range; log as consumed); train cells only; no registered pool touched; scratch only.
- **Remote, ≈ 2 core-s, not metered** (my omission; import checks only). The pro6000 env has numpy and torch but **no scipy, HiGHS or OR-tools**; 24 cores; 60 GB RAM free.

**Screen definitions.**
- **Policy-level relevance of factor j:** the exact expected regret, in the full configuration, of π* of the same configuration with j off (it tracks its own belief from the shared history) is > ε = .5.
- **s0 relevance:** A*(I0) changes when j is off.
- **Eligible:** acting as if no factor were active costs > ε.

## Headline
1. **Track B BLOCKER (F8): at current settings no untouched family meets ≥ 30% per factor.**
   - All six pairs are touched: U+C and S+E were ext-05 holds, U+E and S+C are historical, U+S and C+E are in b5 training.
   - Every triple fails on C (10–13%); U+S+E also fails on E (22%).
   - **A principled range revision passes:** C only at k ≥ 2 with corr ∈ [.25, .45], and p_event ∈ [.3, .6]. Per factor, of eligible: S+C+E .86 / .73 / .67 (jointly .49); U+C+E .97 / .62 / .85 (jointly .61); U+S+E .55 / .58 / .60.
2. **Track B MAJOR (F11): PUBLIC-RAW vs SUPPLIED-FACTORIZED is vacuous.** Flags and q, D_side, corr and p_event are already explicit public inputs.
3. **Track A MAJOR (F1–F3): as specified, one cascade is likely to dominate.**
   - Decomposition and propagation are preprocessors, not alternatives.
   - The likely winner: decompose → greedy-warm-started B&B to a price-scaled budget or certified gap.
   - A1 must contain telemetry-threshold rules, or HEADROOM measures the value of telemetry, not of learning.
4. **Gate GA is not well-posed (F5).** "Relative to SD" is ambiguous; "no single method within .01 of the simple portfolio" points the wrong way.
5. **Missing artifacts (F17).** registry.json and jobplan.json are linked but absent, and the seed sub-ranges are unassigned.

## Track A: portworld
**F1 [MAJOR] Two of the eight methods are not alternatives.**
- Propagation and decomposition are exact, O(n+m), and help every solver. "Always do them" dominates.
- **Fix.** Make them charged, fixed preprocessing. The real menu is {greedy, repair(b), beam(w, b), B&B(b or gap target), reuse+verify(+repair), inspect(which)} × stopping.
- Keep contract rows (input, output, guarantee, work meter, resumable) in code. Warm-started B&B is one option.

**F2 [MAJOR] Dominance risk.** At n ≤ 50 with 2–3 dimensions, three depth-2 public-feature rules are likely strong:
- "decompose; B&B per component to a budget ∝ 1/price; greedy+repair above a size cap";
- "always verify the cache (O(n)), repair, warm-start B&B";
- "inspect iff hidden fraction × failure loss > observation price".

Headroom then needs ≥ 3-way nonlinear driver interactions, or information revealed by computation.

- **Fix.** Put all of the following in A1: these cascades, price-scaled budgets, a **certified-gap stopping rule** (stop when UB − incumbent < price × expected remaining work), and a node-count restart/switch rule.
- The gap rule is the classical anytime-monitoring rule and A3's natural competitor. Without these, A3 can "beat simple rules" by rediscovering them.

**F3 [MAJOR] Headroom must be principled, not manufactured.**
- **Admissible sources:**
  1. **heavy-tailed B&B effort at fixed static features.** It is strongest on the strongly and almost-strongly correlated value–weight classes (Pisinger), so early telemetry such as the gap after a small budget and the node growth rate is genuinely informative;
  2. **stale-but-feasible reuse**, where optimality is not cheaply checkable;
  3. **hidden capacities or conflicts** with an explicit posterior;
  4. **prices** spanning each method's cost crossover.
- **Not admissible:** method-specific random failures, or hidden switches that no computation reveals.
- **Generator:**
  - draw the drivers independently (Latin hypercube over n, dimensions, tightness, conflict density, correlation class, component sizes, hidden fraction, perturbation size and log-prices);
  - centre the compute price where B&B's median quality gain over greedy equals its charge, and sample ±1 decade around that. This is a stated rule, not tuning;
  - freeze the generator before the headroom run.
- **Report, never tune:** the hindsight-best share per method. Drop a method that is best < 5% of the time.
- **The one allowed GA-fail revision** (driver, range, reason) is logged before it is run, and runs on fresh dev instances.

**F4 [MAJOR] A fair public-information bounded-portfolio estimate (BPE).**
- **Cross-fitting:**
  - K = 5 folds, clustered by episode;
  - A1 thresholds, trees (depth ≤ 3), logistic regularization and hand-portfolio constants are tuned by inner CV on the training folds;
  - the "best simple portfolio" is **selected on inner folds** and scored out of fold. Taking the max of out-of-fold scores over ~50 rules is a winner's curse that shrinks HEADROOM.
- **BPE:** GBM or MLP with nested-CV hyperparameters, on the same menu, features and telemetry as A1's telemetry rules, and charged for them (F7).
- **Gates:** one-shot (GA2, for A2) and sequential (GA3, for A3) are gated separately; the max of the two is optimistic.
- **Phase 2 independence:** GA partly pre-answers Phase 2, so Phase 2 must use fresh instances and fresh lineages.
- **Oracle:** the hidden-state oracle is labelled non-deployable (ext-05 F1) and never gates anything.

**F6 [MAJOR] The exact optimum sets instance size.**
- **Oracle.** With no LP/MIP solver available, use a hand-written B&B with a **surrogate/Lagrangian bound**: aggregate the rows with multipliers, then take the fractional-knapsack bound on the conflict-relaxed problem. It is valid, O(n log n) per node, and needs no simplex.
- **Cross-checks:** brute force for n ≤ 20; a 1-D DP without conflicts; a second bound implementation, since the oracle shares code with the B&B method.
- **Pilot:** 200 instances per n ∈ {20, 30, 40, 50} × correlation class. Pick n_max so that p99 certification is ≤ ~5 core-s.
- **Never drop uncertified instances by outcome.** That removes the heavy tail that creates headroom. Cap n by a public rule, or score cap hits against the best certified upper bound (counted and reported).
- **Traces.** Run each anytime method once to its maximum budget and log (work, incumbent, bound). Every budget, rule and hindsight oracle is then read off the traces: ~10–20× cheaper than a method × budget matrix.

**F7 [MAJOR] Compute accounting.**
- **Work units** are deterministic counters (nodes, bound evaluations, moves, scans), not wall time. Fit core-s ≈ a·work per method and report R².
- **Charge the selector** in the same units, per instance: features (component scan, surrogate bound, greedy probe), telemetry, and inference (multiply-adds × a declared rate).
- **Training cost** is reported separately, amortized per 1,000 instances.
- **"At comparable or lower total compute"** double-counts once U subtracts compute. Keep U primary, and add a secondary non-inferiority check: work ≤ 1.1× A1.

## Gate GA
**F5 [BLOCKER for registering GA].**
- ".02 of utility scale (relative to the between-instance SD)" reads as either .02 or .02·SD, and the SD is dominated by price heterogeneity.
- "No single method within .01 of the best simple portfolio" *fails* when the simple rules ≈ one method but a flexible public-information selector beats both. That is the case Track A needs.
- **Proposed GA**, per BPE variant, cross-fitted and episode-clustered, with U in absolute units (quality ∈ [0, 1] − charges):
  - **(i)** H_s = U(BPE) − U(best simple, inner-selected) ≥ .02, with a 95% lower bound > .005;
  - **(ii)** U(BPE) − U(best single) ≥ .02;
  - **(iii)** H_s ≥ 25% of U(oracle) − U(best simple), gated only if the oracle is tight.
- Report the single-vs-simple spread; do not gate on it. Fix the Phase 2 margin ≤ H_s/2 from the dev power analysis.

## Track B: probeworld v3
**F8 [BLOCKER] Factor-relevance screen.** Policy-level, as a fraction of all configurations (s0 in brackets); n = 200 per family; base generator (train cells, k ∈ {1, 2, 8}).

| family | status | U | S | C | E | jointly |
|---|---|---|---|---|---|---|
| U+C | ext-05 hold | .86 (.76) | – | **.10** (.03) | – | .10 |
| S+E | ext-05 hold | – | .60 (.65) | – | **.21** (.27) | .12 |
| U+E | historical | .88 (.62) | – | – | .33 (.51) | .30 |
| S+C | historical | – | .66 | **.14** | – | .14 |
| U+S / C+E | b5 training | .35 / – | **.11** / – | – / **.06** | – / **.18** | .10 / .04 |
| U+S+C | untouched | .48 | **.18** | **.12** | – | .07 |
| U+S+E | untouched | .46 | .32 | – | **.22** | .08 |
| U+C+E | untouched | .91 | – | **.13** | .34 | .11 |
| S+C+E | untouched | – | .69 | **.13** | **.18** | .05 |
| U+S+C+E | untouched | .56 | .38 | **.11** | **.27** | .08 |

- **Why C and E fail.** C is inert at k = 1 by construction, and corr ≤ .25 rarely moves a decision. E only forces reruns, so its mean regret is .3–1.3.
- **The v3 "wide" variant** (n = 150; relevance among eligible). Its rule is that *each factor's parameter spans the regime where it can change a decision*: C only at k ≥ 2, corr ∈ [.25, .45], p_event ∈ [.3, .6], U and S unchanged.

  | family | eligible | per factor (s0) | jointly |
  |---|---|---|---|
  | **U+C+E** | 135/150 | U .97 (.58), C .62 (.11), E .85 (.59) | .61 |
  | **S+C+E** | 133/150 | S .86 (.62), C .73 (.17), E .67 (.43); at regret > 2: .77 / .53 / .41 | .49 |
  | U+S+E | 134/150 | U .55, S .58, E .60 | .24 |
  | U+S+C | – | fails: S .29 | – |
  | k ≥ 2 only (no widening) | – | fails: S+C+E E .27; U+C+E C .26 | – |

- **Fix.**
  - Version the generator as `probeworld-v3`, with the wide ranges in **every** split, singles and pairs included, so that the held triple is not also off-manifold in parameter values.
  - Pre-declare the selection rule, e.g. the largest *minimum* per-factor relevance subject to GSS ≤ .5 (F9). On these numbers it selects **S+C+E** as primary (min .67) and U+C+E as secondary. Register before any model sees v3.
  - Report relevance at ε and at regret > 2, and s0 separately. C acts mostly after s0, so the endpoint cannot be first-probe only (F12).
  - No new factor is needed. The Track B builder is already screening new factors D and T (seen in its running job). Prefer the v3 range revision unless D or T clears the bar by a clear margin *and* stays within the build and test time.

**F9 [MAJOR] Make "global-shift solvability" operational.**
- **π_base** = π* of the nearest training-support configuration: for a triple, the best of its three pair ablations (conservative); for singles-only arms, the single ablations.
- **π_β** = argmax_a [Q_base(I, a) + β_a], with one β ∈ R^10 for the whole family, fitted by coordinate search on the family-mean exact utility (`evaluate_foreign_policy` with the biased argmax).
- **GSS** = [U(π_β*) − U(π_base)] / [V* − U(π_base)]. Register **GSS ≤ .5**.
- **Model-level companion:** fit β on B0's logits using the hold, an oracle-tuned bound on what a shift can do. B2's gain counts as compositional only above B0+β*.

**F10 [MAJOR] Arm matching.**
- **Equal updates and equal examples are automatic.** `cmd_train` draws 4,000 × 64 on-policy episodes on fresh worlds whatever the pool size. "Volume" therefore means distinct configurations, and B1 − B0 isolates configuration count at matched presentations (not epochs). State this.
- **The real B2 − B0 confounds:** per-factor marginal frequency, combination shares, and the decision-type mix (the B-XC caveat).
- **Fix.** Build B2 by replacement, with per-factor marginals and the s0 ε-optimal-action histogram matched to B0 (stratified draws), so that only co-occurrence differs.
- **Dose** = the number of the held triple's constituent pairs in training: 0 (B0), 1, 2, and 3 (= B2, plus the non-held triples).

**F11 [MAJOR] The factorized arms need a raw input that hides the factors.**
- `public_vector` holds the flag bits and the factor parameters, so LEARNED-FACTORIZED is trivial.
- **Option (a).**
  - PUBLIC-RAW = a frozen random 64-d projection of the public vector, then tanh (same information, fixed per campaign).
  - SUPPLIED = the current coordinates.
  - LEARNED = the mixed input plus an auxiliary factor head that feeds the trunk.
- **Option (b).** Drop the arms. Ext-05's BX2 was already a supplied-factor arm and did not fix s0.

**F12 [MAJOR] Counterfactual sets for a triple are 2³ octets.**
- From each base draw, `ablate` every subset to get 8 on-manifold variants (same prices, k and prior). Add near-misses: a triple whose π* equals that of its best pair ablation.
- **Endpoints**, over all visited decisions: change-accuracy P(action changes | optimal changes) and stability P(unchanged | optimal unchanged). A global shift cannot raise both.
- **Worlds.** The greedy s0 action is deterministic per configuration, so s0 needs 1 world; later decisions and gap regret need ≥ 4.

**F13 [MINOR] Memory.**
- DP states at k = 8: none ~4k; U+C and U+E ~30k; **C+E-containing triples ~150k** (6.4 core-s and ~150–250 MB per configuration).
- Shard the holds, and measure pool RSS in the smoke; ext-05's ~2.5 GB per pool will not hold.

## Statistics and confirmation
**F14 [MAJOR] Units and power.**
- **Track B:**
  - the unit is the configuration, nested in seed;
  - the B-XC paired s0 SD ≈ .4 implies ≈ 125 eligible configurations per seed pair for +.10 (80% power, α .05). So use **≥ 200 eligible** (not 60) and **≥ 150 octets**;
  - use **5 seed pairs** for B2 vs B0 (~500 core-s each; with 3 seeds, "3/3" is the only seed-level inference);
  - the claim needs ≥ 4/5 same sign plus a pooled config-bootstrap CI excluding 0, with per-seed values reported.
- **Track A:**
  - the cluster is the episode, with generator seed and lineage above it;
  - n = (2.8·σ_d/margin)². At σ_d .05–.10 and a .01 margin that is 200–800, so plan 1,000 fresh confirmation episodes;
  - a lineage = independent training draw + seed: 3 lineages, paired on the same episodes.
- **Adaptive labels:** the GA revision, and any v3 range change beyond F8.

## Budget and wall time (12 h, 24 cores, 48 core-h)
**F15 [MAJOR] CPU is fine; the risk is serial wall time. Price Track A from its pilot.**
- **Track B ≈ 26k core-s (7 core-h);** ~10 trainings run concurrently (RAM-bound), so ~3 waves of 30–40 min:

  | item | core-s |
  |---|---|
  | labels | ~4k |
  | 28 trainings × ~500 (B0/B2 at 5 seeds, B1/B3 at 3, dose 2 × 3, factorized 2 × 3) | ~14k |
  | evaluations | ~2k |
  | fresh B0/B2 confirmation × 5 | ~6k |
- **Track A ≈ 30–35k core-s (9 core-h) with traces** (×10 without):
  - traces ≈ 14k (8k instances × 6 methods × .3);
  - exact optima 5–10k;
  - Phase 2 and confirmation ~10k.
- **Checkpoints to register:**
  - the B family registered by T+2 h;
  - GA decided by T+5 h, or Track A closes as "not established";
  - confirmations launched by T+8 h;
  - the audit from T+10 h.
- The GPU is unneeded; record it as unused.

## Missing vs the brief
- **F16 [MAJOR] Track C trigger test.** Track C is triggered only if, on close alternatives (|ΔQ*| ∈ [ε, 5ε]), the winning arm's Q-ranking is ≤ .8 **and** a supplied-exact-Q ablation closes ≥ 50% of the regret. Otherwise it is "not triggered".
- **F17 [MAJOR] Artifacts.**
  - Create registry.json (gates, primaries, margins, family rule, endpoints, adaptive flags) and jobplan.json (unit costs from smokes) before any matrix.
  - Assign sub-ranges: Track A dev / train / select / confirm in 300–340M; Track B v3 screen / train / holds / octets / confirm in 6.1–6.9e9.
- **F18 [MINOR]**
  - A3: state the rollouts per option and the value-variance target, and name the learner (e.g. cross-fitted fitted-Q over options).
  - Define portworld's exact continuation for case types (a)–(d).
  - Optional branches start only if ≥ 50% of the budget is left at T+9 h.
