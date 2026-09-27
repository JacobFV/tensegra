# Portworld (Track A): specification, method contracts, generator, headroom study

Code:
- [src/tensegra/campaign06_portworld.py](../../../src/tensegra/campaign06_portworld.py): generator, methods, oracle, utility, public features.
- [research/tools/campaign06_portfolio.py](../../../research/tools/campaign06_portfolio.py): `evaluate`, `pilot`, `headroom`, `plan`.
- [tests/test_campaign06_portworld.py](../../../tests/test_campaign06_portworld.py).

Everything is pure Python + numpy. No torch, no scipy.

## 1. Task family

**Instance (true).** n items, each with:
- an integer value v_i;
- a group g(i);
- a local weight wl_i and a global weight wg_i.

A solution S (a subset of items) must satisfy:
- **Local capacity per group:** Σ_{i∈S, g(i)=g} wl_i ≤ capl_g.
- **Global capacity:** Σ wg_i ≤ capg. This constraint may be non-binding.
- **Known conflicts (public):** no two conflicting items are both selected.
- **Possible conflicts:** public candidate pairs, each real with public prior probability ½; the truth is hidden.
- **Requires links (public):** i ∈ S ⇒ parent(i) ∈ S. The links form a forest inside each group.
- **Hidden capacities:** for a hidden group only the public interval [lo, hi] is known. The truth is uniform on its integers, which is the explicit posterior.

**Episodes.** Each episode has k ∈ [3, 6] instances.
- Instance t is a perturbation of instance t−1. The perturbation has magnitude m, and m = 0 means an identical repeat. It changes values, weights, capacities, hidden caps and conflicts.
- The perturbation summary is public: the counts of changes and m.
- **Cache.** The cache for instance t is the certified optimum of the true instance t−1. It is a supplied solution, independent of any policy. This keeps utility tables free of path dependence.
- The cache may be reusable, stale-but-feasible, or infeasible.

**Planning modes.**

| mode | plan | observations | guarantee |
|---|---|---|---|
| `cons` | hidden caps at lo; possible conflicts treated as real | 0 | every output is truly feasible |
| `insp` | inspect everything: the true instance | n_hidden (priced) | every output is truly feasible |
| `opt` | hidden caps at the posterior median; possible conflicts absent | 0 | may be infeasible (failure loss) |

The posterior feasibility probability of any solution is computed exactly from public information (`posterior_feasible_prob`). It is available as telemetry.

**Utility** (one committed solution per instance): U = quality − c·work − o·observations − L·[infeasible or empty commit].
- **Quality** = value / optimum, verified by the exact checker against the truth.
- **Charged once.** In a sequence, every stage (the greedy incumbent, propagation, a resumed B&B run) is charged at its maximum work. Observations are also charged once.
- **Commit rule:** after an inspection, the best truly feasible output is committed; otherwise the highest-valued output is committed.
- **Selector charges** (design v2 rev. 11) are added in work units:
  - public-feature computation (`feature_work`, deterministic);
  - telemetry calls (they are arms, so they are charged as work);
  - inference: tree comparisons, logistic multiply-adds, or GBT rounds × depth × models, at 1 unit each.

## 2. Methods and their contracts

Work units are deterministic counts of item-level operations. No wall time is used.
- The oracle pilot fitted core-s = work / 16.4M (R² .98).
- Every anytime method is run once to `TRACE_MAX` = 1.024M units. Each budget in the menu and every rule is read off that trace (`BnB.at_budget`, `PD.at_budget`, `trace_arm`).
- The tests verify that a trace read equals a fresh run at that budget.

| method | input → output | guarantee | work units | resumable |
|---|---|---|---|---|
| greedy `G` | plan → feasible S | none | n⌈log₂n⌉ + items touched by add checks (requires-closure) | – |
| local repair `R` (in `GR`) | plan + feasible start → feasible S′ | never worse; local optimum (add / 1-for-1 swap with closure) when finished | items touched per move evaluation; budget 20n²+50 | yes: `run(b)` repeatedly = one run (tested) |
| propagation (`propagate`) | plan + incumbent value → fixed-in/out masks | every solution better than the incumbent respects them (tested vs brute force) | 2 bound evaluations per free item per round, to a fixed point | – |
| decomposition (`components`) | free items → independent components (conflicts, requires, binding shared capacities) | exact: optimum = incumbent ∨ fixed + Σ component optima | union-find scan | – |
| B&B (`BnB`) | plan + warm start → incumbent; certified UB trace | anytime; certifies optimality when the stack empties; the logged global UB is valid at every trace point (tested) | 1/node + bound scan | yes: explicit stack; at_budget(b) = fresh run at b (tested) |
| `PD@b` | GR incumbent → propagate → decompose → per-component GR seed and B&B, smallest component first, shared budget b | exact when every component finishes; certified UB = fixed + Σ component UBs | GR + propagation + seeds + search (charged b unless finished) | per component |
| `RUPD@b` | as PD, warm-started from the verified/repaired cache when it beats GR | as PD | + reuse | per component |
| beam `BM{4,16}` | plan → best complete S of width-w beam over the ratio order | none | bound scans per candidate | – |
| reuse `RU` | cache → verify (popcount + conflict scan); drop the lowest-ratio violating item until feasible; local repair | output feasible for the plan | verify + drops + repair | – |
| `RV` | cache → commit as is if it verifies for the plan, else `RU` | output feasible for the plan | verify (+RU) | – |
| inspect | → reveals all hidden parameters | information only | o per parameter | – |

**Oracle (evaluation only; `exact_optimum`).**
- Greedy+repair, then propagation and decomposition, then B&B per component using min(method bound, **surrogate bound**). The surrogate aggregates the binding rows with multipliers 1/capacity.
- The cap is 8e7 units (~5 core-s).
- **Cross-checks (tests):**
  - brute force for n ≤ 16 (pw-v1/v2/v3);
  - on n = 24–30 it agrees with the plain method-bound B&B, a second bound implementation;
  - certified UBs are ≥ the optimum.
- **Uncertified instances are kept, not dropped.** They are scored against the certified upper bound, which makes their quality conservative, and they are counted.

## 3. Generator knobs and revision log

All knobs are public or produce public quantities, except the true hidden capacities and the true status of possible conflicts. There are no hidden strategy-class bits. Knob sets are versioned in `KNOB_SETS`; old versions stay reproducible (tests run v1–v3).

| knob | pw-v3 value |
|---|---|
| correlation class (Pisinger, public id) | uncorrelated / weakly / strongly / almost strongly / subset-sum-like, uniform |
| n | U[16, 50]; public per-class cap: 40 for strongly and almost strongly correlated (oracle pilot) |
| groups G | {1, 2, 3, 4, 6, 8} (≤ n/3) |
| local tightness | U[.2, .8] ± .1 per group |
| global tightness | U[.2, 1.4] (≥ ~1 is non-binding) |
| within-group conflict density | U[0, .35] |
| cross-group conflicts | 0 w.p. .5, else U[0, .03] |
| requires fraction | U[0, .3] |
| hidden-capacity fraction | 0 w.p. .4, else U[.2, 1] |
| prior width | U[.1, .5] |
| possible conflicts per item | 0 w.p. .5, else U[0, .25] |
| episode length k | U{3..6} |
| perturbation m | 0 w.p. .2, else U[.05, 1] |
| value change for a changed item | ×U[.5, 1.5] |
| weight change | w.p. .6m, ×U[.7, 1.4] |
| conflict churn | .5·m·|E| edges swapped |
| compute price c | log-uniform, c* ± 1 decade, c* = 5.0e-7 |
| observation price o | log-uniform, o* ± 1 decade, o* = 9.7e-3 |
| failure loss L | U[.2, 2] |

The drivers are drawn independently. A Latin hypercube was not used, because episodes are seed-addressed and chunked.

**Revision log** (development instances, builder sub-range 300,000,000–304,999,999; all numbers are development numbers):

| version | reason | development headroom evidence |
|---|---|---|
| pw-v1 | initial | N=1,099 (250 episodes). Oracle − best single only ≈ .017; inspection paid on ~all instances (mean obs cost .015 vs ~7% conservative loss); the cache stayed near-optimal at every perturbation size. HEADROOM (v1 tool) .0016 [−.0002, .0034]. |
| pw-v2 | **Principled heterogeneity.** The observation price range was one-sided, so it was widened upward: log10 o ∈ [−3, −.7]. Perturbations could not break caches, so their size was raised to m ≤ 1, with larger value/weight changes and more conflict churn. Conflict density was widened. | N=1,090. Oracle − best single .042. HEADROOM (v1 tool) .0084 [.0053, .0117]; best simple (tree d3) − single .026. Oracle − best simple .0155 < .02, so GA could not pass even with a perfect selector. |
| pw-v3 | **Design v2 compliance, not a GA-fail revision:** admissible sources only (review F3); Pisinger classes for heavy-tailed B&B effort; per-class size cap from the oracle pilot; prices centred by the stated crossover rule; `opt` mode (hidden constraints with explicit posteriors, failure loss live); propagation+decomposition always-on in B&B arms (BB/RUBB without decomposition removed); RUPD added | see §5 |

**Oracle pilot** (`pilot`, 24 instances per n ∈ {20, 30, 40, 50} × class; 171 metered core-s):
- p99 oracle CPU ≤ 2.2 s for classes 0/1/4 at n = 50.
- Strongly and almost strongly correlated at n = 50: 3/24 and 3/10 were uncertified at 5e7 units, so those classes are capped at n = 40 (p99 .53 s and .43 s; p99 work 9e6, i.e. a heavy tail beyond the largest menu budget).
- **Price crossovers** (insp mode, 160 instances):
  - median(PD@1e6 quality − GR quality) = .018; median work 36.5k, so c* = 5.0e-7;
  - median inspection gain = .039 over a median of 4 hidden parameters, so o* = 9.7e-3.

## 4. Headroom study (`campaign06_portfolio.py headroom`)

**Menu.**
- 45 arms: 3 modes × {G, GR, RV, RU, BM4, BM16, PD0, PD@{1e3,1e4,1e5,1e6}, RUPD@{1e3,1e4,1e5,1e6}}.
- Plus trace-read rules at any `TRACE_GRID` budget.

**Cross-fitting.** K = 5 outer folds, clustered by episode. Every tuned constant, and the choice of the best simple family, comes from training folds only.

**Simple (A1) families.**

| family | what it is |
|---|---|
| `A0_single` | best single arm |
| `cascade_tuned` | 324 cascades: mode × {RV-first, GR-first} × {PD, RUPD} ladder × cap × gap threshold. Cheap-first ladders, and expensive-first (cap 1e6, τ = 0), are members. |
| `strong_baseline` | the design v2 rev. 8 family (540 settings), described below |
| `hand` | hand-written sequential portfolio (80 settings of 3 constants), described below |
| `threshold_any` | depth-1 policy tree |
| `tree_d2`, `tree_d3` | utility-maximizing policy trees with root lookahead, on public features |
| `logistic_tuned` | softmax selector with an expected-utility objective; L2 by inner CV |

- **`strong_baseline`:**
  1. inspect iff θ·hidden_frac·L > o·n_hidden, else `cons` or `opt`;
  2. always verify/repair the cache and warm-start from it (RUPD, with propagation + decomposition always on);
  3. B&B budget = κ/c, read off the trace;
  4. certified-gap stop: stop at the first trace point with (UB − inc)/inc < γ·c·b;
  5. node-count switch rule: if the run is not certified at b_sw, switch to beam-16.
- **`hand`:**
  1. choose the information mode by the least expected loss among inspect (o·n_hidden), conservative (α·voi_gap, the public optimistic-vs-conservative root-bound gap) and optimistic ((1 − posterior feasibility of opt:GR)·(1 + L)), with opt:GR as a paid probe;
  2. verify the cache when m is small;
  3. GR;
  4. escalate PD/RUPD budgets while c·Δb < γ·gap(root bound) and the run is not certified.
- Trees, logistic and GBT choose from arms ∪ the top-5 (training) settings of cascade / strong / hand.
- **Best simple rule:** selected on 4 **inner** folds of each outer training set (no winner's curse), then scored out of fold. The best single method is chosen on the training folds.

**Public-information bounded-portfolio estimates (BPE).**
- **One-shot:** structured multi-output GBT with per-arm models of quality, log-work (Duan smearing) and failure probability, on the price-free public features. It is combined with the public prices and exact observation counts: Û = q̂ − c·Ŵ − o·obs − L·p̂_fail. GBT hyperparameters come from inner CV.
- **Sequential:** for each probe ∈ {cons:GR, cons:PD@1e3, cons:RV+GR, opt:GR}, the probe is paid for and its telemetry observed:
  - root bound, values, gaps;
  - propagation/decomposition stats and the gap after 1e3 units;
  - cache verification;
  - posterior feasibility.

  The same structured GBT then chooses the continuation over all arms plus "stop". A first-call selector, cross-fitted inside each training set, chooses between the one-shot estimate and each probe policy.
- Also reported: a direct multi-output GBT on the menu's utilities (`learned_oneshot_direct`).

**Gates** (design v2 rev. 7; reported separately for `learned_oneshot` and `learned_sequential`):
- **GA-1:** Û − best simple (inner-selected) ≥ .02, with the episode-clustered 95% lower bound > .005.
- **GA-2:** the same margin against the best single method.
- The hidden-state oracle (per-instance best arm) is a non-deployable ceiling only.

**Also reported:**
- per-driver tertile breakdowns (n, correlation class, tightness, conflict density, components, hidden fraction, m, prices);
- hindsight-best share per arm, method and mode, flagging methods best < 5% of the time;
- oracle certification rate and the number of instances scored against the UB.

## 5. Development results (pw-v3)

(filled below)

## 6. Supplied vs learned

| supplied (fixed procedures) | learned / tuned |
|---|---|
| every method, its contract and work meter; propagation, decomposition, the exact checker, posterior feasibility, the oracle (evaluation only) | which method/mode/budget to run (one-shot), and when to stop, continue, switch or inspect after telemetry (sequential) |
| cascade, strong-baseline and hand structures | their constants (tuned on training folds) |
| public features (deterministic, charged) | tree splits, logistic weights, GBT models (cross-fitted) |

## 7. Registered GA run (root launches)

(filled below)
