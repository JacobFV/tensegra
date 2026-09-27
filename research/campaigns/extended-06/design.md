# Extended-06 design (v1, 2026-09-27; internal review pending)

## Track A: portfolio world (a real combinatorial family with complementary methods)

**Task family: constrained packing/assignment with side constraints ("portworld").** Each instance selects a subset of n items (value, weight; multiple resource dimensions) under:
- capacities;
- a **conflict graph** (pairs that cannot co-occur);
- optional **precedence/requires** links;
- optional **uncertain constraint parameters**: some capacities or conflicts are hidden until an `inspect` action reveals them.

Instances come in **episodes of related instances** (k per episode). A later instance is a perturbation of an earlier one, and a cached solution may be **reusable** (still feasible and near-optimal) or **stale** (a perturbation broke feasibility or optimality).

**Utility** = verified solution quality − priced compute − priced observation − failure loss.
- Quality is value / optimal value, verified by an exact checker. The optimum is computed offline by exact search for evaluation only.
- Compute is solver work units × price; observation is inspect count × price.
- An infeasible or empty commit incurs a declared loss. Costs are charged once.

**Methods, each with an explicit contract** (input, output, guarantee, budget, work, resumability):

| Method | Output and guarantee | Notes |
|---|---|---|
| greedy (value/weight ratio with conflict filtering) | feasible solution, no guarantee | O(n log n) |
| local repair (swap/1-opt moves from a start solution) | anytime; improves or keeps | resumable |
| propagation (fix forced in/out items from constraints; reduces n) | reduced instance | fixed-point option |
| decomposition (split into conflict/precedence connected components; solve independently) | exact if the components are solved exactly | wins when the instance is decomposable |
| beam search width b | anytime | budgeted |
| branch-and-bound (DFS with LP-relaxation bound) | anytime; certifies optimality when finished | resumable budget; exact |
| reuse (take the cached solution for a related instance, verify, repair) | exact validity check | cheap when still valid |
| inspect (reveal hidden parameters) | information only | priced |

The right method should **vary with observable structure**:
- decomposability;
- tightness (capacity/total weight);
- conflict density;
- n;
- the fraction of hidden parameters;
- perturbation magnitude since the cached solution;
- the price regime (compute and observation prices are public).

**Headroom study (Phase 1, before any learned controller).** On a broad development population, evaluate:
- each single method under its best budget (A0);
- cascades, e.g. greedy → repair → B&B to budget;
- cheap-first, expensive-first and size-threshold rules;
- depth ≤ 3 decision trees and a tuned logistic selector on public features (A1 family; the best is tuned on development and evaluated on held-out folds);
- a strong hand-written portfolio;
- **an estimate of the public-information bounded-portfolio optimum.** This is a flexible cross-fitted selector over the same method/budget menu (gradient-boosted or MLP on public features and early-computation telemetry), plus a sequential version that may observe a method's status before choosing the next;
- a **hidden-state oracle** (per-instance best in hindsight), as a ceiling only.

- **HEADROOM** = U(best public-information bounded portfolio, cross-fitted) − U(best simple portfolio in A1 ∪ the hand portfolio), with instance-clustered CIs.
- **Gate GA:** HEADROOM ≥ .02 of the utility scale (relative to the between-instance SD, CI excluding 0), and no single method is within .01 of the best simple portfolio. If the gate fails, the task distribution may be changed **once, for a principled reason** (more heterogeneity in the observable drivers). If it still fails, **close Track A** and report it.

**If GA passes**, the Phase 2 arms are (≥ 3 lineages each, fresh held-out instances):

| Arm | Description |
|---|---|
| A0 | best single method |
| A1 | best tuned simple portfolio |
| A2 | learned contextual selector (one-shot choice of method and budget from public features) |
| A3 | learned sequential strategy policy over **options**: call a method to budget, observe status, continue, switch, inspect, reuse, stop. The options are supplied procedures; the policy learns **when** and in **what sequence**. |
| A4 | privileged ceiling |

- **Labels.** Belief-conditioned values for A3 targets are estimated by rollouts over resampled hidden parameters consistent with public information (hidden-parameter posteriors are explicit in the generator). The continuation is named. Case types (a)–(d) and switch metrics follow the brief §11.
- **Primary claim (registered at promotion):** U(A2 or A3) ≥ U(A1) + margin (fixed from development power), at comparable or lower total compute, in 3/3 lineages, on fresh instances.

## Track B: clean compositional generalization (probeworld v3)

**Structural screening (Phase 1, model-independent, exact DP).**
- Enumerate candidate pair and triple families from the probeworld factors: unreliable detection U, side effect S, correlated failure C, events E. If needed, add a versioned factor extension with a meaningful new factor, e.g. **probe cost scaling with hidden type** or a **deadline/horizon pressure** factor.
- For each candidate family, compute on eligible configurations:
  - the **decision-relevance of each factor**: the fraction of configurations whose optimal first (and later) decision changes when that factor is removed;
  - the **interaction residual**: optimal action or Q* under A+B vs what A-alone and B-alone predict;
  - **"global shift" solvability**: the best achievable score on the family by a policy that differs from the training-support optimum only by a global action-rate bias.
- **Select** challenge family (or families) with every factor decision-relevant in ≳ 30% of the eligible population, low global-shift solvability, and enough support. The selection is recorded before any model is evaluated on it. Historical U+E, S+C and U+C remain challenge sets only.

**Balanced counterfactual sets.** For each key decision type, matched configuration quadruples (none / A / B / A+B, plus a near-miss A+B that does not flip). Conditional accuracy on these contrasts is a primary endpoint alongside regret.

**Matched data-volume arms.** 3 seeds each, equal optimizer updates and equal total examples within each volume level:

| Arm | Configurations | Combination set |
|---|---|---|
| B0 | N | base combination set |
| B1 | 2N | base set |
| B2 | N | broader combination variety, never the held-out family |
| B3 | 2N | broader variety |

- **Primary causal contrast: B2 vs B0.**
- **Dose (if cheap):** 0/1/2/all extra pair types at N.
- **Factorized arms:** PUBLIC-RAW (B0 or B2 recipe), SUPPLIED-FACTORIZED (deterministic public factor values supplied; a localization ceiling), LEARNED-FACTORIZED (predict the factors from raw public inputs and use them downstream).
- **Causal factor tests:** on-manifold interventions, i.e. valid public configurations that change one factor. The policy should change only where that factor changes expected utility.

## Track C (conditional)
Only where A or B shows headroom that simple rules miss, plus a specific value bottleneck.
- **Value arms:** C0 monolithic; C1 decomposed (success probability, work cost, observation cost, remaining steps, wrong-commit risk, reuse value; summing to U without double-counting); C2 normalized per-remaining-query; C3 pairwise advantage/regret.
- **Metrics:** decision-relevant regret and ranking on close alternatives, by action class.

## Statistics, integrity, cost
- **Clustering and reporting.** CIs are clustered at the independent unit (configuration or instance; world; lineage). Paired comparisons are used on shared worlds. Absolute competence floors apply. Per-seed values and support counts are always reported.
- **Development vs confirmation.** Before confirmation, source, protocol, hyperparameters, selection rule, metrics and test family are frozen. ≥ 3 lineages. Adaptive replications are labelled.
- **Job plan.** [jobplan.json](jobplan.json) holds every job's dimensions × measured unit costs, generated before any large matrix. A one-job smoke comes before every batch.
- **Budget.** ~48 CPU core-h ceiling; aim ≤ 60% for Phases 1–4 and keep ≥ 20% for confirmation and audit.
