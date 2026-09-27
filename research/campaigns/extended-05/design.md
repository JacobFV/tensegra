# Extended-05 design (v1, 2026-09-27; pending internal review)

## 0. Evidence reconciliation (brief §5)
- **Composition probe rate.** The extended-04 report said ".545 in all 6 seeds". That is corrected on main (3a566812):
  - B1 seeds 0–2 on heldout_comp: .636/.636/.773 (B1 audit);
  - F2 seeds 3–5: .545 (48/88, F audit).

  Both are below .80. They are **distinct experiments**, kept separate.
- **What each probeworld result is:**

  | Experiment | Initializations | Outcome draws | Configurations |
  |---|---|---|---|
  | B1 | new (seeds 0–2) | offset 500 | the registered held-out configurations (128 per split) |
  | F2 | new (seeds 3–5) | new (offset 700) | the **same** 128 configurations |

  So **F2 is not new configuration-level transfer.**
  - The held-out horizon k = 4 is an unseen *intermediate* value (training k ∈ {1, 2, 8}).
  - The historical omitted pairs (unreliable+events, side_effect+correlated) are **inspected challenge sets** from now on, not untouched confirmation data.
- **Baselines and decoding (frozen references, all hashes in extended-03/04 configs):**
  - depworld X1 bootstraps r0–r5: r0–r2 from extended-03; r3–r5 fresh in extended-04.
  - Development-selected deployment mode: greedy + R-mask. A1 and F1b show it ≈ best mode.
  - The strongest training variant, a2-dep (masked-sampling RL, r2 only), is at bootstrap level.
  - The dep_reuse reference.
  - Probeworld: L1/L4 models seeds 0–5, plus π* and the fixed rules.
- **Unresolved mechanisms:**
  - (i) why neither RL nor on-policy supervision moves argmax behaviour above cloning;
  - (ii) whether any consequential intervention has headroom over greedy + R-mask;
  - (iii) where the combination failure arises: belief, value estimate, ranking, or deployment;
  - (iv) why value estimates fail at unseen horizons.

## 1. Track A: deployed-policy improvement with measured headroom (depworld)

**Deployment contract D** (frozen): the X1 bootstrap actor (d1 features, width 1024) scores the candidate set.
- The progress-diagnostic-v1 **R-mask** removes flagged stagnant action keys (release on signature change; verify/abstain never masked; all-masked falls back to the argmax).
- **Greedy argmax** chooses, with ties going to the lowest candidate index (catalog order).
- The tracker updates each step. The episode stops at verify success, abstain, or the 96-step limit.

This is the extended-04 "greedy + R-mask" procedure (`campaign04_deploy.py` mode `r_mask`), bit-for-bit.

**Consequential option set O(I)** (public, from the action catalog; no hidden masks):
- D's own choice;
- every `call(p, b)` over open problems p and allowed budgets b (budget choice/extension);
- every `use_return(h, u)` whose record type fits u (reuse), versus `start_subset`/`start_assign`/`build_route` (recompute);
- `uncommit(assign)` versus `uncommit(select)` (revision scope);
- `abstain` (stop).

A-HR also reports the **complete catalog** one-step deviation on a subsample, to test whether O(I) misses headroom.

**A-HR, headroom gate (evaluation plus branching only; no training).** States are sampled from D's rollouts on development worlds and on 3 fresh lineages (bootstraps r3–r5). For each a ∈ O(I): clone the environment (shared solver, exact solver cache), take a, then follow D to the end. The label is the realized remaining utility, U_final − U_t.
- **Two references are reported, never confused:**
  - (a) the **hidden-state rollout bound** H = E[max_a Q_true^D(I,a) − Q_true^D(I,D(I))]. It is computed on the true world, so it is an *optimistic, non-deployable* upper bound on one-step improvement over D.
  - (b) the **same-information estimate**: the regression of branch returns on public features (Q̂^D). It is unbiased for E[· | visible history] in expectation. Its argmax gain is estimated on held-out states with paired branch evaluation.
- **Gate G1:** the IID-group H ≥ .02 utility per episode, averaged over visited states and summed along trajectories via episode-level evaluation of the "oracle one-step rollout policy". If the gate fails, test one **bounded multi-step option** class: a budget override for the next solver call plus D until the next commit, or a revision option (uncommit-scope plus recompute plus D). If that also fails, **stop Track A** and report "no headroom in the permitted class".

**A-PI, rollout-based cost-sensitive policy improvement (only if G1 passes).**
1. Collect D-visited states on training worlds. Retain a mixture: 50% D rollouts, 25% rollouts of the improved policy (iteration 2 only), 25% dep_reuse-teacher states.
2. Branch all a ∈ O(I) with K repeats where stochastic (solver cache makes branches deterministic given the world).
3. Train a **cost-sensitive ranking head** over O(I) on the frozen actor features, with a small head and disclosed parameters. The loss is a pairwise hinge/softmax weighted by |ΔQ| (the regret of misranking), with near-ties (|ΔQ| < .005) as equivalence sets.
4. Deploy π₁(I) = D(I) unless the ranked best option's predicted gain > margin m (registered on development data).
5. Iterate at most once (RSPI-style, arXiv:0805.2027) if development shows gain.

- **Comparators at matched label cost:**
  - (i) D;
  - (ii) D plus teacher-action supervision on the same states (extended-04 A3-style; must be beaten);
  - (iii) simple public heuristics: always budget 128 then escalate; reuse-if-applicable; minimal revision scope;
  - (iv) dep_reuse.
- **Safety rail.** Deployment-time fallback to D when the ranking head's top-option confidence is low (registered), and the frozen D is always retained. No REINFORCE on argmax actions; no differentiating through argmax.
- **Endpoints (IID group; per lineage):**
  - utility (verified success − declared costs);
  - success;
  - cost;
  - per-step no-progress;
  - intervention rate;
  - **decision-level:** ranking accuracy within O(I), regret of the chosen option, false-positive and missed interventions.
- **Promotion (development screen, 1 lineage):** utility ≥ D + .015 with success ≥ D − .01, **and** beats (ii) by ≥ .01.
- **Confirmation A-CF:** fresh lineages r3–r5 (bootstraps exist), sealed fresh worlds, 512 per condition.
  - **A-CF-H:** utility ≥ D + .01 in 3/3 lineages, success ≥ D − .01, per condition absolute floor ≥ .8 × dep_reuse.
  - The numerical margins are fixed by the development power analysis before confirmation and written into the registry.

## 2. Track B: compositional value-of-computation (probeworld)

**B-LOC, localization (no training).** Existing models on the **historical challenge sets** (the two omitted pairs, separate from price/k shifts): L1/L4 seeds 0–5, greedy.
- For every decision, record: configuration, visible history, exact belief, Q*/V*, the optimal set, the chosen action, the model's Q-head values (L4), the policy argmax, and the gap δ_t = V*(I_t) − Q*(I_t, a_t).
- Localize the **first consequential error** (δ > ε) and classify it:
  - (1) **ranking** (the model's Q-head argmax is ε-optimal but the policy picks otherwise);
  - (2) **value estimate** (the Q-head misorders the chosen versus the optimal action);
  - (3) **belief-dependent** (the error vanishes when the exact belief is supplied as an input to a diagnostic re-run; see B-X arm O);
  - (4) **deployment** (the ε-optimal action exists but was chosen under greedy ties, or is masked).
- Realized regret and accumulated gap regret are reported separately. Their expectations coincide under the registered utility and termination; per trajectory they differ.

**B-SPLIT, new composition holds (registered before any model run).** Flags: unreliable (U), side_effect (S), correlated (C), events (E), giving 6 pairs.
- **Historical challenge pairs:** U+E, S+C.
- **New confirmation holds:** **U+C** and **S+E**. These are never used in training or selection.
- **Training support:** all singles plus the remaining pair U+S, and C+E where the generator supports it, plus the historical pairs *only* in the exposure arm. Selection uses a development split of training configurations.
- **Split-support audit before runs:** labels are functions of visible information; held-out configurations are disjoint by generator parameters; and the held-out pairs are verified never to co-occur in training.
- Transfer is reported separately for fresh instances, new continuous values, omitted combinations, and the unseen intermediate horizon.

**B-X, acquisition vs factorization** (3 seeds each; matched updates and capacity; public inputs only unless labelled):

| Arm | Description |
|---|---|
| **B0** | The L1 action-set baseline, unchanged recipe, trained on the new split. |
| **BX1** | Exposure: more diverse training configurations (more price and k draws, all permitted singles and pairs), same updates. |
| **BX2** | Factorized public state: a **deterministic public-state computation supplied architecturally**. It is the exact Bayesian belief over θ given the visible history, plus explicit per-flag features (detection reliability, event hazard, side-effect cost, remaining horizon, build/use costs). Interactions remain expressible (the MLP/GRU sees all of them jointly). Labelled **supplied belief**, not learned reasoning. |
| **BO** | **Oracle diagnostic:** B0 inputs plus the exact belief, evaluated on the challenge sets only. It is a localization aid, never a claim. |

- **Endpoints on the new holds (U+C, S+E):**
  - uniquely-optimal probe rate (support counts shown);
  - not-optimal probe rate;
  - conditional decision accuracy and regret near the switching boundary (ρ bins with support);
  - value-of-computation ranking accuracy for the Q head;
  - held-out regret;
  - absolute success ≥ .8 × π*.
- **B-X primary (registered):** on the new holds, BX2 or BX1 raises the uniquely-optimal probe rate by ≥ .15 over B0 in 3/3 seed pairs, with not-optimal probing ≤ .10 and regret ≤ B0's. Otherwise the failure is reported as unresolved by exposure or supplied factorization.
- **Statistics:** confidence intervals are clustered by configuration (the 128 configurations × 4 draws are not 512 independent units); per-seed and per-configuration-family values are reported.

## 3. Supporting calibration (inside both tracks)
Decision-relevant only:
- within-state ranking of O(I) (A) and of probeworld actions (B);
- predicted vs actual incremental utility, per action type (never pooled with stop);
- probability that an intervention is beneficial;
- regret of the chosen option;
- risk/coverage per shift.

The baselines are constant forecasts, per-type averages, and public-feature rules. **Aggregate ECE is not a promotion gate.**

## 4. Metric-design checks (applied to every endpoint)
- **Improve or fail by construction?**
  - R-mask-scored metrics are excluded from any claim about the learned choice.
  - The no-progress episode rate is not used (it cannot change under masking).
  - Ranking accuracy is computed only over options with |ΔQ| ≥ .005.
- **Pass by abstaining or never intervening?**
  - Utility includes failures, so abstaining costs success.
  - The intervention rate is reported.
  - A policy identical to D scores exactly D and so cannot pass a strict "≥ D + margin" test.
- **Collapsed denominators?**
  - Absolute floors apply everywhere.
  - Rates carry support counts, and rates with < 20 support are marked "insufficient".

## 5. Structural-attention branch (conditional)
Only if A-PI or BX2 yields a concrete relational interface (e.g. the dependency graph over records and problems used by the ranking head). Then compare graph-as-data attention, keyed memory/message passing, graph-bias attention (B_r = P_q A_r P_kᵀ) and hard routing, with zero, wrong and corrupted-graph controls. Otherwise it is not run.

## 6. Sequencing and stopping
1. Reconcile and design (this document); internal review; job plan.
2. **In parallel:** A-HR headroom (evaluation plus branching) and B-LOC localization plus B-SPLIT audit (CPU).
3. A-PI screen if G1 passes (otherwise the multi-step option test, then stop A). B-X screen.
4. Confirmations A-CF and B-CF on fresh lineages/seeds and sealed worlds or new holds.
5. Independent audits; report.

A failed screen selects the next discriminating diagnostic. Close when both questions reach an evidential boundary or the ceilings are reached.

---

## v2 revisions (after internal review [review/design-review.md](review/design-review.md): 3 BLOCKER, 11 MAJOR; these supersede v1)

**Track A**
1. **G1 is a necessary condition only (F1).** The hindsight-state bound H is reported per option class, with and without abstain. **Abstain/stop gains are a separate class and do not count toward G1** (they arise only in episodes D later fails, so they are hindsight-favourable by construction). The G1 bound is H excluding abstain ≥ .02 per episode.
2. **G1b (deployable estimate) is added (F1, F2).** The policy class is **single-deviation-from-D**: at publicly sampled decision points (a fixed public sampling rule), a learned ranker may replace D's action once per episode; otherwise D is followed. The episode-level gain of such a policy **equals the branched advantage exactly**, so it is estimated from the planned branches with **cross-fitting by world** (regression Q̂^D on public features; argmax over O(I) at the sampled point; the true branch value of the chosen option minus D's).
   - **G1b:** the cross-fitted same-information gain ≥ .01 per episode on the IID group, with a CI excluding 0.
   - The oracle rollout policy (sum over states) is dropped; summing per-state headroom along trajectories is invalid.
3. **O(I) is anchored to D's action kind (F3).**
   - At D-call states: budget alternatives, plus "call a different open problem".
   - At D reuse/recompute states: use_return(applicable-typed) vs recompute.
   - At D commit/uncommit states: revision-scope alternatives (uncommit assign vs select).
   - At other states: "call now" options.
   - Abstain is always its own class.
   - The full catalog is only on a subsample.
   - Measured on dev: 17 options per decision on average; D takes consequential actions at 38% of decisions and never abstains; 89% of its calls use budget 128.
4. **Lineages (F8).** r0–r2 are for headroom and screening; **r3–r5 are reserved for confirmation only**.
5. **Loss (F4).** The ranking loss is linear in realized cost differences (cost-sensitive regression of ΔQ with argmax selection), or a paired regression head. Near-ties (|ΔQ| < .005) are equivalence sets.
6. **Iteration 2 (F5).** Its continuation is named "π₁ (the single-deviation improved policy), then D". Otherwise no iteration.
7. **Teacher comparator (F6).** "Matched label cost" means the same number of labelled states and the same simulator core-seconds, spent on teacher-action labels. It uses 3 seeds.
8. **Infrastructure (F7).**
   - A checkpoint **path-remap layer** (tensegra-campaign0X → structured-latent-dynamics-campaign0X; hashes verified). Hashed configs are never edited.
   - The **D device is registered as CPU** for branching/labels (as in extended-04 Track C) and as GPU for large evaluation. CPU/GPU numerical identity must be tested on a sample, or a single device kept per comparison.
9. **Costs (F17).** Learned-head evaluation is priced at **.35 core-s/episode**. The job plan is re-run with all arms, evaluation-time branching and the multi-step option test (revised ≈ 100k).

**Track B**

10. **New split table version (F12).**
    - The new holds U+C and S+E are removed from training **and** development/selection.
    - A new split-table version with **new seed bases**; the old table is kept for historical reproduction.
11. **Pool sizing from the exact solver before any model run (F11).**
    - Evaluation pools are sized so each primary cell has ≥ 20 configurations with a uniquely-optimal first probe (≈ 1,500 U+C and ≈ 350 S+E configurations, 1 world each).
    - CIs are configuration-level. Cells under 20 are marked "insufficient".
12. **Correlated flag (F10).** It is mostly inert at k = 1, so U+C results are also reported on the registered subset where the pair changes the optimal value (flag-sensitive configurations).
13. **B-HR headroom gate (F13).** B0 (the L1 recipe on the new split) is evaluated on the new holds first. If its probe/regret gap is already small (uniquely-optimal probe ≥ .80 with support ≥ 20), there is no failure to repair, and B-X reports that instead of training arms.
14. **BX2 and a modular arm (F14).** BX2 is kept as a *labelled supplied-belief control*; it is predicted null at the first decision, where the belief equals the prior. A new **BX3 per-flag modular encoder** is added, with matched parameters. All arms are reported **by decision depth**.
15. **BO (F15).** The oracle-belief diagnostic arm is trained on the **extended-04 split** to localize the old models' failure (+2.4k core-s).
