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
