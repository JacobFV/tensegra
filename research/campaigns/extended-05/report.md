# Extended-05 report: decision-relevant metacognition and compositional computational control

**Status: FINAL** (§Headline, Track B and recommendation wording corrected after external review; no numbers changed). The campaign ran on the pro6000 from 2026-09-27T01:27Z, closing at an evidential boundary after about 3.2 h of wall time. Independent audits ([A](review/a-independent-audit.md), [B](review/b-independent-audit.md)) reproduced every reported number from raw rows, and their corrections are applied below. A design review ([review/design-review.md](review/design-review.md)) preceded implementation.

**Sources:**
- Design: [design.md](design.md) (v1 plus v2).
- Registry: [registry.json](registry.json), 14 prospective entries, including adaptive ones labelled as such.
- Decisions: [decisions.md](decisions.md), with timestamps from `date` or git.
- Job plan: [jobplan.json](jobplan.json).
- Seed ranges: [seed-ranges.json](seed-ranges.json).

## Headline: the three questions from the brief

| Question | Answer |
|---|---|
| **Did we learn to choose better computations?** | **Partly.** In depworld, one-step deviations from the competent deployed policy D (imitation + greedy + R-mask) have **no deployable headroom** (A-HR). The hindsight bound is .0077 per episode against a .02 gate, and the same-information estimate is −.002. The one option class with headroom is **delegating to a better supplied sub-policy until the next commit**. A learned public-information controller for that choice improves D by **+.021 to +.028 utility** on fresh imitation actors and sealed worlds (A-CF-T, 3/3). The audit shows this is equivalent to **recovering the supplied teacher**: per-world success is identical in 5/6 lineages. In probeworld, **training exposure to more condition combinations** improves the value-of-computation decision (probe first) on an unseen combination (B-XC, below). |
| **Did that choice generalize across new combinations?** | **In one family, replicated; not in general.** On fresh configurations of the unseen unreliable+correlated combination, with fresh initializations, broader combination exposure raises the uniquely-optimal first-probe rate by **+.21 (95% CI .11–.31)** and lowers gap regret in 3/3 pairs (B-XC, adaptively registered after inspecting that pair family). The side_effect+events hold showed **no failure to repair** (B0 probes at .95–.98). A supplied exact belief (BX2) and per-flag modular encoders (BX3) did not fix first-probe decisions. BX2 did lower gap regret through later decisions. |
| **Did the learned regulator add value beyond simple rules?** | **No.** Learned delegation timing never beat the best simple trigger rule, under free delegation (A-PI-T, A-CF-T) or under per-step delegation costs of .001 and .003 (A-PI-C). In every lineage, "delegate at every commit/revise point" (R2) or "delegate at every call" (R1) matched or beat it by about .001–.002, while delegating fewer steps. |

**Not substituted:**
- **More determinism for better decisions.** The extended-04 finding is kept, and in depworld nothing beyond delegation helped.
- **Calibration for control.** Appraisal quality was not a gate. The aggregate Q-ranking accuracy did not track decision quality (B-Q).
- **Buying a supplied solver for discovering structure.** The teacher and the build operation are supplied.
- **A richer vocabulary for competence.** No new metacognitive variables were claimed.

## Track A (depworld): what was tested

- **Contract.** D is the exact extended-04 r_mask deployment (80/80 re-run match). Its continuation is named in every label.
- **A-HR headroom.** 1,536 episodes, 512 world clusters, 3 lineages; one-step options anchored to D's action kind.
  - Hindsight bound (non-deployable) .0077 per episode (CI .004–.012).
  - Cross-fitted same-information single-deviation gain −.0021 (CI −.0042 to −.0004): public estimates made deviations harmful.
  - The full action catalog added almost nothing. Abstain was its own class and never counted.
- **A-HR2 multi-step option.** Delegate to the public dep_reuse teacher until the next commit or 12 steps.
  - Hindsight .026 (lower CI bound .016, so the pass is on the point estimate).
  - Deployable estimate +.021 to +.025 (CI excludes 0).
  - All the value comes from 70/1,536 episodes that D fails.
- **A-PI-T / A-CF-T.** Frozen controller: ridge, 372 parameters, margin .075, committed before evaluation.
  - vs D: +.035 to +.044 (screen) and +.021 to +.028 (fresh lineages r3–r5, sealed 260M).
  - Not better than the best simple rule. Delegated steps 5.5–7.7% against R2's 3–4%.
  - "Fresh lineages" means new imitation actors; the teacher and controller are shared.
- **A-PI-C.** Per-step delegation cost c ∈ {.001, .003}; controllers retrained on exactly cost-adjusted labels.
  - Always-teacher drops to .872 and .815.
  - The controller still beats D (+.028 to +.044).
  - R2 remains best (.899–.901, +.002 over the learned controller).
- **Why no improvement beyond rules.** The recoverable value is concentrated in rare D failures at identifiable public anchors (commit/revise points). A fixed trigger at those anchors captures it as well as a learned estimate.

## Track B (probeworld): what was tested

- **B-LOC localization** on the extended-04 models and the historical challenge pairs:
  - **52–66% of the first consequential errors are value-estimate errors**: the model's own Q-head argmax is also wrong. This is evidence about the representation's accessible value predictions (an auxiliary head), **not a demonstrated cause** of the policy's choices. The exposure benefit in B-Q came without better Q ranking.
  - 23–38% are ranking errors.
  - 0 come from deployment (ties).
  - A "belief-dependent" class was **withdrawn by the audit**. Other seeds without belief inputs fixed errors as often, so no belief effect is detectable.
- **B-SPLIT v2.**
  - New holds U+C and S+E, removed from all training and selection.
  - Pools sized with the exact solver (≥ 60 eligible configurations).
  - The new holds are weak compositions: the second flag matters in 12–16% of configurations.
- **B-HR.**
  - U+C shows a failure (B0 .52–.65 uniquely-optimal first probe).
  - S+E shows none (.95–.98).
- **B-X** (U+C primary; registered rule ≥ +.15 in 3/3 pairs): **not supported** for any arm.
  - BX1 (exposure): +.10/+.15/+.13, with lower regret.
  - BX2 (belief): fewer probes, lower gap regret via later decisions.
  - BX3 (modular): worse.
- **B-Q.**
  - Exposure's benefit appears in the decisions (first probe, gap regret), **not** in all-pairs Q-ranking accuracy.
  - The registered mechanism, that exposure improves value estimates, is not supported.
- **B-XC, confirmed.** Adaptive registration; fresh U+C configurations (seed base 5.9e9, 1,289 configurations); fresh seeds 20–22.
  - Mean paired first-probe gain +.206 (+.167/+.233/+.217). Per seed (B0 → BX1): 56.7% → 73.3%, 55.0% → 78.3%, 51.7% → 73.3%. This is **conditional first-decision accuracy on the 60 eligible configurations** (of 1,289) shared by all pairs. It is not overall task success.
  - Gap regret lower 3/3 (−.81/−1.09/−1.62).
  - It also passes the stricter original B-X rule.
  - **Caveats:**
    - the pair family had been inspected;
    - **exposure is confounded with data volume:** B0 trains on 384 configurations over 7 combinations, BX1 on 768 over 9, at equal updates and architecture. The benefit is of that broader training distribution; the experiment does not separate more configurations from more combination types or different decision frequencies;
    - **the gain may partly be a shift toward probing** rather than more systematic use of the interacting conditions (BX1 probes more everywhere). The evidence does not choose between these;
    - BX1 about doubles the overall probe rate, and its not-optimal probing rises (still ≤ .10);
    - most of the gap-regret gain comes from later decisions;
    - one weak composition only; it is not new-pair transfer.

## Supplied, learned and generalized

- **Supplied:**
  - the depworld world, solvers and applicability relations;
  - D's imitation actor (from extended-03/04);
  - the recovery rule, the dep_reuse teacher (the sub-policy) and the delegation option semantics;
  - probeworld's exact labels (training only), the split families and the BX2 belief features.
- **Learned:**
  - the public-feature estimate of when delegation helps (372-parameter ridge; a 6.7k-parameter MLP at c = .003);
  - probeworld action choice under action-set supervision;
  - the benefit of wider combination exposure.
- **Generalized:**
  - the delegation trigger transfers to new imitation actors and sealed worlds (A-CF-T);
  - the exposure benefit replicates on fresh U+C configurations and initializations (B-XC).
- **Not shown:**
  - learned regulation beyond simple rules;
  - generalization to a new untouched pair;
  - structural induction.

## Integrity, incidents and accounting

- **Spend.** CPU **44.6k / 172.8k core-s** metered (dev 3.8k), plus ~3.0k local, estimated. GPU occupancy **6.6k / 43.2k s**. About 125k core-s is left unspent.
- **Unit costs.** Evaluation with a learned head was priced at .35 core-s per episode after the review; actual jobs ran well below the plan. There was no overrun (the extended-04 lesson).
- **Incidents** (all in decisions.md):
  - A merge committed with conflict markers in seed-ranges.json. The launch guard refused all launches, and the file was resolved by hand.
  - Controllers were not staged twice, because snapshots exclude results. Both times the failure came at startup, and the one-job smoke caught the second.
  - A missing world namespace.
  - Two decision timestamps were hand-written ahead of wall-clock time and corrected to commit times.
- **Failed launches** (≈ 13 core-s each) are receipted and charged.
- **Worlds.** All ranges (220–280M, 5.1e9–6.0e9) are registered, disjoint and fresh. Each sealed range was used once.
- **Audit-driven corrections.** Adopted throughout, notably:
  - A-PI-T/A-CF-T gains = recovering the teacher;
  - "not better than" rather than "below" the best rule;
  - the B-LOC belief class withdrawn;
  - the BX2 later-decision benefit acknowledged;
  - the B-XC caveats.

## Next-decision recommendation

1. **Stop learning when-to-invoke regulators over fixed option menus in depworld.** With a competent D, the recoverable value sits at public anchors that simple rules capture. Metacontrol research needs environments where the best intervention **varies across instances at the same apparent decision point** and is inferable from available evidence. The target is to beat strong, relatively simple, equally informed heuristics and tuned portfolios. *(Corrected after external review: an earlier draft said 'no fixed rule can express', but any computable policy is a rule.)* Build and measure that headroom first, as A-HR did.
2. **Improvement over the teacher, not recovery of it.** The deployed gains equal the supplied teacher's. The next Track A question is whether any learned procedure can exceed dep_reuse (.90), for example by learning to invoke *different* sub-policies than the ones supplied.
3. **Composition.** The exposure effect is real but narrow. Test it on a genuinely untouched pair or triple family registered in advance, with exposure dose varied. Test whether the gain works through later-decision value estimates, where BX2 also helped, or through first-probe recognition.
4. **Value estimates under shift** remain the common failure (extended-04 B-H6, extended-05 B-LOC). A targeted study of value-head generalization, such as horizon-normalized or decomposed targets, is the most direct lever for both tracks.
