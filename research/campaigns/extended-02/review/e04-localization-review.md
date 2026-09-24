# E04 independent archived-outcome audit

All35 cells ×512 =17,920 raw episodes and56 paired comparisons reconstruct published success, costs, utilities, truncations and paired outcomes. Archive hashes match. Actions agree with counted episode lengths and observations; utilities equal verified success minus cost. Producer address-counter fields were summed, **not independently reimplemented**; this receipt does not extend their semantic scope. No checkpoint inference was rerun.

| Condition | Lightweight | Recurrent | Cheap | Always tool | Cheap first |
|---|---:|---:|---:|---:|---:|
| easy |512|512|452|512|512|
| medium |512|0|434|512|512|
| hard feasible |512|0|445|512|512|
| work tight |0|0|422|0|131|
| expensive tools |512|0|436|512|512|
| expensive travel |512|0|422|512|512|
| obstacle |309|0|430|512|512|

Every denominator512. These are single-initialization development transfer results, not replicated confirmation. Conditions use separate world seeds; within-condition method pairs share worlds. Price-condition differences alone are not paired causal effects of changing price on the identical world.

## Failure localization

- **Recurrent medium:** all512 episodes begin subset construction before inspecting the whole inventory, all contain an incorrect reduction, all finish without a selected assembly and repeatedly issue incomplete `verify`. One raw example also skips capacity and repeatedly retrieves its return. This is an acquisition/generalization failure in sequencing/formalization, not evidence that the exact executor failed.
- **Recurrent hard:** all512 episodes make no reductions/calls, retain no assembly, and finish with repeated incomplete verification. At least the inspection-to-computation transition fails. Do not attribute all recurrent failures solely to return retention or numerical representation.
- **Lightweight obstacle:** the203 failures exactly match203 episodes encountering the obstacle. Previous reductions are correct and assembly is selected; every failed ending repeatedly calls a stale route problem. Recovery/replanning after changed dependency is the clear missing behavior.
- **Lightweight tight work:** all512 fail. In495, assembly succeeds but the policy repeatedly builds route drafts after solver resources become inadequate; in17, assembly never succeeds and the policy repeats constraint insertion. Cheap succeeds422/512 while always-tool succeeds0: useful policy alternatives demonstrably exist from the allowed observations. This supports a budget/fallback curriculum, not a harder solver.
- **Lightweight expensive tools:** all512 succeed, but utility≈.3686 versus cheap-first≈.9136. Always-tool imitation is costly; success alone does not establish resource adaptation.

Next curriculum should retain easy anchors while exposing longer inspection phases, variable counts, insufficient solver budgets, interrupted route validity and cheap environmental motion. Measure closed-loop improvement on new development constructions. Do not add a larger runtime to fix these localized transitions. First keep the strong lightweight/reference policies and test whether recurrent control benefits from broader exposure rather than inferring an architectural impossibility from its narrow bootstrap.

## New objective review

Static inspection of `26c46050` confirms optional `episode_mean` sums policy terms within each episode, averages episodes, and keeps critic regression decision-averaged. Actor advantages stay detached; rewards are incremental verified utility. This addresses the previously documented random decision-count normalization mismatch. Entropy is also summed per episode: that is a declared cumulative-entropy regularizer, potentially favoring longer trajectories, so its strength must be assessed alongside computation/action costs. Default `decision_mean` preserves prior behavior; use an explicit frozen setting for the new comparison. No privileged reward is routed into inference observations. No runtime/gradient test was repeated by this reviewer.

## Accounting

Independent full reconstruction23.471134608CPU seconds; targeted example inspection0.010835984; four-cell localization3.084728656. Total **26.566699248 CPU-core seconds**, no children/GPU/inference. Raw-audit script and JSON include bindings and failure examples. No outcome-driven checkpoint selection occurred in this review.
