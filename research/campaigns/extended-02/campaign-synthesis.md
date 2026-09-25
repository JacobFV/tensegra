# Extended-02 synthesis (phases 1–3)

Phase reports:
- [phase 1](campaign-report.md)
- [phase 2](campaign-report-phase2.md), [claims](claim-map-phase2.md), [audit](review/phase2-independent-audit.md)
- [phase 3](campaign-report-phase3.md), [claims](claim-map-phase3.md), audits [3](review/phase3-independent-audit.md) and [3b](review/phase3b-independent-audit.md)

Every number below comes from sealed, frozen-endpoint evaluations recorded in those reports.

## The headline question

> Can selection produce agents that obtain more verified problem-solving ability from the same computational resources, and does that ability transfer beyond the exact primitive combinations used during training?

**Selection.**
- **Copy-based selection does not help.** At equal RL updates, PBT (exploit/replace/mutate) never beat matched alternatives:
  - vs multistart −0.0097 pooled (E08);
  - adaptive curriculum: no gain (E12);
  - robustness-visible fitness: over-correction to tool avoidance (E13).
- **Allocating depth does help.** Successive halving, where eliminated members' updates go to survivors, beat both PBT and multistart in 3/3 replicates in point estimates (+0.015 and +0.005; E15). It found the good hyperparameters on its own. It still trailed a single learner given those hyperparameters.
- **None of the five population designs reliably produced the best behavioral strategy:** try the direct path first, then call the solver, then escalate budget. Concentrated single-lineage training did so in 5/7 exploratory lineages and 2/3 fresh confirmation lineages (E14 partial).
- **Mechanism.** Selection on short-horizon development utility removes the direct-first mode while it is still immature (E08/E12). Making tool loss visible to selection flips populations to the opposite brittle extreme (E13), or adds noise without fixing it (E21).

**Transfer beyond training combinations.**
- **Resource conditions.** Direct-first controllers match the public teacher on unseen budget and price combinations (E09/E14). They are robust to tool removal and corrupted returns; tool-first controllers are not.
- **New orders of known primitives.** In a three-primitive modular world, controllers execute unseen ordered pairs and three-stage sequences (E16). The registered rule passes on the aggregate reading.
- **Where it breaks.** Transfer failed at a return-binding shortcut that only a novel order exposed: apply whatever solver result exists. Training with wrong-type distractor returns repaired it (E17). Same-type, wrong-provenance returns were handled once a supplied provenance input (effectively a "not a prior record" bit) was added together with exposure (E19→E20: 0.35 → 1.00). m1 cannot represent provenance by construction; no m1 + exposure arm was run.
- **A new primitive** gave a large few-shot head start (0.67–0.98 vs 0.02–0.19 after 20 updates) that was not reliable at 120 updates (E18: partial). Zero-shot use of a never-seen primitive is not identifiable for these controllers.

## Supplied versus learned

**Supplied:**
- exact solvers, typed statuses, validators and draft builders;
- the public cheap-first teacher (bootstrap supervision);
- relational input features v2/m1–m3, including stage typing, provenance and failure counters;
- the niche descriptor in E15.

**Learned:**
- action choice;
- when and whether to compute;
- budget choice and escalation;
- return use;
- stage sequencing without a schedule.

Several repairs in this campaign are **input-contract repairs**: budget relations (E07), provenance (E20) and failure memory (E22). Each removed a specific, localized failure. None is evidence of learned representation discovery.

## Failure mechanisms found

| Failure | Cause | Repair |
|---|---|---|
| Budget escalation not learned (phase 1) | Missing relational input | Supplied features (E07: 57 → 119/128) |
| RL collapse to abstention | Unstabilized policy gradient | KL trust region, normalized advantages |
| Tool dependence | Training where tools are always available, plus selection | Long single-lineage training (not selection) |
| Return-type shortcut | Training never had a wrong-type return present | Distractor exposure (E17) |
| Provenance confusion | No provenance input | m2 features + exposure (E20) |
| Commit perseveration | Memoryless controller | Public failure counters (E22). A residual `choose_item` loop remains in 5 of 22,272 episodes per arm. |
| Omitted semantic composition (RL01/RL02) | Target addressing; pointer addressing produced duplicate edges instead | **Unresolved** |

## Not tested or not established

- Structural (graph-bias) attention advantage.
- Optimization-quality benchmarks: item objectives are constant.
- Confidence calibration.
- Descriptor-based zero-shot primitives.
- Independent-bank replication of the population mechanism: E08/E12/E13/E15/E21 share three banks.
- A recurrent workspace advantage: it collapsed to never calling tools under RL (E11).

## Resources

The ceilings were 48 CPU core-hours and 12 GPU device-hours. GPU use is measured as device occupancy, as accepted by the user. Totals are in [budget.json](budget.json). Failed and capped attempts are charged.
