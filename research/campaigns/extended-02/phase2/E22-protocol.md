# E22: is direct-commit perseveration a memory gap? (pre-registration)

**Observation.** A lineage-level failure recurs across phases. After a direct completion attempt (e.g. `commit_pending`) is rejected, some controllers repeat it dozens of times instead of switching to the solver (E17-r2, E20-r2: 37–50 rejections per failed episode). The lightweight controller is memoryless. Its observation reports only the *last* action's feedback, so once any other action intervenes, the history of rejections is gone from its input.

**Intervention.** Identical to E20 (initialization seeds, streams, mixture with both distractor types, recipes). The one change is features m3 = m2 + public per-stage counters: rejected completion attempts, and own solver calls, since the current stage began. Both are counts of events the actor itself observed.

**Evaluation.** The E20 sealed set (E16 conditions + wrong-type + same-type distractor conditions; seeds 90M/95M/96M).

**Endpoints.**
- Pooled failure rate from commit perseveration: failed episodes with ≥10 rejected completion attempts, over all non-hard conditions, E22 vs E20 per lineage.
- Overall success; retention.

**Decision.** Memory-gap hypothesis supported if pooled perseveration failures fall by ≥50% (E22 vs E20, summed over lineages) with no condition-group mean success dropping by >0.02. It is refuted if they do not fall. This is still a single seed per lineage pairing; lineage-level idiosyncrasy can dominate.
