# E15: depth versus breadth in selection (pre-registration)

**Motivation.** In E08/E12, each PBT lineage received only 300 sequential RL updates, because copying weights adds no depth. In single lineages the direct-first→solver policy emerged after ~700 updates. If the population loss is a *depth deficit* (temporal myopia), selection that reallocates the eliminated members' updates to survivors should reach the combined policy.

**Design.** Identical to E08 PBT r0–r2 (banks, hyperparameter rows, training mixture and streams, development panel, 5 rounds × 6 slots × 60 updates = 1,800 updates). The one change: mode `halving` with survivors per round (6, 3, 2, 2, 2).
- Each survivor receives 6/k consecutive slots per round, so the final survivors have 60 + 120 + 180 + 180 + 180 = 720 sequential updates.
- There is no weight copying and no mutation.
- Survivors are cut by last-slot development utility.
- **Arms:**
  - `plain`: cut by utility alone.
  - `niche`: at each cut (k ≥ 2), first keep the best member whose DEV greedy-first rate is ≥0.5 and the best whose rate is <0.5, then fill by utility.
- The niche descriptor is **supplied by the experimenter from phase-2 findings**. It tests whether protecting a known behavioral axis helps, not autonomous diversity discovery.
- Finalist: the better of the two final survivors by DEV utility.

**Evaluation.** Sealed E09 worlds (70M+), learned arms only, compared with E08 PBT/multistart/single on the same worlds.

**Endpoints.**
1. Finalist greedy-first (control) and no-tools success.
2. Sealed IID utility vs paired E08 PBT and single.
3. Transfer utility.

**Decision.** The depth hypothesis is supported if ≥2/3 finalists of an arm are direct-first (≥0.9) with no-tools ≥0.7 **and** the arm's mean IID utility ≥ the E08 PBT mean. The comparison of `niche` against `plain` isolates diversity protection. Banks are shared with E08: this is a mechanism test, not an independent replication.
