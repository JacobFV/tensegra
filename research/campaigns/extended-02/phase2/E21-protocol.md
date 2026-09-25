# E21: depth-allocating selection with robustness-visible fitness (pre-registration)

**Motivation.**
- E15: halving (depth to survivors) beat PBT and multistart, but its finalists were direct-first in only 1/3 per arm.
- E13: robustness-visible fitness flipped *copying* PBT to tool avoidance, with IID utility ≈ the no-tool greedy.

E21 asks whether depth allocation plus robustness-visible fitness selects the combined direct-first → solver → escalate policy.

**Design.** E15 halving-plain r0–r2, unchanged (E08 banks, hyperparameter rows, streams, schedule 6,3,2,2,2). The one change is E13's `development_world_mix`: the E08 components + 4×4 no-tools + 5×5 tool-invalid, ~20% robustness worlds visible to selection only. Training never contains them.

**Evaluation.** Sealed E09 worlds (70M+), learned arms only.

**Endpoints.**
1. Finalist greedy-first rate and solver calls per episode on control worlds.
2. No-tools success (selection-visible, not held out).
3. IID utility vs paired E15 halving-plain and E08 single.

**Decision.** The combined policy is reached if ≥2/3 finalists have greedy-first ≥0.9, solver calls ≥0.15 per control episode (i.e. they still use tools), no-tools success ≥0.7, **and** mean IID utility ≥ the E15 halving-plain mean (0.933). If finalists avoid tools (calls <0.15) the result is "over-correction as in E13"; otherwise it is mixed.
