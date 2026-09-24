# E13: does robustness-inclusive fitness stop PBT from discarding the direct-first mode? (pre-registration)

Registered after E08/E09/E12 sealed results, as a mechanism test prompted by them. It is not a confirmation of E08.

**Observation motivating it.** In all six PBT populations (E08 and E12, three banks), members that attempt a direct (greedy) commit before calling a solver existed at round 0. Copies of higher-development-utility tool-first members replaced them, and every finalist ended ~100% tool-first. Tool-first finalists fail when tools are unavailable (sealed no-tools success ≈ 0) or when instances exceed the solver's 20-item contract. Single learners that kept or re-acquired the direct-first mode reached the best sealed utility.

**Two explanations.**
- (T) Temporal myopia: direct-first members are still immature when selection happens.
- (D) Distributional myopia: the development panel never contains a world where the tool is unavailable, so fitness cannot reward robustness.

**Intervention.** E13 is identical to E08 PBT r0–r2 (banks, hyperparameter rows, mutation RNG, training mixture and streams, 5×60 actor-critic rounds). The only change is the development (selection) panel. It uses `development_world_mix` = the 8 E08 components + {4×4 with work_limit 0 (no tools)} + {5×5, 64 steps (subset instances exceed the 20-item contract)}, assigned by the usual public-free seed hash (~20% robustness worlds). Training never contains these worlds.

**Endpoints (sealed E09 conditions, same frozen worlds).**
1. Finalist greedy-first rate on `int_4x4_control`.
2. Finalist success on `int_4x4_no_tools` and `xfer_5x5_subset_tool_invalid`. These conditions are now in the *selection* distribution, though not in training. They are reported as selection-visible, not held-out.
3. Sealed IID mean utility vs the paired E08 PBT finalist.

**Decision rule.**
- (D) is supported if ≥2/3 finalists are greedy-first (≥0.9) **and** have no-tools success ≥0.7.
- If the finalists remain tool-first, distributional visibility alone is insufficient, which favours (T) or a representational explanation.
- Anything in between is reported as mixed.

Three replicates share banks with E08/E12; they are not new independent lineages.
