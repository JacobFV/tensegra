# S20 fresh confirmation: acquisition replicates; heldout combination fails

The fixed known-motif confirmation claim **passes in all three independent lineages**, against both original and contextual matched-calibrated baselines. Scratch record accuracy on the three trained motifs is95.898%,94.727%,94.857% for seeds701/702/703 (mean95.161%). Every seed exceeds90% macro, every known cell exceeds80%, and every paired macro improvement against both matched baselines has a positive conditional95% event-bootstrap lower bound. **Heldout3×4 fails:0/512 for every seed and every arm.** No extension or recipe change is inferred.

All nine arms completed4,096 incremental updates before analysis. The frozen loader and exact source/config/parent/exposure/event/target/calibration guards passed unchanged. This is fresh construction confirmation of the learner comparison, conditional on three independently initialized lineages; it is not isolated architectural attribution or broad shape transfer.

| Seed | Arm/policy |3×3 /512|3×4 /512|4×3 /512|4×4 /512|
|---|---|---:|---:|---:|---:|
|701|Record categorical|509|0|501|463|
|701|Original raw|0|0|1|0|
|701|Original matched|3|0|21|0|
|701|Context raw|0|0|3|0|
|701|Context matched|3|0|40|0|
|702|Record categorical|509|0|502|444|
|702|Original raw|0|0|6|0|
|702|Original matched|2|0|67|2|
|702|Context raw|0|0|2|0|
|702|Context matched|2|0|53|1|
|703|Record categorical|511|0|498|448|
|703|Original raw|0|0|3|0|
|703|Original matched|7|0|52|4|
|703|Context raw|0|0|5|0|
|703|Context matched|16|0|84|7|

Each baseline uses its own corresponding independent S12 decay parent model+AdamW. Both baselines within a lineage share that parent; seed201's historical baseline is not reused as replication. All701/702/703 parents are retained, including701's prior competence failure. Record starts from scratch with the same numeric lineage seed. Original/context inherit196,608 presentations and24,576 optimizer steps; record inherits none. All receive the same mixed4096 corpus,32,768 incremental presentations/eight visits and construction-index stream. Original/context negative-query streams also match. Parameter count, objective, inherited history and computational work differ:57,853,781 versus62,677,315 parameters. This limits causal attribution without weakening the observed fixed-learner acquisition comparison.

## Paired uncertainty and TRAIN fitting

| Seed | Record minus matched original, pp [95% CI] | Record minus matched context, pp [95% CI] |
|---|---|---|
|701|94.336 [93.164,95.443]|93.099 [91.862,94.336]|
|702|90.104 [88.607,91.536]|91.081 [89.648,92.448]|
|703|90.755 [89.323,92.122]|87.891 [86.263,89.453]|

Intervals use10,000 stratified resamples of the same sorted512 event IDs per known cell, shared across all seeds/policies/comparators, RNG20020. They quantify event uncertainty conditional on these models, not a population-of-initialization confidence interval. The average over the three fixed lineages has interval[90.712,92.708]pp against matched original and[89.583,91.732]pp against matched context. All raw comparisons also favor record by94.336–95.833pp. Complete paired2×2 transitions, cell counts and continuous effects remain in `s20-analysis.json`; no comparator is discarded.

Record completes128/128 actual TRAIN-panel graphs in every seed. Original matched TRAIN counts are4/128,10/128,9/128; contextual matched counts5/128,6/128,13/128. Original raw counts0/1/1 and contextual0/0/1. Baseline matched TRAIN scores overlap calibration fitting and are optimistic; record has no calibration. Full training loss curves, per-field record TRAIN diagnostics and all per-cell components are retained. This replicates the earlier contrast in acquisition of whole graphs, rather than merely a threshold-policy effect.

## Failure boundary and deterministic examples

Record heldout valid-output counts are366/512,415/512,340/512. Exact presence occurs285/330/258 times, but **no heldout graph has exact kind, value, copy, edge or slot components** in any seed. Invalid reasons are primarily generated node references:145/86/166, with duplicate edges0/2/6 and unsupported multiple pair-slot labels1/9/0. Thus both well-formed wrong graphs and malformed references contribute; validity alone cannot recover the missing structural combination.

`s20-failure-examples.json` retains the lexicographically first semantic ID among incomplete outputs for each seed, cell3×4 or4×4, and valid/invalid category. These deterministic examples illustrate boundaries, not representative sampling or a new selection rule. For valid heldout event0050cd790737… all three seeds first diverge at zero-based NODE index10. Seed701 emits kind5 with public-copy index7 where the gold record requires kind7/value1, despite matching total31 nodes; all later semantic components fail. In known4×4, seed701's first valid failure079c1e1a973a… has every node component and slots exact but wrong edges. Seed702's corresponding example07751a83f4f0… has correct nodes but wrong edges/slots. Seed703's00349426887e… first mismatches copying at node30. Raw records and source artifact hashes are preserved, so these observations can be independently reconstructed without another forward pass.

## Cost and evidence scope

Main outer elapsed2992.71 seconds, conservatively charged**2992.72**, exit0/GPU-free, below4300 cap. Profile33.83 gives total**3026.55 seconds** for S20; unused cap is not elapsed compute. There was no interim confirmation-based selection. The reserved population was opened for the authorized frozen final inference and complete-matrix analysis only.

Main archive399a50f9, config4e648a9a274a7cabcbb84c0b2e6e6a2dcbf37925e2146b0758702086245f6c8b; prospective loader5bf4078f/core22cc4cf3 with independent clearance30e414ff and earlier core4bf94915. Independent raw codec/threshold replay and checkpoint model/AdamW auditing are separately underway; this report's loader checks do not substitute for that audit. No formula, source or policy changed during analysis.

The result supports replicated known-motif acquisition by the fixed sequential learner and preserves a sharp heldout-combination failure. S21's broader-corpus development question was registered from S19 before S20 confirmation outcomes; it is a separate intervention and cannot revise these confirmation criteria or outcomes.
