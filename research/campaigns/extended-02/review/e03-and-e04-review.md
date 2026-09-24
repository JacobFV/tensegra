# E03 acquisition audit and prospective E04 evaluator review

E03 final compressed predictions and summaries were independently reconstructed without model inference. Both width1024 actors attain128/128 verified success,256/256 correct reductions, no truncation, mean cost0.028824921875 and utility0.971175078125. Trace/history actions and feedback agree; costs and compute counts reconstruct; complete selections satisfy the fully inspected public facts; submitted constraints/maps match those facts. Every paired environmental action trace is identical between actors.

There are **128 shared unique development episodes**, reused at six checkpoints, and one initialization per architecture. This is neither768 independent evaluation examples nor a replicated comparison. Earlier checkpoint success is summary-reported; only final600 raw predictions are present in the inspected local archive. Both arms use4800 training episodes and81600 teacher decision presentations. Parameter counts are2,205,698 lightweight versus26,364,930 recurrent, not parameter matched.

Training curves' `training_success=1` describes teacher trajectories. Closed-loop development measures the learned actor: lightweight summary reaches1 at100updates; recurrent summary is0 at100 and1 from200 onward. The residual≈.186 cross-entropy is compatible with interchangeable inspection targets; it is not an episode failure. Final traces show exactly two solver calls, two retrievals and two return uses per episode under always-tool supervised bootstrap. Evaluation supplies no schedule, but the observed behavior is acquired reproduction of that schedule, not evidence of learned resource allocation or new primitive composition.

## Frozen evaluator `1320b3e1`

Static inspection found no teacher, hidden plan or evaluator label entering learned scoring. Hidden specs are archived for audit, while policies receive only world observations through the feature encoders. Checkpoint hashes are enforced; paired arms construct fresh worlds from the same specs and independent declared address seeds. Models are frozen/no-grad. The persistent executor does not expose other actors' private outcomes.

The evaluator itself does not enforce train/test semantic separation. Before E04, bind fresh condition seed ranges, source hashes, checkpoint hashes and selection policy in the registry; inspect those against training history. Same seed conditions with changed public costs permit paired counterfactuals; different seeds do not establish within-episode causal adaptation.

Return-address metrics explicitly test role, retrieval, status and stale-result validity, not optimal-return choice or original-goal semantic correctness. Keep that limited label. Multiple extant records can include irrelevant different-role returns; successful addressing does not automatically establish competition among several plausible same-role answers.

References see full public JSON; learned policies see supplied feature encodings. Neural-vs-neural information matches, but reference comparisons need this disclosure. The default reference compute tariff0 versus model tariff1 is acceptable only with an explicit interpretation or zero public compute price; it is not evidence of fair modeled computational pricing by itself. Freeze priced tariffs before outcome selection and retain actual runtime/CPU comparisons.

The evaluator's per-batch neural wall allocation is throughput accounting, not serial episode latency. Parent CPU excludes the persistent solver and is not the campaign ledger. Models may have different maximum step limits inherited from checkpoints; confirmation must disclose or match these. No evolutionary advantage follows from E03 or this evaluation.

Audit scripts/receipts bind all six input artifacts. Two read-only reconstruction passes used0.1298664CPU seconds plus the final receipt's CPU field; no GPU, solvers or checkpoint inference. The second pass additionally verifies reduction labels against fully observed submitted inputs. Root should charge both inclusive audit passes once.
