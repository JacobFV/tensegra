# C01 N2 development: numerical generalization failure retained

The contextual17,028,136-parameter comparator fails fresh answer acquisition under the frozen two-LR recipe. Calibration selects.0003 at2000updates (1472/2048 answers). Fresh validation answers2940/4096 (71.78%) and auxiliary values580/4096 (14.16%). Both LR runs nevertheless complete4000updates and all prescribed controls; the full2048000 presentations and466.858283148s process occupancy remain charged. N1 remains the strong neural comparator; N2's failure is not evidence that neural-only composition is impossible.

## Existing-output localization

The selected N2 model does learn substantial addressing: primitive4094/4096, destination4096/4096, first operand4074/4096, required second operand3250/3267, full ordered semantic proposal4056/4096. Its answer count conditional on correct lowering is2912/4056. Reversed roles yield3896/4096 full proposals and2784/3896 answers conditional on those correct proposals. Thus wholesale grounding failure cannot explain the main clean numerical/decision weakness.

Cross-entropy reconstructed from fresh exported logits is0.001536 primitive,0.000271 destination,0.014022 first operand and0.017103 binary second operand, versus1.616676 answer and6.102885 auxiliary value. These are evaluation losses, not an unavailable training-batch loss trace.

A separately labeled CPU diagnostic evaluates the frozen selected checkpoint on the first256 ORIGINAL fitting events, regenerated using the full original pool size and seed:255/256 answers and256/256 values, with CE0.007374/0.001711. This3.351s CPU-only diagnostic has no optimizer updates and does not revise selection. The in-pool/fresh contrast supports strong numerical/decision overfitting or generalization failure rather than a simple inability to fit these labels; it does not establish a specific internal memorization mechanism or perfect fit on all16384 examples.

Record/inventory order and fresh-name-no-op controls preserve2940/4096 answers; reverse roles yield2953/4096, unrelated records2920/4096. Full raw outputs and component diagnostics are preserved. The contextual baseline received all public numeric/table/query evidence; this is a retained failure of this architecture/optimization recipe, not a task information restriction.

## Extension decision and comparison limits

The.001 calibration-answer curve at0/250/500/1000/2000/3000/4000 is1014/1049/1033/990/1011/984/1000. Its last1000 gain16/2048 (0.781points) technically fulfills the registered eligibility trigger, even though the endpoint is below initial/chance and below the earlier best. The selected.0003 curve is1014/1413/1441/1412/1472/1433/1422 and declines late. Root explicitly declines a blind extension: eligibility was optional, the weak-arm rise is not credible useful acquisition, and the selected arm regresses. No threshold or historical outcome is changed and no extension ran. The original runner saves selected checkpoints but not final optimizer state; future continuation would need a new registered replay/optimizer decision, not a silent resume.

The numerical success of N1 on the same fresh clean population remains part of the comparison. Hybrid-vs-N1 uses answer-only against answer-only with inherited compute disclosed; the joint hybrid lowering-and-answer gate is separate. No C01 confirmation or broader advantage claim follows merely because N2 fails.

Process exit0 under1800s cap; internal465.567422898s, peak774238208CUDAbytes. GPU released before analysis. [Raw outputs and hashes](../../../results/campaign-01/composition/c01-development/n2/raw/manifest.json), [component localization](../../../results/campaign-01/composition/c01-development/n2/localization.json), [evaluation losses](../../../results/campaign-01/composition/c01-development/n2/evaluation-losses.json) and [fitting-prefix CPU diagnostic](../../../results/campaign-01/composition/c01-development/n2/train-prefix.json) are retained. Full PT snapshots and the selected checkpoint remain at `~/topoformer-campaign-01/composition/c01-development-n2/`.
