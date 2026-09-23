# S02 frozen learned-node edge readout

The bounded existing-head continuation did **not** meet its128/128 exact TRAIN edge criterion. It transiently improved complete edge acquisition, then regressed; the final endpoint is preserved rather than replaced by its better midpoint.

| Extra head updates | TRAIN exact edges, raw | TRAIN exact edges, calibrated | DEV exact edges, raw/cal |
|---:|---:|---:|---:|
|0|0/128|0/128|0/512,0/512|
|1,200|3/128|34/128|0/512,0/512|
|4,800|2/128|4/128|0/512,0/512|

Whole-graph counts happen to equal these edge counts at all measured checkpoints. This does not remove the frozen126/128 TRAIN whole-graph ceiling from two slot errors. The inherited DEV non-edge errors already imposed a zero whole-graph oracle-edge ceiling, so this arm was primarily an acquisition diagnostic for the frozen TRAIN interface.

Source7844351 cached640public-text forwards from the final S01-N128 checkpoint. On every example, original BF16 edge logits and logits recomputed from CPU-round-tripped cached node vectors were bitwise equal (maximum error0.0). Non-edge state hashes remained unchanged. The existing1,836,800edge parameters inherited AdamW moments and step8,192, together with the pair generator and shuffled-data continuation state. The1024-dimensional learned nodes, ontology, non-edge predictions, objective, and TRAIN-only relation calibration policy were fixed. No gold-node representation or gold inference mask was supplied. Clipping applies only to the remaining edge gradients and therefore is not numerically identical to joint-model continuation.

Full occupancy98.68698s; optimizer29.96155s; capture9.74841s; inner process97.34523s. PeakCUDA1,021,264,384bytes; processRSS4,323,856KiB. Raw predictions, targets, calibration floats, parent/data/source/config/cache/checkpoint hashes and full curve are archived under `research/results/campaign-01/semantics/s02-edge-refit`. Immutable node tensors/checkpoints remain on GB10, with hashes in the archive.

The midpoint shows that the frozen learned states permit better edge predictions under additional optimization. The late regression does not establish information absence or an architectural capacity bound. It instead leaves optimization stability, objective weighting and numerical/readout limitations unresolved in this bounded recipe. No extra head architecture or best-checkpoint claim follows. Independently justified full-backbone exposure on the more diverse8,192-construction corpus proceeds as separately registered S03.

Status: **failed restricted frozen-readout acquisition**, one development trajectory; no confirmation or generalization success claim. Independent raw-metric/provenance audit pending.
