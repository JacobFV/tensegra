# Stage 8 semantic acquisition: preliminary measurements

Status: implementation and acquisition probes, **not yet a scaling result**.
The paired scaling study is pending resource freeze. No semantic-runtime coupling
or supervision withdrawal is enabled.

## Standard width and resource evidence

The actual recurrent workspace width is1024 with four distinct phases, reused
twice. The first timing models had57,861,981 parameters and used1.224GB peak CUDA
allocated memory. Four presentations took0.447s in the semantic arm and0.135s in
the no-text-feature arm, excluding evaluation/checkpoint work. This tiny cold
probe is not a reliable asymptotic throughput estimate.

The fixed8-graph acquisition probe is more useful for warm serial throughput:
1000 presentations took35.39s training for semantic and34.26s for no-input;
total walltime including evaluation/checkpointing was81.02s and75.51s. Both used
width1024 and57,859,931 parameters (the smaller training-only value vocabulary
accounts for the parameter difference from timing). Neither used mixed precision.

Generating the shared10k vocabulary pool plus64 heldout graphs took29.66s,
requiring30,491 generator attempts and rejecting20,427 alpha-equivalent duplicates.
It produced all10,064 requested unique graphs. Extracting its10-value categorical
vocabulary took8.09s. These are generation feasibility measurements, not learning.

## Fixed8-graph acquisition probe

Source frozen at776990e; seed0, two matched arms,1000 presentations, batch2 via
serial gradient accumulation. Node supervision starts immediately; identity/value
at100 presentations; edge/slot supervision at300. The8 graphs each have English
and Spanish forms. The64 heldout graphs are diagnostic validation, not a test set
used to authorize semantic competence.

| Final training metric | Semantic English | Semantic Spanish | No-input |
|---|---:|---:|---:|
| Node-type accuracy |95.54%|96.68%|76.65%|
| Identity-copy accuracy |90.72%|89.80%|0%|
| Node-presence F1 |98.68%|98.11%|96.38%|
| Typed-edge F1 |18.89%|19.42%|24.72%|
| Ordered-edge F1 |15.11%|15.93%|20.34%|
| Exact graph recovery |0%|0%|0%|

The semantic model's English training edge count was568 true positives among5421
predictions, against593 true edges. Thus high recall coexists with severe
false-positive decoding. On English validation, semantic typed-edge F1 was15.48%
versus20.88% for no-input. Exact graph recovery remained zero in all evaluated
surfaces. This probe therefore **does not establish graph acquisition**.

The node/type/copy results show partial learned acquisition; edge precision and
whole-graph recovery remain unresolved. Width alone is not sufficient under this
recipe. This does not isolate whether more optimization, edge-head expressivity,
or decoding calibration is the remaining limitation.

## Prespecified next diagnostics

The subsequent runner adds true padded minibatches and optional BF16 autocast
without reducing workspace width. CPU contract tests check batching equivalence,
public padding masks, finite mixed-precision gradients and approximate agreement
with FP32. GPU timing will select resources before freezing the main study.

Because the edge loss balances rare positives against negatives, zero logits are
not necessarily calibrated edge-presence decisions. A separate diagnostic fits
one threshold per relation using only the first8 **training** graphs, both
languages, on the fixed grid[-6,-4,-2,0,2,4,6,8,12]. It maximizes micro-F1 per
relation over dense edge predictions with **predicted**, never gold, node presence.
The same threshold vector is applied to every heldout renderer. Raw zero-threshold
metrics and calibrated metrics are both retained. Full threshold-grid counts are
logged. Ties choose the lowest threshold, including zero-gold relations; this can
still overpredict and must not be tuned after observing heldout results.

Calibration consumes16 privileged training label presentations even at optimizer
step0; that additional decoder-fitting budget is explicit. Calibration is not a
new neural competence result. The failed original acquisition artifacts remain
unchanged.
