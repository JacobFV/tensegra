# Stage 8 semantic acquisition: preliminary measurements

Status: implementation and acquisition probes, **not yet a scaling result**.
The paired scaling study is pending resource freeze. No semantic-runtime coupling
or supervision withdrawal is enabled.

## Standard width and resource evidence

The actual recurrent workspace width is 1024 with four distinct phases, reused
twice. The first timing models had 57,861,981 parameters and used 1.224 GB peak CUDA
allocated memory. Four presentations took 0.447 s in the semantic arm and 0.135 s in
the no-text-feature arm, excluding evaluation/checkpoint work. This tiny cold
probe is not a reliable asymptotic throughput estimate.

The fixed 8-graph acquisition probe is more useful for warm serial throughput:
1000 presentations took 35.39 s training for semantic and 34.26 s for no-input;
total walltime including evaluation/checkpointing was 81.02 s and 75.51 s. Both used
width 1024 and 57,859,931 parameters (the smaller training-only value vocabulary
accounts for the parameter difference from timing). Neither used mixed precision.

Generating the shared 10k vocabulary pool plus 64 heldout graphs took 29.66 s,
requiring 30,491 generator attempts and rejecting 20,427 alpha-equivalent duplicates.
It produced all 10,064 requested unique graphs. Extracting its 10-value categorical
vocabulary took 8.09 s. These are generation feasibility measurements, not learning.

## Fixed 8-graph acquisition probe

Source frozen at 776990e; seed 0, two matched arms,1000 presentations, batch 2 via
serial gradient accumulation. Node supervision starts immediately; identity/value
at 100 presentations; edge/slot supervision at 300. The 8 graphs each have English
and Spanish forms. The 64 heldout graphs are diagnostic validation, not a test set
used to authorize semantic competence.

| Final training metric | Semantic English | Semantic Spanish | No-input |
|---|---:|---:|---:|
| Node-type accuracy |95.54%|96.68%|76.65%|
| Identity-copy accuracy |90.72%|89.80%|0%|
| Node-presence F 1 |98.68%|98.11%|96.38%|
| Typed-edge F 1 |18.89%|19.42%|24.72%|
| Ordered-edge F 1 |15.11%|15.93%|20.34%|
| Exact graph recovery |0%|0%|0%|

The semantic model's English training edge count was 568 true positives among 5421
predictions, against 593 true edges. Thus high recall coexists with severe
false-positive decoding. On English validation, semantic typed-edge F 1 was 15.48%
versus 20.88% for no-input. Exact graph recovery remained zero in all evaluated
surfaces. This probe therefore **does not establish graph acquisition**.

The node/type/copy results show partial learned acquisition; edge precision and
whole-graph recovery remain unresolved. Width alone is not sufficient under this
recipe. This does not isolate whether more optimization, edge-head expressivity,
or decoding calibration is the remaining limitation.

## Prespecified next diagnostics

The subsequent runner adds true padded minibatches and optional BF 16 autocast
without reducing workspace width. CPU contract tests check batching equivalence,
public padding masks, finite mixed-precision gradients and approximate agreement
with FP 32. GPU timing will select resources before freezing the main study.

Because the edge loss balances rare positives against negatives, zero logits are
not necessarily calibrated edge-presence decisions. A separate diagnostic fits
one threshold per relation using only the first 8 **training** graphs, both
languages, on the fixed grid[-6,-4,-2,0,2,4,6,8,12]. It maximizes micro-F 1 per
relation over dense edge predictions with **predicted**, never gold, node presence.
The same threshold vector is applied to every heldout renderer. Raw zero-threshold
metrics and calibrated metrics are both retained. Full threshold-grid counts are
logged. Ties choose the lowest threshold, including zero-gold relations; this can
still overpredict and must not be tuned after observing heldout results.

Calibration consumes 16 privileged training label presentations even at optimizer
step 0; that additional decoder-fitting budget is explicit. Calibration is not a
new neural competence result. The failed original acquisition artifacts remain
unchanged.
