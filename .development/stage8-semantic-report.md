# Stage 8 semantic acquisition: preliminary measurements

Status: implementation and acquisition probes, **not yet a scaling result**.
The paired scaling study is frozen and queued; see the main-study section below. No semantic-runtime coupling
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
| Node-presence F1 |98.68%|98.11%|96.38%|
| Typed-edge F1 |18.89%|19.42%|24.72%|
| Ordered-edge F1 |15.11%|15.93%|20.34%|
| Exact graph recovery |0%|0%|0%|

The semantic model's English training edge count was 568 true positives among 5421
predictions, against 593 true edges. Thus high recall coexists with severe
false-positive decoding. On English validation, semantic typed-edge F1 was 15.48%
versus 20.88% for no-input. Exact graph recovery remained zero in all evaluated
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
one threshold per relation using only the first 8 **training** graphs, both
languages, on the fixed grid [-6,-4,-2,0,2,4,6,8,12]. It maximizes micro-F1 per
relation over dense edge predictions with **predicted**, never gold, node presence.
The same threshold vector is applied to every heldout renderer. Raw zero-threshold
metrics and calibrated metrics are both retained. Full threshold-grid counts are
logged. Ties choose the lowest threshold, including zero-gold relations; this can
still overpredict and must not be tuned after observing heldout results.

Calibration consumes 16 privileged training label presentations even at optimizer
step 0; that additional decoder-fitting budget is explicit. Calibration is not a
new neural competence result. The failed original acquisition artifacts remain
unchanged.

## Frozen main study

The resource freeze is now committed in `stage8-semantic-main-freeze.json`; the
study is queued, not yet a reported learning result. It uses 1k/10k available
training graphs, three paired seeds, semantic/no-text-feature arms, and one
100,000-presentation trajectory per arm with measurements at 0/10,016/100,000.
The 10k-scale measurement is explicitly **10,016**, aligned with batch32.

Optimized batch32 BF16 timing measured warm training steps of 0.1409 s for
semantic and 0.1313 s for no-input: 4.40/4.10 ms per presentation. The first steps
were 0.473/0.665 s. Removing per-field GPU synchronization and caching deterministic
CPU data changed throughput, not the mathematical per-example loss. Parameters
remain FP32; dense operations use BF16 autocast and loss logits return to FP32.
The projected serialized budget is approximately 95–110 minutes for all 1.2M
presentations, preprocessing and evaluation. This estimate is not a completion
claim.

### Diversity and genuinely fresh evaluation

The generated pool contains only four variable-binding and 120 set-operation
canonical constructions at this difficulty; the remaining accepted constructions
are unification examples. The raw N=1k and N=10k corpus distributions therefore
would change lesson mixture substantially. The main study fixes **equal lesson
presentation frequency**, independently cycling through each lesson's available
graphs and pairing English/Spanish renderings. The frequency control uses inverse
lesson-size weights to match that intended mixture. Repeated small-family examples
remain repeated examples, not new unique graphs. Actual unique counts and lesson
exposures are recorded at every checkpoint.

The main corpus uses a new data seed, 800000. Merely changing the seed was not
enough: 14 of its initial evaluation-prefix keys matched previously inspected
canonical graphs. Those keys are skipped only while reserving evaluation; they
remain eligible training examples. The final 64 heldout graphs have zero overlap
with the prior 64-key fixture, whose digest also matches Stage7's principal
heldout set. The new heldout contains 40 unification and 24 set-operation graphs.
It has **no fresh variable-binding constructions**, so this study cannot establish
variable-binding structural transfer. Both aggregate and per-lesson metrics are
required in the final interpretation.

The first eight main training graphs alone fit the edge thresholds. Neither main
heldout outcomes nor the previously inspected validation labels select thresholds.
The recipe, curriculum, sampling mixture and seeds are frozen before main learning
outcomes. No runtime composition or supervision withdrawal is authorized by this
study.

### Raw artifact representation

Raw predictions preserve every tensor field. Binary edge tensors are stored as
shape-tagged, little-endian bitmaps instead of oversized coordinate lists, with
one gzip shard per arm/seed/exposure. The historical acquisition conversion
reduced approximately 478 MB to 22 MB and verified every decoded field exactly.
Original compressed SHA256, converted SHA256 and record counts are committed;
the original is also retained in the remote durable artifact directory. Exact
historical source snapshots are committed independently of branch ancestry.

### Slot-objective capacity audit

A separate gold-only audit found a two-by-two XOR witness in each of the first
1,000 main training graphs. The ordered-slot head adds a source-specific and a
target-specific class logit. Its current loss labels sampled non-edge pairs as
“no slot,” alongside ordered slots on actual edges. Two independent ordered edges
and cross pairs labeled “no slot” can therefore require an XOR classification that this
additive head cannot satisfy with strict class margins. The audit tests absence
of an ordered slot; a cross pair can be either a non-edge or a real unordered
edge. Both receive the no-slot target when sampled. This is an irreducible
component of the slot training objective, not evidence that more data cannot help
other semantic outputs.

The independent edge head can suppress cross pairs during decoding, so the audit
does **not** prove that final graph recovery is impossible. The frozen main study
continues unchanged; any later edge-conditional slot objective would be a separate
experiment. The audit source and per-lesson counts are committed with the timing
artifacts. This limitation must accompany interpretations of ordered-edge and
whole-graph failures.
