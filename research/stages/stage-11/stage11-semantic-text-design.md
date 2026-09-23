# Stage 11 public-text acquisition protocol

Stage 10 established complete fixed-set edge fitting with supplied node codes.
That reference is retained without rerunning its memorization experiment. This
track removes those supplied nodes: actor inputs are public English text and
its visible token inventory only. Privileged compiler graphs remain supervision.

## Observable contract and frozen architecture

Use eight alpha-distinct `unification` constructions from the unchanged pinned
TCN generator, difficulty .5, seeds searched deterministically from11000000.
This existing family has substantial canonical diversity; no broader binding or
set-operation transfer is implied. The initial study is fitting eight known
surfaces, not a512-example generalization test. No heldout data selects its
configuration. Compiler node order, ontology and categorical value vocabulary
are supplied label conventions; the vocabulary is fit only on this training set.

Before training, audit public/target collisions, lexical feature hashes, exact
visible identity copying, token alignment/truncation, node/slot limits, unknown
values and conflicting pair-slot targets. Actor inference receives only
`ActorInput(text, ())`. Its public text defines the copy inventory. Graph size,
gold presence, graph positions, ordered labels and token alignments never enter
actor state. Privileged sampled pair queries select output loss entries after
hidden states are computed; a mechanical test checks this independence.

Reuse the unchanged `SemanticCurriculumActor`: actual width1024, eight workspace
rows, four distinct blocks reused for two microsteps, fixed128node queries.
No runtime or structural-attention module is introduced. Keep its sampled edge
existence/relation objective. Replace only the documented slot mismatch with
edge-conditional supervision: actual unordered edges still receive no-slot;
nonedges receive no slot loss. Actor deployment predicts node presence, edges
and slots. No gold presence/edge mask is used in decoding.

## Training and measurement

One recipe, three paired initialization seeds30/31/32. Full batch of eight
training surfaces, AdamW lr1e-4, clip norm1, BF16 dense operations with FP32
parameters/losses. Preserve the Stage8curriculum: presence/type initially,
value/copy after1000presentations, edges/slots after2000. Maximum4000updates
=32000presentations per initialization,96000total, eight distinct graphs.
Final update4000 is the only advancement endpoint. Observe0/125/250/1000/2000
without checkpoint selection or early stopping. No extra exposure after failure.

Report raw zero-logit edge decoding and a separately fitted TRAIN-only decoder.
Calibration uses one relation threshold minimizing binary classification errors
over predicted-present pairs on these same eight TRAIN graphs; ties choose the
lowest threshold. Predicted absent nodes cannot be restored with gold masks.
An empty predicted-pair population uses a declared zero-threshold fallback with
zero calibration support. The same final thresholds would be frozen for any
subsequent evaluation; no test-specific threshold fitting is allowed. Presence
threshold stays fixed atzero. The calibrated restricted gate requires8/8complete
canonical graphs in every seed at4000. Raw graph correctness remains separate.

Complete metrics require node presence/type, visible copies, values, all typed
edges and ordered slots. Component metrics and loss curves diagnose failure but
cannot substitute for exact graph fitting. A matched fixed-mixture frequency
baseline uses the eight labeled constructions and no text features; repetitions
do not create additional independent labels or change its frequencies.

## Progression and resource rules

Profile20updates with development seed301 and full evaluation/export, width1024,
existing GB10 environment. Fix the main compute budget after measured throughput;
track cap75GPUminutes, intended under20. No GPU work begins before root release.
Every run records source/config/compiler/generator/data/checkpoint hashes,
parameters, workspace/token allocation, exposures, actual unique constructions,
CUDA allocation, process RSS, losses and complete compressed predictions/targets.

Failure at the fixed-set gate stops this track with component attribution. Only
if every seed passes may root authorize a separately frozen fresh known-renderer
study (at least512unique heldout examples where available). Alpha renaming and
unseen surfaces come only after that competence, with identifiable mappings.
No large scaling sweep, generator extension, runtime composition or supervision
withdrawal is part of this acquisition protocol.
