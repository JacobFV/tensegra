# Stage 9 semantic contracts

Baseline `0f44e13`; Stage 8 source and artifacts remain immutable. This track does
not test structural attention or runtime composition.

## Initial evidence and source map

- `semantic_curriculum.SemanticCurriculumActor`: width1024, four reused workspace
  phases, canonical node queries, separate edge-existence/relation head, additive
  source/target slot logits.
- `semantic_curriculum.sampled_pairs/sample_losses`: all gold edges plus up to128
  active-node nonedges; class-balanced slot loss also labels nonedges no-slot.
- `semantic_scaling.targets`: canonical node, copy, value, typed-edge, scalar slot
  targets. Legacy scalar slot assignment would overwrite conflicting multiedges;
  no such pairs occurred in the entire archived construction pool.
- `semantic_scaling.decode/metrics`: predicted presence and edges mask final slot
  predictions. This is why an unsatisfiable slot objective does not make every
  final graph impossible.
- `tcn_data`, `semantic_graph`, pinned vendor: generation, rendering and canonical
  compiler. Exact compiler-order matching is stricter than denotational semantics.
- Stage8 main manifests/curves/prediction shards: archived metric reconstruction,
  not fresh model inference; frozen checkpoints remain on GB10.

## Contracts and cheap diagnostics

`stage9-semantic-contract-audit.py` regenerates archived descriptors read-only.
The 10,128 graph audit covers 30,384 public surfaces. It checks same text against
incompatible alpha-normalized targets; hashes; visible copies; declared vocabulary;
capacities; multiedge slot collisions. It found none in this finite support.
There is no actor token truncation. Maxima:37nodes,85edges,slot3,74tokens.
The pool is4variable-binding,144set-operation,9980unification constructions.
The64heldout comprise24set and40unification; all4binding are training-only.
This cannot establish unseen variable-binding structure generalization.

Mechanical width4 fixtures explicitly prove the additive margin identity and
construct a bilinear XOR solution; these are mathematical tests, not experimental
workspace substitutes. Conditional slot supervision includes unordered real
edges, excludes nonedges, and uses gold only in the loss mask.

## Proposed c2 development diagnostic (pending profile approval)

Hypothesis: explicit pair interaction and/or edge-conditional slot targets remove
the documented head/objective contradiction. Four arms: additive/interaction ×
all-pairs/edge-conditional. Fixed8canonical graphs, node representations are
privileged graph-and-node one-hot codes padded to1024. Node attributes and copies
are oracle supplied; learned edge and slot predictions must reconstruct the
complete graph. This isolates decoder acquisition and cannot establish public
surface understanding. Development seeds101/102, maximum300updates, equal graph
presentations; no threshold tuning on heldout data. Raw zero-logit edge decoder.
Stop after the cap; c3 blocked unless a candidate fits every graph exactly in
both development seeds. Main comparisons later require10/11/12 and fresh data.

The primary1024workspace standard remains unchanged. Decoder rank is an output
head hyperparameter, separately reported. No new generator semantics or runtime
primitives are introduced. Public acquisition and transfer gates remain blocked
pending c1/c2 and root budget/queue approval.

## c2 resource freeze

Profile completed on GB10, existing CUDA environment, source `c379744`, config
`stage9-semantic-oracle-profile.json`. Three updates per arm took0.458s cold
additive/all-pairs,0.065s additive/conditional,0.284s interaction/all-pairs,
0.264s interaction/conditional. Peak allocated GPU memory170MB/392MB; process RSS
is separately recorded. Head parameters1,902,658 versus6,359,106. Node feature
width1024; no neural recurrent workspace or memory tokens are used in this
privileged decoder-isolation task. Every graph/node has a supplied distinct
one-hot code; matching this code is not learned semantic grounding.

The fixed development budget is4arms ×2development seeds ×300updates ×8graphs
=19,200graph presentations. Expected2–3minutes, hard resource cap5GPUminutes.
Selection uses fixed-set development fitting only. No test surfaces, no learned
thresholds, no exposure extension after inspecting outcomes. Advancement needs
exact graph fitting under predicted edge existence in both development seeds;
otherwise public acquisition stays blocked and this bounded failure is reported.

## Faithful affine-head correction (separate diagnostic)

The first c2 fixture removed edge projection biases, whereas Stage8 uses affine
projections. It failed complete acquisition in every arm despite perfect slots
on true edges. This fixture mismatch limits attribution. Before further outcomes,
a separately named `stage9-semantic-oracle-affine` restores exactly those biases;
all other data,300updates,2development seeds and4arms remain fixed. The failed
bias-free result remains reported. Root approved≤2GPUminutes; no extra exposure
or tuned threshold. This is removal of a diagnostic confound, not a new runtime
or architecture. Public acquisition remains blocked until reviewed results.

## Prospective calibrated-contract confirmation

The final posthoc TRAIN diagnostic found strict per-relation score separation,
so the historical raw decoder's failure does not establish absent edge
information. A **separate prospective contract**, approved before its outcomes,
uses the same8-graph/300-update affine oracle diagnostic with all4head/objective
arms and three new initializations10/11/12. Data seed9001000constructs a new fixed
fixture, but small finite families may overlap historical canonical structures;
this remains fixed-set acquisition, not fresh structural generalization.

At the final update300 only, each relation's threshold minimizes unweighted
classification errors on all eight TRAIN graphs and every node pair. Candidate
thresholds are one below the minimum score and every distinct observed score;
strict `score > threshold` predicts an edge. Ties choose the lowest threshold.
No-positive relations choose their maximal score and establish no positive
recall. No test-renderer labels, intermediate-checkpoint selection or fitted
per-example thresholds enter the policy. Raw zero-threshold results remain
parallel. Every arm receives exactly the same decoder-fitting rule.

The new **restricted calibrated acquisition** criterion requires all8graphs
exact in every one of12runs; minimum512does not apply to this deliberately tiny
fixed-set diagnostic and no generalization claim is allowed. Original raw gate
remains unchanged. The four-way result must be reported regardless of outcome.
Estimated≤2minutes, hard3GPUminutes, existing environment. No C3 run is authorized
until the root reviews this result and freezes any further bounded acquisition.
Independent reviewer approved the boundary/tie/absent-class policy before launch.
