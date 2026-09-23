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
