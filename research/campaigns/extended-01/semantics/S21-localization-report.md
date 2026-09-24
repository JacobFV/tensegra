# S21 fixed-endpoint generated-prefix localization

**Broader motif training acquires the added familiar shapes, but the held-out3×4 failure still begins at the first predicate boundary.** In both paired arms, every one of the512 held-out examples predicts an extra `ident` where the next `pred(parent)` node belongs. This is secondary failure localization after both complete arms closed, not a new selection rule or a substitute for the independent primary audit.

The reviewed prospective script ran unchanged after validating the closed successful receipt, frozen config5a13908e, both arms/all four checkpoints, all TRAIN/DEV files and hashes, complete support, identical paired targets, and direct DEV identities against the public cache. Width1024/source77683135; no inference, refitting, calibration or threshold changes.

## Canonical greedy outcomes

All cells contain the same512 paired development examples per arm.

| Cell | Original exact graph | Broad exact graph | Original all node records correct | Broad all node records correct |
|---|---:|---:|---:|---:|
|2×4|0|504|0|511|
|3×3|507|505|510|512|
|3×4, held out|0|0|0|0|
|4×3|507|502|511|508|
|4×4|455|436|490|475|
|5×3|0|504|0|509|
|5×4|0|437|0|474|

The new-motif gains are genuine free-running graph improvements, not only teacher-forced component gains. Known old cells lose26 exact graphs net: paired repairs/regressions are2/4 at3×3,2/7 at4×3, and28/47 at4×4. New cells repair504,504,437 previously incorrect cases respectively. Held-out3×4 has512 wrong→wrong pairs and no correct graph in either arm.

## Held-out boundary and layout

Both arms fail first at the identical zero-based canonical node indices:8 in12 examples,9 in157, and10 in343. Every first discrepancy changes kind/value/copy together: predicted `ident` versus required `pred(parent)`. Thus the model continues a fourth argument when the arity-three fact has ended. Neither arm produces a completely correct node-kind sequence, correct entire node records, or even the correct entity-removed occurrence-tree layout for this cell.

Entity-removed layout matches:

| Generated held-out layout | Original | Broad |
|---|---:|---:|
|trained4×3|388|302|
|trained4×4|1|11|
|other|123|199|
|correct3×4|0|0|

Broad training reduces the number of exactly familiar4×3 skeletons but does not turn any into the correct omitted combination. Both3×4 and trained4×3 have54 public tokens; the additional trained motifs have distinct other lengths. The pattern is consistent with a familiar-layout shortcut, but **does not establish that input length causes it**. No length counterfactual or gold-state reset was performed.

These are canonical-position diagnostics. A node-index permutation alone cannot explain the wrong predicate boundary and absent correct occurrence-tree skeleton; nevertheless, the diagnostic is not an unrestricted graph-isomorphism test. Removing entities is used only to describe layout, never to relax exact recovery.

## Invalid records and downstream errors

Held-out original:395 valid,397 retained EOS;115 invalid generated references and2 conflicting pair-slot outputs. Broad:403 valid,408 EOS;102 invalid references,5 conflicting pair-slot outputs and2 duplicate edges. All malformed terminal records remain in the analysis; no partial graph credit is given.

There are zero held-out cases with all nodes correct, so this population cannot isolate an edge-only defect after successful node construction. In contrast, acquired motifs have such subsets: broad5×4 has474 correct node outputs but437 complete graphs. Its37 node-correct failures are categorized as edge/slot failures; their aggregate differences include92 missing and89 extra typed/ordered edges, with2 missing edges also having a wrong-slot prediction for the same(source,target,relation). That slot diagnostic is per edge, not an assertion that only slots explain those examples.

`leading_nodes_correct` and `all_nodes_correct` remain distinct in the raw diagnostics: the latter disallows a retained extra NODE after an EDGE. Invalid prefixes, extra/missing nodes, missing/extra edges, duplicated edges and terminal conditions are independently visible in the compact per-event artifact.

## Supervision and interpretation

Teacher-forced gold-prefix field diagnostics are retained separately in the JSON, pooled across development. They are not per-cell free-running accuracy and cannot demonstrate deployment correctness. The actual complete-graph outcomes above use generated prefixes only.

This localizes the omitted-combination failure despite strong acquisition of newly trained layouts. It does not show universal failure of public-text learning, prove a particular causal shortcut, change the registered advancement result, or validate programmable attention. No new recipe is selected by this report.

## Artifacts and cost

`research/results/campaign-01/semantics/s21-localization.json` binds all input hashes and aggregate/paired counts. Adjacent `s21-localization-events.json.gz` retains7,168 compact paired per-example diagnostics. Reproduction uses `S21-localization.py` with the frozen config hash and closed archive paths; it fails closed on incomplete matrices or identity/summary errors.

CPU wall time **62.720234s** for the unchanged reviewed analysis; GPU occupancy0, failed attempts0. Historical artifacts untouched. Independent primary raw-score/codec auditing remains a separate responsibility.
