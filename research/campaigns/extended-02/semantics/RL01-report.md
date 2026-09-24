# RL01: relation-only continuation does not repair omitted composition

**The registered promotion criterion fails.** With correct canonical nodes supplied, neither continuation objective recovers any omitted 3×4 graph: **0/512 at updates 0, 256, and 1024**. Redirecting supervised loss from node records toward relations/EOS does not repair this boundary. No further semantic training or confirmation is authorized by this result; the structured-observation agentic track is independent.

## Supplied versus learned

This exploratory paired comparison has **one inherited lineage**, the S21 broad checkpoint (historical seed2101). Two identical copies receive the same 1024 fresh training constructions and 8192 presentations each. Width remains1024 and each model has62,677,315 parameters. AdamW is reset; both receive1024 additional updates, batch8, constant3e-5 learning rate. The relation-only arm zeroes node-field losses and supervises type only at EDGE/EOS; the historical denominator8 is retained. The actor, teacher forcing, representation, and compute path are unchanged.

Evaluation supplies the complete correct NODE prefix, then predicts edges, ordered slots, and termination. It receives no gold edge count, edges, or stopping time. These are **privileged node-boundary diagnostics**, not public-only complete semantic induction, learned scheduling, or attention-bias experiments. Teacher-forced local metrics additionally receive previous correct edge records and therefore are not free-trajectory performance.

The fresh TRAIN1024 covers six known motifs; known DEV512 uses the same motif mixture, and omitted DEV512 contains only3×4. Alpha-equivalent historical constructions and all S21 train/development/confirmation rows were excluded. All sets are development data. The deterministic TRAIN128 panel is training-exposed, not a generalization estimate.

## Acquisition and generalization curves

| Objective / update | TRAIN128 exact | Known512 exact | Omitted512 exact | Omitted local ordered edges correct /35,396 |
|---|---:|---:|---:|---:|
| Common parent /0 |128|496|0|27,597|
| All records /256 |127|489|0|27,600|
| Relation only /256 |128|488|0|27,636|
| All records /1024 |127|484|0|27,535|
| Relation only /1024 |128|485|0|27,549|

At the fixed final endpoint the paired known-event table is474 correct in both arms,10 all-record-only,11 relation-only, and17 wrong in both. The extra14 correct local tuples for relation-only represent only0.040 percentage points. Both final local totals are below their common initialization. The prespecified requirements of at least52 omitted complete graphs and an improvement of26 over the comparator both fail. Known retention does not rescue that failed conjunction.

## Where the omitted predictions fail

Teacher forcing makes ordinary containment and declaration edges nearly perfect but does not repair the cross-layout bindings:

| Gold relation | Support | All-record exact tuples | Relation-only exact tuples |
|---|---:|---:|---:|
| contains |12,288|12,275|12,288|
| declares |3,140|3,139|3,140|
| field:facts |512|509|509|
| field:pattern |512|3|3|
| field:query |512|4|8|
| item |2,048|662|716|
| argument |8,192|5,552|5,439|
| refers_to |8,192|5,391|5,446|

For `field:pattern`, all509 errors in each arm are **target-node errors**; source, relation label, and slot are correct. For `field:query`, target errors account for508/504 cases. Other families include substantial role/slot and source errors. These are narrow observed error locations, not proof of a particular positional-shortcut mechanism.

The five teacher-forced field counts (tag/source/target/role/slot) are **35,393/32,381/30,184/29,907/29,896** for all-record and **35,394/32,354/30,307/29,819/29,879** for relation-only, each with denominator35,396. Fields are correlated; their probabilities must not be multiplied to estimate complete graphs.

**Premature EOS is not the main local failure.** At actual EDGE positions, teacher-forced EOS is predicted only3 times versus2; at the gold EOS position both arms are correct509/512. Free decoding can still compound mistakes: of389 versus281 outputs ending in EOS,29 versus48 emit fewer edges than the gold graph, while253 versus162 emit more. This length comparison is a descriptive termination diagnostic, not evidence that every missing relation is caused by halting.

Strict valid outputs decline from389/512 in the all-record arm to278/512 in relation-only. The other outputs are122 versus224 out-of-range generated node references,1 versus7 duplicate-edge failures, and0 versus3 outputs with pair-slot multiplicity unsupported by the historical scoring representation. The correct supplied nodes do not prevent the learned edge generator from referencing an invalid node address. Partial-edge totals include invalid prefixes solely for localization and are not deployable graph scores.

## Cost, reconstruction, and interpretation

Main runner wall time was251.844s and CPU time256.349s; profile runner wall time22.183s and CPU time21.907s. Outer-process occupancy is separately recorded in the campaign ledger and includes launch overhead; these internal durations do not replace it. Peak allocated CUDA memory was1,381,603,840 bytes and process maximum RSS4,136,472KiB. Both arms visited every training construction exactly8 times using identical streams.

The [CPU localization receipt](RL01-localization.json), reproduced by [the standalone script](RL01-localization.py), reconstructs summary counts and all local tuple decisions across6,912 evaluated graph records, checks matched event pairing, and independently decodes the packed gold edge tensors for all1,024 final omitted outputs. Its measured analysis CPU cost is12.313s. This is archived-prediction reconstruction, **not checkpoint inference**. It retains the original strict codec validity decision rather than pretending to reimplement the entire codec. Input artifact hashes are included. Raw predictions remain in `research/results/campaign-02/rl01-main-v1`.

This test rejects one bounded repair—removing node-record loss while continuing on known motifs—under this model, parent, data, and budget. It does not establish impossibility, an information-capacity limit, or universal failure of relation learning. Exact nodes and mostly correct local syntax coexist with near-total failure on several semantically important cross-layout relations. Further work would need an independently motivated relation/address representation or exposure intervention, rather than another automatic extension of this loss-allocation recipe.
