# RL01 independent archived-metric audit

Scope: one inherited S21 broad parent, development continuation, supplied exact NODE prefix. No new inference, public-text node acquisition, structural-attention comparison, or independent-seed replication.

`audit_rl01.py` reconstructs complete/component graph correctness directly from generated canonical records and bit-packed targets; it separately recomputes teacher-forced tuple equality. All **6,912 rows across 18 cells** match saved summaries and manifests. Repeated checkpoints/arms share 128 TRAIN-panel, 512 known, and 512 omitted constructions; 6,912 is not independent support.

| Arm / update | TRAIN complete /128 | Known complete /512 | Omitted complete /512 | Omitted valid /512 | Local EDGE correct /35,396 |
|---|---:|---:|---:|---:|---:|
| Both /0 |128|496|0|352|27,597|
| all_records /256 |127|489|0|374|27,600|
| relation_only /256 |128|488|0|280|27,636|
| all_records /1024 |127|484|0|389|27,535|
| relation_only /1024 |128|485|0|278|27,549|

Final known paired outcomes (all-records → relation-only): 474 correct→correct, 10 correct→wrong, 11 wrong→correct, 17 wrong→wrong. The one-example marginal advantage is not compelling evidence of improved generalization. Every omitted example remains wrong in both arms.

The relation-only endpoint's 234 invalid omitted generations comprise 224 invalid node references, seven duplicate edges, and three unsupported multiple-slot labels on one node pair. All-records has 123 invalid omitted generations: 122 invalid node references and one duplicate. These are recorded decoder/codec rejection categories, not necessarily distinct underlying learned mechanisms. Valid omitted generations also all fail complete edge and slot correctness: removing invalid outputs alone would not fix the result.

Local teacher-forced tuple scores condition on gold preceding records. Near-78% omitted EDGE accuracy therefore does not establish a successful generated trajectory. Neither fixed-endpoint arm improves local omitted accuracy over initialization; no registered promotion signal appears. Stop this recipe as planned rather than attributing failure to missing node identities or starting an unregistered architecture search.

## Provenance and comparability

- All 13 frozen source hashes match current root source bytes. Every raw artifact hash matches its curve manifest; arm manifests match the enclosing manifest.
- Each arm starts at tensor hash `2d1267174e4f7fbd93ebd77da36bec0748e74f5c412cc0144b56cf64925d683e`, bound to archived parent file hash `a5ad0f6d4fba9804a0ae3ed1ef8dd7b399ed2037dc89acd7b28c07f3bfc56af8`.
- Update-zero raw rows are identical across arms. Optimizers are recorded reset. Each width-1024 model has 62,677,315 parameters.
- Both recorded construction-stream hashes match; visits show exactly eight presentations of each of 1,024 TRAIN constructions (8,192 presentations/arm). The minibatch schedule itself is not re-executed here.
- Frozen input hashes agree with the generation manifest. Evaluation alpha identities are stable across arms/checkpoints and disjoint across TRAIN-panel, known, and omitted sets.
- Full remote training caches and checkpoint tensors are not local: this audit does **not** newly verify their bytes or independently reconstruct the historical exclusion inventory. Those guarantees retain the generator/runner receipts and their limitations.
- Invalid rows retain the controller's recorded termination/rejection status; valid graph scores are independently reconstructed. Public copy canonicalization is not rerun without the public text cache. This is archived-score verification, not rerun inference.

Audit CPU: **20.079648048 seconds**, no children or GPU. Campaign training/inference occupation is separately charged by the coordinator (253.7447 GPU-process seconds; 287.0622 CPU seconds), not inferred from this audit. Raw compact audit JSON preserves every cell, rejection count, source agreement and paired endpoint count.
