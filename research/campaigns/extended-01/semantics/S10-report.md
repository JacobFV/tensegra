# S10: public compiler contracts recover some complete graphs

Status: exploratory frozen-output diagnostic; independent audit passed (`c3a01f8`), all 15,360 records reconstructed independently including exact output hashes and metrics. Source `2085053` (protocol `a5a0ca0`). No training, GPU, score refit, threshold selection, new inference, or confirmation data. Frozen actor width remains 1024.

## Observation

The combined supplied schema/bookkeeping decoder recovers **100/512 (19.53%)** complete canonical DEV graphs from frozen S03 calibrated predictions, versus zero originally. The same combination recovers 39/512 using raw thresholds. Bookkeeping alone recovers 33/512 calibrated; schema masking alone still recovers zero. S01 and S04 also improve, but S04 remains worse than S03; no endpoint is retrospectively replaced.

These are real fresh-construction events relative to the original training split, but they have already been inspected during this campaign. They are development diagnostics, not an untouched generalization confirmation or a three-seed claim. One initialization (201), three training-exposure checkpoints.

## Complete results

| Snapshot | Split | Policy | Baseline | Schema | Bookkeeping | Combined | Combined typed F1 | Combined ordered F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S01 | train | raw | 0/128 | 0 | 0 | 0 | 0.84177 | 0.70801 |
| S01 | train | calibrated | 0/128 | 0 | 0 | 5 | 0.91808 | 0.87885 |
| S01 | development | raw | 0/512 | 0 | 0 | 0 | 0.84455 | 0.70675 |
| S01 | development | calibrated | 0/512 | 0 | 2 | 33 | 0.92264 | 0.88152 |
| S03 | train | raw | 0/128 | 0 | 0 | 8 | 0.95045 | 0.89152 |
| S03 | train | calibrated | 0/128 | 0 | 6 | 30 | 0.97240 | 0.95183 |
| S03 | development | raw | 0/512 | 0 | 2 | 39 | 0.94802 | 0.89342 |
| S03 | development | calibrated | 0/512 | 0 | 33 | 100 | 0.96829 | 0.94954 |
| S04 | train | raw | 0/128 | 0 | 0 | 0 | 0.93632 | 0.85837 |
| S04 | train | calibrated | 0/128 | 0 | 0 | 1 | 0.95912 | 0.91689 |
| S04 | development | raw | 0/512 | 0 | 1 | 3 | 0.93598 | 0.86059 |
| S04 | development | calibrated | 0/512 | 0 | 0 | 19 | 0.95806 | 0.91674 |

All baseline complete-graph counts are zero, hence each newly exact graph is a paired repair and there are no complete-graph regressions. Component errors can still be introduced. For calibrated DEV combined decoding:

| Snapshot | Removed edge errors | Introduced edge errors | Missing predicted reference matches | Identifiers with multiple predicted entity matches | Scope abstentions |
|---|---:|---:|---:|---:|---:|
| S01 | 7617 | 1417 | 685 | 384 | 0 |
| S03 | 6764 | 702 | 338 | 172 | 0 |
| S04 | 5453 | 542 | 302 | 136 | 1 |

## Supplied versus learned

**Supplied:** compiler-wide type admissibility; exactly-one-predicted-scope construction of contains/declares; S05 exact equality of predicted public copy identities for ident→entity references; unordered bookkeeping slots. These provide known output grammar/bookkeeping, not learned semantic induction. They are specific to this compiler’s construction-scoped entity table. No general lexical-scope or pronoun-resolution claim.

**Still learned/frozen:** node presence/type/value/copy, actual argument/item/field associations and their order, binding associations, and all original thresholds. Schema masking only removes invalid associations; it never selects a gold-compatible association. Duplicate predicted entities retain the original S05 all-matching-pairs policy; no gold tie break. Missing/multiple scopes cause bookkeeping abstention, not a supplied gold scope. S04 has one such DEV graph.

The combined intervention introduces some edge errors because predicted node types and copied identities can be wrong. It does not guarantee a correct graph. The calibrated S03 result still fails on 412/512 events. There is no structural-attention intervention here.

## Contract and mechanical verification

Twelve hand-built compiler fixtures cover every compiler branch (including bind), and all 8,704 cached TRAIN/DEV graphs satisfy the included coarse constraints: 8,716 graphs, 673,633 edges. No difficulty-specific fixed node count, tree shape, position, or root rule is used. Gold is read only by the separate audit/scorer. Five CPU unit tests pass, covering target-free decoder signature, no mutation, permutation/identity-renaming equivariance, duplicate reference policy, ambiguous-scope abstention, and structural-slot preservation.

Primary metrics use the historical active-node masking and exact canonical slot/type/copy conventions. The raw per-row `shared_slot_conflicts`, `edge_flip_count`, and `slot_change_count` describe complete stored tensors, including padded nodes; they are **not** active semantic error counts. Removed/introduced errors use active-node metrics. Exact tensor hashes plus immutable archive hashes and frozen deterministic decoder reconstruct every output; redundant padded edge lists are deliberately omitted.

## Compute and export correction

Completed compact run: 58.38 seconds CPU wall, internal audit/evaluation/export 57.54 seconds, maximum RSS 1,020,292 KiB, zero GPU occupancy. Initial identical-policy run was stopped while attempting an unnecessarily large padded-edge-list export before any outcome inspection. Exact occupancy was not captured; charge its declared 300-second timeout ceiling conservatively. Total CPU wall charge 358.38 seconds, plus five mechanical tests (0.007 seconds test body). That failed export and source remain recorded; the repair changed serialization only.

## Decision

This justifies a separately registered fresh, paired confirmation or a subsequent explicitly grammar-constrained model comparison if the coordinator prioritizes it. It does not pass historical semantic transfer gates, does not license composition, and does not establish an attention advantage. Existing S01/S03/S04 predictions and metrics remain immutable.
