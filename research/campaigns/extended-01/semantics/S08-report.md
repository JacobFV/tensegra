# S08: alignment is acquired incompletely and does not guarantee ordered edges

Frozen S01 and S07 inference exactly reproduces all archived raw graph predictions on TRAIN128 and DEV512. There is no new training, threshold fitting, checkpoint selection or confirmation evaluation. Original lossless copy logits make this diagnostic independently reconstructable.

| DEV identifier occurrences (9,780 each) | First-identity position accuracy | Actual occurrence accuracy | Canonical identity accuracy | Mass on actual occurrence | Entropy (nats) |
|---|---:|---:|---:|---:|---:|
| S01 identity supervision | .94427 | .31881 | .94806 | .31778 | .18620 |
| S07 occurrence supervision | .31483 | .88937 | .91391 | .83874 | .33135 |

S01 was not trained to choose the actual occurrence, so its .31881 is not failure at its prescribed target. S07 learns a materially different pointer distribution, but does not acquire the target near perfectly. Its remaining canonical identity errors are also larger. TRAIN occurrence accuracy is .89192, close to DEV .88937; this is not a large observed train/dev alignment gap on the inspected subsets.

Among S07's 8,698 correctly aligned DEV ident occurrences, 6,848 required incoming ordered edges have both the edge and slot correct (78.73%). Among 1,082 incorrectly aligned occurrences, 431 do (39.83%). Correct alignment is associated with better edge retrieval, but is neither sufficient nor a causal intervention. These counts cover required gold incoming edges, not false-positive precision. Entity nodes have zero required incoming ordered edges and provide no support for that association.

The model therefore partially learns the supervised interface while graph reconstruction remains limited by multiple errors. This does not show that recurrence cannot preserve semantics, that occurrence labels are universally better, or that a further unspecified architecture is required. It separates acquisition of a particular auxiliary representation from acquisition of complete canonical relations.

The replay uses the original public-only actor at width1,024, fixed final SHA-guarded checkpoints and historical source/data hashes. Gold spans enter scoring only. TRAIN relation thresholds are reused for descriptive edge association, never fitted to DEV. Lossless `copy-logits.npz`, per-node records, metadata and counts are under `research/results/campaign-01/semantics/s08-alignment`.

Source `dec2d7b`, checkpoint/config freeze `d602b24`. Full occupancy26.94170 s; inner diagnostic25.81028 s; peak CUDA allocation959,284,224 bytes; process RSS2,792,436 KiB. The score domain is the inspected English unification corpus, with two known ordered trees and depth3. Independent audit requested.

## Exact node-renumbering check

A separate CPU-only diagnostic permits arbitrary node renumbering while preserving every node kind, relevant visible identity/value, directed relation label, ordered slot and relation multiplicity. All S01/S06/S07 final graphs remain non-equivalent on TRAIN128 and DEV512, for both raw and calibrated decoding. Every case is rejected already by necessary node-attribute or edge-label multiset invariants; no approximate matching or relaxed semantic criterion is used.

For DEV, S01 fails the node-attribute multiset in 390/512 cases and edge-label multiset in the remaining122; S06 splits504/8, S07 splits481/31. Thus canonical numbering alone is not concealing fully correct graphs in these inspected outputs. This does not prove that canonical-order supervision is the easiest way to learn them.

NetworkX3.6.1 was already installed. The bounded audit took33.09184 CPU seconds with no neural inference; a mechanical fixture verifies renumbering invariance and rejection of changed identities or ordered slots. Raw decisions are in `research/results/campaign-01/semantics/s08-isomorphism`. Canonical primary metrics remain unchanged.

Independent audit completed: reviewer `c538e8a` reconstructed S07 raw decisions/calibration and S08 lossless-logit statistics; `9196ec2` independently checked all3,840 exact-isomorphism rejections. No count discrepancies were found.
