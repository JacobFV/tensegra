# S13 multilingual cache: completed CPU-only preparation

**Data contract passed; no model competence tested.** The coordinator released only bounded CPU generation after source review and a fixed64-TRAIN profile. No trained model was loaded, no gradient step or GPU inference occurred, and S12 confirmation work remained independent.

| Quantity | Result |
|---|---:|
| Allowed TRAIN constructions preserved | 8,192 / 8,192 |
| Exact original English text/graph/target parity | 8,192 / 8,192 |
| New alpha-distinct DEV constructions | 512 |
| New DEV attempts / duplicate keys skipped | 532 / 20 |
| Views per construction | 2: English and Spanish |
| TRAIN / DEV surface presentations available | 16,384 / 1,024 |
| Actual optimizer presentations | **0** |
| Distinct full FP32 / BF16 feature sequences | 17,408 / 17,408 |
| Identity collisions, uncopyable targets, unknown values, slot conflicts | 0 accepted violations; any violation would abort |

The20 skips are declared semantic duplicate/old-population exclusions, not failed-contract examples. No allowed TRAIN construction was dropped. New DEV keys are disjoint from all old8,192TRAIN,512DEV,1,024reserved-confirmation keys and each other. Only old semantic-key metadata was used for exclusion; old reserved targets/outcomes never guided new example selection. New confirmation has not been generated or selected.

Every graph's English/Spanish views stay together. The two views share non-copy targets exactly, while copy indices are computed from each surface. Full output identity remains a surface-local pointer/equality contract, not a hidden-English-name export. Both text and actual FP32/BF16 input-feature collisions are checked against full representable targets. The shared-pair slot guard runs **before** target writes, permits multiple compatible relation labels, and rejects conflicting ordered/unordered or different ordered slots. Historical data and frozen S10 rules remain unchanged.

## Immutable artifacts

Remote dataset: `/home/brandonin/topoformer-campaign01-semantics/data/s13-multisurface-v1`. Exact compressed TRAIN/DEV bytes and audit are also committed under `research/results/campaign-01/semantics/s13-multisurface-cache`.

* TRAIN SHA256: `3c19320cc6c41b23639c07d396a92b81b69d5b6d8ee6eafdf171e56cf37ed4fb`
* DEV SHA256: `87e2be4290b26a174875491d6a15534529e49c2900177ccae0f8f458f48add22`
* Source/config freeze: `9939d22`; slot guard: `8b4230a`; pinned generator: `018c9ce9286fa4961292fe1e024b49fcd7e2dd7f`.

Profile64 costs1.82CPUwallseconds and predicts~119seconds for8,704records before metadata/duplicates. Full build costs129.76CPUwallseconds, maximumRSS912,432KiB. Total131.58CPUseconds, zeroGPU; cap300combined seconds respected. Five cache tests and three identity-contract tests pass. Source preflight cleared; independent full-cache reconstruction is requested.

## Next decision remains separate

This dataset supports the prospective single English-only versus English/Spanish continuation comparison from the same campaign-S11development checkpoint, with matched construction order/exposure and one English-TRAIN calibration policy per model applied to both languages. It does not authorize training before S12's known-renderer outcome supports advancement. No block-order, symbol-renderer, new structural motif, or composition axis was added. Spanish would be a trained renderer, not held-out-renderer transfer. The family remains inherited difficulty.5 unification with its limited ordered-tree motifs.
