# S13 preparation: paired multilingual continuation data

**Status: prepared, not run.** No bulk cache generation, trained-model inference, or GPU allocation is authorized by these files. The configuration deliberately rejects generation until the coordinator versions an explicit release. Advance only after S12 known-renderer confirmation supports a useful next comparison.

## Next controlled question

Warm-start both arms from the same immutable **campaign S11 development seed201 final checkpoint**, binding its exact weights/AdamW/sampler state before execution. This is not Historical Stage11. Compare English-only continuation with balanced English/Spanish continuation at matched construction order and presentation count. Keep actor width1024, architecture, objective, finite output vocabulary, and optimization recipe equal. Profile before setting an exposure budget; no exposure or checkpoint is selected in this preparation.

Minimal change: which visible rendering of the same TRAIN graph supplies each presentation. Training never sees new DEV or old confirmation constructions. Multiple surfaces are not additional independent semantic graphs. Report English exposure reduction, Spanish exposure, optimizer tokens/actual compute, graph visits, and same-graph/both-surface correctness. Use one unchanged **English TRAIN128 calibration policy per model**, applied unchanged to both languages; retain raw scores. This avoids per-renderer threshold tuning and does not give the English-only comparator Spanish calibration labels. Supplied S10 grammar results stay secondary.

## Prepared cache contract

`campaign_semantics_multisurface_data.py`, version `campaign-multisurface-v1`, groups both views in one record per construction. It preserves all8,192 existing allowed TRAIN constructions exactly. Every seed is regenerated through pinned TCN018c9ce9 / difficulty.5; original English text, graph/semantic/public hashes, node/edge arrays and token count must equal the historical cache. Original English target tensors must match exactly. Spanish targets use Spanish visible first-copy positions. All non-copy target tensors must remain identical across surfaces.

The raw Term is reproducibly recoverable from each stored generator seed through the pinned constructor. No cached raw Term is required; the new builder verifies regeneration before accepting a row. Existing source supports English/Spanish/symbols, but S13 declares only English/Spanish. No generator semantics are extended.

Fresh DEV512 begins at a separately registered namespace910200001. Candidate alpha-semantic keys are excluded against **all** old TRAIN8192, DEV512 and reserved-confirmation1024 keys, as well as earlier accepted newDEV keys. Exclusion reads semantic-key metadata only; old reserved labels/outcomes are never used for selection. All old file byte hashes are pinned. Declared maximum100,000 attempts fails closed if insufficient support. New confirmation is reserved later; no old confirmation partition is converted to new development. Every renderer of a construction stays with that construction.

## Reject rather than hide bad targets

For every actual surface:

* Reject invisible identities or overlapping token support for distinct identities (including canonical `red`/`rojo` collapsing in Spanish).
* Reject copy targets inconsistent with the surface's first occurrence, and reject identity labels placed in the finite value channel.
* Reject identical actor text with incompatible representable targets.
* Independently reject incompatible targets with identical **actual FP32 or BF16 token-feature sequences**, including whitespace-equivalent inputs.
* Reject unknown finite values, capacity/slot overflow, truncation, compiler-target mismatch, or altered non-copy semantics.

A contract failure stops cache generation with its error; it does not silently drop/remap/repair the construction. Only already-declared duplicate alpha constructions are skipped. The new validator is dataset-only, never a deployed bilingual lookup. `public_view` passes only the selected visible text in ActorInput; tests poison hidden graph metadata and verify the public projection is unchanged. Surface-local pointers are not persistent cross-surface English IDs.

## Mechanical evidence and pending work

Four CPU cache-preparation tests pass: exact original English regeneration/target parity, Spanish copy position change, translated identity collision rejection, hidden-metadata-independent public projection, actual token-feature collision rejection, and denied unapproved generation. Three separate surface-contract tests pass. These tiny fixtures do not establish full-corpus validity or learned multilingual generalization. Full corpus audit, profile, immutable parent binding, training protocol, and fresh confirmation remain pending.

No historical source/cache bytes changed. No reserved cache was read while preparing or testing these helpers. A future released build will read only its stored semantic-key metadata for exclusion, as stated above. The prospective configuration and source must be reviewed before that build.
