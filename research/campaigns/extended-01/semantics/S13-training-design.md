# S13: controlled acquisition of a second rendered surface

Prepared before model inference; **GPU profile and main runs remain unreleased**. The CPU cache is independently audited separately. This is one development comparison, not confirmation and not held-out-language transfer.

## Hypothesis and intervention

An unchanged public-text actor that now acquires fresh English constructions can acquire the same semantic structures from Spanish when training includes that renderer. Compare English-only continuation with English/Spanish continuation. Both start from the immutable campaign S11 development seed201 endpoint at196,608 presentations, checkpoint SHA `3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e`. Restore model, AdamW moments, sampler state and graph visit counts; keep LR1e-5. No selection among S12 seeds, architecture, loss, copy representation, vocabulary, or graph schema changes.

The8192 construction sequence and sampled negative pairs are identical between arms. Mixed training alternates English/Spanish on successive visits to each construction, with a prespecified independently sampled initial phase (renderer seed913001). This RNG never advances the graph or negative-pair RNG. English-only uses exactly the same construction schedule. Four added epochs give each mixed-arm construction two English and two Spanish presentations. Report actual tokens: different rendered lengths mean equal presentations do not imply equal compute.

## Observable inputs and targets

The actor receives only the selected surface text. Renderer-specific copy labels and canonical graph labels are privileged training targets, never forward inputs. Surface-local first-occurrence copying and identity equality are unchanged. The pinned two-renderer cache preserves original8192 English examples exactly; the new512 DEV constructions are disjoint from historical TRAIN, DEV and S12 confirmation. Both surface versions of each construction remain in the same split. Only the inherited two tree motifs are represented; fresh equality patterns are not new structural motifs.

Use the exact frozen four-class value vocabulary and capacity128. Every step must reject unsupported cache hashes, altered initialization, unknown targets or slot conflicts. Preserve cache and source hashes and per-construction/per-renderer visits.

## Exposure, selection and stopping

Proposed initial main budget:4096 updates×8=32,768 added presentations per arm, endpoint229,376 total. Save fixed curves at added updates0,1024,2048,4096; all outcomes retained. Profile20 optimizer updates plus representative English/Spanish evaluation before root sets occupancy caps. A smaller profile is mechanical only and cannot select recipe.

At each checkpoint calibrate relation thresholds using the same English TRAIN128 examples only, independently for each model. Apply those exact thresholds to both English and Spanish DEV512; no Spanish or DEV threshold fitting. Raw decoding remains parallel. Primary exploratory endpoint: Spanish exact complete graphs with English-TRAIN calibrated decoding. Report English retention, raw complete graphs, exact copy, typed/ordered precision/recall/F1, complete accuracy by graph size, per-example paired surface outcomes and runtime. No policy is selected separately by renderer.

Advance to a separately frozen fresh confirmation only if the mixed endpoint reaches at least52/512 Spanish complete graphs, improves over English-only by at least26 graphs, and loses at most26 English graphs. These are development promotion criteria, not historical gate changes. If Spanish improvement from2048 to4096 is at least13 complete graphs and training losses remain finite, a single separately registered extra4096-update paired tranche may be considered; it is not automatically authorized. If training acquisition is absent or the curve saturates, diagnose component failures rather than continue exposure blindly. Missing/failed outcomes cannot pass. No S12 confirmation outcomes select this architecture or checkpoint.

The supplied schema decoder may be applied only as a separately reported frozen engineering reference after primary learned outputs are saved. No new exact bookkeeping enters the actor.

## Prepared implementation status

The guarded trainer and four `*-prepared.json` configs bind the completed cache audit/bytes and immutable S11 checkpoint. Prepared configs reject execution until a separate coordinator freeze. CPU checks cover the four-visit renderer balance and reject changes to parent, batch size, negative pairs, renderer seed, learning rate, exposure or checkpoint schedule; two tests pass in0.65s using CUDA hidden in an isolated temporary source copy. No model was constructed or forwarded by these tests. Runtime guards verify the inherited first construction/negative-pair batch and archive complete construction/pair sequence hashes, per-renderer visits, optimizer state steps and source provenance. Main endpoints must finish exactly4visits per construction (mixed2English/2Spanish). This is preparation, not a profile release; review and the complete S12 matrix precede any S13 GPU work.
