# S19: public text to sequential typed graph records

## Question and rationale

S18 fails its fixed acquisition, retention and recombination criteria. Even its fitted TRAIN panel contains only10/128 complete contextual outputs, with most remaining errors in edges. Close that workspace-family calibration/exposure search. S19 tests a different learned interface: can an ordinary contextual text encoder and causal, variable-length record decoder acquire complete graphs from the same public text? This is a strong alternative learner, not an isolated attribution to one layer or a new cognitive/runtime capability.

The preliminary record-contract audit finds84–125 target records for S15 TRAIN. The full scored graph, including directed relation labels and ordered slots, is predicted. No term-to-graph compiler, public-text parser, supplied node counts, gold alignment, graph schema expansion or exact equality solver may provide inference targets.

## Supplied and learned

Supplied: existing public tokenizer/features and copy inventory; finite node/relation/value vocabularies; canonical training serialization; typed NODE/EDGE/EOS syntax; capacity128nodes and maximum160records; exact conversion of a selected visible token to the historical copy identity. Learned: contextual token states, node creation/stopping, node kinds/values/copy addresses, ordered source/target references, relation and slot labels. Graph targets and previous gold records are privileged training supervision only. Free-running inference consumes its own preceding records and public text.

Use width1024, two encoder layers and two causal decoder layers, eight heads, FFN width4096, dropout0. No pretrained weights, new primitive family or changes to archived actors. Incremental cached decoding must match teacher-forced causal logits on the same prefix. Separate record-field heads are an explicit within-record factorization. The current gold record must never enter its prediction.

## Data and comparisons

Reuse the frozen S15 mixed4096-construction TRAIN corpus and its existing four DEV512 cells. These are inspected development populations; no fresh confirmation is opened. The primary fresh-known-renderer conditions are trained3×3,4×3,4×4;3×4 remains an untrained composition diagnostic. Use the same existing frozen value vocabulary and public copy convention. Conversion must round-trip all TRAIN targets and independently validate malformed-output behavior before training.

Compare with the complete frozen S18 original, contextual and10pass endpoints, preserving raw and matched-calibration scores. S19 has categorical greedy decoding without an edge-threshold search. It starts from scratch; the earlier actors have196,608 inherited presentations before their32,768 mixed presentations. Parameters, total and incremental presentations, decoder-record exposures, arithmetic and latency must all be reported. This is not a parameter/FLOP/pretraining-matched causal architecture comparison.

## Fixed development recipe and decisions

Development seed1901; batch8;4,096updates/32,768presentations/eight visits. Reset a declared seeded corpus permutation schedule, independent of labels and evaluation. AdamW LR3e-4 with128-update linear warmup and cosine decay to3e-5, betas(.9,.999), weight decay.01, gradient norm cap1; BF16 forward with floating-point cross-entropy. Average losses by applicable field and record, log every field independently, and disclose exact loss aggregation before freeze. No calibration or best-checkpoint selection.

Evaluate fixed endpoints0/1024/2048/4096 on a predeclared128-example actual-TRAIN panel and all four DEV512 cells. Record teacher-forced field accuracy/loss separately from free-running complete graphs. Final4096 determines promotion: at least50% complete TRAIN graphs, and at least52/512 complete graphs in **each** trained-motif fresh cell. Report paired outcomes against all frozen baselines, without turning a threshold pass into an architectural-isolation claim. Heldout composition is separate and requires52/512 before any positive recombination statement.

One exposure extension may be registered after the main, but is not automatic: it requires TRAIN complete≥50%, weighted fresh trained-mixture complete≥5%, and an increase of at least3 percentage points in that same weighted mixture from2048 to4096. It must retain the first result, declare the additional schedule/budget before new outcomes, and use no confirmation for selection. Otherwise inspect teacher-forced versus free-running errors and stop or register a different discriminating test. Do not keep extending to obtain a favorable endpoint.

## Correctness, outputs and budget

Invalid references, duplicate/conflicting records, output overflow, missing EOS and malformed phase order are explicit failures; old metrics must not hide edges to absent nodes. Preserve generated records and packed canonical predictions, targets, per-field errors and validity reasons. Metadata excluded from historical scored tensors must not be described as recovered compiler provenance. A meaningful no-text/frequency diagnostic must disclose retained length/copy information; unchanged strong neural baselines remain primary context.

Before any GPU release: independent codec/public-boundary review, causal/cache/gradient/padding tests, parameter count, source/config/data hashes and representative profiling of training **and worst-case free-running evaluation/export**. Proposed profile ceiling180seconds; the main budget is unset until measured. Planning envelope for this development branch is at most3,600GPU seconds including profiling and any separately registered extension; a profile showing that meaningful full evaluation cannot fit triggers redesign or deferral, not a silent width/data reduction. Root retains sole scheduling authority and approximately8,640seconds for confirmation within the remaining campaign budget.

If promotion is earned, freeze the selected recipe and run at least three independent initialization seeds with all fresh required confirmation cells≥512 and matched alternatives where practical. Existing S15 reserved confirmation remains untouched until then. No current S19 competence, timing or generalization result exists.
