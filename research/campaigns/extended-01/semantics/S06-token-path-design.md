# S06 prospective token-state path

Prepare this single alternative while S04 finishes; no launch is authorized by this file. S04's slowing component acquisition and persistent completegraph0 motivate testing public-text access rather than simply increasing exposure indefinitely. The final S04 outcome determines advancement.

## One representational-path intervention

Reuse the existing feature projection, four distinct recurrent blocks repeated twice, canonical node queries, copy head, typed-edge head and additive edge-conditional slot head. Replace the eight learned workspace rows with an evolving state for each public token (observed54/64tokens). Node queries attend the contextualized token states directly. Recurrent residuals remainFP32 under the original BF16 dense-operation policy; no precision change is intended. Original public projected memory remains available to the block cross-attention. No additional layers, runtime, graph input, history cache, pretrained model or supplied node identity is introduced.

Padded public tokens are masked as self-attention keys and decoder keys; padded residual states are zeroed after every sublayer block. Padding never creates actor-visible nodes. The original eight initial rows remain in the state dict for identical shared initialization but are frozen/unused, an explicit8,192inactive parameters. Total57,853,781; expectedactive57,845,589. Width remains1024. Report actual state lengths, parameter allocation and measured compute;54/64state rows versus8 is substantially more attention/linear compute, not an equal-FLOPs comparison.

The mechanism follows per-token recurrent refinement as in the [Universal Transformer](https://arxiv.org/html/1807.03819v3); it does not invoke its theoretical computation claims or adaptive halting. Existing role-specific edge scoring remains unchanged, rather than adding a second intervention from [deep biaffine parsing](https://arxiv.org/html/1611.01734v3).

## Proposed bounded development comparison

Keep the same8,192TRAIN/512DEV constructions, English surfaces/copy inventory, value vocabulary, seed201 shared initialization, AdamW1e-4, batch8, original node/identity curriculum and sampled balanced-edge/edge-conditional-slot objective. Gold pairs select loss rows only after actor computation. Calibrate on the same first128TRAIN predicted-present pairs with the same relation-threshold rule. No fresh evaluation thresholds or reserved confirmation predictions.

Preferred endpoint65,536presentations, checkpoints0/8,192/32,768/65,536presentations, matching S01-N8192. One variant only; no frontend/depth/width sweep. Raw/calibrated completegraph and components are all reported. Primary comparator is the historical current actor at identical presentations; wall time/token allocation differences remain explicit. Successful development motivates a separately registered paired confirmation, not immediate claims. No programmatically derived S05edges enter this training arm or its primary score.

## Profile and stopping

Before freezing any main budget, run20representative updates at actualwidth1024, batch8, on existing data with16TRAIN/16DEV evaluation records. This profile is mechanical, not recipe selection. Measure optimizer time, full occupancy, peakCUDAallocated and processRSS; extrapolate complete training/evaluation/export cost. Provisional main config refuses execution until its budget_status is explicitly frozen. If cost exceeds the available track allocation, ask coordinator for a budget decision; do not silently shrink width/exposure or change the comparison. Root GPUrelease required separately for profile and main.

CPU width16 fixtures test padded-batch invariance, dense/sample edge consistency and identical common initialization. Source/config/provenance must freeze before any GPU profile. Save failed recipes and all prescribed checkpoints; do not select an early checkpoint as the main endpoint.

## Public positional-feature audit (before GPU profile)

The inherited `encode_public` already appends `[i/len(tokens), sin(i), cos(i), 0]` to each64-bit lexical vector. S06 preserves these public features unchanged. On all8,704alpha-distinct TRAIN+DEV constructions, sorted full-row feature multisets are unique in bothFP32 andBF16 input rounding (zero canonical-target collisions). The normalized position coordinate alone retains54/54 or64/64distinct positions afterBF16 rounding. Lexical-only bags have8,230unique multisets and474collisions between different canonical constructions, confirming that ignoring order would lose relevant distinctions. No positional encoding was added after inspection.

Audit source and encoder hashes, collision examples and8.327CPU seconds are retained in `research/results/campaign-01/semantics/s06-feature-audit.json`. This checks finite raw-feature observability, not whether learned projections/attention preserve or use position correctly; reserved confirmation predictions were untouched. The dataset's existing alpha-disjoint audit means those lexical-bag collisions are not merely renamed copies of one semantic graph.

## Measured profile and frozen main budget

Source d6cb503 profile completed20updates at width1024/batch8 with54/64token states. Full external occupancy10.81017s; optimizer1.63075s including firstupdate.33358s. Remaining19updates average.06827s. Historical current-actor32step profile averages.06430s after its firstupdate at the same width/batch/precision, but uses a different minibatch sequence and corpus prefix: this is an approximate geometry-matched timing reference, not same-input or FLOP-normalized evidence. Exact initial parameter hashes match. More token-state arithmetic need not cause proportional wall time on this hardware.

Actual parameters57,853,781, active57,845,589. PeakCUDAallocated1,225,811,968bytes; processRSS2,950,680KiB. These are separate measurements; `nvidia-smi` reports total device memory asN/A for this GB10 rather than either figure being total capacity. Profile raw/receipt/source hashes: `research/results/campaign-01/semantics/s06-token-profile/s06-token-profile.occupancy.json` and manifest.

Measured steady optimizer time projects~560s for8,192updates, with170–220sestimated data/evaluation/checkpoint overhead. Freeze `configs/campaign-s06-token.json`: seed201,8,192TRAIN/512DEV,65,536presentations, declared0/1,024/4,096/8,192update evaluations, cap900s full occupancy. Same S01-N8192 data/loss/curriculum/threshold policy; one token-state intervention. No further profile needed. Root budget accepted; launch awaits explicit queue release. Provisional config remains as planning history and still refuses main execution.

### Frozen post-training contract diagnostic

Before the final S06 outcome is inspected, apply the unchanged S05 predicted-presence/kind/copy equality rule to S06's final archive, using `campaign-s06-identity-contract-diagnostic.json`. This is CPU-only and changes only the `refers_to` edge plane. Report the primary learned raw/calibrated canonical metrics unchanged; the programmed rule and gold-reference ceiling are separate diagnostic columns. No thresholds or rule parameters are selected from S06 outcomes. This tests the scoped compiler identity contract over learned public copying, not learned reference inference or independent confirmation.
