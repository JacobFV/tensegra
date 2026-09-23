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
