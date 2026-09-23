# Stage 11 return acquisition horizons

## Question and supplied contract

Can the existing 1024-wide workspace and its six ordinary consumers jointly acquire reliable access across longer recurrent horizons? This is one timing-distribution intervention; architecture, persistent six-facet memory, encoders, cross-entropy objectives and optimizer recipe stay unchanged. There is no structural-attention intervention.

Start from immutable Stage 9 factorized/persistent backbones initialized with seeds 10, 11 and 12. Run IDs 30, 31 and 32 identify continuations, not new initialization seeds. Historical Adam state was not archived: both paired arms reset AdamW identically from frozen weights. Neither continuation is an uninterrupted historical trajectory.

## Registered comparison

Each arm receives 1,000 updates of 32 fresh public events, learning rate 0.0003, AdamW defaults, summed six-field cross entropy and gradient clipping at 1. The pair sees identical examples in identical update order. The repeating short schedule is `[0,1,2,4,2,4]`; the wide schedule is `[0,1,2,4,8,16]`. Zero and one exposure match exactly. Wide replaces duplicate weighting at two and four with eight and sixteen; this is not a uniform-delay or FLOP-matched comparison. Report presentations, unique events and actual recurrent microsteps separately. Full backpropagation is used, without truncating the long arm.

Unmodified Stage 9 heads are evaluated on the same fresh events. Saved Stage 10 shared-pool ridge heads are a contextual frozen-backbone reference, not an objective- or exposure-matched causal comparator. No decoder is newly fit to any evaluation data.

## Acquisition diagnostic and stopping rule

Before main training, use the independent Stage 8 seed-0 factorized/persistent checkpoint. Fit one fixed set of 32 events, at most 300 updates per arm. Evaluate at updates 0,60,120,180,240,300 on every delay covered by that arm. Both arms must reconstruct all six fields jointly on all 32 events at every covered delay by the last update. Fresh contexts are reported separately and are not this fixed-set gate. If either arm fails, stop the dependent main comparison and characterize acquisition; do not tune repeatedly or weaken this gate.

Profile batch-32 forward/backward/Adam at delays 0,4,16 and inference at 32 before committing the main compute estimate. No GPU is used until the coordinator releases it. The track ceiling is 75 GPU-minutes, not a target.

## Data and selection

All new data uses the 11-million seed namespace. Profile 11000001; fixed acquisition 11001001; fresh acquisition 11002001. Main training uses 11300000 + pair_index*10000 + update. Validation 11400001 and untouched test 11500001 have 512 events each. These partitions and inherited Stage 9/10 data are disjoint by namespace and event hashes. Generate each entire evaluation batch once, then slice, preserving the same events at all delays and distractor counts.

Evaluate delays 0,1,2,4,8,16,32 and distractors 2,8. Delay 16 is trained coverage for wide and extrapolation for short; 32 is genuine recurrent-length extrapolation for both. Zero is an ingestion regime and remains visible. No delay-32 training, selection or validation-driven recipe change is allowed. Main uses the final update-1000 weights, not a best test checkpoint. Curves through 16 use validation only; final 32 results are diagnostic and cannot select a model.

## Metrics, gates and provenance

Save predictions and targets for all six fields, event identities, exact half-unit scalar accuracy, signed/absolute errors, joint and non-value joint accuracy, and required-argument denominators. Retain historical sixteen-step thresholds: type and primitive >99%, exact scalar and required identities >98% in every prescribed seed/cell. Joint correctness is separately reported. A scalar gain cannot pass the full interface; no result authorizes composition or supervision withdrawal.

Save initial/final checkpoint hashes, source/config hashes, per-field losses, data batch hashes, actual optimizer presentations, recurrent cost, CUDA allocation, process RSS and runtime. Freeze source/config before main evaluation. Three paired seeds and at least 512 examples per main cell are mandatory. Same-event repeats across delays, models and distractors are correlated observations.
