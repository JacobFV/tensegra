# Stage 4 journal

## Scope and design

Stage3 baseline is main8243cba, clean. The user asks to promote identity binding into an architectural object and defer interpretation/execution. This stage keeps data and old experiments unchanged and isolates matching normalization, identity-write protection, read isolation, explicit pointer transitions, and labeled auxiliary supervision.

The Stage3 comparison between .849 average pre-step correctness and .672 complete paths does suggest correlated outcomes. The large gap is strong descriptive evidence against independent homogeneous hop errors, but does not identify its cause: pre-step accuracy and full pre/post path correctness are different events, and graph/sample heterogeneity, cycles and recovery all matter. The stricter complete-path event does not weaken the qualitative dependence signal; it does require care when assigning a specific transition mechanism. New metrics retain per-example pre/post correctness sequences, first-error hazards and conditional persistence/recovery counts. Attractor language remains a hypothesis unless transition evidence supports it.

Raw-query cosine normalization and protected identity are distinct interventions. Pointer propagation Pq A then expected-key write is a deliberately stronger graph-computation prior; attention-key write preserves attention as the information-flow mechanism. The report must not pool these into a claim about logit-space structure.

Random projection and null initialization will be controlled separately. Supervised query/key/null targets affect training losses only, never forward-pass state writes or evaluation routing. Internal transition/rebinding consistency is not an inverse-graph operation and will be labeled precisely.

## Frozen main study launch

Executable source 3b63b88 passes all 260 tests in 2.61 seconds remotely. The six-variant end-to-end smoke completes. Main configuration fixes 23 variants × three seeds × 400 steps, with 20 final matrix cells and two initialization anchors per run. Evaluation examples remain 128/cell, batch16. All main training is remote with two CPU threads. The single unpinned resource benchmark was used only for runtime planning; its metrics are not pooled with the main results. Model checkpoints are retained remotely for post-training audits.

## Follow-up selection rule while main runs are incomplete

The optional adaptive experiment will use protected attention writes with identity-only grounding if that family passes the all-seed gate; otherwise it may use the explicit pointer-write family if that family passes. This preference follows the weaker computational prior, rather than ranking mean OOD accuracy. The partial seed-zero results are already visible, so this is a disclosed follow-up rule, not a retroactive preregistration of the main study.

A follow-up will compare fixed strength with an entropy-controlled coefficient using the same initialization, schedule and maximum strength. Both follow-up arms will freeze the base relation strengths at 4, so the only intervention is confidence modulation; the main study instead learns its nonadaptive strengths. The control will be retrained rather than reusing the main artifact. Confidence will be multiplied by real-entity probability mass so a confident null assignment cannot trigger graph crystallization. It will remain separate from the main69 runs, with its own source and resolved configuration. No implementation or training starts until all-seed stability is verified.

## Main gate and adaptive follow-up

All69 runs completed and independently audited. Only `cosine_pointer_identity` passes: D64/N128 task scores 1,1,.9921875 and complete trajectories 1,1,1. The attention-write alternative fails (task .9921875,.8515625,.9296875; paths 1,.8046875,.921875). Per the disclosed weaker-prior-first rule, the follow-up uses the pointer family. This is a scope limitation: confidence changes payload attention, while explicit Pq A identity writes remain unchanged. The experiment cannot establish confidence-driven emergence of stable binding.

Two fresh matched arms, `pointer_fixed4` and `pointer_adaptive4`, freeze all base relation strengths at4. Adaptive strength is4 times `(1-p_null)*(1-H(p)/log(N+1))`, clamped to[0,1]; no new learned parameters. Three seeds,400steps, identical schedules and the same20-cell evaluation matrix. This is a conditional follow-up on the established task, not an independent confirmatory test set. Main artifacts retain frozen source3b63b88.
