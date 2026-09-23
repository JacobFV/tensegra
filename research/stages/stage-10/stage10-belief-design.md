# Stage 10 belief interface: observation handles versus semantic content

## Hypothesis and limits

Stage 9 located a real coverage hole: IDs entered the learned compatibility input even though they name ledger entries, not semantic content. ID4 occurred only with contradictory primitive values, and ordinary IDs were tied to arrival order. Here we test two small repairs: enforce the role separation architecturally, or teach the raw-ID matcher with independently assigned identities. This is a closed-candidate, noiseless constraint benchmark, not language understanding, structural attention, or runtime composition. Width remains 1024 with four residual phases and FF width2048.

## Prespecified arms

1. `ledger_only`: observation IDs still address the exact ledger, but the sixteen ID coordinates entering the content encoder are zeroed. All allocated parameters match the raw-ID arms; the unused ID weights are disclosed. Content compatibility, nonempty null inference and candidate posterior remain learned. Consistent bijective ID renaming within the public unsigned16-bit handle domain is an architectural guarantee, not an acquired behavior.
2. `raw_randomized`: the content encoder receives raw16-bit IDs, while every training episode receives a fresh independent random bijection over its observation handles. Duplicate and retraction references reuse the same mapped handle. The ID allocator uses an independent RNG; it does not use a winner, target support, primitive value, role or arrival position.
3. `raw_correlated`: the unchanged Stage8 ID/content pairing is retained as a fresh-training reference. This is not a reused historical checkpoint.

All arms use the exact public empty-ledger prior, including after complete retraction. This is supplied uniformly and is not learned prior calibration. All arms retain exact first-write ledger idempotence and removal by identity. There is no new null head and no runtime execution. A finite masking logit implements exact float32 zero probability while keeping supervised loss finite.

## Information, supervision, pairing

Public input is the same typed candidate records, role, action, event content and observation handles. Privileged targets are the exact uniform-on-compatible-support posterior and per-event compatibility labels; they enter losses and evaluation only. Independent equal-content observations remain deterministic repeated constraints, not independent noisy likelihoods.

All arms start from identical complete parameter tensors for each paired seed. They see the same semantic episodes and condition schedule; only the declared neural ID access and training handle assignment differ. Allocated parameter count is expected16,867,395, with16,384 encoder weights structurally inactive in `ledger_only`. Equal parameter count does not imply equal useful input capacity. No attention memory tokens are introduced.

## Development, freeze and budget

Development seed101 only, separate training namespace10,010,000 and validation10,090,000. Profile two updates and representative 512-event evaluation before authorizing a main budget. Then one bounded acquisition probe of at most300 updates establishes finite learning and useful fresh-data improvement; it does not select among several recipes. Stop for diagnosis if loss is nonfinite or acquisition is absent. No test inspection during development.

Proposed main: paired initialization seeds20/21/22, each arm1000 updates ×32 fresh episodes, AdamW lr0.0003, compatibility weight1, gradient norm clip1. Six-condition schedule clean/partial/contradiction/retract/duplicate/reorder, inherited unchanged. Training seeds are10,000,000 + (initialization_seed−20)×2000 + step. Validation10,100,000 + initialization_seed; untouched test10,200,000 + initialization_seed. These do not overlap prior-stage namespaces or development. Source/config freeze precedes main training and test inference. Fixed final checkpoint; no outcome-based best-checkpoint selection. Step0 and scheduled curves are saved. Main authorized only after root resource release.

Track ceiling45 GPU minutes, target under15 including all profiles, development, three-arm main training and compressed evaluation. Expected training approximately40–60 seconds per run based on Stage8; full evaluation/export may dominate. Profile actual runtime, allocated CUDA peak and process RSS separately. No dependency upgrade. GPU jobs serialize with other tracks. No further architectural expansion if the fixed budget fails.

## Required validation/test matrix

Each cell has512 fresh episodes. N8 IID and N16 moderate OOD are mandatory; N32 is separately labeled additional-size OOD. All original conditions remain: clean, independent reorder, same-ID duplication, eight-fold same-ID duplication, contradiction, explicit contradiction retraction, partial, empty. Add full retraction, distinct equal-content IDs, candidate permutation, arbitrary consistent ID renaming, within-range ID permutation, and the Stage9 seenID4 / unseenID100 redundant-event pair. Initial and all intermediate frames remain scored.

Compare clean and consistently renamed episodes frame-by-frame with the same underlying construction. Report probability delta under ID bijections, never just answer invariance. `ledger_only` invariance must hold mechanically before training. For `raw_randomized`, invariance is an empirical learned property. Repeated-ID and full-retraction metrics separately report ordinary impossible mass, null mass/error, supported-candidate probability fidelity and support size. Full-retraction final support alone cannot hide incorrect intermediate uncertainty.

## Gates and interpretation

The inherited full validation gate is unchanged: all prescribed seeds/cells, minimum512; final unique-selection accuracy >.98 IID and >.95 moderate OOD; mean framewise posterior L1 <.05 and impossible mass <.01. Ambiguous support membership is reported separately from a unique correct proposal. Original and expanded matrix decisions remain separate; no missing cell, failed seed or initial-frame exclusion may pass. N32 does not rescue a mandatory failure.

Architectural invariance requires identical outputs under a bijection (up to explicitly reported floating arithmetic tolerance); this is distinct from learned posterior calibration. A learned-ID repair must pass the expanded gate, and its remaining renaming sensitivity is reported without assuming concentration is correctness. Raw confidence calibration uses expected correctness under the privileged joint posterior; it is never coupled to execution.

Preserve all seeds, training exposures and curves, per-example construction keys, compressed predicted/gold posteriors, source/config/checkpoint hashes and independent raw audits. Report programmed guarantees, learned compatibility, calibration and generalization separately. No composition or supervision withdrawal is authorized by any result.


### Development continuation rule frozen before outcomes

The three arms each receive300 updates at development seed101. Continue to the fixed main recipe only if both repair arms have clean accuracy at least.90 on64 fresh development events, all losses are finite, and ledger-only renaming changes probabilities by at most1e-6. Randomized-ID renaming quality at300 is recorded but is not required to reach its final competence gate before the planned1000-update exposure. The correlated reference has the same development budget and is never dropped because it is weak. If either repair misses the clean acquisition condition, stop and report; do not extend exposure or select a new recipe. Development evaluation namespace10,092,000 is separate from profile/curve namespaces. Root authorized profile plus this development run within a three-minute serialized allocation after the semantic track releases the GPU.


Legacy condition names `distinct_equal_seen_id` (ID4) and `distinct_equal_unseen_id` (ID100) retain their Stage9 spelling for comparison. They do not declare those handles unseen after randomized-ID training. Record the actual training inventory, membership of4/100, their observed primitive values, and the fraction of each evaluation cell's ID assignments absent from training. Do not reinterpret arbitrary renaming as a wholly unseen-ID split. The intervention tests nuisance-handle invariance, with actual identity coverage disclosed.
