# Stage 9 return localization

Status: archived numerical reconstruction complete; frozen inference and probes pending. No return interface gate passes, no repair selected, no downstream use authorized. Workspace standard remains 1024.

## Source and evidence map

`return_memory.py` defines the learned field encoders, mixer, initial cross-attention, four recurrent workspace blocks and workspace-only decoder. `return_memory_study.py` defines deterministic regenerated evaluation batches, decoding counts, training delays and checkpoint export. `retention_data.py` constructs exact public events and disjoint supervised targets. `retention.py` implements protected storage; `thinking.py` supplies the recurrent blocks. All five current file hashes match the archived Stage 8 main manifest. The 18 immutable checkpoints remain at `gb10-direct:~/topoformer-stage8-return/results/main`.

The Stage 8 main manifest/config and 18 gzip prediction files under `research/results/stage8/return-main` support the report. `return_diagnostics.py` reconstructs 1,008 evaluation rows and 504 paired-delay comparisons from those files; this is **archived metric reconstruction, not rerun inference**. New outputs and raw file hashes are under `research/results/stage9/return-archive`.

## Pair identity and historical diagnosis

The historical runner calls `make_batch(es,512,d)` independently for every delay. That generator uses a local seeded RNG and generates return events, nonce identity keys and their targets before the distractor tensor. Thus equal data seed/batch size/distractor count implies the same ordered event batch across delays. The analysis verifies all six target arrays are equal before pairing. Historical raw records do not contain nonce hashes; deterministic source supports event identity, while frozen regeneration will add explicit hashes. Matching only the visible categorical targets would not independently establish nonce identity.

Test set, eight distractor rows, factorized persistent memory, paired one-to-sixteen updates:

| Seed | Quantity | correct→correct | correct→wrong | wrong→correct | wrong→wrong |
|---|---|---:|---:|---:|---:|
|0|scalar|363|67|9|73|
|1|scalar|425|55|14|18|
|2|scalar|398|66|5|43|
|0|non-value joint|471|40|0|1|
|1|non-value joint|498|13|0|1|
|2|non-value joint|494|15|0|3|
|0|full joint|333|97|8|74|
|1|full joint|413|66|14|19|
|2|full joint|388|73|5|46|

The net 209-example full-joint decline combines 236 deteriorations and 27 recoveries. The marginal difference is not a forgetting probability. Across seeds the same held-out event set is reused, so pooled counts are descriptive seed-replicates, not 1,536 independently sampled events.

Every field/count agrees with Stage 8. Analysis includes confusion matrices, signed/absolute scalar errors, per-value/sign/magnitude/type/primitive counts, non-value joint and scalar accuracy conditional on non-value correctness. Required argument1 excludes the explicit unary null. All original 18 retention gates remain failed; exact scalar reconstruction blocks each arm.

Persistent arms read intact encoded storage again on every recurrent update; late degradation concerns learned retrieval/representation/readout, not corruption of protected storage. The original trained lengths are0/1/2/4; sixteen and32 are recurrent-length extrapolation. Lifecycle interventions were untrained OOD. New probes must not use the inspected historical test set for selection.

## Frozen-checkpoint plan (before probes)

Capture scalar field encoding before mixing, memory after mixing, initial retrieved workspace, and workspace after1/4/16/32 updates. Use fresh, disjoint probe-training/validation/test events in the9-million seed namespace. Keep checkpoints frozen and report their original seeds0/1/2 as historical frozen replicas, not new initialization seeds. Equal-capacity probes receive1024 coordinates (zero-padding the disjoint scalar field); compare categorical linear and bounded nonlinear probes selected on validation only. Report geometry by actual value before asserting normalization removes magnitude. Oracle facet reads and exact public scalar input are explicitly privileged boundaries, not complete-interface successes.

A profile precedes a frozen full diagnostic budget. No representation repair will be selected before localization. Any later repair uses new seeds10/11/12 and untouched512-example cells. At most two repairs are permitted, with the historical exact reconstruction gate unchanged.

## Frozen probe freeze

Profile at source8838749: three64-event batches, width1024, five workspace boundaries through32updates, facet/persistent checkpoint0 took1.026seconds after load on GB10. Four mechanical tests pass, including exact state agreement with the historical forward. Full frozen matrix estimates~5–8 GPUminutes including ridge solves (18checkpoints ×3072events; no backbone optimizer). Hard ceiling10minutes pending root authorization.

Configuration `stage9-return-frozen-probes.json` fixes all18 checkpoints, all seven boundaries,2,048fresh training events,512validation and512test events (distinct9-million data seeds), eight distractors and all0/1/4/16/32delays. All33 numerical output classes belong to the training domain; this tests fresh contexts, not unseen numerical labels. Original scalar decoder is evaluated at every workspace boundary on identical events. Diagnostic ridge families are1024→33 categorical regression and1024→1 numerical regression followed by rounding/clamping to the half-unit grid. They differ in objective/output prior, so their comparison is not an equal-output-capacity architecture trial. Within each family capacity is equal across boundaries. Scalar premix is zero-padded to1024. Train-only feature standardization and validation-only selection over regularization{.01,.1,1,10}; all boundaries and failures reported. Failure of a ridge probe cannot establish information absence.

## Full scalar-grid result (privileged encoder diagnostic)

The CPU-only frozen encoder check took8.55seconds. Every one of18 scalar affine encoders supports528/528 exact half-unit-grid reconstruction with a newly fitted linear numerical readout on fresh nonce nuisance contexts. The train/test populations both cover all33 bounded labels16times; this is deliberately not numerical-label extrapolation. The isolated context is float/add with consistent operands.

Selecting the correct encoded facet remains528/528 for all six facet checkpoints. After mixed/concatenated encoding, the same linear numerical probe reaches only295–371/528 across the twelve checkpoints. This establishes weaker linear accessibility after merging nuisance fields; a failed linear probe does not establish absence of scalar information. This privileged facet selection does not bypass the historical workspace gate in reported competence decisions. Actual affine/mixed encoding pair distances and norms are saved; no magnitude-destruction claim is inferred from LayerNorm alone.

Workspace boundary probes consume `model.norm(state)[:,0]`, exactly the same normalized scalar row as the original scalar classification head. Thus the baseline/probe comparison does not conflate removal of final LayerNorm with a different decoder. Frozen capture parity is tested against the historical forward at0/1/2 mechanical steps; primary captures retain1024dimensions. Probe source finalized at`e12d546` before test inference, with per-alpha validation counts, saved coefficient hashes, event identities, configuration hash and distinct GPU allocation/process RSS/device-capacity measurements.
