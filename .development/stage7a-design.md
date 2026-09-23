# Stage 7a: isolated semantic return retention and use

The user's reductionist sequence starts with perfect returned facts, before candidate formation or halting. This stage only tests that first interface. No new primitives, projectors, pretrained language models, or TCN scaling runs are introduced. Existing stages remain unchanged.

## Question and controls

Can a model retain a typed event through 1/2/4/8/16 recurrent updates and then use its value with a newly revealed query? Reuse the existing four distinct recurrent workspace blocks. Compare identical parameterizations and initialization:

- `once`: inject the learned typed event once; subsequent updates see unrelated distractors.
- `persistent`: same initial injection, plus read access to the unchanged typed event at every update. This is architecturally supplied memory, not learned exact preservation.

All output heads read only the latent workspace. No exact scalar or symbolic answer is passed to the output heads. A late comparison query arrives only after the retention interval, followed by a required additional cell update. Thus precomputing the eventual task answer at injection is impossible. The protected arm may reread the event; success there establishes learned consumption of persistent memory, not autonomous latent retention.

## Data contract

`retention_data.make_batch(seed, batch_size, feature_dim=32, distractors=2, value_limit=8)` returns `{public, targets}` with tensors. Public fields: `event` dictionary accepted by ThinkingModel.encode_events (values[B,1], types[B,1], operations[B,1], arguments[B,1,2,F], provenance[B,1,F]); `distractors[B,D,F]`; `provenance_keys[B,4,F]` shuffled independent nonce vectors; `query[B,2]` holding threshold and comparison-inversion bit. Targets are separate: value_class[B] for half-unit values -8..8 (33 classes), type[B] (int/float/bool), operation[B] (add/sub/mul/neg/compare), provenance[B] index into shuffled public keys, task[B] for `(value > threshold) XOR inversion`. Generate semantically valid perfect events with explicit ordered operands. Every example's query is withheld until after the delay. No target tensor enters model.forward.

Type cannot be perfectly inferred from numeric integrality: float values may be integer-valued. Provenance keys are random per example; no stable provenance label or token position shortcut. Regenerated seeds/data hashes identify every batch. Train and evaluation seed ranges are disjoint.

## Model contract

`retention.ReturnRetentionModel(feature_dim=32,width=32)` owns a ThinkingModel with the same existing four blocks plus value/type/op/provenance/task readout heads. `forward(public, delay, mode, intervention='none')` returns before-query and after-query logits plus diagnostics. Modes change only availability of the immutable event in recurrent memory; parameters match. Reuse blocks directly to exclude unused grounding, topology, candidate and emission mechanisms from this isolated question. Workspace rows remain distributed features.

For readout, provenance scores compare a learned workspace projection to the public shuffled keys; other heads receive only workspace. Query enters the final recurrent update, not the final task readout directly. Unrelated distractors stay identical across paired modes.

Frozen interventions: remove persistent access after initial injection; substitute a different event in the persistent channel; switch the late query while holding event fixed; change the entire event consistently and regenerate its target. Report intervention support and distinguish correct response to changed facts from corruption sensitivity.

## Measurements and sequencing

First, overfit a fixed small training set with all query conditions to establish the loss and decoder can acquire the interface. Then train fresh paired seeds on generated examples with delays1/2/4, evaluate IID and delays8/16, and more distractors. Log step0 and training curves, per-field accuracy, joint semantic retention, exact task accuracy, accuracy conditional on retained facts, counterfactual sensitivity, recurrent drift, elapsed time, parameter counts and source/config/data/checkpoint hashes.

A provisional competence gate is >=99% joint field retention and >=99% late-query accuracy on the fixed acquisition set; main heldout results remain separate and are never used for architecture/hyperparameter selection. If acquisition fails, characterize the field/loss that fails rather than composing more subsystems. Training budgets are fixed after a timing smoke and before main outcomes. All training is remote on gb10-direct, at most two CPU threads per process, with no local Torch workload.

Multiplying marginal confidence terms is not a joint calibrated probability without independence. The later readiness study must calibrate correctness of the whole candidate and treat schema validity as an exact gate; stable wrong proposals are possible. No readiness mechanism is added here.
