# S11 proposal: one learning-rate-only late continuation

S03 improves with continued exposure, while S04 improves at196,608 presentations then regresses by262,144 under the same1e-4 learning rate. This motivates a bounded optimization control before interpreting a new decoder architecture: change only the restored optimizer's learning rate to1e-5 and compare at the same starting checkpoint, example order and number of subsequent updates.

## Feasibility and exact scope

The immutable S03 checkpoint `model-u16384.pt` is present and its SHA remains `e54b23dc8362dd4d00381b6fd3115e48eef3942c519d8179e4c4e7b0610732af`. CPU inspection confirms model weights, AdamW moments/steps16,384, param-group LR1e-4, pair-sampling RNG, shuffle RNG, order and position, and cumulative visits131,072. The actual active stochastic processes are saved. Global CPU/CUDA RNG states were not archived; the model has zero dropout and sampling uses the two explicit generators. Do not claim a bit-identical restoration of unspecified global RNG state.

The current continuation runner loads optimizer state after constructing AdamW. Merely setting `learning_rate=1e-5` in its configuration would be ineffective: the archived group LR overwrites the constructor value. A separately versioned S11 runner must first restore the optimizer and then set ONLY each param group's `lr` to1e-5. It must test/record unchanged moments, steps and all other optimizer settings. Historical source is not edited.

Require exact initial raw/calibrated TRAIN128/DEV512 prediction and threshold replay, plus the existing independent cloned first-batch/sample-pair RNG check. Compare that first-batch record with S04 as an additional public receipt. Retain original width1,024, actor, objectives, curriculum, batch8, graph data, seed201, calibration policy and vocabulary. No pointer, occurrence or context-read intervention is combined.

## Budget and comparisons

Start at16,384 updates /131,072 presentations. Add8,192 updates /65,536 presentations and stop at24,576 updates /196,608 cumulative presentations. Checkpoints16,384 /20,480 /24,576; no early selection. The exact comparator is S04's already declared24,576-update checkpoint, not its later262,144 endpoint. That comparator has already been inspected, so S11 is a development-selected experiment, not untouched confirmation. No other learning rates or optimizer changes are tried.

Same inference/training geometry as the completed S03 tranche predicts approximately650 full-process seconds; requested external cap900. No new GPU profile should be necessary solely for the LR scalar, but source/reviewer/budget approval and explicit root release remain mandatory. The proposed config is labeled protocol-only and cannot be treated as permission to run the old source.

## Curves and diagnostics

Save per-component sampled training losses and exact full-graph/component outcomes at each declared checkpoint. Also derive raw natural-prevalence and balanced-positive/negative BCE from the losslessly archived TRAIN calibration scores/targets. Report per-relation counts and score supports. These are losses conditional on predicted-present node pairs, not the same distribution as sampled training loss and not a loss on calibrated hard decisions. Reconstruct the identical diagnostics for S03's parent and S04's comparison checkpoint without new neural inference. Extra intermediate evaluation consumes no sampler RNG.

Primary evidence is complete canonical graphs, raw and fixed TRAIN-calibrated decoding. Copy/type/ordered edge metrics and loss changes localize residuals. No test-renderer threshold fitting. A promising search outcome requires>=10% complete DEV graphs, copy>=.95 and ordered F1>.90; it still requires later paired confirmation and transfer controls. Failure does not authorize another LR, more exposure, or hiding the fixed endpoint. Partial improvements with zero complete graphs remain partial acquisition evidence.

The corpus remains8,192 distinct equality/reference patterns but only two ordered trees at depth3. All targets are privileged supervision, all actor inputs public text. This optimization experiment does not directly test programmable attention, new semantic motifs or open-world language understanding.

## Frozen implementation and preflight

`campaign-s11-lr.json` freezes the sole1e-5 override and900-second requested cap. The new runner restores all prior state, overrides only optimizer param-group learning rates, rejects a changed parent/comparator hash or different first-batch/pair-sampling receipt, and preserves exact initial prediction replay. Two CPU mechanical tests pass: optimizer state differs only in LR, and archived BCE computation retains support counts. No GPU profile is needed because the unchanged geometry was already measured. Archived parent/comparator calibration losses are saved separately before new outcomes. The source and config are committed before coordinator release; GPU permission is not implied by this frozen config.
