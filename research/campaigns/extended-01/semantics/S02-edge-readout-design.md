# S02: frozen learned-node edge readout

Development diagnostic selected after S01-N128, while the matched diversity ladder continues unchanged. No GPU work is authorized by this file alone.

## Evidence and question

At65,536 presentations, N128 predicts every TRAIN node presence/type/value/copy correctly, but no complete TRAIN graph. Supplying gold edges restores126/128 complete graphs; two remaining slot mistakes are frozen. Does continued optimization of the existing edge readout acquire exact edges from already learned node states, or does this deficit persist even after removing interference from a moving backbone?

The intervention changes only which parameters may update. It is not a new architecture or a gold-node oracle. Cache the final N128 checkpoint's learned node representations using public-text forwards, with no gold graph inputs. Verify that scoring those cached vectors with the inherited affine edge_source/edge_target reproduces the original frozen edge logits exactly. Preserve original predicted node presence and all non-edge heads in deployment. Stop if cache reproduction fails.

## Optimization and comparisons

Warm-start the original edge_source/edge_target weights AND their inherited AdamW moments, learning rate and step counter. Preserve sampled-pair RNG, shuffled-data schedule and continuation position from the original checkpoint. Freeze every other parameter; explicitly verify its state hash remains unchanged. Keep the original balanced positive/negative typed-edge objective,128 negative pairs per example, batch8, gradient clipping1. Gold pairs affect loss selection only; never mask deployment edges with gold presence.

Curves are fixed at0/1200/4800 additional head-only updates. Use the same128 TRAIN constructions and512 development examples. Fit relation thresholds on predicted-present TRAIN pairs with the same lowest-minimum-error tie rule; then freeze them for development. Export raw/calibrated edge sets, exact typed-edge agreement per graph, whole-graph results, F1, signed score separation and calibrated support. All other predicted components remain the saved original outputs. Development is diagnostic; no reserved confirmation evaluation occurs.

Primary readout competence criterion is128/128 exact TRAIN typed-edge sets, separately for raw/calibrated decoding. Whole-graph ceiling is126/128 because two frozen slot errors cannot be fixed by this intervention. No head-only result can establish fresh full-graph semantic competence: the existing development oracle-edge ceiling is zero because its non-edge predictions are already wrong.

## Resource and interpretation

A mechanical profile first extracts a small public subset and times representative head-only steps at width1024; cached node vectors retain1024 coordinates, edge projections retain the existing128 dimensions. Expected extraction is under10 seconds for640 public forwards; head-only training should be much cheaper than backbone training, but the final cap follows measurement. Coordinator queue release is required for profile and main separately. Initial proposed main ceiling120 seconds, subject to the profile.

Save immutable parent checkpoint hash, feature-cache hash, source/config hashes, inherited optimizer state provenance, non-edge state hashes, complete CPU/GPU occupancy and all failed reproduction checks. A successful frozen readout would show accessible learned edge information under this readout/optimization intervention; it would not establish that the joint objective would have learned it at the same cost. Failure would localize a remaining frozen-interface problem without proving information absence or a universal capacity bound.

## Numerical parity and limitations

Both original dense-evaluation and sampled-training edge formulas are preserved under BF16 autocast. A width16 CPU mechanical fixture compares each formula to the unchanged actor and passed. Main capture additionally requires bitwise equality between original public-forward logits and logits reconstructed from cached node states after a CPU round trip, on every captured example. Export the maximum difference; stop rather than changing precision if this check fails. The cache freezes single-example public forwards; small batched backbone numerical differences from the original joint training are not a learned intervention and should not be mistaken for a capacity result. The edge optimizer still uses batch8.

Profile config is32 updates on8 TRAIN/8 development records, explicitly mechanical; the main config remains128/512 and4,800updates. The profile's shortened schedule is a disclosed reset, while main inherits the original128-example schedule and RNG states. A separate launcher prevents changes to the frozen S01 launcher. No lifecycle, slot, node encoder, inference schema mask, or threshold-policy change is introduced.

The frozen config pins the expected parent checkpoint SHA256 and the runner verifies it before loading. Gradient clipping now acts only on the edge-head gradients, whereas joint training clipped the combined backbone/other-head norm. This is intentionally frozen-backbone optimization, not a claim of identical optimizer updates to a full-model continuation.
