# R11: lower learning rate improves scalar access, but the tail gate still fails

The paired optimizer intervention produces a clear development difference. Continuing the same linear consumer atLR.003worsens several results; reducing only that rate to.0003improves fitting and reused validation. Neither fixed7,200-updateendpoint passes the prespecified advancement screen. No confirmation or replacement of an established consumer follows.

| Endpoint | Mixture calibration minimum /1,024 | Grid calibration minimum /128 | Fitting value at16 /65,536 | Mixture validation minimum /4,096 | Grid validation minimum /128 |
|---|---:|---:|---:|---:|---:|
| Frozen3,600reference |1,021|118|65,057|4,070|116|
|7,200constant.003|1,014|87|64,887|4,064|102|
|7,200decay.0003|1,020|120|65,433|4,089|122|

The gate required mixture calibration≥98%, minimum grid calibration≥122/128and16-updatefitting≥65,405/65,536. Decay passes mixture and fitting but misses grid. Constant fails grid and fitting. The decay-versus-constant calibration difference exceeds the separately declared4/128minimum for a lower-LR-specific effect, but that relative effect cannot turn a failed competence gate into a pass.

Both fork calibration minima concern float+8 at16updates; their validation minima are float+8(constant)andfloat−6.5(decay). Validation is **reused development data**, not untouched confirmation, and cannot override calibration selection.

## Exact matched continuation

Before either fork,3,600updates were replayed from the original Stage11scalar head. Weights, normalization, full loss sequence, sampled-row membership and every archivedR10reference prediction/target/event hash agree exactly. The old optimizer state was reconstructed by deterministic replay; it was not available as a historical checkpoint. Historical index-order hashes were not recorded, so historical order is justified by the frozen sampler recipe plus these replay equalities, not an unavailable old hash.

Both new forks restore the same reconstructed AdamW moments/steps and RNG state. They receive exactly921,600additional presentations, visit355,118distinct delay/event rows and all65,536events, and have identical ordered-index stream and final-RNG hashes. Only learning rate differs. Full model/AdamW/RNG checkpoints are now saved. Both have33,825trainable parameters, unchanged1024-dimensional features and identical numerical labels. Backbone, memory, normalization and non-value outputs remain fixed.

## Paired behavior and learning curves

At16updates on6,656balanced validation events, scalar accuracy changes6,593→6,574constant or6,626decay. Relative to the reference, constant fixes47andbreaks66; decay fixes47andbreaks14. Both retain16wrong cases. Joint semantic counts are6,476reference,6,459constant and6,509decay; non-value joint remains6,539for every arm. These exact paired counts distinguish repairing errors from merely subtracting marginals.

All arms reconstruct all6,656zero-updategrid values. Atoneupdate, reference6,654,constant6,652anddecay6,656. This gives no reason to introduce another ingestion-phase architecture for this development population.

The last300training minibatch losses average.004685constant versus.001107decay. Constant continues to show intermittent large loss blocks; decay's recorded losses are lower and smoother. Together with the exact paired intervention, this supports optimizer sensitivity of the tested linear readout. It does not prove that every remaining scalar distinction is linearly representable, that loss spikes had one universal cause, or that further decay would solve the tails.

## Scope and next decision

These are bounded finite-class return-access results on one deliberately selected historical backbone11and reused development data. They are not a new numerical-range, all-seed, delay32, lifecycle or autonomous-composition result. R04/R05/C04interfaces and all historical gates remain unchanged.

The local sequenceR06–R11now distinguishes useful balanced coverage, unsuccessful extra nonlinear capacity/phase conditioning, and a successful relative optimizer repair that still falls short of the tail contract. Recommend closing this local recipe search for the campaign rather than automatically multiplying exposure or searching rates. A future stronger-tail study should begin with a new protocol and fresh evaluation; current evidence supports neither a uniform98%claim nor absent-information claims.

Full outer process occupancy57.58seconds (GNU time,.01sresolution); load/preparation11.585s, exact replay3.191s, fork fitting5.589s, endpoint evaluation10.950s, remaining child work/export24.446s. Peak CUDA allocation4,835,127,808bytes; peak process RSS8,422,756,352bytes. Lossless remote logits and checkpoints plus all108compact raw cells/source/config/provenance are retained. Independent audit is pending. This experiment concerns supporting scalar interfaces, not programmable attention.
