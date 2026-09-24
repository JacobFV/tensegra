# Scalar-tail branch closeout: R10–R11

The local search is closed after the fixed R11 endpoint; no additional rates, exposure or confirmation are queued. This is a campaign allocation decision, not proof that a stronger scalar consumer is impossible. The confirmed R04original-mixture interface and R05/C04consumers are preserved unchanged.

## Supported conclusion

R10 adds balanced fitting contexts16k→64kat matched3,600updates. The worst reused validation grid improves108→116/128, but the registered calibration minimum improves only115→118(<122required). New-context tails remain imperfect, and the larger training set is less completely fitted.

R11 reconstructs the exact64koptimizer trajectory and compares identical additional3,600updates atLR.003versus.0003. Lowering only LR improves fitting65,057→65,433/65,536and reused validation minimum116→122/128; constant continuation worsens both. Decay still misses the calibration minimum120/128versus122required. No heldout selection or gate relaxation occurs.

The evidence localizes a meaningful optimizer sensitivity; it does not establish missing information in the workspace. Ingestion succeeds perfectly in both new populations. Remaining late-float errors involve both acquisition and fresh-context generalization. No phase, encoder, memory or backbone change is justified by these tests alone.

## Search and resource accounting

Four new development endpoint recipes were evaluated: R10balanced16k/64k, then R11constant/decay continuation. R11also reproduces the earlier64kendpoint for exact optimizer recovery. Both comparisons use one deliberately selected historical backbone11and finite33-value labels. The52type/value strata do not create new numerical ranges. R10introduces65,536distinct fitting events with a16,384prefix and18,432distinct development-evaluation events; R11reuses them. Delays, distractors and heads do not multiply independent event support.

| Released process | Full outer occupancy, seconds |
|---|---:|
|R10mechanical profile|13.44|
|R10development|249.14|
|R11mechanical profile|4.30|
|R11development|57.58|
|Total|324.46|

GNU-time resolution is.01second; root conservatively rounds each charge upward. These are complete experiment-process intervals, including CPU preparation and export while owning the GPU slot, not claims of GPU-active kernel time. The two main comparisons consume4,608,000optimizer presentations including R11's exact replay; mechanical profiles add128,000. All failed endpoints are retained.

The final local CPU analysis replay costs8.1591wall seconds (7.9043user +.2530system), reproduces both summary files byte-for-byte, and never accessesCUDA. Earlier exploratory shell analyses were not separately CPU-instrumented; the receipt explicitly limits its scope. Independent review has separately measured CPU accounting and must not be double-counted here.

## Preservation and followup

No historical source/result file or established checkpoint was changed. All new source/configs/raw predictions and source/config/cache/checkpoint hashes are versioned. Large immutable feature banks, full logits and readout/AdamW/RNG endpoints remain remote with verification hashes. The final summary reader rejects incomplete matrices, bad support, duplicate cells, changed non-value outputs and inappropriate mechanical-profile promotion.

Independent R11raw/cache/state/gate audits pass, including exact endpoint predictions and the retained promotion failure. No GPU jobs remain owned by this branch. Recommended next campaign allocation is to semantic transfer and direct attention confirmation rather than another scalar-tail optimizer sweep. A later robust-tail study needs a new protocol and fresh confirmation, not a retrospective reinterpretation of these failed screens.
