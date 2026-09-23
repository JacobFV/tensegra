# R03 development: optimization exposure on one fixed diverse pool

R02 improved fresh tails as the fitting pool grew, but the16,384-event arm no longer fit all training events. R03 therefore changes only CE optimization exposure. It reuses the identical frozen feature cache, normalization, original-head warm-start, AdamW .003 and batch256 from R02. No new backbone inference, memory, encoding, architecture or loss is introduced.

R02 did not save optimizer state. Replay its trajectory from the original frozen-wide head with the same local generator seed22000011. At900updates require parameter agreement within1e-6 absolute/relative tolerance, exact argmax agreement with the archived R02 readout on every captured split, and exact visited-row bitset agreement. Stop as a mechanical mismatch if any check fails; do not pretend an unmatched trajectory is an exposure-only intervention.

After this check, continue the same optimizer and RNG state to1,800 and3,600updates. Save each prescribed endpoint and actual visited rows/events. The fixed event pool contains16,384 returns×six delays; all endpoints share it. Delay32 remains absent. R02 calibration/validation data are reused transparently for exploratory development, not claimed to be a fresh confirmation.

Endpoint selection rule: maximize the minimum scalar-correct count across all covered calibration delay/distractor cells, then total correct count, then prefer the earlier endpoint. Inspect validation only after this selection is frozen. Report every endpoint's training fit, per-type/value/operator errors and margins; if validation contradicts the calibration-selected trend, disclose it rather than changing the selection rule.

A later main claim requires a new recipe freeze and fresh three-backbone confirmation. No test from R01 is reopened. No historical gate or failed contract changes. Broad subtype and numerical-range reliability must not be inferred from aggregate accuracy.

Estimated process time30–60seconds including lossless cache loading/export; request120-second ceiling based on existing measured head-fitting throughput. The coordinator must release the GPU. Record external process occupancy and reuse provenance hashes, rather than charging only optimizer time.
