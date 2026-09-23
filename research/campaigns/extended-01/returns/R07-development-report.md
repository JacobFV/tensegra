# R07: additional nonlinear readout capacity did not improve acquisition

The fixed 900-update residual MLP performs worse than the linear readout on both fitting examples and fresh contexts. It fails the prespecified calibration-grid advancement criterion. This is a failure of this capacity/optimization recipe, not evidence that the frozen features lack the remaining scalar distinctions.

| Readout | Minimum mixture calibration /1024 | Minimum grid calibration /64 | Minimum mixture validation /4096 | Minimum grid validation /64 |
|---|---:|---:|---:|---:|
| Linear replay | 1022 | 59 | 4088 | 55 |
| Residual MLP | 1006 | 0 | 4032 | 0 |

The nonlinear grid failure is the complete float 5.5 class at ingestion. The linear grid minimum remains float −7.5 at delay16. Both aggregate mixture scores exceed98%, illustrating why the separate balanced-grid criterion matters. No checkpoint or threshold was selected after observing these outcomes.

On the actual balanced fitting pool, ingestion accuracy falls from16,382/16,384 to15,998/16,384; delay16 falls from16,358/16,384 to16,239/16,384. Fresh grid totals at ingestion/one update/sixteen updates are3,327/3,327/3,308 for linear and3,254/3,286/3,291 for nonlinear, each over3,328. The failure therefore includes acquisition, rather than only generalization to new nuisance contexts.

The nonlinear average training loss over the last50 updates is.02620 versus.00405 for linear. Its minimum original-mixture calibration count fluctuates from1,022 at step500 to1,000 at800 and1,006 at900. These are diagnostic curves, not permission to retrospectively select step500; only the frozen900 endpoint has full grid evaluation. Optimization sensitivity is a plausible next question, not an established mechanism.

## Controlled comparison and limits

Both heads start with exactly identical predictions. The residual MLP adds a1024-hidden GELU branch with zero-initialized output, increasing trainable parameters from33,825 to1,117,250. Both use identical train-only normalization, fitting features, 900 AdamW updates, 230,400 sampled rows, and the same row sequence. R06 linear reproduction is exact: maximum parameter difference0 and identical visited-row bitset. Additional readout capacity also changes optimization geometry and computation; this is not a parameter- or FLOP-matched comparison.

The backbone, encoder, memory, non-value heads, data population, loss, and recurrent coverage are unchanged. All fields and joint predictions are retained. No delay32 is captured. R06 calibration/validation populations are explicitly reused development data. This experiment does not replace the audited R05 consumer, change historical gates, or authorize autonomous composition.

External process occupancy was24.02seconds and peak CUDA allocation1,208,889,856bytes. No automatic longer run was launched. Independent raw/cache audit is pending. The next smallest discriminating experiment would hold this readout fixed and reduce its learning rate, with a prospective endpoint and all failures retained; it requires coordinator authorization.

[All per-stratum counts and calibration decision](../../../results/campaign-01/returns/r07-development/summary.json), [raw predictions](../../../results/campaign-01/returns/r07-development/predictions.json.gz), [manifest](../../../results/campaign-01/returns/r07-development/manifest.json.gz), [process receipt](../../../results/campaign-01/returns/r07-development/process.json), and [prospective protocol](R07-protocol.md).
