# R06 development: balanced fitting improves a late scalar tail

Changing only the selected fitting population improves both the prespecified calibration tail and fresh original-mixture reconstruction. It passes the declared development advancement rule, but does **not** establish uniformly reliable exact scalar reconstruction.

| Scalar-head fitting population | Worst original calibration /1,024 | Worst calibration grid cell /64 | Worst original validation /4,096 | Worst validation grid cell /64 |
|---|---:|---:|---:|---:|
| Original mixture | 1,020 | 44 | 4,076 | 41 |
| Balanced legal type/value pairs | 1,022 | 59 | 4,088 | 55 |

Both worst grid cells are float −7.5 at sixteen updates. The balanced calibration minimum strictly improves while its original-mixture minimum remains above98%, satisfying the frozen advancement rule. Validation agrees directionally but remains weak in its worst stratum:55/64 is85.94%, not a uniform98% pass. No endpoint or threshold was chosen from validation.

At validation ingestion and one update, the balanced arm reaches3,327/3,328 grid events; at sixteen updates it reaches3,308/3,328 versus3,295/3,328 for the original-mixture refit. Crucially, **both fresh refits already recover all64 float-zero events at ingestion**. R04's specific whole-class float-zero failure is therefore not reproduced by the fresh original control; do not attribute that class's repair to balancing. The supported improvement concerns the remaining late-value tail in this development run.

Both arms use frozen historical backbone12, the same original head,16,384distinct fitting events,900CE updates,230,400presentations and33,825trainable parameters. They share11,545events selected from one fixed65,536-event pool. The balanced population contains315/316examples per legal type/value pair; it intentionally changes type/primitive prevalence and the induced train-only normalization statistics. This is an explicitly supplied training-population prior, not a new learned semantic invariant or an objective-only causal comparison.

The event generator, values, identities, operand witnesses, architecture, recurrent delay coverage, optimizer and loss remain unchanged. No delay32 data is captured. All non-value heads remain frozen; all fields and joint outcomes are retained. This refit does not replace the audited R04/R05 interface or inherit its decision contract automatically.

External process occupancy was135.48seconds; peak CUDA allocation1,499,566,592bytes. The development backbone was selected because of its earlier failure, and only one new fitting/evaluation population is tested here. Independent audit is pending. Next requires a prospectively frozen, fresh multi-backbone confirmation or another coordinator-selected diagnostic; no automatic composition follows.

[Calibration decision and every grid cell](../../../results/campaign-01/returns/r06-development/summary.json), [raw predictions](../../../results/campaign-01/returns/r06-development/predictions.json.gz), [manifest](../../../results/campaign-01/returns/r06-development/manifest.json.gz), and [process receipt](../../../results/campaign-01/returns/r06-development/process.json).
