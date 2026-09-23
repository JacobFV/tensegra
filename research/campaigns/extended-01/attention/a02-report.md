# A02: changing the score interface repairs the observed soft-routing failure

Exploratory, not confirmation. The registered selection rule chooses fixed λ=8. No graph-context or message-passing baseline was weakened.

| Intervention | N16/D4 | N32/D4 | N16/D8 | N32/D8 | Held-out (2,2) |
|---|---:|---:|---:|---:|---:|
| Trained soft4, seed102 | 1.000 | .980 | .832 | .281 | 1.000 |
| Trained soft8, seed102 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Trained soft4 + log(N/16) | 1.000 | 1.000 | .832 | .949 | 1.000 |
| Frozen A01 soft4, unchanged | 1.000 | .984 | .887 | .316 | 1.000 |
| Frozen A01 soft4, override λ=8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Frozen A01 soft4 + log(N/16) | 1.000 | 1.000 | .887 | .969 | 1.000 |

The frozen intervention changes only structural logit strength. It restores task performance without any new supervised fitting. This establishes a score-interface failure in this model rather than requiring a new value encoder or recurrent architecture. The stronger prior still does not outperform the competitive address-context or message-passing baselines, which were already perfect on the development cells.

The size correction leaves N16 unchanged by construction and largely repairs N32, while fixed λ=8 also repairs the deeper N16 condition. This pattern agrees with finite nonedge mass accumulating over depth, although the learned dynamics are not identical to the equal-score mathematical fixture.

The trained soft4 and size-adjusted models have identical training behavior because all training graphs have N16, where the correction is zero. They are the same learned model with different public size-conditioned evaluation rules, not independent trained mechanisms. Their separate runs check reproducibility and are retained in the resource accounting.

All cells contain the same 256 fresh development examples across arms. Training uses 8,000 generated presentations and initialization seed102; frozen interventions use the earlier seed101 checkpoint. The frozen and newly trained populations are not an initialization-matched comparison. Source/config/data/checkpoint hashes and compact raw predictions are retained.

Full process occupancy was 31.26 seconds. Confirmation will retain soft4 as a reference, compare the selected soft8 with context, message passing and hard attention across three paired seeds, and evaluate fresh data. The largest evaluation shape must be profiled before its resource budget is frozen. No confirmation outcomes have been inspected.
