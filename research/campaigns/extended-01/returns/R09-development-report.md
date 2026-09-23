# R09: binary phase conditioning yields mixed improvements, not advancement

The public ingestion-versus-recurrent readout fails the frozen advancement criterion. Its worst calibration-grid type/value cell falls from59/64 to57/64. Reused validation improves from55/64 to57/64, but validation cannot override the registered calibration decision.

| Consumer | Mixture calibration minimum /1024 | Grid calibration minimum /64 | Mixture validation minimum /4096 | Grid validation minimum /64 |
|---|---:|---:|---:|---:|
| Shared linear | 1022 | 59 | 4088 | 55 |
| Public binary-phase linear | 1023 | 57 | 4090 | 57 |

The worst grid cell remains float −7.5 at sixteen recurrent updates. Balanced validation totals at ingestion/one/sixteen updates change from3,327/3,327/3,308 to3,328/3,327/3,314 of3,328. Actual fitting accuracy at sixteen updates improves from16,358/16,384 to16,367/16,384. These modest changes do not demonstrate that binary phase separation resolves the remaining late-value errors.

The shared classifier exactly reproduces R06: parameter difference0 and identical visited-row bitset. Both arms begin with the same original transformed linear function and use the same balanced fitting rows, train normalization, learning rate.003,900 optimizer updates and230,400 total presentations. The phase consumer has67,650 parameters versus33,825. Its ingestion classifier receives38,247 supervised rows and recurrent classifier192,153; both receive optimizer/weight-decay steps on each mixed batch. This is an additional consumer-capacity and public phase-contract intervention, not an equal-capacity comparison.

The phase bit uses only public t=0 versus t>0. No specific positive delay has its own head; no target or private semantic state controls routing. The backbone, encoder, memory and non-value heads are frozen. All six fields and full-joint predictions remain archived. No delay32, extension, or replacement of the audited R05 interface occurred.

All evaluation populations are reused R06 development data. The initial-phase result cannot establish broader generalization, and the failed grid gate does not show absent scalar information. External occupancy was23.57seconds, peak CUDA allocation1,208,889,856bytes. Independent raw/cache audit is pending. Preserve the mixed outcome and pursue another intervention only under a separately specified question.

[All grid strata and decision](../../../results/campaign-01/returns/r09-development/summary.json), [raw predictions](../../../results/campaign-01/returns/r09-development/predictions.json.gz), [manifest](../../../results/campaign-01/returns/r09-development/manifest.json.gz), [process receipt](../../../results/campaign-01/returns/r09-development/process.json), and [prospective protocol](R09-protocol.md).
