# A09 guarded exposure extension

The global graph-record interface remains underacquired after 2000 updates. On the fresh 512-event development population, exact task accuracy is 85.35% IID, 49.80% at N64/D8, and 14.06% at N128/D32/K8. It fails the registered 98% IID and 95% N64/D8 competence thresholds. This is exploratory evidence from seed701; no confirmation or additional exposure increase follows automatically.

The original 1000-update prefix was reproduced exactly: model tensor SHA256 `be44e0d5a30f109acfeedeec15278a78f68e49f37aa880fc762e16e41fd0240d` and fresh side-by-side checkpoint logits match with maximum error zero. The guard passed before extension. All 2000 updates, including replay, are charged. Optimizer, model, and CPU/CUDA RNG states at1000/1500/2000 are durably preserved remotely and hashed in SHA256SUMS.

| Condition | Exact task | Complete suffix | Exact diagnostic path |
|---|---:|---:|---:|
| IID N32/D4 | 85.35% | 48.63% | 96.88% |
| N64/D8 | 49.80% | 5.27% | 94.73% |
| N128/D32/K8 | 14.06% | 0% | 58.79% |
| Joint, record-logit multiplier2/address16 diagnostic | 15.43% | 0% | 60.94% |

The diagnostic label refers to the predeclared override implementation: selector_scale_override16 multiplies record logits by16/8=2. It does not establish acquired symbolic routing. Instruction-swap task accuracy is87.30%; zero-content15.23%. Wrong-graph clean task is19.34% while agreement with the supplied graph is90.82%.

Monitoring and final evaluation use different registered populations (173M and191M). At1000, monitoring task accuracy was43.75% IID and18.75% N64/D8. Do not treat changes to the final population as paired event improvements. Strong supplied-neighborhood A06 interfaces remain the relevant baselines; these results neither show a unique soft-attention advantage nor establish ordinary graph-text understanding.

Frozen source0880c64; configuration file SHA256 `89c486b467c178ed354d443d1b0023521f855b649ca45abfcfa6727571462f66`. Full process occupancy is60.396638588seconds, persisted atomically remotely before launcher exit; internal evaluation/export wall time58.412202247seconds. CUDA peak2,087,815,680bytes. Launcher2123996 changed orchestration only. GPU was verified idle after completion. Including recovered A08 profile's conservative60-second upper bound, attention accounting throughA09 is1066.046638588seconds; this combines measured occupancy and that explicitly uncertain upper bound.

Raw metrics, paired arrays, source hashes, replay receipt, process log, and remote file hashes: `research/results/campaign-01/attention/a09`. Full training states/checkpoint: `~/topoformer-campaign01/attention/a09-receipted/results` on gb10-direct. Independent raw audit requested.
