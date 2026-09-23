# R02: more fitting diversity improves the failing boundary

This is failure-directed development on historical wide backbone 11, not new confirmation. All consumers use the same frozen backbone and original scalar-head warm-start. One fresh 16,384-event cache supplies nested 4,096/8,192/16,384 fitting prefixes, each with six delays. Every arm receives exactly 900 CE updates, batch256 and learning rate .003; more unique events therefore receive fewer average revisits.

| Fitting events | Exact in-pool scalar at16 | Calibration at16,8 distractors | Validation at16,8 distractors | Worst validation cell |
|---:|---:|---:|---:|---:|
| 4,096 | 4096/4096 | 1001/1024 | 2004/2048 | 2004/2048 |
| 8,192 | 8181/8192 | 1004/1024 | 2017/2048 | 2014/2048 |
| 16,384 | 16251/16384 | 1010/1024 | 2020/2048 | 2019/2048 |

The larger pools improve fresh reconstruction despite receiving fewer repeat optimizer exposures. They also cease fitting every observed event perfectly. The next discriminating question is therefore optimizer exposure at a fixed large pool, not a new decoder architecture.

Every fitting event was sampled at least once. Unique sampled delay/event rows were24,574/48,700/88,839, out of24,576/49,152/98,304 possible rows. Packed visitation bitsets permit independent reconstruction. Minimum numerical-class support rises40→77→180. Normalization is estimated separately from each nested fitting pool, so this comparison includes the natural consequences of additional data for normalization and revisit frequency.

The remaining validation16 errors are exclusively float typed. At eight distractors the float counts are780/824,793/824 and796/824; integer822/822 and boolean402/402 are unchanged and exact for every arm. Thus the largest arm's aggregate98.63% does not imply >=98% reliability within every semantic subtype. Per-value, operator, margin and error-distance records remain available. Zero-step validation is2048/2048 for all three arms.

## Next experiment

R03 will reuse exactly the16,384-event cache and reproduce the900-update trajectory before extending it to1,800/3,600 updates. Since optimizer state was not saved, rerun from the original warm-start with the same RNG stream and verify the900 endpoint against R02. Select among prespecified endpoints using minimum calibration coverage then total correctness; validation reports the development trend without changing thresholds. Original R01 confirmation remains failed and untouched. No composition is promoted from this development result.

## Resources and provenance

Frozen source ce26bce; full process occupancy72.14s, inner runner70.781s including feature capture and exports. Peak allocated CUDA memory1.500GB. All old result bytes remain untouched. Local compact artifacts are in research/results/campaign-01/returns/r02-development; immutable large cache/logits/weights are at gb10-direct:~/topoformer-campaign-01/returns/r02-development. Independent audit is pending.
