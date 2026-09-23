# Return track handoff

Isolated branch/worktree: campaign/returns, campaign-returns. Coordinator alone releases GPU. No current GPU job from this track. Historical source/results remain untouched.

## Completed

R01 development: frozen Stage 11 wide backbone 10, CE900 selected after four ridge candidates and eleven CE checkpoints. Worst fresh validation1016/1024. Confirmation froze CE900/ridge .01 across historical backbones10/11/12, using independent fresh data. Declared0–16 contract FAILED: backbone11 sixteen-step validation1000/1024 for both distractor counts. Backbone10 was also used in development. Other covered cells pass aggregate scalar checks; no scope narrowing or composition is authorized from that alone. Final-test32 counts987/971/990 (eight distractors). All cache/logit/selection/provenance replay checks passed independently. Report/plots are committed.

R02 failure-directed development selected backbone11. One fresh16,384-event cache supplied nested4096/8192/16384 pools, each trained900updates from the original wide head. Worst fresh2048-event validation counts2004/2014/2019; sixteen-step eight-distractor counts2004/2017/2020. Larger pools help generalization but fit imperfectly (16251/16384 at16). Float conditional accuracy remains796/824 in the largest arm; all integer/bool values are exact in that cell. R02 raw/cache/coefficient/visit-bitset audit passed (237f9d0/c1de9d0). Sourcece26bce, report/artifacts8dbedf8, plot38a0127.

## Next

R03 is complete; calibration retained 900 updates. Read the R03 report for all three endpoints and the fixed-fit versus fresh-error distinction. R04 now awaits independent source review and coordinator release. Frozen source is 6b5341b, staged at `~/topoformer-campaign-01/returns/source-r04`; five CPU tests pass. It compares nested fresh 4,096/16,384-event pools with 900 updates each, unchanged original-head warm-start and the same 230,400 optimizer presentations. Three frozen historical backbones receive fresh independent data. Validation/test support is 4,096 events per run, calibration 1,024. All covered-delay validation cells must reach 4,015/4,096 for the narrow aggregate scalar contract. No delay-32 fitting or selection. Expected 300–450 seconds, requested hard cap 600. No GPU job is active.

Large immutable files: ~/topoformer-campaign-01/returns/{r01-profile,r01-development,r01-confirmation/{10,11,12},r02-development}. Readouts, caches/logits and manifests have SHA receipts. R02 cache and ce-16384.pt are the R03 source.

Actual external process occupancy so far:3.96 + 26.19 + 95.90 + 72.14 + 17.38 = 215.57 seconds. Initial return envelope2GPUh. Failures select the next discriminating experiment; do not terminate this campaign after one failed gate.

A conditional consumption draft exists but is not frozen/released: learned scalar scores plus public threshold/polarity into a learned comparator, with query-only/oracle ceilings and dropped/wrong/swapped event controls. Promotion requires a prospective consumed-interface contract and fresh validation, not posthoc exclusion of failing conditions.

R03 completed: external 17.38 seconds, exact 900 replay, calibration retained 900 despite improved training fit at 1800/3600. Detailed report and all endpoints committed. R04 protocol/config freezes 16k/900 on fresh three-backbone populations; awaits source review and GPU release. No job is active from this track.
