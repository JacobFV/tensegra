# Return track handoff

Isolated branch/worktree: campaign/returns, campaign-returns. Coordinator alone releases GPU. No current GPU job from this track. Historical source/results remain untouched.

## Completed

R01 development: frozen Stage 11 wide backbone 10, CE900 selected after four ridge candidates and eleven CE checkpoints. Worst fresh validation1016/1024. Confirmation froze CE900/ridge .01 across historical backbones10/11/12, using independent fresh data. Declared0–16 contract FAILED: backbone11 sixteen-step validation1000/1024 for both distractor counts. Backbone10 was also used in development. Other covered cells pass aggregate scalar checks; no scope narrowing or composition is authorized from that alone. Final-test32 counts987/971/990 (eight distractors). All cache/logit/selection/provenance replay checks passed independently. Report/plots are committed.

R02 failure-directed development selected backbone11. One fresh16,384-event cache supplied nested4096/8192/16384 pools, each trained900updates from the original wide head. Worst fresh2048-event validation counts2004/2014/2019; sixteen-step eight-distractor counts2004/2017/2020. Larger pools help generalization but fit imperfectly (16251/16384 at16). Float conditional accuracy remains796/824 in the largest arm; all integer/bool values are exact in that cell. R02 raw/cache/coefficient/visit-bitset audit passed (237f9d0/c1de9d0). Sourcece26bce, report/artifacts8dbedf8, plot38a0127.

## Current frontier

R04 completed in 398.23 seconds and passed the registered aggregate scalar contract in all 36 fresh validation cells for the 16k/900 arm. Worst counts are 4079/4038/4049 of4096. The paired4k arm still fails backbone11. Full-grid diagnostics expose a whole-class failure: backbone12 maps all64 float-zero returns to−0.5 at ingestion; backbone11 float16-step accuracy is2025/2112. Raw/cache audit passed; generator reconstruction is separately tracked. Report9708d93 and plots00c7c10 retain all failures and full-field counts. Historical registries are untouched.

R05 is being prepared as a learned closed-vocabulary scalar-consumer interface, not autonomous orchestration. Frozen R04 scalar scores plus public threshold/polarity feed a35→1024→2 comparator; query-only and exact-value learned comparator are paired controls. Actor3cae39f is staged immutably at `~/topoformer-campaign-01/returns/source-r05`;10CPUtests pass. No R05 GPU job has run. Reviewer actor preflight pending; coordinator alone releases profile/development.

Analytical query-only Bayes ceiling4324/5445=79.4123% motivated a pre-outcome protocol repair f80e916: ≥15pp dropped-return effect plus ≥80% of positive matched oracle-minus-query gain, while ≥90% supplied-fact changed-subset agreement remains. Clean development >95% only permits fresh confirmation; later composition still needs≥98% across3fresh populations. Preserve actual mixture and separate balanced-use diagnostics. No output-based threshold change occurred.

Large immutable files: ~/topoformer-campaign-01/returns/{r01-profile,r01-development,r01-confirmation/{10,11,12},r02-development}. Readouts, caches/logits and manifests have SHA receipts. R02 cache and ce-16384.pt are the R03 source.

Actual external process occupancy so far:3.96 + 26.19 + 95.90 + 72.14 + 17.38 + 398.23 = 613.80 seconds. Initial return envelope2GPUh. Failures select the next discriminating experiment; do not terminate this campaign after one failed gate.

A conditional consumption draft exists but is not frozen/released: learned scalar scores plus public threshold/polarity into a learned comparator, with query-only/oracle ceilings and dropped/wrong/swapped event controls. Promotion requires a prospective consumed-interface contract and fresh validation, not posthoc exclusion of failing conditions.

R03 completed: external 17.38 seconds, exact 900 replay, calibration retained 900 despite improved training fit at 1800/3600. Detailed report and all endpoints committed. R04 protocol/config freezes 16k/900 on fresh three-backbone populations; awaits source review and GPU release. No job is active from this track.
