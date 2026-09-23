# Return track handoff

Isolated branch/worktree: campaign/returns, campaign-returns. Coordinator alone releases GPU. No current GPU job from this track. Historical source/results remain untouched.

## Completed

R01 development: frozen Stage 11 wide backbone 10, CE900 selected after four ridge candidates and eleven CE checkpoints. Worst fresh validation1016/1024. Confirmation froze CE900/ridge .01 across historical backbones10/11/12, using independent fresh data. Declared0–16 contract FAILED: backbone11 sixteen-step validation1000/1024 for both distractor counts. Backbone10 was also used in development. Other covered cells pass aggregate scalar checks; no scope narrowing or composition is authorized from that alone. Final-test32 counts987/971/990 (eight distractors). All cache/logit/selection/provenance replay checks passed independently. Report/plots are committed.

R02 failure-directed development selected backbone11. One fresh16,384-event cache supplied nested4096/8192/16384 pools, each trained900updates from the original wide head. Worst fresh2048-event validation counts2004/2014/2019; sixteen-step eight-distractor counts2004/2017/2020. Larger pools help generalization but fit imperfectly (16251/16384 at16). Float conditional accuracy remains796/824 in the largest arm; all integer/bool values are exact in that cell. R02 raw audit is pending. Sourcece26bce, report/artifacts8dbedf8, plot38a0127.

## Next

R03 exposure development is authorized in principle, awaiting source review and explicit GPU release. Sourcee05ca8f is staged immutably at gb10-direct:~/topoformer-campaign-01/returns/source-r03. Five CPU tests passed. Configcampaign-r03-development.json reuses R02's exact feature cache and original head. Replay900updates and assert parameter tolerance1e-6, exact argmax and sampled-row bitset before continuing the same optimizer/RNG to1800/3600. Calibration-only endpoint rule is frozen; reused validation is exploratory, not confirmation. No32 exposure. Estimate30–60s, requested120s ceiling. Do not launch without root.

Large immutable files: ~/topoformer-campaign-01/returns/{r01-profile,r01-development,r01-confirmation/{10,11,12},r02-development}. Readouts, caches/logits and manifests have SHA receipts. R02 cache and ce-16384.pt are the R03 source.

Actual external process occupancy so far:3.96+26.19+95.90+72.14=198.19seconds. Initial return envelope2GPUh. Failures select the next discriminating experiment; do not terminate this campaign after one failed gate.

A conditional consumption draft exists but is not frozen/released: learned scalar scores plus public threshold/polarity into a learned comparator, with query-only/oracle ceilings and dropped/wrong/swapped event controls. Promotion requires a prospective consumed-interface contract and fresh validation, not posthoc exclusion of failing conditions.
