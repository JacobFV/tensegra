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

## R05 latest (supersedes preparation notes above)

Development completed at frozen143db40:68.97s external, learned validation2041–2044/2048, query-only1634, oracle2043. Drop effect19.43–19.97pp; changed wrong867–868/871 and swap571–574/580. All development checks pass; report/raw9bd7c5c. No development32capture. Independent audit pending.

Confirmation actor/config/protocol8b619f7 is staged at `~/topoformer-campaign-01/returns/source-r05-confirmation`;11CPUtests pass. Fresh3backbone fits, fixed endpoints learned/oracle2000 andquery1000;8192fit/1024cal/4096validation/test. Causal2048-event prefix preserves underlying validation identity. Test/balanced32 onlyafterfixedfits. Expected300–450s, requestedcap600. Reviewer preflight and coordinator release pending; NO ACTIVE GPU JOB. Do not launch without parent.

Worker `/root/campaign_composition` received the frozen consumer API but no promotion: normalized workspace1024→R04 categorical accessor→softmax33+normalizedquery2→new1024hidden comparator. Planned finalconsumerfiles r05-confirmation/{seed}/learned.pt do not yet exist. This remains an oracle-first-operation learned scalar-consumer interface.

## Latest audited frontier and next profile

R05 confirmation completed352.54seconds, frozen8b619f7. All36covered validation cells exceed98% (worst4080/4081/4076of4096); causal checks pass. Report/raw/gatesdeedda1, plot8ff1cce, independent audit9cc33af, auditedstatus4fdbf8e. Audit verified513raw+513cached predictions and23disjoint populations. Boundary counts remain weak; at test16 most accessor errors do not change binary answers. Composition worker has API/checkpoint locations but coordinator alone promotes any next execution study.

R06 actor7e77c3d and config are staged immutably at `~/topoformer-campaign-01/returns/source-r06`. Reviewer preflight passed;13CPUtests pass. Registrye14c06e. It compares original16k prefix against balanced315/316-per52-strata selection from onefresh65536-event pool, samebackbone12/head/900updates. No phaseconditioning or architecture change. Fresh original and grid cal/val, no32. Profile uses separate seeds and1024fits; requested60seconds. Main estimate150–240seconds, requested300pendingprofile. NO ACTIVE GPU JOB; await explicit root release.

External return process occupancy cumulatively:613.80(R01–R04)+4.18(R05profile)+68.97(R05development)+352.54(R05confirmation)=1039.49seconds. No additional GPU experiments have run.
