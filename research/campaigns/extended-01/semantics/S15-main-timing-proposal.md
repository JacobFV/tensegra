# S15 fixed mains: measured timing proposal

Both90-second mechanical profiles completed unchanged with exact initial TRAIN replay, equal initial model/AdamW steps, construction schedule/visits and common-example negative-pair stream. Each used20updates,160presentations and two curves of128historical calibration plus64DEV examples. No profile outcomes select a recipe, population, calibration policy or endpoint.

| Arm | Full profile s | Optimizer s | Two evaluation curves s | Projected optimizer4096updates s | Projected four main curves s | Projected full main s | Proposed cap s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control |21.8180|1.7606|8.5852|360.5773|194.5976|578.1192|800|
| Mixed |22.7450|1.5842|9.7539|324.4454|221.0892|568.3482|800|

Optimizer cost scales by4096/20. Each main curve keeps calibration fixed at128 and expands DEV64→2048; the combined evaluation timing estimate is `(profile evaluation seconds / 2) × 4 × (128+2048)/(128+64)`. This uses a per-example approximation because the inherited evaluator reports calibration and DEV jointly; it does not claim calibration scales with DEV or assume calibrated success. Fixed threshold-fitting work included in that rate makes the extrapolation conservative when it dominates, while shape-dependent inference and serialization remain uncertainty. Double the remaining measured full-process overhead to cover four rather than two checkpoint exports and startup. The proposed800-second whole-process cap supplies roughly38–41% margin over this estimate. Costs are estimates, not guaranteed wall time or equal FLOPs.

Profile tokens/nodes/edges were9420/5393/12335 versus8332/4803/10697; both had74common presentations with identical pair hashes. Main matched4096uniquegraphs×8visits gives32768presentations perarm but unequal token/node/edge exposures, which remain recorded. Fixed main4096updates and curves0/1024/2048/4096, all2048DEV, historical TRAIN128 and original S11model+AdamW parent remain unchanged. Main configs are still prepared; independently review and root-allocate caps before creating source-bound frozen files, then require separate serial GPU releases. Profiles are not continuation parents. No main or conditional extension is automatically authorized by this proposal.

Exact timing and manifest hashes are in `research/results/campaign-01/semantics/s15-paired-profile-timing.json`; control archive30b22267, mixed2a97f21b. Source remains34e27d86 with launcher/freezer790bdb75, as in the audited profiles.

## Prospective cap revision before any main

Root revised both allocations to1100seconds after independent review highlighted the unavailable calibration/DEV timing decomposition. The578/568-second forecasts above are conditional average-per-example projections, not separate fixed-calibration measurements. A more conservative forecast scales **all** combined profile evaluation time as though it belonged to DEV: four versus two curves ×2048/64 =64 times the two-curve timing. This deliberately also scales fixed calibration overhead, yielding933.37/971.51seconds including the same optimizer and doubled residual overhead. It is a conservative forecast, not a rigorous runtime bound. Root's1100-second caps provide margin without changing any scientific recipe, exposure, curves, data, calibration or selection rules. Unused800-second v2 configs are preserved; v3 configs/output prefixes bind1100. Exact revised arithmetic is `s15-paired-profile-timing-v3.json`; all mains remain unlaunched until separate root releases and independent audit completion.
