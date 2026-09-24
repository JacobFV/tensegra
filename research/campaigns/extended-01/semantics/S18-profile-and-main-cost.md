# S18 profile timing and prospective development budget

Profile source35e94fce/config418864b3 completed both fixed20-update arms and all scheduled evaluations, exit0/no timeout. GNUtime outside the bound GNUtimeoutKILL180 recorded86.52seconds at0.01second precision; conservatively charge86.53. Inner wrapper86.497563seconds, child83.389818, source/data preflight2.614926. GPU was empty before any timing analysis. No profile outcome or threshold selected any recipe, checkpoint, arm, width, policy or exposure.

Immutable local archive is `research/results/campaign-01/semantics/s18-profile/`: raw calibration NPZ/evaluation exports, all manifests, receipts, outer command/utility hashes, and file-hash inventory. Four model+AdamW checkpoints remain in the read-only remote profile namespace under `results/s18-profile-v1/{context,workspace_control}/model-u{24576,24596}.pt`; `retained-checkpoints.json` records exact paths/bytes/SHA256 and all match the profile manifests. Checkpoint bytes total roughly2.7GB and are not duplicated into git; full prediction/calibration artifacts are locally archived. Profile model outcomes remain unexamined for study selection.

| Main projection term (seconds) | Context | Workspace control |
|---|---:|---:|
|4096 optimizer updates from measured20|402.055|506.762|
|Four checkpoints × both128calibration/TRAIN panels and post-state checks|24.528|39.625|
|Four checkpoints × both full2048DEV passes|213.824|349.085|
|Export, conservatively scaling entire measured stage|433.271|1074.820|
|Four full checkpoint writes/hashes|4.577|4.318|
|Model/AdamW setup|1.119|0.364|
|Subtotal|1079.374|1974.973|

Arithmetic is fixed: optimizer multiplier4096/20=204.8; calibration/posthash and checkpoint-export multiplier4/2=2; DEV multiplier(2048/64)×(4/2)=64. Both matched and historical policies are included. Export also receives multiplier64 because the observed phase combines fixed calibration NPZ compression and variable DEV prediction export; this intentionally overcounts the fixed part rather than pretending it is all measured per-example work. Profile exports at control checkpoint0 were notably slower than later exports; all are retained in the projection, with no outcome-based omission. These are planning estimates, not deterministic bounds or measured main runtimes.

The training timer excludes dataset/target/pair preparation and stream hashing. Add an explicit **300-second allowance** for those unprofiled8192combined-update costs (~36.6ms/update), plus **100seconds** for full raw-policy replay reads, setup/preflight/hash/process tails and other unmeasured overhead. These allowances are engineering reserves, not measurements. Two-arm main planning estimate is3454.347seconds; propose a **4000-second hard whole-process cap**, leaving545.653seconds additional margin. All scientific settings, both4096-update arms, all curves and fixed1024width remain unchanged. No automatic extension, missing-arm rescue, checkpoint selection or shortened exposure is authorized on failure.

Separately propose a **240-second frozen-reference cap** for the three early original checkpoints. Existing S17 full2048DEV+128matchedTRAIN original-actor inference measured50.244730seconds (control) and46.721313seconds (mixed), including setup, calibration, export and replay checks. Three times the slower existing endpoint is150.734189seconds; add35seconds reference setup/checkpoint/source/process allowance. This185.734189-second planning estimate leaves54.265811seconds within240. S18 reference does no training, does not reload/re-evaluate final baseline, and preserves the already available S17 final TRAIN/DEV artifacts. Reused original trajectory's484.141408-second historical work remains disclosed separately.

Combined proposed new development caps4240seconds (70.67minutes) fit the≤1.5hour target and preserve the root-managed confirmation reserve. Profile's86.53seconds is already separately charged. Main and reference configs remain unlaunchable until prospective cap/source freezes, independent review and explicit root release; no GPU main/reference call occurred during this analysis.
