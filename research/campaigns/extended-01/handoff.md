# Resume here

Campaign ACTIVE, not a single-stage stop. Baseline123299a7c5312e1f75019fa491618943f54d0de2; started2026-09-23T19:50:55Z, deadline2026-09-24T19:50:55Z. Ceiling43,200 GPU-process seconds, reserve8,640 for confirmation. Root is sole queue owner; budget.json and queue.json are authoritative. Do not restart completed jobs or launch around an active release.

Coordinator: `.worktrees/campaign`, branch `campaign/extended-01`. Worker branches/worktrees `campaign/{returns,semantics,attention,review}` under `.worktrees/campaign-*`; workers commit, root cherry-picks. Active agents `/root/stage9_returns`, `/root/stage9_semantics`, `/root/campaign_attention`, `/root/stage9_review`. Historical main remains123299a; progress branch is pushed at useful verified milestones. Existing old agents are completed, not competing coordinators.

GB10 SSH `gb10-direct`; CUDApython `~/topoformer-stage8-cuda/bin/python`, CPUpython `~/topoformer-pilot/.venv/bin/python`. No installs/upgrades. Local systemPython lacks torch/pytest; use existing remoteCPUenv for tests. Ten combined campaign tests passed earlier; new worker-specific tests also pass. Final integration tests still required. All3953 baseline tracked blobs preserved by independent initial audit; new work is additive.

## Completed evidence

- R01 profile3.96s, development26.19s. CE900/ridge.01 selected on one historical wide backbone. Fresh three-backbone confirmation95.90s FAILED declared scalar0–16 contract: seed11d16=1000/1024; seeds10/12 pass. Independent raw/cache/normalization/selection/split replay passes; plots/report committed. Historical six-field gates unchanged. No composition promotion.
- R02 nested fresh4096/8192/16384 pools, fixedCE900, selected failing backbone11.72.14s. Fresh2048event d16 accuracy97.85/98.49/98.63%; largerpool incompletely fitted. R03 is registered samecache exposure900/1800/3600, exact900 replay required; calibration-only endpoint selection; no32 exposure. R03 sourcee05ca8f ready, reviewer preflight requested, noGPUrelease yet.
- A01 profile2.54s, five-arm dev36.30s. Context/message/hard100% by10updates. Soft4 argmax path perfect yet N32D8task.246; finite payload mixing is the issue.
- A02 trained/frozen-strength diagnostics31.26s. Frozen4→8 restores100% allfivefreshdevelopmentcells withoutweightchange. Trainable strength initialized8 selected over4/size-adjustment. Strongbaselines retained.
- A03 largest-shape profile2.90s. Fresh confirmation frozen:3pairedseeds201/202/203 × soft4/soft8/context/message/hard;500updates;18cells×1024;primaryN64D16; pairedcorruption/permutation; source754bd3d, wording6a89227. Cap300s perpairedseed,900s total. Reviewerpreflight pending; noconfirmationlaunched. Zero-strength otherarms no-op/NAnotgraphablation. Active/allocated params and isolatedforwardtiming reported.
- S01profile charged14s (12.048inner excludesimports). Audited9728alpha-distinct constructions, no detected targetcontradiction/capacityoverflow; split128/1024/8192trainprefixes+512dev+1024reservedconfirmation. Matched65536presentations each, init201. N128 complete603.831s: TRAIN attributesperfect, exactgraph0/128; replacingedges alone126/128 (2sloterrors). DEVexact0/512,copy46.97%,caltypedF1.628. Independentmainandreplacement audits pass. This is not acampaignstop.

## Current work and next releases

S01-n1024 is RUNNING with cap900s, source2c7e987+launcherfef09e9. Await worker GPUfree/actualoccupancy. Next useful shortjobs: A03pairedseed201 confirmation<=300s and R03<=120s oncepreflightpasses; then S01-n8192<=900s. RemainingA03seeds202/203 afterappropriatequeueboundary. No competing GPUjobs.

S02 frozen-node edgehead continuation is being prepared CPUonly bysemanticworker: cache learnedpublicnodefeatures, preserveexistingedgehead/optimizerstate and BF16parity, freezeallotheroutputs;1200/4800headupdates. Exactedgefit primary, wholegraphceiling126/128. Do not mistake privileged cached-node diagnostic forfreshtextgeneralization. Continue diversityladderbeforefrontendchange.

Return fact-use draft is conditional only: existingthreshold/XORquery consumes learnedscalarreadout, query-only/exactvalue controls, wrong/drop/swappedreturns. R01failedcontract cannotpromote it. A future narrowedcontract mustbeprospective andmatchconsumeddistribution, includingfloat-tailcoverage.

## Reporting discipline

Keep all failedconfirmation evidence. Register fresh followups before outcomes; confirmation never tunes recipes. Known priors versus learned interfaces explicit. A/R study widths1024; semanticactor57.85Mparams/1024/8workspace rows. Direct attention supplies reverse schedule and explicit grounding; no planningclaim. Do not call supportingreturn/semanticstudies programmableattentionevidence. CPUaudits trackedseparately; latestreview receipts include>100ssemantic audit notyetallrolled intobudgetCPUtotal. Check worker commits for pending results/review updates and cherry-pick, never reset their branches.
