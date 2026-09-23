# Attention track handoff

Current branch: `campaign/attention`. Root is the sole GPU scheduler; never launch without its explicit slot release.

## Completed

- A01 profile: source53aa9f6,2.54 seconds full occupancy.
- A01 development: controller snapshot57b53e3, same model source53aa9f6; five arms500updates,36.30 seconds full occupancy. Raw compact artifacts at `research/results/campaign-01/attention/development-53aa9f6`. Independent reviewer reconstructed175cells and confirmed shared initial state hashes.
- Context/message/hard reach100% on all development cells by10updates. Soft4 has perfect mean-head argmax paths but only24.6% taskaccuracy atN32/D8. Edge mass.626 versuscontext.982. This is a routing-versus-weighted-payload distinction, not evidence of lost identity.
- Eight CPU mechanical tests pass in0.64seconds afterA02instrumentation. Width32onlymechanical fixtures; experimentwidth1024.

## Frozen confirmation in progress

A02 and the largest-condition profile are complete (details below). A03 source is frozen at `27fc238` in `/tmp/campaign-a03-source-final` on GB10. Seed201 completed. Seeds202 and203 remain unrun and await explicit coordinator GPU release after S8192. Do not change their protocols from inspected seed201 results.

Run only after release: `timeout 300 bash research/campaigns/extended-01/attention/run-confirmation.sh SEED` in that frozen directory. Report the external controller occupancy and GPU-free status immediately, before artifact copying or analysis. Compact exports exclude checkpoints; preserve checkpoints non-destructively under `~/topoformer-campaign01/attention/a03-confirmation`.

After all three seeds finish, run `research/tools/campaign_attention_analysis.py` over the complete archive, obtain the independent review, and write the final report. Shared confirmation events across seeds are paired replicates, not additional independent observations.

## Provenance/storage

A01 full checkpoints remain `/tmp/campaign-a01-results` and are also copied to `~/topoformer-campaign01/attention/a01-results-preserved` onGB10. Compact predictions/config/source/checkpointhashes are tracked locally. Source snapshots immutable. No attentionGPUjobcurrentlyrunning.

Historical files unchanged. The backward schedule, explicitnodealignment, functionalrelations and known-keycontext prior are supplied. No learnedplanning or independentsemanticbinding claim. Hard/message routing coincide on clean functionalgraphs. Wrong/missing topology can make clean answers unidentifiable; corruption is not a guaranteed soft-recovery test.

## Confirmation update

A02 completed31.26seconds; trained soft8 and frozen soft4→override8 each reach1.0 across the development cells. Soft4+size adjustment largely repairs size but not depth. Strength8 denotes trainable initialization; the frozen override is exactly8. A03 largest-shape profile completed2.90seconds.

A03 source frozen27fc238, deployed `/tmp/campaign-a03-source-final`. Root releases one paired seed at a time; run `timeout 300 bash research/campaigns/extended-01/attention/run-confirmation.sh SEED` there. Seeds201/202/203 and all15configs are frozen. Soft4 has20finalcells including two frozenoverride8 diagnostics; otherarms have18. Final1024cases/condition, intermediatecurves separate256cases. Root is still sole GPUqueueowner.

Seed201 completedall5arms,78.11seconds, rawcompactarchive223da5c. Seeds202/203 have NOT run. Do not change selection or configs. Reviewer is auditing seed201; analysis/report waits the full prescribed matrix. Root plans202/203 after S8192. Fullprocess receipt `research/results/campaign-01/attention/a03-confirmation/process-201.json`.

Eleven CPUtests pass1.35seconds. Confirmation raw includes clean and supplied topology mass separately, CUDA forward-only latency, and paired node permutations restored to original coordinates. `parameters_with_gradient` is autograd participation, not acquired/functional capacity; last-update nonzero gradient count is separate.

Optional predicted-grounding note97fc6e2 is planning only, not promoted. A03 completion comes first; common immutable key matching may add little beyondStage3. A content-dependent multi-neighbor task is another possible future branch, but no implementation or GPU authority has been granted for it yet.

## Unpromoted next mechanism

`content-selector-proposal.md` is the preferred next design if A03 motivates further work: multiple typed neighbors, public fresh attribute codes, learned content-dependent selection, strong dynamic neighbor-attention baseline. Primary GAT/GATv2 methods have been read. No implementation or GPU work is authorized until A03 completes and the coordinator promotes this branch. `predicted-grounding-options.md` remains an alternative outline; immutable matching alone may duplicate Stage3.

## A03 complete / A04 profile queued

All A03 confirmation seeds finished unchanged:201=78.11s,202=76.17s,203=75.20s. All15 models/raw results are committed through c09e86b. Report/plots6fa7ec2, audit-status correction ba78548. Independent reviewer e20e787 reconstructs456cells and checks all15 checkpoint hashes. One seed203soft4 intermediate payload argmax changes under node permutation; final task and routes stay equal. Report preserves this caveat.

A03: soft8/context/message/hard100% across8clean1024-event cells/all3seeds; soft4 N64D16 about11.59% despite100% diagnostic argmax paths. Frozen strength override8 repairs all soft4 errors. Message passing is roughly twice as fast; no unique attention advantage. Total attention GPU occupancy through A03=302.48seconds. Durable checkpoints under `/home/brandonin/topoformer-campaign01/attention/a03-confirmation`.

Root promoted A04 CPU implementation after full A03. Current source freeze2ba70af, immutable remote `/tmp/campaign-a04-source-2ba70af`. Modules `campaign_attention_selector.py` / `_study.py`, six CPU mechanical tests pass1.43s. Profile config `campaign-a04-profile.json`: width1024,12updates,64 N64/D8/K8 evaluation examples. Root has NOT released its GPU slot yet; S03 is using GPU. Profile cap60s, await explicit release. No A04 model outcome exists. Older unused snapshot fd0937e is superseded, not run.

Profile command after release: `PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/time -f 'PROCESS_OCCUPANCY_SECONDS=%e' timeout 60 ~/topoformer-stage8-cuda/bin/python -m topoformer.campaign_attention_selector_study --config configs/campaign-a04-profile.json --output /tmp/campaign-a04-profile` in frozen source directory. Report GPU free before analysis. Register development budget from profile before four-arm1000-update controller `run-a04.sh` launches. Reviewer requested supplied targets and changed-answer/terminal swap conditionals; all now exported/tested. Optimizer keeps A03 weight_decay1e-4. Context is a keyed neighborhood selector with supplied hard address mask, not an unconstrained graph-token transformer. Exact gather and hard-neighbor attention are one algebraically equivalent strong arm.

## Latest: v2 profile queued after generator-contract audit

A04 profile3.64seconds, development63.01seconds complete; raw6c00dae, report17e0e4c. All three graph arms acquire100% tested clean task accuracy by25updates; no graph stays low. Paired swapped-code changed-answer subset is163/256 and all strong arms correct. No soft advantage. Checkpoints durable `~/topoformer-campaign01/attention/a04`.

**Do not run the old A05 confirmation controller/configs.** Its draft17f062d is superseded-before-run. A generator-only CPU audit found v1 long paths coalesce: ~1.4 terminal identities from128starts after32hops. No A05 model/profile outcome exists. Old draft/status preserved; A04 remains a valid narrow matching result.

Root authorized a separately versioned balanced generator and new screen. Source53bf204 in immutable remote `/tmp/campaign-a05-v2-source-53bf204`, reviewer preflight clear. Generator `block_permutation_v2` uses independent permutation per relation/source-group/target-group. After first instruction exactlyN/K identities remain and all subsequent within-active-group maps are bijective. Three512-graph CPU audits verify exactly16 identities atN128/K8/D32, all same-code start pairs stay distinct. Nine CPU tests pass1.55seconds.

Root GPU slot still pending after S04. Run only after explicit release: from frozen53bf204 dir, `PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/time -f 'PROCESS_OCCUPANCY_SECONDS=%e' timeout 60 ~/topoformer-stage8-cuda/bin/python -m topoformer.campaign_attention_selector_study --config configs/campaign-a05-v2-profile.json --output /tmp/campaign-a05-v2-profile`. Profile12updates plus64largestN128/D32/K8, not model selection. Signal free immediately. Then register measured main budget before four-arm1000-update `run-a05-v2.sh` (separate root release). No new GPU work has started. New development seed501/91Mtrain/92Meval,256cases, originalA04conditions plus long-path and matched frozen content-scale16 (not graphλ) interventions. Source/configs frozen before outcomes. Main preliminary cap300seconds; profile informs estimate.

## Latest: v2 complete; A05b queued, not run

A05-v2 profile3.76s (57432a3), development149.54s (raw7afff11), allfour1000-update arms complete. Final noncoalescingN128/D32/K8 task: soft.671875, exactgather.7421875, keyedcontext.21875, no-graph.1640625. Graph-arm mean-head path diagnostics all1.0. Frozen content-scale16 gives soft/gather1.0 task+all-node+complete suffix trajectories; context task.5234375. IID/throughD8 allstrongarms1.0 from25updates. Report in6704bd8 plus interpretation correctiond47d409. Full attention process occupancy through v2=522.43seconds. Checkpoints durable `~/topoformer-campaign01/attention/a05-v2/{arm}/checkpoint.pt`.

Root authorized A05b frozen address-concentration diagnostic before confirmation. Source6704bd8, remote immutable `/tmp/campaign-a05b-source-6704bd8`. Reviewer preflight clear. Config `campaign-a05b-context.json`: zero updates, checkpointSHA enforced, four512-event newdevelopment cells (seed102M), content learned/16 crossed with address learned/16. Initial/final tensor hashes must agree. Address changes affect both retrieved attributes and payloads (d755312), so do not attribute improvement solely to values. Ten CPU tests pass1.32s.

**No A05b GPU run yet.** Root queues it after S06 main, cap60seconds, estimated10–20seconds. Await explicitrelease and verifyidle. Run from frozen dir with `PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/time -f 'PROCESS_OCCUPANCY_SECONDS=%e' timeout 60 ~/topoformer-stage8-cuda/bin/python -m topoformer.campaign_attention_selector_study --config configs/campaign-a05b-context.json --output /tmp/campaign-a05b-context`. SignalGPUFREE immediately. Then fetchcompactraw and reportallfourpaired outcomes. No automatic confirmation launch.

A06 promotion plan7026124 reserves freshseeds601/602/603, train141M+seed×100k, monitoring142M, final151M; final protocol waits A05b. Keep allstrongbaselines; don't select an artificial softadvantage over diffuse keyed retrieval. Reviewer is auditingA04/v2raw; A03fullaudit e20e787 alreadycomplete. Every earlier stage/artifact unchanged.
