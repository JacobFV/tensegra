# Attention track handoff

Current branch: `campaign/attention`. Root is the sole GPU scheduler; never launch without its explicit slot release.

## Completed

- A01 profile: source53aa9f6,2.54 seconds full occupancy.
- A01 development: controller snapshot57b53e3, same model source53aa9f6; five arms500updates,36.30 seconds full occupancy. Raw compact artifacts at `research/results/campaign-01/attention/development-53aa9f6`. Independent reviewer reconstructed175cells and confirmed shared initial state hashes.
- Context/message/hard reach100% on all development cells by10updates. Soft4 has perfect mean-head argmax paths but only24.6% taskaccuracy atN32/D8. Edge mass.626 versuscontext.982. This is a routing-versus-weighted-payload distinction, not evidence of lost identity.
- Eight CPU mechanical tests pass in0.64seconds afterA02instrumentation. Width32onlymechanical fixtures; experimentwidth1024.

## Next, already registered

A02protocol: `a02-protocol.md`. Source83b4a6a (code), deployed full snapshot `/tmp/campaign-a02-source-a1ebaa2`; controller `run-a02.sh`. Root has approved scope but has NOT released the GPU slot yet. S01-n128 is currently using the GPU. A02 must run under outertimeout120.

Three fresh seed102 training arms:soft4,soft8,sizeadjust;500updates. Three frozenA01soft4 interventions:unchanged,override8,sizeadjust. Fresh devseeds12020000/12030000. Configs `campaign-a02-*.json`. Outputs new `/tmp/campaign-a02-results`. No A02outcomes exist yet.

AfterA02: choose soft setting by registered rule, profile largestN64/D16 evaluation with a mechanical seed (never confirmation), then register three-seed confirmation including soft4reference,bestsoft,context,message,hard. Primary cells1024fresh examples. Targetedcorruption cellsN16/D4andN64/D16 share underlyingevents via `data_group`. Need pairednodepermutation restoration/control and actualforward latency before final confirmation.

## Provenance/storage

A01 full checkpoints remain `/tmp/campaign-a01-results` and are also copied to `~/topoformer-campaign01/attention/a01-results-preserved` onGB10. Compact predictions/config/source/checkpointhashes are tracked locally. Source snapshots immutable. No attentionGPUjobcurrentlyrunning.

Historical files unchanged. The backward schedule, explicitnodealignment, functionalrelations and known-keycontext prior are supplied. No learnedplanning or independentsemanticbinding claim. Hard/message routing coincide on clean functionalgraphs. Wrong/missing topology can make clean answers unidentifiable; corruption is not a guaranteed soft-recovery test.

## Confirmation update

A02 completed31.26seconds; trained soft8 and frozen soft4→override8 each reach1.0 across the development cells. Soft4+size adjustment largely repairs size but not depth. Strength8 denotes trainable initialization; the frozen override is exactly8. A03 largest-shape profile completed2.90seconds.

A03 source frozen27fc238, deployed `/tmp/campaign-a03-source-final`. Root releases one paired seed at a time; run `timeout 300 bash research/campaigns/extended-01/attention/run-confirmation.sh SEED` there. Seeds201/202/203 and all15configs are frozen. Soft4 has20finalcells including two frozenoverride8 diagnostics; otherarms have18. Final1024cases/condition, intermediatecurves separate256cases. Root is still sole GPUqueueowner.

Seed201 completedall5arms,78.11seconds, rawcompactarchive223da5c. Seeds202/203 have NOT run. Do not change selection or configs. Reviewer is auditing seed201; analysis/report waits the full prescribed matrix. Root plans202/203 after S8192. Fullprocess receipt `research/results/campaign-01/attention/a03-confirmation/process-201.json`.

Eleven CPUtests pass1.35seconds. Confirmation raw includes clean and supplied topology mass separately, CUDA forward-only latency, and paired node permutations restored to original coordinates. `parameters_with_gradient` is autograd participation, not acquired/functional capacity; last-update nonzero gradient count is separate.

Optional predicted-grounding note97fc6e2 is planning only, not promoted. A03 completion comes first; common immutable key matching may add little beyondStage3. A content-dependent multi-neighbor task is another possible future branch, but no implementation or GPU authority has been granted for it yet.
