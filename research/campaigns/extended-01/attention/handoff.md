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
