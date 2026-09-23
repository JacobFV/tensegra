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
