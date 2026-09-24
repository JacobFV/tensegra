# A14 measured profile and main budget

Source67dc519 completed the fixed profile: fresh seed1499,100 training updates/1600 presentations, two32-event development cells,32joint events under all five frozen policies, and a separate uninstrumented benchmark. Full-process occupancy22.198607951seconds, internal21.021181339seconds. No confirmation seed has run.

Training excluding development monitoring took1.559903819seconds; monitoring.105004126seconds. Linear training projection is93.59seconds for6000updates. Five instrumented joint-policy forwards total7.764485946seconds for32events, projecting248.46seconds for1024events. Add the two smaller core shapes, three development checks, full public hashing, checkpoint/raw export and variation: estimate360–400seconds per main seed; retain provisional cap480, subject to root authorization. Do not scale the entire profile wall time, which includes a separate benchmark absent from main. Nine A06 engineering references remain estimated45–70seconds/cap120 from prior frozen runs.

The separate benchmark uses a fresh batch16 joint graph with canonical pretokenized record order. Generation/tokenization/oracle setup cost.013882212seconds. Each policy has one separately charged warmup and three bare-forward repeats:

| Policy | Warmup seconds | Three forward seconds |
|---|---:|---|
| Original | .5107 | .4859, .4860, .4828 |
| Both hard | .6545 | .6363, .6384, .6392 |
| Shared soft | .4911 | .4889, .4915, .4941 |
| Shared hard | .4933 | .4923, .4983, .4896 |
| Privileged oracle | .4923 | .4956, .4908, .4936 |

These describe this profile implementation/workload, not statistically established latency superiority or main instrumented cost. Dense one-hot hardening does not imply a sparse implementation. Allocated parameters4,575,275; gradient/nonzero-gradient tensor participation2,412,562. Training+monitor CUDA peak500,482,560bytes; post-cleanup instrumented inference peak1,516,155,392bytes. Bare-forward peaks are1,042,956,288original/sharedsoft,1,445,609,472both-hard and1,044,004,864sharedhard/oracle. Optimizer/gradients are cleared before inference; these scopes remain explicitly distinct.

Frozen inference initial/final tensor hashes agree. Profile joint task counts1/3/2/4/32 of32 and complete suffix0/0/0/0/32 for original/both-hard/sharedsoft/sharedhard/privilegedoracle are retained as timing-population outcomes only. They cannot select a seed, policy or gate. All main seeds, examples, training endpoints and rules remain unchanged.

Raw/benchmark/receipt commit9a42d5db at `research/results/campaign-01/attention/a14-profile`. Full checkpoint and optimizer/RNG state remain remotely at `~/topoformer-campaign01/attention/a14-profile-receipted/results`, with file hashes archived locally. Independent profile audit requested. GPU verified free; no main job launched. Attention accounting1665.929478161seconds includes A08 profile's uncertain60-second upper bound.
