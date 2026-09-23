# S03: optimizer-preserving exposure continuation

Doubling presentations on the selected8,192-construction corpus improved fresh graph components again, but did not acquire complete graphs. This is a development learning curve, not a confirmation or evidence of a universal architectural limit.

| Cumulative presentations | DEVcopy (graph-macro) | DEVtypedF1 raw/cal | DEVorderedF1 raw/cal | DEVexact raw/cal |
|---:|---:|---:|---:|---:|
|65,536|.95768|.53647/.83892|.57624/.85424|0/512,0/512|
|98,304|.97245|.54258/.86680|.66494/.88942|0/512,0/512|
|131,072|.98442|.71833/.89140|.80217/.92627|0/512,0/512|

TRAIN128 diagnostic final copy=.99044, calibrated typed/orderedF1=.89729/.92834, complete0/128. Final DEVpresence510/512 exact, kind17,109/17,337 (357complete sequences), copy12,761/12,972 (363complete sequences), all4,365finite values correct. These marginal improvements do not establish joint graph correctness.

The last interval orderedF1 gain.03685 exceeds the prospectively declared.02extension rule. A continuation to262,144total presentations is therefore justified within the existing search protocol, subject to a separately frozen endpoint/config and root compute release. It must retain the same model/data/optimizer trajectory; no additional frontend or loss is selected from these results.

Source d743464 preserved the parent model, every AdamW state (all inherited step8,192), pair/shuffle generators, order/position, counts and cumulative curriculum. Initial replay matched every TRAIN128/DEV512 raw graph, calibrated edge set and threshold exactly. The first actual resumed batch matched the independent cloned-state index/pair-hash prediction. Historical global RNG was unavailable; the model has no nonzero dropout and its stochastic sampling/shuffling uses the two saved explicit generators. Full manifests retain these checks.

This tranche added65,536presentations,495.14106optimizer seconds and642.37245seconds full process occupancy. Cumulative presentations131,072, unique TRAINgraphs8,192, visits16each, optimizer tokens7,744,288. Width1024,8workspace rows,57,853,781parameters unchanged. PeakCUDA1,224,941,568bytes; processRSS3,830,744KiB. No reserved confirmation predictions.

Raw/calibration/curve/provenance artifacts: `research/results/campaign-01/semantics/s03-current-n8192-131k-dev201`. Immutable full checkpoints remain on GB10 with hashes. Independent raw/provenance audit pending.
