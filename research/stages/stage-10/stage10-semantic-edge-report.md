# Stage 10 semantic edge fitting

**Restricted acquisition passes after the prespecified exposure increase.** All
three new initializations reconstruct every one of the eight fixed TRAIN graphs
at 1200 updates, using either the raw zero threshold or TRAIN-fitted relation
thresholds. This is not language understanding or generalization to new graphs.
No architecture, head, objective or optimizer changed.

## Historical failure localization

Read-only CPU inference inspects the frozen Stage 9 seed10 additive/
edge-conditional affine decoder. The five residual mistakes are false-positive
`argument` edges. Four connect one predicate to the four arguments of a different
fact; the fifth connects a fact predicate to a query variable. Every graph node
is an explicitly supplied distinct one-hot code here, so these are edge-score
errors, not ambiguity about public language.

The maximum negative argument score is4.1030 while the minimum positive is3.8644
(gap−.2386). A single threshold per relation cannot remove that ranking overlap.
The full edge-loss gradient norm is.000367. For each erroneous edge the
infinitesimal negative-gradient direction lowers its logit by approximately
.000153per unit step. This rules out an exactly zero local correction signal,
but does not prove what Adam or a longer trajectory will do. It is not an actual
optimizer update. Historical raw binary predictions survive at0/25/100/300;
intermediate logits, gradients and optimizer states were not archived, so no
such historical trajectory is invented.

[Full frozen diagnosis](../../results/stage10/semantic-edge/frozen-seed10.json)
records node identities, labels, margins, weighted gradients and available raw
prediction trajectories, with the checkpoint hash. The historical five errors
are distinct from the six new seed21errors below.

## Frozen paired experiment

Only additive/edge-conditional is retained; earlier slot arms shared the same
edge behavior and are not independent replications. New initializations20/21/22
train the unchanged affine decoder with node width1024, output rank128 and
1,904,450parameters. The eight graphs and their privileged node identity codes
are unchanged from Stage 9's confirmation fixture. Presence, node attributes and
copy identities remain supplied; only typed edges and slots are reconstructed.
This oracle-node diagnostic has no recurrent workspace or attention memory.

Each initialization follows one fixed1200-update Adam trajectory, lr.01, on all
eight graphs per update. The primary paired endpoints were declared as300 and
1200 before outcomes;0/25/100/600 are observational checkpoints. No checkpoint
was selected and no run stopped early. Relation thresholds at every checkpoint
use only these TRAIN graphs, minimizing entry-classification errors with a
lowest-threshold tie rule. Raw scores are reported in parallel.

| Initialization | Raw exact at300 | TRAIN-calibrated exact at300 | Calibrated edge errors at300 | Raw exact at1200 | Calibrated exact at1200 |
|---|---:|---:|---:|---:|---:|
|20|0/8|8/8|0|8/8|8/8|
|21|0/8|5/8|6|8/8|8/8|
|22|0/8|8/8|0|8/8|8/8|

At600 updates all calibrated graphs are correct, while raw exact counts are1/8,
4/8and4/8. This checkpoint was not selected as the primary result. At the frozen
1200 endpoint the raw decoder also fits every graph, so success in this fixture
no longer depends on threshold fitting. Neither result establishes calibrated
probabilities outside the fixed TRAIN population.

The six new seed21errors at300 are all false-positive argument edges. Their
scores fall from3.94–4.09 at300 to−4.56…−3.66 at1200. Thus the observed repair
is not merely a changing threshold: their actual logits cross the raw zero
boundary. The full graph also remains correct, excluding a repair that simply
rejects every edge. Exact score trajectories at0/25/100/300/600/1200 are saved;
[the focused error analysis](../../results/stage10/semantic-edge/seed21-error-trajectories.json)
reports contemporaneous and frozen300-threshold decisions separately.

## What this establishes

The remaining Stage 9 failure was not a proven inability of this decoder to
represent the fixed targets. In the tested paired runs, greater exposure alone
removes the residual ranking errors and the raw-decoder error. No alternative
representation or new symbolic mechanism was necessary for this tiny oracle
fixture. This is evidence about finite-budget acquisition, not a universal
exposure prescription or convergence theorem. The experiment does not replay
historical seed10's missing Adam state, so it does not claim that those exact
five historical errors were repaired by continuation.

Gate: **passed_restricted** for complete fixed-set fitting at1200 across every
prescribed new seed. Only eight distinct TRAIN graphs are used; repeated
presentations and repeated checkpoints are not independent graph samples.
The inherited language/transfer gates remain untested and unchanged. Public-text
scaling, runtime composition and supervision withdrawal were not attempted.
This track tests a supporting semantic interface, not programmable attention.

## Resources, artifacts and verification

Main exposure is28,800graph presentations and3,600optimizer updates, with eight
unique graphs total. Recorded training time is45.11seconds; model-run walltime
is49.83seconds including checkpoint/export work. Mechanical cold/warm profiles
are separate and were used for runtime estimates only. GPU allocated memory,
process RSS and actual node dimension are recorded independently in the
[per-seed summary](../../results/stage10/semantic-edge/exposure-summary.json).
Existing GB10/PyTorch CUDA dependencies were reused without upgrades.

All18checkpoints remain immutable at
`gb10-direct:~/topoformer-stage10-semantic/results/stage10-semantic-edge-exposure/`.
The repository includes exact FP32 score arrays, targets, predictions, thresholds,
loss curves, source snapshots and configuration/data/checkpoint hashes. Gzip
preserves the full score data; no score quantization or selected-error-only export
substitutes for the raw record. One focused mathematical gradient test passed.
Independent raw-metric/checkpoint verification is pending at this report revision.
