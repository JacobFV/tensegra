# Stage 10: bounded semantic edge acquisition

## Frozen failure localization

Historical seed 10, additive/edge-conditional affine decoder, is inspected without
training. All five errors are false-positive `argument` edges. The minimum true
positive score is 3.8644 and maximum negative score 4.1030: within-relation ranking
overlap remains, so another single threshold cannot eliminate all five. The
full-loss negative-gradient direction reduces each erroneous score locally;
edge-gradient norm is 0.000367. These are infinitesimal SGD diagnostics, not actual
Adam steps or a proof of convergence. Historical raw binary predictions exist at
0/25/100/300 updates; earlier logits, gradients and optimizer checkpoints do not.
Four historical slot arms share their edge behavior; they are not four independent
replications. This stage retains only additive/edge-conditional.

## Exposure-only comparison (frozen before outcomes)

Hypothesis: the residual fixed-set ranking failures reflect incomplete acquisition
within 300 updates and can become seed-robust with a prespecified longer budget.
The single intervention is optimizer exposure: checkpoints 300 versus 1200 on the
same trajectory. No head, loss, representation, optimizer or threshold policy
changes. Three fresh paired initializations 20/21/22; Adam lr .01; affine decoder,
node width 1024, rank 128; same eight fixed graphs generated from seed 9001000.
Development/profile initialization 201 is separate. The fixture has two binding,
three set and three unification graphs and is not an unseen-structure test.

Inputs are privileged distinct graph/node one-hot identities padded to width
1024; node attributes/copy targets are supplied. Learned heads produce directed
relation existence and ordered slots. All graph labels used by the loss and
calibration are TRAIN labels. At each declared observation checkpoint
0/25/100/300/600/1200, one threshold per relation minimizes TRAIN classification
errors, with lowest-threshold tie breaking. No-positive relations establish no
positive recall. Raw zero-threshold metrics remain parallel. No intermediate
checkpoint is selected and no run stops early for success.

Primary comparison: complete graph fitting at 300 versus 1200 updates, separately
for each paired seed. Advancement requires all eight graphs exact at 1200 in all
three seeds under the calibrated contract. Raw exact fitting is a separate
outcome. This is restricted acquisition, not a 512-example generalization gate.
No fresh data, language transfer, runtime composition or supervision withdrawal
is tested. Failure at the declared cap stops dependent work; no automatic extra
steps or architectural repairs.

Exposure: 1200 updates × 8 graphs × 3 seeds = 28,800 graph presentations; eight
unique graphs total. Parameters are 1,904,450. There is no recurrent workspace or
attention memory in this decoder-isolation fixture; the 1024-dimensional node
input standard remains explicit. Rank is an output scorer parameter.

Profile: three updates, separate seed 201, same graph set and full raw export,
maximum 30 seconds. Main estimate under two GPU minutes, hard five minutes,
within the track's 30-minute ceiling. Profile measurements precede launch. Exact
FP32 edge scores, labels and calibrated/raw outputs are losslessly compressed at
every checkpoint to support per-error trajectories and independent replay.
Source/config/data/checkpoint hashes, optimizer losses, parameter count, CUDA
allocated memory, process RSS and training/wall time are retained separately.

## Measured resource freeze

On the configured GB10 CUDA environment, three cold updates took0.197training
seconds and1.070wall seconds. A separately labeled30-update mechanical profile
(seed201, outcomes not used for model selection) took0.566training seconds and
1.311wall seconds. This projects approximately68training seconds for the3600
main updates, plus checkpoint/export overhead. Main budget remains hard5minutes,
expected under2minutes. CUDA allocation173MB; process RSS1.72millionKiB is
recorded separately. No further profiling or recipe selection follows.
