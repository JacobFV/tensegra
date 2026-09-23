# Stage 10: semantic invariance, cross-delay readout, and fixed-set acquisition

Status: belief main study and final independent audits pending. Baseline `60ad5ee`; width1024. Composition and supervision withdrawal remain blocked. This report separates three supporting-interface questions; none directly tests programmable attention.

## Experiment scope

- Beliefs compare content-only matching with IDs retained in the ledger, raw-ID matching with independent randomized training handles, and a correlated-ID reference. The prior and ledger semantics are supplied in every arm; matching and nonempty posterior behavior remain learned.
- Returns freeze the three Stage9 backbones and change only scalar readout fitting. Source-specific decoding and one matched-row shared decoder use fresh train/calibration/test events. Non-value heads remain unchanged.
- Semantics isolate one edge head on eight fixed graphs with supplied node identities. The sole intervention is a predeclared increase from300 to1200 updates. This is fitting, not public-language understanding.

See [design](stage10-design.md), [registry](stage10-registry.json), and [independent review](stage10-review.md) for contracts, source/configuration freezes and stopping rules.

## Belief invariance

Main results pending. The development acquisition check passed for both repairs, but randomized-ID predictions still changed under renaming despite correct final proposals. The main gate therefore retains framewise posterior fidelity, impossible mass and intermediate retractions rather than substituting answer accuracy.

## Returned scalar access across time

The fresh cross-delay matrix shows that a successful decoder at one time need not work later. On test examples with eight distractor rows, averaged descriptively over three frozen backbones:

| Decoder fit | Evaluation delay | Exact scalar accuracy |
|---|---:|---:|
| One-step only |1|98.44%|
| One-step only |32|50.65%|
| 32-step only |32|92.90%|
| Short pool0/1/2/4 |32|75.00%|
| Shared pool0/1/2/4/8/16 |32|89.58%|

The32-step-specific probe is diagnostic: it trains at the evaluated delay. The shared decoder's32-step result is genuine readout-length extrapolation. Short and shared pools each use8192 feature rows from the same2048 distinct events, the same coefficient capacity, train-only standardization and source-support-only calibration. They differ in delay coverage, not total fitting rows. Backbones were originally trained only through delay4; covering longer delays for a decoder does not retrain the backbone.

At32, shared counts are451/512,475/512,450/512; short-pool counts340/512,411/512,401/512. At16, shared counts473/512,488/512,479/512; short439/512,474/512,459/512. These are paired events, not independent observations for every decoder/delay/seed. All scalar coverage, scalar extrapolation and full16-step retention gates fail. Non-value errors persist unchanged.

Delay0 is a distinct difficult readout regime: its own decoder reaches91.28% but transfers to delay1 at16.34%; shared delay0 accuracy is77.34%. The shared linear readout does not uniformly improve every time. No new representation or larger probe was added to rescue these results.

This supports delay-dependent linear accessibility and a benefit from broader readout fitting. It does not prove that information is permanently destroyed, nor uniquely identify query drift versus recurrent mixing versus a nonlinear representation. Persistent memory remains intact. No returned-fact-use or second-crystallization training follows the failed gate.

## Semantic edge fitting

Frozen historical seed10 had five false-positive argument edges with overlapping relation scores. Their local gradients point toward correction, but a nonzero gradient is not a convergence guarantee. The five historical errors are distinct from the six errors in the new seed21 trajectory.

The registered experiment retains the same1024-dimensional privileged node codes, affine head, loss and optimizer. Three fresh initialization seeds20/21/22 follow one1200-update trajectory each;300 and1200 are prespecified endpoints, not selected checkpoints.

| Seed | Raw exact at300 | TRAIN-calibrated exact at300 | Raw exact at1200 | TRAIN-calibrated exact at1200 |
|---|---:|---:|---:|---:|
|20|0/8|8/8|8/8|8/8|
|21|0/8|5/8|8/8|8/8|
|22|0/8|8/8|8/8|8/8|

The six false argument-edge logits in new seed21 cross from3.94–4.09 to−4.56…−3.66. Thus the final success is not merely recalibration: the raw zero threshold also recovers every graph. Every real edge and ordered slot remains correct. All three seeds pass restricted fixed-set acquisition at the declared endpoint.

This supports a finite-budget acquisition explanation for this fixture. It does not establish language understanding, novel-graph transfer, or that historical seed10 would necessarily converge under continued Adam training; its optimizer state was unavailable. There are eight distinct training graphs, not thousands of independent semantic examples. No public-text experiment was launched.

[Full semantic report](stage10-semantic-edge-report.md) and independent audit retain all288 graph decisions,234 threshold records and18 checkpoint hashes.

## Claim boundaries and next decisions

Architectural ID invariance, learned compatibility, readout consistency, calibration and fixed-set fitting are distinct achievements. Exact storage alone does not guarantee successful access; a failed original decoder does not imply absent information. A finite300-update failure is not a theoretical representational limit.

These tracks do not modify QK geometry. They neither validate programmable attention nor make every runtime-interface gate a universal prerequisite for studying it. A future attention comparison can have its own matched-input claim and prerequisites. Runtime composition and supervision withdrawal remain separate blocked decisions.

## Verification and resources

The immutable source snapshot passed562 tests and six subtests. Final historical-byte and raw-metric review is pending. All experiments use the existing GB10 environment; the three-GPU-hour ceiling is retained. Per-track reports retain parameters, feature/memory allocations, distinct examples, exposure, timing and hashes. Final compute accounting will be completed after the belief matrix finishes.
