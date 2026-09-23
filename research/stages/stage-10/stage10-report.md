# Stage 10: semantic invariance, cross-delay readout, and fixed-set acquisition

Experiments complete; final independent verification in progress. Baseline `60ad5ee`; width 1024. Composition and supervision withdrawal remain blocked. This report separates three supporting-interface questions; none directly tests programmable attention.

## Experiment scope

- Beliefs compare content-only matching with IDs retained in the ledger, raw-ID matching with independent randomized training handles, and a correlated-ID reference. The prior and ledger semantics are supplied in every arm; matching and nonempty posterior behavior remain learned.
- Returns freeze the three Stage9 backbones and change only scalar readout fitting. Source-specific decoding and one matched-row shared decoder use fresh train/calibration/test events. Non-value heads remain unchanged.
- Semantics isolate one edge head on eight fixed graphs with supplied node identities. The sole intervention is a predeclared increase from 300 to 1200 updates. This is fitting, not public-language understanding.

See [design](stage10-design.md), [registry](stage10-registry.json), and [independent review](stage10-review.md) for contracts, source/configuration freezes and stopping rules.

## Belief invariance

Both repairs pass every registered validation cell across all three initialization seeds: 90 required N8/N16 cells and 45 separately reported N32 cells per arm. Untouched test data independently give the same 90/90 required and 45/45 additional-cell counts for both repairs. The correlated-ID reference passes only 78/90 required and 39/45 additional cells on each split; every seed fails the expanded contract, while all original conditions pass.

The content-only arm retains IDs as ledger addresses but zeros their neural feature coordinates. Its posterior is exactly unchanged under consistent observation-ID renaming. This invariance is architectural; content compatibility is learned. Independent randomized-ID training also acquires the mean posterior contract, but does not supply exact invariance.

A concrete randomized-training tail illustrates the distinction. For seed20, test N8 renaming changes an individual posterior coordinate by as much as .66490. Twelve of512 episodes exceed .01 maximum change, three exceed .05, and two exceed .1. All final argmaxes remain correct. In the largest case all renamed IDs were seen in training; the error is an intermediate posterior that spreads mass onto two impossible ordinary candidates before later evidence resolves the answer. This descriptive tail analysis does not retrospectively alter the registered mean-metric gate.

The supplied prior and exact insertion/removal ledger are common to all arms. No result here establishes learned retraction bookkeeping in the unrestricted recurrent comparator, open-world probability semantics, or language grounding. The observed role-separation advantage concerns opaque observation handles in this typed, noiseless interface.

## Returned scalar access across time

The fresh cross-delay matrix shows that a successful decoder at one time need not work later. On test examples with eight distractor rows, averaged descriptively over three frozen backbones:

| Decoder fit | Evaluation delay | Exact scalar accuracy |
|---|---:|---:|
| One-step only |1|98.44%|
| One-step only |32|50.65%|
| 32-step only |32|92.90%|
| Short pool0/1/2/4 |32|75.00%|
| Shared pool0/1/2/4/8/16 |32|89.58%|

The32-step-specific probe is diagnostic: it trains at the evaluated delay. The shared decoder's32-step result is genuine readout-length extrapolation. Short and shared pools each use 8192 feature rows from the same 2048 distinct events, the same coefficient capacity, train-only standardization and source-support-only calibration. They differ in delay coverage, not total fitting rows. Backbones were originally trained only through delay 4; covering longer delays for a decoder does not retrain the backbone.

At 32, shared counts are451/512,475/512,450/512; short-pool counts 340/512,411/512,401/512. At 16, shared counts 473/512,488/512,479/512; short 439/512,474/512,459/512. These are paired events, not independent observations for every decoder/delay/seed. All scalar coverage, scalar extrapolation and full16-step retention gates fail. Non-value errors persist unchanged.

Delay0 is a distinct difficult readout regime: its own decoder reaches 91.28% but transfers to delay 1 at 16.34%; shared delay 0 accuracy is 77.34%. The shared linear readout does not uniformly improve every time. No new representation or larger probe was added to rescue these results.

This supports delay-dependent linear accessibility and a benefit from broader readout fitting. It does not prove that information is permanently destroyed, nor uniquely identify query drift versus recurrent mixing versus a nonlinear representation. Persistent memory remains intact. No returned-fact-use or second-crystallization training follows the failed gate.

## Semantic edge fitting

Frozen historical seed 10 had five false-positive argument edges with overlapping relation scores. Their local gradients point toward correction, but a nonzero gradient is not a convergence guarantee. The five historical errors are distinct from the six errors in the new seed 21 trajectory.

The registered experiment retains the same 1024-dimensional privileged node codes, affine head, loss and optimizer. Three fresh initialization seeds 20/21/22 follow one 1200-update trajectory each; 300 and 1200 are prespecified endpoints, not selected checkpoints.

| Seed | Raw exact at 300 | TRAIN-calibrated exact at 300 | Raw exact at 1200 | TRAIN-calibrated exact at 1200 |
|---|---:|---:|---:|---:|
|20|0/8|8/8|8/8|8/8|
|21|0/8|5/8|8/8|8/8|
|22|0/8|8/8|8/8|8/8|

The six false argument-edge logits in new seed 21 cross from 3.94–4.09 to −4.56…−3.66. Thus the final success is not merely recalibration: the raw zero threshold also recovers every graph. Every real edge and ordered slot remains correct. All three seeds pass restricted fixed-set acquisition at the declared endpoint.

This supports a finite-budget acquisition explanation for this fixture. It does not establish language understanding, novel-graph transfer, or that historical seed 10 would necessarily converge under continued Adam training; its optimizer state was unavailable. There are eight distinct training graphs, not thousands of independent semantic examples. No public-text experiment was launched.

[Full semantic report](stage10-semantic-edge-report.md) and independent audit retain all 288 graph decisions, 234 threshold records and 18 checkpoint hashes.

## Claim boundaries and next decisions

Architectural ID invariance, learned compatibility, readout consistency, calibration and fixed-set fitting are distinct achievements. Exact storage alone does not guarantee successful access; a failed original decoder does not imply absent information. A finite 300-update failure is not a theoretical representational limit.

These tracks do not modify QK geometry. They neither validate programmable attention nor make every runtime-interface gate a universal prerequisite for studying it. A future attention comparison can have its own matched-input claim and prerequisites. Runtime composition and supervision withdrawal remain separate blocked decisions.

## Verification and resources

The immutable source snapshot passed 562 tests and six subtests. No package or test source changed afterward. Final historical-byte and raw-metric review is in progress. All experiments used the existing GB10 environment. Recorded job timers total about 21.2 minutes; conservative accounting charges 25 minutes, below the 180-minute ceiling. No jobs remain. [Compute accounting](stage10-compute.json) and per-track manifests retain dimensions, parameters, feature/memory allocations, distinct examples, exposure, timing and hashes.
