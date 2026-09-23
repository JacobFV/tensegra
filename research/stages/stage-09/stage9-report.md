# Stage 9: representation contracts, localization, and minimal repairs

Stage 9 improves attribution, not autonomous competence. A supplied empty-evidence prior repairs the original protected-belief contract; learned return readout improves substantially; semantic calibration recovers much of a fixed-set decoding failure. Expanded beliefs, exact return retention, and all-seed semantic acquisition still fail. Public-text training, composition, and supervision withdrawal remain blocked.

Baseline: `0f44e13`. Experimental workspace width: **1024**. Versioned Stage 9 implementations preserve historical source and results. See the [design](stage9-design.md), [registry](stage9-registry.json), [gate decisions](stage9-gates.json), [compute accounting](stage9-compute.json), and [independent audit](stage9-review.md).

## Outcomes

| Interface | Observation | Decision |
|---|---|---|
| Protected beliefs, original matrix | Supplied empty-ledger prior passes all48 validation cells across three frozen initialization seeds | Restricted architectural-contract pass; not learned prior calibration |
| Expanded belief contract | ID/content combinations and intermediate retraction uncertainty remain brittle | Failed |
| Workspace-only return reconstruction | New CE readout averages99.02% exact scalar at one step; original85.22% | Improved learned readout, not complete retention |
| Return at16 updates | Original74.41%, ridge89.19%, CE84.90% exact scalar | All9 seed/readout gates fail |
| Semantic head realizability | Additive margin contradiction reproduced; interaction and conditional objectives express required fixtures | Restricted mathematical/fixture pass |
| Fixed-node semantic acquisition | Prospective TRAIN-calibrated graphs: seed10=6/8, seeds11/12=8/8, identical across four arms | All-seed gate fails; public-text acquisition blocked |

Means are descriptive, not substitutes for seedwise gates. Return percentages shown use test examples with eight distractor rows; gates retain both prescribed distractor conditions. Required belief/return cells contain512 examples; semantic fixed-set fixtures contain only8 graphs and make no generalization claim.

## 1. Is the belief failure the prior, learned null inference, or both?

For the protected model on the original matrix, the dominant recorded defect is the specified empty-evidence prior. Ordinary candidate logits are equal; learned null mass `u` produces posterior L1 exactly `2u`, verified within1.01e-7. Frozen gradients are nonzero: the objective supplies pressure to remove null mass, with the initial-frame contribution averaged across frames. This establishes the error mechanism and local gradient, not why the optimizer historically stopped there.

The minimal A2 rule supplies the public uniform non-null prior whenever the active observation ledger is empty, including after full retraction. It changes no nonempty posterior. Both comparators receive equivalent public empty-ledger metadata. Protected A2 passes48/48 original validation cells; recurrent A2 still fails10/48. Expanded conditions fail for both. Recurrent residual errors often concern ordinary candidates, not null. Protected ID failures expose unseen ID/content combinations even for an ID seen during training. A new null head is therefore not justified as a universal repair.

Duplicate observation IDs are idempotent ledger entries; equal content under distinct IDs creates distinct ledger entries. In this deterministic noiseless benchmark, repeating the same constraint leaves the target posterior unchanged; independent noisy measurements are not modeled. This programmed bookkeeping and supplied closed-world prior do not establish open-world uncertainty calibration. [Belief report](stage9-belief-report.md)

## 2. Where does exact scalar information first become unreliable?

The affine scalar field code preserves all33 half-unit values across fresh nuisance contexts:528/528 in all18 frozen checkpoints. The correctly selected facet after mixing is likewise528/528. Actual affine encodings do not exhibit the hypothesized magnitude collapse. Merged-memory linear probes are weaker; that is a limitation of the tested readout, not proof of absent information.

The original workspace decoder is already unreliable at initial retrieval. Frozen-backbone probes and separately trained readout replacements recover substantial early accuracy without changing memory, encoder, attention, or recurrence. The first demonstrated failure is therefore the learned retrieval/workspace/readout boundary; the experiment does not uniquely assign all initial error to attention versus representation versus decoding.

The two permitted repairs were a categorical ridge refit and a cross-entropy refit. Three fresh paired backbones were trained with the original0/1/2/4 delays. Readout training used those same short delays;16/32 remain recurrent-length extrapolation. CE reaches99.02% one-step scalar accuracy, but only84.90% at16; ridge reaches89.19% at16. All full retention gates fail, including non-value failures unchanged by scalar-head replacement. [Return report](stage9-return-report.md)

## 3. Does recurrence lose information, retrieve it poorly, or expose decoder mismatch?

Decoder mismatch explains a substantial recoverable fraction. Delay-specific probes recover more late-step scalar information than the original head, while a single short-delay-trained decoder remains brittle at longer recurrence. This supports length-dependent accessibility/interference as well as readout mismatch. It does not distinguish irreversible loss from query drift or a changing representation uniquely. Persistent exact storage remains intact.

Historical paired events—not differences of marginal accuracies—give facet-persistent one-to16-step scalar counts: correct→correct1186, correct→wrong188, wrong→correct28, wrong→wrong134 across1536 seed/example replicates. Full joint counts are1134/236/27/139. Seed replicates share underlying events and are not1536 unique events. New per-field, non-value-joint, conditional-value, confusion, signed-error and intervention outcomes are retained. Corruption scores distinguish original facts from actually supplied facts. Release/overwrite remain historical OOD, not trained lifecycle competence.

## 4. Does fixing the slot objective/head improve complete acquisition?

The additive margin fixture is impossible because diagonal and cross-pair sums coincide. This is an objective/head mismatch, not proof that final graph decoding is impossible: a separate edge head can suppress non-edges. Edge-conditional slot supervision removes that redundant requirement; interaction scores express the fixture.

In privileged fixed-node fitting, every arm learns real-edge slots correctly. Raw final graph recovery remains0/8. A subsequent diagnostic found within-relation score separation: training-only relation thresholds recover8/8 development graphs. Consequently, the original raw failure cannot be described simply as absent graph information.

A separately frozen prospective confirmation uses three fresh initialization seeds, four head/objective arms, final300-update checkpoints, and deterministic training-only relation thresholds. Seeds11/12 recover8/8; seed10 recovers6/8 with five remaining argument-edge errors. Slot changes do not change complete recovery within a seed. Calibration is consequential, but complete acquisition is not robust across all prescribed seeds. The old raw gate remains failed; the new calibrated gate also fails. No public-text run follows. [Semantic report](stage9-semantic-report.md)

## 5. Are semantic failures model behavior, coverage, or unidentifiable targets?

The finite observability audit examines10,128 canonical graphs and30,384 surfaces. It finds no observed incompatible public-input targets, tokenizer/hash collisions, uncopyable required identities, target-vocabulary overflow, graph capacity overflow, or conflicting pair-slot targets in that support. This is not a proof for every possible generator output.

Coverage is materially uneven: only four binding constructions;120 training set-operation constructions, with24 held out. Missing fresh binding-family coverage cannot count as transfer success. Surface diversity is not canonical structural diversity. The prospective fixed fixture overlaps earlier canonical constructions and is explicitly acquisition-only.

Observed failures include model/readout/calibration behavior under a representable supplied-node contract. Broad semantic transfer remains unestablished because fresh public-text acquisition was blocked and available family diversity is limited. No generator semantics were silently expanded.

## 6. Which conclusions concern programmable attention?

These experiments concern supporting belief and neural/runtime interfaces. They do not test a new structural-attention intervention and cannot validate metric programming. Programmed guarantees (ledger/prior), learned matching/readout, calibration, generalization, and composed competence remain separate categories. No composed competence is claimed.

## Reproducibility, limits, and stopping

Historical numerical reconstruction is distinguished from fresh frozen-checkpoint inference. Independent audits reconstruct192 historical belief cells,1008 return rows,18,944 semantic rows, plus Stage9 belief/probe/repair/calibration outcomes. Frozen checkpoint and coefficient hashes are verified; calibration is independently replayed on CPU. The audit confirms all1747 baseline files byte-identical.

Model dimensions, parameter counts, memory allocation, presentations, unique examples, runtime and source/config/data hashes are recorded per track. Equal parameter count is not equal attention cost. New studies use paired seeds and validation-only selection; inspected historical tests serve diagnosis only. Privileged probes and fixed-node results cannot pass deployable interface gates.

Recorded experiment timers total roughly18 minutes; conservative accounting reserves2000 seconds including profiles/export overhead, below the10800-second ceiling. CPU audits are separate. No jobs remain. The three-GPU-hour limit is a ceiling, not an obligation to continue after failed gates.

Recommendation: retain composition and withdrawal blocks. Any next stage should separately propose an ID-invariant evidence interface, diagnose late-workspace readout accessibility, or establish seed-robust calibrated edge acquisition. Stage9 does not authorize or implement those further changes.
