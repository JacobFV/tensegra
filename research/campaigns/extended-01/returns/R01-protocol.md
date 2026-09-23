# R01: compose broader backbone training with a stronger scalar readout

## Hypothesis and prior evidence

Stage 10 improved readout transfer using a shared categorical ridge consumer on frozen short-trained backbones. Stage 11 improved long-delay non-value retention through broader backbone training, but left scalar errors. R01 tests their simplest combination before changing representations or memory.

The initial DEVELOPMENT experiment uses only frozen Stage 11 wide-continuation checkpoint 10. Backbones and all non-value heads remain frozen. Public typed return events and protected persistent memory are supplied, as before. Privileged exact scalar labels supervise only a replacement scalar decoder from the same normalized 1024-dimensional workspace row. This is supporting-interface research, not a structural-attention intervention.

## Prespecified first development comparison

Fresh fit seed 20100101 supplies 4,096 unique events, each at delays 0/1/2/4/8/16 with two distractors: 24,576 feature rows. Calibration seed 20200101 and validation seed 20300101 each supply 1,024 different events. Every evaluation event is paired across these six delays and two/eight distractors. Individual event hashes enforce disjointness. Delay 32 is not captured or exposed during development; it cannot select a recipe.

Arms:

1. The unchanged Stage 11 scalar classifier.
2. A categorical ridge consumer with 33,825 coefficients, train-only standardization, and regularization selected from .01/.1/1/10.
3. A same-capacity CE consumer using the same standardized features and labels. Warm-start by algebraically transforming the original head so its initial logits are preserved. AdamW learning rate .003, 1,000 updates, batch256, fixed sampling seed. Evaluate calibration at steps0/100/.../1000 and retain the best checkpoint.

Both consumers select using the minimum exact-correct count over covered calibration cells, then summed correct count; ties retain the earliest declared candidate. This rewards coverage rather than hiding an ingestion failure in a pooled mean. Validation is untouched until every candidate/checkpoint selection is frozen. Head-fit objectives, regularization and optimization exposure differ; this is a comparison of readout recipes, not an isolated loss-function intervention.

All 33 numerical labels must be represented in the fitting pool. This is fresh-context generalization within the same bounded half-unit vocabulary, not novel numerical-class extrapolation. The unchanged benchmark's type/value frequencies are retained. Report every value/type/primitive, sign/magnitude, argument/provenance target identity and delay group; a balanced grid may be a separately named later diagnostic.

## Budget, profile and advancement

First profile the exact capture/readout path using 64 mechanical-profile events per partition, one ridge solve and ten CE updates. It is not a primary experiment and missing classes are permitted only there. Standard latent width remains 1024. Estimate initial development at 1–3 GPU-minutes; request a maximum5-minute release only after the profile. The coordinator owns all queue/budget decisions.

Record training-set fit separately from fresh calibration/validation. Prefer a recipe that improves the weak boundary without losing covered-delay accuracy. A failed first recipe selects the smallest discriminating follow-up; it does not end the authorized campaign. Likely branches are more data/exposure, phase-conditioned consumers if zero versus recurrent states conflict, or an independently justified nonlinear decoder. Do not combine these changes before inspecting R01.

No confirmation or composition is claimed from one checkpoint. Any main competence claim requires three paired confirmation seeds, fresh fitting/calibration and at least1,024 untouched validation/test events, with the selected recipe frozen first. The new narrow consumed-scalar contract requires >=98% in every declared consumed condition and seed. Historical six-field gates remain unchanged; copying cannot pass reconstruction.

## Provenance and outputs

Verify frozen checkpoint SHA against the archived Stage 11 manifest. Save source/config/checkpoint hashes, compressed lossless feature caches and hashes, fitted readout weights, every calibration candidate score, fieldwise predictions/targets/event identities, class counts, training curve, parameter counts, row/event/optimizer exposure and wall-clock/GPU/RSS measurements. Other field predictions must be identical between arms. Keep all Stage 1–11 files unchanged.
