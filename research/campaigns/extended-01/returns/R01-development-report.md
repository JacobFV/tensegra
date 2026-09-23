# R 01 development: a reliable scalar consumer is now plausible

Frozen Stage 11 wide backbone 10 plus a supervised CE readout reconstructs every training scalar at every covered delay. On fresh validation, its worst delay/distractor cell is 1016/1024 (99.22%). This is one development checkpoint, not confirmation or composition competence.

## Matched inputs and learned part

All arms use exactly the same normalized 1024-dimensional scalar workspace row, unchanged backbone, and unchanged non-value heads. The CE classifier has 33,825 parameters, like categorical ridge. It is warm-started from the original classifier through an algebraically equivalent train-standardized representation. Six-field predictions remain available; only scalar predictions differ.

The fitting set has 4,096 unique events and 24,576 delay/event rows over 0/1/2/4/8/16. Every one of 33 half-unit labels is present, with 38–585 unique events per label. Calibration and validation each have 1,024 disjoint events, paired across delays/distractors. This is a bounded-vocabulary fresh-context experiment, not numerical-range extrapolation.

Four ridge regularizations and eleven CE checkpoints were inspected on calibration only. Worst-cell exact accuracy, then summed correctness, selected ridge .01 and CE step 900. CE training used 1,000×256 optimizer presentations; ridge uses a closed-form solve. These are recipe comparisons with different objectives/exposure, not a pure loss-function intervention. No delay 32 features were captured.

Eight-distractor fresh validation counts, each out of 1,024:

| Consumer | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| Unchanged | 918 | 951 | 935 | 911 | 894 | 872 |
| Categorical ridge | 828 | 1021 | 1024 | 1024 | 1022 | 1018 |
| CE refit | 1024 | 1024 | 1024 | 1024 | 1023 | 1016 |

CE reaches 4096/4096 training accuracy at all six delays. Ridge fits recurrent training states almost perfectly but reaches only 3355/4096 at zero; its zero-step failure therefore includes acquisition within this readout family. CE's calibration counts at delays 0/1/2/4/8/16 with eight distractors are 1024/1024/1024/1024/1024/1017. Its worst validation cell across both distractor counts remains 1016/1024 at delay 16/eight distractors.

Of the eight errors in that cell, seven differ by one half-unit and one by a full unit. Type-conditioned exact counts are integer 404/405, float 403/410, boolean 209/209. Primitive-conditioned counts are add 204/205, subtract 199/202, multiply 201/203, negate 203/205, compare 209/209. Per-value, signed-error, magnitude, ordered identity and joint value×type×primitive support is retained in every raw row. Small subgroup support must not be interpreted as a precise worst-value generalization guarantee.

## Decision and accounting

The smallest discriminating next step is confirmation of the selected CE 900 recipe on all three frozen wide backbones with new data, not another memory/encoding intervention. Keep ridge .01 and unchanged heads as paired references. Freeze 900 updates and .01 before generating confirmation data; do not select new checkpoints on confirmation validation. Delay 32 will first be evaluated only in the final confirmation test and cannot select the recipe.

Profile process occupancy was 3.96 s; development occupancy 26.19 s (runner 24.87 s), including export. Development peak allocated CUDA memory was 563.85 MB; process RSS 2.872 GB. Persistent memory/backbone dimensions remain unchanged. Raw predictions and manifest are in research/results/campaign-01/returns/r 01-development. Lossless feature/logit caches and selected coefficients remain immutable on GB 10 under ~/topoformer-campaign-01/returns/r 01-development.

Independent raw-count/selection audit passed all 90 cells for both profile and development, reconstructed the chosen heads, and confirmed unchanged non-value predictions. Coefficient replay/provenance audit is pending. No confidence-calibration claim is made from ridge class scores. No returned-fact use or composition experiment has run yet.
