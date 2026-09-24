# S16 frozen normalization diagnostic

All six fixed models completed all1,024 instances without timeout, omission, checkpoint/threshold mutation or recipe change. Full outer wall114.73s; conservative charge114.74s. GPU was free before analysis. Raw artifacts, hashes, receipts and paired summary are archived under `research/results/campaign-01/semantics/s16-main/`. The independently audited normalizer is a supplied English-grammar/case/known-vocabulary engineering prior; it does not establish learned renaming invariance or general language understanding.

The normalizer makes original and consistently renamed inputs identical, but it does not improve graph exactness on the original population. Calibrated decay exactness decreases in all three seeds; constants remain zero. Copy accuracy stays high. This separates a mechanically enforceable public-identifier invariance contract from graph prediction correctness.

| Seed | Arm | Policy | Original exact /1024 | Normalized exact /1024 | Repairs | Regressions | Original copy | Normalized copy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 701 | constant | raw | 0 | 0 | 0 | 0 | 0.976039 | 0.980060 |
| 701 | constant | calibrated | 0 | 0 | 0 | 0 | 0.976039 | 0.980060 |
| 701 | decay | raw | 2 | 1 | 1 | 2 | 0.989763 | 0.990254 |
| 701 | decay | calibrated | 66 | 48 | 27 | 45 | 0.989763 | 0.990254 |
| 702 | constant | raw | 0 | 0 | 0 | 0 | 0.987228 | 0.990916 |
| 702 | constant | calibrated | 0 | 0 | 0 | 0 | 0.987228 | 0.990916 |
| 702 | decay | raw | 6 | 3 | 3 | 6 | 0.995734 | 0.995012 |
| 702 | decay | calibrated | 131 | 103 | 60 | 88 | 0.995734 | 0.995012 |
| 703 | constant | raw | 0 | 0 | 0 | 0 | 0.981701 | 0.980398 |
| 703 | constant | calibrated | 0 | 0 | 0 | 0 | 0.981701 | 0.980398 |
| 703 | decay | raw | 11 | 12 | 11 | 10 | 0.994505 | 0.994422 |
| 703 | decay | calibrated | 139 | 120 | 64 | 83 | 0.994505 | 0.994422 |

Each repair/regression is paired on the same original semantic instance, with frozen original predictions under the corresponding raw or calibrated policy. There are1,024 distinct instances, reused across models;6,144 model-instance cells are not independent population samples. Constants remain0/1,024 for both policies. Across the three decay checkpoints, calibrated exact model-instance counts change336→271 (151 repairs,216 regressions); raw19→16 (15 repairs,18 regressions). These aggregate counts are descriptive only, and seed variation is retained above.

| Seed | Arm | Policy | Typed-edge micro-F1 original→normalized | Ordered-edge micro-F1 original→normalized |
|---|---|---|---|---|
| 701 | constant | raw | 0.715686 → 0.745481 | 0.766808 → 0.770014 |
| 701 | constant | calibrated | 0.894002 → 0.881678 | 0.915879 → 0.915059 |
| 701 | decay | raw | 0.874751 → 0.871887 | 0.939788 → 0.944931 |
| 701 | decay | calibrated | 0.960533 → 0.958224 | 0.979280 → 0.977916 |
| 702 | constant | raw | 0.688433 → 0.639830 | 0.777617 → 0.708394 |
| 702 | constant | calibrated | 0.906745 → 0.904761 | 0.928531 → 0.926160 |
| 702 | decay | raw | 0.901177 → 0.896107 | 0.965407 → 0.969548 |
| 702 | decay | calibrated | 0.970784 → 0.968802 | 0.985538 → 0.982291 |
| 703 | constant | raw | 0.723197 → 0.746956 | 0.766230 → 0.819064 |
| 703 | constant | calibrated | 0.907527 → 0.905151 | 0.927956 → 0.909791 |
| 703 | decay | raw | 0.916930 → 0.911130 | 0.957315 → 0.949360 |
| 703 | decay | calibrated | 0.970428 → 0.966874 | 0.983579 → 0.981197 |

Node presence/count/F1, node-type accuracy, identity-equivalence accuracy, copy accuracy, typed and ordered micro-counts for every endpoint/policy are retained in `paired-summary.json`; metric-sufficient predictions and original per-instance component metrics are retained in all six immutable prediction artifacts. Calibrated typed-edge and ordered-edge F1 decline in every endpoint.

No normalized-renamed forwards were run. Their prediction implications follow from the audited full FP32/BF16 actor-input equality, preserved token positions/equality classes, and restoration through original public-token indices. They must be labeled a programmed invariance consequence under this restricted renderer and public case convention. The study does not show that the frozen model itself learned arbitrary identifier handling, nor that canonicalization repairs its graph semantics. Selection among endpoints, threshold retuning and further training are outside this diagnostic.

Timing: all-six preflight14.342909s; forward/decode38.614227s; scoring/reconstruction40.571447s; export1.903061s. The480s cap was unchanged; full duration114.73s lies below cap and prospective conservative planning estimate.
