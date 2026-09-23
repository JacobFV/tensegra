# Stage 6 TCN language main study

Producer source: `4c36557224e638cb6c03811be5ffbe7ea2583b46`. Analyzer source: `61474e3` copied to `/tmp` on `gb10-direct`; the frozen producer checkout was unchanged. All 12 runs completed: seeds 7/19/43 × four arms, 256 updates, batch four, 240 training and 48 heldout constructions, width 32, canonical node capacity 128, three recurrent microsteps (one in single_pass). Evaluations at 0/64/128/256 cover English, Spanish, symbols and controlled entity alpha-renaming: 48 JSONL rows and 192 renderer/seed/step cells. Full configuration and loss curves are archived.

Task choice accuracy at update 256 (mean ± sample SD across three paired seeds):

| Arm | English / Spanish / symbols | Entity renaming | Paired ordinary-renderer difference from single_pass |
|---|---:|---:|---:|
| single_pass | 22.92% ± 0.00 pp | 20.14% ± 5.24 pp | reference |
| recurrent | 14.58% ± 0.00 pp | 13.89% ± 2.41 pp | −8.33 ± 0.00 pp |
| semantic_supervision | 14.58% ± 0.00 pp | 24.31% ± 6.01 pp | −8.33 ± 0.00 pp |
| multisurface_consistency | 17.36% ± 4.81 pp | 21.53% ± 9.39 pp | −5.56 ± 4.81 pp |

The random-choice baseline is 19.44%. Paired entity-renaming differences from single_pass are −6.25 ± 3.61 pp recurrent, +4.17 ± 3.61 pp semantic-supervision, and +1.39 ± 8.42 pp multisurface. These SDs describe three seed differences, not confidence intervals. English step-zero task accuracy is 20.14% ± 3.18 pp for single_pass and 19.44% ± 6.70 pp for the three recurrent arms. The matching task scores across ordinary renderers do not establish renderer-invariant semantics.

Exact canonical graph accuracy and exact 64-bit lexical-hash fidelity are zero for every final arm/split. Semantic supervision improves partial decoder metrics: on English its macro node precision/recall are 91.77%/91.11%, node-type accuracy 54.45%, and typed-edge precision/recall 0.630%/62.30%. Multisurface gives 95.90%/79.69% node precision/recall, 51.85% type accuracy and 0.730%/49.08% typed-edge precision/recall. Very low edge precision leaves extensive false-positive structure despite high node scores. These findings do not establish full graph induction, exact semantics, or a downstream task advantage from recurrence/semantic supervision.

Raw graph true positives, predicted/gold populations and defined-example populations are retained in summary counts. Macro precision for unsupervised heads is undefined for some seeds; `n` can be one or two rather than three, and null is preserved. Separately named `pooled_graph_rates` use pooled raw denominators and should not be confused with macro means. Privileged heldout-label majority-position baselines are separated from model metrics.

All producer source hashes, combined original+renamed data hashes, config hashes, row data/config identities, and 12 producer-declared checkpoint SHA256 values verified. Initial state hashes match across arms for each seed and agree between all evaluation/loss rows and the run manifest. Semantic train/eval IDs are disjoint. The source/data audit includes compiler/renderer provenance; complete primary coverage and all four renderer labels were confirmed. `archive-manifest.json` records local archive file hashes and exact source identity. Checkpoint binaries remain remote; their verified hashes and byte counts are archived.

Interpretation limits: canonical node alignment, role vocabulary, lexical hashes and typed semantic schema are supplied. Spanish receives paired task supervision and extra surface exposure in the multisurface arm, so its comparison is not an isolated consistency-loss causal estimate. Symbols are withheld. Entity renaming is a controlled alpha-renaming test, not heldout motifs/compositions or unrestricted vocabulary generalization. The separate controlled execution study must be reported separately.

Recorded final elapsed timestamps sum to 798.33 seconds across runs, excluding the final evaluations/checkpoint writes; they include earlier evaluations. `runtime-cost.json` preserves per-seed timing. Analyzer execution, source/checkpoint verification and standalone figure rendering succeeded remotely. The task/graph figures were visually inspected and compressed raw-curve identity was checked locally. No training was launched during analysis.
