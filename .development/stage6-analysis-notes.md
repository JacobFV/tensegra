# Stage 6 artifact analysis

Run the analyzer directly to avoid package `__init__` importing Torch:

```sh
python src/topoformer/thinking_analysis.py RESULTS OUTPUT --source-root .
python src/topoformer/thinking_analysis.py LANGUAGE_RESULTS LANGUAGE_OUTPUT --track language --source-root .
```

`--no-plots` needs only the standard library. Plotting lazily imports matplotlib, uses Agg, and saves standalone PNG heatmaps, per-loss/annealing curves, and readiness calibration/risk coverage. No training or checkpoint loading occurs. The input is streamed one JSONL row at a time; only compact cells persist. Summary JSON does not duplicate raw examples, traces, calibration rows, or language failure examples.

Controlled execution and TCN semantic decoding remain separate tracks. Controlled aggregates include sample mean/SD across seeds, explicit count/denominator populations, paired-seed contrasts against local control, separately paired frozen event interventions, runtime status and refusal-reason populations, task/exact semantic/trajectory/transition-set accuracy, conditional task given exact result, microstep histograms, forced exits, incomplete learned halts, projection overlap, grounding/null diagnostics, and encoder versus post-recurrent workspace event round trips. Oracle-trace aggregates live separately under `privileged_oracles`. Readiness calibration preserves fractional posterior targets: risk is posterior unready mass among proposals above threshold, not task-error risk.

Actual language renderer/split names are retained (including controlled entity renaming when present). Decoder node/typed-edge/type/lexical/exact-canonical metrics stay distinct from execution metrics. Label-derived majority-position baselines are separate privileged baselines. TCN node/edge precision/recall are producer macro averages; raw decoder denominators were not originally emitted, so the analyzer does not invent them. No unsupported heldout motif or full language induction claim follows from these artifacts. Runtime refusal combines unavailable and nonnumeric operands, so a pure type-validity or independent lowering score cannot be reconstructed.

Duplicate evaluation/training rows, mixed source hashes, unequal controlled initialization hashes, unequal same-seed training data, and unequal same-condition evaluation data raise errors. Manifest/source/data digests and semantic split disjointness are audited where available. Source checks compare against `--source-root` and report mismatches instead of hiding them. Configuration, input artifact, analysis source and checkpoint files receive SHA256 identities; checkpoint hashes cannot prove agreement with an independently expected digest that the producer did not emit. Core declared seed/variant/step/depth or renderer coverage reports missing cells; optional OOD/intervention coverage is visible in aggregates. One seed has null SD, zero conditional population has null rate, and missing diagnostics remain unsupported.

Validation: six standard-library fixture tests pass via `python3 tests/test_thinking_analysis.py`, covering paired mean/SD and conditional nulls, identity failures, runtime populations, event-stage separation, microstep distribution, duplicate training rows, partial grid detection, file integrity, and privileged language baseline separation. Local `python3 -m pytest` is unavailable (pytest is not installed); matplotlib is also not installed locally. Root runs Torch/full-suite and figure smoke validation remotely. No experiment has been launched by this worker.

## Main-study schema additions

The analyzer now retains explicit null decoder metrics, raw node/typed-edge populations, defined-example counts, and separately named pooled graph rates. It validates TCN per-seed initial states and data/config identity across evaluation/loss rows, rejects duplicate loss rows and impossible graph counts, includes renamed examples in modern data-hash verification, and checks checkpoint bytes against producer manifest hashes. Old pilots remain readable with explicit legacy denominator/provenance labels. Controlled aggregates retain observation protocols, paired contrasts flag protocol differences, and training curves retain teacher-forcing rates and objective names.

Validation expanded to nine passing standard-library tests including modern decoder null/count handling, initial-state mismatch rejection, renamed-data hash checks, checkpoint corruption detection and protocol mismatch labeling. Remote plotting was validated on the completed language pilot without modifying its frozen checkout. Main-study changes do not retroactively rewrite the archived pilot summary.

## Disjoint seed shard merge

```sh
python src/topoformer/thinking_analysis.py configs/stage6-controlled.json COMBINED_DIR \
  --merge-shards SEED0_DIR SEED1_DIR SEED2_DIR
python src/topoformer/thinking_analysis.py COMBINED_DIR ANALYSIS_DIR --source-root FROZEN_REPO
```

The canonical input must declare the full seed list. Resolved shard configs must differ **only** in seeds; supplied canonical fields must match them, and omitted defaults are inherited from the shared resolved config. Source manifests and hashes must match, evaluation row source hashes must agree, variant/seed populations must be complete/nonoverlapping, training updates must cover every configured step, and the core depth/checkpoint grid must be present. Duplicate rows, out-of-shard rows/checkpoints, and missing final checkpoints (when checkpoint saving is enabled) are errors. Evaluation-economy grid filtering must match the runner's final schema before merging main artifacts.

Merging streams one decoded JSONL line at a time and retains only identity/count sets. It archives each shard's exact original manifest, resolved config, and compressed original log under `shards/NNN/`. `combined_from` records original manifest/source/config/log hashes, rows, seeds, variants and checkpoint hashes. Checkpoint bytes are copied and rehashed; no model state is deserialized. The merged manifest preserves shared fields and substitutes the canonical seed union. Combined metrics are gzip JSONL, automatically recognized by normal analysis. Outputs are staged and renamed only on success; an existing output directory is rejected. This records partitions of one paired study, not additional independent studies.

Validation: 11 standard-library tests pass, including disjoint merge and rejection of overlap, missing seeds/runs, config/source mismatch and missing final checkpoints. A real CLI merge followed by gzip-input analysis verified complete coverage and merged-log hash agreement. No experimental training runs were launched.
