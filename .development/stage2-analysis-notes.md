# Stage 2 artifact analysis notes

`src/topoformer/study_analysis.py` is an artifact-only analysis CLI. Its core
path uses the Python standard library and never imports Torch. Optional SVG
figures are produced with a lazy Matplotlib import:

```bash
python src/topoformer/study_analysis.py RESULTS/metrics.jsonl REPORT_DIR --plots
```

The output directory must not already exist. The command writes `summary.json`
and `report.md`; `--plots` additionally writes standalone SVG figures. The JSON
summary is the lossless analysis product. Markdown tables are intended for
review and the later narrative report, not as a replacement for raw JSONL.

## Locked analysis rules

- Only rows marked `status: complete` contribute. Non-finite evaluations,
  incomplete rows, and duplicate run identities are counted in `exclusions`.
- Seed comparisons require both modes in the same suite, domain, training
  configuration, trajectory count, corruption, evaluation size, split, and
  checkpoint. Metrics from multiple evaluation graphs are averaged within a
  seed before population SD or a paired effect is calculated.
- Paired effects are baseline error minus treatment error. Because oracle
  metrics are matched within each exact case, this is also the reduction in
  oracle-excess error; no ratio is formed for censored observations.
- Soft strengths are selected separately per suite/domain/training
  configuration using the final validation checkpoint averaged over the common
  validation seed set. Test rows are never read by strength selection.
- `soft1` is contrasted with `permuted1`, and `soft4` with `permuted4`. This
  keeps supplied-bias magnitude matched.
- Efficiency thresholds are fixed at oracle plus 10%, 25%, and 50% of the
  checkpoint-zero oracle-to-zero gap. `s_epsilon` is the first observed
  checkpoint at a given training count. `n_epsilon` is the smallest tested
  training count with any observed crossing. Unreached results remain null and
  carry `censored: true`.
- Validation AUC is trapezoidal area divided by optimizer-step budget. It is
  null unless the curve includes both step 0 and the complete requested budget.
  Fixed-size efficiency normalization is recorded from the smallest nested
  training subset and is never refit for larger counts.
- `efficiency_identity` follows the same efficiency rules but remains a
  separate suite; it is never pooled with identity-free efficiency runs.
- Corruption precision and recall are calculated from recorded realized graph
  quality. Transfer evaluation graphs are macro-averaged within seed before
  cross-seed population SD.
- Learned and typed coefficients preserve every signed value at every
  checkpoint, indexed by layer, head, and relation channel when present. The
  analysis does not assume positivity or a trend with depth.

The optional plots show population SD error bars and faint per-seed efficiency
curves. Parameter count, optimizer examples, training seconds, and process peak
RSS are retained as resource accounting. No statistics are synthesized for
missing or partial cases.
