# Stage 4 artifact analysis

The analyzer reads raw JSONL and the resolved runner config; it performs no model execution, fitting, checkpoint selection, or adaptive experiment launch. It preserves final-checkpoint task performance as primary and reports canonical complete grounding paths separately.

The canonical trajectory is initial query grounding followed by each post-update grounding, D+1 states. It avoids multiplying duplicate pre/post measurements of the same recurrent state. Per-example bitstrings support pooled first-error survival and integer transition counts. Error persistence and recovery share the denominator of transitions starting wrong, including initial errors. The product of per-state marginal accuracies is a descriptive reference only: trajectory difficulty, selection, revisits and reconvergence confound an attractor interpretation.

Complete grid/seed, source/config, fixed budget, finite values, paired data/schedule and initial-versus-final pairing are checked. Canonical completion must match recomputation from raw bits. Paired contrasts disclose differing initialization hashes rather than claiming all interventions are identical-parameter comparisons. Pointer writes are explicitly a stronger algorithmic prior.

Each variant has fixed-range 0–1 PNG/SVG task/complete-path heatmaps. The report includes every depth/size cell's mean, sample SD and individual seed values, plus initialized/final anchors, paired task differences, persistence counts and within-cell feature/failure associations. Pearson coefficients are computed from examples within variant × N × D, pooled across seeds; constant features or outcomes are unavailable, not zero. Feature means include measurements after errors and cannot establish causal predictors. JSON retains aggregates, not duplicated trajectory arrays.

The stability gate is task AND canonical complete-path >=.95 at N128/D64 in each declared seed. Missing gate cells are unavailable. Heatmaps and gate checks do not select checkpoints or authorize a new experiment.

Validation: eight focused analyzer tests passed on the remote CPU environment, covering denominators, canonical bitstring aggregation, missing cells/seeds, data/source/budget mismatch, all-seed gate, constant correlations and artifact hashes (including lossless gzip JSONL input). Stored compressed bytes are hashed; raw trajectory arrays remain only in the original metrics artifact. Analyzer code uses only stdlib except optional Matplotlib plotting; the existing package __init__ still imports Torch.

Post-run CLI correction: direct `python3 -S src/topoformer/binding_analysis.py --help` now uses a sibling helper import and works with all site packages disabled. Package/module use retains relative imports. A ninth test exercises this standalone invocation from an unrelated directory. Main/adaptive frozen analyzer provenance remains unchanged; their module invocations were valid.
