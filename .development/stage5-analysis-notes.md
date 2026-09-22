# Stage 5 artifact analysis

`runtime_analysis.py` is a standalone, Torch-free CLI. It streams plain/gzip JSONL one run at a time and retains compact statistics; it never copies full traces or per-example arrays into its output. Run large artifacts on the remote machine anyway, since a single decoded row includes its training history.

```
python src/topoformer/runtime_analysis.py metrics.jsonl.gz analysis --config config.json
```

The analyzer checks declared variant/seed coverage, complete depth × distractor grid and extras, duplicate conditions, paired data hashes, common source/config identity and the supplied configuration hash. Initialization and training-schedule equality are disclosed for paired contrasts rather than assumed. Raw counts are summed with their actual denominators; across-seed means and sample standard deviations remain separate from pooled rates. Undefined conditional rates stay `null` and cannot be silently converted into perfect accuracy. Zero coverage has undefined risk and zero unconditional correct execution.

Plots show end-task, exact execution, binding and complete trajectory separately, with the same [0,1] scale. Neural controls' counterfactual exact-lowering audits do not become actual symbolic execution. Depth D excludes initial resolve (D+1 scheduled operations). The size axis counts distractor objects, not all runtime nodes. Risk/coverage plots separate noise, ambiguity and invalid schemas. Learning curves include step zero and the declared warm boundary; they are diagnostics, not checkpoint-selection rules.

Representative raw failure cases are not used to estimate failure frequencies. Main observations and interpretation belong in the final report only after frozen-source experiments complete. Tests run with `python3 -m unittest discover -s tests -p test_runtime_analysis.py` without Torch imports.

Compatibility audit: remote runner smoke (all ten variants) and oracle lifting pilot both analyze successfully with their resolved configs. These initial artifacts predate the final runner's runtime-node and failure-count additions. A regression test protects neural `None` execution metrics, distinct numeric-result metrics, metric-semantics provenance and compact runtime-node ranges/seed means. Raw generic failure/invocation/recovery counts are retained but their interpretation must follow each variant's actual-versus-counterfactual semantics.
