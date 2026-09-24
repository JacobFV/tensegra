# C04 aggregate analysis prepared before main outcomes

`research/tools/campaign_composition_confirmation_analysis.py` reads saved CPU tensors and JSON only. It imports no model/training/runtime code and runs no inference. Missing lineages remain explicitly pending, with missing filenames and any available failed process receipt. Aggregate gates and uncertainty remain pending until all three lineages have outputs. Completed lineages must match model seeds1601/1602/1603 and distinct test event seeds560000003/561000003/562000003.

The report retains every workspace cell (both views, distractors2/8, all six delays), separate ordered-lowering-and-answer joint counts, answer-only counts, persistent refusals and correct-lowering conditional counts. All three neural arms remain visible with fixed4000 primary and CLEAN-selected secondary results. Public controls and primitive/type/value/query-gap strata are preserved from the original summaries. Paired target ordering and N1 initialization/sample-stream invariants are checked; input artifacts are SHA256 recorded.

Causal counts are independently recalculated from saved clean/drop/wrong/swap outcomes for both views, workspace delays0/1/16 and supplied-copy. They expose all attempted events, actual supplied-present/absent counts, prediction refusals, original-label correctness, changed-support correctness, unchanged-support errors, drop effect and normalized query-to-copy gain. Nonpositive normalization denominators remain undefined/fail. Supports below256 or changed agreement below90% fail; no refusal is removed from the attempted denominator or repaired with a label.

There are198 paired answer-only comparisons: every workspace cell versus supplied-copy and each neural endpoint; supplied-copy versus each neural endpoint; and all three primary neural pairs. Each reports left-only/right-only/both-correct/both-wrong counts per seed. The2000-replicate percentile bootstrap independently resamples event indices within each fixed lineage, using deterministic seed6042026+lineage and identical draws across every arm/view/delay of that lineage. Lineages have distinct event populations, so draws are independent across lineages. Their mean has equal lineage weights. Repeated cells never become extra independent examples. Seed means, min/max and sample standard deviation are shown separately; the interval is conditional event uncertainty, not a population confidence interval over three training seeds. Shared finite arithmetic signatures are disclosed rather than claimed independent numerical computations.

The overlap report reads each arm's private visited-view audit: typed ordered operands/operation, with and without exact threshold/polarity. It reports pool counts, unique signatures, multiplicity-weighted event overlap fractions, and each ordered pair of lineages' test-to-actually-visited-training overlap. Keys/destinations/names remain excluded. Inherited acquisition overlap remains explicitly unmeasured; no zero-overlap or numerical extrapolation claim is inferred.

CPU checks: four contract tests passed in0.08s with CUDA hidden, covering refusal accounting, undefined normalization, paired bootstrap/seed dispersion, and all-missing pending behavior. Raw profile reader smoke processed28 hybrid/copy cells,8 causal cells,198 pairings and72 cross-lineage overlap cells in0.816428s by reusing mechanical profile artifacts three times solely for software coverage; this is not confirmation evidence. A final guard rejects such repeated mechanical namespaces in actual confirmation aggregation. No main outcome was read.

After all lineages finish, run with existing CPU Python and CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES="" /home/brandonin/topoformer-pilot/.venv/bin/python \
  research/tools/campaign_composition_confirmation_analysis.py \
  --directory /home/brandonin/topoformer-campaign-01/composition \
  --output /home/brandonin/topoformer-campaign-01/composition/c04-aggregate.json
```

The output path must be new. Independent reviewer raw audit remains separate. This analysis-only addition changes no frozen acquisition/config/source recipe or GPU budget.
