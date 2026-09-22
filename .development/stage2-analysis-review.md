# Stage 2 analysis review

## Verdict

**Spec: fail. Quality: changes required before any Stage 2 result is interpreted or published.**

The artifact-only boundary, complete-run filter, validation-only soft-strength selection, population-SD convention, signed-coefficient preservation, and basic censor representation are good foundations. However, the current analyzer silently reduces several scientifically distinct runner metrics to one scalar, labels that scalar incorrectly, does not enforce graph-level matching for paired effects, and accepts a truncated curve as a complete-budget AUC. These are result-changing defects. The full experiment may continue producing raw runner artifacts, but `study_analysis.py` must not be used to report them in its present form.

Review basis: `.development/stage2-design.md`, the uncommitted `.development/stage2-methods.md`, `.development/stage2-analysis-notes.md`, `.superpowers/sdd/plan-02/analysis-review.diff`, and the emitted schema in `src/topoformer/study.py`. The worker-reported 133-test run and 104-row CLI smoke were accepted as prior evidence; unchanged tests were not rerun. Full experimental results were still running and were not reviewed.

## Critical findings

### C1 — Test metrics are selected incorrectly and then mislabeled

`_metric()` at `study_analysis.py:176-180` searches for `normalized_mse`, then raw `deterministic_rollout_mse`, then raw `stochastic_rollout_mse`, then `mse`. The runner emits test fields named `one_step_mse`, `one_step_normalized_mse`, `stochastic_rollout_mse`, `stochastic_rollout_normalized_mse`, `deterministic_rollout_mse`, and `deterministic_rollout_normalized_mse` (plus matched oracle and zero fields) at `study.py:256-268`.

Consequently, every normal test evaluation is flattened to **raw deterministic rollout MSE**. The aggregate table has no metric dimension, while plots label it “normalized error” (`study_analysis.py:527`) and the report gives no outcome/horizon/unit label. One-step and stochastic rollout outcomes, all normalized test variants, and all oracle/zero references disappear. This invalidates suite aggregates, paired contrasts, corruption-reference contrasts, terminal tables, and plots.

Required repair: expand each evaluation into explicitly named metric rows rather than choose the first available field. At minimum preserve and aggregate one-step, deterministic rollout, and stochastic rollout errors in raw and normalized units, with their matched oracle and zero references. Add `metric` (and preferably `unit`/`normalization`) to every grouping, pairing stratum, summary record, table, and plot. Labels must state the exact outcome, normalization, split, and rollout horizon where relevant. Add schema-level integration tests using a runner-shaped evaluation that contains all fields and asserts that none are silently dropped or relabeled.

### C2 — Paired effects do not enforce exact graph/reference matching

`paired_effects()` groups observations only by seed and mode (`study_analysis.py:73-89`). `_paired_contrasts()` stratifies on suite/domain/config/count/corruption/fraction/eval-size/split/step but omits metric identity and graph identity (`study_analysis.py:277-287`). Repeated evaluation graphs are independently averaged within each mode. If a mode is missing a graph, contains a duplicate graph, or has a different graph/trajectory set, it still produces a plausible paired effect from nonmatching sets. The current unit test expressly blesses averaging repeated rows but never verifies identical graph keys.

The runner provides `graph_hash`, `graph_seed`, and `trajectory_seed`. Required repair: within every seed and exact analysis case, key observations by an explicit reference identity (at least graph hash + trajectory seed, and supplied/evaluation condition where applicable), reject duplicate identities, require identical reference-key sets for the two modes, compute graph-level matched differences, then macro-average those differences within seed. Report unmatched/duplicate reference counts and keys. Transfer must average matched graph effects within seed before population SD. Add tests for reordered graphs, one missing graph, duplicate graph records, mismatched trajectory seeds, and the valid matched multi-graph case.

### C3 — AUC certifies the observed maximum as though it were the configured budget

`_efficiency_summary()` defines `budget = max(point.step)` and immediately asks `complete_normalized_auc()` whether the curve ends at that value (`study_analysis.py:341-342`). A curve ending at step 300 therefore passes as a “complete” 300-step AUC even though the study’s requested budget is 600. This contradicts the locked rule that AUC is null unless both step 0 and the complete requested budget are present.

Required repair: take the requested optimizer budget from an explicit analysis input/config artifact or serialize it into each runner row, and pass that value to `complete_normalized_auc`. Do not infer it from the curve. Validate the exact expected checkpoint set or, at minimum, endpoints and strictly unique steps. Record the budget alongside every AUC. Add an integration test where a row marked complete has a 0..300 curve under a 600-step configuration and must yield null plus an exclusion/validation error.

## Important findings

### I1 — Duplicate and nonfinite exclusion is incomplete

Run deduplication keeps the first identity and discards later copies (`study_analysis.py:253-267`). A partial first row can suppress a later complete row, and two conflicting complete rows are treated as a harmless duplicate. The identity also does not include every training-condition discriminator (for example an explicit node-identity value beyond truthiness/config evolution or future schema fields), nor is content equality checked. Within-run duplicate evaluation identities are not detected at all.

Nonfinite validation only examines whichever scalar `_metric()` happens to choose. Other outcome/reference fields can be nonfinite without exclusion. `_learned_summary()` silently drops nonfinite coefficients rather than excluding or flagging the checkpoint. JSON from the runner forbids NaN, but the analyzer claims general artifact validation and should enforce the schema it reports.

Required repair: validate a canonical run identity and evaluation identity; reject/exclude all conflicting duplicates rather than order-dependently choosing one. Either collapse byte/content-identical duplicates deterministically or exclude the whole ambiguous identity and report it. Validate every analyzed metric and reference field, coefficient, step, and resource field used. Keep reason-specific counts (`duplicate_runs_identical`, `duplicate_runs_conflicting`, `duplicate_evaluations`, `nonfinite_<metric>`).

### I2 — Soft-lambda selection is validation-only, but selected and unselected results are not distinguished in downstream reporting

The selection input is correctly restricted to final validation-curve values from complete soft runs (`study_analysis.py:220-232`), so test and runtime-corruption evaluations do not choose lambda. The grouping is suite/domain/training configuration and uses common validation seed sets, which is appropriate. Runtime-corruption runs inherit the clean-trained validation choice, also appropriate.

However, the selected mapping is merely printed; terminal tables, contrasts, and plots include both soft strengths without marking which was selected. A reader cannot tell the preselected primary comparison from fixed-strength sensitivity analyses. Selection also accepts `max(step)` rather than explicitly requiring the configured final checkpoint.

Required repair: preserve all candidates, mark every soft aggregate/contrast as `selected_by_validation`, and produce a clearly separated primary selected-lambda view plus fixed-strength sensitivity rows. For runtime corruption, join to the clean training-condition selection only. Require the configured final validation checkpoint and matched candidate seed set; if absent, report selection as unavailable rather than falling back to an earlier maximum. Never select independently by evaluation corruption or test metric.

### I3 — Efficiency reference and censoring validation is too trusting

`efficiency_thresholds()` correctly implements `oracle + f*(zero-oracle)`, invalidates a zero gap, takes the first observed checkpoint, chooses the smallest tested count, and leaves failures censored. `_efficiency_summary()`, however, independently mixes stored runner thresholds/steps with a second derivation from the first point of the smallest-count curve (`study_analysis.py:343-356`). It does not verify that stored and derived values agree, that every mode/count uses the same matched validation reference, that count values are unique and on the configured grid, or that normalization provenance is consistent. Missing references default to zero, which can turn malformed input into an apparently valid summary or exception. Censored `n_epsilon` is represented in the derived record, but per-count rows omit `n_epsilon` by design and the Markdown table does not print an explicit censor flag.

Required repair: use one authoritative derivation from validated normalized validation curves and explicit matched oracle/zero references; compare against serialized runner thresholds as an integrity check. Require finite references, a strictly positive gap, the expected count grid, unique strictly increasing checkpoints, the configured budget, and identical normalization provenance across nested counts/modes. Make censoring visible in JSON and Markdown for each threshold and do not average reached-only `n_epsilon` values in plots as if censoring were absent. If speedup ratios are added, compute them only for exact matched seeds where both modes attained the threshold; otherwise keep them null/censored.

### I4 — Plot grouping pools domains and obscures conditions

Efficiency plotting groups by `suite:mode` (`study_analysis.py:484-509`), pooling all domains before its population mean/SD and threshold-count line. That violates domain-specific comparisons and makes the faint per-seed lines connect unrelated domain observations. Terminal plot labels contain only mode, evaluation corruption, and eval size (`study_analysis.py:521`): corruption plots omit domain and train corruption/fraction; transfer plots omit training-size condition and domain; heterogeneous/learned plots omit mechanism/information condition. Repeated terminal rows become visually indistinguishable. All terminal plots also inherit C1’s false “normalized error” label.

Required repair: facet or emit separate figures by suite, domain, metric, and meaningful training condition. Encode both train and evaluation corruption for corruption/runtime robustness, fixed-vs-mixed training sizes for transfer, and mechanism/information for heterogeneous results. Include normalization and outcome in y-axis/title, indicate selected lambda, and show censoring rather than dropping failures from threshold plots.

### I5 — Corruption-reference contrasts are under-specified and can overwrite references

`_corruption_reference_contrasts()` builds a dict whose value is a single scalar per domain/seed/graph/corruption/fraction (`study_analysis.py:293-311`). Before C1 is fixed it has no metric dimension; afterward it would still need metric, trajectory/reference identity, split, and evaluation condition. Duplicate keys overwrite silently. It also skips all clean-trained treatments, so the dedicated runtime-corruption comparison is not summarized against clean-trained `none`; retrained models and runtime-corrupted clean models are not presented as separate estimands.

Required repair: define and label at least two distinct comparisons: (1) retrained-on-corruption treatment versus its exact clean-none reference on matched test data, and (2) clean-trained treatment under runtime corruption versus clean-trained none under the same runtime corruption, using validation-selected lambda. Match every graph/trajectory/metric record exactly, flag duplicates/unmatched records, and aggregate matched effects within seed.

## Minor quality findings

- Resource summaries pool domain, training configuration, trajectory count, train size, mechanism, and identity condition by only suite/mode (`study_analysis.py:382-391`). Report them at a useful condition granularity; parameter counts especially should not be averaged across architectures/configurations that differ.
- Aggregate graph precision/recall is computed over raw rows, not first macro-averaged per seed (`study_analysis.py:209-216`), so its `n` counts graph records while error `n` counts seeds. Use the same per-graph-within-seed convention and label `n_seeds` and `n_graphs` separately.
- `seed_values` uses JSON object keys, which stringify integer seeds on disk. A list of `{seed, value, n_graphs}` records is clearer and preserves type/denominator metadata.
- The Markdown terminal filter excludes validation rows but would include any future nonvalidation split without an explicit supported-split check (`study_analysis.py:426`). Render named sections for validation, test, and runtime corruption.
- The summary calls itself “lossless” in the notes, but it drops most runner metrics and graph-level observations. Either retain the graph-level analyzed records/provenance or describe the product as aggregated and point to JSONL as the lossless artifact.

## Test gaps and acceptance criteria

The existing unit tests cover pure threshold arithmetic, endpoint-aware AUC in isolation, simple seed pairing, validation-only selection, and coefficient flattening. They do not exercise a realistic runner row or generated report/plot semantics. The 104-row smoke can succeed while every test metric is the wrong quantity because it checks execution rather than schema meaning.

Before approval, add runner-shaped end-to-end fixtures that establish all of the following:

1. Every one-step/deterministic/stochastic metric, oracle, and zero reference survives with an exact metric label and unit; no raw value is labeled normalized.
2. Pairing succeeds only when run case, seed, metric, graph hash, trajectory seed, split, checkpoint, size, and corruption condition match; graph order does not matter, while missing/duplicate references are excluded and reported.
3. Transfer effects are matched per graph, macro-averaged within seed, then summarized across seeds.
4. Soft selection reads only the configured final validation checkpoint on common seeds; test and runtime corruption cannot change it, and downstream rows identify selected versus sensitivity strengths.
5. A 0..300 curve is censored/invalid for a configured 600-step budget; step 0 and 600 with invalid duplicate/intermediate steps are rejected; a valid 0..600 curve gets the correct trapezoidal AUC.
6. Zero-oracle gaps that are nonpositive, missing, or nonfinite remain invalid/censored; unreached thresholds stay explicit and reached-only plotting does not conceal censoring.
7. Conflicting duplicate runs/evaluations and nonfinite values in any analyzed metric/reference/coefficient are excluded with stable reason counts.
8. Plot/report labels distinguish domain, metric/unit, training configuration, train versus runtime corruption, transfer train sizes, selected lambda, and censoring.

After those changes, rerun the focused analysis tests and a runner-schema smoke. The already reported unchanged full test run need not be repeated unless shared code or runner output changes.

## Scoped re-review of `fcac02d`

The fix wave resolves C1, C3, I2, I4, and I5, and resolves the graph/trajectory portion of C2. It retains all 18 runner evaluation fields with exact metric names, uses the configured optimizer budget, distinguishes validation-selected strengths, separates plots by domain/metric/training regime with censor marks, and gives retraining and runtime corruption separate estimands. I1 and I3 are materially improved: conflicting run duplicates are excluded, duplicate evaluation identities and nonfinite metric/coefficient values are counted, nested-count normalization/reference consistency is checked, and malformed curves do not receive AUCs. The supplied evidence was 13 focused passing tests plus a 104-row real-schema CLI smoke with plots and 18 retained metrics; those commands were not rerun.

**Re-review verdict: one important integrity defect remains; no critical defect remains in the reviewed paths.**

### Remaining important defect — paired modes are not verified to share the recorded schedule and initialization

The design and methods require paired variants within a case to use identical sampled schedules and common parameter initialization, and the runner records `schedule_hash` and `shared_initialization_hash` specifically to audit those claims. `_evaluation_rows()` does not carry either hash into flattened rows, and `_matched_pair_summary()` matches only seed plus validation checkpoint or test graph hash/trajectory seed. Artifacts with different schedules or common initial tensors will therefore still be reported as paired effects.

Minimal fix: carry `schedule_hash` and `shared_initialization_hash` from each run into every flattened row. For each exact seed/case contrast, require equality of both hashes across baseline and treatment before computing any validation or test effect. Exclude a mismatch with a distinct `pairing_provenance_mismatch` reason and retain the two hash values for diagnosis. Add one focused test in which graph/trajectory identities match but one hash differs, and assert `n_pairs == 0`; add the valid matching-hash case. `initialization_hash` itself is expected to differ for architecture-specific tensors, so the audit should use the explicitly shared hash rather than demand identical full states.

### Remaining nonblocking hardening

- Efficiency curve construction filters nonfinite points before validating the step sequence. A nonfinite intermediate value can disappear while endpoints still make the curve look valid. Treat any present nonfinite step/value as an invalid curve rather than filtering it away.
- `thresholds_by_train_count` republishes runner `threshold_steps` even when that count's curve is invalid, and stored threshold steps are not cross-checked against the authoritative rederived crossings. Derive these per-count values from validated curves, or null/censor them whenever the curve is invalid; count discrepancies as artifact-integrity exclusions.
- `_paired_contrasts()` omits a result whose only failure is `duplicate_references`, because its emission condition checks pairs, incomplete, and unmatched only. Upstream evaluation deduplication covers ordinary test duplicates, but duplicate validation checkpoints can be silently absent from the contrast report. Include `duplicate_references` in the condition.
- Resource rows are grouped correctly in JSON but the Markdown table displays only suite/mode, so multiple conditions appear identical. Add domain/config/count or omit the ambiguous table rows from the human report.

The documented CLI command remains compatible with runner output: absent `--config`, it reads the metrics file's sibling `summary.json` and extracts its `config` object. A standalone config JSON also works through `--config`.

## Final scoped recheck of `6d448ce`

**Approved for the reviewed Stage 2 analysis scope. No blocking defects remain from C1–C3 or I1–I5.**

The final fix carries `schedule_hash` and `shared_initialization_hash` into flattened observations and rejects missing, internally inconsistent, or baseline/treatment-mismatched provenance per seed before computing an effect. The exclusion is explicit and provenance-only failures are emitted. It also rejects nonfinite efficiency points without allowing filtered endpoints to certify a curve, suppresses stored crossings for invalid curves, emits duplicate-reference-only contrasts, and makes Markdown resource conditions distinguishable.

Evidence accepted without rerun: 15 focused remote tests passed, including parameterized missing/mismatched provenance regressions, and the real-schema CLI smoke produced paired results successfully. The remaining risks are ordinary report-scale/readability concerns rather than statistical correctness blockers; raw full-study outputs should still be checked for nonzero exclusion counts before narrative interpretation.

## Plotting and evaluation-step follow-up for `7e202ca`

**Approved. No analysis-math or pairing regression found.** Test evaluations now inherit the run's final `completed_steps`, while conflicting explicit evaluation checkpoints are excluded instead of entering the wrong stratum. This aligns terminal metrics with their actual trained checkpoint and leaves metric expansion, exact reference matching, provenance gating, and within-seed aggregation unchanged. Transfer figures now use numeric node-count axes with mode lines and training-regime panels; efficiency color consistency and seed jitter change presentation only.

Evidence accepted without rerun: 17 focused tests passed, and a real-schema smoke completed with zero exclusions and 23 PNG/SVG figure pairs.
