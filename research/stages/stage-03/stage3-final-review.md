# Stage 3 independent scientific review

## Status

**Whole-study verdict: APPROVED for integration.** Source, pairing, completed numerical artifacts, final narrative, and presentation pass the scientific review. No blocking or moderate implementation, leakage, pairing, arithmetic, or interpretation finding remains. This approves the report's bounded conclusions, including its negative milestone decision.

This independent review owns only this document. No training, product edits, or redundant test runs were performed. Reviewed design/methods, prior architecture/leakage reviews, traversal data/model/oracle integration, runner, analyzer, and raw main/keyed artifacts. The six IID pilot rows remain a separate family and are not pooled with the main study.

## Verified source and experimental integrity

- Main artifacts contain the complete **39-run grid: 13 variants × 3 seeds**, each with 400 optimizer steps and checkpoints 0/25/50/100/200/400. They record a clean source checkout at `2adc87ca90d42682aa3a592ccdec6f90201c9c98`.
- The stronger keyed supplement contains the complete **9-run grid: 3 variants × 3 seeds**, recording a clean checkout at `27ca0a2c1f6f77f0072eba69be513cd985603280`.
- All recorded product source-file hashes were independently compared with their corresponding Git revisions and match for both families.
- Every main variant within a seed has the same training schedule hash and condition-specific evaluation data hash. Every ordinary recurrent parameter has the same initialization hash across all 13 variants, excluding the intentionally changed grounding/strength parameters. The keyed variants additionally have identical **full** initial-state hashes within each seed.
- All main numeric fields are finite, and all clean direct and exact-attention oracle task/path accuracies are exactly 1.0. The keyed clean exact-attention controls likewise score exactly 1.0.
- Comparison with the pre-Stage-3 tree at `aed7404` shows no changes to Stage 1/2 source or existing result artifacts; changes are new Stage 3 code/configuration/tests/docs/results and README updates.
- Independent standard-library calculations from raw JSONL agree with all **3,497 main** and **807 keyed** aggregate metric mean/sample-SD cells, and all individual effects in **630 main** and **30 keyed** paired comparison records. Stage 3 intentionally reports sample SD, unlike Stage 2's population SD; these are descriptive and are not confidence intervals.

## Scientific contracts

The model input allowlist in `grounding_study.py:151–154` contains only runtime entity keys, supplied adjacency, metadata IDs, shuffled memory keys/values, start key, and instructions. Training (`285–298`) uses only final-answer cross-entropy. Gold intermediate nodes, token-node mappings, clean adjacency, and destination labels are evaluation-only diagnostics. Random identity keys precede and are independent of payloads/topology; the shuffled memory contains the payload information by task definition. No target or intermediate-state leakage was found.

The learned route is differentiable `Pq A Pkᵀ`, with null mass excluded rather than redistributed. Source-row/destination-column adjacency is consistent with data generation and attention. Grounding is recomputed from the updated recurrent state except in the explicitly frozen control. The hard control blocks grounding gradients with argmax and must remain an initialized routing control, not a learned discrete grounder. The graph-input control transports destination information into source-token values, so destination-token attention accuracy is not mechanism-comparable to its retrieval.

The first entity-key coordinates are deliberately aligned at initialization; the query is given the start key, and instructions externally schedule one recurrent update per edge. Thus deeper evaluation has proportionally more computation and an explicit relation schedule. These are substantial priors, not leaked answers. Neither arbitrary lexical identity discovery nor autonomous program execution is measured.

Corruption rewires supplied functional edges while retaining clean targets. It can destroy information needed to recover the clean answer. Exact oracles follow the supplied graph, so corrupted-oracle failure is not a failure of exact routing. Clean and corrupted conditions use common generation seeds and identical underlying clean task data.

## Material interpretation requirements

1. **Task classes collide.** Uniform guessing is 1/8, but this is not an upper bound on graph-free prediction using memory class frequencies or return/cycle priors. Empirical no-graph controls often achieve 20–25%. Above-12.5% answer accuracy alone does not establish graph use. Keep exact grounding/path diagnostics and empirical controls beside task accuracy.
2. **Main soft success is bounded and prior-dependent.** At depth 4, primary soft task/path means are 0.8672/0.7396; at depth 32 on 16 nodes they are 0.6771/0.3568. Random initialization is 0.2188/0.0000 at depth 4 and 0.2396/0.0000 at depth 32. This is evidence for feasibility with an identity-coordinate prior, not discovery of identity binding from scratch.
3. **Joint extrapolation is a failure for the learned default.** At depth 32 and 64 nodes, primary soft reaches 0.2448 task accuracy but only 0.0234 complete grounding trajectories. Known cosine grounding reaches 0.7500/0.6823. Main learned-temperature/shared-strength variants also have low joint exact-path rates. Report this explicitly rather than infer scalability from separate depth and size axes.
4. **Structural strength has a tradeoff.** Strength-4 soft reaches 0.8438 task/0.6719 exact path at depth 32 on 16 nodes, but only 0.4010 task/0.1771 path at depth 4 on 64 nodes and 0.1901/0.0000 in the joint condition. This is not an across-condition best setting and must not be retrospectively selected on test conditions as a universal winner.
5. **The keyed graph-input control materially changes the architecture comparison.** With the shared graph-free cosine content prior β=8 and structural strength initialized to 16, graph-input-keyed achieves task/path 0.9974/0.9245 at depth 4 and 0.9714/0.7083 at depth 32, versus soft-keyed 0.4740/0.2083 and 0.3620/0.0234. The main graph-input failure therefore cannot be treated as a bound on graph-as-data methods. The keyed family is separate; do not pool it into the main paired analysis.
6. **Shared prior does not imply identical mechanistic effect.** The β=8 content prior attracts attention to the current identity, which retrieves the successor message in graph-input but competes with destination routing in soft attention; λ=16 compensates without making the mechanisms identical. The supplement demonstrates a strong alternative under a declared prior, not universal graph-input superiority. It also fails jointly: graph-input-keyed reaches only 0.2682 task/0.0208 exact path at depth 32/64 nodes.
7. **Depth is not distinct entities.** Functional random graphs permit cycles and revisits. Report measured distinct visited nodes at long depth, and avoid equating a depth-32 instruction sequence on 16 nodes with 32 novel bindings.
8. **Diagnostics are not all operational.** None-mode grounding/null probes are unused by its inference; frozen post-update probes do not change its cached routing; graph-input next-token attention has the source-message caveat. High answer accuracy with lower complete-path accuracy, including keyed graph-input, must remain visible.

## Final gate scope

The completed `stage3-report.md` arithmetic, milestone conclusion, main-versus-supplement labeling, diagnostic negative results, figure links/labels, exact test evidence, and source-bound artifact manifests were reviewed. No training rerun was required.

## Narrative/artifact review

The 46 files listed in `results/stage3/audit.json` independently match their SHA-256 hashes. This manifest keeps 39 main, 9 keyed, and 6 IID pilot runs separate. Main depth and keyed joint-depth figures were visually inspected: condition labels, task/path panels, error bars, and family-specific legend entries are legible and agree with the data.

An independent direct-raw check covered **266 main narrative table mean/SD cells** and parameter counts. Five last-digit rounding discrepancies were sent to the report author (soft-8 D4 query grounding; known inference SD; learned-temperature total-time SD; period-4 total-time SD; untyped inference SD). Other checked cells agree.

### Important interpretation finding: traversal exists before training

The parent identified, and this reviewer independently verified, a critical limitation from the already-recorded step-zero validation diagnostics. At depth four, `soft` strength 8, `known`, `hard`, `learned_temperature`, `shared_strength`, and `period4` all have **1.0 initial complete grounding paths** while initial answer accuracy is approximately 0.1016. For default soft, training raises validation answer accuracy to 0.8385 but lowers complete path accuracy to 0.7656. Hard improves answer accuracy to 0.8021 at step 25, then deteriorates to 0.2995 with complete path 0.0417 at step 400. These are preservation/readout and optimization-damage results, not evidence that those variants learned traversal from an initially non-traversing state.

The supplement has the same issue: `graph_input_keyed` starts with complete path 1.0 (answer 0.1068), and `soft_keyed` starts with complete path 0.9948 (answer 0.1016). Their final validation complete paths are 0.9505 and 0.2839 respectively. The strong graph-input supplement therefore demonstrates preservation and readout of initialized traversal under the declared content prior.

**Strength 4 supplies the clearest acquired-routing evidence:** its initial validation query grounding is 0.3828 and complete path is 0; after training these become 0.9701 and 0.9271, with answer accuracy 0.9505. This still operates under identity-coordinate initialization, not arbitrary latent/semantic binding. These step-zero observations concern the recorded depth-four validation condition; no unmeasured initial depth-32 performance should be inferred.

The final narrative must place this distinction prominently in the executive result, supplement, and milestone conclusion. It must also avoid calling corrupted exact-traversal accuracy a maximum achievable clean-answer accuracy: that procedure is a diagnostic, not a proved Bayes-optimal ceiling. Correct final answers after a failed complete-path diagnostic can result from shared payload classes, recovery to the destination, or imperfect alignment between diagnostic argmaxes and answer readout, so the gap cannot be uniquely attributed to class collisions.

## Final resolution and evidence

Narrative/methods revision `36fea1b` resolves the material interpretation findings. Its executive result explicitly distinguishes initialized strength-8 routing from acquired strength-4 routing; the training section presents step-zero through step-400 task/query/path measurements, and the supplement discloses its near-perfect initialized paths and subsequent optimization damage. The stronger keyed graph-input result is prominent, with the current-identity versus destination-token bias interaction explained and main/supplement families kept separate. Joint large/deep failures remain explicit and large-model/interpreter follow-ups remain deferred.

The report now describes corrupted exact traversal as a diagnostic rather than a Bayes-optimal ceiling, lists multiple explanations for task/path gaps, qualifies the known-cosine mechanism comparison, and distinguishes uniform random class guessing from an empirically stronger graph-free baseline. It also discloses that `random_init` changes both projection alignment and the null-logit prior (0.65 to zero), so its null failure cannot isolate projection initialization alone.

Final numerical checks cover **290 main/keyed table mean/sample-SD cells**, model parameter counts, and **72 scalar values** in the new validation-trajectory table. The previously flagged last-digit table discrepancies were corrected. Two additional nonblocking double-rounding nits in the new validation table were sent to its author: soft-4 step-50 answer accuracy should display .661 rather than .662, and hard step-400 answer accuracy should display .299 rather than .300. Neither changes any interpretation or underlying artifact.

The final manifest independently verifies **all 50 artifact SHA-256 hashes**. It records main training source `2adc87ca90d42682aa3a592ccdec6f90201c9c98`, keyed training source `27ca0a2c1f6f77f0072eba69be513cd985603280`, and final analyzer source `85e6017dd848dabff878669d7a87643b6f3e21a0`. The exact-source gate at that analyzer revision records **215 passed in 2.17 seconds**; this reviewer did not redundantly rerun it. The final analyzer change adds training-curve rendering without changing training or aggregation.

Both new primary/keyed training-curve figures were visually inspected and correctly show sample SD, optimizer checkpoints, fixed validation examples, separate task/path axes, initial routing, and optimization damage/recovery. Main depth and keyed joint-depth figures also passed visual inspection. All **20 final narrative links** resolve. Stage 1/2 source and raw results remain unchanged.

**Recommendation: integrate Stage 3.** The evidence supports bounded routing feasibility and acquisition under declared priors, together with important failures of random initialization, optimization preservation, and joint size/depth transfer. It does not establish learned semantic binding, a scalable latent interpreter, an architecture-wide advantage of structural logits over graph-input methods, or an information-theoretic corruption ceiling.

Coordinator closeout: both nonblocking validation-table rounding corrections were applied before integration (soft-4 step 50: .661; hard step 400: .299).
