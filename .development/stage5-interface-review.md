# Stage 5 learned-interface review

Independent inspection of `runtime_tasks.py`, `runtime_execution.py`, `runtime_model.py`, and `runtime_study.py` during development. This is a living review, not approval of an unfinished experiment freeze.

## Boundaries that are sound in the inspected implementation

The public/gold dictionaries are separate and the network receives only public tensors. The task runtime contains initial nodes and values, never precomputed call results or returned wrappers. Label generation runs the oracle interpreter on a copy. Predicted execution also runs on a copy and uses a persistent actual register; gold actions and traces enter only the subsequent evaluation routine. Invalid actions reject instead of restoring the correct path.

Initial node types, lexical keys, scalar values, numeric index payloads, frame flags, and directed typed graph relations are exposed. Candidates are shuffled and opaque node IDs are not neural features. Lexically equivalent names, repeated integer literals, and repeated index labels share observable keys. This prevents an accidental identity answer key through repeated-value candidates.

The lowerer selects a **semantic selector**, not necessarily a unique runtime entity. In particular, choosing a shadowed name is followed by exact scope resolution using the supplied current frame. Selecting a field/index slot supplies its name/index to the actual current object, rather than jumping directly to that slot's value. This is a valuable controlled boundary, but does not demonstrate neural selection of the correct lexical binding instance. The runtime performs that disambiguation.

Operation cues are tokenized synthetic operation words; reference cues are noisy random lexical feature vectors. Clauses are segmented and scheduled externally. The benchmark is not ordinary language parsing or learned program scheduling. Nested object access, immutable aliases, and builtin arithmetic sequences are represented; the family called composition currently means sequential pure builtin application, not learned lowering of user-defined nested call frames. The independent exact language supports more than the trained benchmark.

The policy estimator uses the sum of sampled operation/selector log probabilities across the supplied schedule, a detached end-answer correctness reward, and a lagged scalar baseline. This is a valid score-function estimator for discrete execution, distinct from pathwise differentiation. Invalid executions receive zero reward. Sparse reward and actions sampled after an eventual early rejection can cause high variance, but do not invalidate the estimator. Frozen-boundary maintenance is labeled separately.

## Findings communicated to owners before freeze

1. **World intervention:** the runner initially removed the wrong-graph flag without generating corrupted data, making that control a no-op. Public adjacency and exact executor must use the same corrupted world, with original labels retained and the number of changed edges recorded.
2. **Unused graph inputs:** ordinary and protected-learned neural modes initially ignored adjacency. Graph-as-token memory is being added so all matched controls can observe connectivity; extra message passing or logit bias should be the mode-specific intervention.
3. **Evaluation flags:** ambiguous/invalid arguments were initially silently ignored by the task generator. Held-out word ordering was accidentally mapped to paraphrase tokens instead. All advertised conditions must alter the observable task as intended. Fresh random names already appear in every split; an extra renamed condition must be described as a paired intervention or an additional fresh sample, not a new lexical capability.
4. **Metric attribution:** complete trajectories obtained by feeding a neural control's auxiliary lowering into the exact executor are an external-runtime audit, not a trajectory of that control's own computation. Its untrained standalone lifter must not be reported as a meaningful lifting failure. Conditional execution and task metrics need the same distinction.
5. **Timing:** execution timing must exclude the second oracle run used to score traces. Report actual symbolic time separately from diagnostic interpreter overhead.
6. **Confidence accounting:** hypothetical local gates on every clause differ from actual executed invocations after earlier rejection/deferral. Report both explicitly. Aggregate probability over observably equivalent selectors, and normalize entropy by the valid semantic support rather than padded batch width; otherwise repeated values and padding create artificial uncertainty. Genuine ambiguous examples require an explicit ambiguity/defer interpretation rather than silently treating an arbitrary gold branch as uniquely correct.

All findings were sent to the model, task, and runner owners. Final resolution and focused regression evidence are required before this review can approve the frozen experiment protocol.

### Model follow-up

The model owner has implemented graph edge-record tokens for all four neural controls, including the protected learned-register model. Inspection confirms source/destination projections retain direction, relation embeddings retain edge type, and graph-data/soft modes add their own interventions on top of the same edge-record interface. Semantic-group confidence now aggregates public equivalence groups and uses per-example valid support, with padding invariance covered by a focused test. These address findings 2 and the model-side part of 6; final integrated tests still need task-provided groups.

### Integrated pre-freeze re-review

Re-inspected the integrated task/model/runner changes through task commit `6bb92b6`, model commit `efbe539`, and runner commit `110cf1a`. Wrong-graph generation now supplies the same changed edges to tensor inputs and exact execution. Held-out surface order is distinct, missing-reference and invalid-schema conditions are real, and undefined queries no longer receive ordinary task-accuracy denominators. Neural control trajectory/lifting fields are marked not applicable, with exact-lowering audits separately labeled. Fixed confidence thresholds are declared in advance; they are not optimized on test performance.

Task-generation labels now also have independent generator-side scalar/leaf checks against the interpreter, avoiding reliance on the evaluator alone as its own correctness oracle. Raw `n0`-style IDs remain deterministic host identifiers but never enter neural inputs; shuffling candidates and random lexical keys prevent a numerical-ID shortcut.

The neural comparator is a small clause transformer with a recurrent GRU controller and ordinary content attention over node/edge records. It should not be called a pure decoder-only transformer. It receives the same lowering auxiliary supervision as the supervised runtime variant, plus direct numeric-result supervision; this extra target favors the comparator but means losses are not literally identical. Exact execution is the intended architectural privilege, not a learned reasoning achievement.

Two final metric corrections were requested before freezing: actual confidence error counts under the permuted-binding intervention must score the **executed** selector instead of its pre-intervention prediction, and training-time neural accounting must not silently include gold diagnostic interpreter runs. These are metrics corrections, not an observed execution or model-input leak.

A single scalar-lifting representation revision is being evaluated using shallow oracle pilots only, after the original shallow oracle exposed a learned-output bottleneck. Preserve both pilot artifacts and disclose the revision; no OOD measurements may be used to select that representation. Approval of the main freeze remains conditional on the pending metric corrections and focused integrated tests after that revision.

### Final source review and verification

At `9965a15`, actual invocation correctness is computed from the executed trace's operation and semantic selector, including after permutation. Training timing is honestly labeled `neural_and_diagnostic_train_seconds`; actual predicted interpreter time remains separate. Undefined-query task and conditional execution/lifting rates are `None`, and neural-controller exact-runtime audit metrics remain distinct from neural numerical predictions.

The RBF scalar-lifting revision is documented as an explicit bounded numerical representation prior, followed by a learned MLP. It contains no output rule, preserves scalar gradients, and does not discretely look up the target label. Both shallow pilot configurations, raw rows, and exact source archives are committed. This is representation selection on shallow development data, not OOD selection.

An independently copied current source snapshot ran **67 focused tests successfully** on the remote machine with two CPU threads, covering runtime model, runner, generated tasks, exact runtime, and language. The complete diff from Stage 4 contains additions only. No Stage 1–4 file was modified.

One final narrow metric correction was requested: risk–coverage `correct` counts must also mask undefined queries, because their stored numeric zero is a placeholder rather than a correct answer. With that correction and its focused regression verification, the interface is approved for the planned frozen experiment. Remaining limitations are scientific scope, explicitly documented above, rather than implementation blockers.

The final correction is committed as `c62c62f`. Inspection confirms undefined targets are masked from confidence correctness and `defined_examples` is recorded. The new regression forces an accidental valid zero-valued completion on an undefined query and verifies two answered / zero correct at every threshold. The runner owner reports all **11 runner tests passing** after this fix. **Approved for the frozen Stage 5 experiment.**
