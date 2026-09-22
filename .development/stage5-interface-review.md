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
