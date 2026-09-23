# Stage 5: learned lowering into a protected runtime

The user's specification is the authority: implement a tiny auditable immutable language, then isolate learned semantic binding, exact execution and learned lifting. Preserve every Stage 1–4 source/config/test/artifact. No pretrained model, arbitrary host execution, loops, mutation, recursion or closures.

## Scope and scientific boundary

Two complementary deliverables share one runtime. The exact language supports let, integers, records, arrays, access, pure definitions/calls, lexical frames and returns. The learning benchmark uses public segmented surface instructions and the same initial runtime for every control. Scheduling/segmentation is supplied, not learned program induction. Generated canonical and paraphrased phrases, randomized names, shuffled candidates and noisy reference features test a controlled lowering boundary; this is not open-domain language understanding.

A runtime node has a stable ID and type. Bindings, values, field/index slots, function definitions, call instances, argument positions, frames and returned values are distinct. Typed schemas validate before transition; no invalid invocation may partially change state. Return/value nodes are allocated only during actual execution, never precomputed into model inputs. Immutable aliases share the underlying value; name resolution follows lexical parent scopes.

## Learning and controls

A small transformer encodes each public surface clause and observable reference/candidate descriptions. It predicts operation and binding distributions with null capacity. Protected registers store actual current runtime identities/types/frames; ordinary MLPs cannot overwrite them. Exact execution uses predicted lowered operands and actual current state, never a gold reset after an error. Learned lifting maps the runtime scalar result and public output-style/comparison cue back to a neural output task. Numeric execution accuracy is measured before lifting.

Compare exact oracle lowering, supervised cold lowering, answer-only cold lowering, reduced/removed auxiliary supervision after a supervised warm start, ordinary neural prediction, graph-as-data/message passing, soft structural attention, protected learned transitions without exact execution, and exact transitions. All receive the same initial values, names, graph, surface instruction schedule and output cue; oracle identities are explicitly privileged. Permuted binding/wrong-graph interventions and type-invalid examples are separate tests.

Argmax plus exact Python execution has no pathwise task gradient. Weak supervision must use a declared score-function/policy-gradient estimator or a separately labeled soft surrogate; merely freezing the lowerer is not task-supervised acquisition. Include a warm frozen-binding maintenance control if practical. Step-zero measurements distinguish supplied priors from learned behavior.

## Confidence

Local per-primitive confidence combines operation/binding entropy or margin with actual type/schema validity. No global mode. Fix thresholds using validation only; use low/medium/high noise and genuinely ambiguous cues. Low confidence defers (counts as unanswered, not correct); medium confidence can retain soft hypotheses; high confidence plus validity invokes exact execution. Report risk versus coverage, wrong execution, unnecessary deferral, schema rejection and conditional accuracy. Do not call repeated identical inference recovery. A soft-to-correct hypothesis change must be measured explicitly.

## Experiment protocol

Three paired seeds, shallow training depths 1–4. Evaluate depths 4/8/16/32 and object counts 8/32/64, plus renamed identifiers, held-out templates, lexical shadows, nested calls, argument reversal, aliases, invalid schemas and uncertainty. Fixed final checkpoints; no OOD checkpoint selection. A bounded pilot sets a feasible CPU budget before freezing the main configuration. Separate gold targets/trace metadata from observable inputs by API. Save source/config/data/initialization hashes, raw counts, step-zero/learning curves, failure cases, complete trajectory accuracy, execution conditional on correct lowering, and lifting accuracy conditional on correct runtime results.

The central result must distinguish binding, primitive recognition, schema, execution, lifting and interference failures. Exact runtime correctness is established independently with tests. Report scope and any unimplemented/evaluated aspect explicitly; do not conflate interpreter support with learned benchmark coverage.

## Resource policy

Train and run full Torch tests on gb10-direct, existing venv, two CPU threads, sequential jobs and bounded batches. Keep local work to source edits and small artifacts. No new package needed beyond existing Torch/pytest/plotting setup. Commit notes, source and compressed raw artifacts; integrate verified work to main under existing authorization.
