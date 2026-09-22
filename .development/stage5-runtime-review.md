# Stage 5 independent runtime and design review

Reviewed `tiny_runtime.py`, `tiny_language.py`, their tests, and `stage5-design.md` during implementation (initial runtime commit `9ada36b`). This review covers the exact runtime and declared experimental design; it does not certify the separately evolving neural benchmark or experiment results.

## Exact semantics

No blocking issue found in the inspected core. The parser constructs a small AST and the evaluator dispatches an explicit finite set of language forms. There is no host-language `eval`, dynamic source execution, or user-supplied executable callback in language execution. The `Primitive.lower/lift` callback hooks are trusted application interfaces, not syntax available to generated programs.

Bindings, immutable scalar/record/array/function values, field/index slots, argument slots, call instances, frames, and returned wrappers have distinct node identities. Alias bindings point to the same underlying value. Array order and noncommutative builtin arguments are preserved. Reading a record or array resolves a slot then its value; name resolution follows lexical parent frames.

Each call allocates a distinct invocation and fresh frame. User function frames are parentless, receive parameters and global function definitions, and cannot capture outer data variables. Nested function definitions and direct/indirect recursive activation are rejected. Global function-name availability is the explicitly implemented namespace convention; this is not closure semantics. Functions are evaluated synchronously: the public `call` primitive returns the result value after internally creating its frame, argument positions, and returned wrapper. It is not a resumable coroutine or independently learned call/return scheduler.

Call validation rejects invalid arity and builtin operand types before allocation. The transaction also rolls back nodes, edges, trace, counter, lexical maps, function registry, builtin registry, and activation stack if a user body fails after partial execution. Other exposed primitive methods validate before writing. Return wrappers and scalar call results are allocated during execution, not at definition time. A trace node-count boundary now tests that return/result nodes did not exist at call entry.

The runtime has Python-level public dictionaries and trusted construction APIs; its immutability contract concerns language operations, not an adversarial host mutating those dictionaries. This is appropriate for a controlled interpreter, not a security sandbox claim.

## Independent checks

Manual lightweight Python checks, loading only the new runtime modules without Torch, passed for nested-block return, user-function composition, immutable aliases, lexical shadow restoration, indirect-recursion rollback of the complete runtime dictionary, rejection of captured outer variables, bad arity, and out-of-range indexing. The core author reports all initial 11 focused tests passing. Two extra regression tests (indirect recursion rollback and nested-block return) and a documentation correction to include argument slots in `value_of`'s schema were requested; verify their final test result before integration.

Previously committed Stage 1–4 files were unchanged in the inspected diff. New work remains additive.

## Design assessment and conditions for the scientific report

The design appropriately distinguishes exact language support from learned benchmark coverage. Segmentation and scheduling are supplied computational structure, not learned planning. The same public schedule and initial runtime information must be exposed to matched controls; an unscheduled ordinary transformer is a secondary comparison with a stronger interface difference.

Gold paths, post-execution returned nodes, gold-restricted candidates, and teacher-forced register resets must never enter model observables. Type/schema masks may depend on the actual current state and predicted primitive, not the gold operation. After a mistaken binding, subsequent execution must follow that mistaken state or reject, rather than silently resume the reference trajectory.

Argmax lowering followed by exact execution has no pathwise task gradient. Weak-supervision results require an explicitly described estimator or differentiable surrogate; an unchanged/frozen lowerer at auxiliary weight zero is a maintenance control, not task-supervised acquisition.

Confidence must be tested on uncertain and genuinely ambiguous inputs. Choose thresholds using validation only, count deferrals as unanswered, report risk and coverage, and distinguish schema rejection from semantic correctness. Repeating the same inference without new evidence or a learned update is not recovery. Conditional execution accuracy and conditional lifting accuracy should be reported separately so boundary failures do not obscure exact semantics.

The runtime core is ready for integration subject to the final focused regression checks. The benchmark/model and raw experiment artifacts require their own leakage, fairness, gradient, and provenance review.
