# Stage 5 methods: learned lowering, exact execution, learned lifting

This stage tests a deliberately narrow neural/runtime boundary. A learned lowerer selects semantic operations and operands from noisy lexical features; a small immutable interpreter executes those selections through protected state; a learned output head interprets the resulting scalar. The experiment does not ask the network to discover the language semantics or invent a program schedule. The independent exact language has broader syntax than the learned benchmark. Those two deliverables must not be conflated.

The main experiment freezes source at `2e95c9f` with 400 updates, 200 warm-start updates, three paired seeds and ten variants. Results remain pending at this methods revision. The final report and archived resolved configuration record actual coverage and provenance. Earlier stages are preserved.

## Exact language and runtime

[`tiny_language.py`](../src/topoformer/tiny_language.py) tokenizes and parses a finite grammar into syntax trees. It supports `let`, numeric literals, immutable records/arrays, field/index access, top-level pure function definitions, calls, lexical blocks and returns. Arithmetic uses explicit builtins (`add`, `sub`, `mul`, `neg`). There is no generated host-language `eval`, mutation, loops, recursion, closures or arbitrary executable callback in program syntax. A leading record literal needs parentheses because an initial brace introduces a lexical block.

[`tiny_runtime.py`](../src/topoformer/tiny_runtime.py) assigns append-only stable IDs. Names, binding nodes and values are distinct. Field and index slots connect to values through `value_of`; aliases have separate bindings sharing a value. Function definitions, invocations, argument positions, frames and returned wrappers have distinct identities. Directed typed edges include binding/slot/value relations, callee/argument relations, lexical parents and returns. Return wrappers and computed scalar nodes are allocated during execution, never precomputed into the model's initial world.

Name resolution follows lexical parent scopes. User functions have fresh parentless frames containing parameters and available global function definitions; they do not capture caller data. Direct and indirect recursive re-entry is rejected. Calls run synchronously: the public convenience method returns a result value after internally allocating its call/frame and returned wrapper. This is an exact interpreter, not a learned or resumable call scheduler. Failed calls roll back graph, scope and trace state transactionally; invalid reads validate before changing state.

Primitive descriptors expose a typed schema, validity constraints, executor and trusted lowering/lifting hooks, corresponding to `(Σ,V,T,L,U)`. The runtime implements name resolution, slot reads, calls and returns; the neural experiment supplies a separate learned adapter. Public Python dictionaries are inspectable but assumed read-only by clients. Language immutability is not a claim of an adversarial host security sandbox.

## What the neural experiment observes and learns

[`runtime_tasks.py`](../src/topoformer/runtime_tasks.py) supplies already segmented, execution-ordered clauses. The seven learned operation labels are `resolve`, `field`, `index`, `add`, `sub`, `rsub`, and `mul`. A clause has a small vocabulary of synthetic operation words plus a noisy random continuous lexical reference key. Candidate descriptions expose keys, node types, scalar values, index payloads, current/root frame indicators and the entire initial typed graph. Candidate rows are shuffled; opaque runtime IDs and fixed candidate positions are not neural inputs.

Reference keys identify observable semantic selectors, not unique gold entities. Two shadow bindings named `x` share a name key; repeated index `2` slots share an index key; repeated literal values share a literal key. A supervision mask accepts every equivalent selector. The exact runtime then resolves that selected name in the actual scope, or reads the selected field/index from the actual current object. Consequently, name-selector accuracy is **not learned lexical-scope disambiguation**. Scope semantics remain supplied by the interpreter.

Arithmetic clauses select a builtin operation and a scalar operand. `rsub` reverses the current-register/operand order. This tests learned operation recognition and ordered operands around exact arithmetic. It does not test neural selection of arbitrary user-function definitions, construction of nested argument frames or program planning. The independently tested language supports nested user-function calls; the learned family called “composition” consists of sequential pure builtin applications.

The six families are nested alternating record/array lookup, lexical shadowing followed by arithmetic, builtin composition, noncommutative argument ordering, immutable aliases, and mixed lookup/output interpretation. `depth=D` means D operations **after** initial resolution, hence D+1 clauses. `nodes=N` means N additional distractor bindings, **not N total runtime nodes**. Increasing depth also grows the runtime graph. Reports include actual candidate-node counts.

Nested lookup, alias and mixed worlds each include a matched decoy chain with identical depth, field names, index labels and shape but a different root binding and terminal scalar. Swapping only the initial name selector therefore changes the answer; the remaining references are unchanged. Both chains are checked by the exact executor. This prevents a unique terminal-field key from revealing the result without retaining object context. There is one such deep decoy per world, rather than one per distractor binding, to bound dense graph cost.

Output styles are independently sampled: numeric result classification over integers −64…64, binary comparison with an observable comparison cue, or three-class sign interpretation. Generation keeps results within that benchmark range; the interpreter's arithmetic is not itself a 129-class operation. Learned lifting is a small MLP over scaled scalar result, style embedding, comparison cue and fixed Gaussian radial-basis features centered at all 129 integers from −64 through 64 (width 0.75). These continuous numerical features are an explicit bounded-vocabulary prior. They contain no sign/comparison/report rule, do not clip the scalar, and still require learned output weights. This tests a narrow latent-output round trip, not natural-language generation or unrestricted downstream reasoning.

A paired pre-main shallow oracle pilot isolated a scalar-only lifting optimization failure. The same 400-update one-seed pilot with radial-basis features changed validation output accuracy from .375 to .6875; separate shallow evaluation after the change was .703125. Config and training schedule matched, with only the model source hash changed. Those pilots used the earlier generator without the matched decoy chain and are archived separately; they are not main results or OOD evidence. The representation change was selected using shallow diagnostics.

## Architecture and protections

[`runtime_model.py`](../src/topoformer/runtime_model.py) uses a small clause transformer with within-clause positions, a learned reference projection and separate operation/cosine-selector heads. Query and candidate projections start randomly. Candidate representations also consume learned type/value/payload features. A learned null candidate permits unbound lowering. Training on valid examples does not teach unanswerable inputs explicitly.

[`runtime_execution.py`](../src/topoformer/runtime_execution.py) deep-copies the initial world, lowers predicted actions into exact primitives and maintains actual protected entity/type/frame state. Generic MLPs cannot write these registers. There is no gold register reset after a wrong action: subsequent primitives use the mistaken current state or reject. Symbolic state need not be a token's complete latent vector, and ordinary attention is not forced to become one-hot.

The public and gold dictionaries are separated. Forward inputs contain neither gold selectors, gold paths nor future return nodes. Oracle label generation executes a separate copy. Evaluation consults gold only after predicted execution. This causal boundary is tested independently from predictive performance.

## Controls and losses

All neural variants share the observable clauses, candidates, initial values, directed graph and output cues. Ordinary neural prediction receives graph edge-record tokens encoding ordered endpoints and relation types. Graph-as-data additionally applies typed message passing. Soft structure additionally induces an attention-logit bias from learned grounding and a relation mixture. Protected learned transition maintains a separate GRU-updated register without exact runtime semantics. Each unrolls once per supplied clause. They are small matched recurrent neural controls, not architecture-optimal baselines.

| Variant | Lowering supervision and execution |
|---|---|
| `oracle` | Privileged exact operation/selector targets; exact runtime; learned lifter |
| `supervised` | Cold learned lowering, auxiliary weight 1; exact runtime |
| `task_only` | Cold lowering, no auxiliary labels; answer-reward policy gradient |
| `warm_weak` | Supervised warm start, then weight 0.1 plus answer-reward policy gradient |
| `warm_task` | Supervised warm start, then no auxiliary labels plus policy gradient |
| `warm_frozen` | Same warm start, then frozen lowering; lifter continues training |
| `neural` | Ordinary graph-token attention/recurrent prediction |
| `graph_data` | Additional typed message passing |
| `soft_structure` | Additional grounded structural logit bias |
| `protected_learned` | Separate learned register, no exact transition |

Supervised lowering sums operation cross-entropy and negative log probability mass on equivalent selectors. Exact variants train the lifter on end-task labels from the actually executed scalar. Invalid execution has no correct task answer. Argmax and host-side exact execution have no pathwise gradient. Policy variants sample the complete action schedule and use REINFORCE with detached final-answer correctness, summed action log probabilities, and a lagged scalar reward baseline (decay 0.9). Invalid executions receive zero reward. This sparse estimator is distinct from differentiating through the interpreter and can have high variance. Its reward is final output correctness, not exact numeric-result correctness: a wrong program may still produce the correct sign or comparison label. Failure to recover exact computation under this reward must therefore be interpreted alongside reward non-identifiability and sparse credit assignment, not as proof that task-only binding is unlearnable.

Neural controls receive auxiliary lowering supervision at weight 1 and an extra numeric-result classification target in addition to the output target. This favorable baseline supervision is disclosed rather than calling objectives identical. Their auxiliary lowerings can be audited through an external interpreter, but that audit is not their own neural execution trajectory. Their unused standalone lifter is not scored as a lifting failure.

## Separately disclosed semantic-read supplement

The main sweep remains frozen. After launch, while partial first-seed outcomes were visible, a separate stronger-control experiment was specified to address a retrieval-interface confound. Main neural controllers learn auxiliary semantic selectors but do not directly consume those distributions. The supplement supplies learned-selector graph-data, learned-selector protected recurrent state, and privileged oracle-semantic-selector protected recurrent state controls under the same 400-update, three-seed schedule.

The added read sums selector probabilities against observable node memory and combines operation probabilities with existing operation-head weights before the learned recurrent update. Graph-data reads its message-enriched memory. No exact pointer operation, arithmetic or lexical-resolution algorithm is added. The change introduces no parameters or random draws; an independent comparison found identical seeded parameters and bitexact default outputs against main source `2e95c9f` in all five modes. Artifact hashes must still establish pairing before comparing completed runs.

The privileged condition supplies one-hot operations and uniform probability over all semantically equivalent selectors, not a gold unique runtime entity. Reported effective interface accuracy there describes oracle information, not acquired grounding. Its own numeric answers remain neural predictions. This supplement was declared after seeing partial main results and must not be presented as preregistered or silently merged into the original ten variants. See [runner notes](stage5-runner-notes.md) and [`stage5-selector.json`](../configs/stage5-selector.json).

## Curriculum and evaluation

The frozen main configuration [`stage5.json`](../configs/stage5.json) specifies seeds 0/1/2, 400 updates, batch 16, width 32, learning rate 0.001 and a 200-update supervised warm start. Training cycles through depths 1–4 and distractor counts 8/16, alternating canonical/paraphrased templates. Checkpoints include step 0 and updates 25/50/100/200/300/400. Final evaluation crosses depths 4/8/16/32 with distractor counts 8/32/64, using 64 examples per condition and seed. The bounded pilot preceded the source/config freeze. Main evaluation uses the fixed final checkpoint, with no OOD checkpoint selection.

Fresh lexical keys and candidate shuffles appear in every split. The extra `renamed` condition drops its name flag and reuses the same evaluation seed/settings as D4/N8, so it is an identical redundant condition, not an independent sample or distinct lexical-generalization test. Fresh opaque names and continuous reference keys across generated batches test feature-identity transfer only; they do not establish a held-out textual-vocabulary capability. Other tests cover held-out word order/templates, medium/high reference noise, missing-cue ambiguity, invalid schemas, permuted operand bindings, and a corrupted runtime world. Corruption changes observable scalar-value edges and the executor's world together while retaining clean targets. Permutation intentionally breaks the lowering-to-runtime correspondence and is an intervention rather than another trained model.

Missing-cue and invalid-schema examples have no defined answer. Deferral or rejection is reported as abstention, never correct task execution. Placeholder numeric labels are excluded from their answer accuracy. Family/style class frequencies should accompany results because output imbalance can make coarse end-task accuracy easier than exact arithmetic.

## Local confidence and metrics

Confidence combines normalized entropy and top-two margin of operation and selector distributions, multiplied by non-null probability. Selector probabilities aggregate over public semantic-equivalence groups; normalization uses each example's valid groups rather than batch padding. The score is not a calibrated probability. Per-primitive thresholds sweep 0, .05, .1, .2, .4, .6, .8 and .95. A gate also requires actual runtime type/schema validity. This implementation selectively invokes or defers; it does not run an automatic medium-confidence inference loop or learn independent confidence policies for several register subspaces. Any selected operating threshold must come from validation, not OOD optimization.

Metrics separate per-clause selector and primitive accuracy; whole-schedule correct lowering; exact runtime result; complete semantic trajectory; end-task output; and lifting supplied the correct result. Scalar trajectory states compare values, while object states compare entity identity. Distinct newly allocated scalar nodes with equal values are therefore semantically equal. Conditional execution given correct lowering isolates executor reliability. Task accuracy given correct binding is weaker conditioning because primitive recognition may still fail.

Confidence tables distinguish actual executed invocations from hypothetical local gates after an earlier stop. Report invocation/deferral/rejection counts, accepted incorrect lowerings, unnecessarily deferred correct lowerings, risk versus coverage and defined-answer task accuracy. Recovery must mean a wrong actual intermediate state later becomes correct, not repeated unchanged inference. Neural controls' external-runtime audit is labeled separately throughout.

Timing records neural training/evaluation and symbolic execution separately. Stored execution time includes world copying/execution but excludes oracle trace scoring; whole-run wall time additionally includes evaluation, logging and diagnostics. Peak RSS is a process high-water mark. Neither timing is a hardware-independent complexity law.

## Reproducibility and interpretation

The runner records resolved config/source hashes, per-seed initialization/final parameter hashes, paired data/schedule hashes, step-zero diagnostics, training losses, final raw counts, condition/family summaries and failure traces. Checkpoints remain on the compute machine; compressed raw metrics, configs, summaries, plots and audits are committed. Three-seed mean and sample standard deviation summarize observed runs, not a precise population-confidence claim.

A positive result would show that learned lowering drives exact protected computation and learned output interpretation beyond training depth. It would not show spontaneous crystallization, language induction, human-like cognition or superiority to all graph/message-passing approaches. Binding, primitive recognition, schema, execution and lifting errors are reported separately; interpreter capability is not evidence that a neural model learned all supported syntax.
