# Stage 5: learned lowering into a protected executable runtime

**Experiment status: implementation and protocol preparation; results pending.**

This report will distinguish the independently verified tiny language from the narrower learned semantic-selector benchmark. No empirical success claim is made before the paired runs and artifact audit are complete.

## Question and scope

Can a small model bind noisy lexical input to semantic operations/operands, use exact typed state transitions locally, and lift the result into a learned output task? Clauses and their schedule are supplied; lexical scopes and arithmetic semantics are executed exactly. See [methods](stage5-methods.md), [design](stage5-design.md), [runtime review](stage5-runtime-review.md), and [learned-interface review](stage5-interface-review.md).

## Results to populate from frozen artifacts

- Step-zero and training curves, with warm-start maintenance separated from cold acquisition.
- Three-seed depth × distractor-object-count matrices for result, complete semantic trajectory and output accuracy; actual runtime node counts alongside nominal counts.
- Binding/primitive accuracy and execution conditional on complete correct lowering.
- Learned lifting conditional on correct runtime output and independently supplied correct results.
- Matched neural, graph-as-data, soft-structure and protected learned-transition controls.
- Confidence risk/coverage under noisy, missing-cue and invalid-schema conditions, without crediting abstention as an answered task.
- Permuted correspondence and wrong-world interventions; paired hashes and fairness audit.
- Failure traces partitioned into binding, primitive, schema, execution and lifting errors.
- Runtime/neural resource costs, source/config hashes, raw artifact links and reproducibility commands.

## Interpretation constraints

The exact interpreter supports user-defined function composition; the learned benchmark uses segmented builtin-operation sequences. Runtime name resolution handles lexical shadowing after the network selects a name. Output lifting predicts numeric/sign/comparison classes, not free-form text. No result here alone establishes general program planning, autonomous language induction, spontaneous crystallization or a reason to move to pretrained language models.
