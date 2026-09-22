# Stage 5: learned lowering into a protected executable runtime

**Experiment status: the frozen 30-run main sweep is in progress; results and the separately declared nine-run stronger-control supplement are pending.** This draft states the reporting framework, not empirical conclusions.

## What is being tested

The question is whether learned semantic lowering can drive an exact typed runtime, then return its result to a learned output task. The independent language supports immutable bindings, records, arrays, pure definitions/calls, lexical frames and returns. The learned benchmark is narrower: externally segmented and ordered clauses select seven operations and observable semantic operands. Lexical name resolution, scope traversal and arithmetic semantics remain exact supplied algorithms.

The model sees synthetic operation words, noisy continuous lexical reference keys and a shuffled full initial runtime world. It does not learn to parse free prose, construct an execution schedule or plan calls to arbitrary user-defined functions. Output lifting predicts numeric-report, comparison and sign classes, optionally rendered with fixed phrases. It is not learned natural-language generation.

See [methods](stage5-methods.md), [design](stage5-design.md), [runtime review](stage5-runtime-review.md), and [learned-interface review](stage5-interface-review.md).

## Protocol and controls

Main source `2e95c9f` uses three paired seeds, ten variants, 400 updates and 64 examples per condition. Training depths are 1–4 post-resolution operations; evaluation depths are 4/8/16/32. The count axis is 8/32/64 additional distractor bindings, **not total graph nodes**. Every program has an initial resolve clause, making total schedule length D+1. Actual runtime-node ranges will accompany final tables.

Main controls share parameters, public information and data schedules. Ordinary neural prediction observes directed typed edge-record tokens; graph-data and soft-structure controls add respective structure pathways. Neural controls receive lowering supervision and direct numeric-result labels. A protected learned register tests architectural separation without exact transitions. Privileged oracle lowering bounds the execution/output interface. Cold supervised, cold answer-only, warm weak, warm task-only and warm frozen variants separate acquisition from maintenance.

The separately disclosed supplement addresses a remaining retrieval-prior confound: it feeds predicted selector/op distributions directly to graph-data and protected learned controllers, and includes an oracle-selector protected learned control. This was specified after partial first-seed main results became visible, uses the same budget and adds no exact semantics. It will be reported separately, with pairing and unchanged-default verification.

## Results awaiting complete artifacts

The final report will contain three-seed means and sample standard deviations, paired contrasts and denominators for:

| Boundary | Primary measure | Necessary distinction |
|---|---|---|
| Semantic lowering | Selector and primitive accuracy; all decisions correct | Selector equivalence is not unique entity/scope disambiguation |
| Exact computation | Runtime result and complete semantic trajectory | Neural result classification is a different quantity |
| Conditional execution | Exact result given all lowering decisions correct | Null denominators remain undefined |
| Output lifting | End-task output; correct-result lifting audit | Correct sign/comparison can conceal a wrong scalar |
| Generalization | Depth × distractor-count matrices | Depth also changes actual graph size |
| Confidence | Actual invoked errors, risk and coverage | Deferral is unanswered, not correct task execution |

The core table will compare shallow and joint-OOD anchors. Separate heatmaps will show output, exact-result and trajectory performance, rather than treating one as a proxy for another. Family/style breakdowns and majority-class frequencies will contextualize aggregate output accuracy. The external interpreter audit of a neural control's auxiliary lowering will never be called that neural controller's own symbolic trajectory.

## Acquisition, maintenance and reward interpretation

Step-zero curves distinguish learned behavior from supplied priors. Warm curricula are identical through update 200, after which weak supervision, answer reward and frozen lowering diverge. Changes in selector, operation, execution and lifting accuracy through update 400 will be inspected separately.

The policy reward is the final lifted output label. Because comparison/sign outputs are many-to-one functions of numeric results, a semantically wrong program can earn reward. A negative answer-only result would reflect this identification problem together with sparse credit and optimization; it would not establish that task-only semantic binding is impossible. The frozen warm control isolates whether continuing policy updates help or damage an already trained interface.

## Confidence, uncertainty and invalid inputs

The implemented gate is local selective invocation of exact primitives, with type/schema validation. It does not add new evidence or another neural inference pass after deferral. Threshold curves under medium/high noise will report actual reachable invocations separately from hypothetical clause gates. Missing-reference and invalid-schema examples have undefined answers; their appropriate rejection/deferral rates are reported without placeholder-label task accuracy.

No null examples occur during training. Unbound rejection is therefore an extrapolation stress test, not a trained skill. Confidence is based on entropy/margin and semantic groups, not a calibrated probability. No OOD-selected operating threshold will be used to manufacture a favorable task score.

## Failure analysis, resources and reproducibility

Final failure analysis will distinguish selector errors, primitive errors, schema rejection, incorrect execution after correct lowering, and lifting errors after correct execution. Recovery means the actual wrong state later matches the correct semantic state, not a fresh gold restart. Failure-case examples are illustrative; aggregate raw counts supply frequencies.

The report will link compressed raw records, resolved configs, source hashes, analysis/audit outputs, plots and representative traces. Timing will separate actual symbolic copying/execution from neural-plus-diagnostic overhead; it is not a hardware-independent speed claim. Source/config/initialization/schedule/evaluation hashes will establish reproducibility and paired comparisons.

The success criterion remains a learned lowering/exact execution/learned lifting round trip with meaningful OOD structural advantage. Interpreter correctness alone is insufficient, and exact computation with weak output lifting is partial success. No result here would establish spontaneous crystallization, general language induction or broad superiority to graph/message-passing architectures.
