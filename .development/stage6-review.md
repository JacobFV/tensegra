# Stage 6 independent implementation review

Status: approved for a bounded wiring pilot after coordinator verification of the reported remote tests; this is not approval of full-study scientific completeness. Review covers the new Stage 6 implementation and does not launch runtime or training jobs. The coordinator must attach remote test evidence before final approval.

## Reviewed boundaries

The protected session accepts only public scalar registers and predicted typed candidates. Its source checks container shapes before mutation, reads a snapshot for each batch, validates numeric types and bounds independently of gold, and stages append-only commits. Same-identity incompatible proposals conflict symmetrically; identical retries are idempotent. Candidate readiness is checked independently. Invalid operations do not mutate memory; unrelated valid operations may commit. Initial result-namespace collisions cannot overwrite literals. These properties are covered by the committed focused tests; this reviewer has inspected but not executed them.

The semantic compiler uses canonical paths and ordered edges, deduplicates identifier entities within one construction, and records a construction-wide scope and explicit bind edges. It is a structural compiler, not a lexical-scope inference or arbitrary rewrite engine. The TCN adapter generates a single construction before rendering its surfaces; language variants therefore share semantics instead of relying on seed equivalence across different generators.

The current workspace model uses four distinct blocks shared across recurrent microsteps, independent candidate readiness heads, overlapping routing, and additive event injection. The output projection consumes workspace only. Driver enforcement of a post-return update and fully autonomous evaluation is pending runner completion.

## Open findings before freeze

1. The in-progress runner initially exposed exact result contents as ordinary memory features. This creates a parallel channel around event-drop/wrong-value interventions. The implementer has been asked to remove result contents from this actor channel or intervene every result-content channel and explicitly describe the causal estimand.
2. Initial auxiliary selection truncates flattened hypotheses to the first two candidates. This can supervise arbitrary alternatives of a shuffled clause and omit final comparison supervision. Selection must be designed and tested against actual availability and progressive evidence, with privileged selection restricted to training.
3. Independent token hashing without token positions makes cross-attention permutation invariant. Reversing comparison operands preserves the token multiset but flips the correct answer. Ordered/role-sensitive public features and a paired-context actor-input test are required. The same issue affects ordered expression operands.
4. Runner/model interfaces, hidden-field/future canaries, mandatory post-event recurrence, teacher-forced/free-running separation, and source-preservation evidence await completed implementation and remote verification.

## Interpretation limits

Controlled tasks supply typed literal memory, public result slots and identities, three evidence arrivals, and two expression lanes. They use bounded integers and a binary downstream comparison. Performance is not unrestricted language grounding, arbitrary graph rewriting, lexical scope discovery, or general autoregressive language modeling. Actual TCN surface-to-semantic measurements must be reported independently from these supplied-interface execution tasks. General graph rewrites, mutation, undo and recursion remain unsupported.

At preliminary review, tracked changes since the Stage 5 completion commit `1082074` are new Stage 6 files only; existing Stage 1–5 source, configs, tests and results remain unchanged. Untracked in-progress modules must also be included in the final preservation audit.

## Follow-up while runner is under construction

The runner now withholds nonliteral register contents from actor memory and includes token-position features, closing the two initial bypass/operand-order source findings. Data now supplies final comparison readiness when context is visible and uses a symmetric real/decoy operand distribution. Privileged trace-selected training targets replace arbitrary flattened hypothesis truncation.

Further issues sent directly to the runner implementer: task-only updates still include emit supervision; zero auxiliary weights under anneal-all retain teacher forcing; the emit target originally marks root availability rather than downstream comparison completion; event-drop initially bypasses the post-return-step guard because only injections update its clock; a fixed 12-step budget makes exact depth-16/32 trajectories infeasible. Hidden/future mutation canaries, actual TCN learned evaluation, complete intervention controls and telemetry remain to be checked. These are not an approval to freeze experiments.


## Pilot re-review

Latest source closes the initial blockers: result payloads enter only through typed events; public token features preserve order; training targets select trace candidates explicitly; task-only/all-zero auxiliary updates disable teacher forcing and emit supervision; emit targets require the final comparison; post-return recurrence tracks actual returns even when messages are dropped; the common evaluation cap is 80 steps. Event roundtrip heads now contribute to transition-supervised training and emit metrics. Readiness scoring compares each predicted proposal with the current evidence posterior after inference. Free evaluation receives only the public record.

The runner implementer reports seven focused remote tests passing. Inspected tests include all-hidden-field mutation invariance of free actor traces/workspace, future-frame prefix invariance, context-order feature sensitivity, exact-result memory isolation, and finite separated losses. The compiler implementer reports four wheel-installed remote tests passing, including resource availability. This reviewer did not launch any runtime or training job; coordinator verification remains the release gate.

Remaining scientific limitations to resolve or report before final study claims:

- Linear auxiliary decay reaches zero at `step == config.steps`, but training ends before that update; the current schedule establishes decay toward zero, not a measured zero-support training phase.
- Separately trained event variants do not isolate causal use by one frozen policy. Same-checkpoint interventions are needed; holding exact execution fixed is a separate diagnostic from allowing actions to respond to corrupted events.
- `public_graph` rejects these controlled episodes because they do not contain a supplied observable graph. It is not yet a measured graph-as-data baseline.
- The TCN compiler/renderer is measured as data infrastructure, not yet as learned surface-to-semantic generalization. The separate learned TCN benchmark remains outstanding.
- `trajectory_exact` currently requires the exact flattened reference action order, so swapping independent same-step actions is counted as a trajectory mismatch. Interpret it as exact reference scheduling, not general semantic trace equivalence.

The final tracked preservation check against `1082074` contains only new Stage 6 modules/docs/tests/vendor files, plus additive `pyproject.toml` package-data declarations for the vendor license, manifest and grammar JSON resources. Existing Stage 1–5 runtime/model/data sources, configs, tests and results are unchanged. In-progress runner files still need inclusion in the coordinator's final diff/test audit.
