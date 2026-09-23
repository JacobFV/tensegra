# Stage 7: isolated semantic interfaces

**Status: final audits and progressive/scaling diagnostics in progress. Autonomous recomposition remains blocked.**

Stage 7 successfully isolates trainable interfaces, but the frozen studies do not establish a competent composed system. Complete-evidence typed proposal selection passes. Calibrated readiness, return retention, and reliable joint halting/task behavior do not pass across all seeds. Earlier-stage source and artifacts remain unchanged.

## Results and gate decisions

| Interface | Frozen evidence | Decision |
|---|---|---|
| A: complete proposal | 100% full ordered proposals for each of three seeds on IID and moderate-OOD validation/test sets, 1,024 examples per cell | Pass within supplied typed-instruction scope |
| A2: progressive evidence | Separately trained recurrent public-instruction posterior study | Final results pending |
| B: learned-proposal readiness | Test precision 99.45/99.02/99.12%; executable recall 52.31/81.78/38.40% | Fail: seed 2 misses 50% recall; no execution coupling |
| C: return retention | All nine seed/arm runs fail; test joint accuracy after 16 updates is 9.30/10.72/9.38% for once/protected/gated arms | Fail; returned-fact use remains blocked |
| E: halting | Validation exact timing 99.02/100/99.22%; joint timing/task 99.02/92.97/99.22% | Fail all-seed joint gate; seed 1 also has test timing/rejection errors |
| F: semantic graphs | Small-set acquisition fails exact graph reconstruction; renderer-matched cardinality diagnostic in progress | No semantic/runtime integration |

Readiness uses a frozen proposal model and an explicitly supplied public register-type table. Its candidate-specific calibration beats a global scalar on grouped independent examples, but this is not a test of competing interpretations inside one shared world. The separate supplied-score calibration experiment also fails its coverage requirement.

Return retention reveals substantial acquisition error before long recurrence: at one update, unseen test values are only 57.5–58.2% accurate, and joint semantic reconstruction is 13.3–15.1%. Additional recurrence worsens it. Exact protected records survive, but every arm reuses a width-32 encoder that mixes value, type, operation, ordered argument vectors, and provenance. These data do not isolate protection as ineffective; the shared neural encoding/readout is already a bottleneck. Untrained late-use heads are not evidence about learned return consumption. C2 was not trained.

Halting is no longer universally stuck at the minimum. Two seeds learn strong variable-time behavior, and the remaining seed has perfect validation timing but weaker task acquisition. Its heldout test also exposes premature stopping and rejection errors. The prespecified joint gate remains failed; successful seeds do not override the failure.

## What is supplied and what is learned

| Interface | Supplied prior | Learned behavior under evaluation |
|---|---|---|
| Proposal | Typed records, ordered fields, nonce keys, query destination | Relevant instruction selection, primitive and ordered pointers |
| Readiness | Exact public types/schema validity; privileged calibration labels | Correctness confidence for a particular proposed structure |
| Return | Perfect typed event; immutable storage in protected arms | Semantic reconstruction through recurrent workspace under distractors |
| Halting | Certified keyed evidence, deadline, supervised sufficiency time | Evidence use, task decision, and stop/reject timing |
| Semantic graphs | Pinned generator/compiler and privileged graph targets | Typed nodes, identities, ordered edges, and canonical graph recovery |

Protected storage and exact execution are architectural capabilities. Typed instruction retrieval is not language parsing. Synthetic supplied-score calibration is not calibration of learned candidates. Canonical graph reconstruction is a restricted compiler-aligned comparison, not unrestricted semantic equivalence.

## Reports and reproducibility

- [Proposal/readiness report](stage7-ab-notes.md), including raw predictions, frozen thresholds, reliability curves, reverse-order interventions and tiny delta failures.
- [Return-retention report](stage7-c-report.md), including all delay curves, interventions, exact identity denominators and failure examples.
- [Halting report](stage7-halt-notes.md), separating pure timing, joint correctness, oracle timing and exact public-rule bounds.
- [Semantic acquisition/scaling notes](stage7-semantic-notes.md), including failed recipes, loss diagnosis, renderer fixes, actual unique graphs and exposure.
- [Independent review](stage7-review.md), [gate registry](stage7-gates.json), [budget decisions](stage7-budget-notes.md), and [reproduction commands](stage7-runbook.md).

Raw metrics, configs, source snapshots/hashes, checkpoints or durable checkpoint manifests, plots, and deterministic failures accompany each track. Small-set acquisition and frozen main results are kept separate. Validation determines competence; final test reports assess the frozen recipe. No failed interface underwent supervision withdrawal or dependent composition.
