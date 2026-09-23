# Stage 7: isolated semantic interfaces

**Status: experiments in progress. No autonomous recomposition is authorized.**

Stage 7 separates proposal construction, confidence calibration, and symbolic-return retention. Adaptive halting and semantic graph acquisition are independent diagnostics. Earlier-stage source and artifacts remain unchanged.

## What is supplied and what is learned

| Interface | Supplied prior | Learned behavior under evaluation |
|---|---|---|
| Proposal | Typed instruction records, ordered fields, nonce entity keys, query destination | Select the relevant instruction; predict primitive and ordered register pointers |
| Readiness | Exact schema validity; calibration labels during training | Estimate correctness of a particular proposal and select a usable high-precision threshold |
| Return | Perfect typed event and, in protected arms, immutable external storage | Recover event semantics through the recurrent workspace despite distractors |
| Halting | Public certified evidence and deadline; supervised sufficient-evidence time | Identify evidence, solve the task, and stop or reject at an appropriate time |
| Semantic graphs | Pinned procedural generator/compiler and privileged graph targets | Recover typed nodes, identities, ordered edges, and complete semantics from surfaces |

Protected storage and exact execution are architectural capabilities. A learned readout from protected storage is a distinct result. Typed instruction retrieval does not establish natural-language semantic induction. Synthetic supplied-score calibration cannot establish readiness of learned candidates.

## Prespecified decisions

The machine-readable [gate registry](stage7-gates.json), [design](stage7-design.md), [budget decisions](stage7-budget-notes.md), and [independent review](stage7-review.md) define interpretation and blocked dependencies. Validation controls gate decisions; untouched test results assess frozen recipes. Every required seed and condition must be present. Small-set acquisition and empty evaluation sets cannot satisfy competence gates.

Main subsystem reports and audited final gate decisions will be added after the frozen studies complete. Failed gates block their dependent phases; they do not block independent diagnostics.
