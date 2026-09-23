# Stage 6 scientific report

**Status: TCN main complete; controlled main and final regression evidence pending.** Stage 6 implements a recurrent current-state workspace, candidate-local typed proposals, protected append-only arithmetic execution and learned return injection. A separate TCN experiment measures public surface-to-semantic decoding. These tracks share a cell and graph schema; an integrated language-controlled runtime has not been demonstrated. See [methods](stage6-methods.md) for interfaces, objectives and limitations.

## Evidence inventory

| Artifact | Protocol | Evidential use |
|---|---|---|
| [TCN wiring pilot](results/stage6/pilots/language/README.md) | 3 seeds × 4 arms, 8 updates, 18 train/9 evaluation constructions | Pipeline/resource validation only |
| [Controlled timing configuration](../configs/stage6-timing.json) | 1 seed × 2 arms, 8 updates | Resource/wiring check only |
| [TCN main configuration](../configs/stage6-language.json) | Frozen source `4c36557`; 3 seeds × 4 arms, 256 updates, 240 train/48 evaluation constructions | Completed; [audited summary](results/stage6/language/summary.json) |
| [Controlled main configuration](../configs/stage6.json) | Config `462a5e3`; 15 arms × 3 seed shards, 256 updates, batch 2, depth-4 training; evaluation at 4/8/16/32 | Final source freeze and completed artifacts pending |

Pilot artifacts remain labeled pilots. They are not pooled with main experiments. No partial main-run metrics are used for conclusions.

## Completed TCN main result: no recurrent or semantic advantage at this budget

All 12 runs completed 256 updates. The [audited summary](results/stage6/language/summary.json) contains all 48 checkpoint rows and 192 seed/arm/checkpoint/renderer cells, including renaming. The source, data, configuration, row identity and producer checkpoint digest checks passed against the frozen source; semantic train/evaluation identities were disjoint and initial states matched across arms within seed. Each cell evaluates 48 heldout constructions. Means ± sample SD below are percentages across three seeds, not uncertainty intervals over independent corpora.

| Arm | English | Spanish | Symbols | Renamed English |
|---|---:|---:|---:|---:|
| Single pass | 22.92 ± 0.00 | 22.92 ± 0.00 | 22.92 ± 0.00 | 20.14 ± 5.24 |
| Recurrent | 14.58 ± 0.00 | 14.58 ± 0.00 | 14.58 ± 0.00 | 13.89 ± 2.41 |
| Graph supervised | 14.58 ± 0.00 | 14.58 ± 0.00 | 14.58 ± 0.00 | 24.31 ± 6.01 |
| Paired surfaces | 17.36 ± 4.81 | 17.36 ± 4.81 | 17.36 ± 4.81 | 21.53 ± 9.39 |

Random-choice expectation is 19.44% in each renderer condition. Spanish is withheld from the first three arms but exposed during paired-surface training; symbols is withheld from every arm. Equal final choice rates across the three ordinary renderers do not establish successful cross-language grounding. The recurrent and graph-supervised arms each finish 8.33 percentage points below single pass (paired-seed SD 0); paired surfaces finish 5.56 points below it (SD 4.81). No recurrent or graph-supervised choice advantage is demonstrated within this 256-update budget. This is a bounded negative result, not evidence that a larger budget or another architecture cannot learn the task.

At step zero, English choice accuracy was 20.14 ± 3.18% for single pass and 19.44 ± 6.70% for every recurrent arm. Renamed English changed from 15.97 ± 4.34% to 20.14 ± 5.24% for single pass; recurrent arms started at 21.53 ± 9.39%. Graph supervision's final renamed score of 24.31 ± 6.01% is a small-corpus observation, not evidence of broad lexical generalization. Full checkpoint/per-lesson values and paired contrasts are retained in the summary.

Whole 64-bit lexical-hash fidelity and exact canonical graph accuracy are **0% for every arm, renderer and recorded checkpoint**. Exact canonical graph equality is a strict conjunction over presence, type, lexical hash and typed edges; its failure alone does not quantify every partial skill. Partial graph scores independently reveal poor relational decoding: final supervised typed-edge precision is below 0.86% in every renderer condition. High node presence precision/recall therefore does not establish semantic grounding.

Final per-example macro graph means (%; seed SDs, defined populations and separately pooled count-based rates remain in the summary):

| Supervised arm | Renderer | Node P / R | Typed-edge P / R | Node type accuracy |
|---|---|---:|---:|---:|
| Graph supervised | English | 91.77 / 91.11 | 0.630 / 62.30 | 54.45 |
| Graph supervised | Spanish | 90.70 / 90.63 | 0.650 / 61.33 | 50.73 |
| Graph supervised | Symbols | 91.46 / 90.34 | 0.635 / 61.39 | 52.86 |
| Graph supervised | Renamed English | 90.42 / 91.73 | 0.619 / 62.74 | 53.39 |
| Paired surfaces | English | 95.90 / 79.69 | 0.730 / 49.08 | 51.85 |
| Paired surfaces | Spanish | 95.35 / 81.88 | 0.713 / 50.14 | 51.67 |
| Paired surfaces | Symbols | 96.86 / 78.92 | 0.854 / 51.90 | 50.95 |
| Paired surfaces | Renamed English | 94.68 / 85.67 | 0.719 / 52.42 | 52.00 |

Unsupervised decoder precision is undefined for some seeds that predict no nodes/edges. The summary retains nulls and defined-example counts; it does not replace them with zero or silently use three seeds for those means. Node-type and lexical metrics are scored at gold positions. These output-decoder measurements say nothing about protected execution because this language track has no runtime integration.

## Controlled main results pending

The controlled table will report learned numeric/decision/joint task accuracy, exact runtime result/comparison, reference-order trajectory and transition-set equality, task conditional on exact result, and complete/incomplete halts at each depth. Runtime validity does not establish task correctness. Reference-order trajectory equality penalizes alternate ordering of independent operations; transition-set equality is a separate metric.

Readiness analysis will show fractional-posterior calibration/Brier score, threshold risk/coverage, premature attempts and executed/deferred/rejected/conflict/duplicate populations. Risk here is local posterior unready mass, not downstream task-error risk. Returned-value use will compare the same frozen checkpoint under drop/shuffle/wrong-value interventions and report encoder versus post-recurrent workspace reconstruction with their support counts. If no correct executions occur, a null conditional is not a failed zero-percent conditional estimate.

Compute analysis will retain microstep distributions, forced caps, expected training compute, wall time and observation protocol. Complete-evidence fixed/recurrent contrasts will be separated from progressive-evidence runtime contrasts. Auxiliary curves will show the actual zero-support training interval and cold-versus-warm task-only behavior. Final artifact links, source/config/data/checkpoint digests and missing-cell audits remain pending.

## Requested capability and metric coverage

“Measured, pending” below refers to controlled instrumentation awaiting completed main-study values; it is not a successful capability claim. The TCN results above are complete.

| Capability or metric | Coverage | Boundary |
|---|---|---|
| Four distinct phases reused over microsteps; current state only | Architectural/tests | No latent-history cache; unroll reuses parameters |
| Distributed workspace and overlapping candidate routes | Architectural/tests; overlap measured, pending | Output slots do not define thoughts; overlap alone does not measure interference |
| Role-specific grounding/null and typed structural bias | Architectural/tests; entropy/null measured, pending | Detailed grounding margin/head edge-mass metrics are not established by current artifacts |
| Candidate primitive, ordered operands, destination and readiness | Measured, pending | Learned proposals; supplied memory/IDs and clause segmentation |
| Readiness calibration, risk/coverage and refusal populations | Measured, pending | Local ambiguity labels; unavailable/nonnumeric refusal is not a standalone type-accuracy score |
| Exact primitive arithmetic and formal validation | Architectural/tests; runtime result measured, pending | Fixed scalar subset; task correctness is independently scored |
| Concurrent independent proposals, conflicts and idempotence | Architectural/tests; event populations measured, pending | Same-snapshot append-only transitions |
| General graph rewrite library, arbitrary scope mutation, undo | Unsupported | Canonical graph compiler and scalar transition library are narrower |
| Non-destructive event injection and post-return recurrence | Architectural/tests; roundtrips measured, pending | Learned output need not preserve exact symbolic values |
| Causal return-channel contribution | Frozen interventions measured, pending | Later actions may change; no fixed-future-execution estimand |
| Adaptive halt and task-to-emit gradient | Architectural/tests; halt distributions measured, pending | Hard bounds and soft priors remain supplied; training uses hazard surrogate |
| Discrete primitive acquisition from task reward | Task-only controls measured, pending | REINFORCE surrogate does not imply successful acquisition |
| Independent grounding/topology/transition/readiness annealing | Implemented; main arm coverage pending | Zero support is distinct from merely decayed support; cold/warm separated |
| Structure-off and protected learned-return controls | Pending implementation/freeze | No claims until exact retained guarantees and measured arms are recorded |
| Depth 8/16/32, controlled motif/composition transfer | Instrumented; completed grid pending | Bounded synthetic arithmetic, not arbitrary program generalization |
| Actual TCN surface-to-graph/choice performance | Measured; bounded negative result | Canonical decoder; no graph isomorphism or lexical string generation |
| TCN multilingual/renaming generalization | Measured; no broad generalization established | Spanish exposure is arm-dependent; reused bounded renamed vocabulary |
| TCN learned graph driving protected runtime | Unsupported | Separate tracks; no end-to-end language-controlled execution |
| Multi-token autoregressive language output | Unsupported | One bounded answer/choice; token clock is an architectural interface |
| Autonomous cognition or unrestricted reasoning | Unsupported | Neither architecture nor these bounded benchmarks establishes such claims |

## Interpretation and outstanding evidence

Final conclusions must distinguish what was constructed, what survived architectural tests, what was learned under teacher forcing, what the free policy executed, and what the workspace correctly emitted. Exact protected semantics do not transfer exactness to learned readout. Failure to acquire useful autonomous proposals, generalize, preserve returned values or stop appropriately will be reported as a result.

Remaining work is to archive complete controlled main manifests and rows, run artifact-only audits against its frozen source, fill the controlled quantitative tables and plots, attach representative failures, and record final regression/preservation evidence. Existing Stage 1–5 behavior and artifacts are outside this study's modification scope. The completed language study does not establish the proposed learned capabilities within its budget.
