# Stage 6 scientific report

**Status: methods and evidence-coverage draft; main quantitative results pending.** Stage 6 implements a recurrent current-state workspace, candidate-local typed proposals, protected append-only arithmetic execution and learned return injection. A separate TCN experiment measures public surface-to-semantic decoding. These tracks share a cell and graph schema; an integrated language-controlled runtime has not been demonstrated. See [methods](stage6-methods.md) for interfaces, objectives and limitations.

## Evidence inventory

| Artifact | Protocol | Evidential use |
|---|---|---|
| [TCN wiring pilot](results/stage6/pilots/language/README.md) | 3 seeds × 4 arms, 8 updates, 18 train/9 evaluation constructions | Pipeline/resource validation only |
| [Controlled timing configuration](../configs/stage6-timing.json) | 1 seed × 2 arms, 8 updates | Resource/wiring check only |
| [TCN main configuration](../configs/stage6-language.json) | Frozen source `4c36557`; 3 seeds × 4 arms, 256 updates, 240 train/48 evaluation constructions | Running at initial draft; final manifest/grid/analysis pending |
| Controlled main | Final source, arms and budget pending | No main quantitative claim yet |

Pilot artifacts remain labeled pilots. They are not pooled with main experiments. No partial main-run metrics are used for conclusions.

## Main results to fill from completed, audited artifacts

The final language table will report choice accuracy, node/typed-edge precision and recall (macro and pooled), node type accuracy, lexical-hash fidelity and exact canonical graph accuracy for English, Spanish, symbols and controlled entity renaming, at step zero and each declared checkpoint. It will retain per-lesson results, random-choice and privileged majority-position baselines, paired seed contrasts, supports and failures. Spanish exposure differs in the paired-surface arm; no isolated consistency-loss claim follows.

The controlled table will report learned numeric/decision/joint task accuracy, exact runtime result/comparison, reference-order trajectory and transition-set equality, task conditional on exact result, and complete/incomplete halts at each depth. Runtime validity does not establish task correctness. Reference-order trajectory equality penalizes alternate ordering of independent operations; transition-set equality is a separate metric.

Readiness analysis will show fractional-posterior calibration/Brier score, threshold risk/coverage, premature attempts and executed/deferred/rejected/conflict/duplicate populations. Risk here is local posterior unready mass, not downstream task-error risk. Returned-value use will compare the same frozen checkpoint under drop/shuffle/wrong-value interventions and report encoder versus post-recurrent workspace reconstruction with their support counts. If no correct executions occur, a null conditional is not a failed zero-percent conditional estimate.

Compute analysis will retain microstep distributions, forced caps, expected training compute, wall time and observation protocol. Complete-evidence fixed/recurrent contrasts will be separated from progressive-evidence runtime contrasts. Auxiliary curves will show the actual zero-support training interval and cold-versus-warm task-only behavior. Final artifact links, source/config/data/checkpoint digests and missing-cell audits remain pending.

## Requested capability and metric coverage

“Measured, pending” below means implemented instrumentation whose completed main-study values have not yet been entered; it is not a successful capability claim.

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
| Actual TCN surface-to-graph/choice performance | Measured, pending | Canonical decoder; no graph isomorphism or lexical string generation |
| TCN multilingual/renaming generalization | Measured, pending | Spanish exposure is arm-dependent; reused bounded renamed vocabulary |
| TCN learned graph driving protected runtime | Unsupported | Separate tracks; no end-to-end language-controlled execution |
| Multi-token autoregressive language output | Unsupported | One bounded answer/choice; token clock is an architectural interface |
| Autonomous cognition or unrestricted reasoning | Unsupported | Neither architecture nor these bounded benchmarks establishes such claims |

## Interpretation and outstanding evidence

Final conclusions must distinguish what was constructed, what survived architectural tests, what was learned under teacher forcing, what the free policy executed, and what the workspace correctly emitted. Exact protected semantics do not transfer exactness to learned readout. Failure to acquire useful autonomous proposals, generalize, preserve returned values or stop appropriately will be reported as a result.

Remaining work is to archive complete main manifests and rows, run artifact-only audits against each frozen source, fill the quantitative tables and plots, attach representative failures, and record final regression/preservation evidence. Existing Stage 1–5 behavior and artifacts are outside this study's modification scope. No positive scientific conclusion is claimed in this draft.
