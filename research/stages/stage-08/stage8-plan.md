# Stage 8: protected belief updates and factorized return memory

Authorization: user requested the Stage8 priorities and explicitly set **1024 latent dimensions as the standard**. This stage starts implementation and experiments, not merely planning. Baseline `884b37e` and all Stage1–7 artifacts remain unchanged.

## Independent tracks

1. Progressive candidate beliefs: compare a generic four-phase recurrent workspace, protected additive candidate logits with learned evidence increments, and exact support/update oracle. Inputs are public typed evidence and joint candidates. Track impossible-hypothesis mass, posterior accuracy/calibration, reordering, duplicate evidence, contradiction and explicit retraction. Protected ledger/update semantics are architectural priors, not learned reasoning.
2. Return semantics: separate value/type/primitive/ordered arguments/provenance in typed factorized memory. Compare against a width1024 compressed-event interface, varying persistent availability independently. Report capacity, parameter count and memory token count. Establish zero/one-step reconstruction before long retention and use.
3. Separate semantic graph curriculum/scaling: preserve pinned generator/compiler; learn node/type/identity/edge components with explicit supervision. Sweep available graph cardinality independently from optimizer presentations, same heldout instances and renderer exposure, actual unique counts. No runtime coupling.

Readiness is deferred until progressive belief competence. Halting is preserved, not expanded. No full composition, supervision annealing, new runtime primitive families or pretrained models.

## Execution and validation

Width1024 is the default of every new learned architecture and main configuration; smaller widths are labeled unit-test or historical ablations only. Use isolated remote environments and frozen snapshots. Test tensor contracts, causal/private-target separation, exact oracle semantics, control fairness and zero-evidence behavior before training. Benchmark memory/time at width1024 before freezing paired budgets. GPU runs are serialized until measured safe; CPU tests use at most two threads. No local training.

Use three paired seeds and validation-selected frozen thresholds; heldout test once per frozen recipe. Preserve step0, losses, raw per-frame predictions, configs, source/data/checkpoint hashes, resource counts, failure cases, curves and independent gate audits. Acquisition revisions remain separate from confirmatory main recipes. Prior thresholds remain reference: >98% complete proposals IID, >95% moderateOOD; near-ceiling ordered operands; retention16 type/op>99%, values/identities>98%. Recomposition remains forbidden even after individual passes until explicitly evaluated later.

Owners: belief_state/belief_study, return_memory/return_memory_study, semantic_curriculum each have separate workers; independent reviewer owns stage8-review. Root owns budgets, hardware, integration and overall report.
