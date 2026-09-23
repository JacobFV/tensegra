# Stage 7: independently acquired neural–symbolic interfaces

The user's full Stage7 specification supersedes the narrower initial stage7a scope. Existing Stage1–6 files/results remain unchanged. No new runtime primitive family, LLM, agent behavior or larger runtime. Each subsystem trains independently under direct supervision; no failed interface is used to demonstrate another's competence. All experiments run remotely with bounded CPU/memory. Shared implementation worktree is `.worktrees/retention`, branch `feat/interfaces`.

## Tracks and boundaries

A: complete-evidence primitive/destination/ordered operand construction, without execution, readiness, halting or returned values. Existing add/sub/mul/neg/compare primitives only. Frequent noncommutative cases, keyed randomized names/order, unseen names/sets/value ranges/compositions. Component and full ordered metrics; missing/permuted/reversed controls. Strong role-bearing synthetic input priors must be disclosed. Progressive A2 follows only after complete-evidence competence.

B: candidate-specific correctness calibration, independent from proposal failure. Controlled supplied candidate hypotheses and known posterior support permit readiness to be measured on its own. Explicit primitive×ordered-arguments×schema×stability factors; product is a score, not assumed calibrated probability. Train calibration on a dedicated split, freeze thresholds, assess reliability/precision/coverage versus a global gate. Include stable-wrong, multiple-valid and zero-valid cases. Using supplied hypotheses must not be called learned candidate construction. Actual proposal coupling waits for GateA and GateB.

C: perfect typed event retention and late-query use. Initial stage7a design remains a starting point, extended to delays1/2/4/8/16/32, operand identities, wrong type/provenance controls, and protected register/read-gate/persistent-memory/release controls. Reuse the existing four blocks; generic MLPs cannot mutate typed return records. Workspace readout has no direct exact-answer bypass. Retention and downstream use are separately scored and gated. C2 training/composition depends on C competence; simple heldout late-use diagnostics may be reported without claiming a passed composition gate.

D: independently supervised evidence-timing/halting task with known sufficiency, early/late/misleading/no-solution episodes. No symbolic execution or proposal generation. Fixed/minimum/oracle/learned/bias controls; measure task and stop accuracy, early/late compute and nonconstant timing. No-solution must remain a reject outcome, not fabricated certainty.

E: separate TCN semantic scaling, using existing pinned generator/compiler. Start resource-feasible1k/10k/100k construction budgets, escalating only after timing and acquisition checks;1M conditional on resources. Decomposed semantic/ordered-edge/identity metrics, not exact lexical hashes alone. Disjoint semantic constructions and heldout renderer/lexicon, baselines and step0/learning curves. Record actual examples/optimizer exposure independently of dataset cardinality. No runtime coupling.

Tiny graph-delta targets reuse existing arithmetic/lookup/resolve/substitution semantics only, with explicit auditable transition labels and no broad rewriting claims. Begin with arithmetic reduction under supplied operation/operands; score delta prediction independently from proposal and readiness. Do not add a general rewrite engine.

## Prespecified gates

A: before progressive evidence, >99% primitive and each required operand, >98% full ordered proposal on IID and moderate OOD (stricter than later >95% OOD composition floor). Noncommutative ordering near ceiling. Require all paired seeds to pass on validation before running progressive/composed studies; report heldout test separately.
B: heldout high-threshold precision>99%, coverage>=50% of executable candidates (declared operational meaning of usable), full reliability/support counts, local advantage in mixed readiness groups over matched global scalar. Never select threshold on test outcomes.
C: at16 recurrent updates, type/primitive>99%, discrete value and each identity>98%, including unrelated activity. Exact protected-register integrity is architectural and separate from learned workspace reconstruction.
D(return-use): downstream decisions>95% plus substantial causal sensitivity (at least20pp drop when a required return is removed or corrupted, scoped controls). C competence required before training multi-operation composition.
E(halt): >95% correct stop/reject on heldout timing examples, premature rate<1%, correct variable timing across at least3 sufficient-evidence arrival steps and >20pp task advantage over minimum-step stopping. Those numerical operationalizations are declared before outcomes.
F(TCN): positive heldout typed-edge and semantic learning trend across exposure budgets, materially above simple baselines and meaningful heldout-renderer/lexicon transfer. No arbitrary graph-presence-only pass. Quantitative interpretation with counts/curves; composition blocked absent evidence.

## Experiment policy

Test and small fixed-set overfit first, then bounded timing smoke. Freeze paired main budgets before inspecting test outcomes. Log source/config/data/initial/checkpoint hashes, per-loss curves, raw outputs, interventions and gate denominators. Train<=2CPUthreads per process, at most3 simultaneous jobs initially, no local Torch workload. Preserve all old source files; new modules/configs/notes only. Do not spend on huge TCN runs without measured feasibility. If competence fails, record why and stop dependent phases, not unrelated independent tracks.

All confidence factors can be correlated. Stability can support a persistently wrong answer. Report calibrated empirical correctness and schema validity separately. More recurrence, permanent symbolic availability, and direct privileged labels remain supplied priors, not learned reasoning claims.
