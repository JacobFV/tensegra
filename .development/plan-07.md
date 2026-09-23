# Stage 7 implementation plan

**Goal:** acquire and evaluate proposal, readiness and return interfaces independently; halt and semantic-scaling diagnostics remain separate.
**Spec:** [stage7-design.md](stage7-design.md); user full Stage7 request is authoritative.
**Method:** authorized subagents with isolated file ownership, tests before implementation, independent review, remote bounded CPU runs.

- [ ] TrackA/B + elementary deltas: `interface_proposals.py`, `interface_readiness.py`, `interface_study.py`, tests/configs; complete-evidence benchmark, factorized candidate confidence and independent delta prediction. Gates block A2 and coupled execution.
- [ ] TrackC: `retention_data.py`, `retention.py`, `retention_study.py`, tests/configs; typed event retention, explicit register controls, late-use diagnostics, frozen interventions and gate decisions.
- [ ] TrackD: `halting_study.py`, tests/configs; separate evidence-timing supervision without runtime, stable public/target separation and no-solution rejection.
- [ ] TrackE: `semantic_scaling.py`, tests/configs; pinned TCN/compiler reuse, decomposed graph metrics and resource-bounded dataset/exposure sweep.
- [ ] Review every interface for leakage, shortcuts, parameter/data matching, exact-vs-learned behavior and gate enforcement; run full baseline/regression remotely.
- [ ] Freeze and run independent paired studies after acquisition/timing probes; no composition or supervision withdrawal before competence. Archive raw metrics/configs/provenance/failures and per-track gate reports.
- [ ] Independent artifact audits, concise combined report, preserve Stage1–6, commit/push/integrate and clean owned worktree.

Root owns overall scope/budgets/integration. Workers own only their new modules/tests/configs/track notes and must coordinate shared contracts. Earlier stages are not edited to simplify this work. Progress/gate results live under `.development`.
