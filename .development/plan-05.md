# Stage 5 implementation plan

**Goal:** learn semantic lowering and output lifting around an independently correct protected runtime.
**Architecture:** exact runtime/language; generated observable tasks; neural lowering and matched neural controls; protected execution adapter; isolated training/evaluation/reporting.
**Tech stack:** Python, existing PyTorch and pytest; remote two-thread CPU execution.
**Spec:** [stage5-design.md](stage5-design.md).

## Ownership and implementation tasks

- [x] Runtime worker: `tiny_runtime.py`, `tiny_language.py`, semantic tests and runtime notes. Test integer/record/array access, nested pure calls, distinct invocations, lexical shadowing, aliases and transactional rejection before implementing each behavior. Export `Runtime`, typed primitive schemas, and `execute(source)` with exact result and trace.
- [x] Data/adapter worker: `runtime_tasks.py`, `runtime_execution.py`, tests and notes. Generate six requested families from initial observable state; split surface instructions from gold lowering/result traces; randomize candidate order/names; never precompute returns into inputs. Execute predicted actions on persistent protected state. Test no gold dependence, wrong binding persistence, schema rejection and deterministic paired data.
- [x] Model worker: `runtime_model.py`, tests and notes. Implement small clause transformer, independent operation/binding probabilities with null, neural controls consuming identical observables, protected-register interface and learned output lifting. Test gradients, null/confidence behavior, name/order handling and disabled-structure equivalence where applicable.
- [x] Coordinator/runner worker: `runtime_study.py`, configs/tests. Freeze API after data/model workers agree; supervised, cold task-only and supervised-to-weak curricula, explicit estimator for nondifferentiable execution; step-zero/curves, paired seeds, all failure channels and resource hashes. Pilot before fixed main budget.
- [x] Analysis/review workers: artifact-only analysis, audit, report and standalone plots; exact-runtime oracle sweep; verify conditional denominators, fair inputs, no future leakage, independent source/data hashes and all requested controls.
- [ ] Integrate, run old/new tests, commit raw artifacts/report, push main and clean only the task-owned worktree.

## Review focus

1. Repeated names in nested scopes must not accidentally receive gold scope or candidate masks.
2. Wrong model choices persist into later operations; error metrics cannot teacher-force recovery.
3. Call frames/returns are created at runtime; static input graphs contain no evaluated result.
4. Task losses through discrete execution need an explicit estimator; beta zero is not silently no learning.
5. Abstention/type rejection are not successful task answers; oracle and learned lifting are measured separately.

User supplied detailed design, execution scope, subagent preference and commit/push authorization; proceed without another approval cycle. Shared-worktree ownership prevents conflicting edits.
