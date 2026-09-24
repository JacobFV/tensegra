# Protocol v1 independent static review

Snapshot `55ceb550`; source, seven tests, protocol documentation and actual package import path inspected. No new solver/test workload run by reviewer; root owns the real-environment pytest invocation.

## Correctness boundary

The bounded Dijkstra implementation uses nonnegative exact integer edges and stale-distance checks; exhaustive subset/CSP enumeration has a distinct budget-exhausted return. The positive-payload verifier independently recomputes arithmetic, assignment compatibility, selected totals and path costs. Inspection found no algorithmic error in these bounded contracts. This is static review, not exhaustive algorithm verification.

`validate_result` intentionally does **not** prove shortestness, optimality, infeasibility, or semantic reduction correctness. A longer valid path can validate, as can a feasible suboptimal subset. The executor's `optimal` and `exhaustive` tuple entries are claims from trusted code, not proof objects. Reports should call them executor guarantee metadata. A returned infeasible result needs an independent exhaustive reference in mechanical audits; timeout must remain unknown feasibility unless a separately validated incumbent exists.

Constrained subset means one item per category **present in supplied arguments**. Missing a required world category can yield an optimal valid result for the wrong instance. This is a decisive intended reduction-correctness test, not a primitive bug. World validation must verify categories, every required conflict, units/capacity/objective direction, public state versions and mapping back to actual items; it must not repair omitted inputs and credit the proposal.

Work units have heterogeneous definitions. Subset combinations do not count each summed item or conflict scan; intersection constructs a set outside ticks; validation and serialization consume time. A strategy can trade these costs. Keep deterministic work as a public episode currency, actual process CPU as experimental expenditure, and report their relationship rather than identifying them.

## Actual import and isolation risk

Importing `topoformer.campaign02_protocol` executes `topoformer.__init__`, which imports `attention` and therefore torch. Spawn repeats these imports **before** `_child` sets its 2GiB address-space limit. Consequently import memory is not bounded by that limit, while subsequent tiny allocations may fail if existing virtual memory already exceeds it. The earlier package-stub test does not cover this case. Require actual-environment `test_isolated` and a small timeout case before claims of usable process isolation. Source should either use a deliberately lightweight child entry point or choose and disclose a measured compatible memory policy; do not silently remove all resource limits.

The parent waits wall-budget plus0.25seconds for spawn/import/execution. This is an operational API timing rule, not pure solver runtime. On a cold heavy import tiny logical computations can time out. Account startup and parent serialization separately from measured solver work; report whether main environments use trusted in-process or isolated mode. Trusted mode is not hostile-code isolation, though the actor only supplies size-bounded data.

`execute_isolated` terminates and joins one fixed child. The child runs trusted bounded code and does not spawn more processes, so that limited assumption is reasonable. It is not a sandbox for arbitrary executable tools. The outer job wrapper's subreaper handles campaign accounting. Unknown killed-child CPU must be recovered there, never charged as zero just because `Result.cpu_seconds` is0 with its explicit unknown flag.

## Additional defensive tests recommended

- Positive verifier accepts a feasible nonoptimal result: preserve this as a deliberate limitation test, preventing later metrics from treating `True` as optimality.
- Dropped required category gives a protocol-valid optimum but world-invalid reduction.
- Exact status/provenance behavior on a timeout with a feasible incumbent versus no payload.
- Fresh process actual imports under memory cap; invalid inputs cannot trigger unbounded allocation before safe parsing.
- Real child resource receipts reconcile once at the outer root; do not add per-return CPU to the same root receipt again.
- `validate_result` currently assumes a Result-like object: `None` raises AttributeError rather than False. This is not an actor escape under data-only controlled returns, but a cheap strict Result/API-version check would make tampering diagnostics safer. Exclude false-version results when validating provenance.

No new dependency or runtime primitive is recommended. Most remaining risk is at the world/compiler boundary and accounting interpretation, not the small exact algorithms themselves.
