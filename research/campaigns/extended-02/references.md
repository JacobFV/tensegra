# Public-information reference policies v1

`ReferencePolicy.choose(observation, catalog)` receives the same current public
observation and syntactic action catalogue as learned actors. It never receives
a world, generation seed, planted solution, evaluator, or gold action mask.
`choose_index` requires exact catalogue matching; no nearest-action correction.

Three supplied policies establish solvability, not learned orchestration:

- `cheap`: inspect inventory, choose a local greedy item per category, follow
  locally cheapest forward edges in the visible DAG, verify or abstain. No combination search/backtracking.
- `always_tool`: inspect, explicitly add capacity/funds/incompatibility
  constraints, call optimization, retrieve its addressed result, commit it,
  inspect map, build/call route, retrieve a distinct result, deliver and verify.
- `cheap_first`: use greedy when available; otherwise construct the optimization
  problem, escalate catalogue budgets after timeout, and consider routing when public travel-cost savings could outweigh call overhead.

The reference chooses result records by public primitive/problem provenance and
state version, not array position alone. `retrieve` precedes protected exact
`use_return`; this is supplied addressing logic plus exact access, not learned
scalar reconstruction. All generated subset values are one: the environmental
objective is feasibility; optimality is not needed for verified completion.

All combinatorial work occurs through charged primitive calls. Greedy scans,
reduction choices, and policy selection have measured controller process CPU.
Deterministic work units are not asserted equal CPU costs. `run_episode` records
controller CPU, complete episode CPU/wall, simulator solver CPU/work, actions and
feedback. Generation/import costs must be separately charged by the runner.

First proposed experiment (requires coordinator scheduling): 32 paired seeds per
registered difficulty, all three policies, ≤120 CPU seconds, no GPU. Report raw
verified success, cost/work/latency, constraints added, returned records addressed,
replanning and unavailable/timeout outcomes. Register exact difficulty parameters,
seed partitions, and data/source hashes before launch. The small unit fixtures
only test API mechanics and are not empirical evidence of a leverage region.

Known scope: these heuristics inspect all items; they are not optimal observation
policies. Largest-budget means largest currently offered catalogue budget. Calls
restart after timeout; there is no implicit resumed solver state. Validated timeout incumbents are consumed for the feasibility task; their
certificate must pass the executor validator. They never establish optimality. No empirical leverage claim is made before the paired
reference experiment.

## Travel-cost update

Cheap routing is a genuine zero-solver-work multihop policy. It locally selects
the least-cost outgoing forward edge, breaking ties toward greater progress.
The cheap-first policy scans the resulting greedy path and compares a bound on
possible travel savings against four action prices plus one estimated edge scan.
This is a supplied heuristic, not an optimal metareasoning policy. The scan's
actual controller CPU remains recorded. Public travel budgets constrain it.

Largest-budget calls pay actual expanded work, not allocation size: always-tool
is not automatically wasteful merely because it requests1024. Its real overhead
comes from unnecessary solver execution, explicit building/retrieval actions,
and potentially seeking optimality when a cheap feasible choice already works.
Small-budget escalation restarts searches; it can therefore cost MORE than one
large call. These are measured comparisons, not assumed advantages.

`research/tools/campaign02_reference_profile.py` consumes an explicit JSON
configuration and writes lossless compressed per-episode simulator histories,
paired world hashes, raw costs/CPU, cell summaries, and source/config hashes.
The coordinator freezes conditions and runs it; no profile has been launched
by this worker. Generation and serialization are included in self-plus-child
CPU accounting. No GPU is used.

## Persistent execution and feasibility update

Profiler configuration explicitly selects `executor: isolated` (historical
per-call process) or `persistent` (one owned data-only `BoundedSolver` context).
The persistent context closes before final CPU accounting. Child startup,
transport/solve wall time and restart counts are recorded. Live child CPU is
read from Linux process accounting for the between-episode budget check; final
self-plus-reaped-child CPU remains authoritative. The runner never reuses solver
solutions or world state between episodes.

Result addresses use a separately configured `address_seed_start` stream, varied
per paired episode and identical across comparison modes. Handles do not encode
semantic difficulty. `model_compute_tariff` defaults to zero for reference
heuristics; a nonzero fixed harness tariff is charged before each decision and
reported separately from measured CPU. It is not inferred from neural width.

Timeout returns are usable only when their public certificate-valid flag is
true; exact-access remains required. This respects feasibility versus optimality
and avoids wasting restarts on an already verified feasible incumbent.
