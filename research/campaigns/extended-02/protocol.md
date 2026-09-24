# Computation protocol v1

Calls contain primitive/version, immutable ordered public arguments, source-state
version, caller identity, deterministic work cap and wall cap. Results preserve
that provenance, status, immutable payload, observed CPU time, work and executor
certificate. No solver reads a world or knows the requested environmental goal.
The environment constructs explicit instances and independently scores reduction
correctness; solver feasibility alone never establishes semantic correctness.

`Budget(work_units=10000, wall_seconds=2)` and
`Call('sub', (2, 7), source_version=0, caller='candidate-1', budget=budget)` feed
`execute(call)` for trusted bounded simulator inputs, or `execute_isolated(call)`
for data-only process isolation. Descriptor registry documents input ordering.
Public inputs must use exact integers (bool is rejected), strings and tuples;
integers, nesting, total elements, graph dimensions and combinatorial arity are
capped. The API is not an arbitrary host-code sandbox.

## Contracts

- Lookup, threshold filter, intersection, add/sub/mul and three-way comparison
  return exact results. Source handles are preserved by the caller/world; they
  do not enter numerical semantics.
- Shortest path: `(n, edges[(u,v,nonnegative_weight)], start, goal)` returns
  `(path, distance)`. Directed Dijkstra succeeds with an optimal distance;
  exhausted frontier establishes unreachable. Duplicate edges use minimum cost.
- CSP: `(domains, forbidden[(variable_i,value_i,variable_j,value_j)])` returns
  an assignment satisfying every supplied binary incompatibility. Enumeration
  exhaustion is infeasible; reaching a work/wall limit is timeout.
- Subset: `(weights, values, capacity)` returns `(indices,value,weight)`.
- Constrained subset: `(items[(category,weight,cost,value)], capacity,max_cost,
  forbidden_pairs)` returns `(indices,value,weight,cost)`. Select exactly one per
  distinct category appearing in the supplied instance, maximize value. Minimum
  cost is represented explicitly by `value=-cost`. Missing categories or omitted
  constraints are not repaired. The environmental validator must reject a
  reduction which omits a required category. Exhaustive enumeration establishes
  optimality/infeasibility; a timeout may retain a feasible incumbent, never a
  certified optimum. Empty instance is the vacuous feasible selection.

Statuses are success, infeasible, invalid, unavailable, timeout, unknown.
Current deterministic executors use all except unknown, retained for future
solver uncertainty. No resumable state is currently promised. Certificates are
executor claims, not externally verifiable proofs of infeasibility/optimality.
`validate_result` independently checks positive feasibility, arithmetic and
provenance. An external exhaustive reference is needed to audit optimality or
negative claims; a valid path validator does not prove shortest distance.

## Resources and limitations

Work units count scans, heap expansions/edge scans, complete CSP assignments and
constraint checks, or subset combinations. Units differ by algorithm; do not
interpret them as equal CPU operations. Bounded tuple processing, sums and
validation add real CPU work outside expansion counts. Time uses monotonic wall
and process CPU. In-process timing excludes orchestration. Isolated calls spawn
fixed executor code with OS CPU and 2 GiB address-space limits; parent allows the
public wall budget plus 0.25 s launch grace, then kills the child. Successful
isolated CPU reporting includes child startup/import CPU. A killed child's CPU
is explicitly unknown and work is an upper bound, not measured zero consumption;
the coordinator must account child CPU externally (or conservatively charge the
OS cap). Parent launch/serialization/validation CPU also belongs in campaign cost.
No uncontrolled children, network, generated eval or filesystem API is exposed.
Spawn startup can dominate tiny calls; trusted simulator mode avoids that cost.

## Initial mechanical verification

Seven dependency-free test functions passed, including 64 independently
exhaustive subset cases, positive-payload tampering, provenance mismatch,
negative weights, timeout versus infeasible, and one isolated spawn. Parent CPU
0.023 s (child startup separate); wall under one second. Local Python lacks
pytest/torch, so this initial check loaded the module using a package stub to
avoid unrelated package imports. Standard pytest remains required in the
configured project environment. No GPU work or new dependency installation.
