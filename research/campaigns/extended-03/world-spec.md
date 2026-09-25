# depworld-v1: a changing dependency graph of computations (build specification)

Status: specification for Stage A. The implementation lives in `src/topoformer/campaign03_depworld.py`, with tests in `tests/test_campaign03_depworld.py`. It reuses the extended-02 typed protocol and solvers (`campaign02_protocol.py`: `constrained_subset`, `csp`, `shortest_path`) unchanged, with `execute` / `BoundedSolver` / `validate_result`.

## Purpose

Test whether a learned controller can **bind, reuse, invalidate and revise** results in a chain of *data-dependent* computations under *changing requirements*. The environment does not script a stage order. The goal is the final deliverable; the agent chooses which computations to run, reuse or revise.

## Entities (hidden spec; public only after inspection)

- **Items.** Each item has `handle`, `category`, `weight`, `price`, `duration` (int 1–4) and `slots` (the allowed start slots, a subset of `range(S)`). Handles are opaque random strings.
- **Requirements.** `capacity`, `funds` and `incompatible` item pairs (as in extended-02); `deadline` D; and `slot_capacity` (the number of items one slot may hold). Items that share a slot conflict.
- **Map.** A directed DAG with integer travel times; start 0, destination L−1, as in extended-02's generator.
- **Planted feasibility.** The generator plants one full solution (selection → assignment → route) that meets the deadline. Planting is never public.

## The dependency chain

1. **Selection:** one item per category under capacity, funds and incompatibility. Solver: `constrained_subset`. Draft built from inspected items plus requirement constraints (as extended-02 `start_subset` + `add_constraint`).
2. **Assignment:** give each **selected** item a start slot from its `slots`. Items in the same slot must respect `slot_capacity`. The public forbidden-pair list comes from the requirements (e.g. pairs that cannot overlap in time: item i at slot s and item j at slot t with `s ≤ t < s + duration_i`). Solver: `csp`, over domains = the selected items' slots and forbidden = the conflicts among selected items. **The CSP instance is a function of the current selection**, so changing the selection changes the instance.
   - Finish time T = max over selected items of (slot + duration).
3. **Route:** a path from the current position to the destination whose travel time is ≤ D − T. Solver: `shortest_path` (the deadline check is the agent's job: the shortest-path result carries its distance). **Route feasibility is a function of the current assignment's finish time.**
4. **Deliver + verify:** the agent commits the selection, commits the assignment, delivers along the route, then verifies. Each commit is validated against the **current** requirements and the **current** committed upstream pieces.

A locally valid choice may make a downstream stage infeasible. For example, a selection with long durations may leave no route that meets the deadline. The agent must then **revise** upstream (uncommit and recompute).

## Change events (public, mid-episode)

With probability p_event per episode (set per condition), a public event fires once, after a public step count sampled from a per-world range. Event kinds:
- `edge_closed(u,v)`: the map changes. Route results become stale; selection/assignment results stay valid.
- `capacity_reduced(new)`: the selection may become invalid. If the committed selection violates the new value it is **revoked** (public), and downstream assignment/route results depending on it become stale.
- `slot_closed(s)`: the assignment is invalid if it uses slot s. It is revoked if committed; selection results stay valid.
- `deadline_moved(new)`: the route may no longer meet it. The assignment may need a revision (earlier finish).

Every event increments a public `requirements_version` and names the affected requirement. The planted solution is regenerated where needed, so every world remains feasible after its event (a generator guarantee, verified by tests).

## Records and applicability metadata (public)

Every computation record carries:
- `primitive`, `status`, `certificate_valid`, `work_units`, `budget`;
- `problem` (the handle of the registered problem it solved);
- `problem_snapshot`;
- `depends_on`: `{"requirements_version", "selection_id", "assignment_id"}`, whichever apply, taken at draft time.

**Registered problems:**
- Drafts get handles `problem_{n}`.
- The world may also pre-register **foreign problems** and records at t = 0:
  - an "earlier shift plan": registered problems of each primitive with solved results for a different requirement set or a different selection;
  - "other-order" problems: registered with a snapshot that differs from any current request.
- Some foreign results are **still applicable**: same snapshot as the current request would produce and current dependencies, so reuse is correct and cheaper. Others are not.
- Foreign records are **registered** problems, so registry membership does **not** separate own from foreign. This is deliberate: it breaks the extended-02 m2 shortcut relationally.

**Applicability** is the evaluator's definition and the reference teacher's rule. A record is applicable to the current request of its primitive iff:
- its type matches, **and**
- its snapshot equals the instance that the current dependencies would produce (request match), **and**
- its `depends_on` versions equal the current ones (dependency match), **and**
- its status is usable and its certificate valid.

The actor sees the raw public metadata. The encoder (d1) must expose request-match and dependency-match *relations* computed from public state. Age and ownership are irrelevant: an old record that is still applicable is reusable.

## Actions (public catalogue; syntactic availability only)

- **Inspection:** `inspect(item|map|requirements)`.
- **Drafts:** `start_subset`, `add_constraint(problem, c)`, `start_assign` (requires a committed selection), `add_constraint(problem, "conflicts")`, `build_route` (requires map and a committed assignment).
- **Solver use:** `call(problem, budget)` (budgets 16/128/1024 + remaining), `retrieve(handle)`, `use_return(handle, as ∈ {select, assign, route})`.
- **Direct paths:** `choose_item`, `commit_pending`, `choose_slot(item, slot)`, `commit_assignment`, `move(dest)`.
- **Revision:** `uncommit(select|assign)` (explicit revision; uncommitting selection also revokes the assignment).
- **Other:** `verify`, `think`, `abstain`.

**Rejections carry reasons:**
- `capacity`, `funds`, `incompatibility`;
- `slot_conflict`, `slot_unavailable`;
- `deadline`, `stale_dependency`, `type_mismatch`, `not_applicable`;
- `missing_edge`.

## Attempted-action record (public typed memory)

The observation includes `attempts`: a list of `{action_kind, action_key, outcome_status, reason, dependency_versions_at_attempt}` for every completion/commit/use_return/move/call. `action_key` is a canonical hash of kind + arguments. The d1 encoder must expose, per candidate:
- this exact action was attempted before, and its last outcome and reason;
- whether relevant dependencies changed since that attempt.

## Costs and utility

Utility = verified success − (action + observation + travel + work + compute costs). `work_price` must make recomputation materially costly, e.g. 2e-4 per work unit, tuned in the Stage A profile, so validated reuse beats always-recompute on cost at equal success.

## References (public observations only)

- `dep_greedy`: direct paths only; never calls solvers.
- `dep_recompute`: always builds fresh drafts and calls solvers for every needed result, escalating budgets; never reuses existing records.
- `dep_reuse`: validate-and-reuse. It uses any applicable record (by the applicability definition above, computed from public metadata); otherwise it computes. It revises upstream when downstream is infeasible, e.g. when no route meets the deadline, it uncommits the assignment and tries an alternative, then the selection.
- `dep_naive_reuse`: uses the most recent same-type record without applicability checks. It is a control and should fail under staleness or foreign records.

## Stage A gate (leverage profile)

On development seeds, across the conditions (sizes, event rates, foreign-record rates):
- `dep_reuse` success ≥ `dep_recompute` success − 0.01, with materially lower cost;
- `dep_naive_reuse` substantially worse where stale or foreign records exist;
- `dep_greedy` materially worse where CSP/deadline coupling bites;
- some worlds require revision: `dep_reuse` success without its revision logic is materially lower.

## Non-goals

No language front end. No new solver algorithms. No hidden-state inputs to actors.
