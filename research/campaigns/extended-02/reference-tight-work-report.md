# Tight-work reference diagnosis and versioned follow-up

This is an archived-trace audit of exploratory development anchors, not new
inference or held-out confirmation. Original recipes and outcomes remain intact.
The compact audit binds the source archive hash and retains representative tails.

| Reference | Success /512 | Failure localization |
|---|---:|---|
| Cheap |424|88 cannot select a feasible greedy subset; no solver calls|
| Always tool |0|512 consume the entire128 work allocation on subset search;490 obtain and commit a valid subset, but cannot execute routing;22 have no subset|
| Cheap first |158|354 abstain after timeouts;298 already have a subset,56 do not|

Of the354 cheap-first failures,330 make one16-work call and retain112 work;
24 make two16-work calls and retain96. Every last solver status is timeout.
The frozen catalogue offers16/128/1024 allocations. After spending16, the next
larger supported allocation128 is unavailable. The reference correctly avoids
restarting the same exhausted deterministic search with the same16-work cap,
but then abstains. This is a discretized action-interface limitation plus a
missing non-solver route fallback, not proof that128 work is intrinsically
insufficient. Conversely, always-tool consumes its work on optimization even
though the environmental criterion requires feasibility and it has an incumbent.

The cheap policy demonstrates that much of the route portion can be completed
without solver work. These traces alone do not establish how many failed
solver-policy episodes a repaired policy will solve; no counterfactual outcome
is credited without running the versioned intervention.

## Prespecified minimal follow-up proposal

Separate two changes in a small2×2 development comparison:

- original versus `cheap_first_fallback_v2` reference;
- original public catalogue versus `include_remaining_budget=true`.

The new reference inherits the existing choices but falls back to the public
locally greedy forward walk when routing cannot make another useful call. It
does not run Dijkstra, inspect hidden state, or change any constraint. The
remaining-budget option is world version`ff16df47`: an explicit public action
choice, not a hidden completion of requested budgets. An optional64 allocation
is another intervention and should not be silently bundled with this comparison.

Run all arms on paired fresh development worlds, including standard and tight
conditions. Retain solver work, controller CPU, verified success, utility,
remaining-budget distributions, and reference failures. If the improvement is
useful, frozen confirmation must use fresh instances. No learned resource-policy
or evolutionary claim follows from improving a supplied reference.

Audit work: two local archive scans used approximately0.545 CPU seconds total;
the saved reconstruction pass records0.272 seconds. Ten mechanical API fixtures
passed in0.004 seconds. No new environment experiment was launched by this worker.

Profiler `shared_condition_seeds=true` now explicitly pairs conditions on the
same base-plus-index generation seed and separate address seed. The default
remains false. This option is profiler metadata, never a WorldSpec field.
For the proposed catalogue-only comparison, identical generation arguments
except public budget metadata preserve the same hidden world. Arbitrarily
changing generator size with the same seed would not establish identical worlds.
Raw world hashes may differ because they include the public catalogue contract;
`instance_key` records the generation seed. Summary paired2×2 outcomes retain
both-correct, each-arm-only and both-wrong counts. Repeated policies/conditions
are not counted as independent support; `unique_seed_instances` is explicit.
