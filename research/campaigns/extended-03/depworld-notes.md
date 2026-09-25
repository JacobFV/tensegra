# depworld-v1: Stage A implementation notes

Status: Stage A build, branch `campaign/e03-depworld`. The build contract is [world-spec.md](world-spec.md). This note records what was built, where it deviates from the spec (with reasons), the d1 encoder, the tuning decisions and the leverage profile.

## Files

| File | Contents |
|---|---|
| `src/topoformer/campaign03_depworld.py` | `DepSpec`/`DepItem`, `generate_depworld`, `DepWorkshop`, `DepObservation`, `depworld_executor`, the public applicability rule (`current_request`, `current_dependencies`, `relations`, `applicable`), the evaluator-only audit (`audit_record`, per-call reduction audit), `relevant_dependencies` (attempt memory), `action_catalog`, the d1 encoders, `DepReference` (5 modes) |
| `tests/test_campaign03_depworld.py` | 27 tests (see below) |
| `research/tools/campaign03_leverage_profile.py` | CPU-only leverage profiler (in-process `execute`) |
| `campaign02_training.public_frame` | dispatches `feature_version == "d1"` or `observation.version == "depworld-v1"`; a mismatch raises |
| `campaign02_policy.PolicyConfig` | accepts `"d1"` |
| `campaign02_references.make_reference` | `"dep_<mode>"` prefix branch |
| `campaign02_population` | `world_family="depworld"` in config validation and `main()`. The depworld source hash is added to `source_hashes` only for depworld runs, so historical resume hashes do not change. |
| `research/tools/campaign02_evaluate.py` | `world_family="depworld"` conditions. `episode_counts` now uses `.get` for arguments, counts `stale_dependency` as stale, and adds depworld counts (`reason/*`, `dep_uses`, `dep_invalid_uses`, `dep_foreign_uses`, revisions, revocations, calls) when an outcome has `reuse_audit`. |

`campaign02_protocol.py` is unchanged. No historical results were touched.

## World as built

### Items, requirements and map
- **Items.** Opaque handles, category, weight 1–7, price 1–9, duration 1–4, and allowed start slots (a random subset of `range(S)` of size 2–4).
- **Planted plan.** One item per category, packed back to back from slot 0 in random order. Planted durations are shrunk if needed so every start is below `S`.
- **Requirements.** `capacity`, `funds`, `incompatible` pairs, `deadline`, `slot_capacity = 1` and `closed_slots`. All of them, including the incompatible pairs, stay hidden until `inspect(requirements)`.
- **Map.** The extended-02 generator: a forward chain, a 75% chance of a (0, L−1) shortcut, and extra forward edges with probability 0.25.

### Chain
- **Selection.** `constrained_subset` over the draft's items. The draft's value column is `5 − duration`, a declared public rule ("shorter jobs preferred"), so the exact solver returns a feasible selection with the smallest total duration.
- **Assignment.** `csp` over the committed selection, in category order. The domains are the allowed slots minus the closed ones. `conflicts` forbids every overlapping pair (a single machine), and an optional `finish_by = b` keeps only starts with `s + d ≤ b`. The finish time is `T = max(s + d)`.
- **Route.** `shortest_path` from the current position. Moving, or using a route, is rejected with reason `deadline` if `T + travel + distance > deadline`. That check is atomic, so no partial travel happens.
- **Verify.** Re-checks the selection and assignment against the *current* requirements, then the position and the deadline.

### Events
There is at most one event per world, fixed at generation. It is independent of the agent, and it fires at the end of step k, with k sampled from `(n_items+9, n_items+19)`.
- `edge_closed` closes an edge on the pre-event shortest path, provided the destination stays reachable.
- `capacity_reduced` lowers capacity below the canonical selection's weight when the planted weight allows it.
- `slot_closed` closes a slot used by the canonical assignment that is not a planted start.
- `deadline_moved` moves the deadline earlier, aimed below the canonical plan's arrival when possible.

Every event keeps the planted plan feasible, because the post-event values are built around it and the pre-event values are looser. An event bumps the global `requirements_version` and the affected requirement's version, then publishes `{kind, argument, step, requirements_version, affected, revoked}`. Committed pieces that violate the new requirement are revoked: `capacity_reduced` can revoke the selection and assignment, and `slot_closed` can revoke the assignment.

### Records and foreign problems
- **Record fields.** Each record carries `primitive`, `status`, `certificate_valid`, `work_units`, `budget`, `problem` (a registered handle), `problem_snapshot`, `depends_on` and `created_step`. The payload stays hidden until `retrieve`.
- **Foreign problems.** They are registered as `problem_0…` at t=0 (so drafts continue at `problem_{n}`). Each one has a solved record, and its evaluator-only label never appears in any observation. The eight kinds:

| Kind | Applicable? |
|---|---|
| `subset_current` | Yes |
| `subset_earlier` (different capacity or funds, versions 0) | No |
| `subset_version_mismatch` (same snapshot, versions 0) | No (dependency mismatch only) |
| `route_current` | Yes |
| `route_other_goal` (same map, another goal) | No (request mismatch only; walking it is silent harm) |
| `route_earlier` (older map with a perturbed or extra edge, version 0) | No |
| `csp_current` (the canonical selection's instance) | Yes, once that selection is committed |
| `csp_other_selection` | No |

### Applicability (public rule, the teacher's and evaluator's definition)
`applicable(obs, record, primitive)` holds when all four conditions hold:
1. The type matches.
2. **Request match:** the snapshot equals `current_request(obs, primitive, snapshot_options(snapshot))`, the instance the current public dependencies would produce under the snapshot's own declared options (subset exclusions, csp `finish_by`).
3. **Dependency match:** `depends_on` equals `current_dependencies(obs, primitive)`.
4. The status is usable (success, or timeout with a valid certificate) and the certificate is valid.

`audit_record(env, handle)` recomputes the same relations from the hidden spec and state, and adds `payload_valid_now`, a semantic check of the payload against the current true instance (for routes: optimal from the current position). The environment also logs a per-call reduction audit (`reductions[*].correct/request_match/dependency_match`) and a per-use audit (`reuse_audit`: hidden applicability, foreign, label). All of this is evaluator-only.

### Attempted-action record
Each `commit_pending`, `commit_assignment`, `use_return`, `move`, `call`, `uncommit` and `verify` appends `{action_kind, action_key, arguments, outcome_status, reason, dependency_versions_at_attempt, step}`. `action_key` is a canonical hash of kind plus arguments. `dependency_versions_at_attempt = relevant_dependencies(pre-action observation, action)` is a public function, so encoders can recompute it now and answer "have the dependencies changed since this attempt?"

## Deviations from world-spec.md (and why)

1. **`slot_capacity` is fixed at 1.** A single machine: no two selected jobs may overlap in time, which already implies that two jobs sharing a start slot conflict. The CSP stays binary, so it lowers exactly to the frozen `csp(domains, forbidden)`. A capacity of 2 or more would need ternary constraints. `DepSpec` rejects other values.
2. **`depends_on.requirements_version` is a dict of per-requirement versions** for the requirements the primitive's instance reads: subset reads capacity, funds and incompatible; csp reads slots; route reads map. A single global counter would make `edge_closed` invalidate selection results, which the spec says stay valid. The global `requirements_version` is still public.
3. **Route records depend on the map only**, not on `assignment_id`. The shortest-path instance is map + position + destination, and the deadline is checked when the route is used. After an assignment revision, the old route result therefore stays applicable (old-but-valid reuse). `build_route` still requires a committed assignment, as the spec says.
4. **Selection and assignment ids are content-addressed** (hashes of the chosen handles). Re-committing the same selection makes earlier or foreign csp records for it dependency-matched again.
5. **Two draft options were added** so that revision can request an alternative: subset `exclude(item)` and csp `finish_by(bound)`, with bounds 1…S+3. Request match is evaluated under the snapshot's own options. The d1 encoder also exposes `canonical_match` (no options) separately.
6. **Subset value = 5 − duration** (a declared public rule). Without it the selection solver would be blind to the deadline coupling. The coupling still bites through packing, finish time and route length.
7. **Extra declared rejection reasons:** `category_coverage`, `missing_dependency`, `incomplete`, `already_committed`, `retrieve_required`, `unknown_item`, `travel_budget`, `work_budget`. `REASONS` lists the spec's reasons first.
8. **`commit_assignment` also rejects with `deadline`** if `T + travel > deadline`.
9. **Events are aimed and scheduled deterministically** (see above), and **an event scheduled after delivery is moot.** Without that rule a faster agent that had already delivered lost to a later deadline move, which cost one development episode in the 64-per-cell profile.
10. **Public versions start at 1.** Version 0 is reserved for foreign "earlier shift" records.
11. **Attempt entries also carry `arguments` and `step`,** which the recompute reference and the evaluator use.
12. **`dep_naive_reuse` does not re-apply a record it has already applied in the episode.** The literal "most recent same-type record" loops forever after the first assignment revision (measured: success 0.875 with no foreign records or events, all loop failures). It still performs no applicability check.
13. **`dep_recompute` treats as "fresh"** only records created after the public stage-open step: the last commit or uncommit of the upstream piece, a revocation, or an event touching the stage's inputs.

## d1 encoder (observation 82 dims; candidate 153 dims)
The d1 encoder uses no handle spelling (handles enter only equality relations and hashes), no hidden labels, no gold actions and no solved answers. **There is no single "applicable" bit.** The encoder exposes the component relations instead: type, request, canonical, dependency (split into requirements and selection), and usable. All of them can be derived from the public observation, and the tests check that their conjunction reproduces `applicable()`.

**Observation.** The encoding covers:
- **Requirements:** known flag, capacity, funds, deadline, slot_capacity, number of closed slots, number of incompatible pairs, `requirements_version`, and a changed flag for each of the six requirements.
- **Plan state:** selection and assignment committed, finish time, position, destination, arrived, travel, and deadline slack.
- **Route bound:** whether an applicable retrieved route is known, its distance, whether the current plan meets the deadline, and the needed finish bound.
- **Progress:** inspection progress and map known.
- **Counts:** problems, records, retrieved, pending, pending assignment.
- **Pending selection:** pending weight and price, and the capacity and funds remaining.
- **Events:** last event kind (one-hot, or none), selection revoked, assignment revoked, event fired this step.
- **Resources:** remaining work, steps and travel; prices.
- **Feedback:** status one-hot and reason one-hot over `REASONS`.
- **Attempts:** attempt count, rejected-attempt count, uncommit count.

**Candidate.** A kind one-hot (17), the use or uncommit target (3), the inspect target type (3) and the action's primitive (3), followed by these blocks. Each block is zeros when it does not apply:
- **Record** (retrieve/use_return, 26): present, type match to the use, request match, canonical match, dependency match, requirements match, selection match, usable, status one-hot (5), certificate valid, retrieved, finish_by option and value, number excluded. **Payload facts** (only after `retrieve`):
  - Route: distance, starts at the current position, ends at the destination, meets the deadline with the current finish time.
  - Csp: implied finish time, whether it meets the needed bound, whether its items equal the current selection.
  - Subset: total duration, whether it is valid under the current requirements.
- **Draft** (call/add_constraint, 28): request match and dependency match of the draft, constraints present, finish_by, number excluded, size.
  - Call: budget, whether the budget exceeds the most work used on this snapshot, whether it timed out at this budget before, number of usable current records, whether an infeasible current record exists, whether the budget equals the remaining work.
  - add_constraint: constraint one-hot, already present, bound, whether the bound is at most the needed bound, the excluded item's duration and the category's remaining alternatives.
- **Start** (3): number of drafts of this primitive, drafts matching the current request and dependencies, live applicable records.
- **Item** (inspect/choose_item, 14): known, weight, price, duration, number of slots, pending, selected, category already pending, fits the remaining capacity and funds, conflicts with pending items, duration rank and cost rank within the category.
- **Slot** (choose_slot, 10): duration, slot, allowed, closed, overlaps another pending slot, finish time, whether the finish meets the needed bound, same as pending, item already has a pending slot.
- **Move** (7): edge weight, destination, direct edge from the target to the destination, minimum outgoing edge, meets the deadline, step size.
- **Commit/uncommit** (10): for `commit_pending`: coverage, complete, weight and price OK, incompatibility present. For `commit_assignment`: coverage, allowed, clash, finish time, meets the bound. For `uncommit`: the target is committed.
- **Attempt** (the last `ATTEMPT_BLOCK = 30` entries): attempted before, count, last-outcome one-hot (8), last-reason one-hot (18), dependencies changed since.

It costs about 0.8 ms per step to build the catalogue and encode it. The catalogue averages 113 candidates, with a maximum of 215, over s3/s4 worlds with events and 4 foreign records.

## Tuning decisions (Stage A profile)

- **`deadline_slack = 1`.** With slack 0–1 over the planted arrival, about 10–25% of worlds need a voluntary revision.
- **`work_price = 2e-4`,** the spec's example, kept as the default. The profile reprices at 1e-4, 2e-4 and 5e-4. The pooled reuse cost advantage is −11.2%, −13.0% and −15.0% respectively.
- **`step_limit = 96`, `work_limit = 8192`.** An s4 CSP can use about 1,000 work units per exhaustive infeasibility proof.
- **Event timing `(n_items+9, n_items+19)`.**
  - Extended-02-style early timing (8–24) fired before most commits, so events revoked almost nothing (0–9% of episodes).
  - `(n+6, n+22)` reached about 30% post-commit exposure.
  - `(n+11, n+21)` made fewer events fire before episodes ended.
  - The chosen `(n+9, n+19)` fires in 47–94% of episodes (depending on the reference's speed) and revokes in 4–17%.
- **Event aiming.** Initially events were random: capacity cuts rarely hit the committed selection, and closed slots were rarely used. They now target the canonical public pipeline's likely commitments, which is agent-independent.
- **`slot_choices = (2, 4)`, durations 1–4, reference initial budget 128** (escalating to 1024, then the remaining work).

## Leverage profile (128 development worlds per cell)
Command: `PYTHONPATH=src python research/tools/campaign03_leverage_profile.py --examples 128`. Seeds are `2_000_000_000 + cell*100_000 + i`, and each cell pairs the same worlds across references. Sizes: s3 = 3 categories × 3 choices, 7 locations, 6 slots; s4 = 4 × 3, 8 locations, 7 slots. Utility and cost are at work_price 2e-4. "Invalid uses" counts uses of records that the hidden audit marks not applicable. Wall time was 75 s on CPU.

| size | p_ev | foreign | reference | success | utility | cost | work | calls | revisions | revised eps | event fired | revoked eps | invalid uses | foreign uses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 | 0.0 | 0 | greedy | 0.312 | 0.279 | 0.0338 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 0 | recompute | 0.984 | 0.913 | 0.0717 | 123 | 3.32 | 0.15 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 0 | reuse | 0.984 | 0.914 | 0.0707 | 121 | 3.23 | 0.15 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 0 | naive_reuse | 0.984 | 0.914 | 0.0707 | 121 | 3.23 | 0.15 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 0 | reuse_norevise | 0.883 | 0.815 | 0.0675 | 113 | 3.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 2 | greedy | 0.273 | 0.239 | 0.0341 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 2 | recompute | 0.984 | 0.912 | 0.0720 | 124 | 3.37 | 0.16 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 2 | reuse | 0.984 | 0.921 | 0.0639 | 99 | 2.52 | 0.16 | 0.12 | 0.00 | 0.00 | 0.00 | 0.75 |
| s3 | 0.0 | 2 | naive_reuse | 0.820 | 0.754 | 0.0662 | 104 | 2.66 | 0.76 | 0.29 | 0.00 | 0.00 | 0.88 | 1.59 |
| s3 | 0.0 | 2 | reuse_norevise | 0.875 | 0.816 | 0.0593 | 86 | 2.29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.71 |
| s3 | 0.0 | 4 | greedy | 0.281 | 0.248 | 0.0330 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 4 | recompute | 0.992 | 0.920 | 0.0725 | 129 | 3.34 | 0.12 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 |
| s3 | 0.0 | 4 | reuse | 0.992 | 0.935 | 0.0570 | 73 | 2.05 | 0.12 | 0.09 | 0.00 | 0.00 | 0.00 | 1.16 |
| s3 | 0.0 | 4 | naive_reuse | 0.820 | 0.762 | 0.0582 | 68 | 2.05 | 0.93 | 0.25 | 0.00 | 0.00 | 1.42 | 2.52 |
| s3 | 0.0 | 4 | reuse_norevise | 0.906 | 0.853 | 0.0529 | 61 | 1.87 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.12 |
| s3 | 0.5 | 0 | greedy | 0.367 | 0.334 | 0.0333 | 0 | 0.00 | 0.00 | 0.00 | 0.14 | 0.01 | 0.00 | 0.00 |
| s3 | 0.5 | 0 | recompute | 0.984 | 0.912 | 0.0727 | 126 | 3.48 | 0.13 | 0.10 | 0.44 | 0.04 | 0.00 | 0.00 |
| s3 | 0.5 | 0 | reuse | 0.984 | 0.913 | 0.0711 | 123 | 3.36 | 0.13 | 0.10 | 0.44 | 0.04 | 0.00 | 0.00 |
| s3 | 0.5 | 0 | naive_reuse | 0.984 | 0.913 | 0.0710 | 122 | 3.34 | 0.13 | 0.10 | 0.44 | 0.04 | 0.05 | 0.00 |
| s3 | 0.5 | 0 | reuse_norevise | 0.898 | 0.832 | 0.0666 | 108 | 3.17 | 0.00 | 0.00 | 0.43 | 0.04 | 0.00 | 0.00 |
| s3 | 0.5 | 2 | greedy | 0.273 | 0.240 | 0.0333 | 0 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 0.00 |
| s3 | 0.5 | 2 | recompute | 0.953 | 0.877 | 0.0760 | 140 | 3.58 | 0.22 | 0.13 | 0.34 | 0.03 | 0.00 | 0.00 |
| s3 | 0.5 | 2 | reuse | 0.953 | 0.886 | 0.0668 | 113 | 2.72 | 0.18 | 0.11 | 0.27 | 0.04 | 0.00 | 0.63 |
| s3 | 0.5 | 2 | naive_reuse | 0.852 | 0.785 | 0.0666 | 106 | 2.62 | 0.60 | 0.22 | 0.22 | 0.02 | 0.98 | 1.57 |
| s3 | 0.5 | 2 | reuse_norevise | 0.875 | 0.815 | 0.0603 | 90 | 2.42 | 0.00 | 0.00 | 0.26 | 0.04 | 0.00 | 0.61 |
| s3 | 0.5 | 4 | greedy | 0.266 | 0.231 | 0.0344 | 0 | 0.00 | 0.00 | 0.00 | 0.13 | 0.04 | 0.00 | 0.00 |
| s3 | 0.5 | 4 | recompute | 0.969 | 0.895 | 0.0742 | 133 | 3.52 | 0.15 | 0.10 | 0.35 | 0.04 | 0.00 | 0.00 |
| s3 | 0.5 | 4 | reuse | 0.969 | 0.911 | 0.0581 | 78 | 2.06 | 0.13 | 0.09 | 0.25 | 0.06 | 0.00 | 1.27 |
| s3 | 0.5 | 4 | naive_reuse | 0.750 | 0.688 | 0.0618 | 80 | 2.38 | 1.08 | 0.30 | 0.23 | 0.06 | 1.53 | 2.65 |
| s3 | 0.5 | 4 | reuse_norevise | 0.898 | 0.843 | 0.0555 | 73 | 1.88 | 0.00 | 0.00 | 0.23 | 0.06 | 0.00 | 1.23 |
| s3 | 1.0 | 0 | greedy | 0.391 | 0.355 | 0.0354 | 0 | 0.00 | 0.00 | 0.00 | 0.32 | 0.06 | 0.00 | 0.00 |
| s3 | 1.0 | 0 | recompute | 1.000 | 0.922 | 0.0775 | 138 | 3.84 | 0.14 | 0.12 | 0.83 | 0.17 | 0.00 | 0.00 |
| s3 | 1.0 | 0 | reuse | 1.000 | 0.925 | 0.0750 | 134 | 3.65 | 0.14 | 0.12 | 0.83 | 0.17 | 0.00 | 0.00 |
| s3 | 1.0 | 0 | naive_reuse | 1.000 | 0.925 | 0.0752 | 134 | 3.64 | 0.14 | 0.12 | 0.83 | 0.17 | 0.14 | 0.00 |
| s3 | 1.0 | 0 | reuse_norevise | 0.875 | 0.805 | 0.0703 | 121 | 3.40 | 0.00 | 0.00 | 0.81 | 0.17 | 0.00 | 0.00 |
| s3 | 1.0 | 2 | greedy | 0.242 | 0.209 | 0.0335 | 0 | 0.00 | 0.00 | 0.00 | 0.27 | 0.03 | 0.00 | 0.00 |
| s3 | 1.0 | 2 | recompute | 0.984 | 0.903 | 0.0814 | 159 | 3.91 | 0.20 | 0.16 | 0.80 | 0.12 | 0.00 | 0.00 |
| s3 | 1.0 | 2 | reuse | 0.984 | 0.915 | 0.0691 | 118 | 2.88 | 0.18 | 0.13 | 0.64 | 0.10 | 0.00 | 0.64 |
| s3 | 1.0 | 2 | naive_reuse | 0.852 | 0.787 | 0.0648 | 99 | 2.55 | 0.49 | 0.23 | 0.55 | 0.05 | 1.09 | 1.66 |
| s3 | 1.0 | 2 | reuse_norevise | 0.859 | 0.796 | 0.0630 | 98 | 2.61 | 0.00 | 0.00 | 0.62 | 0.10 | 0.00 | 0.62 |
| s3 | 1.0 | 4 | greedy | 0.305 | 0.271 | 0.0334 | 0 | 0.00 | 0.00 | 0.00 | 0.19 | 0.02 | 0.00 | 0.00 |
| s3 | 1.0 | 4 | recompute | 0.969 | 0.887 | 0.0822 | 161 | 3.88 | 0.20 | 0.14 | 0.84 | 0.06 | 0.00 | 0.00 |
| s3 | 1.0 | 4 | reuse | 0.977 | 0.911 | 0.0653 | 106 | 2.34 | 0.16 | 0.09 | 0.47 | 0.06 | 0.00 | 1.13 |
| s3 | 1.0 | 4 | naive_reuse | 0.828 | 0.762 | 0.0662 | 107 | 2.08 | 0.73 | 0.23 | 0.41 | 0.05 | 1.50 | 2.52 |
| s3 | 1.0 | 4 | reuse_norevise | 0.906 | 0.846 | 0.0605 | 92 | 2.04 | 0.00 | 0.00 | 0.44 | 0.06 | 0.00 | 1.12 |
| s4 | 0.0 | 0 | greedy | 0.148 | 0.111 | 0.0375 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 0 | recompute | 0.992 | 0.795 | 0.1968 | 715 | 4.07 | 0.16 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 0 | reuse | 0.992 | 0.796 | 0.1959 | 714 | 4.03 | 0.16 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 0 | naive_reuse | 0.992 | 0.796 | 0.1959 | 714 | 4.03 | 0.16 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 0 | reuse_norevise | 0.875 | 0.704 | 0.1707 | 599 | 3.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 2 | greedy | 0.133 | 0.096 | 0.0368 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 2 | recompute | 0.977 | 0.763 | 0.2139 | 797 | 4.18 | 0.20 | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 2 | reuse | 0.977 | 0.786 | 0.1905 | 695 | 3.32 | 0.20 | 0.14 | 0.00 | 0.00 | 0.00 | 0.66 |
| s4 | 0.0 | 2 | naive_reuse | 0.875 | 0.689 | 0.1859 | 674 | 3.07 | 0.47 | 0.20 | 0.00 | 0.00 | 0.79 | 1.44 |
| s4 | 0.0 | 2 | reuse_norevise | 0.859 | 0.713 | 0.1460 | 486 | 2.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.65 |
| s4 | 0.0 | 4 | greedy | 0.172 | 0.134 | 0.0379 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 4 | recompute | 0.969 | 0.803 | 0.1661 | 561 | 4.00 | 0.19 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 |
| s4 | 0.0 | 4 | reuse | 0.969 | 0.851 | 0.1179 | 342 | 2.55 | 0.19 | 0.15 | 0.00 | 0.00 | 0.00 | 1.12 |
| s4 | 0.0 | 4 | naive_reuse | 0.781 | 0.640 | 0.1413 | 447 | 2.71 | 0.98 | 0.30 | 0.00 | 0.00 | 1.35 | 2.46 |
| s4 | 0.0 | 4 | reuse_norevise | 0.852 | 0.750 | 0.1020 | 275 | 2.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.10 |
| s4 | 0.5 | 0 | greedy | 0.156 | 0.118 | 0.0379 | 0 | 0.00 | 0.00 | 0.00 | 0.09 | 0.01 | 0.00 | 0.00 |
| s4 | 0.5 | 0 | recompute | 0.945 | 0.751 | 0.1942 | 696 | 4.34 | 0.27 | 0.19 | 0.39 | 0.02 | 0.00 | 0.00 |
| s4 | 0.5 | 0 | reuse | 0.945 | 0.754 | 0.1915 | 690 | 4.23 | 0.27 | 0.19 | 0.39 | 0.02 | 0.00 | 0.00 |
| s4 | 0.5 | 0 | naive_reuse | 0.945 | 0.755 | 0.1906 | 685 | 4.20 | 0.27 | 0.19 | 0.39 | 0.02 | 0.05 | 0.00 |
| s4 | 0.5 | 0 | reuse_norevise | 0.812 | 0.652 | 0.1604 | 550 | 3.55 | 0.00 | 0.00 | 0.36 | 0.02 | 0.00 | 0.00 |
| s4 | 0.5 | 2 | greedy | 0.195 | 0.157 | 0.0379 | 0 | 0.00 | 0.00 | 0.00 | 0.13 | 0.01 | 0.00 | 0.00 |
| s4 | 0.5 | 2 | recompute | 0.977 | 0.808 | 0.1684 | 567 | 4.25 | 0.16 | 0.15 | 0.46 | 0.07 | 0.00 | 0.00 |
| s4 | 0.5 | 2 | reuse | 0.977 | 0.837 | 0.1393 | 441 | 3.14 | 0.14 | 0.12 | 0.34 | 0.06 | 0.00 | 0.72 |
| s4 | 0.5 | 2 | naive_reuse | 0.805 | 0.652 | 0.1524 | 492 | 3.73 | 0.84 | 0.28 | 0.34 | 0.08 | 0.92 | 1.55 |
| s4 | 0.5 | 2 | reuse_norevise | 0.875 | 0.753 | 0.1221 | 364 | 2.80 | 0.00 | 0.00 | 0.34 | 0.06 | 0.00 | 0.70 |
| s4 | 0.5 | 4 | greedy | 0.188 | 0.150 | 0.0370 | 0 | 0.00 | 0.00 | 0.00 | 0.14 | 0.02 | 0.00 | 0.00 |
| s4 | 0.5 | 4 | recompute | 0.961 | 0.804 | 0.1569 | 510 | 4.19 | 0.14 | 0.13 | 0.51 | 0.11 | 0.00 | 0.00 |
| s4 | 0.5 | 4 | reuse | 0.969 | 0.856 | 0.1126 | 314 | 2.55 | 0.11 | 0.09 | 0.28 | 0.05 | 0.00 | 1.08 |
| s4 | 0.5 | 4 | naive_reuse | 0.781 | 0.658 | 0.1229 | 358 | 2.62 | 0.91 | 0.28 | 0.24 | 0.05 | 1.47 | 2.46 |
| s4 | 0.5 | 4 | reuse_norevise | 0.898 | 0.801 | 0.0978 | 247 | 2.34 | 0.00 | 0.00 | 0.27 | 0.05 | 0.00 | 1.04 |
| s4 | 1.0 | 0 | greedy | 0.234 | 0.196 | 0.0383 | 0 | 0.00 | 0.00 | 0.00 | 0.27 | 0.02 | 0.00 | 0.00 |
| s4 | 1.0 | 0 | recompute | 0.945 | 0.759 | 0.1859 | 643 | 4.75 | 0.34 | 0.24 | 0.94 | 0.10 | 0.00 | 0.00 |
| s4 | 1.0 | 0 | reuse | 0.945 | 0.764 | 0.1817 | 634 | 4.50 | 0.34 | 0.24 | 0.94 | 0.10 | 0.00 | 0.00 |
| s4 | 1.0 | 0 | naive_reuse | 0.945 | 0.764 | 0.1816 | 633 | 4.48 | 0.34 | 0.24 | 0.94 | 0.10 | 0.10 | 0.00 |
| s4 | 1.0 | 0 | reuse_norevise | 0.758 | 0.614 | 0.1438 | 465 | 3.70 | 0.00 | 0.00 | 0.90 | 0.10 | 0.00 | 0.00 |
| s4 | 1.0 | 2 | greedy | 0.211 | 0.172 | 0.0392 | 0 | 0.00 | 0.00 | 0.00 | 0.30 | 0.05 | 0.00 | 0.00 |
| s4 | 1.0 | 2 | recompute | 0.992 | 0.794 | 0.1983 | 710 | 4.51 | 0.15 | 0.11 | 0.86 | 0.15 | 0.00 | 0.00 |
| s4 | 1.0 | 2 | reuse | 0.992 | 0.823 | 0.1694 | 587 | 3.48 | 0.15 | 0.11 | 0.67 | 0.11 | 0.00 | 0.59 |
| s4 | 1.0 | 2 | naive_reuse | 0.859 | 0.675 | 0.1842 | 656 | 3.56 | 0.64 | 0.21 | 0.55 | 0.06 | 1.02 | 1.53 |
| s4 | 1.0 | 2 | reuse_norevise | 0.891 | 0.734 | 0.1570 | 535 | 3.12 | 0.00 | 0.00 | 0.62 | 0.11 | 0.00 | 0.58 |
| s4 | 1.0 | 4 | greedy | 0.164 | 0.126 | 0.0378 | 0 | 0.00 | 0.00 | 0.00 | 0.27 | 0.05 | 0.00 | 0.00 |
| s4 | 1.0 | 4 | recompute | 0.961 | 0.736 | 0.2247 | 841 | 4.59 | 0.15 | 0.09 | 0.87 | 0.12 | 0.00 | 0.00 |
| s4 | 1.0 | 4 | reuse | 0.961 | 0.782 | 0.1788 | 644 | 2.92 | 0.16 | 0.11 | 0.52 | 0.09 | 0.00 | 1.10 |
| s4 | 1.0 | 4 | naive_reuse | 0.719 | 0.510 | 0.2087 | 771 | 3.51 | 1.30 | 0.33 | 0.52 | 0.05 | 1.43 | 2.45 |
| s4 | 1.0 | 4 | reuse_norevise | 0.883 | 0.724 | 0.1587 | 554 | 2.52 | 0.00 | 0.00 | 0.48 | 0.09 | 0.00 | 1.07 |

### Stage A gate

| Clause | Result (pooled over all 18 cells, 2,304 worlds per reference) | Holds? |
|---|---|---|
| `dep_reuse` success ≥ `dep_recompute` − 0.01, with materially lower cost | Success 0.975 vs 0.974. The worst per-cell gap is +0.000 (reuse never trails). Cost −13.0% pooled; −11% to −17% at 2 foreign records and −20% to −29% at 4; −0.4% to −3.3% with no foreign records (leverage there comes only from route reuse after assignment revision and from events). | **Yes**, pooled and in the foreign-record region; the cost advantage is small with no foreign records |
| `dep_naive_reuse` substantially worse where stale or foreign records exist | Foreign > 0: success 0.812 vs 0.975, utility 0.697 vs 0.868. Events only (foreign = 0, p_event > 0): 0.969 vs 0.969, utility equal. | **Yes for foreign records; no for event-staleness alone** (see the limitations) |
| `dep_greedy` materially worse where CSP/deadline coupling bites | 0.240 vs 0.975 pooled; 0.13–0.39 in every cell | **Yes** |
| Some worlds require revision (reuse without revision is materially lower) | `dep_reuse_norevise` 0.871 vs 0.975 pooled; 0.76–0.91 per cell. 9–24% of reuse episodes revise. | **Yes** |

## Tests (`tests/test_campaign03_depworld.py`, 27 tests)
- **Planted feasibility,** by brute force over all selections, assignments and shortest routes, before and after the event. It covers 60 worlds × 2 small sizes, all four event kinds, and the planted plan itself.
- **Generation:** deterministic and JSON-safe.
- **No hidden information before inspection,** including identical d1 encodings for worlds that differ only in hidden facts.
- **Foreign records** hide labels and payloads until `retrieve`.
- **Dependency gating and uncommit.**
- **Each event kind:** it is public and versioned, and it revokes what it should.
- **Every declared rejection reason** is triggered on a hand-built world.
- **Attempted-action record** and its encoded retry features: attempted, last reason, dependencies changed only when relevant state changed.
- **Public `applicable()` against the hidden audit** on random mixed-policy states (40 worlds with events and 4 foreign records).
  - Always sound: public-applicable implies hidden-applicable, and the payload is valid now.
  - Exactly equal whenever inspection is complete: more than 2,000 checks, with both classes represented.
- **Foreign kinds:** applicable and inapplicable kinds both occur, and registry membership never separates them.
- **Version mismatch:** a record that differs only in dependency version is visible through the dependency relation alone.
- **Rename invariance** of the full encoded trajectories (including renamed foreign snapshots and selection ids).
- **Dimension stability and finiteness** over random states and sizes.
- **Relations, not an applicability bit:** the relations' conjunction equals `applicable()`, and the catalogue offers all uses (no masks).
- **References** stay in the catalogue, and `make_reference` works for `dep_` and `modular_` names.
- **A small leverage profile** reproduces the gate ordering.
- **Executor contract** for all three primitives: timeout, invalid input, the isolated default path, and the environment's work-contract check.
- **Utility accounting.**
- **`collect_teacher`** with a width-8 `CandidatePolicy` (d1), plus `supervised_loss`.
- **A tiny `PopulationRun`** with `world_family="depworld"`.
- **`episode_counts`** on a depworld outcome.

The full campaign suite (`-k campaign0` plus this file) gives 168 passed. That is the 141 existing tests plus the 27 new ones, with no regressions.

## Known limitations and items for the coordinator
- **Staleness is detected by the world.** Every commit or use is validated physically against the current requirements, so a stale own record yields an explicit rejection reason (`missing_edge`, `slot_unavailable`, `capacity`, `stale_dependency`), not silent harm. `dep_naive_reuse` also never re-applies a used record. Naive reuse therefore loses almost nothing when events are the only source of staleness. Its losses come from foreign request mismatches that are accepted physically: routes to another goal, earlier-map routes that are valid but not shortest, and valid-but-slow earlier-shift selections.
  - If Stage B/C needs staleness to cost success, either make naive literal (it loops after revisions) or add an event kind whose stale result stays physically valid but suboptimal, such as an edge that becomes *slower*.
- **Event exposure depends on speed.** Faster references finish, or deliver, before a scheduled event more often: 47% vs 83% of s3 reuse episodes see the event at foreign = 4 vs 0. Paired comparisons share worlds but not event exposure. A progress-triggered event, for example after the Nth commit, would equalize exposure but departs from the spec's "public step count".
- **Foreign snapshots reveal facts before inspection.** Like any public record, they expose the item facts and requirements of the instance they solved. The d1 encoder exposes only relations, and request match needs a completed inspection.
- **No reuse leverage without foreign records or events.** With foreign = 0 and no events, reuse and recompute differ by under 1.5% in cost. The leverage region is foreign-record conditions and revision- or event-heavy worlds.
- **The revision search is a heuristic.** It excludes the longest-duration item of a selection shown dead by an exhaustive infeasible CSP. About 1–5% of worlds fail for recompute and reuse alike, although the planted plan exists.
- **s4 is compute-dominated.** Exhaustive CSP infeasibility proofs cost up to about 1,000 units, so utility at work_price 5e-4 drops to about 0.5–0.6 for solver references. If s4 is used for training, check that the cost scale is intended.
- **Stage B preflight is not done yet.** The relations it needs are exposed, and a paired-state audit over the actual encoded tensors should be written next.
