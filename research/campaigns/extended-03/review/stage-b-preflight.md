# Stage B preflight: encoded-input audit of d1 and the P1 ablations (extended-03)

**Verdict: GATE PASSES.** Every required d1 distinction (1–10) is preserved on the actual encoded tensors. The ablations `d1-noapp` (X3) and `d1-noattempt` (X4) erase exactly their intended distinctions and keep all the others. There are **no d1 encoder defects that block training**. Two minor, non-blocking findings (F1, F2) and three design disclosures for the P1 readout (F3–F5) follow. The root agent decides on them.

- **Auditor.** An independent Stage B subagent. It did not write the d1 encoder. It restarted from the interrupted WIP (`campaign/e03-preflight`, caf1a785) and re-verified every part of it (see "Relation to the WIP").
- **Branch.** `campaign/e03-preflight-v2`, from `campaign/extended-03` at 27c0de1d (post-rename; code in `src/tensegra/`).
- **Tool.** `research/tools/campaign03_preflight.py`. It writes [`stage-b-preflight.json`](stage-b-preflight.json), about 8 CPU-s, with an in-process solver and no GPU.
- **Gate test.** `tests/test_campaign03_preflight.py` (20 tests). A failing pair, a failed catalogue check or a failed functional ablation check fails the suite.
- **Ablations.** Implemented in `src/tensegra/campaign03_depworld.py` (`FEATURE_MASKS`, `OBSERVATION_NAMES_D1`, `CANDIDATE_NAMES_D1`, `encode_public`). They are selectable through `public_frame(obs, v)`, `PolicyConfig.feature_version` and the frozen X3/X4 boot configs.

## Method

Each pair is built by **stepping real `DepWorkshop` instances**. The worlds are hand-built `DepSpec`s with the in-process `campaign02_protocol.execute` behind `depworld_executor`. Pairs are compared on the exact tensors the policy consumes: `campaign02_training.public_frame(observation, version)`, which returns the candidate set, the observation vector (82) and the candidate matrix (153 per row). Nothing is compared on raw observations or on encoder helpers.

Each pair declares its **carrier**, the part of the input that must carry the distinction:
- `row`: the named `use_return`/commit/move candidate row(s);
- `observation`: the observation vector;
- `state`: observation + candidate set + full matrix;
- an explicit coordinate list.

For every version the JSON records:
- the observation diff and the row diff (semantic coordinate names);
- whether the candidate sets are equal, and how many matrix rows differ;
- construction preconditions, e.g. the same status/certificate/retrieved flags, the same problem, the same pending set, the same step.

Expected verdicts per version: D = must differ, = = must be identical, r = (ablation) the relations must be gone and only declared payload feasibility facts may remain, ? = report only.

**Hand world** (tool header). Six items, A1–A3 and B1–B3, over two categories.
- **Selections:** {A1,B1} is valid. {A1,B3} fails on capacity, {A1,B2} on funds, {A2,B1} on incompatibility, and {A3,B1} on capacity.
- **Slots:** slot 3 is closed.
- **Map:** 0→1→2→3 costs 4 and 0→3 costs 6. With deadline 9, `move(3)` from 0 misses the deadline.
- **Foreign records:** 11 registered kinds, the generator's 8 plus `subset_other_order` (request-only), `route_slow` (request-only, physically deadline-failing) and `route_missing_edge`. They are built from the world's true facts, as the generator does.

**Functional ablation checks (distinction 11).** These run on generated worlds (dev seeds 2,000,007,000+, p_event 1, 4 foreign records), along `dep_reuse`, `dep_naive_reuse` and `dep_recompute` trajectories: 523 states and 62,152 candidates. At every state:
1. **`d1-noapp` is invariant to the relation values.** `relations()` and `_draft_match()` are monkeypatched to random request/canonical/dependency/requirements/selection bits, and the `d1-noapp` tensors must not change. `d1` changes in 523/523 states.
2. **`d1-noattempt` is invariant to the attempt log.** Outcomes and reasons are shuffled, dependency snapshots perturbed and entries dropped, and the `d1-noattempt` tensors must not change. `d1` changes in 257/257 states that have attempts.
3. **`d1-noattempt` equals `d1`** on every unmasked coordinate.
4. **`d1-noapp` equals `d1`** on every unmasked coordinate except the four that read the need bound (see F3). Measured differences occur only there: `constraint.bound_within_need` 1,298, `payload.meets_bound_or_selection_valid` 360, `slot.meets_bound` 127.
5. Masked coordinates are zero, and the dimensions equal d1's (82/153).

**Sensitivity of the gate (mutation checks, run once, not committed).**

| Mutation | Result |
|---|---|
| Masking `record.dependency_match`/`requirements_match` in d1 | Fails 1c, 4a (both variants), 2b, 2d |
| Masking `attempt.last_reason.*` in d1 | Fails all eight "after1/after2" reason pairs plus the deliver pair |
| An empty `d1-noattempt` mask | Fails 6-after*, 6-move, 7a, 7b |
| Dropping the payload proxies from the `d1-noapp` mask | Fails 1d/1e/4d/4g-retrieved |
| Leaving the applicability-derived route bound in the `d1-noapp` context | Fails functional check 1 |

**d1 is byte-identical to 27c0de1d.** The observation and candidate matrix were checked along 939 states (8 dev seeds × greedy/recompute/reuse/naive_reuse, events and foreign records). The ablation plumbing does not change d1.

## Per-distinction results

| # | Distinction (brief) | Pairs | d1 | d1-noapp | d1-noattempt |
|---|---|---|---|---|---|
| 1 | Own current applicable result vs a same-type result of another registered (foreign) problem, with the same status/certificate/retrieved flags | 12 | **PASS** (all differ, on relation coordinates) | erased (metadata); relations erased, feasibility residual (retrieved) | kept |
| 2 | Current vs stale result of the **same problem** after a requirement change | 6 | **PASS** | erased (1 residual: route distance) | kept |
| 3 | Old-but-applicable vs stale. An old applicable result is **identical** to a fresh one | 7 | **PASS** (3c/3d/3e identical) | erased | kept |
| 4 | Foreign applicable vs foreign not applicable (dependency-only, request-only, both) | 14 | **PASS** | erased (metadata); relations erased, feasibility residual (retrieved) | kept |
| 5 | Wrong type vs right type for a use | 8 | **PASS** | kept (`record.type_match`, primitive) | kept |
| 6 | Rejection reasons (capacity/funds/incompatibility; slot_conflict/slot_unavailable; deliver deadline/missing_edge; move deadline): immediately, after 1 and after 2 intervening actions | 14 | **PASS** | kept | immediate kept (last-step feedback); **after ≥1 action erased** |
| 7 | Identical retry vs never tried; vs retry after dependencies changed (pending change, event, revision, inspection) | 7 | **PASS** (see F1) | kept | erased (7e is carried by `req.*` and kept, as intended) |
| 8 | Committed vs revoked after an event; which requirement changed | 10 | **PASS** | kept | kept |
| 9 | Renaming invariance (items, record handles, foreign snapshots, selection ids; three reference trajectories with an event); ownership invariance; `problem_n` spelling | 10 | **PASS** (all identical) | identical | identical |
| 10 | No hidden-state leakage (uninspected item facts, planted plan, future event under both triggers, unretrieved payload, evaluator label) | 12 | **PASS** (all identical; sensitivity control differs) | identical | identical |
| 11 | Ablations: dimensions; exact erasure per pair; functional invariance | all of the above + 523 states | n/a | **PASS** | **PASS** |

**Catalogue.** In four states (the full chain with 14 records and 14 problems, unretrieved and retrieved; after a selection revocation; after edge_closed), the candidate set contains `retrieve` and all three `use_return` uses for every record. It also contains every `call` budget for every problem, `commit_*`, `verify`, both `uncommit`s, every `move` along a known out-edge and every `choose_slot` of a selected item. There are no duplicates.

Totals: 100 pairs, **100 PASS**. Gate:

```
{'d1_all_distinctions': True, 'ablations_exact': True, 'catalog': True, 'functional': True, 'dimensions_equal': True, 'pass': True}
```

## Ablation definitions (as implemented)

Both ablations keep the same dimensions as d1, and the masked coordinates are zero.

**`d1-noapp` (X3): request/dependency-match relations and everything derived from them.**
- **Candidate coordinates masked:**
  - `record.request_match`, `record.canonical_match`, `record.dependency_match`, `record.requirements_match`, `record.selection_match`;
  - `payload.route_starts_at_position` and `payload.route_ends_at_destination`, the route request components;
  - `payload.csp_items_equal_selection`, the selection dependency;
  - `draft.request_match`, `draft.dependency_match`;
  - `start.matching_drafts`, `start.live_applicable_records`.
- **Observation coordinates masked:** `route.applicable_known`, `route.distance`, `route.plan_meets_deadline`, `route.need_bound`.
- **Context change:** the encoder context also drops the applicability-selected best route. `need_bound` then falls back to deadline − travel wherever it is read (see F3).
- **Kept:** type match, usable status, certificate, retrieved flag, snapshot options, and the payload feasibility facts (`payload.route_distance`, `payload.route_meets_deadline`, `payload.finish_or_total_duration`, `payload.meets_bound_or_selection_valid`).

**`d1-noattempt` (X4): everything computed from `observation.attempts`.**
- **Masked:** the 29-coordinate per-candidate block `attempt.*` (attempted, count, last outcome, last reason, dependencies changed), and the observation counters `attempts.count`, `attempts.rejected`, `attempts.uncommits`.
- **Kept:** the one-step `feedback.status.*`/`feedback.reason.*`.

**Plumbing.**
- `campaign02_training.DEPWORLD_FEATURE_VERSIONS = ("d1", "d1-noapp", "d1-noattempt")` is dispatched by `public_frame`. A mismatch in either direction raises.
- `PolicyConfig` accepts both ablation names.
- The frozen `configs/campaign03/p1-boot-x3-*/x4-*` configs load. They build a policy of their feature version, and `collect_teacher` feeds the masked tensors (test). Before this change `PolicyConfig` rejected `d1-noapp`/`d1-noattempt`, so the X3/X4 bootstraps would have failed at launch.
- `campaign03_depworld.py` changed, so the depworld `source_hashes` of new runs differ. No extended-03 run exists yet.

## Findings

**F1: minor d1 gap, non-blocking.** `attempt.dependencies_changed` does not register a requirements inspection (pair 7e-bit).
- `relevant_dependencies(commit_pending)` is the pending set, the requirement *versions* and the selection. It does not include whether the requirements are *known*.
- So a `commit_pending` rejected with `missing_dependency` (requirements not inspected), followed by `inspect(requirements)`, still shows the deps-changed bit at 0 on the retry. The same holds for `use_return` as select/assign.
- The distinction itself is preserved at the state level (pair 7e: `req.known`, `req.*` and `pending.*_left` differ, and both ablations keep them). A model can learn the interaction.
- References always inspect requirements first, so no teacher trajectory hits this case.
- **Proposed registered fix (only if root wants the bit exact):** add `"requirements_known": o.requirements is not None` to `relevant_dependencies` for `commit_pending`, `commit_assignment` and `use_return`.
- **Cost of the fix:** it changes the attempt record and `DepReference._repeat_failure` semantics, so re-run the depworld suite and the leverage profile. It also changes the identical-retry metric definition. **Recommendation: accept as disclosed; no change before P1.** When the P1 identical-retry metric is implemented, it must use `relevant_dependencies` so that it matches the encoder.

**F2: disclosure; not hidden-state leakage.** The public foreign-snapshot channel reaches two d1 payload coordinates before inspection (pairs 10x).
- **csp:** after `retrieve`, a foreign csp record's `payload.finish_or_total_duration` is computed from the record's own snapshot durations. Those are true item facts when the record was built from them. With nothing inspected, B1's duration (2 vs 1) is visible through this coordinate (10x-csp).
- **route:** a retrieved route record's `payload.route_distance` reflects the map (edge weight) before `inspect(map)` (10x-route).
- **Nothing leaks unretrieved** (10x-csp-meta), nor through subset records (10x-subset), since selection facts require all rows to be inspected.
- **Why it is not a leak:** the snapshot, and after `retrieve` the payload, are public observation fields. Any actor with raw observations sees them. Pairs 10a–10g, which hold the public facts equal, are all identical.
- **Correction to depworld-notes.** The notes (limitations) say "the d1 encoder exposes only relations". That is inaccurate for these two coordinates. Update the notes.
- **Optional fix:** compute the csp finish from `known_items` durations, zero if any snapshot item is uninspected, and gate `payload.route_distance` on `known_edges is not None`. **Recommendation: no change** (not required by any distinction; changes d1).

**F3: ablation design.** `d1-noapp` needs more than a coordinate mask.
- The observation's route bound (`route.*`) is the minimum distance over retrieved routes that are request- *and* dependency-matched. It is an applicability readout.
- It also feeds `need_bound`, and through it four candidate coordinates.
- A mask-only ablation (the WIP approach) therefore leaks applicability into X3. The mutation check confirms this: functional check 1 fails.
- As implemented, `d1-noapp` drops that bound in the context. The four need-bound coordinates then use the deadline − travel fallback that d1 already uses when no applicable route is known. These are the only unmasked coordinates on which `d1-noapp` differs from d1 (functional check 4).

**F4: X3 residual (by design).** With payloads retrieved, `d1-noapp` still separates 12 of the 1/2/4 pairs, but only through payload **feasibility facts**:
- route distance (1e/1f/2c/4d/4e/4f);
- selection total duration and current validity (1a/1b/4b/4c);
- csp finish (1d/4g).

No relation coordinate survives; the check is enforced (`r`). These are genuine physical facts, not applicability: an earlier-shift selection can be valid but mismatched, and a route to another goal has a legitimate distance. Report them with R3.

**F5: X4 residual (by design).** `d1-noattempt` keeps:
- last-step feedback, so a reason is visible at the *immediate* next decision (6-*-immediate);
- record-derived call history (`call.timed_out_at_budget`, `call.infeasible_same_record`, `call.usable_same_records`), which comes from records, not the attempt log.

R4's identical-retry comparison is therefore cleanest on commit/use_return/move retries. For calls, X4 can still avoid re-calling a budget that timed out.

## Relation to the WIP (caf1a785)

Reused: the hand world, the pair skeleton and the mask coordinate names. Verified and corrected:
1. **Names.** The names now live next to the encoder. They are checked against the encoder by test: relation coordinates against `relations()` on more than 100 rows, the attempt block, and observation counters. One mislabel was fixed: the uncommit flag lives in the *first* commit coordinate, now named `commit.coverage_or_uncommit_target_committed`.
2. **Route-bound leak.** The noapp route-bound/need-bound leak (F3) is fixed.
3. **Handles in the leakage pairs.** They were computed with the wrong address seed, so the WIP retrieved a non-existent handle. Fixed.
4. **Rename trajectory.** The `dep_reuse` rename trajectory finished before its step-30 event, so no event was covered. It now uses a progress-triggered event and runs three references.
5. **Report-only pairs.** They were mixed with required ones (`None`). Required pairs now have strict expectations, and the retrieved-payload ablation check is explicit (`r`).
6. **New pairs:**
   - progress-trigger future events (10c/10d);
   - unretrieved-payload hiding with a sensitivity control (10e);
   - the evaluator label (10f);
   - a same-snapshot foreign record (10g);
   - selection revision and requirements inspection (7d/7e);
   - revoked vs never committed (8h);
   - request-only foreign kinds (1b, 4c, 4f);
   - the functional invariance checks.

## Full pair table

Evidence is the list of d1 coordinates that differ: the row for `row` carriers, observation then row for `state`/`observation` carriers, and the first differing step for trajectory pairs. Full per-version diffs, including the ablations' residual coordinates, are in the JSON.

#### D1

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 1a-unretrieved | use_return as select: own current result vs foreign registered same-type (subset_earlier), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 1b-unretrieved | use_return as select: own current result vs foreign registered same-type (subset_other_order), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.request_match` | PASS |
| 1c-unretrieved | use_return as select: own current result vs foreign registered same-type (subset_version_mismatch), unretrieved | row | D/=/D | differ | same | differ | `record.dependency_match`, `record.requirements_match` | PASS |
| 1d-unretrieved | use_return as assign: own current result vs foreign registered same-type (csp_other_selection), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.selection_match` | PASS |
| 1e-unretrieved | use_return as route: own current result vs foreign registered same-type (route_other_goal), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.request_match` | PASS |
| 1f-unretrieved | use_return as route: own current result vs foreign registered same-type (route_earlier), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 1a-retrieved | use_return as select: own current result vs foreign registered same-type (subset_earlier), retrieved | row | D/r/D | differ | differ | differ | `payload.finish_or_total_duration`, `payload.meets_bound_or_selection_valid`, `record.canonical_match`, `record.dependency_match`, `record.request_match` (+1) | PASS |
| 1b-retrieved | use_return as select: own current result vs foreign registered same-type (subset_other_order), retrieved | row | D/r/D | differ | differ | differ | `payload.finish_or_total_duration`, `payload.meets_bound_or_selection_valid`, `record.canonical_match`, `record.request_match` | PASS |
| 1c-retrieved | use_return as select: own current result vs foreign registered same-type (subset_version_mismatch), retrieved | row | D/r/D | differ | same | differ | `record.dependency_match`, `record.requirements_match` | PASS |
| 1d-retrieved | use_return as assign: own current result vs foreign registered same-type (csp_other_selection), retrieved | row | D/r/D | differ | differ | differ | `payload.csp_items_equal_selection`, `payload.finish_or_total_duration`, `record.canonical_match`, `record.dependency_match`, `record.request_match` (+1) | PASS |
| 1e-retrieved | use_return as route: own current result vs foreign registered same-type (route_other_goal), retrieved | row | D/r/D | differ | differ | differ | `payload.route_distance`, `payload.route_ends_at_destination`, `record.canonical_match`, `record.request_match` | PASS |
| 1f-retrieved | use_return as route: own current result vs foreign registered same-type (route_earlier), retrieved | row | D/r/D | differ | differ | differ | `payload.route_distance`, `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |

#### D2

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 2a-unretrieved | own subset result from before capacity_reduced(6->5) (stale: request+dependency) vs the same draft re-called after the event, unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 2a-retrieved | own subset result from before capacity_reduced(6->5) (stale: request+dependency) vs the same draft re-called after the event, retrieved | row | D/r/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 2b | own subset result from before a same-value capacity event (dependency-only stale) vs re-called | row | D/=/D | differ | same | differ | `record.dependency_match`, `record.requirements_match` | PASS |
| 2c-unretrieved | own route from before edge_closed(1,2) (stale) vs route recomputed after, unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 2c-retrieved | own route from before edge_closed(1,2) (stale) vs route recomputed after, retrieved | row | D/r/D | differ | differ | differ | `payload.route_distance`, `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 2d | own csp from before slot_closed(4) (slot unused by the selection: dependency-only stale) vs re-called | row | D/=/D | differ | same | differ | `record.dependency_match`, `record.requirements_match` | PASS |

#### D3

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 3a-unretrieved | selection result from before edge_closed (still applicable) vs the same result from before capacity_reduced (stale), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 3b-unretrieved | route from before capacity_reduced (still applicable) vs the same route from before edge_closed (stale), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 3a-retrieved | selection result from before edge_closed (still applicable) vs the same result from before capacity_reduced (stale), retrieved | row | D/r/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 3b-retrieved | route from before capacity_reduced (still applicable) vs the same route from before edge_closed (stale), retrieved | row | D/r/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 3c | old-but-applicable selection result (pre-edge_closed) vs a fresh re-call after the event (same request) | row | =/=/= | same | same | same | (identical) | PASS |
| 3d | old-but-applicable route (pre-capacity_reduced) vs a fresh recompute after the event | row | =/=/= | same | same | same | (identical) | PASS |
| 3e | old-but-applicable selection result (pre-slot_closed) vs a fresh re-call after the event | row | =/=/= | same | same | same | (identical) | PASS |

#### D4

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 4a-unretrieved | use_return as select: foreign applicable (subset_current) vs foreign not applicable (subset_version_mismatch): dependency-only mismatch, unretrieved | row | D/=/D | differ | same | differ | `record.dependency_match`, `record.requirements_match` | PASS |
| 4b-unretrieved | use_return as select: foreign applicable (subset_current) vs foreign not applicable (subset_earlier): request + dependency mismatch, unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 4c-unretrieved | use_return as select: foreign applicable (subset_current) vs foreign not applicable (subset_other_order): request-only mismatch, unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.request_match` | PASS |
| 4d-unretrieved | use_return as route: foreign applicable (route_current) vs foreign not applicable (route_other_goal): request-only mismatch (other goal), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.request_match` | PASS |
| 4e-unretrieved | use_return as route: foreign applicable (route_current) vs foreign not applicable (route_earlier): earlier map, version 0, unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 4f-unretrieved | use_return as route: foreign applicable (route_current) vs foreign not applicable (route_slow): request-only mismatch (wrong edge weight), unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.request_match` | PASS |
| 4g-unretrieved | use_return as assign: foreign applicable (csp_current) vs foreign not applicable (csp_other_selection): other selection, unretrieved | row | D/=/D | differ | same | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.selection_match` | PASS |
| 4a-retrieved | use_return as select: foreign applicable (subset_current) vs foreign not applicable (subset_version_mismatch): dependency-only mismatch, retrieved | row | D/r/D | differ | same | differ | `record.dependency_match`, `record.requirements_match` | PASS |
| 4b-retrieved | use_return as select: foreign applicable (subset_current) vs foreign not applicable (subset_earlier): request + dependency mismatch, retrieved | row | D/r/D | differ | differ | differ | `payload.finish_or_total_duration`, `payload.meets_bound_or_selection_valid`, `record.canonical_match`, `record.dependency_match`, `record.request_match` (+1) | PASS |
| 4c-retrieved | use_return as select: foreign applicable (subset_current) vs foreign not applicable (subset_other_order): request-only mismatch, retrieved | row | D/r/D | differ | differ | differ | `payload.finish_or_total_duration`, `payload.meets_bound_or_selection_valid`, `record.canonical_match`, `record.request_match` | PASS |
| 4d-retrieved | use_return as route: foreign applicable (route_current) vs foreign not applicable (route_other_goal): request-only mismatch (other goal), retrieved | row | D/r/D | differ | differ | differ | `payload.route_distance`, `payload.route_ends_at_destination`, `record.canonical_match`, `record.request_match` | PASS |
| 4e-retrieved | use_return as route: foreign applicable (route_current) vs foreign not applicable (route_earlier): earlier map, version 0, retrieved | row | D/r/D | differ | differ | differ | `payload.route_distance`, `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match` | PASS |
| 4f-retrieved | use_return as route: foreign applicable (route_current) vs foreign not applicable (route_slow): request-only mismatch (wrong edge weight), retrieved | row | D/r/D | differ | differ | differ | `payload.route_distance`, `record.canonical_match`, `record.request_match` | PASS |
| 4g-retrieved | use_return as assign: foreign applicable (csp_current) vs foreign not applicable (csp_other_selection): other selection, retrieved | row | D/r/D | differ | differ | differ | `payload.csp_items_equal_selection`, `payload.finish_or_total_duration`, `record.canonical_match`, `record.dependency_match`, `record.request_match` (+1) | PASS |

#### D5

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 5a-unretrieved | use_return as select: csp record (wrong type) vs subset record (right type), unretrieved | row | D/D/D | differ | differ | differ | `primitive.constrained_subset`, `primitive.csp`, `record.canonical_match`, `record.dependency_match`, `record.request_match` (+3) | PASS |
| 5b-unretrieved | use_return as route: foreign applicable csp (wrong type) vs applicable route, unretrieved | row | D/D/D | differ | differ | differ | `primitive.csp`, `primitive.shortest_path`, `record.canonical_match`, `record.dependency_match`, `record.request_match` (+3) | PASS |
| 5c-unretrieved | same subset record: use_return as assign (wrong) vs as select (right), unretrieved | row | D/D/D | differ | differ | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match`, `record.selection_match` (+3) | PASS |
| 5d-unretrieved | type relation only: use_return as select of a csp record vs of a route record (both wrong type), unretrieved | row | D/D/D | differ | differ | differ | `primitive.csp`, `primitive.shortest_path` | PASS |
| 5a-retrieved | use_return as select: csp record (wrong type) vs subset record (right type), retrieved | row | D/D/D | differ | differ | differ | `payload.csp_items_equal_selection`, `primitive.constrained_subset`, `primitive.csp`, `record.canonical_match`, `record.dependency_match` (+4) | PASS |
| 5b-retrieved | use_return as route: foreign applicable csp (wrong type) vs applicable route, retrieved | row | D/D/D | differ | differ | differ | `payload.csp_items_equal_selection`, `payload.finish_or_total_duration`, `payload.meets_bound_or_selection_valid`, `payload.route_distance`, `payload.route_ends_at_destination` (+10) | PASS |
| 5c-retrieved | same subset record: use_return as assign (wrong) vs as select (right), retrieved | row | D/D/D | differ | differ | differ | `record.canonical_match`, `record.dependency_match`, `record.request_match`, `record.requirements_match`, `record.selection_match` (+3) | PASS |
| 5d-retrieved | type relation only: use_return as select of a csp record vs of a route record (both wrong type), retrieved | row | D/D/D | differ | differ | differ | `payload.csp_items_equal_selection`, `payload.finish_or_total_duration`, `payload.meets_bound_or_selection_valid`, `payload.route_distance`, `payload.route_ends_at_destination` (+4) | PASS |

#### D6

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 6-capacity-funds-immediate | commit_pending rejected for capacity vs funds: the next decision | state | D/D/D | differ | differ | differ | `pending.weight`, `pending.price`, `pending.capacity_left`, `pending.funds_left`, `feedback.reason.capacity` (+5) | PASS |
| 6-capacity-funds-after1 | commit_pending rejected for capacity vs funds, then one repair choose_item to the SAME pending set | state | D/D/= | differ | differ | same | `attempt.last_reason.capacity`, `attempt.last_reason.funds` | PASS |
| 6-capacity-funds-after2 | commit_pending rejected for capacity vs funds, repair + think (two intervening) | state | D/D/= | differ | differ | same | `attempt.last_reason.capacity`, `attempt.last_reason.funds` | PASS |
| 6-capacity-incompatibility-immediate | commit_pending rejected for capacity vs incompatibility: the next decision | state | D/D/D | differ | differ | differ | `pending.weight`, `pending.capacity_left`, `feedback.reason.capacity`, `feedback.reason.incompatibility`, `attempt.last_reason.capacity` (+3) | PASS |
| 6-capacity-incompatibility-after1 | commit_pending rejected for capacity vs incompatibility, then one repair choose_item to the SAME pending set | state | D/D/= | differ | differ | same | `attempt.last_reason.capacity`, `attempt.last_reason.incompatibility` | PASS |
| 6-capacity-incompatibility-after2 | commit_pending rejected for capacity vs incompatibility, repair + think (two intervening) | state | D/D/= | differ | differ | same | `attempt.last_reason.capacity`, `attempt.last_reason.incompatibility` | PASS |
| 6-funds-incompatibility-immediate | commit_pending rejected for funds vs incompatibility: the next decision | state | D/D/D | differ | differ | differ | `pending.price`, `pending.funds_left`, `feedback.reason.funds`, `feedback.reason.incompatibility`, `attempt.last_reason.funds` (+3) | PASS |
| 6-funds-incompatibility-after1 | commit_pending rejected for funds vs incompatibility, then one repair choose_item to the SAME pending set | state | D/D/= | differ | differ | same | `attempt.last_reason.funds`, `attempt.last_reason.incompatibility` | PASS |
| 6-funds-incompatibility-after2 | commit_pending rejected for funds vs incompatibility, repair + think (two intervening) | state | D/D/= | differ | differ | same | `attempt.last_reason.funds`, `attempt.last_reason.incompatibility` | PASS |
| 6-slot_conflict-slot_unavailable-immediate | commit_assignment rejected for slot_conflict vs slot_unavailable: the next decision | state | D/D/D | differ | differ | differ | `feedback.reason.slot_conflict`, `feedback.reason.slot_unavailable`, `attempt.last_reason.slot_conflict`, `attempt.last_reason.slot_unavailable`, `commit.assign_allowed` (+2) | PASS |
| 6-slot_conflict-slot_unavailable-after1 | commit_assignment rejected for slot_conflict vs slot_unavailable, then choose_slot to the SAME pending assignment | state | D/D/= | differ | differ | same | `attempt.last_reason.slot_conflict`, `attempt.last_reason.slot_unavailable` | PASS |
| 6-slot_conflict-slot_unavailable-after2 | as above, plus one think (two intervening actions) | state | D/D/= | differ | differ | same | `attempt.last_reason.slot_conflict`, `attempt.last_reason.slot_unavailable` | PASS |
| 6-deliver-deadline-missing_edge-after1 | deliver (use_return as route) rejected for deadline vs missing_edge, after one think: the attempted rows | `attempt.last_reason.capacity` ... | D/D/= | differ | differ | same | `attempt.last_reason.deadline`, `attempt.last_reason.missing_edge` | PASS |
| 6-move-deadline-vs-untried-after1 | move(3) rejected for deadline, then think vs never tried (step-matched) | state | D/D/= | differ | differ | same | `attempts.count`, `attempts.rejected`, `attempt.attempted`, `attempt.count`, `attempt.last_outcome.rejected` (+1) | PASS |

#### D7

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 7a | commit_pending already rejected on this exact state (identical retry) vs never attempted (same state) | state | D/D/= | differ | differ | same | `attempts.count`, `attempts.rejected`, `attempt.attempted`, `attempt.count`, `attempt.last_outcome.rejected` (+1) | PASS |
| 7b | retry after the pending set changed (rejected at another pending set, same reason) vs identical retry | state | D/D/= | differ | differ | same | `attempt.dependencies_changed` | PASS |
| 7b-bit | as 7b, carrier = attempt.dependencies_changed only | `attempt.dependencies_changed` | D/D/= | differ | differ | same | `attempt.dependencies_changed` | PASS |
| 7c | commit_assignment rejected (slot_conflict), then a slot requirement revision (event) vs unchanged | `attempt.dependencies_changed` | D/D/= | differ | differ | same | `attempt.dependencies_changed` | PASS |
| 7d | commit_assignment rejected, then a selection revision (uncommit + recommit) vs unchanged | `attempt.dependencies_changed` | D/D/= | differ | differ | same | `attempt.dependencies_changed` | PASS |
| 7e | commit_pending rejected (missing_dependency: requirements unknown), then inspect requirements vs think | state | D/D/D | differ | differ | differ | `req.known`, `req.capacity`, `req.funds`, `req.deadline`, `req.slot_capacity` (+4) | PASS |
| 7e-bit | as 7e, carrier = attempt.dependencies_changed only (report) | `attempt.dependencies_changed` | ?/?/= | same | same | same | (identical) | PASS |

#### D8

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 8a | after the event: capacity_reduced(3) revokes selection vs capacity_reduced(5) no revocation | observation | D/D/D | differ | differ | differ | `req.capacity`, `plan.selection_committed`, `plan.assignment_committed`, `plan.finish_time`, `plan.deadline_slack` (+5) | PASS |
| 8b | after the event: slot_closed(2) revokes assignment vs slot_closed(4) no revocation | observation | D/D/D | differ | differ | differ | `plan.assignment_committed`, `plan.finish_time`, `plan.deadline_slack`, `event.assignment_revoked`, `commit.coverage_or_uncommit_target_committed` | PASS |
| 8c | after the event: capacity_reduced(5) no revocation vs slot_closed(4) no revocation | observation | D/D/D | differ | differ | differ | `req.capacity`, `req.n_closed_slots`, `req.changed.capacity`, `req.changed.slots`, `pending.capacity_left` (+2) | PASS |
| 8d | after the event: edge_closed(1,2) vs deadline_moved(8) | observation | D/D/D | differ | differ | differ | `req.deadline`, `req.changed.deadline`, `req.changed.map`, `plan.deadline_slack`, `event.last.edge_closed` (+1) | PASS |
| 8e | after the event: capacity_reduced(5) no revocation vs edge_closed(1,2) | observation | D/D/D | differ | differ | differ | `req.capacity`, `req.changed.capacity`, `req.changed.map`, `pending.capacity_left`, `event.last.edge_closed` (+1) | PASS |
| 8f | after the event: slot_closed(4) no revocation vs deadline_moved(8) | observation | D/D/D | differ | differ | differ | `req.deadline`, `req.n_closed_slots`, `req.changed.slots`, `req.changed.deadline`, `plan.deadline_slack` (+2) | PASS |
| 8g | after the event: capacity_reduced(3) revokes selection vs slot_closed(2) revokes assignment | observation | D/D/D | differ | differ | differ | `req.capacity`, `req.n_closed_slots`, `req.changed.capacity`, `req.changed.slots`, `plan.selection_committed` (+6) | PASS |
| 8a-row | uncommit rows (target committed?) after: capacity_reduced(3) revokes selection vs capacity_reduced(5) no revocation | row | D/D/D | differ | differ | differ | `commit.coverage_or_uncommit_target_committed` | PASS |
| 8b-row | uncommit rows (target committed?) after: slot_closed(2) revokes assignment vs slot_closed(4) no revocation | row | D/D/D | differ | differ | differ | `commit.coverage_or_uncommit_target_committed` | PASS |
| 8h | selection revoked by capacity_reduced(3) vs never committed (after the same inspection) | observation | D/D/D | differ | differ | differ | `req.capacity`, `req.version`, `req.changed.capacity`, `count.problems`, `count.records` (+12) | PASS |

#### D9

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 9o-a-unretrieved | ownership/age invariance: own current select result vs foreign APPLICABLE subset_current, unretrieved | row | =/=/= | same | same | same | (identical) | PASS |
| 9o-b-unretrieved | ownership/age invariance: own current assign result vs foreign APPLICABLE csp_current, unretrieved | row | =/=/= | same | same | same | (identical) | PASS |
| 9o-c-unretrieved | ownership/age invariance: own current route result vs foreign APPLICABLE route_current, unretrieved | row | =/=/= | same | same | same | (identical) | PASS |
| 9o-a-retrieved | ownership/age invariance: own current select result vs foreign APPLICABLE subset_current, retrieved | row | =/=/= | same | same | same | (identical) | PASS |
| 9o-b-retrieved | ownership/age invariance: own current assign result vs foreign APPLICABLE csp_current, retrieved | row | =/=/= | same | same | same | (identical) | PASS |
| 9o-c-retrieved | ownership/age invariance: own current route result vs foreign APPLICABLE route_current, retrieved | row | =/=/= | same | same | same | (identical) | PASS |
| 9a-reuse | consistent renaming of item handles, record handles (address seed) and foreign snapshots/selection ids: observation, candidate set (mapped) and matrix identical at every step of a dep_reuse trajectory (capacity event after the first completion, 11 foreign records) | state | =/=/= | same | same | same | (identical) | PASS |
| 9a-naive_reuse | consistent renaming of item handles, record handles (address seed) and foreign snapshots/selection ids: observation, candidate set (mapped) and matrix identical at every step of a dep_naive_reuse trajectory (capacity event after the first completion, 11 foreign records) | state | =/=/= | same | same | same | (identical) | PASS |
| 9a-recompute | consistent renaming of item handles, record handles (address seed) and foreign snapshots/selection ids: observation, candidate set (mapped) and matrix identical at every step of a dep_recompute trajectory (capacity event after the first completion, 11 foreign records) | state | =/=/= | same | same | same | (identical) | PASS |
| 9b | problem-handle spelling: reverse the problem_n names so own drafts and foreign problems swap spellings (records, retrieved, attempts, action keys renamed) | state | =/=/= | same | same | same | (identical) | PASS |

#### D10

| Pair | Constructed pair | Carrier | Expected d1/noapp/noatt | d1 | d1-noapp | d1-noattempt | d1 tensor evidence | Result |
|---|---|---|---|---|---|---|---|---|
| 10a | uninspected item A1 (weight/price/duration/slots) differs; everything else inspected; A1 pending; own subset draft + call (the draft covers inspected items only) | state | =/=/= | same | same | same | (identical) | PASS |
| 10b | future event (slot_closed@60 vs deadline_moved@60, step trigger) and planted solution differ; full chain before the event | state | =/=/= | same | same | same | (identical) | PASS |
| 10c | future event under the progress trigger (capacity_reduced after completion 1 vs edge_closed after completion 2) vs no event; everything before the first completion commit | state | =/=/= | same | same | same | (identical) | PASS |
| 10d | no event scheduled vs capacity_reduced after completion 1 (progress); everything before that commit | state | =/=/= | same | same | same | (identical) | PASS |
| 10e | foreign route record with identical metadata but a different (hidden) payload; not retrieved; full inspection | state | =/=/= | same | same | same | (identical) | PASS |
| 10e-control | as 10e, then retrieve the record: the payload becomes public and must now be visible | state | D/D/D | differ | differ | differ | `route.distance`, `route.need_bound`, `payload.route_distance` | PASS |
| 10f | evaluator-only foreign label differs (route_current vs route_earlier) on an identical record | state | =/=/= | same | same | same | (identical) | PASS |
| 10g | uninspected A1's hidden facts differ while a foreign subset record (and its public snapshot) is identical in both worlds; everything but A1 inspected; retrieve the record | state | =/=/= | same | same | same | (identical) | PASS |
| 10x-csp | DISCLOSURE: foreign csp record for {A1,B1}; B1's hidden duration differs (2 vs 1) and so does the record's public snapshot; nothing inspected; retrieve the record; inspect requirements | state | ?/?/? | differ | differ | differ | `payload.finish_or_total_duration` | PASS |
| 10x-csp-meta | DISCLOSURE: as 10x-csp, record NOT retrieved (metadata only) | state | =/=/= | same | same | same | (identical) | PASS |
| 10x-route | DISCLOSURE: foreign current-route record; hidden edge (2,3) weight differs (1 vs 2) and so do the record's snapshot and payload; map NOT inspected; retrieve | state | ?/?/? | differ | differ | differ | `payload.route_distance` | PASS |
| 10x-subset | DISCLOSURE: foreign current-subset record; hidden A1 weight differs (2 vs 3); A1 NOT inspected (others are); retrieve | state | ?/?/? | same | same | same | (identical) | PASS |

