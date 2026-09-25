# Phase-3b independent audit: E19, E20, E21, E22

Auditor: independent re-derivation from raw sealed rows on `gb10-direct` (read-only, nothing launched, no GPU). All code in this directory was written fresh for this audit and shares nothing with earlier audit scripts:

- `phase3b_raw_remote.py`: main raw pass. It covers success and utility, distractor facts, failure anatomy, a recomputation of the m3 counters from traces, and seed ranges.
- `phase3b_detail_remote.py`: follow-up probes, covering rejected-action kinds, the anatomy of the E22 failures, E17 r0/r1 identity, foreign-record use, and whether E17 successes coincide with a goal-valid distractor.
- `phase3b_static.py`: config diffs and AST facts about the encoders.
- `phase3b_assemble.py`: assembles `phase3b-independent-audit.json`.

**Measured CPU:** 24.7 core-s in total (17.9 raw pass, 6.8 detail pass, 0.02 static), measured with `getrusage`/`process_time`. The budget was ~600.

**Row integrity:** every arm file has 256 rows per condition. Row seeds equal the world-file seeds, and row `spec_hash` equals the world `spec_hash`. The worlds are byte-identical by spec_hash in two comparisons:
- e20-sealed vs e22-sealed, for all 31 conditions;
- e20-sealed vs e17-sealed (16 conditions) and vs e19-sealed (8 same_* conditions).

The E21 and E15 roots have identical worlds for the 8 iid conditions and for int_4x4_control/no_tools.

## Verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | E19 numbers, distractor types, success tied to zero-distractor worlds | **CONFIRMED**; the mechanism statement is **QUALIFIED** |
| 2 | E20 1.000 on same_*/distr_*/pairs; triples 1.000/0.999/0.919; r2 residual is perseveration | **CONFIRMED** |
| 3 | E20 configs = E17 + m2 + same_type_distractors=2; m2 does not read `prior` | **CONFIRMED** (see concern C2) |
| 4 | E21 numbers; configs differ only by development_world_mix | **CONFIRMED** |
| 5 | E22 0/1/125 → 3/0/3 failures, perseveration 126 → 0; m3 = E20 + feature version; counters public and stage-reset | **CONFIRMED**, with two wording qualifications |
| 6 | Sealed seeds disjoint from training and development | **CONFIRMED** |

### 1. E19 (root e19-sealed, 8 same_* conditions × 256)

**Mean success over the 8 conditions:**
- e17rl-r0 / r1 / r2: **0.3672 / 0.3672 / 0.3472**. The claim is .367/.367/.347.
- Teacher `reference-modular_cheap_first`: **1.000**.
- No-tool greedy `reference-modular_cheap`: **0.6260**. The claim is .626.
- `modular_always_tool`: also 1.000.

**Distractor types:** 2,094 distractors in 2,048 worlds. By primitive: 755 constrained_subset, 812 csp, 527 shortest_path. Distractor counts per world: 636 with none, 730 with one, 682 with two.
- **0 of 2,094** have a primitive outside the goal's stage primitives.
- Every snapshot's primitive matches its tuple primitive.
- Wrong-type check (e20 distr_*): **0 of 1,520** distractors are of a goal primitive.

**Success tied to zero-distractor worlds (E17 learned arms):**

| | zero-distractor worlds (636) | worlds with ≥1 distractor (1,412) |
|---|---|---|
| r0 | 636/636 = 1.000 | 116/1,412 = 0.082 |
| r1 | 636/636 = 1.000 | 116/1,412 = 0.082 |
| r2 | 595/636 = 0.936 | 116/1,412 = 0.082 |

**Qualification.** All 116 successes in distractor worlds (per arm) came from a `use_return` on a record the actor did not create. The actor succeeded by reusing the distractor. For r0 these 116 are exactly a subset of the 194 distractor worlds that contain a distractor that happens to be **valid for the goal**:
- success with a goal-valid distractor: 116;
- failure with a goal-valid distractor: 78;
- failure with no goal-valid distractor: 1,218;
- success with no goal-valid distractor: 0.

Every E17 failure in r0/r1 used a foreign record, as did 1,290 of 1,337 for r2. The reading "fails almost exactly when any distractor exists" holds. The underlying mechanism is stronger than the report says: E17 *always* takes the prior record and succeeds only when that record is accidentally correct.

E17 r0 and r1 have identical outcomes on all 2,048 E19 episodes and all 5,888 e17-sealed episodes, but different action sequences (709 and 917 identical). They are distinct checkpoints that are outcome-equivalent, not a duplicated arm.

### 2. E20 (root e20-sealed; paired against e17-sealed and e19-sealed on identical worlds)

| Group mean success (r0 / r1 / r2) | E17 | E20 |
|---|---|---|
| same_* (8) | 0.3672 / 0.3672 / 0.3472 | **1.000 / 1.000 / 1.000** |
| distr_* (6) | 1.000 / 1.000 / 0.9199 | **1.000 / 1.000 / 1.000** |
| IID pairs (3) | 1.000 / 1.000 / 0.9414 | 1.000 / 1.000 / 1.000 |
| held-out pairs (3) | 1.000 / 1.000 / 0.8997 | 1.000 / 1.000 / 1.000 |
| triples (6) | 1.000 / 1.000 / 0.8366 | **1.000 / 0.9993 / 0.9186** |

- **Rule check:** the E20 − E17 gain on same_* is +0.633 / +0.633 / +0.653. Retention is never negative beyond −0.0007 (r1 triples, one episode).
- **E17-r2 iid_S-R:** 0.824 → 1.000.
- **Residual failures:** the E20 r2 failures are triple_A-R-S 0.852, triple_A-S-R 0.852 and triple_R-A-S 0.809, i.e. the triples where select follows assign.
- **Residual failure anatomy:** all 125 E20-r2 non-hard failures have **37–45** rejected completion actions (claim ≥10; the report's 37–45 is exact).
  - Every rejected action is `commit_pending` (5,268 total; reasons: capacity 3,587, funds 1,681).
  - 0 of 125 failures touch a foreign record.
- The single r1 failure (triple_A-S-R) has 36 rejections.
- **Distractor use:** E20 and E22 learned arms perform **0** `use_return` or `retrieve` actions on foreign records across all same_* episodes.

### 3. E20 training configs and the m2 encoder

**Config diffs:** e17→e20 boot and rl-single, r0–r2, flattened.
- The only differences are `address_namespace`, `policy.feature_version` m1→m2, and `same_type_distractors: 2` added to all 9 `world_mix` entries.
- For rl-single, `initial_checkpoints` also differ: each points at its own e20-boot checkpoint.
- Initialization seeds, population seed, training and development seed starts, hyperparameters and methods are identical.

**m2 encoder:** `encode_action_m2` reads the record's `problem` and `problem_snapshot` and `o.problems`. AST check: the string constant `"prior"` and any private (`_`) attribute appear in neither `encode_action` (m1) nor `encode_action_m2`. `observe()` strips only `payload` from records, so `prior=True` *is* visible in the observation but is not read by the encoder.

### 4. E21 (e09v2-sealed-e21 vs e09v2-sealed-e15)

| | robust r0 | r1 | r2 | plain r0 | r1 | r2 |
|---|---|---|---|---|---|---|
| IID utility (8 iid_*, equal weight) | **0.8219** | **0.9278** | **0.8881** | 0.9330 | 0.9356 | 0.9294 |
| Greedy-first, int_4x4_control | **0.246** | **0.008** | **0.117** | 1.000 | 0.789 | 0.000 |
| Solver calls per control episode | 1.59 | 1.27 | 1.68 | 0.35 | 0.64 | 1.39 |
| No-tools success | **0.516** | **0.000** | **0.008** | 0.723 | 0.492 | 0.000 |
| Transfer utility (10 xfer_*) | 0.710 | 0.838 | 0.829 | 0.894 | 0.852 | 0.818 |

Greedy-first is the first of {choose_item, commit_pending} vs {call, start_subset}; all 256 control episodes contain one. All claimed numbers match.

**Configs:** e15-halving-plain → e21-halving-robust differs only in `development_world_mix`. That key is absent in E15, and in E21 it has 10 entries identical to E13's (including the `work_limit: 0` no-tools and 5×5 entries). `world_mix` and all other keys are identical.

**Minor qualification:** "transfer drops" holds on the mean (0.792 vs 0.855), but r2 transfer rises in the pairing (0.829 vs 0.818).

### 5. E22 (roots e20-sealed, e22-sealed; 29 non-hard conditions × 256 per arm)

| | r0 | r1 | r2 |
|---|---|---|---|
| E20 failures (perseveration ≥10 rejected completion) | 0 (0) | 1 (1) | 125 (125) |
| E22 failures (perseveration) | 3 (0) | 0 (0) | 3 (0) |
| E22 triples | 0.9993 | 1.000 | 0.9993 |

**Configs:** e20→e22 (boot and rl-single, r0–r2) differ only by `address_namespace`, `feature_version` m2→m3, and the rl `initial_checkpoints` (own e22-boot).

**m3 counters, from the code** (`ModularWorkshop.step`):
- `_stage_rejections` increments when the action kind is commit_pending, commit_assignment, use_return, deliver or move and the returned feedback status is rejected or invalid_input.
- `_stage_calls` increments on every `call`.
- Both reset to 0 exactly when `len(self._completed)` changes, i.e. on stage completion.

Every increment is conditioned on the actor's own action and the feedback returned to it, so there is no hidden input. The encoders (`encode_observation_m3`, `encode_action_m3`) read only `o.stage_rejections` and `o.stage_solver_calls`.

**Raw check:** I recomputed both counters from the logged action/feedback sequence (reset on a `stage_completed` feedback). They match the logged pre-action observation on **39,071 of 39,071 steps** in 1,200 E22 episodes that contain at least one rejection.

**Qualifications:**
- (a) The report says "No condition-group mean dropped by more than 0.001". E22 r0 IID pairs dropped **0.0013** (one failure in 768). The registered threshold (0.02) is still met by a wide margin.
- (b) E22's 6 failures are not perseveration by the registered definition, but 5 of 6 are a *different* loop: 9 inspects, then **55 consecutive `choose_item`**, with no completion attempt, until the 64-step cap. The ≥10-rejected-completion metric cannot see this loop because `choose_item` is not a completion action. World 96010180 (same_iid_S-A) fails in both r0 and r2. The counts are small (6 vs 126), so the conclusion stands. The claim "perseveration removed" is metric-specific.

### 6. Sealed-seed disjointness

For each of 21 runs (e17-, e20-, e22-boot/rl-single r0–2 and e21-halving-robust r0–2), I compared:
- the training seed intervals recorded in `state.json` (`allocations[].training_seed_interval`);
- the development seeds read from the actual development row files (first and last file per run; the sets are equal).

**Ranges found:**
- training at 1.000e9–1.080e9+14,400 (modular) and 700M–745M (E21);
- development at 1.090e9–1.097e9 (+127) and 595M–597M.

**No overlap** with sealed spans 90.00M–96.07M+256 (E19/E20/E22) or 70.0M–71.7M / 75.0M–75.1M+256 (E21).

**Caveat:** training coverage comes from recorded intervals, not logged training worlds.

## Concerns

**C1. Same-type distractors are not all "unrelated instances".**
- Every constrained_subset distractor (755/755) is built over *this episode's item set* with no constraints. The csp distractors draw random values.
- 214 of 2,094 same-type distractors (10.2%) are **valid solutions of the goal**. 194 of 1,412 distractor worlds (13.7%) contain one: 59/256 in same_iid_S, and ~33 per condition for the select pairs.
- This inflates E17's E19 success (all 116 distractor-world successes per arm) and makes "right type, wrong provenance" partly "right answer, wrong provenance". It does not affect E20, which never touches foreign records.

**C2. m2 is close to a supplied provenance oracle.**
- Distractor records carry `problem="prior_i"`. Actor drafts are always named `problem_{n}` (action catalog).
- The m2 bit "record's problem is one of this episode's drafts" is therefore exactly "not a prior record" for computation records.
- The bit is public and legitimately derivable, and the report says it is supplied. But E20's 1.000 is expected once this bit exists and training worlds contain the same distractor distribution (sealed same_* conditions use the same generator as training, so this is IID robustness).
- The design confounds input and data (as registered). The interpretation "provenance binding *required* a provenance input" rests on code inspection: m1 record-candidate features cannot separate own and prior same-type records. No m1 + same-type-training arm was run.

**C3. The E22 perseveration metric is definition-bound.** The residual E22 failures are `choose_item` loops that the metric does not count (§5b). The memory-gap interpretation fits `commit_pending` loops only.

**C4. Minor wording issues** (§4 transfer r2; §5a 0.0013 > 0.001). There is also a carried-over disclosure: RL lineage 0 and boot lineage 2 share training seeds starting at 1.04e9 in E20 and E22 as in E17. This does not affect the sealed evaluation.

No numerical claim in the E19/E20, E21 or E22 sections was refuted.
