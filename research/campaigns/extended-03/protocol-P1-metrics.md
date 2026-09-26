# P1 metric definitions (registered before any P1 result)

Status: operational definitions for [protocol-P1.md](protocol-P1.md), registered on `campaign/e03-p1-analysis` while the 12 P1 bootstraps were training. No P1 training, development or sealed output had been read when this was written. The rules R1–R5 and their thresholds are unchanged. This document fixes only how each quantity is computed from raw rows. Items marked **[registered operationalization]** are choices where the protocol text was ambiguous. They are fixed here, before results.

Implementation:
- `src/tensegra/campaign03_p1_audit.py` is the evaluator-only step audit. It is logged in `outcome["p1_audit"]`.
- `research/tools/campaign02_evaluate.py` runs the sealed evaluation and writes raw rows.
- `research/tools/campaign03_p1_analysis.py` computes the metrics and verdicts.
- `tests/test_campaign03_p1_analysis.py` holds the tests.

If this text and the code disagree, the code at the registration commit is what was registered, and the discrepancy is reported.

## 1. Raw data

- **One row per (arm, condition, world).** `<eval>/<condition>/<arm>.jsonl.gz` holds one row per sealed world, with the world spec in `worlds.jsonl.gz`.
- **Arms:**
  - the 24 learned endpoints: the bootstrap and the RL endpoint of X1–X4 × lineage r0–r2;
  - the five references `dep_greedy`, `dep_recompute`, `dep_reuse`, `dep_naive_reuse` and `dep_reuse_norevise`.
- **Pairing.** All arms in a condition run on the same worlds and the same address seeds, so every comparison is paired by world.
- **Sealed run.** Every learned endpoint is evaluated once, in a single sealed run. Worlds start at seed 110,000,000 + 100,000·i, with 256 worlds per condition.
- **Checkpoint identity.** Each checkpoint binding carries `endpoint` ∈ {boot, rl}, `arm`, `lineage` and `feature_version`. These are copied to `summary.json → results[].checkpoint_binding`. The fallback is the name pattern `p1-{endpoint}-{arm}-r{r}`.
- **What each row holds:**
  - the step history (action kind and arguments, feedback status and reason, requirement version, cumulative work);
  - `reuse_audit`: the environment's hidden applicability audit of every use_return that reached payload application;
  - `p1_audit`, a per-step record written **before** the action is applied. It contains:
    - the public need;
    - the stage-open step;
    - the lists of applicable and reusable records;
    - for a use_return: the target's public and hidden applicability, creation step and completability;
    - for a call: the called primitive;
    - the event audit;
    - the public attempt log, with `relevant_dependencies` snapshots.
- **The audit changes nothing in the episode.** A test checks that the audited world reproduces `DepWorkshop` exactly: same history, same outcome.

## 2. Aggregation conventions (all metrics)

- **Condition-level value, per arm:** a pooled ratio of sums over the condition's 256 episodes, Σ numerator / Σ denominator.
  - Means such as success and utility are pooled ratios with denominator = episodes.
  - A rate whose denominator is 0 in a condition is **undefined** (n/a) there. It is not 0.
- **Group value:** the equal-weighted mean of the condition-level values over the group's conditions (protocol: "equal-weighted over the conditions of a group").
  - Undefined conditions are dropped from the mean.
  - When two arms are compared (R2, R3, R4), the mean uses only the conditions defined for **both** arms. **[registered operationalization]**
- **Per-lineage values** are always reported. "≥ 2/3 lineages" means at least 2 of the 3 lineage indices.
  - "Lineage pairs" are X*k*-r*i* vs X1-r*i* at the same index *i*, per the freeze pairing amendment.
- **Endpoints.** R1, R3, R4 and R5 read the **RL endpoints** (the final frozen endpoints). R2 reads the X2 bootstrap and RL endpoints. Bootstrap endpoints of the other arms are reported descriptively.
- **Per-condition readings.** Every rule is recomputed with its group replaced by each single condition of the group. Readings that differ from the aggregate verdict are listed next to it (protocol: "both the aggregate and the per-condition reading where they differ").
- **Uncertainty.** A 95% percentile bootstrap, with 1,000 resamples and seed 20260926:
  - worlds are resampled with replacement within each condition;
  - the same index draw is applied to both arms of a comparison (paired);
  - it is reported for the R2 reuse gain and the R3/R4 differences, per lineage.
  - **Verdicts use the point estimates and the protocol thresholds; the intervals are descriptive.** **[registered operationalization]**

## 3. Metric definitions

### Success, utility, cost, work per success
- **Success:** `outcome.verified_success` (verify accepted). Truncated episodes count as failures. The learned decision cap is 96, equal to the world step limit.
- **Utility:** `outcome.utility` = success − cost.
- **Cost components** (per-episode means, each reported), using each world's own prices:
  - action = steps × action_price;
  - observation = observations × observation_price;
  - travel = travel_distance × travel_price;
  - work = work_units × work_price;
  - compute = compute_units × compute_price.
- **Compute units.** Learned arms are charged 1 compute unit per decision (neural_work_per_forward = 1.0). References are charged `reference_compute_tariff` = 1.0 per decision.
- **Work per success** = Σ solver work_units over **all** episodes (failures included) / Σ successes, per condition. It is undefined when there are no successes.

### Correct-reuse rate (R2 primary)
This is a per-*opportunity* rate. The unit is a **stage window**, not a single decision, so long dithering does not weight an opportunity more. **[registered operationalization]**

1. **Public need at a decision.** Computed from the public state before the action:
   - `select` if no selection is committed;
   - otherwise `assign` if no assignment is committed;
   - otherwise `route` if the position is not the destination;
   - otherwise none.

   The primitive for each need is constrained_subset, csp and shortest_path respectively.
2. **Stage window.** A window is the set of decisions sharing (need, stage-open step).
   - The stage-open step is `campaign03_depworld._stage_open(observation, primitive)`, the teacher's own freshness boundary. It advances on successful uncommits, upstream commits, revocations and events affecting the stage's requirements.
   - A revocation or an event therefore opens a new window.
3. **Reusable record at a decision.** A record *r* of the needed primitive is reusable if all of the following hold:
   - (a) status `success`;
   - (b) it is **applicable by the public teacher rule** `applicable(observation, r, primitive)`: type, request match under its own options, dependency match, usable status and certificate;
   - (c) it is **pre-existing**: `created_step ≤ stage-open step`. It was not produced by a solver call in the current window. This is the same boundary `dep_recompute` uses to decide what is "fresh". Old own records and foreign records both qualify.
   - (d) it **would complete** (evaluator-only hidden check): applying it now would be accepted by the environment and leave the episode completable.
     - select: some slot assignment of the selection meets the deadline with a shortest route from here;
     - assign: finish time + travel + shortest remaining route ≤ deadline;
     - route: the path is accepted and ends at the destination.

   Retrieval is not required, because the agent can retrieve and then use.
4. **Opportunity:** a window in which at least one decision has a non-empty reusable set.
5. **Correct reuse:** an opportunity window in which the agent executes `use_return(h, as=need)` with *h* reusable at that decision, the use succeeds, and **no solver call for the needed primitive has been made earlier in the window**. If it recomputes first and reuses afterwards, that is not correct reuse.
6. **Correct-reuse rate** = Σ correct windows / Σ opportunity windows. It is also reported per stage (select, assign, route) and as opportunities per episode.

**Why (d).** Without it, 30–40% of "applicable" pre-existing records cannot be used at all: for example a foreign csp result that commits but leaves no deadline-feasible route, or an applicable route that misses the deadline. The `dep_reuse` teacher correctly declines these. Calibration on development seeds 2,050,000,000+ (24 worlds per condition, conditions iid_f2, noevent_f2, foreign4 and events_train_kinds_p1), before any P1 output existed:
- `dep_reuse` scores **1.000** in every condition (75/75 opportunity windows);
- `dep_recompute` scores **0.000** (0/75);
- `dep_naive_reuse` falls in between (0.59–0.88).

The variant without (d), "applicable pre-existing", is reported as a secondary (`correct_reuse_rate_applicable_pre`). On it `dep_reuse` scores about 0.7.

**Caveat.** (d) uses hidden state, and so does the primary definition. It is an evaluator label only; no actor sees it.

### Invalid-reuse rate (R3)
- **Denominator:** *audited uses*, the use_return actions that passed the environment's gates (type match, usable status and certificate, retrieved) and so had their payload applied. These are exactly the entries in `outcome.reuse_audit`.
- **Numerator:** audited uses with `applicable_hidden = false`, the environment's hidden applicability at the time of use.
- **Rate** = Σ invalid / Σ audited uses.
- Gated-out use_returns (type_mismatch, not_applicable, retrieve_required) are counted separately as `use_attempts − audited_uses`. They are not in the rate. **[registered operationalization]**
- **Secondaries:** the invalid-use episode rate (fraction of episodes with ≥ 1 invalid use) and the own/foreign split.

### Stale-use rate (R3)
- Audited uses whose record's `depends_on` differs from the current dependency versions (`dependency_match_hidden = false`), divided by audited uses.
- This covers records computed under older requirements or an older selection, whether own (event staleness) or foreign (earlier-shift or version-mismatched).
- Stale uses are a subset of invalid uses; request-only mismatches are invalid but not stale. **[registered operationalization]**

### Revision quality (secondary)
- **Event episodes:** episodes in which the event fired. Every episode has at most one event.
- **At the event,** the audit records:
  - the pieces the event **revoked**;
  - the pieces **kept** (still committed);
  - for each kept piece, whether it can still be completed. This is a hidden check:
    - selection: some assignment under the new closed slots meets the new deadline with a shortest route on the new map;
    - assignment: finish + travel + shortest route ≤ deadline.
- **Invalidated pieces** are the revoked pieces plus the kept pieces that can no longer be completed. **Valid kept pieces** are the rest.
- **Over-revision.** After the event step, the agent does either of the following:
  - it successfully uncommits a valid kept piece (`uncommit select` when the selection is valid, or `uncommit assign` when the assignment is valid);
  - it calls a solver on the primitive of a valid kept piece (constrained_subset while the selection is valid, csp while the assignment is valid).

  Route computations are never over-revision, because the route is never committed.
- **Revision quality** = fraction of event episodes that end in verified success **without** over-revision, i.e. that "recompute only the invalidated pieces" and finish.
- **Components reported:** over-revision rate, post-event success, and post-event work units. **[registered operationalization]**

### Identical-retry rate (R4)
The environment's attempt log records, for every commit_pending, commit_assignment, use_return, move, call, uncommit and verify: `action_key` (a canonical hash of kind + arguments), the outcome status, the reason, and `dependency_versions_at_attempt = relevant_dependencies(observation before the action, action)`. This is the exact quantity the d1 encoder's `attempt.dependencies_changed` bit compares.

- **Primary kinds** are commit_pending, commit_assignment, use_return and move. Per decisions.md F5, this is where X4's residual channels do not leak attempt memory.
- **Rejection** means `outcome_status == "rejected"`.
- **Identically retried rejection:** a rejected attempt *j* is identically retried if the **next** attempt with the same `action_key` has a `dependency_versions_at_attempt` equal to *j*'s. By construction, that next attempt's last outcome was *j*'s rejection.
- **Identical-retry rate (primary)** = Σ identically retried rejections / Σ rejections, over the primary kinds. It is a per-rejection rate: "how often a rejection is blindly repeated". **[registered operationalization]**
- **Secondaries:**
  - (i) per attempt: identical retries / primary-kind attempts;
  - (ii) all attempt kinds, with "failure" = any non-success outcome, including call timeouts. For calls, X4 keeps record-derived call history (F5);
  - (iii) excluding rejections with reason `missing_dependency` (F1 sensitivity);
  - (iv) per kind.
- **F1 caveat.** `relevant_dependencies` does not include whether the requirements were inspected. So a commit_pending rejected for `missing_dependency`, retried after `inspect(requirements)`, counts as an identical retry although the state changed. This was accepted before training (decisions.md F1). The metric keeps the encoder's definition, secondary (iii) shows its effect, and R4 reporting notes it.
- **Calibration.** No reference ever makes an identical retry: the teachers check `_repeat_failure`.

### Transfer ratio (R5)
- The held-out ratio is, per X1 RL lineage, success(heldout_deadline_moved) / IID success, where IID success = the mean of success(iid_f0) and success(iid_f2). It is undefined if IID success is 0.
- The larger-size ratio (larger_s4) is reported as a secondary.

## 4. Rules: exactly what each reads

Sealed condition names (freeze amendment): `iid_f0`, `iid_f2`, `noevent_f2`, `events_train_kinds_p1`, `heldout_deadline_moved`, `foreign4`, `larger_s4`, `work_price_x4`. Groups:

| Group | Conditions |
|---|---|
| IID | iid_f0, iid_f2 (freeze amendment) |
| R3 | foreign4, events_train_kinds_p1, heldout_deadline_moved **[registered operationalization]** |
| all | all eight |

- **R1** (X1 RL; IID group). Lineage *r* passes on success if its IID group success is ≥ 0.95. The utility criterion is: the mean over the three lineages of IID group utility ≥ the IID group utility of `dep_reuse` − 0.02. Supported iff ≥ 2 lineages pass on success **and** the utility criterion holds.
- **R2** (X2 bootstrap vs X2 RL, same lineage; IID group). Lineage *r* passes iff all three hold:
  - (a) correct-reuse rate(RL) − correct-reuse rate(boot) ≥ 0.25, over IID conditions where both are defined;
  - (b) work per success(RL) < work per success(boot). This is the group mean of condition values. An RL value that is undefined fails; a boot value that is undefined while RL is defined passes;
  - (c) IID success(RL) ≥ IID success(boot) − 0.02.

  Supported iff ≥ 2 lineages pass. **[registered operationalization: R2 reads the IID group.]**
- **R3** (X3 RL vs X1 RL, paired by lineage index; R3 group). The pair passes iff the group invalid-reuse rate of X3 minus that of X1 is ≥ 0.05, **or** the same holds for the stale-use rate. Supported iff ≥ 2 pairs pass.
  - A pair with no defined condition (no audited uses in either arm) fails.
  - Read with F4 in mind: X3 keeps payload feasibility facts.
- **R4** (X4 RL vs X1 RL; all eight conditions). Supported iff either:
  - (a) the identical-retry rate of X4 minus that of X1 is ≥ 0.05 in ≥ 2 of 3 lineage pairs; **or**
  - (b) the lineage-mean all-conditions success of X1 minus that of X4 is ≥ 0.02.

  Clause (b) is read at the **lineage mean**, like R1's utility clause. **[registered operationalization]** Per-pair success differences are also reported. F1 and F5 apply.
- **R5** (X1 RL). Lineage passes iff the held-out ratio is ≥ 0.9. Supported iff ≥ 2 lineages pass. It is reported as a secondary.

## 5. Ambiguities resolved here (all before results)

1. **R2 conditions.** The protocol names no group for the reuse rate or work per success. The IID group is used for all three R2 clauses, the same group as "IID success". All eight per-condition readings are reported. iid_f0 has no foreign records, so reuse opportunities there come only from the agent's own older records, and the per-condition readings will show this.
2. **R3 "event conditions".** These are the two p_event = 1 conditions: `events_train_kinds_p1` and `heldout_deadline_moved`. `deadline_moved` bumps no dependency version (deadline is in no primitive's READS), so in that condition the applicability input matters only through its two foreign records. It is included for fidelity to the protocol text. The per-condition readings show the other two conditions separately.
3. **R4 conditions and clause (b).** No group is named. Attempt memory can matter wherever a rejection can occur, so all eight conditions are used. The success clause is read at the lineage mean.
4. **Rate denominators.** Invalid and stale rates are per audited use, identical retry is per rejection, correct reuse is per stage-window opportunity, and each has a per-episode or per-attempt secondary. A rate with no support is undefined, not zero, and is dropped from group means (§2).
5. **"Correct reuse"** requires completability (hidden). Section 3 gives the reason and the calibration.
6. **"Rejection"** is the status `rejected`. For the all-kinds secondary, any non-success counts.
7. **Stale ⊂ invalid.** Stale means dependency-version mismatch, whether own or foreign.

## 6. What the protocol asks for that cannot be measured exactly

- **"Recomputing only the invalidated pieces"** uses a hindsight-feasibility criterion (hidden). A public-information agent may reasonably revise a piece that is still feasible but looks risky. The teacher never does (calibration: 0 over-revisions).
- **F1.** The identical-retry bit ignores requirement inspection, so the identical-retry metric inherits it (§3).
- **R2 when X2 never succeeds.** Work per success is undefined when there are no successes; the handling is registered in §4.

## 7. Commands (root launches; paths on the pro6000)

```bash
# 1. After all 24 endpoints complete: bind every frozen endpoint (verifies each finalist's sha256).
python research/tools/campaign03_p1_configs.py checkpoints --results /home/brand/tensegra-campaign03/results --output configs/campaign03
# 2. The sealed config (protocol seeds 110,000,000+, 256 worlds per condition, the five references).
python research/tools/campaign03_p1_configs.py sealed --checkpoints configs/campaign03/p1-sealed-checkpoints.json \
  --references dep_greedy,dep_recompute,dep_reuse,dep_naive_reuse,dep_reuse_norevise --output configs/campaign03
# 3. One sealed run of all 24 endpoints (under the campaign02_job.py wrapper, like the other jobs).
python research/tools/campaign02_evaluate.py configs/campaign03/p1-sealed.json \
  --output /home/brand/tensegra-campaign03/results/p1-sealed --device cuda --threads 1
# 4. Metrics and verdicts (standard library only).
python research/tools/campaign03_p1_analysis.py /home/brand/tensegra-campaign03/results/p1-sealed \
  --output research/results/campaign-03/p1-analysis
```

A smoke must not touch the sealed worlds. Use `sealed --examples 4 --seed-start 2100000000`, which writes `p1-sealed-nonprotocol-*.json` with its own address namespace.
