# P1 independent audit (raw-row reconstruction)

Auditor: independent subagent. It did not write the evaluator changes, the metric definitions or `campaign03_p1_analysis.py`.

Branch: `campaign/e03-p1-audit`, from `campaign/extended-03` e4d0801f.

Data: `/home/brand/tensegra-campaign03/results/p1-sealed`, copied read-only. Config sha256 `f31d4c74…8b77`, which matches both the file and `summary.json`.

Numbers: [p1-independent-audit.json](p1-independent-audit.json).

Tool: `research/tools/campaign03_p1_audit_reconstruct.py`. It uses the standard library only and was written from `protocol-P1-metrics.md` before the root's analysis tool was opened. The root tool was read afterwards, only to explain the two secondary discrepancies in §1.

```
python research/tools/campaign03_p1_audit_reconstruct.py <p1-sealed> --output recon.json --details details.json --workers 4
```

## Summary

- **All five verdicts reproduce exactly.** Every per-lineage value behind them also matches, as do all primary per-condition metrics (29 policies × 8 conditions). No primary or rule value differs from the root's by more than 1e-12.
- **Two secondaries differ, and neither feeds a rule.** The root's "truncation" column is also uninformative.
- **Integrity is clean.**
- **Four claims need weakening or correction** (§7):
  1. The X2 "emergent reuse" framing: the recompute-boot already reuses, mainly foreign records.
  2. R4: it is supported only through collapsed or degenerate comparators, and its r2 pair rests on a single world.
  3. R5: lineage r0 "passes" only because its IID baseline collapsed.
  4. The "truncation 0.000" reading.

## 1. Reconstruction vs the root's analysis

| Rule | Auditor | Root | Per-condition readings that differ (auditor = root) |
|---|---|---|---|
| R1 | not supported (0/3 success; mean util 0.231 vs 0.903−0.02) | not supported | none |
| R2 | not supported (1/3: r2 only) | not supported | none |
| R3 | SUPPORTED (3/3) | SUPPORTED | none |
| R4 | SUPPORTED (retry clause 3/3; success gap −0.331) | SUPPORTED | iid_f0: no |
| R5 | not supported (1/3) | not supported | none |

**Per-lineage values, all identical to the root's:**

- **R1.** IID success is .092, .938 and .000. IID utility is −.024, .860 and −.144.
- **R2.** The reuse gain is −.183, −.480 and +.539. Work per success goes 147.2→142.5, 148.9→181.7 and 161.6→117.4. IID success goes .955→.705, .947→.945 and .951→.945.
- **R3.** The invalid-use difference is .413, .433 and .456. The stale-use difference is .313, .342 and .311.
- **R4.** The X4 identical-retry rate is .527, .426 and .974; X1's is .000, .064 and .000. The X1−X4 success gap per pair is −.439, +.375 and −.927.
- **R5.** The held-out ratio is 2.043, .871 and n/a.

**Metric cells compared:** 3,944, covering 13 rates and 4 support counts for every arm and condition. Success, utility, work/success, correct-reuse rate, invalid rate, stale rate, identical retry (primary, per-attempt and all-kinds), revision quality, over-revision rate, opportunities, audited uses, rejections and event episodes all match to within 1e-9.

**Discrepancies greater than 0.001 (secondaries only):**

1. **`correct_reuse_rate_applicable_pre`, 88 cells** (for example x1-boot-r1 on heldout: auditor .652, root .745).
   - The root counts a window as correct-pre if it contains *any* use_return of an applicable pre-existing handle before a call, **whether or not the use succeeded**.
   - The auditor keeps step 5's "the use succeeds" requirement. The metrics doc says the secondary only drops (d).
   - The root's secondary is therefore slightly inflated. It enters no rule.
2. **`identical_retry_rate_no_missing_dependency`, 22 cells**, all X3 or X4 (for example x3-boot-r1 on iid_f2: auditor 579/858 = .675, root 587/858 = .684).
   - The root removes missing_dependency *attempts* from the sequence before it searches for the next same-key attempt. A non-md rejection whose next same-key attempt was an md rejection with different dependencies is then matched to a later attempt.
   - The auditor drops only md rejections from the denominator.
   - The registered text ("excluding rejections with reason missing_dependency") allows either reading. This is the F1 sensitivity secondary and enters no rule.
3. **The "truncation" column is 0.000 in every cell, but this is not a measurement.**
   - `row.truncated` is never set: 0 of 59,392 rows. The decision cap equals the world step limit, so the world ends the episode instead.
   - Failed episodes that stop at step 96 are common. Examples: X1-rl-r2 on iid_f2, 256/256; X1-rl-r0, 209/256; X3 bootstraps, about 85/256.
   - Success is unaffected, because such episodes count as failures either way. Any text that reads "no truncation" should be removed.

## 2. Spot checks, verified by hand from `history` and `trace`

The checks were made independently of `p1_audit` where possible. 60 random (condition, policy) files were drawn with seed 20260926, and one counted event was taken from each.

**Correct reuse: 59/59 verified.** For each sample, from the trace observation before the step:

- the action is `use_return(h, as=need)` and it succeeded;
- *h* is a success record of the needed primitive, with `created_step` ≤ the stage-open step;
- the stage open was **recomputed independently** from history and public events, and it equals the audit's `open` in 59/59;
- the history between the open step and the use contains **no call** on a problem of the needed primitive.

Sources in the sample: 49 were foreign records (created_step 0); 10 were own older records.

**Identical retry: 38/38 verified.**

- Each is the same action (kind and arguments), and the first attempt was rejected.
- No identical action occurs between the two attempts.
- The public dependency state is equal at both attempts: requirement_versions, selection_id, assignment_id, position, travel, and whether the handle was retrieved.
- In 0/38 did requirement inspection happen in between, so there is no F1 case in the sample.
- All 38 were use_return rejections: category_coverage, already_committed, deadline, missing_dependency, retrieve_required, not_applicable, stale_dependency and capacity.

## 3. The X2-bootstrap reuse anomaly: real reuse, not a metric artifact, but not learned from reward

Every counted correct reuse of the six X2 endpoints was classified by source. Per condition, the counts below are correct / opportunities.

| endpoint | iid_f0 (own records only) | iid_f2 | foreign4 | heldout | source of correct reuses (all conditions) |
|---|---|---|---|---|---|
| x2-boot-r0 | 13/19 | 167/174 | 280/289 | 189/359 | 1,218 foreign, 101 own-older |
| x2-boot-r1 | 14/21 | 167/172 | 274/289 | 193/193 | 1,215 foreign, 88 own-older |
| x2-boot-r2 | 1/29 | 152/184 | 265/297 | 192/294 | 1,209 foreign, 35 own-older |
| dep_recompute (reference) | 0/21 | 0/177 | 0/291 | 0/198 | none |

- **Foreign records.** In the foreign-record conditions, the X2 bootstraps take the teacher-free shortcut almost every time: retrieve a foreign record and `use_return` it without calling a solver. This happens at the select and assign stages in nearly every opportunity (iid_f2: 49/49 assign in all three lineages; 56/56 select for r0 and r1, 56/66 for r2) and for most route opportunities.
  - These uses are hidden-applicable and completable, and they succeed. Invalid-use rate 0.000 and stale-use rate 0.000 for all X2 bootstraps.
  - **Example:** x2-boot-r0, iid_f2, seed 110100001. Steps 1–11 inspect, step 12 retrieves foreign record `receb93665434736a`, and step 13 does `use_return(as=select)`. `dep_recompute` would instead do `start_subset` at step 12.
  - **Example:** seed 110100000. The foreign route `ra1e2c36efef97ff5` (created_step 0) is retrieved at step 25, tried as a route out of order at step 26 (rejected, missing_dependency), and used successfully as the route at step 40. No shortest_path call is made in the episode.
- **Own older records.** These are always route records reused after an upstream re-assignment. Examples:
  - seed 110000014 (iid_f0): the route was computed at step 26 after an edge_closed event at step 18. The assignment was then redone by `use_return(as=assign)` at step 35, which opens a new route window. The step-26 route, still valid, was reused at step 36.
  - seed 110000025: a slot_closed event revoked the assignment.

  `dep_reuse` does exactly this: 21/21 of iid_f0's opportunities are this pattern. It is legitimate reuse under the registered definition.
- **No artifact pattern was found.** There were 0 cases of a record "pre-existing" only because it was computed out of order, while a different stage was needed. There were 0 correct reuses with an earlier same-primitive call in the window. Each classification was confirmed against the independently recomputed stage open.
- **Consequence for R2.** The protocol premise "the X2 teacher never reuses, so any reuse in X2 is learned from reward" is **false for the endpoints produced**. The recompute-imitating bootstraps already reuse at .82 (r0 and r1) and .43 (r2) on the IID group.
  - The mechanism is imitation generalization: the student picks retrieve/use_return on foreign applicable records where the teacher would start a draft. No reward is involved.
  - The R2 comparison therefore does not measure "emergence from zero". Two lineages *lose* reuse under RL:
    - X2-rl-r1 stops reusing foreign **routes** (iid_f2: 49 → 2 route reuses) and recomputes them. Work per success rises 148.9 → 181.7.
    - X2-rl-r0 degrades: success .955 → .705, and fewer episodes reach the route stage.
  - Only r2 gains (+.539). This is mostly own-route reuse after re-assignment (iid_f0: 1/29 → 17/18) plus the remaining foreign routes.
  - R2 "not supported" stands, but its interpretation must say that the baseline was not reuse-free.

## 4. RL collapse characterization (descriptive)

The table pools all 8 sealed conditions: 2,048 episodes per endpoint. "Cap" means the episode failed and stopped at the 96-step world limit.

| endpoint | success | cap | mean steps | dominant actions | where failures stall |
|---|---|---|---|---|---|
| x1-boot-r0 | .941 | .053 | 31.4 | inspect .36, retrieve .17 | select committed |
| **x1-rl-r0** | .180 | **.820** | 83.1 | **add_constraint .79** | 1,638 with selection committed, assign never committed |
| x1-boot-r2 | .946 | .037 | 30.6 | inspect .37, retrieve .16 | — |
| **x1-rl-r2** | .000 | **1.000** | 96.0 | **retrieve .74** | 1,961 with assignment committed, route never applied |
| x2-boot-r0 | .856 | .142 | 38.5 | inspect .30, retrieve .28 | — |
| x2-rl-r0 | .564 | .436 | 57.6 | inspect .48, retrieve .36 | 513 with assignment committed; 299 never commit a selection |
| x4-boot-r0 | .845 | .155 | 37.8 | inspect .30, use_return .21 | — |
| x4-rl-r0 | .619 | .381 | 53.8 | use_return .29, retrieve .27, uncommit .10 | commit/uncommit churn; 307 never commit a selection |
| x4-boot-r1 | .926 | .071 | 32.2 | inspect .35 | — |
| x4-rl-r1 | .527 | .473 | 59.9 | **add_constraint .64** | 857 with selection committed |
| x1-rl-r1 (reference) | .902 | .098 | 36.6 | inspect .31 | — |

**The failure mode is a loop of *successful*, state-preserving actions that runs until the step cap. It is not abstention, rejection or wrong commits:**

- **X1-rl-r0** and **X4-rl-r1** get stuck in the csp draft of the assign stage. They add constraints forever:
  - X1-rl-r0 alternates `add_constraint(finish_by, bound=8)` and `(… bound=9)` until step 96 (iid_f2, seed 110100000);
  - X4-rl-r1 repeats `add_constraint(conflicts, problem_3)`.

  Neither ever calls csp.
- **X1-rl-r2** commits the selection and the assignment, then `retrieve`s the same route record until step 96: 72% of its failed-episode steps are (need=route, retrieve). It never applies a route and never verifies successfully (verify returns "incomplete" 109 times). Zero abstains.
- **X2-rl-r0** loops on `retrieve` at the select stage (42% of failed-episode steps) or on `inspect` at the route stage (37%).
- **X4-rl-r0** mixes retrieve loops with commit/uncommit churn and long runs of rejected use_returns (up to 78× `missing_dependency` or `retrieve_required` in single worlds).
- **The median number of repeats of a single action key** in a failed episode is 38 (X1-rl-r0), 71 (X1-rl-r2), 80 (X2-rl-r0), 81 (X4-rl-r0) and 77 (X4-rl-r1).
- **These loops are invisible to the identical-retry metric**, which counts only *rejected* attempts. Collapsed X1 RL scores identical retry 0.000 while looping on one action for 70+ steps.
- **Development curves.** Recomputed from `development/round-*-attempt-*.jsonl.gz`, 128 dev worlds per attempt; they are identical to `state.json` allocation successes.
  - X1-r0: .95 through attempt 20, then .88–.84 (attempts 21–25), then .57, .21, .10, .10.
  - X1-r2: .95 through attempt 24, then .92, .61, .13, .00, .00.
  - X4-r0: .95 → .84 from attempt 13, .23 at attempt 22, back to .74.
  - X4-r1: .93 until attempt 27, then .48 and .54.
  - X2-r0: drifts to .77, ending at .70.
  - Stable: X1-r1, X2-r1/r2, X3 (all, .67–.86) and X4-r2.
  - **Step-cap failures follow the curves** (X1-r2: .05 → 1.00). The last-attempt dev action mix already shows the sealed signature: X1-r0 add_constraint .81, X1-r2 retrieve .73, X4-r1 add_constraint .64.
  - **The collapse is visible in monitoring data before the sealed run.** The root's summary is accurate, except that X1-r0 declined to .84 from attempt 21, not only in attempts 26–29.

## 5. Confound readings for R3/R4 (NOT registered, descriptive)

| reading | R3 | R4 retry clause (pairs passing; X4 − X1 per lineage) | R4 success clause (lineage-mean X1−X4) |
|---|---|---|---|
| registered (RL vs RL) | 3/3; invalid diff .41/.43/.46 | 3/3; +.527/+.362/+.974 | −.331 (no) |
| (a) bootstrap endpoints (X3/X4-boot vs X1-boot) | 3/3; invalid diff .446/.608/.262; X1-boot invalid .000 | 2/3; +.268/+.263/**−.083** (X1-boot retries .66/.57/.73) | +.037 (yes; mostly X4-boot-r0 heldout .195) |
| (b) r1 only, RL | pass; invalid .447 vs .014, stale .342 vs .000 | pass; .426 vs .064 | +.375 (yes) |
| (b′) r1 only, boot | pass; .608 vs .000 | pass; .833 vs .570 | +.009 (no) |

- **R3 is robust.** X1 makes essentially no invalid or stale uses at any endpoint (≤ .03 in any single condition), and X3 makes 26–61% at both endpoints and in every lineage. The collapse of X1 RL r0/r2 does not produce R3. X1-rl-r0/r2 still make 266–598 audited uses per condition, all valid.
- **R4 is fragile.**
  - **Pair r2 rests on one world.** X4-rl-r2 has rejections in only one condition (noevent_f2) and in only one world there: seed 110200178, 38 rejected `use_return`s with reason deadline, 37 of them identical retries. X1-rl-r2 hits the same world with 2 rejections and 0 retries. `x4-boot-r0` and `x4-boot-r1` have the same single-world loop (77 rejections). The pair "passes" at +.974. On an episode basis, X4-rl-r2 retries identically in 0.05% of episodes vs 0.83% for X1-rl-r2, the opposite direction.
  - **Pair r0 compares X4-rl-r0 with a collapsed X1-rl-r0.** X1-rl-r0 makes only 94 rejections across 2,048 episodes, because it rarely reaches a commit, and it retries none of them. X4-rl-r0 identically retries in 54% of episodes.
  - **Only pair r1 is a clean comparison.** It passes by rejection rate (.426 vs .064) and by episode (28.0% vs 7.0% of episodes with an identical retry).
  - **At bootstrap the clause passes 2/3 by rejection rate.** But X1-boot, which has attempt memory, already retries 57–73% of its rejections identically. By episode, X4-boot exceeds X1-boot in r0 only (5.4% vs 2.3%); r1 is 0.8% vs 1.5% and r2 0.7% vs 2.6%.
  - Retries are highly concentrated: in X4-boot-r1, 95% of identical retries fall in 5 episodes.

### Bootstrap-endpoint readings of R1 and R5 (NOT registered, descriptive)

The registered rules read the RL endpoints. Read at the X1 **bootstrap** endpoints instead:

- **R1 would be supported.**
  - IID success is .961, .955 and .961, so 3/3 lineages reach ≥ .95.
  - The lineage-mean IID utility is .8834, against a threshold of .9026 − .02 = .8826. The margin is only +.0008.
- **R5 would be supported (3/3).** The held-out ratios are .927, .937 and .951.

The negative R1 and R5 verdicts are therefore a property of the late RL collapse under the latest-only endpoint rule. Imitation acquisition alone reaches them, with R1's utility clause passing by a very thin margin. This does not change the registered verdicts, but any claim like "the controller did not acquire the task" should be worded as "RL degraded an acquired policy in 2/3 lineages".

## 6. Integrity

All checks pass.

- **Sealed seeds.**
  - Each condition's `worlds.jsonl.gz` has 256 worlds with seeds exactly 110,000,000 + 100,000·i + [0, 255]; the full range is 110,000,000–110,700,255.
  - All 29 policy files in each condition have the identical (seed, spec_hash, semantic_spec_hash) sequence as the worlds file: 8 × 29 checks, all identical.
- **Disjointness.**
  - Bootstrap training seeds used: 3.000e9, 3.020e9 and 3.040e9 (+0 to +4,800), for r0/r1/r2.
  - RL training seeds used: 3.400e9, 3.420e9 and 3.440e9 (+0 to +14,400).
  - Dev seeds: 3.900e9, 3.901e9 and 3.902e9 (+0 to +127).
  - Initialization seeds: 300000–5, 300100–5 and 300200–5.
  - None overlaps the sealed range, and none overlaps across lineage indices.
  - Within a lineage index, X1–X4 share seeds, as the freeze pairing amendment specifies. Boot and RL of a lineage share the 128 dev worlds, which is monitoring only.
- **Checkpoints.**
  - For all 24 endpoints, the sha256 of the file named in `p1-sealed.json` = the config sha = the sha of the run's latest checkpoint. The latest is round-0-slot-5-attempt-5 for boot (6 checkpoints) and round-4-slot-5-attempt-29 for RL (30 checkpoints).
  - Each sha also equals the `summary.json` binding and the `p1-sealed-checkpoints.json` entry.
  - For all 12 RL configs, every `initial_checkpoints` entry equals the `p1-banks.json` path and sha, and the bank files hash to those values.
- **Feature versions and teachers**, from the train config, the sealed binding and the summary's `policy_config`:
  - X1: d1, dep_reuse.
  - X2: d1, dep_recompute.
  - X3: d1-noapp, dep_reuse.
  - X4: d1-noattempt, dep_reuse.

  All 24 endpoints are consistent with the protocol.
- **Receipts.** All 12 bootstrap, 12 RL and 1 sealed `occupancy.json` show `exit_code` 0. The smoke-boot-envfail receipt shows exit 1, as documented in decisions.md.

## 7. Claims that must be weakened or corrected

1. **X2 / R2.** Do not describe X2 as "a policy whose reuse can only come from reward".
   - The recompute-boot already reuses: .82/.82/.43 on the IID group, and up to .97 per condition. The source is overwhelmingly **foreign** records, taken up by imitation generalization.
   - R2 "not supported" should be reported as "RL did not increase an already-high reuse rate in 2/3 lineages (it decreased it)", not as "reuse did not emerge".
   - The one passing lineage (r2) gains mainly through own-route reuse after re-assignment.
2. **R4 "SUPPORTED" must be qualified.**
   - The retry clause reaches 3/3 only because pair r0 compares against a collapsed X1 that rarely commits, and pair r2 rests on a single sealed world.
   - Only r1 is a clean pair, so the registered 2/3 bar is not met on clean pairs.
   - At bootstrap, the retry clause passes 2/3 by rejection rate but 1/3 by episode.
   - Recommended wording: "supported by the registered point estimate; driven by lineages whose X1 comparator collapsed or by one world; clean evidence in lineage r1 only". F1 remains a caveat, though no F1 case appeared in the 38 hand-checked retries.
3. **R3 SUPPORTED stands.** It holds at both endpoints, in every lineage, and in r1 alone. Mention F4, per the protocol.
4. **R5.** Lineage r0's pass (ratio 2.04) is an artifact of a collapsed IID baseline (.092). It should be reported as degenerate, not as transfer. The verdict (not supported) is unchanged.
5. **R1** "not supported" is driven by the late RL collapse (r0 and r2). X1-rl-r1 (.938) is close to the 0.95 bar, and read at the bootstrap endpoints R1 would pass (3/3 success; utility margin +.0008) and so would R5 (see §5). The collapse is visible in the dev curves, and the latest-only endpoint rule captured it as registered.
6. **"Truncation 0.000"** in the root tables is not a measurement and should not be cited. Up to 100% of an endpoint's episodes end at the 96-step limit.
7. **Secondary definitions.** The root's `correct_reuse_rate_applicable_pre` counts unsuccessful uses, and its `no_missing_dependency` retry variant rematches sequences. Report both as implementation choices or align them. No verdict depends on either.
