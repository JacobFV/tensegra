# Extended-02 phase 3: independent raw-data audit (E15–E18)

Auditor: independent agent, branch `campaign/e02-phase3-audit` (from `campaign/extended-02`), 2026-09-24.
Scope: the phase-3 claims in `campaign-report-phase3.md` and `decisions.md` (phase-3 section), recomputed from the raw episode rows on `gb10-direct:~/topoformer-campaign02/results/`.

## Method

The audit was strictly read-only on GB10: no jobs, no GPU, nothing written there. All code is new, uses only the Python 3 standard library, and does not use `campaign02_e09_analysis.py` or any earlier audit script.

**Scripts**
- `review/audit_phase3_remote.py` streams every `<root>/<condition>/<arm>.jsonl.gz` row and every `worlds.jsonl.gz` in the eight roots: `e09v2-sealed-r0..r2`, `e09v2-sealed-e15`, `e16-sealed`, `e16-sealed-distractors`, `e17-sealed` and `e18-sealed`. It extracts:
  - seed, spec_hash, success, utility and steps for each row;
  - the number of `prior` records in the first observation;
  - `stages_completed`;
  - counts of `use_return` rejected with "return type mismatch": overall, with `as="select"`, and with `as="select"` on a handle whose record primitive is `csp`.

  It also reads all 66 `state.json`/`protocol.json` files (allocations, lineage, finalist), the seeds of every development prediction file, and all `*-process/occupancy.json` receipts.
- `review/audit_phase3_worlds_remote.py` compares E16 worlds with E17/E18 worlds by content. It also counts rejected `commit_pending` actions per failed e17rl-r2 episode.
- `review/audit_phase3.py` runs locally and computes all means, paired differences, decision rules, integrity checks, seed-overlap checks and ledger checks. Its output is `phase3-independent-audit.json`.

**Conventions**
- Condition means are unweighted means over the 256 worlds of each condition.
- Group means weight each condition equally.
- Paired differences are computed per world, matched on seed and required to have identical spec_hash. The CI is a normal approximation over pooled worlds, not a bootstrap.

**Measured audit CPU**
- Remote extractor: 461.4 CPU-s (462.6 s wall).
- Worlds/commit checker: 4.5 CPU-s, run twice.
- Local analysis: 0.4 CPU-s.
- A few small inspection probes: about 5 CPU-s.
- **Total ≈ 475 core-s (0.13 core-h) of CPU.** No GPU. This is not yet in `budget.json`.

## Verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | E15 numbers and halving mechanics | **CONFIRMED** (one wording qualification) |
| 2 | E16 success, localization (12,195), rule "supported" 2/3 | **QUALIFIED**: numbers exact; the rule passes only on the aggregate reading |
| 3 | E16 distractor diagnostic = zero-distractor fraction | **CONFIRMED** |
| 4 | E17 numbers, rule passes | **QUALIFIED**: passes on the mean reading, fails a per-lineage IID reading; r2 localization details **REFUTED** in magnitude |
| 5 | E18 means and PARTIAL | **CONFIRMED** |
| 6 | Paired-world integrity | **QUALIFIED**: seeds and world content identical, but spec_hash differs E16 vs E17/E18 |
| 7 | Sealed-seed disjointness | **CONFIRMED** |
| 8 | Held-out integrity of training mixes | **CONFIRMED** |
| 9 | Ledger | **CONFIRMED** |

### 1. E15: CONFIRMED

**Sealed IID utility** (8 IID conditions, equal-weighted):

| Arm | r0 | r1 | r2 |
|---|---:|---:|---:|
| Halving-plain | .9330 | .9356 | .9294 |
| Halving-niche | .9330 | .9284 | .9294 |
| PBT | .9289 | .8959 | .9284 |
| Multistart | .9282 | .9292 | .9251 |
| Single | .9430 | .9430 | .9302 |

Plain transfer utility is .8937/.8522/.8176, matching the report.

**Paired differences** (plain halving minus the comparator):

| Comparator | IID per replicate | IID pooled [95% CI] | Transfer pooled |
|---|---|---|---|
| PBT | +.0042 / +.0396 / +.0010 | +.0149 [+.0120, +.0178] | +.0519 |
| Multistart | +.0048 / +.0064 / +.0043 | +.0052 [+.0029, +.0075] | +.0149 |
| Single | −.0099 / −.0074 / −.0007 | −.0060 | −.0324 |

Niche minus plain: 0 / −.0072 / 0; pooled −.0024; transfer −.0118. All of these match the report.

The pooled CIs agree with the reported bootstrap intervals to within about 0.0005.

**Qualification.** The report's "beats PBT in 3/3 replicates" is true for point estimates only. The per-replicate paired CI for r0 includes zero: +.0042 [−.0005, +.0088]. The r2 margin is +.0010 [+.0004, +.0016].

**Mechanics** (all 6 runs, reconstructed from `state.json` allocations):
- Members per round were 6, 3, 2, 2, 2.
- Each survivor received 6/k consecutive slots, and every consecutive slot added +60 cumulative updates.
- Each member's training-seed stream is contiguous across rounds.
- Every finalist has 1,320 − 600 (bootstrap import) = **720 RL updates**.

**Cuts.** I recomputed every cut independently:
- **Plain arm:** the top k by last-slot development utility.
- **Niche arm, k ≥ 2:** the best member with greedy-first ≥ 0.5 and the best with < 0.5, then fill by utility.

All 24 cuts match the kept sets recorded in the state files and the lineage `halving` events. The finalist is the best final-round member by utility in 6/6 runs. The finalist is member 0 in **5/6** runs; the exception is niche-r1, where member 3 won. Member 0 carries the single-recipe row (lr 3e-5, entropy .003, KL .3).

**Plain finalists, no-tools success:** .723 / .492 / .000, matching the report.

### 2. E16: numbers CONFIRMED; rule QUALIFIED

**Success rates**
- RL r0–r2: 1.000 on all 6 `iid_*` conditions.
- `heldpair_A-S`: .773 / .941 / .000.
- R→S: .961 / 1.000 / .902. R→A: 1.000 for all three.
- Hard assign: .746 / .824 for all three RL lineages, exactly equal to `modular_cheap_first`.
- The teacher scores 1.000 everywhere else.

**Localization** (`heldpair_A-S`, rl-r2):
- All 256/256 failures completed assign, did not complete select, and ran to 64 steps.
- The count of `use_return(as=select)` rejected with "return type mismatch" on a **csp** handle is **12,195**, exactly as claimed.
- rl-r0: 58 failures, all after assign, with 1,447 such rejections. rl-r1: 15 failures, with 675 rejections.
- In R→S, rl-r2 issues 1,218 as-select mismatches and rl-r0 issues 255. The claim of the same error at a lower rate holds.

**Registered rule, recomputed per lineage:**

| Lineage | IID-pair mean | Held-out-pair mean | Triple mean | Held-out ≥ 0.9×? | Triples ≥ 0.8×? |
|---|---:|---:|---:|---|---|
| r0 | 1.000 | .911 | .893 | pass | pass |
| r1 | 1.000 | .980 | .954 | pass | pass |
| r2 | 1.000 | .634 | .480 | fail | fail |

"Supported" at 2/3 is correct **on the aggregate (mean) reading** of the protocol. This reading is weaker than it looks:
- r0 passes the held-out criterion by 0.011.
- If the criterion is applied to each held-out pair and each triple separately, **only r1 passes (1/3)**. r0 fails on A→S (.773), A-R-S (.801) and R-A-S (.781).
- The protocol text ("held-out-pair success", "triple success") does not specify aggregation, and the report does not say which reading it used. The report should state that the aggregate reading is the one applied.

### 3. E16 distractor diagnostic: CONFIRMED

RL success is .383 on S-R, .324 on S-A and .324 on A-R, identical in all three lineages. These equal the zero-distractor fractions from `worlds.jsonl.gz` specs: 98/256, 83/256 and 83/256.

The coincidence is exact, world by world:
- On these three IID conditions, every RL success is in a zero-distractor world (0 successes with ≥1 distractor).
- Every zero-distractor world succeeds (0 failures).

Other checks:
- For all 256 rows of every learned arm, the number of `prior` records in the first observation equals the spec's distractor count.
- The teacher (`modular_cheap_first`) and `modular_always_tool` score 1.000 on all 6 distractor conditions. The references ran on the same seeds and spec_hash.

The only partial exception is on a held-out pair: rl-r0 on `distr_heldpair_R-S` succeeds in 86 worlds with distractors.

### 4. E17: numbers CONFIRMED; rule QUALIFIED; r2 localization magnitudes REFUTED

**Success rates (verified)**
- Held-out A→S: 1.000 / 1.000 / .805.
- r0 and r1: 1.000 on every non-hard condition, including all 6 distractor conditions.
- r2:
  - S→R .824; R→S .895;
  - triples .812–.867, of which the S-after-A triples are .816–.848;
  - distractor conditions .805–1.000, mean .920.
- No E17 lineage issues a single `as=select` return-type mismatch (0 in all three).

**Rule**
- **A→S pair:** E16 .773/.941/.000 (mean .572) → E17 1.000/1.000/.805 (mean .935). The gain is **+.363** and the minimum is .805.
- **Pair plus S-after-A triples:** the gain is +.372 and the minimum is .828.
- **IID pairs:** the lineage mean is **.980**, but per lineage it is 1.000 / 1.000 / **.941**.

The rule therefore passes on the mean reading and **fails if "with IID pairs ≥ 0.95" is applied per lineage**. Reading the rule per lineage matches the protocol's own "no lineage falls below 0.8" clause. The report does disclose r2's S→R .824, but it still labels the rule "supported" without saying that the IID clause was read as a mean.

**r2 localization (REFUTED in its specifics)**
- **Rejections per failed episode:** the report says 25–35 rejected direct commits per failed episode. I measured **38–50** rejected `commit_pending` actions per failed select-containing episode, with medians of 41–49 by condition. For example, `iid_S-R` has 45 failures with exactly 43 rejections each: 1,247 capacity and 688 funds rejections in total.
- **Failure rate:** the report says about 18% of select-containing worlds fail. Pooled over the 11 non-distractor, non-hard select-containing conditions, the failure rate is **13.2%**. By condition it ranges from 0% (iid_S) and 10.5% (R→S) to 19.5% (A→S).

The qualitative finding stands. Every failed r2 A→S episode (50/50) is this commit perseveration, not a return-binding failure.

### 5. E18: CONFIRMED

The 11 assign-containing, non-distractor, non-hard conditions are A, S-A, A-R, R-A, A-S and the six triples. Mean success by arm (r0 / r1 / r2):

| Arm | r0 | r1 | r2 |
|---|---:|---:|---:|
| adapt120 | .852 | .896 | .621 |
| scratch120 | .834 | .435 | .812 |
| zeroshot | .112 | .000 | .141 |
| adapt20 | .980 | .668 | .677 |
| scratch20 | .060 | .021 | .194 |

Adapt120 minus scratch120 is +.018 / +.461 / −.191. One lineage is ≥ +.10 and one is ≤ 0, so the registered result is **PARTIAL**, as reported.

Retention on the select/route-only conditions: adapt120 .896/.995/.891 versus zeroshot .993/1.000/.894, matching the report.

### 6. Paired-world integrity: QUALIFIED

Within every root and condition (1,071 arm files), all arms have the same 256 distinct (seed, spec_hash) pairs as `worlds.jsonl.gz`.

The following groups are identical in both seed and spec_hash:
- Across `e09v2-sealed-r0/r1/r2/e15`: all 26 conditions, including the 8 IID and 10 transfer conditions used in E15.
- Across `e16-sealed-distractors`, `e17-sealed` and `e18-sealed`: all 6 `distr_*` conditions.

**E16 versus E17/E18: spec_hash differs on all 17 non-distractor conditions.** The cause is an explicit empty `distractors` field in the E17/E18 specs. On content, all 17 conditions × 256 worlds match exactly:
- same seed order;
- specs equal once the empty field is removed;
- no non-empty distractors;
- identical `address_seed`.

So the worlds are the same, and seed-paired E16-vs-E17 comparisons are valid. The claim that the roots share the same spec_hash is false, and any tool that pairs worlds by spec_hash across these roots will find no matches.

### 7. Seed disjointness: CONFIRMED

**Sealed spans** (from configs and observed rows):
- 70,000,000–75,100,256 (e09/e09v2, all variants);
- 80,000,000–85,100,256 (e14);
- 90,000,000–95,050,256 (e16/e17/e18).

**Checked against 1,515 used ranges:**
- every `training_seed_interval` in all 66 run states;
- observed development-prediction seeds and protocol development ranges;
- every `seed_start` in non-sealed configs;
- phase-1 `training_seed_start`, `validation_seed` and `first_seed`, each given a conservative 1e6 window.

**No overlap was found.** The nearest used seeds below the sealed band are phase-1 configs at 22M–26M. Everything else is ≥ 300M; phase-3 streams are at ≥ 1.0e9.

**Side observation (not a sealed leak).** Two RL runs reuse the same training worlds as another lineage's bootstrap:
- `e16-rl-single-r0` and `e17-rl-single-r0` start their training stream at 1,040,000,000. This is the same stream as `e16-boot-r2`/`e17-boot-r2` (1,040,000,000–1,040,004,800).
- Likewise, `e18-base-rl-r0` starts at 1,240,000,000, which equals `e18-base-boot-r2`.

So RL lineage 0 trains on the first 4,800 worlds that bootstrap lineage 2 was supervised on. The lineages are therefore not fully independent in training data. The effect is probably negligible, but it should be disclosed.

### 8. Held-out integrity: CONFIRMED

Both the repo configs and the remote `protocol.json` world mixes were checked.

**E16/E17 boot and RL; E18 adapt and scratch.** The training stage orders are exactly {S}, {R}, {A}, S→R, S→A and A→R. None of these runs contains R→S, A→S, R→A or any 3-stage order. The development panels use the same mixture (`development_world_mix` is empty).

**E18 base boot and RL.** The orders are {S}, {R} and S→R only, with no assign stage.

**Distractors.** E16 runs have 0 distractors; E17 and E18 runs have 0–2.

### 9. Ledger: CONFIRMED

- `phase2_charged_cpu_core_seconds` = 145,800.745469256, exactly the sum of the 87 `phase2_jobs` (these include the phase-3 jobs).
- `total_charged_cpu_core_seconds` = phase 1 + phase 2 = 147,498.34 s (40.97 core-h of the 48 core-h ceiling).
- **Spot checks.** Seven jobs were compared with their `occupancy.json` receipts: `e15-halving-plain-r0`, `e16-sealed`, `e16-sealed-distractors`, `e17-sealed`, `e18-sealed`, `e18-adapt-r2` and `e09v2-sealed-e15`. CPU and wall time match exactly in all seven.
- **Full comparison.** Every one of the 85 jobs with a remote receipt matches exactly. The two jobs without a remote receipt are local analysis and audit entries.
- Every e15–e18 receipt on GB10 appears in the budget.

## Step caps and other checks for invalidating problems

- **Modular step caps match.** All 24 E16/E17/E18 training runs use `train.max_steps` = 64, and every sealed modular world has `step_limit` 64. Learned arms reach the 64-step cap, while references use at most 32 steps. No cap asymmetry exists in E16–E18.
- **E15/E08 learned arms use `max_steps` 48**, but `iid_4x5` (and some transfer conditions) allow 64 steps. This is the step-cap asymmetry already disclosed in phase 2. It applies equally to every learned arm in the E15 comparisons, so it does not bias halving vs PBT/multistart/single.
- **Distractor visibility to references.** Reference rows carry no trace, so their observations cannot be inspected directly. They were run on identical seeds and spec_hash, and the teacher scores 1.000 with and without distractors. For learned rows, the first-observation prior records match the spec in 100% of rows.
- **E16 → E17 changes only the distractors, as registered.** Initialization seeds, training streams (1.00/1.02/1.04e9 boot; 1.04/1.06/1.08e9 RL) and development seeds are identical between E16 and E17. E18 adapt and scratch share their streams (1.32/1.34/1.36e9).

## Concerns, in order of weight

1. **E17 "supported" depends on reading the IID clause as a lineage mean.** Read per lineage, r2 is .941 < .95. The report should label this a "mean-reading pass; per-lineage fail (r2)".
2. **E16 "supported" 2/3 depends on aggregating held-out pairs and triples.** Per condition it is 1/3, and r0 clears the aggregate held-out criterion by only 0.011.
3. **The E17 r2 localization numbers are wrong.** The report gives 25–35 rejections per failed episode; the measured range is 38–50. It gives about 18% of select-containing worlds; the pooled measure is 13.2% (10.5–19.5% per condition).
4. **E15 "3/3 vs PBT" is true in point estimates only.** For r0 the paired CI includes 0, and r2's margin is +0.001.
5. **spec_hash differs between E16 and E17/E18** for identical worlds; pairing must use seeds.
6. **Training-stream reuse** between RL lineage 0 and bootstrap lineage 2 (E16, E17, E18 base). This affects independence only.
7. **This audit's own ~475 core-s CPU charge** should be added to `budget.json`.

No problem was found that invalidates a headline number. Every recomputed success and utility figure matches the report to the reported precision.
