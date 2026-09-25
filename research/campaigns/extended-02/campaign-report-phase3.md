# Extended-02 phase 3: depth-allocating selection, and composition across primitives

Phase 3 continued on user instruction after the phase-2 closure ([campaign-report-phase2.md](campaign-report-phase2.md)). The same ceilings and deadline apply, and device-occupancy GPU accounting was accepted by the user. Decisions are logged in [decisions.md](decisions.md) (phase-3 section).

## Summary so far

1. **Selection that allocates depth beats selection by copying (E15).** Successive halving gives the eliminated members' update slots to survivors, who keep training sequentially. In point estimates, it beats PBT in 3/3 replicates (pooled sealed IID utility +0.015 [+0.012, +0.018]; transfer +0.052) and matched multistart in 3/3 (+0.005 [+0.003, +0.007]). Per replicate, the r0 interval vs PBT includes zero (+0.004 [−0.0005, +0.009]) and the r2 margin is +0.001.
   - It found the profile-tuned hyperparameter row on its own in 5/6 runs.
   - It still trails the single learner that was *handed* that row (−0.006, 3/3).
   - The registered depth hypothesis, that most finalists reach the robust direct-first mode, is **not supported**: 1/3 per arm.
   - Supplied niche protection did not help (−0.002).
2. **Learned controllers compose three primitives in unseen orders, with one sharp, localized failure (E16).**
   - In a new modular workshop (select/route/assign over subset search, shortest path and CSP), RL controllers trained on singles and three ordered pairs solved every training composition (1.000).
   - They also solved most held-out orders and three-stage sequences, and the registered transfer rule passes (2/3 lineages) **on the aggregate reading**: mean over held-out pairs and mean over triples. Applied per condition, only 1/3 lineages pass, and r0 clears the aggregate held-out criterion by 0.011.
   - **Where it fails:** select after assign, a transition never seen in training (0.00 / 0.77 / 0.94 across lineages).
   - **Mechanism:** the policy applies the previous stage's CSP return as the select result, over and over. This is a return-binding shortcut that only a novel composition exposes.
3. **Wrong-type return exposure repairs it (E17).**
   - **The E16 controllers are broadly brittle.** When 0–2 irrelevant prior solver results of the wrong type are present, success on *training* compositions falls from 1.000 to 0.32–0.38, while the teacher stays at 1.000.
   - **The repair works.** Training with such distractors (E17; nothing else changed) lifts assign→select from 0.57 to 0.94 (lineage means) and distractor-world success to 1.000 in 2/3 lineages. The registered rule is met.
   - **One lineage shows a separate weakness:** it perseverates on rejected direct commits.
4. **A never-seen primitive gives a large few-shot head start that does not reliably persist (E18).**
   - Three lineages were trained without the assign stage. They score ~0 on it zero-shot, as expected: the new actions were never active, so zero-shot success is not identifiable.
   - After 20 supervised updates on the full mixture, they reach 0.67–0.98 on assign-containing sealed worlds, versus 0.02–0.19 for the same 20 updates from scratch. That point is unregistered and descriptive.
   - At the registered 120-update endpoint the difference is +0.02 / +0.46 / −0.19, so the rule outcome is **partial**.
7. **Commit perseveration is a memory gap (E22).** Public per-stage counters of rejected completion attempts (m3) eliminated the recurring "repeat a rejected commit" failure: 126 → 0 perseveration failures over 22,272 sealed episodes per arm, with no retention loss. Most of the prior failures came from one lineage.
6. **Depth + robustness-visible selection fails (E21).** Halving with no-tool/oversized worlds in the selection panel reached the combined policy in 0/3 runs and lowered sealed utility vs plain halving (0.822/0.928/0.888 vs 0.933/0.936/0.929).
5. **Provenance binding needs a provenance input (E19/E20).** Prior results of the *right* type but for an unrelated instance collapse the E17 controllers (0.35 mean success, below the no-tool greedy's 0.63). Their m1 inputs carry no provenance. Adding public provenance features (m2), and training with same- and wrong-type distractors, restores 1.000 in all three lineages with no retention loss (registered rule supported 3/3). The two changes were made together.

## E15: depth versus breadth in selection

Protocol: [phase2/E15-protocol.md](phase2/E15-protocol.md). E15 is identical to E08 PBT (banks, hyperparameter rows, streams, mixture, 1,800 RL updates). Mode `halving` keeps 6 → 3 → 2 → 2 → 2 survivors. Survivors receive the freed slots sequentially, reaching 720 updates of depth versus 300 per PBT lineage. There is no weight copying or mutation. The `niche` arm keeps the best member of each greedy-first niche at every cut. The niche descriptor is supplied from phase-2 findings.

| Sealed (70M+ worlds) | r0 | r1 | r2 |
|---|---:|---:|---:|
| Halving (plain), IID utility | 0.933 | 0.936 | 0.929 |
| Halving (niche), IID utility | 0.933 | 0.928 | 0.929 |
| E08 PBT | 0.929 | 0.896 | 0.928 |
| E08 multistart | 0.928 | 0.929 | 0.925 |
| E08 single (pre-tuned row) | 0.943 | 0.943 | 0.930 |
| Halving (plain), transfer utility | 0.894 | 0.852 | 0.818 |
| Plain finalist greedy-first / no-tools success | 1.00 / 0.72 | 0.79 / 0.49 | 0.00 / 0.00 |

| Paired (plain halving − …) | IID per replicate | IID pooled | Transfer pooled |
|---|---|---|---|
| PBT | +0.004 / +0.040 / +0.001 | +0.0149 [+0.0124, +0.0175] | +0.0519 |
| Multistart | +0.005 / +0.006 / +0.004 | +0.0052 [+0.0033, +0.0071] | +0.0149 |
| Single | −0.010 / −0.007 / −0.001 | −0.0060 [−0.0076, −0.0046] | −0.0324 |
| Niche − plain | 0 / −0.007 / 0 | −0.0024 | −0.0118 |

**Interpretation.** At equal RL updates, a selection procedure that concentrates depth on survivors outperforms both exchange-by-copying (PBT) and independent search with final selection. Halving discovered the hyperparameter row that the single baseline was given. The behavioral mode of the finalist still follows its surviving lineage's trajectory: in r2 that lineage is tool-first, exactly as the single learner from the same bank member. Protecting a supplied behavioral niche kept direct-first members alive, but the final pick by development utility on tool-available worlds still chose tool-first members. The limits of phase 2 (shared banks, one environment) apply.

## E16: composition across primitives

Protocol: [phase2/E16-protocol.md](phase2/E16-protocol.md). Environment: `workshop-modular-v1` ([source](../../../src/topoformer/campaign02_modular.py)).
- **Stages.** The public goal is an ordered tuple of distinct stages, each with a direct path and a budgeted solver path:
  - **select:** constrained subset;
  - **route:** shortest path;
  - **assign:** 5 tasks × 3 slots with forbidden pairs, solved by CSP.
- **Stage gating.** Completing actions count only for the current stage.
- **Supplied:** solvers, draft builders, validators, and stage-typed candidate features (m1).
- **Leverage profile (dev seeds 460M).** The public cheap-first teacher solves all 15 sequences (1.00). The no-tool greedy solves 0.36–0.55 whenever assign is involved. A 6×4 assign region is budget-bound for every method (0.81–0.89) and is reported separately.

**Training.** Three fresh lineages. Each has a 600-update supervised bootstrap from the teacher, then 1,800 single-lineage actor-critic updates. The training mixture is S, R, A, S→R, S→A, A→R. In training, select is never preceded by another stage, and route is never followed by one.

Sealed success (90M+ seeds, 256 worlds each):

| Condition | boot r0/r1/r2 | RL r0 | RL r1 | RL r2 | Teacher | No-tool greedy |
|---|---|---:|---:|---:|---:|---:|
| IID singles and pairs (6) | 0.66–1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.36–1.00 |
| Held-out R→S | .85/.84/.86 | 0.961 | 1.000 | 0.902 | 1.000 | 0.863 |
| Held-out R→A | .95/.90/.75 | 1.000 | 1.000 | 1.000 | 1.000 | 0.488 |
| Held-out **A→S** | .61/.68/.59 | 0.773 | 0.941 | **0.000** | 1.000 | 0.387 |
| Triples, S before A (3) | .10–.82 | 0.95–1.00 | 1.00 | 0.88–1.00 | 1.000 | 0.37–0.43 |
| Triples, **S after A** (3) | .58–.70 | 0.78–0.82 | 0.89–0.93 | **0.000** | 1.000 | 0.43–0.45 |
| Hard assign 6×4 (A; S→A) | .65–.73 | 0.746; 0.824 | same | same | 0.746; 0.824 | 0.18–0.23 |

**Registered rule:** compositional transfer is **supported on the aggregate reading** (r0 and r1 pass: mean held-out-pair success ≥0.9× IID and mean triple success ≥0.8× IID; r2 fails). The protocol did not specify aggregation. Per condition only r1 passes, and r0 fails A→S (.773), A-R-S (.801) and R-A-S (.781). The per-lineage aggregates are .911/.980/.634 (held-out pairs) and .893/.954/.480 (triples), against an IID of 1.000.
- RL substantially improves composition over the bootstrap. For example, bootstrap r1 solves the S→R→A triple only 0.105 of the time.
- On the budget-bound hard-assign region, all RL lineages equal the teacher exactly.

**Failure localization** ([phase2/E16-localization.json](phase2/E16-localization.json)):
- **All failures stop right after assign.** Every failed A→S episode for rl-r2 (256/256) completes assign and then never completes select.
- **The error is a return-type mismatch.** After assign, rl-r2 issues 12,195 `use_return(csp record, as=select)` actions, each rejected with "return type mismatch", until the step limit. The partially failing lineages show the same pattern mixed with rejected direct commits.
- **R→S shows it too, at a lower rate:** the route return is present when select becomes current.
- **Why training never caught it:** in training, whenever select was current, no computation record of any type existed. The policy learned "when the stage is current, apply a return" rather than binding a return to its type. The m1 features do mark type agreement between a return and its intended use, but nothing in training forced the policy to use that feature while select was current.

This is a concrete instance of the prompt's warning that copied or returned results must be addressed correctly: return *addressing* can look perfect in trained compositions and still fail under a new one.

## E17: wrong-type return exposure

Protocol: [phase2/E17-protocol.md](phase2/E17-protocol.md). E17 is identical to E16 (initialization seeds, recipes, streams, mixture, features, held-out splits). The one change: training worlds start with 0–2 public, certificate-valid prior computation records of primitives **absent from that goal**. These are wrong-typed for every stage, uninformative and uncharged, and the teacher ignores them.

**Diagnostic on E16 first.** The same sealed worlds with distractors added cause the E16 controllers to fail on *training* compositions:
- S→R 0.383, S→A 0.324 and A→R 0.324, identically in all three lineages. This equals the fraction of worlds that drew zero distractors, so almost any distractor causes failure.
- The teacher stays at 1.000.
- The E16 policies bind "the current stage" to "whatever computation record exists".

| Sealed success | E16 RL r0 / r1 / r2 | E17 RL r0 / r1 / r2 |
|---|---|---|
| IID singles/pairs (no distractors) | 1.000 all | 1.000 all except r2 S→R 0.824 |
| Held-out A→S | 0.773 / 0.941 / 0.000 | **1.000 / 1.000 / 0.805** |
| Held-out R→S | 0.961 / 1.000 / 0.902 | 1.000 / 1.000 / 0.895 |
| Triples with S after A | 0.78–0.93 / 0.00 (r2) | 1.000 / 1.000 / 0.82–0.85 |
| Distractor worlds (3 IID + 3 held-out pairs) | 0.00–0.65 | **1.000 / 1.000** / 0.81–1.00 |
| Hard assign (budget-bound) | = teacher | = teacher |

**Registered rule: passes on the mean reading, fails on a per-lineage reading.**
- The mean A→S gain is +0.36, above the required 0.2.
- The minimum lineage is 0.805, clearing the 0.8 floor.
- The IID-pair mean over lineages is 0.980 (≥0.95 required).
- **Per lineage, r2's IID-pair mean is 0.941 < 0.95.** Its S→R success is 0.824. The protocol's own "no lineage below 0.8" clause is per lineage, so the stricter reading is defensible.
- No E17 lineage issues a single wrong-type `as=select` use (the auditor counted zero).

**The r2 residual failure is not about composition.** In 13.2% of select-containing sealed worlds (10.5–19.5% by condition; corrected by the audit from my initial ~18%), r2 repeats a rejected direct commit instead of switching to the solver. Each failed episode has 38–50 capacity/funds rejections (corrected from 25–35). This is the same perseveration pattern seen in phase 2's tool-first finalists after corrupted returns. It is a lineage-level weakness of the select stage.

**Interpretation.** The failure boundary found in E16 was not a limit of "composition" as such. It was a missing *return-binding* skill that the training distribution never demanded. Exposing the controller to wrong-typed returns during training, without ever showing the held-out order, repaired both the held-out order and robustness to distractors. This is consistent with the prompt's emphasis on testing wrong, stale and absent results. The distractors vary type only. Same-type distractors, which would require provenance binding (e.g. an old subset result for a different draft), were **not** tested.

## E21: depth allocation with robustness-visible fitness

Protocol: [phase2/E21-protocol.md](phase2/E21-protocol.md). E21 is E15 halving-plain with E13's selection panel (~20% no-tool and 25-item worlds; training unchanged).

| Sealed | r0 | r1 | r2 |
|---|---:|---:|---:|
| Greedy-first (control) | 0.25 | 0.01 | 0.12 |
| Solver calls per control episode | 1.59 | 1.27 | 1.68 |
| No-tools success (selection-visible) | 0.52 | 0.00 | 0.01 |
| IID utility | 0.822 | 0.928 | 0.888 |
| E15 halving-plain IID utility | 0.933 | 0.936 | 0.929 |
| Transfer utility | 0.710 | 0.838 | 0.829 |

**Registered outcome: mixed.**
- **The combined policy was not reached:** 0/3 finalists meet the direct-first + tool-use + no-tools criteria.
- **Not an over-correction either:** the finalists still call solvers, unlike E13's tool-avoiding populations.
- **Worse than plain halving:** utility and transfer both drop.
- **Development dynamics:** the surviving lineages oscillated between modes in the final rounds. Final-round greedy-first rates fell from ~1.0 to 0.07–0.37 in the selected member.

Making robustness visible to depth-allocating selection added selection noise without selecting the robust mode. Across all five population designs tested (PBT, PBT + curriculum, PBT + robust fitness, halving, halving + robust fitness), **none** reliably produced the combined policy. Concentrated single-lineage training produced it in 5/7 lineages in phase 2 and in 2/3 fresh lineages in E14.

## E19/E20: provenance binding under same-type distractors

Protocol: [phase2/E19-E20-protocol.md](phase2/E19-E20-protocol.md). **Same-type distractors** are 0–2 public, certificate-valid prior results of primitives *in* the goal, solved for unrelated instances: right type, wrong provenance. Sealed seeds are 96M+ (conditions S→R, S→A, A→R, R→S, A→S, R→A, S, A).

| Sealed success (mean over lineages r0 / r1 / r2) | E17 (m1, wrong-type training) | E20 (m2 + both distractor types) |
|---|---|---|
| Same-type distractor conditions (8) | 0.367 / 0.367 / 0.347 | **1.000 / 1.000 / 1.000** |
| Wrong-type distractor conditions (6) | 1.000 / 1.000 / 0.920 | 1.000 / 1.000 / 1.000 |
| IID pairs | 1.000 / 1.000 / 0.941 | 1.000 / 1.000 / 1.000 |
| Held-out pairs | 1.000 / 1.000 / 0.900 | 1.000 / 1.000 / 1.000 |
| Triples | 1.000 / 1.000 / 0.837 | 1.000 / 0.999 / 0.919 |
| Teacher on same-type conditions | 1.000 | 1.000 |
| No-tool greedy on same-type conditions | 0.626 | 0.626 |

**E19 diagnosis.** With a same-type prior record present, the E17 controllers fail almost exactly when any distractor exists. Their m1 candidate features cannot distinguish their own result from an unrelated same-type result, so this is an input gap, as predicted.

**E20 repair.** It combines two supplied changes, as registered:
- the m2 public provenance features (the record's problem is one of this episode's drafts, and that draft is unchanged since the call; the records' `prior` field is never read);
- training worlds with both distractor types.

The registered rule (+≥0.2 in ≥2/3 lineages, retention loss ≤0.05) is **supported 3/3**: +0.63 to +0.65 with no retention loss. The r2 commit perseveration on S→R also disappeared (0.824 → 1.000).

**Residual.** E20 r2 still fails 15–19% of the three triples in which select follows assign (0.809–0.852). Every failure is the direct-commit perseveration (37–45 rejected commits per failed episode), with no return-binding error.

**Interpretation.** Return *type* binding could be learned from m1 inputs once wrong-type returns appeared in training (E17). Return *provenance* binding required a provenance input. Given that input plus exposure, all three lineages became fully robust to both kinds of irrelevant results.

## E22: commit perseveration as a memory gap

Protocol: [phase2/E22-protocol.md](phase2/E22-protocol.md). The lightweight controller is memoryless, and its observation shows only the last action's feedback. m3 adds public per-stage counters of rejected completion attempts and own solver calls; everything else is identical to E20 (seeds, streams, mixture, recipes).

| Sealed (E16 set + wrong- and same-type distractor conditions; 29 non-hard conditions × 256 worlds per arm) | r0 | r1 | r2 |
|---|---:|---:|---:|
| E20 failures (perseveration: ≥10 rejected completion attempts) | 0 (0) | 1 (1) | 125 (125) |
| E22 failures (perseveration) | 3 (0) | 0 (0) | 3 (0) |
| E20 triple success | 1.000 | 0.999 | 0.919 |
| E22 triple success | 0.999 | 1.000 | 0.999 |

**Registered rule: supported.**
- Pooled perseveration failures dropped 126 → 0, well past the required 50% reduction.
- No condition-group mean dropped by more than 0.001.

**Caveat.** The evidence is concentrated in lineage r2. The pairing makes it the same lineage (same initialization and streams) with only the feature input changed, but a single lineage's trajectory can still differ for idiosyncratic reasons.

**Interpretation.** The failure recurred across phases (tool-first finalists under corrupted returns, E17-r2, E20-r2). It is consistent with a controller that cannot remember how many times its direct attempt has already failed. A supplied public counter, a form of exact typed memory, removes it. The recurrent workspace, which in principle could integrate this history, was not retested here: phase-2 E11 showed it collapsing under RL.

## E18: a held-out primitive

Protocol: [phase2/E18-protocol.md](phase2/E18-protocol.md). Three fresh lineages trained only on S, R and S→R (with distractors): 600 supervised + 1,800 RL updates, no assign stage anywhere. They are compared on assign-containing sealed conditions (11 conditions, 90M+ worlds, 256 each). The arms (same supervised stream within each lineage) are:
- **zero-shot:** the base endpoint;
- **adapted:** base + N supervised updates on the full mixture;
- **scratch:** fresh initialization + the same N updates.

| Assign-containing sealed success | r0 | r1 | r2 |
|---|---:|---:|---:|
| Zero-shot (base) | 0.112 | 0.000 | 0.141 |
| *Adapted, N = 20 (unregistered, descriptive)* | *0.980* | *0.668* | *0.677* |
| *Scratch, N = 20 (unregistered, descriptive)* | *0.060* | *0.021* | *0.194* |
| Adapted, N = 120 (registered) | 0.852 | 0.896 | 0.621 |
| Scratch, N = 120 (registered) | 0.834 | 0.435 | 0.812 |
| Adapted − scratch at 120 | +0.018 | +0.461 | −0.191 |
| Select/route-only retention: base → adapted@120 | 0.993 → 0.896 | 1.000 → 0.995 | 0.894 → 0.891 |

**Registered rule: partial.** Only 1/3 lineages meet the +0.10 threshold, and 1/3 is negative.

- **What prior training buys:** a very fast start on the new primitive. At 20 updates the gap is +0.48 to +0.92, consistent with reuse of stage-generic skills (gating, return binding, budget escalation).
- **Why that doesn't settle transfer:** by 120 updates both kinds of learner are dominated by lineage-specific failures. scratch-r1 collapses on held-out orders (0.435); adapted-r2 drops to 0.621; adapted-r0 loses 0.10 of retention.
- **Development curves:** they tell the same story. Scratch catches up by ~40 updates, then both oscillate between 0.63 and 1.0.
- **Classification:** this is few-shot *retraining* with privileged teacher supervision. It is not descriptor-based zero-shot use of a new primitive, which the controllers cannot do by construction.

## Phase-3 resources

Ledger: [budget.json](budget.json). Through E18 the campaign totals are **41.0 of 48 CPU core-hours** and **7.7 of 12 GPU device-hours**. At low concurrency a 1,800-update RL lineage costs ~700–750 CPU core-seconds, versus ~5,700 under phase-2 contention.

## Independent audit (phase 3)

[review/phase3-independent-audit.md](review/phase3-independent-audit.md) is a separate reconstruction from raw rows with its own code (~475 CPU core-s).

**Confirmed:**
- all E15, E16, E17 and E18 success and utility numbers;
- the E15 halving mechanics (all 24 cuts recomputed);
- the E16 localization count (12,195);
- the E16 distractor diagnostic, exact world by world: every success is a zero-distractor world;
- E18 PARTIAL;
- sealed-seed disjointness against 1,515 ranges;
- held-out integrity of every training mixture;
- the ledger (85 receipts match exactly).

**Qualified or corrected (all incorporated above):**
- E16 and E17 rule outcomes depend on reading criteria as means.
- The E17 r2 localization magnitudes (corrected).
- The E15 per-replicate interval for r0.

**Additional disclosures:**
- **Pairing across roots.** E16 and E17/E18 sealed worlds are identical in content and seeds, but their spec_hash differs because of an explicit empty `distractors` field. Pairing across these roots must use seeds.
- **Shared training worlds.** RL lineage 0 of E16, E17 and E18-base starts its training stream at the same seed as bootstrap lineage 2 (e.g. 1.04e9), so those lineages share their first 4,800 training worlds. This weakens independence slightly and does not affect sealed evaluation.
- **Step caps.** The modular experiments have no step-cap asymmetry (learned max_steps = world step_limit = 64).
