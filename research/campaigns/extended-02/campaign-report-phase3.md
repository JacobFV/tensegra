# Extended-02 phase 3: depth-allocating selection, and composition across primitives

Phase 3 continued on user instruction after the phase-2 closure ([campaign-report-phase2.md](campaign-report-phase2.md)). The same ceilings and deadline apply, and device-occupancy GPU accounting was accepted by the user. Decisions are logged in [decisions.md](decisions.md) (phase-3 section).

## Summary so far

1. **Selection that allocates depth beats selection by copying (E15).** Successive halving gives the eliminated members' update slots to survivors, who keep training sequentially. It beats PBT in 3/3 replicates (pooled sealed IID utility +0.015 [+0.012, +0.018]; transfer +0.052) and matched multistart in 3/3 (+0.005 [+0.003, +0.007]).
   - It found the profile-tuned hyperparameter row on its own in 5/6 runs.
   - It still trails the single learner that was *handed* that row (−0.006, 3/3).
   - The registered depth hypothesis, that most finalists reach the robust direct-first mode, is **not supported**: 1/3 per arm.
   - Supplied niche protection did not help (−0.002).
2. **Learned controllers compose three primitives in unseen orders, with one sharp, localized failure (E16).**
   - In a new modular workshop (select/route/assign over subset search, shortest path and CSP), RL controllers trained on singles and three ordered pairs solved every training composition (1.000).
   - They also solved most held-out orders and three-stage sequences, and the registered transfer rule passes (2/3 lineages).
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

**Registered rule:** compositional transfer is **supported** (r0 and r1 pass: held-out pairs ≥0.9× IID and triples ≥0.8× IID; r2 fails).
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

**Registered rule: supported.**
- The mean A→S gain is +0.36, above the required 0.2.
- The minimum lineage is 0.805, clearing the 0.8 floor.
- The IID-pair mean is 0.980 (≥0.95 required).
- **Per-lineage caveat:** r2 individually drops to 0.824 on S→R.

**The r2 residual failure is not about composition.** In ~18% of worlds containing select, r2 repeats a rejected direct commit (capacity/funds rejections, 25–35 per failed episode) instead of switching to the solver. This is the same perseveration pattern seen in phase 2's tool-first finalists after corrupted returns. It is a lineage-level weakness of the select stage.

**Interpretation.** The failure boundary found in E16 was not a limit of "composition" as such. It was a missing *return-binding* skill that the training distribution never demanded. Exposing the controller to wrong-typed returns during training, without ever showing the held-out order, repaired both the held-out order and robustness to distractors. This is consistent with the prompt's emphasis on testing wrong, stale and absent results. The distractors vary type only. Same-type distractors, which would require provenance binding (e.g. an old subset result for a different draft), were **not** tested.

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
