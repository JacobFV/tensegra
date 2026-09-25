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
3. **E17** tests whether training with wrong-typed distractor returns repairs that failure. *(In progress.)*

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

Protocol: [phase2/E17-protocol.md](phase2/E17-protocol.md). *(Results pending.)*
