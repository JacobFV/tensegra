# Extended-03 proposal: managing a changing dependency graph of computations

Status: **proposal, not started.** It follows the extended-02 closeout ([extended-02/closeout.md](extended-02/closeout.md)). Budget, ceilings and authorization are to be set when the campaign is launched.

> Extended-02 answered "will the agent call a solver?" Extended-03 asks: **can it manage a changing dependency graph of computations — binding, reusing, invalidating and revising results correctly?**

## Central task (one, not a collection of interface fixes)

**Data-dependent composition with changing requirements and multiple reusable computational results.**

A workshop episode chains real data dependencies:

```text
selected items ──► assignment domains and durations ──► route and timing constraints
     (subset)            (CSP over the chosen items)          (shortest path / schedule)
```

- **Real dependencies.** The items chosen in the first computation define the domains and durations of the second, and the assignment defines time windows for the third. A choice can be locally valid but make a downstream stage infeasible or expensive. The agent must then **revise an earlier decision**, and know which downstream results that revision invalidates.
- **Changing requirements.** Mid-episode, public events can change a capacity, a time window or an edge. Some results stay valid; others become stale.
- **Many results.** Several results per primitive exist at once: the agent's own earlier results (some still valid, some stale), results for other registered problems, and foreign results, some of which are accidentally valid.
- **Provenance and recovery are necessary.** Rejections carry meaningful reasons. Blind recomputation is costly, and blind reuse fails.

The stage order is **not** announced as a fixed script. The goal states the final requirement, and the agent decides which computations to run, reuse or revise.

## Design requirements (from extended-02 lessons)

1. **Preflight observability audit (gating).** Before any training, construct paired states for every distinction the task requires, and verify that the actual encoded observation **and** full candidate set differ:
   - current vs stale;
   - same-type other-problem;
   - reusable old;
   - rejection reasons;
   - retry-after-change vs identical retry.

   Commit the audit as a test. A failing preflight blocks training.
2. **Applicability, not ownership.** The return interface should expose applicability relations: type compatibility, matching request/arguments, compatible dependency versions and usable status. The evaluation must include **old-but-still-valid** results, and it should reward validated reuse over recomputation where reuse is cheaper. It must not reward "never touch foreign records". The test breaks the registry-membership shortcut relationally: foreign results for *registered* problems, versions of the *same* problem, and reusable old results.
3. **Explicit attempted-action record** (typed memory), replacing ad-hoc counters. Each entry holds the action + arguments, the input/state version, the outcome, the reason, and whether its dependencies have changed. Legitimate retries after new information stay allowed.
4. **Controllers.** The lightweight controller with explicit typed memory is the reference. The recurrent workspace is a matched alternative with identical information, and it must earn its cost on a named requirement (history integration or argument binding). Neither is crippled.
5. **Frozen, independently verified components.** Solver implementations are frozen and versioned. Final goals are verified by independent validators. Reduction correctness and solver correctness are audited separately.
6. **Fresh evaluation and controlled overlap.**
   - Sealed worlds are generated only after the recipe freezes.
   - No sealed world is reused across a researcher-adaptive intervention without being labelled as such.
   - Training streams must not overlap across lineages; that overlap was disclosed in extended-02.
   - A versioned **content-based world comparison key** replaces spec_hash pairing.
7. **Pre-specified aggregation.** Every decision rule names the conditions, weights, per-lineage requirements, any worst-condition floor, and the confidence procedure. Old rules are preserved, never silently changed.
8. **Metrics.**
   - Verified success, and utility with cost components kept separate.
   - **Reuse efficiency:** recomputation avoided when valid, and invalid reuse attempts.
   - Stale-use rate.
   - Revision quality: were only the invalidated results recomputed?
   - Repeated-rejection rate.
   - Area under the adaptation curve, updates to sustained competence, and retention (for any adaptation study).
9. **Population search stays secondary.** Successive halving is the default search reference. Evolutionary comparisons come only after the task has measurable, observable good resource strategies, and then with fresh banks, three independent lineages and matched compute.

## Staged plan

| Stage | Deliverable | Gate |
|---|---|---|
| A | Dependency-graph world v1: three data-coupled primitives, public change events, applicability metadata, independent validators; references (greedy, always-recompute, validate-and-reuse teacher) | References show a **reuse-leverage region**: validate-and-reuse beats always-recompute at equal success, and naive reuse fails |
| B | Preflight observability audit for all required distinctions (encoded tensors) | All distinctions preserved; the audit is committed as a test |
| C | Lightweight controller + typed attempted-action memory; supervised bootstrap, then stabilized actor-critic (KL trust region) | Development competence; reuse/stale/revision metrics registered |
| D | Sealed evaluation on fresh worlds: new dependency patterns, new change-event types, longer chains | Pre-registered rules with explicit aggregation |
| E | Matched recurrent-workspace arm, with the same information | Must beat the reference on a named requirement |
| F (optional) | Successive-halving search vs a single learner at matched compute, fresh banks | Separate budget |

## Carried-over open questions from extended-02

- The m1 + same-type-training arm that would separate input from data in the E20 repair. Superseded by requirement 2; run only if cheap.
- The residual `choose_item` loop: check whether the attempted-action record removes it.
- The omitted semantic composition (RL01/RL02, 0/512). This is outside this proposal's agentic track; it needs its own representation hypothesis.
- Graph-bias (programmable) attention: only if a task arises where relation-induced bias addresses an identified requirement, with oracle-vs-learned grounding labelled.
