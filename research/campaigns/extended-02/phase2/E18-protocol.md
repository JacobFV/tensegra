# E18: a held-out primitive — zero-shot, few-shot adaptation, and retraining (pre-registration)

**Question.** Do skills learned on two primitives transfer to a third, never-seen primitive (the CSP-backed `assign` stage)? The candidate skills are stage gating, return type binding, budget escalation, and direct-first-then-solver behaviour.

**Base lineages (3 fresh).**
- Initialization seeds 91000+, streams 1.2e9+.
- 600 supervised bootstrap updates (public `modular_cheap_first` teacher), then 1,800 single-lineage actor-critic v2 updates (E16 recipe).
- The mixture is S, R, S→R (×2), with 0–2 wrong-typed distractor returns (the E17 repair). **No assign stage ever appears.** Roster inspection, `choose_slot`, `commit_assignment`, `start_assign` and csp calls are absent from every training catalogue.

**Arms, per lineage.**
- **zero-shot:** the base RL endpoint, evaluated directly. The assign action kinds have never been active in its inputs, so a zero-shot success claim is **not identifiable**. The measurement is reported, and near-zero is expected.
- **adapted:** base RL endpoint + 120 supervised updates (6 slots × 20, development curve after each) on the full E17 mixture (S, R, A, S→R, S→A, A→R, with distractors).
- **scratch:** a fresh initialization + the same 120 supervised updates, same streams.
- **Reference:** E17 bootstrap endpoints (600 supervised updates on the full mixture). A curve context, not a matched arm.

**Evaluation.** Frozen latest checkpoints on the E16 sealed conditions (90M+), plus distractor variants.

**Primary endpoint.** Mean success over assign-containing sealed conditions (A, S→A, A→R, R→A, A→S, all six triples).

**Decision.** Transfer to a new primitive is supported if adapted − scratch ≥ 0.10 in ≥2/3 lineages. It is negative if ≤ 0 in ≥2/3 lineages; anything else is partial. Select/route-only conditions are reported for retention (catastrophic forgetting check).

**Disclosure.** Adaptation uses privileged teacher supervision. This is few-shot *retraining*, not descriptor-based zero-shot use.
