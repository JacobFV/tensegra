# Protocol A3: on-policy imitation without the anchor (DAgger-style), adaptive follow-up

Registered 2026-09-26T22:10Z, **after** A2 and its audit. This is a researcher-adaptive follow-up and is labelled as such.

**Motivation.** The audited A2 finding: imitation-only (dep_reuse CE on the policy's own sampled states, RL off) gave the largest sampled-mode gain, but it ran with the bootstrap KL anchor (0.3), which holds the policy near the bootstrap. The dep_reuse teacher sits ~.04 utility above the bootstrap.

**Hypothesis.** Removing the anchor lets on-policy teacher supervision move the policy toward the teacher and improve deployed utility over the imitation bootstrap. This would be an improvement over imitation via **supervised on-policy rehearsal, not RL**, and is to be reported as such.

**Arm.** `configs/campaign04/a3-dagger-x1-r2.json`: a2-imit with `anchor_kl_weight` 0 and no anchor checkpoint, the only difference.
- Kept: rehearsal λ = 0.5, RL loss off, tranche KL 0.3, entropy .003, lr 3e-5, the same seeds, streams and 1,800 updates.
- Registered deployment mode: greedy.

**Screening.** Fresh worlds **190,000,000 + 100,000·i**, 256 per condition, over iid_f0, iid_f2, events_train_kinds_p1 and foreign4. Modes greedy, sampled and r_mask. Comparators on the same worlds: the X1-r2 bootstrap, a2-imit final and dep_reuse. Both the final and the P2a-rule deployed checkpoint are reported.

**Promotion rule** (IID group, final checkpoint), all of:
1. greedy utility ≥ bootstrap greedy + .01;
2. greedy utility ≥ the bootstrap's best mode + .005;
3. **per-step** no-progress rate ≤ bootstrap greedy's + .01 (episode-level rates cannot separate modes; see F1);
4. greedy utility ≥ a2-imit greedy + .005, so that removing the anchor is what helped.

**Confirmation if promoted (F3, pre-registered here):**
- Apply the A3 recipe to fresh bootstraps r3–r5, each anchored to nothing and starting from its own bank.
- Evaluate on sealed worlds **200,000,000 + 100,000·i**, 512 per condition, greedy vs each bootstrap greedy.
- **F3-H:** utility ≥ bootstrap + .01 in 3/3 lineages and success ≥ bootstrap − .01, with the per-step no-progress clause.

If A3 is not promoted, the A2 localized reason stands, and the anchored/unanchored imitation-only comparison is added to it.

**Budget.** A3 run ≈ 4.5k plus screening ≈ 2.5k. F3, if triggered: 3 runs ≈ 13.5k plus evaluation ≈ 3.5k (uses the confirmation reserve).
