# Extended-03 P1 report: acquisition, emergent reuse, and input ablations in depworld-v1

**Status: final for P1.** The post-review corrections are in §3 and §8. The results were reproduced from raw rows by an independent auditor ([review/p1-independent-audit.md](review/p1-independent-audit.md)). The corrections the audit requested are applied below.

**Design:**
- Protocol: [protocol-P1.md](protocol-P1.md), frozen before training.
- Metric operationalization: [protocol-P1-metrics.md](protocol-P1-metrics.md), registered before any sealed result.
- Analysis: `research/results/campaign-03/p1-analysis/`.
- Host: pro6000 ([infrastructure.md](infrastructure.md)).

## 1. Headline

Imitation of a public validate-and-reuse teacher **acquires** the dependency-graph task. Under the registered RL recipe, most of what imitation acquired **is then lost late in training**.

- **Bootstraps.** Every bootstrap on the full d1 inputs (X1, X2, X4) reaches 0.93–0.98 sealed IID success, with near-zero invalid reuse.
- **Collapse under RL.** Actor-critic v2, the extended-02 recipe run for 1,800 updates, collapsed two of three X1 lineages and degraded three more lineages in other arms. The registered latest-checkpoint rule captured those collapsed states.
- **Input ablations.** Removing the applicability relations (X3) produces the predicted failure: 41–46% invalid reuse vs ~0% with them. This is the most robust result.
- **Attempt-memory ablation (X4).** Its registered rule passes, but only on fragile comparisons.

## 2. Verdicts (registered rules, point estimates; audit-qualified)

| Rule | Registered verdict | Audited reading |
|---|---|---|
| **R1** acquisition (X1) | **Not supported.** 0/3 lineages reach ≥ .95 IID success; lineage-mean utility .231 vs a bar of .883 | The failure is **RL degrading an already-acquired policy**. At the X1 *bootstrap* endpoints (descriptive, not registered) R1 would pass narrowly: success .961/.955/.961; utility .8834 vs .8826, a margin of +.0008. |
| **R2** emergent reuse (X2) | **Not supported.** Only r2 passes (+.54, [.46, .60]); r0 −.18 and r1 −.48 | **RL did not increase reuse that was already high, and it lowered reuse in 2/3 lineages.** The protocol assumed that "any reuse in X2 is learned from reward". That premise is false: the X2 *bootstraps*, imitating a teacher that never reuses, already reuse correctly at .82/.82/.43 (IID). About 92% of those reuses are valid retrievals of foreign records. The audit found no metric artifact and zero invalid or stale uses. The one RL gain (r2) comes mostly from reusing the agent's own route records. |
| **R3** applicability input is used (X3 vs X1) | **SUPPORTED, 3/3.** Invalid use X3 .41–.46 vs X1 ≤ .014; stale use .31–.34 vs 0 | **Robust.** It also holds at the bootstrap endpoints (differences .45/.61/.26) and in the one uncollapsed pair (r1). X1 has essentially no invalid uses anywhere. |
| **R4** attempt memory is used (X4 vs X1) | **SUPPORTED** by the retry clause (3/3). The success clause fails: X1 − X4 = −.331, because X1 collapsed | **Fragile.** Pair r0 compares against a collapsed X1 (94 rejections in 2,048 episodes). Pair r2 rests on one world (seed 110200178, a use_return repeated 37 times); counted per episode, its direction reverses. Only r1 is a clean pair (.426 vs .064), which is 1/3. At the bootstraps, X1 *with* attempt memory already repeats 57–73% of its rejections identically. Supplied attempt memory is therefore not shown to be used as a retry suppressor. |
| **R5** transfer (X1) | **Not supported** (1/3) | The r0 "pass" (ratio 2.04) is an artifact of collapsed IID success (.092) and is **not** transfer. At the bootstraps (descriptive): .93/.94/.95. |

## 3. Findings

1. **Late RL collapse is the dominant P1 fact.**
   - **What the collapsed policies do.** They loop on actions that succeed but have no effect, until the 96-step limit. X1-rl-r0 alternates `add_constraint` on finish-by bounds 8 and 9 and never calls the csp solver. X1-rl-r2 retrieves the same route record repeatedly after committing the assignment. X4-rl-r1, X2-rl-r0 and X4-rl-r0 loop the same way. They do not abstain, accumulate rejections, or commit wrong answers.
   - **It was visible in development.** X1-r2 held .95 for 24 attempts, then fell to .92, .61, .13, .00, .00. X1-r0 was already slipping to .84–.88 from attempt 21, then fell to .57, .21, .10, .10. The action mix at the last development attempt matches the sealed loops.
   - **Stable runs.** X1-r1, X2-r1/r2, all X3 runs (which *improve* under RL: IID-group success .76–.78 → .79–.82) and X4-r2.
   - **Implication.** The extended-02 AC v2 recipe (lr 3e-5, entropy .003, KL .3) is **not stable over 1,800 updates in depworld**.
     - **Correction after review:** an earlier draft called the loops a "reward-neutral attractor". That is too strong. Training reward is the increment in verified success minus resource costs. Accepted actions earn nothing, unnecessary actions still cost, and failing to finish forfeits the terminal reward. So the loops are *not* neutral under the task utility.
     - **What is established:** the training dynamics produced policies that repeatedly take low-immediate-cost actions and never reach the high-value outcome. **Why** the updates became destructive is unresolved. Candidates: policy drift, critic error, advantage normalization, entropy pressure, a shifting state distribution.
     - The KL term anchors only to a copy of the policy taken at the start of each 60-update tranche. It never anchors to the bootstrap, so small per-tranche drifts can accumulate.
     - Training samples actions while evaluation is greedy; this difference has not been examined around the collapse boundary.
2. **The identical-retry metric cannot see these loops.** It counts only rejected attempts. Collapsed X1 RL scores 0.000 while repeating one accepted action more than 70 times. Loop detection needs both **idempotent-repeat** detection and **short-cycle** detection. For example, alternating a finish-by bound between 8 and 9 changes the draft every time but returns to states already visited.
3. **Imitation generalizes to reuse the teacher never showed.** X2's reuse is valid (0 invalid, 0 stale) and consistent across lineages at the select and assign stages. This reads as the teacher's use_return pattern for fresh results generalizing to foreign applicable records through the supplied applicability relations. It is a finding about imitation plus supplied relations. It is **not** a finding about reward-driven discovery.
4. **Applicability relations are necessary and used.** X3 fails through exactly the predicted mechanism: invalid and stale reuse, with 0.52–0.88 success across conditions and endpoints. RL partly compensates (X3 RL improves) but does not close the gap. This validates the Stage B preflight design: when the relation is erased, the behaviour it supports disappears.

## 4. Supplied vs learned (P1)

- **Supplied:**
  - the world, solvers and validators;
  - the public teachers (dep_reuse, dep_recompute), used as bootstrap supervision;
  - d1 relational features: request match, dependency match, the attempt record.
- **Learned:**
  - action sequencing across the select → assign → route dependency chain;
  - revision after events: revision quality .95–.96 for the X1 bootstraps on events_train_kinds_p1 (lower, .54–.88, on the held-out deadline_moved kind);
  - reuse of valid foreign records by imitation-trained X2, without demonstration.
- **Not shown:**
  - reward-driven emergence of reuse;
  - RL improvement over imitation on the full inputs;
  - use of attempt memory for retry suppression;
  - transfer beyond the bootstrap-level descriptive readings.

## 5. Integrity (audit)

- **Worlds and seeds.** Every condition used 256 sealed worlds at 110,000,000 + 100,000·i. All 29 policies ran identical worlds within a condition. The sealed seeds are disjoint from all training, development and initialization ranges, and those ranges are disjoint across lineages. Arms are paired within a lineage, as the freeze amendment specifies.
- **Checkpoints and arms.** All 24 checkpoint hashes match the config, the files and each run's latest checkpoint. The RL initial checkpoints match `p1-banks.json`. Arm feature versions and teachers match the protocol.
- **Job receipts.** All 25 exited 0.
- **Reconstruction.** 3,944 metric cells were reconstructed independently. No primary metric or rule value differs by more than 1e-9.
- **Two secondary-metric implementation choices differ from the audit** and affect no verdict: `correct_reuse_rate_applicable_pre` counts failed use_returns, and `identical_retry_rate_no_missing_dependency` drops attempts before pairing. **The truncation column in the analysis output is not a measurement** (the row field is never set) and must not be cited. Failed collapsed episodes do run to the 96-step limit.

## 6. Cost (pro6000, ledger in [budget.json](budget.json))

| Item | CPU (core-s) | Wall |
|---|---:|---:|
| Smoke (2 jobs) | 51 | 51 s |
| 12 bootstraps | 7,056 | ~10 min (concurrent) |
| 12 RL runs | 94,886 | ~2.5 h (concurrent) |
| Sealed evaluation | 10,555 | 2.8 h |
| **Total** | **112,547 / 172,800** (65%) | GPU occupancy 20,119 / 43,200 s (47%) |

## 7. Next steps (not started)

1. **P2: RL stability, registered before running.** One targeted comparison on X1 and X2, run on fresh sealed seeds:
   - **(a)** A KL anchor to the *bootstrap* policy, not only to the previous policy.
   - **(b)** A progress-aware step cost, or a registered no-progress-repeat penalty.
   - **(c)** Development-gated early stopping with a registered rule. This is endpoint selection on development data only, never on sealed data.

   Each variant needs a matched control, the unchanged recipe on fresh lineages. The key question is whether RL can *improve* on imitation (utility, work per success) without collapse.
2. **A no-progress loop metric:** repeated successful actions with unchanged public state. Register it alongside identical retry.
3. **X2-style reuse generalization:** test whether imitation-trained reuse survives foreign records whose applicability differs subtly (request match without dependency match), on fresh worlds.
4. **R4 redesign.** A clean attempt-memory test needs a non-collapsed comparator and per-episode as well as per-rejection readings, with worlds where identical retry is tempting and costly.
5. **Budget.** A full-scale P2 (≥ 6 RL lineages at 1,800 updates, ~8k core-s each) does not fit the remaining CPU allowance above the 20% reserve (~26k core-s usable). P2 therefore needs a new budget window or reduced scope, which is the user's decision.

## 8. Post-review notes (2026-09-26)

A review of P1 at fac0d822 accepted the findings and qualified them:
- **Attempt memory.** R4 is **inconclusive**, not "supported with caveats". P1 does not robustly show that the controller uses the attempt record for recovery.
- **Transfer metric.** A ratio over a collapsed IID baseline is meaningless. Future transfer criteria need an **absolute competence floor** in addition to a relative degradation bound. P1's registered verdicts stand.
- **The X2 bootstraps' reuse** is useful generalization under supplied relations, not reward-driven discovery. The mechanism (applicability relations making "use a valid old result" resemble "use a fresh result") is plausible but not isolated.
- **Supplied relations.** Request and dependency matching are supplied computations over public state. P1 shows the controller *uses* them; it does not show it could induce them.
- **The capability exists in the saved bootstrap checkpoints; the failed component is the improvement procedure.** The bootstraps are kept as reference agents. The P1 endpoints and verdicts are not replaced.
- **Next step.** The full P2 in §7 is superseded by a bounded diagnostic **P2a** ([protocol-P2a.md](protocol-P2a.md)): a collapse audit on archived checkpoints plus one controlled comparison of a fixed bootstrap anchor. A confirmatory P2b needs a fresh budget, only after P2a identifies a credible mechanism.
