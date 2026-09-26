# Extended-03 P2a report: diagnosing the RL collapse, and one bootstrap-anchor comparison

**Status: final for P2a.** Screening only, non-confirmatory. The results were reproduced from raw rows by an independent auditor ([review/p2a-independent-audit.md](review/p2a-independent-audit.md)), and the audit's corrections are applied below.

**Sources:**
- Protocol: [protocol-P2a.md](protocol-P2a.md), registered before implementation.
- Part D: `research/results/campaign-03/p2a-partd/`.
- Screening analysis: `research/results/campaign-03/p2a-analysis/`.

## 1. Headline

**P1's "RL collapse" is a collapse of the greedy (argmax) policy, not of the learned action distribution. It is deterministic, and a fixed KL anchor to the bootstrap prevents it over the full horizon. The anchor achieves this by holding the policy essentially at the bootstrap. RL did not improve on imitation in any tested mode.**

| Question | Answer (one lineage, X1-r2; screening) |
|---|---|
| Does the collapse reproduce? | **Yes, bit-exactly.** C0's weights, optimizer state and development curves are identical to P1's X1-rl-r2 run. |
| Is it a loss of the learned distribution? | **No.** Sampled success stays at .80–.93 through attempt 29 in all three X1 lineages; greedy falls to .09 and .00 in the two that collapsed. |
| Does a fixed bootstrap anchor prevent it? | **Yes over this horizon.** C1 greedy IID success is .949, equal to the bootstrap's (an exact tie, 486/512). |
| Does RL improve on the bootstrap? | **No.** C1 utility is .864 vs .872, and work per success is 1.22× the bootstrap's. Under sampling, every RL policy does more work per success. |
| Is the anchor "stable learning"? | **Mostly a freeze.** KL(bootstrap ‖ C1) stays at .014–.045, versus .26–.69 for C0 at comparable attempts. |

## 2. Part D: where and how the collapse happens (archived X1 runs; descriptive)

1. **Greedy vs sampled.** In r0 and r2, greedy and sampled first diverge by at least .10 at attempt 26; they never do in r1. After that, greedy loops (idempotent repeats plus short cycles reach about .7 of decisions) while sampled rollouts escape. At the bootstrap, greedy (.95) already beats sampled (.85–.88).
2. **Drift is local to the policy's own states.** KL(bootstrap ‖ checkpoint) on the checkpoint's own greedy states jumps at the collapse: for r2 it goes .41 → .69 → .96, with argmax disagreement .21 → .75. On the bootstrap's states it stays at .22–.53, and the per-tranche KL stays at ≈ .01–.03. Many small steps accumulate into a different argmax path on states the bootstrap rarely visits.
3. **The critic does not favour loops.** The standardized advantage of loop actions is below that of productive actions in 7 of 8 pre-collapse checkpoints.
4. **Critic gradients (corrected per the audit).**
   - The critic's gradient on the shared encoder is 30–80× the actor's **only at the bootstrap**, where imitation never fitted the value head. Afterwards the ratio is mostly 1.5–7×, and in r2 it falls below 1 just before the collapse.
   - The stable lineage r1 shows the same pattern, so this does **not** localize the collapse.
   - The earlier "clip fraction 1.00, so the step direction is the critic's" argument is withdrawn: clipping only rescales, and AdamW normalizes per coordinate. The effective update follows the actor at r2 attempt 25.
   - "Critic-dominated updates" remains at most a weak candidate.
5. **Reward accounting.** Per-step rewards telescope exactly to verified success − cost on the success, step-cap, loop-to-cap and abstain paths. The truncation path is biased in principle but unused (max_steps = step limit = 96).
6. **Measurement caveat.** Part D's no-progress definition differs from the registered screening one (.068 vs .031 for the bootstrap on the same worlds). Part D and screening rates are not comparable.

## 3. Part C: screening readings (IID group, greedy unless stated; audit-reproduced)

| Policy | Success | Utility | Work/success | No-progress | To cap |
|---|---|---|---|---|---|
| Bootstrap | .949 | .872 | 137.0 | .037 | .023 |
| C0 final (= P1 X1-rl-r2) | .000 | −.145 | n/a | .712 | 1.000 |
| C0 deployed (attempt 24) | .900 | .820 | 135.7 | .177 | .100 |
| C1 final (= deployed) | .949 | .864 | 166.8 | .032 | .051 |
| dep_reuse | .979 | .905 | 126.2 | 0 | 0 |

The registered readings:
- **S0, replay:** true.
- **S1, preservation:** true. Correct wording: **the collapse was prevented, with the policy held within about 0.02–0.05 nats of the bootstrap.** The unmodified bootstrap satisfies S1 by definition, so S1 shows prevention, not improvement.
- **S2, improvement:** false.
- **S3, deployment vs learning:**
  - C0's deployed checkpoint (.900) is a deployment rule sidestepping a collapsed endpoint, not stable learning.
  - It is also weaker than the bootstrap on new worlds: −.049 success and .177 no-progress. The registered 0.02 margin on 128 development worlds did not protect it.
  - C1's final checkpoint is preserved; deployment adds nothing.

**What C1 changed while held near the bootstrap:**
- None of its 256 greedy action sequences on iid_f0 match the bootstrap's.
- Solver calls per episode rise from 3.6 to 5.7.
- The budget mix shifts from about 95% budget-128 to roughly equal thirds of 16/128/1024. This accounts for the +22% work.
- Successes are unchanged world by world: at most one world flips per condition.
- Policy entropy stays at .84–.96, while C0's rises to 1.60.

**Early warning signs in C1.** It shows the same signatures that preceded C0's collapse, at much lower levels. Sampled no-progress is .031 vs .013, sampled step-to-cap is .066 vs .014, and greedy step-to-cap is .051 vs .023. S1 reads greedy success only and does not see them.

## 4. Interpretation

- **The improvement procedure fails in two separable ways.**
  - (a) Greedy deployment of a stochastically trained policy is fragile: RL moves probability mass so that the argmax path on self-visited states becomes a loop, while the sampled distribution stays competent.
  - (b) In either mode, RL does not find cheaper or more reliable behaviour than imitation. Under sampling it spends more work per success.
- **What the anchor does and does not do.** It addresses (a) by staying close to the bootstrap. It does nothing for (b).
- **What the next experiment needs.** A procedure that improves verified utility while keeping the deployed (greedy) policy's behaviour on its own states under control. Examples: train-time evaluation of the argmax policy, entropy/temperature-aware deployment, a trust region measured on self-visited states, or a critic fitted before the actor moves. The anchor alone is a preservation device, not an improvement mechanism.
- **Not separated here:**
  - the anchor's effect vs simple entropy control;
  - the anchor weight (one value);
  - generality beyond one lineage;
  - sealed-world confirmation.

## 5. Integrity (audit)

- **Replay.** The C0 config and protocol are byte-identical to P1's. Tensors and optimizer state are identical at 11 attempts, and the development rows are identical once timing is stripped.
- **C1.** It differs only in the anchor: weight 0.3, bank sha verified. The anchor is active on 1,799/1,800 updates, and the tranche KL is unchanged.
- **Deployment decisions** re-derive exactly from the raw development rows, and the checkpoint hashes match.
- **Screening seeds** (120,000,000 + 100,000·i) are disjoint from every other range. Worlds are identical across policies and modes, and the sampling seeds are fixed and paired.
- **Discrepancies.** 2 of 627 values differ by more than .001; neither affects a reading.
- **Two notes:**
  - The dep_reuse rows carry no trace, so its zero loop rates follow from the registered rule, not from measurement.
  - The protocol's phrase "utility = verified outcomes per total cost" should read "success − cost", as in P1 and the code; the protocol file is corrected with a note.

## 6. Cost (pro6000)

| Item | CPU (core-s) |
|---|---:|
| Part D audit | 3,251 |
| C0 + C1 | 7,035 |
| Screening | 2,825 |
| **P2a total** | **13,111** |

- **Campaign total:** 125,657 / 172,800 CPU core-s. The 20% reserve is intact: 12.6k of usable allowance remains. GPU occupancy is 26.9k / 43.2k s.
- **Unmetered:** subagent development and audit CPU on the pro6000 (a few thousand core-s in total) is not metered by the job wrapper.
- **Profiling.** The runs are CPU-bound. Neural forward, backward and optimizer steps are about 16–20% of CPU; encoding and collation are about 40%.

## 7. Next steps (not started)

1. **P2b (needs a fresh budget).** Target *improvement*, not preservation. Keep the anchor or a self-state trust region as the safety rail, and add one mechanism aimed at (b):
   - a critic warm-up before actor updates, or
   - train-time argmax evaluation with a registered deployment temperature.

   Run on three fresh lineages, with sealed seeds and the P1 comparator. Report greedy and sampled modes, and pre-register the loop-signature early-warning metrics (sampled no-progress, step-to-cap) as secondary endpoints.
2. **Anchor vs entropy control.** A cheap matched arm that holds entropy at the bootstrap's level without the anchor.
3. **Speed.** Optimize observation encoding and collation (about 40% of CPU) before any larger study. The GPU is not the bottleneck.
4. **Deployment policy as an explicit research object.** Greedy vs sampled vs temperature is a registered choice with its own evaluation. It is not an implementation detail.
