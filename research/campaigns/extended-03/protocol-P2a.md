# P2a: bounded diagnostic of RL collapse, and one bootstrap-anchor comparison

Status: **registered before implementation and before any P2a run** (2026-09-26).

- **Scope.** Mechanism screening within the current budget window. It is **not confirmatory**. A confirmatory P2b needs a fresh budget and is designed only if P2a finds a credible stabilization mechanism.
- **Motivation.** [report-P1.md](report-P1.md) §3 and §8 (post-review): imitation acquired the task, and the inherited RL updater failed to preserve it.
- **What does not change.** P1's endpoints and verdicts are preserved. The P1 bootstraps are kept as frozen reference agents.

## Question

Can the RL stage **preserve** the competence imitation acquired, and then **improve** verified utility or computation cost without destroying it?

The first test is **one** preservation intervention. A reward change, a progress penalty, or a state buffer is not combined with it.

## Part D: collapse audit on archived runs (no new training)

**Subjects:**
- X1-rl-r0 and X1-rl-r2, which collapsed, and X1-rl-r1, which stayed stable.
- Their development checkpoints at attempts 18–29 (the 60-update tranche endpoints).
- Their shared bootstrap endpoints.

**Probe worlds:** 64 fixed development-distribution worlds on fresh probe seeds (1,990,000,000+, never used), from the IID group (f0 and f2).

**Measurements, per checkpoint:**
1. Greedy and sampled rollouts: success, utility, action mix, idempotent-repeat rate, short-cycle rate (definitions below), steps to the cap. Sampled rollouts use fixed sampling seeds and 4 samples per world.
2. KL(π_bootstrap ‖ π_ckpt) and KL(π_prev-tranche ‖ π_ckpt), on states visited by the bootstrap's greedy rollouts and on states visited by the checkpoint's own rollouts.
3. Critic error (value vs realized return-to-go) and the mean advantages assigned to loop actions (idempotent repeats and cycle members) vs productive actions (solver calls, successful commits, use_return, move).
4. Gradient-norm contributions of the actor, critic, entropy and KL terms on a fixed rollout batch, plus their cosine similarity on the shared parameters. The actor and critic share the encoder.

**Reward accounting check:** tests on explicit trajectories confirming that per-step rewards telescope to the episode's task utility (verified success − costs) on three paths: success, step-cap failure, and truncation. The terminal handling for each is to be verified.

Part D is descriptive: it localizes where and how the collapse happens. It identifies no cause on its own.

## Part C: one controlled comparison (new training)

**Subject:** lineage X1-r2. It has the sharpest collapse, and its P1 RL run is the replay target.

| Arm | Update rule |
|---|---|
| **C0 (unchanged)** | The P1 X1-r2 RL config replayed exactly: same bank, seeds, streams and hyperparameters. It measures reproducibility against P1's X1-rl-r2 as well as serving as the control. |
| **C1 (bootstrap anchor)** | C0 plus a fixed anchor term, `anchor_kl_weight · mean KL(π_bootstrap ‖ π_current)` over states visited in the rollout batch. π_bootstrap is the frozen X1-r2 bank checkpoint (sha256 in `configs/campaign03/p1-banks.json`). `anchor_kl_weight` = 0.3, the same scale as the existing tranche KL, which is kept. This is one value, fixed with no tuning. |

- **Matching.** Both arms run the full P1 horizon (5 rounds × 6 slots × 60 updates = 1,800), through the historical collapse region at attempts 24–29. Architecture, initialization, reward (the unchanged task utility) and episode streams all match.
- **Registered deployment rule** (a separately reported policy, never applied to screening or sealed data): among a run's tranche checkpoints, deploy the **latest** one whose development success ≥ bootstrap development success − 0.02 **and** whose development utility ≥ bootstrap development utility − 0.02. The development data is P1's 128 development worlds for this lineage. If no checkpoint qualifies, **roll back** to the bootstrap.
- **Reporting.** Both arms report (1) the **final raw checkpoint** and (2) the **deployed checkpoint**.

## Screening evaluation (non-sealed, non-confirmatory)

- **Seeds:** fresh screening seeds 120,000,000 + 100,000·i, 256 worlds per condition, never used before.
- **Conditions:** iid_f0, iid_f2, events_train_kinds_p1, foreign4 (the P1 definitions).
- **Policies:**
  - the X1-r2 bootstrap;
  - C0 final and C0 deployed;
  - C1 final and C1 deployed;
  - P1's X1-rl-r2 endpoint, which checks replay behaviour on new worlds;
  - the dep_reuse reference.
- **Modes:** greedy (as in P1), plus sampled with 1 fixed-seed sample per world.

**Metrics:**
- success; utility per episode (verified outcomes per total cost, failures included); work per success, which is never used alone;
  - *Post-hoc wording note (audit):* "utility" is computed as verified success − cost per episode, as in P1 and the code. The phrase "per total cost" is inaccurate; the computation is unchanged.
- correct reuse and invalid use (the P1 definitions);
- **idempotent-repeat rate:** actions whose action_key equals the previous accepted action's, with the public state unchanged;
- **short-cycle rate:** an accepted action that returns the public decision state (draft constraints, commitments, retrieved set, position) to a state already seen in the preceding 6 steps, without an intervening solver call or new inspection;
- per-episode no-progress counts (either of the above);
- steps-to-cap rate.

## Screening readings (decided before running; not claims)

| Reading | Definition |
|---|---|
| **S0** (replay) | Does C0 reproduce collapse? Final development success < bootstrap development success − 0.20. If it does not, GPU/numerical nondeterminism enters the explanation and is reported. |
| **S1** (preservation) | C1 final greedy screening IID-group success ≥ bootstrap − 0.02, **and** C1's no-progress rate ≤ the bootstrap's + 0.02. |
| **S2** (improvement) | C1 final IID-group utility ≥ bootstrap + 0.01, **or** work per success ≤ 0.9 × bootstrap with success not below bootstrap − 0.02. |
| **S3** (deployment vs learning) | Deployed vs final, for each arm. If the deployed checkpoint helps only where the final one fails, the result is a deployment rule, not stable learning. |

**Next-step rule:** if S1 holds, propose a confirmatory P2b (fresh lineages, sealed seeds, a new budget) for the anchor. If S1 fails, P2b is not proposed for the anchor; Part D's localization then guides the next candidate (e.g. critic isolation, a clipped update, or trust-region control). S2 alone does not justify a claim of reward-driven improvement from one lineage.

## Budget (current window, CPU allowance above the 20% reserve ≈ 26k core-s)

- Estimates:
  - Part D: ≤ 3k core-s.
  - Part C: 2 runs at ≈ 7.5–8k core-s each.
  - Screening evaluation: ≈ 3–4k core-s.
  - Total: ≈ 22k.
- **Profiling first.** The P1 runs were CPU-bound, and "GPU occupancy" is an accounting convention. The first P2a job records a phase breakdown (environment step, encoding, forward/backward, export) before the Part C runs are sized.
- **If the estimate exceeds the allowance:** drop the sampled screening mode before shortening Part C. A run that stops before the historical collapse region cannot establish stability.
