# Probeworld (Track B): specification, labels, splits, ladder

**Status:** built 2026-09-26 on `campaign/e04-probeworld`. The benchmark and exact labels are checked by tests. The ladder has **not** been run; only a 1-seed smoke on a reduced pool exists (below). Design authority: [design.md](design.md) Track B, with v2 revisions 7, 10 and 11; review findings F13, F14 and F16.

- Code: `src/tensegra/campaign04_probeworld.py` (pure Python, no torch or numpy).
- Ladder: `research/tools/campaign04_probeworld_train.py` (torch, CPU).
- Tests: `tests/test_campaign04_probeworld.py`.
- Exact-label report: `research/results/campaign-04/probeworld/exact-label-report.json`.

## 1. Environment

**Hidden type θ** (4 types). Public features (f1, f2) ∈ {0,1,2}×{0,1} give a declared prior P(θ | f) (`FEATURE_PRIORS`).

| θ | probe | exact b1 | exact b2 | prop |
|---|---|---|---|---|
| H heuristic-solvable | solved | solved | solved | reduced |
| M exact-easy | fails | solved | solved | reduced |
| F feasible-hard | fails | **timeout** (solved if reduced) | solved | reduced |
| X infeasible | fails | **timeout** | infeasible (certificate) | conflict w.p. p_conflict, else reduced |

**Episode.** An episode has k related queries, with k ∈ {1, 2, 4, 8}.
- Queries are iid given f, and in the correlated condition given a hidden binary z that shifts P(H) by ±corr.
- After each terminal action the verifier reveals that query's θ, as public audit feedback.

**Actions** (each at most once per requirement epoch; inspect at most once per query; b1 is not offered after b2):

| action | charge | effect |
|---|---|---|
| probe | c_probe (+ D_side if θ≠H, side-effect condition) | candidate if "solved". Failure is detected w.p. q; q = 1 means reliable. |
| exact_b1 / exact_b2 | c_b1 / c_b2 | solved → verified candidate. **timeout = unknown**: no candidate, and the posterior keeps F and X. |
| inspect | c_inspect | noisy easy/hard signal (flip prob η): a partial reveal. |
| prop | c_prop | conflict → X certain; otherwise the instance is reduced (b1 then solves F), and the posterior changes. |
| build | C_build (once per episode) | reusable structure |
| use | C_execute + C_return + C_verify (three ledger lines) | verified answer: solution or infeasibility |
| commit | 0; −L if wrong | **Irreversible.** Commits the best candidate. A probe candidate is correct iff θ = H; a stale candidate is wrong. |
| commit_infeasible | 0; −L if wrong | irreversible; correct iff θ = X |
| abstain | 0 | ends the query |

**Events** (condition flag). After any non-terminal action, once per query, a requirement change fires with probability p_event. It invalidates the declared subset:
- the solution candidates (they become stale);
- the propagation reduction;
- the epoch usage of probe, b1, b2, prop and use, which may then be re-run.

Type evidence and the inspect signal stay valid.

**Conditions (flags).**
- unreliable detection: q < 1;
- irreversible probe side effect: D_side > 0;
- correlated heuristic failure: corr > 0;
- events: p_event > 0.

**Utility and charging.**
- U = R·(verified successes) − Σ costs, with R = 100.
- Each cost is one ledger line, charged exactly once. This is tested, including use's three components and the side effect.
- Oracle DP label compute is an **offline** CPU cost (F16), never subtracted from episode utility.

**ρ_k.**
- ρ_k = C_shortcut / (C_build/k + C_execute + C_return + C_verify), with declared C_shortcut = c_b2, the direct exact route that resolves every type.
- We also report **ρ_eff**, with C_shortcut_eff = the expected per-query cost of π* when the structure is disabled. This is the quantity at which the optimal strategy actually switches (§4).

## 2. Exact labels

**Information state.** I_t = (query index, built, revealed H/non-H counts [correlated only]; posterior over θ rounded to 12 digits, usage bits, reduction flag, candidate kind and validity, event flag).
- It is computed from the visible history only (`public_state_from_history`), with the step record (action, outcome, event, revealed θ).
- The hidden θ and z are used only by `Episode` to simulate outcomes and charges.

**DP.** `ExactSolver` runs memoized DP over this finite space, under the declared prior. It produces:
- Q*(I, a) for every available action;
- V*(I);
- the ε-optimal set A*(I), with **registered ε = 0.5** (price units; R = 100).

The continuation is always π* (`CONTINUATION = "pi_star_exact_dp_v1"`). The deterministic representative used for teacher forcing is the lowest action index in the exact-optimal set.

**Case types** are a function of (I_t, a_t, I_{t+1}). Precedence is a > b > c > d.
- **(a) unjustified:** Q*(I_t, a_t) < V*(I_t) − ε.
- **(b) rational attempt that failed:** a_t is ε-optimal and its own outcome is a failure (probe failed, b1 timeout, wrong commit).
- **(c) invalidated:** a_t is ε-optimal with no own failure, and either a requirement-change event fired, or the new evidence leaves none of the planned next actions ε-optimal. The planned next actions are A* at the modal non-event successor of a_t (the optimal-set change of F13).
- **(d) direct:** anything else (continue or terminate).

**Validation** (`tests/test_campaign04_probeworld.py`, 22 tests, all passing on the pro6000):
- **DP = brute force.** DP matches an independent expectimax over raw visible histories, with beliefs from explicit enumeration over hidden worlds (z, θ_1..θ_k), on 7 hand-sized cases. These cover unreliable detection, inspect/prop, events, side effects, build/use with k = 2, and the correlated case with k = 3 (rel. tol. 1e-9).
- **Monte Carlo.** The mean utility of π* over 3,000 worlds per config matches V*, within 4.5 SE, on 3 dev configs.
- **Labels are a function of the visible history.** Over random-policy rollouts (> 500 distinct histories), identical histories give identical labels, and the state rebuilt from the history equals the simulator's state. Two episodes with different hidden θ (H vs an undetected M failure) and the same visible history have identical labels. Their posterior is the Bayes value .5/.8.
- **Timeout ≠ infeasible.**
  - After a b1 timeout: P(F) = P(X) = .5, no candidate, and Q(commit_infeasible) = .5R − .5L.
  - After a b2 certificate: P(X) = 1 and Q = R.
- **Irreversible commit and charging.**
  - After a commit, the query cannot be revisited; a wrong commit costs L exactly once.
  - A stale candidate after an event is wrong.
  - Each cost is charged once.
- **Splits and determinism.** The splits are disjoint in generator parameters. The generator is deterministic, the DP is deterministic, and the rollouts replay.
- **Other checks.** Direct-success episodes are present in every split. π* strategy is monotone in ρ_k and k.

### Worked example (task brief)

The setup: heuristic cost 1, 80% success, exact fallback 100, reliable detection, R = L = 200.
- Probe-first: expected cost 1 + .2·100 = **21**.
- Exact directly: **100**.
- Q*(probe) = 179 vs Q*(exact) = 100.
- The failed probe is case (b), not (a); choosing exact directly is case (a).

| variant | Q*(probe first) | Q*(exact) | A* |
|---|---:|---:|---|
| base (reliable detection) | 179.0 | 100.0 | probe |
| unreliable detection q = .5 | 149.0 | 100.0 | probe (commit on "solved" risks L) |
| unreliable q = .5, L = 1000 | 99.0 | 100.0 | **exact** |
| probe side effect D = 50 | 169.0 | 100.0 | probe (flips at D = 395) |
| probe side effect D = 500 | 79.0 | 100.0 | **exact** |
| heuristic success 30% | 129.0 | 100.0 | probe (1 + .7·100 = 71 < 100) |
| correlated failure corr = .15, k = 4 | 716.0 | 637.0 | probe (later queries adapt to revealed failures) |
| events p = .3 (candidate invalidated) | 174.56 | 93.7 | probe |

## 3. Generator and splits (by generator parameters only)

**Price sampling.** Prices are sampled inside a price-region cell of a 3×3 grid:
- x = c_probe / c_b1, with bins [.01, .05), [.05, .2), [.2, .8];
- y = C_build / c_b2, with bins [.3, 1), [1, 3), [3, 10].

Other prices: c_b1 is log-uniform in [8, 50]; c_b2 = c_b1·U[1.5, 2.5]; C_exec + C_ret + C_ver = c_b2·U[.1, .3]; L ~ U[50, 200].

| split | price cells | k | condition combos | config seeds |
|---|---|---|---|---|
| train / dev / test_iid | 8 cells (centre held out) | {1, 2, 8} | ≤ 2 flags, minus 2 held-out pairs (9 combos) | 4.1e9 / 4.2e9 / 4.3e9 + i |
| heldout_price | centre cell (1, 1) | {1, 2, 8} | train combos | 4.4e9 + i |
| heldout_k | train cells | **4** | train combos | 4.5e9 + i |
| heldout_comp | train cells | {1, 2, 8} | **unreliable+events**, **side_effect+correlated** (each flag seen singly in training) | 4.6e9 + i |

- **Worlds.** Eval world seeds are split base + 5e7 + 1000·i + r. Training worlds are 8e9 + 1e8·seed + n. All ranges are disjoint and outside the ranges already used.
- **Case types never define a split.** This is tested.
- **heldout_comp is used for no tuning.** No hyperparameter was chosen on any held-out split. The ladder uses fixed defaults and the **final** checkpoint only.

**π* statistics per split** (40 configs × 10 worlds each):

| split | direct-success episodes | case b | case c | case d | mean P(build) |
|---|---:|---:|---:|---:|---:|
| train | .33 | 359 | 382 | 3604 | .32 |
| dev | .38 | 346 | 380 | 3360 | .30 |
| test_iid | .36 | 347 | 379 | 3525 | .23 |
| heldout_price | .34 | 356 | 356 | 3002 | .23 |
| heldout_k | .27 | 398 | 450 | 3671 | .36 |
| heldout_comp | .43 | 118 | 388 | 3228 | .35 |

## 4. ρ_k / k strategy switch (exact labels)

- **Fixed prices:** prior f = (1, 1), c_probe 3, c_b1 20, c_b2 45, C_exec + ret + ver = 5, L 150.
- **Swept:** C_build and k.
- **Measured:** P(π* builds), computed exactly over the outcome tree.
- **No-structure π* cost per query:** 14.65, so ρ_eff = 14.65 / (C_build/k + 5).

| k \ C_build | 10 | 20 | 40 | 60 | 120 | 240 |
|---|---|---|---|---|---|---|
| 1 | **.55** (ρ_eff .98) | 0 (.59) | 0 | 0 | 0 | 0 |
| 2 | **.80** (1.46) | **.55** (.98) | 0 (.59) | 0 | 0 | 0 |
| 4 | **1** (1.95) | **1** (1.46) | **.55** (.98) | 0 (.73) | 0 | 0 |
| 8 | **1** (2.34) | **1** (1.95) | **1** (1.46) | **1** (1.17) | 0 (.73) | 0 (.42) |

- **Where the switch happens.** The optimal strategy switches from probe-first (no structure) to build-then-use at **ρ_eff ≈ 1**.
- **P(build) = .55 at ρ_eff = .98.** This is a contingent strategy: π* probes first and builds only after the probe fails.
- **The declared ρ_k overstates the switch.** The declared ρ_k (C_shortcut = c_b2) puts the switch at ρ_k ≈ 3, because the shortcut is itself a probe-first mix. Both are reported.
- **Monotonicity.** The switch is monotone in C_build and in k (tested).

## 5. Model and factorization ladder

**Model** (`ProbeNet`, identical in every rung, 129,189 parameters at hidden 128):
- linear input projection → GRUCell(128) → tanh trunk;
- heads: policy (10), value (1), Q (10), stage (6), dependency (5), switch (1), previous-case (4).

**Inputs per step, all public:**
- the config's public vector: prices / R, q, D_side, corr, p_event, η, p_conflict, k, log ρ_k, flags, declared prior;
- the previous visible step record: action, outcome, event bit, revealed θ;
- the available-action mask;
- the fraction of queries done.

Privileged labels appear **only** in losses.

**Losses:** cumulative ladder, with every rung keeping the RL loss.

| rung | loss added |
|---|---|
| L0 | advantage actor-critic on U: MC return-to-go with the shared value head as baseline, value MSE, entropy .01 |
| L1 | + optimal-action-set imitation: set-valued CE, −log Σ_{a∈A*} π(a) |
| L2 | + stage/dependency supervision: stage set-CE on the stages of A*; BCE on 5 dependency bits (candidate valid, candidate exact, reduction valid, structure built, event this query) |
| L3 | + switch/rollback supervision: BCE on "none of the planned actions remains ε-optimal"; CE on the previous step's case type |
| L4 | + counterfactual Q* regression for all available actions (Huber, Q/R) |

**Matching** (F14):
- same initialization per seed (torch seed 1000 + s);
- same config/world stream per seed;
- same sampling RNG seed;
- Adam lr 1e-3, grad-clip 1;
- the same number of updates and episodes: 4,000 × 64 = 256k episodes per run;
- all heads present in every rung;
- labels computed for every rung, and used only where the weight is > 0.

Seeds: 0, 1, 2.

**Metrics** (eval pools: 128 configs × 4 worlds = 512 episodes per split, for dev, test_iid, heldout_price, heldout_k and heldout_comp):
- **Free-running greedy:**
  - utility regret vs V*(I_0), realized;
  - **gap regret** Σ_t [V*(I_t) − Q*(I_t, a_t)]: an unbiased, lower-variance estimate of the same expectation (performance-difference identity);
  - absolute success, total cost, wrong commits;
  - case counts;
  - switches, meaning a change of action stage within a query, split into justified (ε-optimal) and unjustified and grouped by the preceding step's case type;
  - value-head calibration: 10 quantile bins of V̂ vs realized return-to-go, and vs V*;
  - Q-head vs Q* on taken actions;
  - build-rate vs ρ_k, ρ_eff and k, against π* on the same worlds;
  - a per-condition breakdown.
- **Teacher-forced** (π* histories): rate of argmax ∈ A*, mean Q* gap of the argmax.

Reference policies on the same worlds: π*, fixed exact-b2, fixed probe-first.

## 6. Supplied vs learned

| Mechanism | Supplied | Learned |
|---|---|---|
| environment, prior table, outcome model, event invalidation subset | yes | - |
| exact DP labels Q*, V*, A*, case, switch, stage, dependency (oracle, offline compute) | yes (training targets only) | - |
| available-action mask; queries-done fraction (public bookkeeping) | yes (inputs) | - |
| belief tracking, dependency validity, when to switch, strategy vs ρ_k | - | yes: L0 from utility only; L1–L4 with the supplied supervision above |

## 7. Smoke results and CPU budget

**Smoke** (dev, metered; reduced pools: 96 train configs, 32 per eval split × 4 worlds = 128 episodes; seed 0; 1,500 updates × 64). Figures are gap regret per episode (V* ≈ 230–340 per episode).

| rung | train CPU | dev | test_iid | heldout_price | heldout_k | heldout_comp |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 128 core-s | 107.3 | 96.9 | 60.0 | 82.6 | 87.5 |
| L4 | 148 core-s | 5.6 | 5.0 | 1.9 | 4.1 | 6.1 |

- **Reference regrets** (realized): fixed exact-b2 89–136; fixed probe-first 65–160.
- **Smoke L4 on heldout_k:** the build rate by ρ_eff bin [0, .5) / [.5, 1) / [1, 2) / [2, 4) was 0 / .29 / .92 / 1, against π* at 0 / .20 / 1 / 1. Smoke L0: 0 in every bin.
- **L0 at 1,500 updates** had not converged: its training regret was still falling, and its greedy policy never builds. **This is a smoke, not a result.**

**Full-ladder CPU estimate** (core-s):

| item | estimate |
|---|---:|
| labels: 384 train + 5 × 128 eval configs | ~0.4k |
| 15 training runs at ~0.1 core-s/update × 4,000 | ~5.5–6k |
| 15 evaluations | ~0.5k |
| references + summary | ~0.05k |
| **total** | **≈ 6.5k**, within the ~12k target |

- **Memory.** About 1 KB per DP state, so a train pool of 384 configs holds ≈ 2.5 GB per training process. Run ≤ 5 trainings concurrently.
- **Wall.** About 7 min per training run, single-threaded.

## 8. Launch (root only; CPU only; 1 thread per job)

Run these from a clean worktree at the probeworld commit, with `SHA` = `research/tools/campaign04_remote.py snapshot` output.
- `PY` = /home/brand/tensegra-campaign03/env/bin/python
- `R` = /home/brand/tensegra-campaign04/results
- `T` = research/tools/campaign04_probeworld_train.py

```
# 1. labels (one job, ~0.4k core-s)
campaign04_remote.py launch-cmd b-labels SHA -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PY $T labels --out $R/b-labels/labels
# 2. ladder: 5 rungs x 3 seeds (<= 5 concurrent; ~2.5 GB RSS each)
for rung in L0 L1 L2 L3 L4; do for s in 0 1 2; do
  campaign04_remote.py launch-cmd b-train-$rung-s$s SHA -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
    $PY $T train --labels $R/b-labels/labels --out $R/b-train-$rung-s$s/run --rung $rung --seed $s
done; done
# 3. evaluation (after each run) + references + summary
campaign04_remote.py launch-cmd b-eval-$rung-s$s SHA -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  $PY $T eval --labels $R/b-labels/labels --run $R/b-train-$rung-s$s/run
campaign04_remote.py launch-cmd b-refs SHA -- env CUDA_VISIBLE_DEVICES= $PY $T references --labels $R/b-labels/labels --out $R/b-refs.json
$PY $T summarize --runs $R/b-train-*/run --out $R/b-ladder-summary.json
```

Defaults: `--n-train 384 --n-eval 128`; train `--updates 4000 --batch 64 --lr 1e-3 --hidden 128`; eval `--worlds 4` (512 episodes per split).
