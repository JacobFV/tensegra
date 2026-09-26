# Protocol B2: does an own-continuation value head fix B-H6?

Registered 2026-09-26T20:45Z, before any B2 run. The only B2 data that exist are dev smokes on reduced pools (below). None of them was used to choose a threshold or a split. Parent: [protocol-B1.md](protocol-B1.md) (B-H6 NOT SUPPORTED, decisions.md 19:55Z). Spec: [probeworld.md](probeworld.md).

## Question

B-H6 failed: value-head reliability error was .13–.29 R on held-out splits, against .05 R. The suspected cause (decisions.md 19:55Z) is a **continuation mismatch**: the value head is not trained on the continuation it is scored against.

B2 tests this directly. It adds a head whose target *is* the scored continuation:

- **Continuation:** the model's own free-running greedy policy at the current parameters (`own_greedy_policy_current_params_mc_v1`).
- **Scoring:** calibration against realized return-to-go on the held-out splits, as in B-H6.

## Correction to the premise (read before interpreting)

The B1 value head `V` is **not** trained toward V*. In `losses()` its target is the Monte-Carlo return-to-go of the **sampled** (stochastic, on-policy) training rollouts: the actor-critic baseline, weight 0.5 inside the RL loss. V* appears in no loss; Q* is the L4 target of the separate Q head.

So the B1 mismatch is **sampled training policy vs greedy evaluation policy**, not π* vs own. Relevant B1 data, already seen and quoted here so it cannot be selected on later:

| rung | held-out reliability error vs own return | held-out reliability error vs V* |
|---|---:|---:|
| L4 | .03–.16 | .02–.15 |
| L1 | .07–.24 | .05–.23 |

- **The two columns nearly coincide**, because the greedy policy is near-optimal (regret 1–5 per episode on V* ≈ 230–340).
- **The in-distribution error is not small either:** L1 dev/test_iid .06–.16; L4 .03–.06.
- **Prior expectation:** a pure continuation fix may not suffice, and a generalization failure is at least as likely. B2 registers both readings (below) so that either outcome is informative.

## Design

**`--own-value`** option of `research/tools/campaign04_probeworld_train.py`. It is off by default, and off is **bit-identical to B1**: tested against a frozen copy of the B1 tool, on parameters, log, meta and eval.

- **Head:** `v_own = Linear(128, 1)` on the trunk features z, the same form as `V`.
  - It is created last in `__init__`, so every B1 parameter has the identical initialization.
  - Parameter cost: 129 of 129,318.
- **Stop-gradient:** v_own sees `z.detach()`.
  - It has its own Adam (lr 1e-3, grad-clip 1).
  - The main optimizer and the main gradient clip cover only the B1 parameters.
  - **The policy, trunk and every B1 head are therefore bit-identical to the B1 run with the same seed.** Dev check: L1 and L4, 150 updates on reduced pools gave identical parameter digests, identical training logs and identical values for every B1 eval key.
- **Target:** realized return-to-go / R of the model's **own free-running greedy rollouts**.
  - The rollouts are collected with `run_batch(..., "greedy", need_labels=False)`. No DP label is read; a test uses a solver stand-in that raises on any access.
  - **Frequency:** after every update, 64 episodes on train-pool configs. The last collection follows the last update.
  - **Loss:** one MSE Adam step per collection, over all steps of those 64 episodes. That makes 4,000 v_own steps and 256k greedy episodes per run.
- **Rollout RNGs and seeds:**
  - Config choice uses `random.Random(11000 + seed)`.
  - World seeds are 8e9 + 1e8·seed + 6e7 + n. These are disjoint from the B1 sampled-training worlds (< 8e9 + 1e8·seed + 256k) and from every eval world.
  - B1's data and action RNGs are not touched.
- **Inputs:** the unchanged public inputs, so no label enters an input.
- **Compute:** charged in `train_meta.own_value.cpu_s`, about +19% per run (dev smoke: 18.6 of 97.8 core-s).

**Runs:**
- rungs **L4** (registered) and **L1** (contrast), seeds 0, 1, 2;
- the B1 labels (`/home/brand/tensegra-campaign04/results/b-labels/labels`), with 4,000 × 64 updates and all other defaults as B1;
- the final checkpoint only.

**Evaluation:** the unchanged B1 eval: 512 episodes per split, all five splits. It adds:

| key | contents |
|---|---|
| `v_own_calibration_own_return` | same 10-quantile binning as `value_calibration_own_return` |
| `v_own_vs_vstar` | v_own against V* |
| `vstar_as_predictor_own_return` | V*(I_t) used as the predictor of own return: the **noise-floor reference** for the binned metric |
| `failure_prediction` | AUROC of −prediction for wrong commits; see below |

The `failure_records.json` file keeps the raw records.

Failure prediction:
- **Commit level:** every commit or commit_infeasible the model makes. The positive class is a wrong commit. Predictors are read at the pre-commit state.
- **Episode level:** predictors at the first step. The positive class is an episode with ≥ 1 wrong commit.
- **Predictors:** v_own, V, Q̂(taken), Q*(taken) (oracle reference), V*.

## Hypotheses and decision rules (per-seed values always reported)

Scored by `research/tools/campaign04_b2_score.py B2_DIR B1_DIR --models`.

| ID | Claim | Supported if |
|---|---|---|
| **B2-P (primary)** | Own-continuation targets make the value estimate calibrated | For **L4**, in **≥ 2/3 seeds**, v_own reliability error ≤ **.05 R** on **each** of heldout_price, heldout_k and heldout_comp (the B-H6 form). L1 is reported with the same rule, descriptive only. |
| **B2-S1 (policy unchanged)** | Stop-gradient leaves the policy untouched | Every B1 parameter tensor is bit-identical to the B1 `model.pt` of the same rung and seed. Every B1 eval key (including regret) is identical on all five splits. **Any difference is an implementation fault**, not a finding: B2 is then void until explained. |
| **B2-S2 (failure prediction)** | v_own anticipates wrong commits | Pooled held-out, commit level. In ≥ 2/3 seeds: AUROC(v_own) ≥ .70 **and** the lower end of the stratified bootstrap 95% CI (1,000 resamples) > .5. A seed with < 10 pooled wrong commits is "insufficient". B1 L4 had 3–20 wrong commits per seed across the held-out splits, so this may well be **insufficient**. Episode level and all other predictors are descriptive. |

**Registered reading of B2-P for L4** (the scorer emits it):
1. **B2-P supported:** continuation mismatch supported. Own-return targets calibrate where the jointly trained V head did not. The comparison is the per-seed V head pass/fail from the same eval.
2. **B2-P fails, but v_own passes .05 R on dev and test_iid in ≥ 2/3 seeds:** **not continuation.** The value estimate fails to generalize to held-out prices, k and compositions.
3. **Otherwise:** **not continuation.** The fit or capacity of a linear head on policy-shaped features is the bottleneck. `vstar_as_predictor_own_return` tells whether the metric itself is noise-limited: dev smoke floor .007–.04 in-distribution.

**Confound declared:** under stop-gradient, v_own is a linear probe on features shaped by the other losses. V, by contrast, can shape the trunk. So a B2-P failure could be a capacity effect rather than evidence that the continuation "does not matter". Reading 2 or 3 names this; B2 does not test a trunk-gradient variant.

Other declarations:
- The **.05 R threshold** is the B1 threshold, unchanged.
- **Nothing is tuned on held-out splits.** The frequency and episode count (1 × 64) were fixed from the smoke's cost only.

## Dev smokes (metered, not results)

**Bit-identity:** a frozen copy of the B1 tool (48286d97) against this tool with the option off and on.
- Settings: tiny (L4, 4 updates) and mid (L1 and L4, 150 × 64 updates, 24 train / 16 eval configs).
- Parameter digests, training logs, meta and eval were all identical. Option on: the B1 parameters and every B1 eval key were identical.
- Torch 2.14.0+cu130. The tiny digest is pinned in the unit test as `B1_GOLDEN`.

**Convergence smoke:** L4, seed 0, 800 × 64 updates, 48 train / 32 eval configs × 4 worlds.

| | dev | test_iid | heldout_price | heldout_k | heldout_comp |
|---|---:|---:|---:|---:|---:|
| V vs own return | .15 | .11 | .07 | .20 | .20 |
| v_own vs own return | .13 | .09 | .07 | .21 | .25 |
| V* vs own return (floor) | .01 | .04 | .12 | .04 | .10 |

- v_own tracked V closely, and training MSE fell to .04–.08.
- These are an undertrained 800-update smoke with regret 1–13 per episode. **Not a result.**

## Budget

- **Training:** 6 runs × ≈ 610 core-s. B1 took ≈ 505–520 core-s per run, plus about 19% for the own rollouts. ≈ 3.7k core-s.
- **Evaluation:** 6 × ≈ 45 core-s ≈ 0.3k.
- **Total ≈ 4.0k core-s** CPU-only.
- **Memory and wall:** ~2.5 GB RSS per training. Run ≤ 6 concurrently (the machine is shared). ~9 min wall each.
- **Dev spent** (metered, results/dev/b2-*-process): 368 core-s.
  - tests-new 16;
  - identity 125;
  - convergence smoke 174;
  - full test file 52 (26 passed).
