# Extended-07 P0b: gradient-flow audit (RAWF / LRN / SUP) and the 2×2 design spec

**Status:** static analysis plus mechanical tests, no protocol training (branch `campaign/e07-contracts`).

**Sources:**
- Code: `research/tools/campaign04_probeworld_train.py` (`ProbeNet`, `fuse_step`, `run_batch`, `losses`, `cmd_train`).
- Tests: `tests/test_campaign07_gradflow.py`. They build each arm through the trainer's own `train` command (`--updates 0` saves the untouched initial weights) and use the trainer's `run_batch` / `losses`.
- Numbers: [p0-audit/gradflow_report.json](p0-audit/gradflow_report.json).
- Coupling probe: `research/tools/campaign07_gradcoupling_probe.py` → [p0-audit/coupling_init.json](p0-audit/coupling_init.json) and [p0-audit/coupling_final.json](p0-audit/coupling_final.json).
- Factor semantics: [factor-contract.md](factor-contract.md).

## 1. Architecture as launched (extended-06 B-FACT / B-FACT-C)

Common to all arms (`--rung L1`, hidden 128, Adam lr 1e-3, 4,000 updates × 64 episodes, one global `clip_grad_norm_(all parameters, 1.0)`):

```
x_t = [public config vector (72 incl. v3) | prev record | action mask | qfrac] (+ 23 exact factors: SUP only)
          |
     inp: Linear(72 -> 128)            <- SUP: reads x[:, :72] only; factors bypass the GRU
          | tanh
     gru: GRUCell(128, 128)   h_t  ----(recurrent; not detached across steps: BPTT over the episode)
          |
   trunk: Linear(128,128) + tanh  ->  z_t  ("trunk state")
          |                          \
          |                    aux: Linear(128,128)-tanh-Linear(128,23)  -> phi_hat_t   [LRN only]
          |                          |                                         |
          |                          |  MSE(phi_hat_t, factor_features(public state_t))  (aux loss, LRN)
          |                         sg()                                       
          v                          v
   fuse: tanh(W_f [z_t ; phi_t] + b_f)   phi_t = 0 (RAWF) | exact factors (SUP) | sg(phi_hat_t) (LRN)
          |
          +--> pi (policy logits, masked)   <- policy-gradient, entropy, set-valued imitation
          +--> v  (value)                   <- value MSE (0.5)
          +--> q, stage, dep, switch, case  <- weight 0 under L1 (exact zero gradients)
```

**The historical LRN, verified exactly.**
- prediction = aux(z_trunk).
- The policy **does not** read concat(z, sg(pred)) directly. It reads **tanh(W_f [z_trunk ; sg(aux(z_trunk))] + b_f)**, a fusion layer that the value head and the unused heads share.

**The other arms.**
- **RAWF** is the same fusion layer with φ = 0.
- **SUP** feeds the exact factors to the fusion layer only. The recurrent state never sees them.
- **B0** (reference) has no fusion layer: the heads read z_trunk.

## 2. Verified gradient-flow table

Each loss term is back-propagated separately with `torch.autograd.grad`. The decomposition is checked to reproduce `losses()`'s total. The table lists the parameter groups that receive a **nonzero** gradient; "0" means an exactly zero gradient. The test asserts this table for every arm (`test_gradient_flow_table`, `test_total_l1_gradient_zero_on_unused_heads`).

| loss term (weight in L1) | inp | gru | trunk | aux | fuse (z cols + bias) | fuse (φ cols) | pi | v | q / stage / dep / switch / case |
|---|---|---|---|---|---|---|---|---|---|
| policy gradient (1; advantage detached) | ✓ | ✓ | ✓ | 0 | ✓ | LRN ✓, SUP ✓, **RAWF 0** | ✓ | 0 | 0 |
| entropy bonus (−0.01) | ✓ | ✓ | ✓ | 0 | ✓ | as above | ✓ | 0 | 0 |
| set-valued imitation (1) | ✓ | ✓ | ✓ | 0 | ✓ | as above | ✓ | 0 | 0 |
| value MSE (0.5; critic shares the trunk and fusion) | ✓ | ✓ | ✓ | 0 | ✓ | as above | 0 | ✓ | 0 |
| **aux factor MSE (1; LRN only)** | **✓** | **✓** | **✓** | **✓** | **0** | **0** | 0 | 0 | 0 |
| stage / dep / switch / case / Q\* (0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0 (exact zeros)** |

**Per module:**
- **Recurrent encoder** (inp + gru) and **shared trunk**: updated by every actor term and, in LRN, by the aux loss. The aux gradient at step t also flows back through h into earlier steps' GRU and input computations.
- **Auxiliary predictor** (LRN): updated **only** by the aux loss. The policy/value gradient is cut by the `detach` on φ.
- **Fusion layer:** updated by the actor terms only, never by the aux loss. RAWF's 23 φ columns never receive a gradient (their input is 0).
- **Policy head:** policy-gradient + entropy + imitation. **Value head:** value MSE only. There is **no Q or value head in the auxiliary path**. The Q\*/stage/dep/switch/case heads exist but carry weight 0 under L1: their gradients are exactly 0, so Adam leaves them at initialization.

**Gradient norms at initialization** (seed 35, 12 dev_smoke configurations, `gradflow_report.json`). In LRN the trunk receives:

| term | trunk gradient norm |
|---|---|
| value | .82 |
| imitation | .20 |
| policy gradient | .20 |
| aux | .12 |

The aux head itself receives .31.

## 3. Parameter counts (hidden 128)

| arm | total | exact-zero gradient under L1 on b6 data | trained |
|---|---|---|---|
| B0 | 129,701 | 3,354 (q, stage, dep, switch, case heads) | 126,347 |
| RAWF | 149,157 | 3,354 + 2,944 (fusion φ columns, input ≡ 0) = 6,298 | 142,859 |
| LRN | 168,636 | 3,354 | 165,282 |
| SUP | 149,157 | 3,354 + 128 (φ column of `exp_hard_cost_rel`, input ≡ 0 on b6) = 3,482 | 145,675 |

- The LRN − RAWF total difference of 19,479 equals the aux MLP (128·128 + 128 + 128·23 + 23).
- The **trained** LRN − RAWF difference is 22,423, because RAWF's φ columns are dead capacity.
- All totals match the historical `train_meta.json` (`test_parameter_counts`).

## 4. Initialization differences under the same seed

`cmd_train` calls `torch.manual_seed(1000 + seed)` and then constructs the model in this order:

`inp, gru, trunk, pi, v, q, stage, dep, switch, case, [aux], fuse`

The results (`test_initialization_differences`, using the trainer's own saved initial weights):

| pair | tensors that differ | tensors present in one arm only |
|---|---|---|
| RAWF vs SUP | none: all 24 tensors bitwise equal (SUP's `inp` has the same 72-dim input because the factors bypass it) | none |
| RAWF vs LRN | **`fuse.weight`, `fuse.bias`**: `aux` is created *before* `fuse` and consumes the RNG first | LRN: `aux.0.{weight,bias}`, `aux.2.{weight,bias}` |
| B0 vs RAWF / LRN | none among the common tensors (inp … case are bitwise equal) | fuse arms: `fuse.*` (+ `aux.*`) |

So LRN − RAWF compares two different fusion-layer draws. The extended-06 audit noted this; it is now pinned by a test.

## 5. Coupling paths from the aux loss into actor training (besides the trunk)

| path | historical behaviour | test / measurement |
|---|---|---|
| **Shared trunk / encoder** | The aux gradient is added to the actor gradient on inp, gru and trunk (BPTT through h). | `test_aux_coupling_paths`: pre-clip pi/v/fuse gradients are identical with and without the aux term; trunk/gru/inp gradients differ. |
| **Global-norm clipping** | One `clip_grad_norm_(all params incl. aux, 1.0)`. The aux gradient raises the norm, so it shrinks **every** actor parameter's step whenever the clip binds. | `test_aux_coupling_paths` asserts the clip coefficient falls when aux is added. Measured magnitude below. |
| **Optimizer state** | One Adam instance over all parameters. Adam moments are per parameter, so the aux head's state is separate. The **trunk/gru/inp moments mix actor and aux gradients**: the aux term changes the second-moment denominator and the step direction of shared parameters. | The Adam state difference follows from the shared gradients (the probe measures the gradient share). |
| **Loss normalization** | None adaptive. The total is rl + imit + 1.0·aux, each a mean. The aux MSE is an unweighted mean over 23 coordinates and all steps (`test_aux_mse_is_unweighted_mean`), dominated by `build_value_rel` / `cost_use_rel` (factor-contract.md). | |
| **Torch RNG** | Consumed only at construction. A full training step leaves `torch.get_rng_state()` unchanged. | `test_aux_coupling_paths` |
| **Python RNGs** | `data_rng` (configuration + world per episode) has fixed consumption, so every arm sees the same configurations and world seeds at every update (common random numbers on data). `act_rng` draws **one uniform per active decision**: once two arms' episode lengths differ, all later action-sampling draws are shifted. That is noise, not bias. | `test_action_rng_one_draw_per_decision` |
| **Compute** | LRN computes `factor_features` (23 Python-side targets) at every decision plus the aux forward/backward. SUP computes `factor_features` as inputs. RAWF computes neither. | Historical `cpu_s_total`, seeds 35–39 (mean): B0 399, **RAWF 438, SUP 453 (+3.4%), LRN 467 (+6.5%)** core-s. Seeds 30–32: RAWF 503, SUP 523, LRN 537. |

**Coupling magnitude.** The probe used the LRN trainer on the real b6_B0 pool (read-only), with dev_smoke worlds. Its columns:
- **clip binds:** fraction of updates where the total gradient norm exceeds 1 (with aux / actor alone);
- **clip coef ratio:** clip coefficient with aux divided by the coefficient for the actor alone;
- **aux share:** aux share of the squared gradient norm on inp/gru/trunk;
- **cos:** cosine between the actor and aux gradients on inp/gru/trunk.

| setting | ‖g_actor‖ | ‖g_aux‖ | clip binds (with aux / actor only) | clip coef ratio | aux share | cos | aux MSE |
|---|---|---|---|---|---|---|---|
| first 300 updates from the seed-35 init (Adam steps taken) | 4.09 | .21 | 97% / 97% | .993 | 1.1% | +.19 | .118 → .075 |
| final LRN s35 checkpoint (40 batches, no step) | 1.01 | .10 | 40% / 40% | .997 | 1.1% | +.20 | .0059 |

**Reading:**
- In gradient terms the auxiliary loss is a **small but consistently aligned** perturbation of the shared layers. Its magnitude on the shared layers is ≈ 10% of the actor gradient's (√1.1%), and its cosine with the actor gradient is +.2.
- Its effect through the clip coefficient is < 1%.
- Its effect on the trunk can still accumulate over 4,000 updates, because Adam normalizes per parameter and the direction is consistent.
- The Phase-2 2×2 is what can tell whether this shaping (S) matters. The table cannot.

**What LRN − RAWF bundled** (historical):
1. S, aux shaping of the trunk/encoder;
2. R, the policy reading sg(predictions);
3. a different fusion-layer initialization;
4. RAWF's 2,944 dead φ weights versus live ones in LRN;
5. the aux head's gradients inside the global clip norm (< 1% effect);
6. trajectory-induced desynchronization of the action RNG;
7. about 6.5% more CPU, with no behavioural effect.

## 6. Design spec: a genuine 2×2 (S × R). Specification only; not implemented.

**Factors:**
- **S** (shaping): does the aux loss update the trunk/encoder?
- **R** (reading): does the policy read the predictions?

**Arms:** S0R0, S1R0, S0R1, S1R1.
- Historical RAWF ≈ S0R0 without a predictor.
- Historical LRN ≈ S1R1 with a global clip and a different fusion init.
- SUP stays outside the 2×2, as an exact-input reference.

### 6.1 Model (identical modules, dimensions and initialization in all four arms)

```
z      = trunk(gru(tanh(inp(x_base)), h))                     # same as today
z_in   = z            if S1 else z.detach()                   # S: does the aux gradient reach the trunk?
pred   = aux(z_in)                                            # aux exists and is trained in ALL four arms
phi    = pred.detach() if R1 else phi_const                   # R: does the policy read it?  phi_const = zeros(23)
z_f    = tanh(fuse([z ; phi]))                                # heads read z_f (pi, v, and the zero-weight heads)
loss   = actor_L1(z_f) + w_aux * MSE(pred, factor_features(public state))   # w_aux = 1 in all four arms
```

**Constant input in R0.** `phi_const = 0`, as in RAWF, so the φ columns are dead in R0 and live in R1. That asymmetry is part of R. The alternative, a fixed training-mean target vector, only moves the fusion bias; it can be offered as a sensitivity arm.

**Consequences:**
- S0R0 is RAWF plus a **disconnected linear-readout probe** of the trunk. Its predictions measure how much factor information the actor-trained trunk carries, with no effect on the actor (see 6.2).
- **S0R1 still sees a trunk that changes through policy learning.** Its predictor reads a detached, drifting representation that is shaped only by the actor loss. The policy's φ input is therefore non-stationary, and prediction quality depends on what policy learning happens to put into the trunk. S0R1 measures "reading predictions of an unshaped trunk", not "reading good predictions".
- Contrasts:
  - S effect: S1R0 − S0R0;
  - R effect: S0R1 − S0R0;
  - interaction: S1R1 − S1R0 − S0R1 + S0R0.

### 6.2 Optimization: a disconnected aux loss cannot change actor training

- **Two parameter groups:**
  - **A** (actor) = everything except `aux.*`;
  - **X** = `aux.*`.
- **Two Adam optimizers** (opt_A, opt_X), same lr, and **two clips**: `clip_grad_norm_(A, 1.0)` and `clip_grad_norm_(X, 1.0)`.
- **One backward pass** on actor + w·aux is valid:
  - the actor loss never reaches X (φ is detached in R1 and constant in R0);
  - in S0 the aux loss never reaches A (`z.detach()`).

  So in **S0**, A's gradients, clip coefficient and Adam moments are exactly those of the actor loss alone.
- In **S1** the aux gradient reaches A's trunk/gru/inp. It therefore enters A's clip norm and Adam moments: that is the S treatment. The aux head's own gradients never enter A's clip.
- An optional `--clip-mode global` reproduces the historical coupling for a check against old LRN.

### 6.3 Initialization and RNG

- **Construction.** Construct once under `torch.manual_seed(1000 + seed)`. The module list and order are identical in all four arms (`inp … case, aux, fuse`), so every tensor is bitwise equal across arms.
- **Explicit copy.** Also support `--init-from PATH`, which loads a saved reference initial `state_dict` and asserts identical shapes. Log `sha256(initial state_dict)` in `train_meta.json` for every arm.
- **Torch RNG.** Unused during training (already true; keep the test).
- **Action sampling.** Replace the shared `act_rng` stream with **counter-based uniforms**: u(update, batch slot, step) from `random.Random(f"{seed}:{upd}:{slot}:{t}")` or a pre-drawn table. Common random numbers then survive trajectory divergence. `data_rng` is unchanged.
- **Common valid-history bank** (design Phase 2). The first stage trains all four arms on the same bank of public histories, with teacher-forced replay, imitation and aux losses, and no on-policy sampling. This makes the state distribution identical across arms. The free-running on-policy stage and evaluation come after.

### 6.4 Exact code changes (`research/tools/campaign04_probeworld_train.py`; defaults leave every historical path bit-identical)

1. **`ProbeNet.__init__`.** Add a `factor_mode="sr"` with attributes `shape: bool` and `read: bool`. For `arch == "fuse"` and mode `sr`, create `aux` then `fuse` exactly as `learned` does, in all four arms. Register a buffer `phi_const = zeros(23)` (identical across arms).
2. **`ProbeNet.fuse_step`.** Add the `sr` branch: `pred = self.aux(z if self.shape else z.detach())`; `self.last_aux = pred`; `phi = pred.detach() if self.read else self.phi_const.expand(len(z), -1)`.
3. **`run_batch`.**
   - Compute `aux_t` targets whenever the model has `aux` (modes `learned` and `sr`).
   - Add `sample_u(upd, slot, t)` behind `--action-rng counter`. This needs `run_batch` to receive `upd` and the slot index; it already has `i` and the step index `len(steps)`.
4. **`cmd_train`.**
   - New arguments: `--shape {0,1}`, `--read {0,1}` (imply `--arch fuse --factor-mode sr`), `--clip-mode {split,global}` (default `split` for `sr`), `--action-rng {stream,counter}` (default `counter` for `sr`), `--init-from PATH`.
   - Build groups A and X, `opt_A`, `opt_X`, and the two clips (or one global clip when `--clip-mode global`).
   - Set w_aux = `--aux-weight`, default 1.
   - Log `shape`, `read`, `clip_mode`, `action_rng`, `init_sha256`, and per-group gradient norms and clip coefficients every `--log-every`.
5. **`cmd_eval` / `replay_history` / model loading.** Rebuild `sr` models from `meta["fuse"]["shape"/"read"]`, and log `last_aux` predictions for the Phase-1 runner.
6. **Common-history stage** (Phase 2, separate flag `--bank PATH`). Teacher-forced replay of stored public histories: the `run_batch` "teacher" mode generalized to arbitrary stored action sequences. The loss is imitation + aux with RL off, followed by the on-policy stage.

### 6.5 Tests that would verify it (to be added with the implementation)

1. **Init identity.** For all four arms, the initial `state_dict`s are bitwise equal and the logged sha256 matches. `--init-from` reproduces the same weights.
2. **Gradient partition per arm.** Extend `test_gradient_flow_table`:
   - aux loss: S0 → {aux} only; S1 → {aux, trunk, gru, inp};
   - actor terms never reach aux (all arms);
   - R0: the fusion φ columns get exactly zero gradient; R1: nonzero.
3. **S0 isolation (the key test).** Train S0R0 for N updates with w_aux ∈ {0, 1, 1000}. The group-A parameters, the opt_A state and the sampled trajectories must be **bitwise identical**. For S0R1, where φ depends on aux by design: with the aux parameters held fixed, one actor step must not depend on w_aux (clip and optimizer isolation).
4. **Clip split.** In S1, A's clip coefficient is computed without the `aux.*` gradients, and `--clip-mode global` reproduces the historical coefficient.
5. **RNG.**
   - `torch.get_rng_state()` is unchanged by training.
   - With `--action-rng counter`, the uniforms for (upd, slot, t) are identical across arms even when episode lengths differ.
   - `data_rng` consumption is identical across arms.
6. **Compute parity.** All four arms compute `factor_features` and the aux forward/backward. Log per-arm CPU and check it matches within noise.
7. **Goldens.** The existing extended-04/05/06 golden and trainer tests (`test_campaign06_trackb.py`, `test_campaign06_bfactc.py`, `test_campaign05_*`) still pass unchanged. The `learned`, `supplied` and `none` modes are untouched.

## 7. Tests run

`tests/test_campaign07_gradflow.py` has 11 tests: the gradient table × 4 arms, zero heads, parameter counts, initialization differences, coupling paths, action RNG, LRN targets from public history, and aux MSE. Together with `tests/test_campaign07_factor_contract.py` (5 tests), all **16 pass** on pro6000 (`e07-dev-contracts-pytest`, 7.3 core-s).

## Compute

| where | job | CPU (core-s) |
|---|---|---|
| pro6000, metered | e07-dev-contracts-pytest | 7.3 |
| | e07-dev-contracts-gradreport | 3.1 |
| | e07-dev-contracts-gradreport2 | 3.1 |
| | e07-dev-contracts-coupling-final | 10.0 |
| | e07-dev-contracts-coupling-init | 41.7 |
| | e07-dev-contracts-pytest-final (17 passed, including `test_campaign06_trackb::test_factor_features_public`) | 7.4 |
| | **total metered** | **72.6** (dev budget 1,500) |
| local | pure-Python audit, Q\* comparison, contract tests, development | ≈ 25 |
