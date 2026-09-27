# Extended-07 Phase 2 infrastructure (branch `campaign/e07-p2infra`)

**Status:** built, tested and smoked. Nothing here is registered and nothing has been launched; the root decides which Phase-2 experiments run once the P1 analysis is in.

**Code:**
- `research/tools/campaign04_probeworld_train.py`: new options, all additive and default off.
- `research/tools/campaign07_p2.py`: `init`, `bank`, `oof`, `seeds` and `jobs`.
- `research/tools/campaign07_diag.py`: S×R and consumer checkpoints.
- `research/tools/campaign07_p2_costs.py`: smoke cost table.
- `scripts/campaign07_p2_smoke.sh`: the metered smokes.
- Tests: `tests/test_campaign07_p2.py`.

## 1. Genuine S × R 2×2 (gradient-flow.md §6, implemented)

`train --arch fuse --shape {0,1} --read {0,1}` selects factor mode `sr`. All four arms build the identical module list `inp … case, aux, fuse`, so every tensor has the same shape and position.

```
pred = aux(z if S1 else sg(z))                      # the aux head exists and trains in all four arms
phi  = sg(pred) if R1 else phi_const                # heads read tanh(W_f [z; phi])
loss = actor_L1 + w_aux * MSE(pred, exact factors)  # bank stage: imitation + w_aux * MSE
```

- **Common initialization.** Construct once per seed with `campaign07_p2.py init --seed S`, which saves `init.pt` and its sha256. Every arm then trains with `--init-from init.pt` (a strict copy: same keys and shapes) and optionally `--init-sha`. `train_meta.p2` logs both the constructed hash and the copied hash. Because the aux layer is created before fuse in every arm, the fusion columns and the aux head are bitwise equal across arms.
- **R0 constant.** `--phi-const zeros` is the default: `phi_const = 0`.
  - It is the historical RAWF control.
  - It is data-free, so it is identical across arms and seeds by construction.
  - It is a non-persistent buffer: it is not in the state_dict, which keeps the common-init hash equal in all four arms. It is restored from `train_meta.fuse.phi_const`.
  - Consequence: in R0 the 2,944 φ columns are dead, and in R1 they are live. That difference is part of R.

  `--phi-const train_mean` is a sensitivity option. It uses the mean of the exact factor labels over all bank decisions (it needs `--bank`), and it is the same vector for every R0 arm. It only shifts the fusion bias by W_φ·m.
- **Optimization.** `--clip-mode split` is the default for S×R.
  - Group A (everything except `aux.*`) and group X (`aux.*`) each get their own Adam and their own `clip_grad_norm_(·, 1.0)`. `gnorm_A` and `gnorm_X` are logged.
  - **In S0** the aux loss never reaches A. A's gradients, clip coefficient and Adam moments are therefore exactly the actor's.
  - **In S1** the aux gradient enters A only through inp/gru/trunk, via backprop through `aux(z)`, including BPTT over earlier steps. It is part of A's clip norm and Adam moments; that is the S treatment. The aux head's own gradients never enter A's clip.
  - **Historical-coupling reference:** `--clip-mode global --action-rng stream` uses one Adam and one clip over everything. With that flag, S1R1 reproduces historical LRN **bit for bit** (tested).
- **Action RNG.** `--action-rng counter` is the default for every P2 run. The action uniform is `Random("e07-act:<seed>:<episode world seed>:<step>")`, so arms keep common random numbers after their trajectories diverge. `data_rng` is unchanged, and torch RNG is consumed only at construction (tested).
- **Aux weights per group.** `--aux-group-weights G1=… G4=…` uses the factor-contract groups; unnamed groups keep weight 1. The default is equal weights, as historically. Note that `build_value_rel` carries 32% of the target variance and dominates the unweighted MSE. The default is deliberately not changed.

## 2. Common valid-history bank (brief §12)

**Building the bank.** `campaign07_p2.py bank --labels TB --pool b6_B0 --source … --world-base W` records valid public histories on the **training pool b6_B0 only**. Sources:
- `pistar:R`: pi*, with the lowest-index tie-break;
- `eps:E:R`: pi* with probability-E uniform deviations, counter-RNG seeded;
- `model:NAME=RUN:R`: a frozen checkpoint sampling on-policy with a counter RNG.

Each decision carries exact labels:
- the available set;
- the solver `opt_set` (the trainer's imitation target);
- Q\* over the available set and V\*;
- the 23 exact factor targets;
- the reward and the return-to-go.

The file is written with exclusive create. A `.json` sidecar holds the sha256, per-source coverage (behaviour eps-optimal rate and regret) and per-combination counts.

**Proposed protocol bank:** pistar:2, eps:0.1:2, eps:0.3:2, RAWF-s30:4, B0-s30:4.
- The model sources are the **dev lineages** (s30), never the P1/P2 evaluation lineages.
- This gives 384 configurations × 14 = 5,376 episodes, shared by every arm and every seed.

**Training on the bank.** `train --bank F --bank-updates N`:
- It replays stored histories teacher-forced. The inputs are rebuilt exactly as in `run_batch`; the test asserts that replay equals the on-policy forward bitwise.
- The loss is set-valued imitation plus w·aux. RL is off, because the stored actions are not the model's. The value loss is optional (`--bank-value-weight`, default 0).
- The episode stream is `Random(7500 + seed)`, identical for every arm of a seed.

All arms therefore get the same label access, the same coverage and the same optimizer budget (N × batch).

**Optional stages.**
- On-policy fine-tuning is separately flagged: `--finetune --updates U`. Without the flag, `--updates > 0` is refused.
- Free-running evaluation is done with `campaign07_diag.py`.

## 3. Controlled consumers (brief §10)

**Choice: a from-scratch recurrent consumer with the SUP architecture.** The factors enter only the fusion layer, and the GRU reads the same public inputs. This is trained on the bank under an input contract; the only thing that differs between contracts is φ.

A consumer on a **frozen** public-context representation from a historical checkpoint was rejected as the default. That trunk was co-trained with its own policy under on-policy RL and already encodes that policy's decision computation, so the consumer could route around φ. It would also tie every consumer to one lineage. The frozen variant is still available as `--frozen-trunk RUN`, which loads inp/gru/trunk and freezes them.

**Predictors.** An LRN-style aux head on its own encoder: `--shape 1 --read 0 --predictor-only`, trained on the bank with the aux MSE only.
- K-fold: K = 5 folds over **training configurations**, never episodes. The folds are stratified by combination and seed-free (`bank_folds`), and are selected with `--bank-folds K --bank-exclude-fold k`.
- One full-data predictor is trained for deployment.

**Out-of-fold predictions.** `campaign07_p2.py oof` predicts every bank decision with the fold model that never saw its configuration; this is asserted, as is the bank hash. It writes:
- per-coordinate RMS and MAE;
- per-group nMAE;
- the full predictor's in-sample error.

**Contracts** (`--phi-contract`):

| contract | φ at training |
|---|---|
| `exact` | the exact labels |
| `oof` | the out-of-fold predictions |
| `mix` | each sampled episode is noisy with probability `--mix-p` (declared default .5): exact + σ_c·N(0,1) per decision and coordinate, where σ_c is the OOF RMS of coordinate c; deterministic in (noise seed, seed, update, slot); unclipped |

- The oof and mix contracts are bank-only; on-policy fine-tuning would read exact values and is refused.
- Consumers train only on the allowed training combinations: the b6_B0 combinations none, U, S, C, E and U+S. S+C+E is never seen. `--bank-combos` can restrict further.

**Evaluation** uses `campaign07_diag.py run` with the same record schema. Model specs:
- `NAME=RUN::exact` gives kind CONS-exact;
- `NAME=RUN::pred=PREDRUN` gives kind CONS-pred. The predictor runs alongside on the same public history, with state [consumer h | predictor h], and interventions are allowed.

Populations: b6c_hold_SCE, SCE octets and UCE octets. Protocols B (free-running) and A-pistar are run with the ivs `none` and `exact`.

**Compute** is reported separately: every predictor, oof and consumer is its own job with its own receipt, and `train_meta.p2.cpu_s_bank` and `cpu_s_onpolicy` are logged.

## 4. Diag runner on new checkpoints

- **Historical runs** load through the identical constructor call (tested).
- **S×R kinds** are S0R0, S1R0, S0R1 and S1R1.
  - `pred` is always logged. In R0 it is the disconnected probe of the trunk.
  - Interventions are allowed only where the policy reads the prediction (S0R1, S1R1, CONS-pred). R0 is refused, like RAWF.

## 5. Tests (`tests/test_campaign07_p2.py`, 26 tests at aca42d90, 31 after §10; all pass on pro6000)

The mechanical tests:

| test | what it shows |
|---|---|
| (e) historical goldens | B0/RAWF/LRN/SUP after 3 updates are bitwise equal to goldens captured with the **unmodified** trainer (base c3974f5c; receipt `e07-dev-p2-golden`, 3.0 core-s); a direct old-copy comparison runs when `_ref/` is present, and passed |
| S1R1 historical coupling | S1R1 with `global` + `stream` equals historical LRN bit for bit (model digest and losses) |
| (c) common init | all four arms start bitwise identical (same init sha, including fuse and aux); `init` equals the trainer's construction; `--init-from` copies exactly; bad sha or bad keys are refused |
| (a) S0R0 isolation | aux weight 0 / 1 / 1000 gives bitwise-identical actor parameters, trajectories and gnorm_A after 4 updates, while the probe differs; it also holds in bank mode |
| (a) S0R1 | one actor step is independent of w; after 2 steps the arms differ, by design |
| (b) R0 | the actor gradient on aux is exactly 0; policy logits are invariant to perturbed aux parameters; the φ columns get zero gradient; R1 reads the prediction (logits change) but its actor gradient on aux is still 0 |
| (d) S1 vs S0 | identical forward; identical gradients on pi, v, fuse and aux; the trunk/gru/inp gradient difference equals the aux-loss trunk gradient |
| split clip | gnorm_A is the norm of A's gradients only |
| RNG | the counter RNG gives the same uniform per (seed, episode, step) across arms; torch RNG is untouched by training |
| bank | labels are exact (re-derived along each history); world seeds are unique; replay equals the on-policy forward bitwise; bank training is identical across arms (aux loss equal in all four, imitation loss equal within each R level); `--finetune` is required for on-policy updates |
| predictor-only | only aux, inp, gru and trunk change |
| folds | assigned by configuration and stratified |
| OOF | every prediction comes from the fold model that excluded that configuration; mismatched fold runs are refused |
| consumers | the three contracts, deterministic mix, SUP architecture enforced |
| PredictedConsumer | forward equals the consumer reading the predictor's output |
| diag | runs on the S×R arms and consumers (kinds, interventions only on reading kinds, free-run equals greedy `run_batch`) |
| seeds, jobs | seed-range disjointness; job-matrix validation |

**Regression suites.** These pass unchanged together with the new tests (106 passed; `e07-dev-p2-pytest3`, 260 core-s):
- `test_campaign07_{gradflow,diag,factor_contract}`;
- `test_campaign06_{trackb,bfactc}`;
- `test_campaign05_{trackb,bxc}`.

## 6. Smoke (pro6000, metered `e07-dev-p2-smoke-*`, seed 47, 24 b6_B0 configurations, batch 64)

Every batch type exited 0.

| batch type | job CPU (core-s) | peak RSS (MiB) | unit cost |
|---|---|---|---|
| bank (3 sources × 24 configurations = 72 episodes, 855 decisions) | 4.0 | 2,150 | 0.038 core-s/episode |
| init | 1.1 | 623 | — |
| 2×2 bank stage, per arm (20 updates) | 5.7–6.0 | 2,271 | 0.037–0.038 core-s/update |
| 2×2 on-policy, S0R0 and S1R1 (10 updates) | 6.1–6.2 | 2,274 | 0.095–0.100 core-s/update (early updates) |
| predictor (fold 0, fold 1, full; 20 updates) | 5.6–5.7 | 2,273 | 0.027–0.037 core-s/update |
| oof (K = 2) | 1.1 | 638 | — |
| consumers exact / oof / mix (20 updates) | 5.6–5.8 | 2,270 | 0.032–0.034 core-s/update |
| eval: diag on b6c_hold_SCE (24 configurations) + 8 SCE octets, 10 model entries | 5.0 | 976 | 6.5 ms per configuration × model; 0.016–0.025 core-s per octet × model |

- **Fixed cost per training job:** about 2 core-s of b6_B0 label loading, plus about 1.5 core-s of start-up.
- **RSS:** about 2.3 GB, dominated by the b6_B0 DP tables. The bank itself is small.
- **Smoke dev total:** 27 metered smoke steps over two passes, ≈ 126 core-s in all (including the first smoke, whose bank contained only 'none' configurations; that bug is fixed: `--n-configs` now spaces the configurations evenly).

## 7. Proposed seeds and world ranges (not registered)

Checked programmatically by `campaign07_p2.py seeds` and `test_seed_ranges_disjoint` against seed-ranges.json: no clashes.

| use | proposal | range |
|---|---|---|
| P2 lineages | seeds 40–44 | training worlds 8e9 + 1e8·s = [1.20e10, 1.25e10); on-policy / fine-tuning, plus support-reference offset 7e7 if the diag `support` command is used |
| smoke seed | 47 | [1.27e10, 1.28e10) (dev) |
| test seed | 97 | [1.77e10, 1.78e10) (dev) |
| protocol bank worlds | W = 1.29e10 | W + 1e6·source + 1000·cfg_idx + r, within [1.29e10, 1.291e10) |
| smoke bank worlds | W = 1.295e10 | [1.295e10, 1.296e10) (dev) |
| P2 test configurations | dev_smoke | 6.890–6.891e9 (dev; nested in the ext06 dev_smoke sub-range, like the ext07 P0/P1 dev blocks) |

## 8. Launch commands and full-run cost estimates

`python research/tools/campaign07_p2.py jobs <SNAPSHOT_SHA> [--onpolicy 4000] [--finetune U] [--bank-updates N]` prints the validated `campaign07_remote.py launch-cmd` lines.
- Job names are `e07-p2-*`, each with `--needs` and `--mkdir`.
- They are ordered bank → init → 2×2 → predictors → oof → consumers → eval. `--needs` refuses a job whose inputs are missing.

Estimates for 5 seeds (40–44), using the smoke unit costs and 4,000 updates × 64 as the historical optimizer budget:

| package | jobs | estimate (core-s) |
|---|---|---|
| bank (5,376 episodes) | 1 | ~210 |
| init | 5 | ~6 |
| 2×2 on the bank (4,000 updates) | 20 | 20 × ~155 = ~3,100 |
| predictors (5 folds + full, 4,000 updates) | 30 | 30 × ~135 = ~4,000 |
| oof | 5 | ~50 |
| consumers × 3 contracts | 15 | 15 × ~137 = ~2,050 |
| eval (b6c_hold_SCE 400 + SCE 320 + UCE 160 octets; 4 arms + 6 consumer entries) | 5 | 5 × ~140 = ~700 |
| **bank-mode package total** | 81 | **≈ 10,100 (≈ 12k with a 20% margin)**; ≈ 45 min wall at 4 concurrent |
| optional: historical-style on-policy 2×2 (`--onpolicy 4000`) | +20 | +20 × ~470 = ~9,400 (from the historical LRN 474 core-s / 4,000 updates), plus eval +~40% |
| optional: on-policy fine-tuning after the bank (`--finetune 4000`) | 0 extra jobs | +~9,400 |

Predictor cost can be halved with `--bank-updates 2000` for predictors. The job matrix uses one value for all stages; edit it if needed.

**Memory:** about 2.3 GB per job. Four concurrent jobs is well under the launch helper's 16 GiB free-memory guard.

## 9. Compute used by this builder

| where | what | core-s |
|---|---|---|
| metered, pro6000 | golden 3.0; pytests 3.7 + 5.4 + 260.1 + 44.7 + 6.0 + 113.9; smokes ≈ 147 | ≈ 584 (dev cap 3,000) |
| local (GB10 host; editing and compiling only, no torch) | | ≈ 5 |

## 10. Revision after the P1 diagnosis (coordinator priorities: consumers, then separate-predictor READ arm, then 2×2)

This revision builds on top of the snapshot c7d22aff that was launched as P2-SCREEN. Everything in §1–§9 is unchanged; everything here is additive, and the historical goldens still pass.

**Priority 1: controlled consumers whose training contract forces factor reliance.** These were already built (§3): the exact and exact+noise (`mix`) contracts, evaluated on both exact and predicted inputs. The predictor used for the predicted inputs is **never the consumer's own trunk**. Two predictor sources are now supported:
- **Separately trained predictor** with its own encoder: K-fold out-of-fold on the training configurations for training, and the full-data predictor at evaluation (`::pred=<pred-ffull run>`).
- **Historical LRN aux predictions**, new: `::pred=<e06-tb-lrn-sXX/run>`. The LRN's own encoder, trunk and aux head are replayed on the same public history.
  - The test asserts the prediction equals the historical LRN's `last_aux` bitwise.
  - The job matrix pairs P2 seed 40+i with LRN lineage 35+i, in stage `evalX`.

**Priority 2: separate-predictor READ arm** (`--arch fuse --factor-mode sep`, kind `SEP`). The predictor has its own recurrent encoder, `pinp/pgru/ptrunk + aux`, trained by the factor MSE only. Its **detached** outputs feed the fusion layer read by the policy and value heads. The factor channel is therefore not a function of the policy trunk.
- The recurrent state is [policy h | predictor h] (`state_size`).
- By default it uses the split optimizer and clip: X = aux + the predictor's encoder, A = everything else.
- Shared tensors are copied from the seed's common S×R `init.pt` (`--init-from` accepts the S×R init for `sep`). The predictor encoder keeps its own construction draw.
- It trains on the bank (imitation + aux) or on-policy (`--updates`, RL + imitation + aux), like LRN.
- Tests:
  - the actor loss puts exactly zero gradient on the predictor;
  - the aux loss reaches only the predictor, never inp/gru/trunk;
  - the policy reads the prediction;
  - one actor step does not depend on w_aux (0 / 1 / 1000);
  - bank replay equals the on-policy forward bitwise;
  - the diag runner accepts it, and interventions are allowed.

**Priority 3: the S×R 2×2** is unchanged (§1).

**Near-miss.** The diag runner logs every octet member, including `role = near_miss`, for every evaluated model, so near-miss accuracy is first-class in the per-decision records. It is scored as in P1.

**Job matrix.** `campaign07_p2.py jobs SHA [--stages …]`. Stages run in priority order:
1. bank, init;
2. P1-predictor, P1-oof, P1-consumer;
3. P2-sep (+ P2-sep-onpolicy with `--onpolicy U`);
4. P3-2x2-bank (+ P3-2x2-onpolicy);
5. evaluation:
   - evalA: consumers on exact and own-predictor inputs;
   - evalX: consumers on historical-LRN predictions, plus SEP;
   - evalB: the 2×2.

The launched P2-SCREEN (c7d22aff) already has bank, init, predictors, oof, consumers, the 2×2 and a combined `e07-p2-eval-s*`. **The only additions it needs are `--stages P2-sep evalX`** (10 jobs, names `e07-p2-sep-bank-s4x` and `e07-p2-evalX-s4x`).

**Smoke (metered, seed 47, all exit 0):**

| job | CPU (core-s) | peak RSS | unit cost |
|---|---|---|---|
| `e07-dev-p2-smoke-sep-bank` (20 bank updates) | 6.9 | 2,281 MiB | 0.052 core-s/update |
| `e07-dev-p2-smoke-sep-onp` (10 on-policy updates) | 6.6 | 2,284 MiB | 0.116 core-s/update |
| `e07-dev-p2-smoke-eval` (15 model entries on 24 b6c configurations + 8 SCE octets: 4 S×R arms, SEP-bank, SEP-onp, 3 contracts × {exact, own predictor, historical LRN s35}) | 7.2 | 984 MiB | — |

**Added cost for 5 seeds:**

| addition | estimate (core-s) |
|---|---|
| SEP bank arm (4,000 updates) | 5 × ~215 = **~1,080** |
| evalX (4 model entries) | 5 × ~60 = **~300** |
| optional SEP on-policy arm (4,000 updates) | +5 × ~470 = ~2,350, plus its evalX entry |

**Seed-range note.** The smoke bank used worlds 12,950,000,000 + 1e6·j + 1000·idx + r with j < 3. That is **up to 12,952,024,000**, which is beyond the registered smoke-bank entry [12.950e9, 12.951e9). Please widen that entry to [12.95e9, 12.96e9) as originally proposed in §7. The dev test seed 97 (worlds [1.77e10, 1.78e10)) is also still unregistered.
