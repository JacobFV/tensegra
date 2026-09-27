# Extended-05 Track B tooling (B-LOC, B-SPLIT, B-HR, B-X)

**Status:** built on `campaign/e05-trackb` (from `campaign/extended-05` @ d3e1d279, design v2). Tooling, split audit,
hold sizing estimate and the B-LOC localization of the existing models are done. **No B-X model has been trained**;
all training/evaluation launches below are root launches.

| file | role |
|---|---|
| `src/tensegra/campaign05_probeworld.py` | split table v2, supplied public state (BO/BX2), decision records, sizing rule, audit helpers (pure Python) |
| `research/tools/campaign04_probeworld_train.py` | additive options, all default off: `labels --split-set b5`, `train --train-split/--inputs/--arch`, `eval --split-set b5 --episode-rows --split-worlds`, `references --split-set b5` |
| `research/tools/campaign05_bloc.py` | B-LOC localization (torch; remote) |
| `research/tools/campaign05_bsplit_audit.py` | split-support audit, hold sizing estimate, BX2 triviality (pure Python) |
| `research/tools/campaign05_bx_score.py` | B-HR gate and B-X endpoints with configuration-clustered bootstrap CIs |
| `research/tools/campaign05_trackb_smoke.sh` | metered dev smoke (cost per update / eval) |
| `tests/test_campaign05_trackb.py` | 13 tests, incl. extended-04 bit-identity goldens |
| `research/results/campaign-05/trackb/` | `bsplit-audit.json`, `hold-sizing-estimate.json`, `bx2-triviality.json`, `bloc-existing.json` |

## 1. Bit-identity of the extended-04 paths (B1/B2/F2)
- Goldens were captured with the **unmodified** trainer on the pro6000 (torch 2.14.0+cu130, metered
  `trackb-golden-20260927T013415`): label pickles and labels_meta (B1 split set), B1 train L1/L4 (state dict, log,
  meta), B1 eval (offset 500), F2 eval (offset 700), B2 `--own-value` (params, eval, failure records), references.
- `test_b1_b2_f2_paths_bit_identical_under_defaults` reproduces all of them with the modified tool (pass). The
  extended-04 split table is pinned separately (`test_extended04_split_table_unchanged`); `campaign04_probeworld.py`
  is not modified.
- Every new key/file (`train_split`, `inputs`, `arch`, `split_set`, `*_episodes.jsonl.gz`, shards) is absent under
  defaults.

## 2. B-SPLIT: split table `probeworld-split-v2` (design v2 revisions 10-12)
Flags U (q < 1), S (D_side > 0), C (corr > 0), E (p_event > 0). All 6 pairs are **generable**: `make_config` samples
each flag's parameter independently, so no pair needs an alternative.

| split | cells | k | combos | config seed base | pool |
|---|---|---|---|---|---|
| b5_train | 8 train cells | 1, 2, 8 | none, U, S, C, E, U+S, C+E | 5.1e9 | 384 |
| b5_dev (selection) | train cells | 1, 2, 8 | same 7 | 5.2e9 | 128 |
| b5_test_iid (fresh instances) | train cells | 1, 2, 8 | same 7 | 5.3e9 | 128 |
| b5_heldout_price (new continuous values) | centre cell | 1, 2, 8 | same 7 | 5.4e9 | 128 |
| b5_heldout_k (unseen intermediate horizon) | train cells | 4 | same 7 | 5.5e9 | 128 |
| **b5_hold_uc** (new hold) | train cells | 1, 2, 8 | U+C | 5.6e9 | sized (sec. 3) |
| **b5_hold_se** (new hold) | train cells | 1, 2, 8 | S+E | 5.7e9 | sized (sec. 3) |
| b5x_train (BX1 exposure only) | train cells | 1, 2, 8 | the 7 + U+E, S+C | 5.8e9 | 768 |
| heldout_comp (historical challenge, extended-04 table) | train cells | 1, 2, 8 | U+E, S+C | 4.6e9 (unchanged) | 128 |

- U+C and S+E are in **no** training, dev/selection, test_iid, price or k configuration (the extended-04 table had
  both in training and dev; it is kept unchanged for historical reproduction).
- World seeds: `base + 5e7 + 1000 i + r`; B-X evaluates at offset 900 (B1 used 500, F2 700 on heldout_comp).
- **How compositional are the holds?** Exact regret, in the pair's world, of acting optimally for the configuration
  with one flag switched off (k = 2, 32 configs per pair; `bsplit-audit.json` "pairs"):

  | pair | role | drop U / S | drop C / E | both > eps |
  |---|---|---:|---:|---:|
  | U+C | new hold | 60.7 (91%) | **0.42 (16%)** | 16% |
  | S+E | new hold | 8.4 (59%) | **0.29 (12%)** | 12% |
  | U+E | historical | 54.6 (94%) | 1.84 (38%) | 34% |
  | S+C | historical | 6.7 (53%) | 0.11 (9%) | 9% |

  Both new holds are genuine but **weak** compositions: the C / E component alone changes the optimal policy by > eps
  in only 12-16% of configurations (U+E, the historical pair where the extended-04 models failed, is the strongest).
  Registered mitigations (v2): U+C is also reported on its flag-sensitive subset and by k stratum (C is inert at k = 1:
  14/172 configurations sensitive in the sizing sample).

## 3. Split-support audit (`campaign05_bsplit_audit.py audit`, local pure Python, 45 core-s): **18/18 PASS**
- A1 all 6 pairs generable; A2 both new holds require both conditions on a non-zero fraction.
- B1-B6 generator-parameter sets: families disjoint (except the BX1 pool by design), holds absent from every
  training/selection combo set, historical pairs only in b5x_train, every hold flag seen singly in b5_train.
- C1-C6 generated configurations at full pool sizes (holds: first 4,000): parameters recovered from public prices
  (2 configurations sit within price rounding of a bin edge and are consistent with their generator cell), no hold
  pair in any training/selection configuration, `split_of_params` (v2) finds no leak, no duplicate configuration
  (also vs extended-04 train/dev/heldout_comp), configuration and world seed ranges disjoint from every extended-04 range.
- D1-D3 labels **and the BX2/BO features** are functions of the visible history (2,071 distinct random-policy histories,
  0 conflicts); simulator state = public state rebuilt from the history.

**Hold sizing (revision 11).** Registered rule (applied by the labels job, before any model run): each hold pool is
the smallest prefix of its configuration stream with **>= 60 configurations whose first action is uniquely probe**
(U+C additionally >= 20 of them in the flag-sensitive subset), cap 4,000, evaluated with 1 world per configuration.
Estimate from the exact solver (`hold-sizing-estimate.json`, local 253 core-s):

| hold | sample | s0 uniquely-probe | projected pool (rate +- 2 se) | flag-sensitive | label CPU |
|---|---:|---:|---:|---:|---:|
| U+C | 480 | 22 (4.6%; k=1: 8, k=2: 12, k=8: 2) | 1,309 (924-2,244) | 110/480 (10 eligible) | ~0.45 core-s/config, ~590 |
| S+E | 160 | 33 (20.6%) | 291 (222-422) | n/a | ~0.24 core-s/config, ~70 |

## 4. B-X arms (trainer options; matched updates 4,000 x 64, L1 recipe, final checkpoint)

| arm | command options | inputs / architecture | params |
|---|---|---|---|
| B0 | `--train-split b5_train --rung L1` | public (68) | 129,189 |
| BX1 | `--train-split b5x_train --rung L1` | public; 768 configs incl. U+E, S+C (labelled) | 129,189 |
| BX2 | `--train-split b5_train --rung L1 --inputs bx2` | public + 21 supplied features (labelled supplied) | 131,877 (+2.1%: input layer only) |
| BX3 | `--train-split b5_train --rung L1 --arch modular` | public; 4 flag-gated encoders `tanh(Wx + sum_f flag_f MLP_f([param_f, x]))`, hidden matched | 128,905 (hidden 121; -0.2%) |
| BO | extended-04 labels, `--rung L1 --inputs belief`, seeds 0-2 | public + exact belief (4); same data stream as B1 L1 s0-2 | 129,701 |

BX2 features (`BX2_FEATURES`): exact belief over theta (4); miss rate 1-q, P(false 'solved'), P(H | 'solved');
D_side, expected side cost; corr, revealed H / not-H fractions; active event hazard, event fired; remaining queries,
last query, built, amortized build over the remaining queries, c_use, c_b2, log rho over the remaining horizon.

**Does BX2 make the decision trivial?** No (`bx2-triviality.json`, pi*-visited decisions, 80 b5_train and 2 x 40 hold
configurations x 8 worlds, local 53 core-s). A majority-action table from training decisions, keyed on:

| key | all hold decisions (7,188): determined | first decision (640): determined |
|---|---:|---:|
| bookkeeping only (usage, candidate, reduction, event, built) | .82 | .41 |
| belief only | .35 | .34 |
| belief + bookkeeping | .41 (coverage .50; .81 on covered) | .34 |
| belief + bookkeeping + flags + remaining horizon (in-sample purity on the holds) | .92 | .71 |

Most later decisions are forced by bookkeeping (commit after an exact solution, and so on), with or without the belief.
The decision the extended-04 models get wrong is the **first** one, and there the belief adds nothing: it equals the
declared prior, already a public input (review F14; BX2 is predicted null at s0). Even with every discrete BX2 feature,
29% of first decisions on the holds need the continuous price/horizon/reliability trade-off. So BX2 is a
supplied-belief **control**, not a solved task.

## 5. Evaluation (B-X endpoints; `campaign05_bx_score.py`)
- Per arm x seed and group (new holds pooled, each hold, U+C flag-sensitive, U+C k = 1 / k > 1, challenge U+E / S+C,
  dev, test_iid, price, k), per family (combo, k):
  - s0 uniquely-optimal probe rate, with support in **configurations**, plus a per-query-first-decision variant;
  - not-optimal probe rate;
  - decision accuracy and mean delta **by decision depth** (s0 / later query-first / within query);
  - near-boundary accuracy and delta by rho_eff bin, and first-decision accuracy by |probe margin| bin;
  - Q-ranking accuracy (n/a for the L1 recipe: the Q head is untrained);
  - realized and gap regret; success, and the floor >= .8 x pi* on the same worlds.
- **CIs.** 95% bootstrap over configurations (1,000 resamples), paired for arm - B0. Support < 20 is marked insufficient.
- **B-HR gate.** Run from B0 alone (`--bhr-only`), per hold:
  - no_failure: >= .80 with >= 20 eligible configurations in >= 2/3 seeds;
  - evaluable: <= .70 in >= 2/3 seeds;
  - otherwise ambiguous.
- **B-X primary.** Scored only on evaluable cells: probe_unique >= B0 + .15 in 3/3 seed pairs, not-optimal <= .10,
  and regret non-inferior (upper CI of arm - B0 <= `--ni-regret`, **to be registered**; default 1.0 per episode).
  BO is never a claim.

## 6. B-LOC on the existing models (historical challenge pairs, original eval worlds)
`campaign05_bloc.py`, metered `trackb-bloc-20260927T015725` (**8.8 core-s**), output `bloc-existing.json`. It reproduces
every model's eval.json heldout_comp probe rates, supports and gap regret exactly (12/12).

First consequential error (delta > .5) per episode, pooled over 6 models per rung (3 B1 + 3 F2):

| rung | pair | episodes | with error | at s0 | ranking | value estimate | unlocalized (no Q head) | deployment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | U+E | 1,368 | 400 | 260 | 126 (32%) | 274 (68%) | - | 0 |
| L4 | S+C | 1,704 | 666 | 436 | 197 (30%) | 469 (70%) | - | 0 |
| L1 | U+E | 1,368 | 445 | 256 | - | - | 445 | 0 |
| L1 | S+C | 1,704 | 603 | 376 | - | - | 603 | 0 |

- **Where.** 57-65% of first errors are at the episode's first decision, and 80-90% in the first query. The dominant
  patterns are:
  - U+E: prop when probe is optimal (L4 122/400);
  - S+C: inspect when probe or prop is optimal.
- **What.** For L4, the Q head agrees with the wrong choice at 52-85% of first errors (Q-head argmax not eps-optimal),
  so the error is in the **learned value estimate**, not a policy-readout (ranking) mismatch.
- **Belief class pending.** The belief class awaits the BO arm (extended-04 split, root launch). At s0 the belief equals
  the prior, so BO cannot resolve s0 errors by construction; it can only resolve the 35-43% of later first errors.
- **Pair asymmetry.** The s0 uniquely-probe deficit is concentrated on **U+E**: .12-.75 per model, n = 32 episodes
  = 8 configurations. S+C is at .64-.86 (n = 56 = 14 configurations).
- **Regret.** Realized regret per model is -1.3 to 6.3 (S+C often negative); accumulated gap regret is 1.5-4.2. Their
  expectations coincide, but per trajectory they differ. First errors carry ~25-40% of the gap regret.

## 7. Launch sequence (root; CPU only; 1 thread per job; <= 3 concurrent on the shared host)
From a clean worktree at the Track B commit: `SHA=$(research/tools/campaign05_remote.py snapshot)`. Shorthands:
`PY=/home/brand/structured-latent-dynamics-campaign03/env/bin/python`, `R=/home/brand/structured-latent-dynamics-campaign05/results`,
`R4=/home/brand/structured-latent-dynamics-campaign04/results` (read-only), `T=research/tools/campaign04_probeworld_train.py`,
`E="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1"`, `L=research/tools/campaign05_remote.py launch-cmd`.
Seeds: B-X arms 10, 11, 12 (fresh); BO 0, 1, 2 (same data stream and torch seed as B1 L1 s0-2, differing only by the belief input).

```
# 1. labels, split table v2 + sized holds (one job; ~1.4k core-s, peak RSS ~8 GB while building b5x_train)
$L bx-labels $SHA --cpu-cap 4000 -- $E $PY $T labels --split-set b5 --out $R/bx-labels/labels

# 2. BO on the extended-04 split (independent of 1; may run in parallel)
for s in 0 1 2; do
  $L bx-bo-s$s $SHA -- $E $PY $T train --labels $R4/b-labels/labels --out $R/bx-bo-s$s/run --rung L1 --seed $s --inputs belief
done
#    after each: challenge-set evaluation only, on the B1 worlds (offset 500)
$L bx-bo-eval-s$s $SHA -- $E $PY $T eval --labels $R4/b-labels/labels --run $R/bx-bo-s$s/run --splits heldout_comp --world-offset 500 --episode-rows
#    B-LOC final (12 existing models + BO; list the runs explicitly, no globs)
$L bloc-final $SHA -- $E $PY research/tools/campaign05_bloc.py --labels $R4/b-labels/labels \
   --runs $R4/b-train-L1-s0/run $R4/b-train-L1-s1/run $R4/b-train-L1-s2/run $R4/f-btrain-L1-s3/run $R4/f-btrain-L1-s4/run $R4/f-btrain-L1-s5/run \
          $R4/b-train-L4-s0/run $R4/b-train-L4-s1/run $R4/b-train-L4-s2/run $R4/f-btrain-L4-s3/run $R4/f-btrain-L4-s4/run $R4/f-btrain-L4-s5/run \
   --bo-runs $R/bx-bo-s0/run $R/bx-bo-s1/run $R/bx-bo-s2/run --out $R/bloc-final/bloc.json --records $R/bloc-final/records.jsonl.gz

# 3. B0 and the B-HR gate (after 1)
EV="--split-set b5 --episode-rows --world-offset 900 --split-worlds b5_hold_uc=1 b5_hold_se=1"
for s in 10 11 12; do
  $L bx-b0-s$s $SHA -- $E $PY $T train --labels $R/bx-labels/labels --train-split b5_train --out $R/bx-b0-s$s/run --rung L1 --seed $s
  $L bx-b0-eval-s$s $SHA -- $E $PY $T eval --labels $R/bx-labels/labels --run $R/bx-b0-s$s/run $EV
done
$L bx-refs $SHA -- $E $PY $T references --labels $R/bx-labels/labels --out $R/bx-refs.json --split-set b5 --world-offset 900 --split-worlds b5_hold_uc=1 b5_hold_se=1
$PY research/tools/campaign05_bx_score.py --bhr-only --arm B0 $R/bx-b0-s10/run $R/bx-b0-s11/run $R/bx-b0-s12/run --out $R/bx-bhr.json
#    STOP here if B-HR is no_failure on both holds (report "no failure to repair"); otherwise register --ni-regret, then:

# 4. BX1 / BX2 / BX3 (same seeds; <= 3 concurrent)
for s in 10 11 12; do
  $L bx-bx1-s$s $SHA -- $E $PY $T train --labels $R/bx-labels/labels --train-split b5x_train --out $R/bx-bx1-s$s/run --rung L1 --seed $s
  $L bx-bx2-s$s $SHA -- $E $PY $T train --labels $R/bx-labels/labels --train-split b5_train --inputs bx2 --out $R/bx-bx2-s$s/run --rung L1 --seed $s
  $L bx-bx3-s$s $SHA -- $E $PY $T train --labels $R/bx-labels/labels --train-split b5_train --arch modular --out $R/bx-bx3-s$s/run --rung L1 --seed $s
done
#    eval each as in 3 (bx-bxN-eval-s$s, same $EV)

# 5. score (pure Python + numpy; metered)
$PY research/tools/campaign05_bx_score.py --arm B0 $R/bx-b0-s1{0,1,2}/run --arm BX1 $R/bx-bx1-s1{0,1,2}/run \
   --arm BX2 $R/bx-bx2-s1{0,1,2}/run --arm BX3 $R/bx-bx3-s1{0,1,2}/run --ni-regret <registered> --out $R/bx-score.json
```
(Expand the brace lists explicitly when the command goes through `metered.sh` or `launch-cmd`, which run without a shell.)

## 8. Compute estimate and spent
Unit costs come from the B1 receipts and this branch's metered smoke (`trackb-smoke-20260927T020139`).
- **Training per update** (smoke, relative to B0): BX1 +2%, BX2 +5%, BX3 +12%.
- **Label DP** on the pro6000: U+C 0.60 core-s per configuration, S+E 0.21.

| item | core-s |
|---|---:|
| labels (b5_train 384, b5x_train 768, 4 x 128 eval, heldout_comp 128, U+C ~1,309 [924-2,244], S+E ~291) | ~1.4k (1.2-2.0k) |
| BO: 3 trainings (~525) + 3 challenge evals (~15) + B-LOC final (~20) | ~1.65k |
| B0: 3 trainings (~510) + 3 evals (~85: 5 x 512 episodes + ~1,600 hold episodes + shard loads) + references (~80) | ~1.9k |
| **B-HR gate**: stop here if no failure on both holds | |
| BX1 / BX2 / BX3: 9 trainings (530 / 535 / 570) + 9 evals (~85) | ~5.7k |
| scoring (bootstrap) | ~0.05k |
| **total** | **~10.7k (13.4k with 25% contingency)**; ~5.0k if B-HR stops after B0 |

- **Against the v1 job plan** (B-SPLIT labels .5k, B-X training 5.7k, eval .8k, B-LOC .25k): the difference comes from
  BO (+1.65k, design revision 15), BX3 (+1.8k incl. eval) and the sized holds (+0.9k labels, +0.5k eval).
- **Memory.** Training RSS is ~2.5 GB (b5_train) and ~5 GB (b5x_train). Evaluation holds one pool or shard at a time
  (<= ~2 GB).
- **Wall.** ~9-10 min per training (1 thread).

**Spent by this tooling task:**
- **Remote, metered: 343 core-s** (ledger dev receipts under `~/structured-latent-dynamics-campaign05/results/dev/`):
  - `trackb-golden` 8.6;
  - `trackb-tests-new` 55.9;
  - `trackb-bloc` 8.8;
  - `trackb-tests-full` 98.3 (39 passed: 26 extended-04 + 13 new);
  - `trackb-smoke` 171.2.
- **Local, pure Python** (Dell GB10, estimated from process time): **~0.47k core-s**:
  - audits 88;
  - triviality 53;
  - hold sizing 253 (+ an earlier 71 rate probe);
  - checks ~5.

## 9. B-XC tooling (registry entry B-XC; branch `campaign/e05-b5c`)
Additive; every b5 / b1 path is bit-identical (section 9.3). No b5c label, training or evaluation has been run.

### 9.1 What changed
- **Split `b5c_hold_uc`** (`pw5.SPLITS5C`): `(TRAIN_CELLS, TRAIN_K, (U+C,), 5_900_000_000)`.
  - It is a fresh draw of the b5_hold_uc family: same cells, k and combo, but a configuration seed base never used before.
  - It is kept **outside** `SPLITS5`, so split table v2, `split_of_params`, `family_params` and the B-SPLIT audit are
    unchanged. `split_params`, `generator_params`, `split_config` and `world_seed` consult `SPLITS5C` after `SPLITS5`.
  - Sizing: the same registered rule (`SENSITIVITY_FLAGS` / `HOLD_TARGET_SENSITIVE` entries copied from b5_hold_uc):
    the smallest prefix with >= 60 s0-uniquely-probe-optimal configurations, >= 20 of them flag-sensitive for the
    correlated flag; cap 4,000; 1 world per configuration.
- **Trainer** `--split-set b5c` (`campaign04_probeworld_train.py`):
  - `labels` builds ONLY `b5c_hold_uc` (shards + `_s0.json` + `labels_meta.json` with `split_set.name = b5c`); no b5
    pool is rebuilt.
  - `eval` and `references` default to `b5c_hold_uc` only. `--episode-rows`, `--world-offset`, `--split-worlds` and
    `--out-name` work as for b5.
  - Models trained on `bx-labels/labels` (b5_train / b5x_train) are evaluated with `--labels <b5c labels dir>`: eval
    reads only the run's `train_meta.json`/`model.pt` and the evaluation pools from `--labels`.
- **Scorer** (`campaign05_bx_score.py`):
  - New groups `b5c_hold_uc`, `b5c_uc_flag_sensitive`, `b5c_uc_k1`, `b5c_uc_k_gt1`, with the same metrics as
    b5_hold_uc (the first two with configuration-clustered CIs). They select no b5 row, so b5 outputs are unchanged.
  - `--bxc` implements the registered primary exactly: mean paired (BX1 - B0) s0 uniquely-optimal first-probe rate
    >= +.10 with all 3 pair differences > 0, AND gap regret lower for BX1 in 3/3 pairs.
  - Secondaries: BX1 not-optimal probe <= .10 per seed; success floor (>= .8 x pi*) per arm and seed;
    flag-sensitive subset readings (gain, gap regret, CIs); configuration-clustered 95% CIs per pair and for the mean
    paired difference (configurations resampled jointly across the 6 runs).
  - Validity checks are reported, and the verdict is `invalid` if any fails: 3 + 3 runs, paired seeds {20, 21, 22},
    train splits b5_train / b5x_train, L1 public flat, world offset 900, rows only on b5c_hold_uc with 1 world, and the
    same configurations in every run.
  - `--episodes-name` reads a non-default episode file.
- **Seed-range registry** (`seed-ranges.json`) gains four entries, all pairwise disjoint (checked by
  `campaign05_hr.check_seed_ranges` and `test_campaign05_bxc.py`):
  - b5 split table v2 [5.1e9, 5.9e9);
  - **B-XC b5c_hold_uc [5.9e9, 6.0e9)**;
  - B-X/B-Q training worlds (seeds 10-12) [9.0e9, 9.3e9);
  - B-XC training worlds (seeds 20-22) [1.0e10, 1.03e10).

  Note: `campaign05_hr.source_hashes()` records this file's hash, so future A-HR receipts will show the new hash.
- **Split-support audit** `campaign05_bsplit_audit.py audit-b5c`: **11/11 PASS** at 4,000 configurations (local,
  pure Python, ~1 core-s; `research/results/campaign-05/b-xc/b5c-split-audit.json`).
  - E1-E4 definition:
    - same family as b5_hold_uc, new base;
    - generator parameters absent from every b5 training/selection/test family;
    - sizing rule identical;
    - seed registry disjoint.
  - F1-F5 configurations: land in their generator parameters; seeds disjoint from every earlier pool; **no
    configuration equal to any of 10,688 earlier configurations** (split table v2 incl. the first 4,000 of each hold
    stream, and the extended-04 pools); `split_of_params` finds only the U+C hold family; eval worlds (offsets
    500/700/900) unique and disjoint from every earlier eval world.
  - G1-G2: labels are functions of the visible history (713 distinct random-policy histories, 0 conflicts); simulator
    state = public state rebuilt from the history.

### 9.2 Tests (`tests/test_campaign05_bxc.py`, 9 tests)
- `test_b5_split_table_unchanged`: split table v2 digest equals the one computed with the unmodified module.
- `test_b5_paths_bit_identical`: the b5 labels (every pickle, shard, s0 file, labels_meta), B0/BX1 training (state
  dict, log, meta), `eval --split-set b5 --episode-rows` (eval.json and episode rows), `references --split-set b5`,
  `score()` and `--bhr-only` on tiny runs are identical to goldens captured with the **unmodified** tools of base
  0923cff2 (pro6000, torch 2.14.0+cu130, metered `b5c-golden-20260927T031610`).
- `test_scorer_b5_outputs_unaffected_by_b5c_groups`: `score()` on synthetic b5 runs equals the unmodified scorer's
  output digest.
- `test_b5c_end_to_end_tiny`: labels b5c -> only b5c files; 6 trainings on a b5 labels dir -> eval/references on the
  b5c labels dir (offset 900, 1 world) -> `--bxc` valid.
- Also: the b5c definition, seed ranges, a small b5c audit, the b5c group metrics (= b5_hold_uc metrics on the same
  rows), and the B-XC primary rule (confirmed / mean below .10 / a zero pair / gap regret 2/3 / invalid offset).
- The existing `test_campaign05_trackb.py` (incl. the B1/B2/F2 goldens) passes unchanged.

### 9.3 Launch sequence (root; CPU only; 1 thread per job; <= 3 concurrent)
Shorthands as in section 7 (`SHA`, `PY`, `R`, `T`, `E`, `L`). Trainings use the existing `bx-labels` (b5_train /
b5x_train); only the evaluation uses the b5c labels. Seeds 20, 21, 22 are fresh.
```
# 1. labels: ONLY b5c_hold_uc (sized; ~0.9k core-s, range ~0.5-1.3k)
$L bxc-labels $SHA --cpu-cap 3000 -- $E $PY $T labels --split-set b5c --out $R/bxc-labels/labels

# 2. trainings (independent of 1; may start in parallel; <= 3 concurrent)
for s in 20 21 22; do
  $L bxc-b0-s$s  $SHA -- $E $PY $T train --labels $R/bx-labels/labels --train-split b5_train  --out $R/bxc-b0-s$s/run  --rung L1 --seed $s
  $L bxc-bx1-s$s $SHA -- $E $PY $T train --labels $R/bx-labels/labels --train-split b5x_train --out $R/bxc-bx1-s$s/run --rung L1 --seed $s
done

# 3. evaluations on the fresh U+C configurations (after 1 and the matching training)
EVC="--split-set b5c --episode-rows --world-offset 900 --split-worlds b5c_hold_uc=1"
for s in 20 21 22; do
  $L bxc-b0-eval-s$s  $SHA -- $E $PY $T eval --labels $R/bxc-labels/labels --run $R/bxc-b0-s$s/run  $EVC
  $L bxc-bx1-eval-s$s $SHA -- $E $PY $T eval --labels $R/bxc-labels/labels --run $R/bxc-bx1-s$s/run $EVC
done

# 4. references (after 1)
$L bxc-refs $SHA -- $E $PY $T references --labels $R/bxc-labels/labels --out $R/bxc-refs.json --split-set b5c --world-offset 900 --split-worlds b5c_hold_uc=1

# 5. score (after 3; metered)
$PY research/tools/campaign05_bx_score.py --bxc --arm B0 $R/bxc-b0-s20/run $R/bxc-b0-s21/run $R/bxc-b0-s22/run \
   --arm BX1 $R/bxc-bx1-s20/run $R/bxc-bx1-s21/run $R/bxc-bx1-s22/run --out $R/bxc-score.json
```
(Expand the loops explicitly when a command goes through `metered.sh` or `launch-cmd`: they run without a shell.)

### 9.4 Compute estimate (from the B-X receipts)
| item | basis | core-s |
|---|---|---:|
| labels b5c_hold_uc | b5_hold_uc: 1,528 configurations, 857 core-s (0.56 per configuration); pool size is stream-dependent (projection 924-2,244) | ~0.9k (0.5-1.3k) |
| B0 x 3 trainings | bx-b0-s10/s11: 381 / 377 | ~1.15k |
| BX1 x 3 trainings | bx-bx1-s10/s12: 470 / 462 | ~1.40k |
| 6 evaluations (b5c only, ~1.5k episodes each) | U+C share of the 77-106 core-s b5 evaluations | ~0.25k |
| references | U+C share of bx-refs (63) | ~0.03k |
| scoring (`--bxc`, 1,000 resamples) | | ~0.02k |
| **total** | | **~3.75k (~4.7k with 25% contingency)** |

- **Wall time:** labels ~15 min, trainings ~7-8 min each.
- **Memory:** training RSS ~2.5 GB (B0) and ~5 GB (BX1).

### 9.5 Bit-identity evidence and spent
- **Goldens.** Captured with the unmodified base tools (`b5c-golden`, 37.4 core-s). The new tools reproduce them:
  `b5c-tests`, 159.1 core-s, 22 passed (9 new + 13 existing Track B tests, incl. the B1/B2/F2 goldens).
- **Other suites.** `tests/test_campaign04_probeworld.py`: 26 passed (41.3). `test_campaign05_hr.py -k seed_range`:
  2 passed (1.4).
- **Real B-X data.** The B-X scoring of the 12 real B-X runs, re-run with the old and the new scorer
  (`b5c-rescore`, 38.8), gives byte-identical output (sha256 `8d554f73...`). This equals the committed
  `research/results/campaign-05/b-x/bx-score.json`.
- **Spent.** Remote, metered: **278 core-s**. Local, pure Python: **~5 core-s** (b5c audit ~1, digests/checks).
