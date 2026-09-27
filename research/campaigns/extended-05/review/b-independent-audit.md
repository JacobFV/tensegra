# Extended-05 Track B: independent audit (B-SPLIT, B-HR, B-X, B-Q, B-XC, B-LOC)

**Auditor:** independent; I did not build the Track B tooling. Branch `campaign/e05-b-audit`, from
`campaign/extended-05` @ 88bd3b0c.

**Method.**
- I read `campaign.md`, `design.md` (v1 + v2), `registry.json` (B-LOC, B-SPLIT, B-X, B-Q, B-XC), `decisions.md`,
  `seed-ranges.json` and `trackb-tooling.md`.
- I did **not** open `campaign05_bx_score.py` or `campaign05_bloc.py`. Every metric below is re-derived from the raw
  episode rows and decision records, using definitions taken from the design and registry.
- I read the environment module (`campaign04_probeworld.py`), the split module and the trainer, but only to decode row
  formats, rebuild configurations and replay models.
- Data are read-only copies of the pro6000 run directories. Nothing under `results/` was modified. My outputs went to
  `~/structured-latent-dynamics-campaign05/audit-b/`, plus two metered receipts under `results/dev/`.

Machine-readable companion: [b-independent-audit.json](b-independent-audit.json). Scripts (all under `research/tools/`):

| script | where | what |
|---|---|---|
| `campaign05_b_audit.py` | local, pure Python | B-HR, B-X, B-Q, B-XC metrics and verdicts from episode rows; config-clustered bootstrap; regret decomposition |
| `campaign05_b_audit_split.py` | local, pure Python | split and pool integrity; b5c freshness; my own exact DP (does not use `ExactSolver`); episode re-simulation; sizing rule |
| `campaign05_b_audit_pools.py` | pro6000, metered | dumps the configurations actually stored in every label pickle |
| `campaign05_b_audit_bloc.py` | local | B-LOC first-error classification re-derived from `records.jsonl.gz` |
| `campaign05_b_audit_bloc_control.py` | pro6000, metered | replays the BO models and the other public-input L1 models at every first-error state |
| `campaign05_b_audit_bloc_control_summary.py` | local | summarizes that control |
| `campaign05_b_audit_extras.py` | local | B-Q Q-ranking; s0 vs later-decision decomposition; cross-rule checks |

## Verdict summary

| item | root claim | audit |
|---|---|---|
| **Split v2 integrity** | 18/18 pass | **Confirmed.** The stored pools equal the registered generator, config for config. There are 0 U+C and 0 S+E configurations in any training or selection pool. The historical pairs appear only in b5x_train. |
| **b5c integrity** | 11/11 pass | **Confirmed.** 1,289 configurations, 0 collisions with 10,688 earlier ones. Seed base 5.9e9 is disjoint. Evaluation rows are b5c only, rep 900. |
| **Labels = f(visible history)** | D1-D3 / G1-G2 pass | **Confirmed** by my own DP and re-simulation: 65 episodes, 561 decisions, 0 mismatches, max \|ΔQ\| 5e-7 (row rounding). |
| **Hold sizing rule** | applied | **Confirmed** (§1.4). |
| **B-HR** | U+C evaluable, S+E no failure, U+C flag-sensitive ambiguous | **Reproduced exactly.** |
| **B-X primary** | not supported for any arm | **Reproduced.** BX1 passes the ≥ .15 lift bar in 1/3 pairs; BX2 and BX3 in 0/3. |
| **B-X / B-Q point metrics** | decisions.md tables | **Reproduced**: 630 values, max \|diff\| 1.1e-14. |
| **B-XC primary** | confirmed | **Reproduced exactly.** Mean paired gain +.206 (my config-clustered 95% CI .113 to .307); pairs +.167/+.233/+.217; gap regret lower 3/3. All validity checks pass. |
| **B-LOC classes** | L4 value-estimate 57-66%, ranking 23-38%, belief-dependent 6-17% | **Counts reproduced exactly** (100% per-episode agreement). The value-estimate range is **52**-66%, not 57-66% (f-btrain-L4-s5 is .524). The **belief-dependent class is not identified** (§3.2). |
| **Adaptive labelling of B-XC** | registered after B-X/B-Q were inspected | **Accurate.** The registration commit precedes every B-XC training, label and evaluation receipt. |

## 1. Split integrity (check 1)

### 1.1 Stored pools against the registered generator
I dumped the configurations inside every label pickle on the pro6000 (`baudit-pools`, metered 67.5 core-s). I then
regenerated each configuration locally from a hard-coded copy of the registered split table (seed base + i; cells, k,
combos). The generator is `make_config`, which is unchanged from extended-04.

- **All 9 pools are bit-equal to the registered generator**, with contiguous indices:
  b5_train 384, b5x_train 768, b5_dev / test_iid / heldout_price / heldout_k 128 each, b5_hold_uc 1,528,
  b5_hold_se 315, b5c_hold_uc 1,289.
- **New holds are absent.** Counting flags on the stored configs (not on the combo labels), U+C and S+E are 0 in
  b5_train, b5x_train, b5_dev, b5_test_iid, b5_heldout_price and b5_heldout_k.
- **Historical pairs appear only in the exposure pool.** U+E and S+C are 0 everywhere except b5x_train (84 and 76).
- **Selection** uses final checkpoints. The trainer samples only from `--train-split`, and I found no use of b5_dev
  to select.
- **Launch commands** (all B-X, B-Q and B-XC training receipts) use only `--train-split b5_train` or `b5x_train`
  on `bx-labels`.
- For reference, the extended-04 train pool contains U+C (54) and S+E (42). This is expected: that is why the design
  moved to split v2. It is relevant to BO and to the existing B-LOC models, which were trained on that pool.

### 1.2 b5c freshness
- **No reuse.** None of the 1,289 b5c configurations equals any of 10,688 earlier configurations: the split v2 pools,
  the first 4,000 of each b5 hold stream, and the extended-04 pools. There are also 0 collisions on the
  (c_probe, c_b1, c_b2, C_build, L) price tuple.
- **Seeds.** Configuration seeds are 5.9e9 + [0, 1,288]; no other registered base is within 4,000.
- **Evaluation worlds.** Every b5c evaluation row has world rep 900 (seed 5.95e9 + 1000 i + 900). There is exactly one
  row per configuration, configurations 0..1,288. The evaluation pool contains b5c only.
- **What is and is not fresh.** B-XC trained B0 and BX1 on the **same** b5_train / b5x_train pools as B-X; only the
  seeds (20–22, i.e. torch init 1020–1022, data stream 7020–7022, training worlds 1.0e10+) and the evaluation
  configurations are fresh. "Fresh initializations" is accurate. "Fresh training data" would not be.

### 1.3 Labels depend only on visible history
65 episodes were sampled: 40 from b5c (B0 s20, BX1 s21), 15 from b5_hold_uc and 10 from b5_hold_se. For each one:

- I regenerated the configuration.
- I replayed the hidden world with `Episode(cfg, world_seed)` and the recorded actions. All 561 visible records
  (a, o, event, reveal) and every realized utility match.
- At every decision I rebuilt the public state from the visible history alone and solved it with **my own exact DP**
  (my recursion; environment outcome model only). Q* matches to 5e-7, which is row rounding. The ε-optimal sets and
  the available sets match in 561/561 decisions, and V*(s0) matches exactly.

### 1.4 Hold sizing rule
The registered rule: the smallest prefix with ≥ 60 s0-uniquely-probe-optimal configurations, ≥ 20 of them
flag-sensitive for C (U+C), cap 4,000, 1 world each.

- **Prefix rule on the label job's own s0 records:**
  - b5_hold_uc: n = 1,528 (60 eligible, 33 flag-sensitive eligible);
  - b5_hold_se: n = 315 (60 eligible);
  - b5c_hold_uc: n = 1,289 (60 eligible, 36 flag-sensitive).

  All three equal the stored pool sizes.
- **My DP on b5c:** I recomputed s0 facts for **all 1,289** b5c configurations. They give 0 mismatches in probe_unique and in flag_sensitive, and V* differences of 0. Applying the rule to my own records gives **n = 1,289**, with 60 eligible and 36 flag-sensitive eligible.
- **My DP on the b5 holds:** all 122 eligible or boundary configurations of the two b5 holds agree (0 mismatches).
- **The binding constraint** was the 60-eligible target in all three pools, not the flag-sensitive one.

## 2. Reconstruction from raw episode rows (check 2)

**Definitions I used:**
- s0 = the episode's first decision.
- Uniquely-optimal first-probe rate = P(first action = probe | s0 ε-optimal set = {probe}), with ε = .5.
- Not-optimal probe = P(probe first | probe ∉ s0 ε-optimal set).
- Realized regret = V*(s0) − U.
- Gap regret = Σ_t δ_t.
- CIs: 95% percentile bootstrap over configurations (1,000 resamples). Paired differences resample the same
  configurations in both runs. There is one world per configuration, so configuration clustering equals episode
  resampling here.

Every point estimate below equals the root's `bx-score.json` / `bq-score.json` / `bxc-score.json` to within 1e-14.
The root CIs agree with mine to bootstrap noise; for example, the mean B-XC gain is root .106–.305 and mine .113–.307.

### 2.1 B-HR (B0 = L1 recipe on split v2, seeds 10–12)

| cell | probe_unique (support) | verdict |
|---|---|---|
| U+C | .633 / .650 / .517 (60 each) | evaluable (≤ .70 in 3/3) |
| S+E | .983 / .950 / .967 (60) | no failure |
| U+C flag-sensitive | .727 / .727 / .576 (33) | ambiguous |

- **Provenance of the .70 threshold.** Design v2 registered only the ≥ .80 "no failure" condition. The ≤ .70
  "evaluable" threshold is from the tooling (df55e56c, 02:05:39Z). That commit precedes every B-X model run, so this
  is a disclosed tooling-level rule, not post hoc.

### 2.2 B-X primary on U+C (registered: lift ≥ +.15 in 3/3, not-optimal ≤ .10, regret NI upper CI ≤ 1.0)

| arm | lift per seed (paired 95% CI) | not-opt probe | Δ realized regret (CI) | Δ gap regret (CI) | pairs passing all |
|---|---|---|---|---|---|
| BX1 | +.100 (−.02, .22) / +.150 (.05, .25) / +.133 (.02, .25) | .045/.060/.022 | −2.58 (−4.2, −1.1) / −1.51 (−3.4, .4) / −1.30 (−3.1, .6) | −.75 (−1.37, −.11) / −.83 (−1.44, −.24) / −.53 (−1.30, .16) | **1/3** |
| BX2 | −.183 / −.050 / −.067 | .008/.030/.008 | −1.21 / −1.28 / −.94 | **−.90 (−1.58, −.31) / −1.18 (−1.94, −.53) / −.69 (−1.27, −.21)** | 0/3 |
| BX3 | +.033 / .000 / +.067 | .079/.051/.042 | +2.33 / +1.40 / +1.43 | +3.35 / +1.35 / +1.71 | 0/3 |

**Verdict:** not supported for any arm, matching root. Seed 11 sits exactly on the bar (.800 vs .650 + .150).

**BX2 lowers gap regret in 3/3 pairs, with CIs that exclude 0.** Its gains are as large as BX1's or larger. They come
entirely from **later** decisions: later-decision gap 1.68/2.06/1.98 vs B0's 2.69/3.54/2.67. At s0 the belief equals
the prior and BX2 is worse there (s0 gap .61/.71/.50 vs .49/.41/.51). The decisions.md table leaves BX2's gap regret
blank; see claim C3.

### 2.3 B-Q (diagnostic; U+C inspected)

| | seed 10 | seed 11 | seed 12 |
|---|---|---|---|
| probe_unique, B0-L4 vs BX1-L4 | .600 vs .717 | .467 vs .750 | .617 vs .733 |
| gap regret, B0-L4 vs BX1-L4 | 3.42 vs 2.36 | 3.84 vs 2.76 | 3.35 vs 2.42 |
| all-pairs Q-rank accuracy (\|ΔQ*\| ≥ .5; ~345k pairs), B0-L4 vs BX1-L4 | .938 vs .915 | .927 vs .920 | .935 vs .932 |
| s0 Q-head argmax ε-optimal on the 60 probe-unique configs, B0-L4 vs BX1-L4 | .70 vs .75 | .25 vs .57 | .90 vs .88 |

- All numbers above match root.
- **Decision-relevant Q check (mine).** The s0 row shows that the Q head's own first-decision choice improves clearly
  in 1/3 seeds only. So B-Q does **not** show that exposure works *through* better value estimates. That was B-Q's
  registered hypothesis, and neither the aggregate ranking nor the s0 Q-head check supports it.

### 2.4 B-XC primary (BX1 − B0, L1, seeds 20–22, fresh b5c U+C configurations)

| pair | B0 → BX1 probe_unique (60) | gain (95% CI) | Δ gap regret (CI) | Δ realized regret (CI) | not-opt probe B0 → BX1 (848) |
|---|---|---|---|---|---|
| 20 | .567 → .733 | +.167 (.065, .275) | −.81 (−1.84, −.02) | −2.74 (−5.26, −.66) | .026 → .050 |
| 21 | .550 → .783 | +.233 (.129, .353) | −1.09 (−1.92, −.37) | −1.69 (−3.76, .36) | .021 → .059 |
| 22 | .517 → .733 | +.217 (.101, .333) | −1.62 (−2.93, −.58) | −4.03 (−7.02, −1.53) | .022 → .044 |
| **mean** | | **+.206 (.113, .307)** | 3/3 lower | | CI of Δ excludes 0 in 3/3 |

**Primary: CONFIRMED**, reproduced exactly.

**Validity checks (independently verified):**
- 3 + 3 runs, paired seeds {20, 21, 22};
- train splits b5_train / b5x_train;
- L1, public inputs, flat architecture, 4,000 × 64 updates;
- evaluation rows only on b5c_hold_uc, world rep 900, one world per configuration, the same 1,289 configurations in
  every run;
- the s0 ε-optimal sets in the rows agree with the label job's s0 file for 6 × 1,289 episodes.

**Subgroups.**
- Flag-sensitive subset (36 eligible): mean gain +.259 (.133, .397), gap lower 3/3.
- k = 1 (21 eligible): +.206 (.02, .37).
- k > 1 (39 eligible): +.205 (.10, .33).

**Cross-rule checks (mine):**
- **B-XC passes the original, stricter B-X rule.** Gains are ≥ .15 in 3/3; not-opt is .050/.059/.044, below .10;
  the regret NI upper CIs are −.66/+.36/−1.53, all ≤ 1.0. This is the strongest fact in B-XC's favour.
- **The B-X data would also have passed the B-XC rule** (mean lift .128, all > 0, gap lower 3/3). So the adaptive rule
  was lenient enough to pass on the inspected data. The fresh replication, not the rule, is what carries the
  evidence.

**What moved (s0 decomposition, mine):**
- **Probe propensity.** BX1's overall first-probe rate roughly doubles (.094/.085/.072 → .175/.187/.143).
- **Correct vs wrong probes.** On the 60 eligible configurations BX1 adds +10/+14/+13 correct probes. On the
  848 not-optimal configurations it adds +20/+32/+18 **wrong** probes.
- **Discrimination still improves.** P(probe | unique) − P(probe | not-opt) goes .54/.53/.49 → .68/.72/.69.
- **Where the gap-regret gain sits.** s0 gap is .43/.85/.46 → .41/.27/.64 (worse for BX1 in pair 22). Later-decision
  gap is 3.35/2.90/3.93 → 2.56/2.38/2.14. So **most of the gap-regret gain comes from later decisions, not from the
  first-probe decision** that the headline names.

## 3. B-LOC (check 3)

### 3.1 Classification reproduced
From `bloc-final/records.jsonl.gz` (12 models × 512 episodes, heldout_comp, B1 worlds offset 500 and F2 worlds
offset 700):

- **First error.** I re-derived the first consequential error (first δ > .5) in all 2,114 error episodes, with 0 index
  mismatches.
- **The class rule** is recovered from the design definitions:
  - **ranking**: the L4 Q-head argmax over the available actions is ε-optimal;
  - otherwise **belief-dependent**: a **majority (2 of 3)** of the BO models choose ε-optimally at that state;
  - otherwise **value-estimate** (L4) or **unlocalized** (L1);
  - **deployment** (ε-optimal tie) never occurs.
- **Agreement.** This rule agrees with the recorded class in 2,114/2,114 episodes, and the per-model counts equal
  `bloc.json` exactly. The majority aggregation is not stated in the design or tooling note; "any" would give
  different counts (1,865/2,114 agreement).
- **BO replay check.** I replayed the 3 BO models at every first-error state (`baudit-bloc-control`). Their argmax
  equals the recorded `bo_argmax` in 6,342/6,342 cases.
- **Per-model fractions (L4):**
  - value-estimate .659/.572/.620/.583/.587/**.524**;
  - ranking .227–.377;
  - belief .063–.160.

  The decisions.md range "value-estimate 57–66%" should read **52–66%**.
- **Definitional caveat.** The value-estimate class includes 106/627 cases where the Q head does *not* misorder the
  chosen vs the optimal action but its argmax is a third, non-optimal action. The design's parenthetical ("the Q-head
  misorders the chosen versus the optimal action") strictly covers only the other 521.
- **Sample.** A stratified sample of 16 first errors (4 per class) is in the JSON (`bloc.sample`). Each hand-checked
  entry is consistent: chosen action, ε-optimal set, Q-head argmax and BO argmaxes.

### 3.2 "Belief-dependent" is not identified (new control)
- **136 of 374 belief-dependent first errors (36%) occur at the episode's first decision.** There the exact belief
  equals the declared prior, which is already a public input. BO has no extra information at those states, so these
  are "fixed" by a *different model*, not by belief. The tooling note's statement that "BO cannot resolve s0 errors
  by construction" is wrong in practice, because BO is a separately trained model.
- **Control.** At the same first-error states I replayed the **other public-input L1 models**: extended-04 seeds 0–5,
  excluding the erring model, no belief input. The table compares the rate at which a majority of 3 models choose
  ε-optimally.

  | first errors | n | BO (oracle belief) majority | other public L1 seeds, majority of 3 (exact average over subsets) |
  |---|---|---|---|
  | all | 2,114 | .207 | **.271** |
  | at s0 | 1,328 | .120 | .222 |
  | later (belief can differ from prior) | 786 | .354 | **.353** |
  | later, L4 errors | 370 | .292 | .355 |
  | later, L1 errors | 416 | .409 | .351 |

- **Reading.** Supplying the exact belief "fixes" first errors no more often than simply using a different public-input
  seed. The 6–16% (L4) and 15–37% (L1) belief-dependent shares measure **seed-to-seed variation**, not belief
  dependence. The qualitative reading "not belief tracking" survives, arguably strengthened: no belief effect is
  detectable over the control. But the class should be relabelled as "resolved by another model", or reported against
  this control.
- **Idiosyncratic errors.** About 27% of first errors are fixed by other public seeds, so a sizable share of the
  historical errors is seed-specific rather than systematic.

### 3.3 BO training contract
- The three BO receipts run `train --labels R4/b-labels/labels --rung L1 --seed s --inputs belief`, with no
  `--train-split`: the extended-04 "train" pool of 384 configurations, the same pool as B1 L1 s0–2.
- `train_meta` equals B1 L1 s0–2 except `inputs` = belief, in_dim 72 vs 68 and params 129,701 vs 129,189 (same
  updates, batch, lr, hidden and loss weights).
- **"Only the belief input differs" is accurate for the data stream, not for initialization.** The torch seed
  (1000 + s) and the data seed (7000 + s) are the same, so the sampled configurations and world seeds are identical.
  But the first layer's shape differs, so every initial weight differs (the RNG consumption shifts), and on-policy
  trajectories diverge. §3.2 shows why that matters.

## 4. Regret metrics (check 4)

**Definitions:**
- Realized regret = V*(s0) − U(episode), one draw of hidden types and outcomes.
- Gap regret = Σ_t [V*(I_t) − Q*(I_t, a_t)] ≥ 0 by construction; the observed minimum is 0.

**Why the two agree in expectation.**
- Their difference is Σ_t [Q*(I_t, a_t) − r_t − V*(I_{t+1})]. This is a sum of martingale increments, because the
  DP's public belief is the exact posterior of the simulator (verified in §1.3). So E[realized] = E[gap].
- Per trajectory they differ. An episode whose hidden types happen to be favourable (e.g. θ = H and the cheap probe
  succeeds) realizes U > V*(s0).

**Why realized regret goes negative.** 31–40% of episodes have negative realized regret in every run. On shared
evaluation worlds the luck term is **common to all arms**:
- B-X U+C, 12 runs on the same 1,528 worlds: realized − gap is negative in all 12 (−.28 to −2.11; per-run SE .7–1.1).
- B-XC, a different world set: residuals are mixed (+1.02, −.08, +1.55, −.91, −.69, −.86).
- B-LOC: the 6 B1-world models (offset 500) all have negative realized regret on S+C (−.2 to −1.3), while the F2-world
  models (offset 700) are mostly positive. The sign tracks the world set, not the model.

**Gap regret is the steadier measure.** Confirmed:
- per-run SE of gap regret is **2.2–4.5× smaller** (e.g. B0 s10: .38 vs .89);
- paired CI widths for arm differences are **2–3× narrower** (B-XC pair 20: gap 1.8 vs realized 4.6).

**Caveat.** The registered B-X non-inferiority test used *realized* regret. The B-XC primary switched to *gap* regret
(adaptive, but defensible), and B-XC also passes the realized-regret NI (§2.4).

## 5. Adaptive registration of B-XC (check 5)

Timeline from git commit times and pro6000 receipts (UTC):

| event | time |
|---|---|
| B-X NI tolerance registered (9226196a) | 02:06:33 |
| B-X arms launched / finished | 02:43:34 / 02:55:24 |
| B-X scored (f30a1b0f) | 02:56:25 |
| B-Q registered (6a77eae7), trainings started | 02:57:51, 02:57:54 |
| B-Q scored (d8ba0363) | 03:07:54 |
| **B-XC registered (0923cff2)** | **03:08:34** |
| B-XC trainings `bx-xc-{b0,bx1}-s2{0,1,2}` start (snapshot 9226196a) | 03:09:03–03:09:12 |
| b5c tooling merged (f22ece8d) / logged (ef0db162) | 03:22:14 / 03:22:54 |
| **bxc-labels start** (snapshot ef0db162) | **03:23:05** |
| bxc evaluations start / refs | 03:35:48–03:35:56 |
| B-XC confirmed (22d326cc) | 03:40:34 |

- **Order.** Registration precedes every B-XC training, the b5c label build and every B-XC evaluation. The B-XC,
  B-X, B-Q and B-SPLIT registry entries are byte-identical between 0923cff2 and HEAD, so there were no post-hoc
  edits.
- **Labelling is accurate.** The pair family (U+C), the arm (BX1), the lowered threshold (+.10 mean, vs +.15 per
  pair) and the switch to gap regret were all chosen after B-X and B-Q had been inspected. The registry says so.
- **"Fresh configurations + fresh initializations" is accurate**, with the §1.2 caveat that the training pools are
  unchanged.
- **Not new-pair transfer.** This is a configuration-level replication within the inspected U+C family, one held-out
  pair. The registry "scope" line states this correctly.
- **Dependence.** The 3 pairs share the same 60 eligible configurations, so "3/3 pairs" is replication over
  initializations, not over independent configuration samples. The configuration-clustered CI of the mean
  (.113–.307) is the right uncertainty statement, and its lower end is close to the .10 threshold.
- **Minor.** Run directories are named `bx-xc-*`, not the `bxc-*` of the tooling's launch plan. This is harmless: the
  scorer's `runs` list points to them.

## 6. Claims to weaken or correct

- **C1. B-LOC belief class.**
  - As written: "belief-dependent (fixed by oracle belief) 6–17%" and "the combination failure is … not belief
    tracking".
  - Correct to: "a majority of oracle-belief models choose correctly at 6–16% (L4) and 15–37% (L1) of first errors.
    Other public-input seeds do so at least as often (27% vs 21% overall; 35% vs 35% after the first decision), so no
    belief effect is detectable."
  - Drop "fixed by oracle belief". State that 36% of the belief-class cases occur at s0, where belief = prior.
- **C2. B-LOC value-estimate range.** Change 57–66% to **52–66%**. Also disclose the majority-of-3 BO rule and that
  value-estimate includes Q-argmax-wrong-but-not-misordered cases (106/627).
- **C3. B-X reading "it is not … a lack of factorized inputs".**
  - This is too strong. BX2 (supplied belief and per-flag features) lowers gap regret on U+C in 3/3 pairs, with CIs
    that exclude 0 (−.69 to −1.18 per episode; BX1: −.53 to −.83), through later decisions.
  - It does not help the **first-probe** decision (expected: belief = prior there).
  - Correct to: "supplied factorization does not fix the first-probe decision on U+C; it reduces later-decision gap
    regret".
  - Report BX2's gap regret in the decisions table.
- **C4. B-Q.** The registered mechanism ("works through better value estimates") is not supported. Aggregate Q-ranking
  is flat or lower, and the s0 Q-head argmax improves clearly in 1/3 seeds only. The existing reading ("shows up in the
  decisions, not aggregate Q ranking") is fine, but should not be read as localizing the gain to value estimates.
- **C5. B-XC headline.**
  - As written: "exposure improves value-of-computation decisions (whether to probe first) and gap regret on an unseen
    combination".
  - Add the following:
    - **(a)** BX1 roughly doubles the overall first-probe rate. Its not-optimal probe rate rises in 3/3 pairs
      (+.024/+.038/+.021, CIs exclude 0), though it stays ≤ .10 and discrimination improves.
    - **(b)** Most of the gap-regret gain is from **later** decisions. The s0 gap is mixed (worse in pair 22).
    - **(c)** The adaptive B-XC rule would also have passed on the inspected B-X data. The confirmatory weight comes
      from the fresh replication, which also passes the stricter original B-X rule; that is worth stating.
    - **(d)** Fresh configurations and fresh initializations, **same training pools**.
    - **(e)** One weak pair family (C matters in about 16% of configurations), 60 eligible configurations shared by
      all pairs; mean-gain CI .11–.31.
- **C6. Tooling-note statement** "BO cannot resolve s0 errors by construction" (trackb-tooling §6) is false in
  practice: 136 s0 cases were classified belief-dependent. Correct it, or exclude s0 from the belief class.

No numeric result was found to be wrong apart from C2. All verdicts (B-HR, B-X not supported, B-XC confirmed) stand as
registered.

## 7. Compute

- **Metered (pro6000, CPU only, 1 thread, `metered.sh`):**
  - `baudit-pools` 67.5 core-s (receipt `results/dev/baudit-pools-20260927T041113-1081562-process`);
  - `baudit-bloc-control` 5.2 core-s (`results/dev/baudit-bloc-control-20260927T041819-1082515-process`);
  - **total 72.7 core-s.**
- **Unmetered remote.** A few `ls`/`cat` inspections and one `tar | base64` copy of the run directories (~15 MB, no
  model.pt): an estimated ≤ 10 core-s.
- **Local (Dell GB10, pure Python, 1 core; estimated from process time):**
  - `campaign05_b_audit.py` 32;
  - split audit, sampled run 122;
  - split audit, full b5c prefix 612;
  - B-LOC re-derivation ~5;
  - extras ~30;
  - summaries/checks ~10;
  - **total ~810 core-s.**
- The host load stayed at ~0 besides my jobs; the user's workloads were not disturbed.
