# Extended-06 Track B: structural screen, registered challenge families, counterfactual octets, arms, launch

**Status:** built on `campaign/e06-trackb` (from `campaign/extended-06`, rebased on design v2 `1ff2c29a`). The structural screen has been **run (metered)** and the challenge families are **registered from it before any model exists**. No b6 label, training or evaluation job has been run; those are root launches (section 8).

| file | role |
|---|---|
| `src/tensegra/campaign06_probeworld.py` | probeworld-v3: factor registry, generator gen3 (design v2 ranges), D/T extension, exact screening primitives, split table v3, design-v2 arms, counterfactual octets, derived factor features (pure Python) |
| `research/tools/campaign06_bscreen.py` | exact structural screening (`run`, `summarize`); both selection rules |
| `research/tools/campaign04_probeworld_train.py` | additive b6 options, all default off (b1/b5/b5c bit-identical; goldens pass) |
| `research/tools/campaign06_bscore.py` | held-out endpoints, octet and intervention metrics, configuration-clustered CIs, seed-paired contrasts |
| `tests/test_campaign06_trackb.py` | 12 tests (11 pure Python + 1 torch end-to-end) |
| `research/results/campaign-06/b-screen/` | `screen.json` / `screen.md` (official gen3 screen), `gen3/*.jsonl` (raw records), `gen2-v1ranges/` (superseded official gen2 screen) |

## 1. Screening method (exact, model-independent)

Every quantity below comes from the exact DP of a configuration and of **every sub-combination at the same prices, prior and k** (each factor switched off with `ablate`, i.e. one parameter set to its off value, a valid public configuration).

- **Population.** Per family: configurations from the family's own screening stream (`SCREEN_BASE + 1e6 * family index + i`), the 8 training price cells, k in {1, 2, 8} (C only at k >= 2). **Eligible** = V*(I0) > eps (attempting is worthwhile); every sampled configuration was eligible.
- **Per-factor decision relevance**, for each f in the family (c_-f = c with f off):
  - `first`: pi*_{c-f}(I0) is not eps-optimal in c (the optimal first decision changes);
  - `any`: following pi*_c, some reachable decision exists at which pi*_{c-f} (tracking its **own** belief from the shared visible history) is eps-suboptimal in c (the optimal first **or later** decision changes);
  - `regret`: V*(c) - V^{pi*_{c-f}}(c) > eps.
- **Interaction residual** at I0 (the same public state in every sub-combination):
  - order-1 additive prediction Q_0 + sum_f (Q_f - Q_0) (for triples also the order-2, pairwise inclusion-exclusion prediction);
  - reported as the median max_a |Q - Q_add| and the fraction of configurations whose additive argmax is not eps-optimal;
  - `joint flip`: no one-factor ablation's first action is eps-optimal for the full family.
- **Global-shift solvability.** The reference policies are the **training-support-optimal policies nearest to c**: pi*_{c-f}, f in the family (the one-factor ablations). The shortcut class is pi_{r,a,b}(I) = argmax_x [Q*_{c-f}(I~, x) + b 1{x = a}], with **one** (reference, action, bias) shared by the whole family.
  - `GS_regret` (the design-v2 "closable fraction"): 1 - Reg_best / Reg_0, where Reg is the mean exact whole-episode regret over the family (bias at every decision), b in {-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30} price units, a in {probe, b1, b2, inspect, prop}, and Reg_0 = the best unbiased reference.
  - `GS_first`: the same on the first decision (accuracy; b on a .25 grid in [-100, 100], every action).
  - `GS_vec`: a free per-action bias vector (a global action-rate prior).
  - If nothing is left to recover, GS = 1 (fully solvable by a support policy).
- **Registered rules.**
  - **Rule A** (builder's; committed at `930c51d3`, before any screen): every factor `regret` >= .30 and `first` >= .15; GS_first and GS_regret <= .35; joint flips >= 15 per 1,000 draws; order-1 additive first action wrong >= .10.
  - **Rule B = registry B-SCREEN** (design v2; official): among untouched triples, the primary has the **largest minimum per-factor relevance (`any`)** subject to GS_regret <= .5, and the next passing triple is the secondary.
    - The D/T extension is a whole-generator alternative, "used only if it clears the rule more clearly". It is adopted only if its best passing triple exceeds the best untouched triple's minimum by > .05. This margin was fixed before the gen3 results were seen.

## 2. Generator history: why probeworld-v3 is what it is

| version | factors / ranges | how screened | outcome |
|---|---|---|---|
| gen1 | v1 U,S,C,E + D (deadline) + G (b1 times out on unreduced M) | local pilot only (n = 10/family; local CPU) | G irrelevant (first/any <= .1): b1 is rarely on the optimal path. Replaced before any metered run. |
| gen2 | v1 ranges; k in {1, 2, 4}; + **D** (deadline 1 or 2 non-terminal actions per query) + **T** (every exact call on a hard type costs t * c_b2, t in [.5, 2.5]) | **official, metered** `e06-tb-bscreen` (1,911 core-s): 6 singles and 15 pairs at n = 150; 20 triples at n = 60 | **No family passed rule A.** (1) C and E are weak under v1 ranges (single-factor `any` .07 / .15). (2) Probing is already optimal first in most base configurations, so D, U and S all push the same way, and their pairs are shift-solvable (U+D GS_regret .99, S+D .92). (3) T's relevance is mostly after the first decision. Closest: U+T (fails the order-1 interaction check, .06), D+T (T regret-relevance .25). |
| **gen3** | **design v2 revision 1**: C only at k >= 2 with corr in [.25, .45], p_event in [.3, .6]; k in {1, 2, 8}; D/T kept as a versioned extension (unused unless it clears rule B more clearly) | **official, metered** `e06-tb-bscreen-v3` (2,931 core-s): the 4 untouched triples at n = 80; the D/T triples that could plausibly compete (S+E+T, C+E+D, C+E+T) at n = 24 | S+C+E primary, U+C+E secondary (below) |

- The gen2 table (pairs and triples) is in `research/results/campaign-06/b-screen/gen2-v1ranges/screen-gen2.md`.
- The other D/T triples were bounded, from the gen2 screen, by their factors whose ranges did not change (U, S, D, T): each bound is <= .45, below the untouched triples' minima.

**Official gen3 screen** (relevance `first / any / regret>eps`; 95% Wilson CI of the minimum `any` in brackets):

| family | n (eligible) | per-factor relevance first / any / regret>eps | min any [CI] | all factors `any` | joint flip at s0 | order-1 (order-2) additive first action wrong | GS_first (vec) | **GS_regret** | rule B |
|---|---|---|---|---|---|---|---|---|---|
| **S+C+E** | 80 (80) | S .40/.80/.75; C .07/.66/.59; E .07/.66/.59 | **.66** [.55, .76] | .59 | 1 | .09 (.03) | .67 (.50) | **.26** | **primary** |
| **U+C+E** | 80 (80) | U .30/.90/.91; C .10/.55/.41; E .21/.68/.72 | **.55** [.44, .65] | .53 | 0 | .23 (.04) | .37 (.50) | **.27** | **secondary** |
| U+S+E | 80 (80) | U .24/.59/.57; S .25/.59/.55; E .23/.53/.56 | .53 [.42, .63] | .29 | 5 | .21 (.11) | .67 (.78) | .60 | fails GS |
| U+S+C | 80 (80) | U .28/.62/.66; S .05/.17/.17; C .05/.36/.31 | .18 [.11, .27] | .15 | 1 | .38 (.07) | .75 (.75) | .00 | passes, lowest |
| C+E+T (ext.) | 24 (24) | C .04/.67/.33; E .08/.88/.67; T .00/.88/.71 | .67 [.47, .82] | .63 | 0 | .12 (.08) | 1.0 (1.0) | .00 | not adopted |
| S+E+T (ext.) | 24 (24) | S .50/.58/.50; E .04/.54/.38; T .17/.71/.62 | .54 | .17 | 0 | .00 (.04) | 1.0 (1.0) | .41 | - |
| C+E+D (ext.) | 24 (24) | C .12/.50/.58; E .25/.71/.71; D .58/.92/.92 | .50 | .46 | 0 | .25 (.04) | .33 (.33) | .00 | - |

- **GS details.**
  - S+C+E: the best shared shift ("prop -2" on the E-ablated policy) closes 26% of the regret gap (2.78 -> 2.06 per episode).
  - U+C+E: "inspect +2" on the E-ablated policy closes 27% (6.60 -> 4.80).
  - U+S+E: "probe -4" on the S-ablated policy closes 60% (4.25 -> 1.70), i.e. mostly a probe-rate shift.
- **Rule A** (reported, superseded): no gen3 family passes it either. S+C+E and U+C+E fail `first` >= .15 (C and E act mostly after the first decision) and the s0 joint-flip support.

## 3. Registration (registry B-SCREEN; proposed to root, recorded before any model run)

- **Primary held-out challenge family: S+C+E (side effect + correlated + events).**
  - It has the largest minimum per-factor relevance among untouched triples: .66 [.55, .76], with S .80.
  - 59% of eligible configurations need all three factors.
  - GS closable fraction .26 <= .5.
- **Secondary: U+C+E** (minimum .55 [.44, .65]; GS .27).
- **Never in training, dev or selection:** S+C+E, U+C+E and the 4-factor combination (their superset). The audit (`audit_split_table`, tested) checks this on generator parameters.
- **Historical challenge sets** (evaluation only, fresh v3 draws; `b6_hist_*`): U+E, S+C, U+C.
  - Under design v2 some arms train on a historical pair: B2/B3 on U+C and U+E; dose3 on S+C.
  - Every such arm is flagged in `labels_meta` (`hist_in_training`). A historical challenge reading is only a challenge for arms that did not train on it.
- **The extension was not adopted.** Its best triple, C+E+T (.667 at n = 24), does not beat S+C+E (.662) by the .05 margin.
  - Disclosed alternative: a per-slot reading would have put C+E+T in the secondary slot. It is not used, for three reasons: the registry text restricts the rule to untouched probeworld-v3 triples; the n = 24 estimate is imprecise; and it would require T in every training arm.
- **What the primary does and does not test.** The composition content of S+C+E is mostly in **later** decisions.
  - First-decision relevance is C .07, E .07; joint flip at s0 is 1/80; the order-1 additive first action is wrong in 9%.
  - The small first-decision content is partly shift-solvable (GS_first .67 on 6 errors).
  - Therefore first-decision endpoints are **secondary**. The primary endpoints are later-decision accuracy, gap regret and the octet contrasts at every decision type (design v2 revisions 1 and 6).

## 4. Balanced counterfactual octets (`counterfactual_set`, label part `cf`)

- **Members.** For a held-out triple, each octet has 8 members: every sub-combination (none, 3 singles, 3 pairs, the triple) at the **same** prices, prior and k, built with `restrict`/`ablate`.
  - Each member carries exact labels (eps-optimal set, unique flag, pi*, Q*) at every decision type whose visible history is possible in it.
- **Decision types** (visible-history prefixes; the decision is taken after the prefix):
  - `first`;
  - `after_probe_solved`, `after_probe_failed`, `after_b1_timeout`;
  - `q2_after_H` / `q2_after_notH`: first decision of query 2 after query 1 was solved exactly and revealed H or not-H. This is where C's relevance arrives.
- **Flip levels per decision type:**
  - `flip` (primary): no **single-factor** member's optimal action is eps-optimal for the triple. B0's support contains the singles, not the constituent pairs.
  - `flip_ablations`: no **pair** member's optimal action is eps-optimal.
  - `changed_vs_none`.
- **Near-miss** (for unique-optimum flips): a triple configuration with every factor still ON but moved toward its weakest ON value (lambda in {.5, 1} per factor, then all together). Its unique decision is predicted by a single-factor member of its own (no flip) and differs from the full member's.
- **Support** (local estimate, 24 octets per family on **dev** seeds 6.85e9/6.86e9; not protocol):

  | decision type | S+C+E: unique / flip / near-miss / flip vs pairs / changed vs none | U+C+E: same |
  |---|---|---|
  | first | 12 / 5 / 1 / 0 / 13 | 14 / 0 / 0 / 0 / 5 |
  | after_probe_solved | 24 / 0 / 0 / 0 / 3 | 17 / 1 / 1 / 0 / 17 |
  | after_probe_failed | 12 / 0 / 0 / 0 / 3 | 21 / 0 / 0 / 0 / 4 |
  | after_b1_timeout | 14 / 1 / 0 / 0 / 2 | 14 / 0 / 0 / 0 / 4 |
  | q2_after_H | 10 / 2 / 1 / 0 / 6 | 15 / 1 / 1 / 0 / 5 |
  | q2_after_notH | 15 / 3 / 2 / 0 / 16 | 14 / 0 / 0 / 0 / 6 |

- **Pair-level flips are absent.** A triple's decision is essentially always predicted by one of its pairs (consistent with the screen's joint flip ~1%). So "flip vs pairs" is not an evaluable contrast. The octet contrast is **vs singles**, and it is sparse:
  - S+C+E: ~0.46 flip units per octet;
  - U+C+E: ~0.13 flip units per octet.
- **Registered sizes (proposal).** S+C+E **320** octets (expected ~150 flip units, ~50 with a near-miss); U+C+E **160** octets (expected ~20 flip units: descriptive only).
- **Always reported, with larger support:**
  - member-wise accuracy on all octets;
  - `changed_vs_none` conditional accuracy (~40% of S+C+E units);
  - the one-factor intervention pairs (below).

**On-manifold one-factor interventions** (`one_factor_pairs`, 12 per octet): pairs of members that differ in exactly one factor, both with a unique optimum. The scorer reports:
- P(model changes its action | the optimal action changes);
- P(model changes | the optimum is unchanged): false change;
- P(the new action is optimal | the optimum changes).

"Policy change only where expected utility changes" = high sensitivity and low false change.

## 5. Matched data-volume arms (design v2 revision 4; registry B-ARMS)

"Volume" = the number of distinct configurations. Every arm trains 4,000 updates x 64 on-policy episodes (256,000 episodes, the L1 recipe), so updates and total examples are equal across all arms and volume levels. Only the number of distinct configurations and their combination types differ. Consequence: at 2N, each configuration is visited about half as often.

Composition at N = 384 (`arm_composition`; each type draws from its own seeded stream shared by every arm; B0 is the first half of B1 type by type):

| arm | configs | none | U | S | C | E | pair slots (64 x size) | U / S / C / E frequency |
|---|---|---|---|---|---|---|---|---|
| B0 | 384 | 64 | 64 | 64 | 64 | 64 | U+S 64 | 1/3, 1/3, 1/6, 1/6 |
| B1 | 768 | 128 | 128 | 128 | 128 | 128 | U+S 128 | same |
| **B2** | 384 | 64 | 64 | 106 | 43 | 43 | U+S 22, U+C 21, U+E 21 (dose 0 variety) | same |
| B3 | 768 | 128 | 128 | 213 | 85 | 86 | U+S 43, U+C 43, U+E 42 | same |
| dose1 | 384 | 64 | 96 | 64 | 64 | 32 | U+S 32, S+E 32 | same |
| dose2 | 384 | 64 | 106 | 85 | 43 | 22 | U+S 22, S+E 21, C+E 21 | same |
| dose3 | 384 | 64 | 112 | 80 | 32 | 32 | U+S 16, C+E 16, S+E 16, S+C 16 | same |

- **B0 base set.** None, the 4 singles and **U+S**, the one pair that is a constituent of neither held triple. B0 needs some pair slots, otherwise "per-combination share" cannot be matched.
- **B2 = B0 by replacement.**
  - It keeps every per-factor frequency, the none share and the pair share of B0.
  - It spreads the pair slots over all 3 non-constituent pairs of the primary (dose 0), so it differs from B0 only in combination variety (1 vs 3 pair types) and in the single-type mix this forces.
  - Dropped configurations are replaced by configurations of the added types **matched on the optimal-first-action class** (probe / exact / gather / structure / terminal). The class multiset of B2 equals B0's; `unmatched_first_class` is reported.
- **B3 = B1 by the same procedure.**
- **Dose d = B0 with the first d constituent pairs of S+C+E** (registered order S+E, C+E, S+C: non-historical first, then screening order), sharing the pair slots with U+S. The per-factor frequencies are kept. dose0 = B0.
- **Primary contrast: B2 vs B0** on S+C+E, with **5 seed pairs (seeds 30-34)**. B1, B3, dose1-3 and the factorized arms use seeds 30-32.
  - Training worlds are 8e9 + 1e8 * seed (registered range [1.1e10, 1.15e10)).
  - The torch init is 1000 + seed and the data stream 7000 + seed, identical across arms for a seed.
- **Secondary-hold caveat.** B2's U+C and U+E are constituents of U+C+E, so B2 vs B0 on U+C+E is dose-confounded (reported as such).

## 6. Factorized arms (design v2 revision 5; registry B-FACT)

The raw factor parameters are already public inputs, so the factors supplied or learned are **derived decision quantities** (`FACTOR_FEATURES`, 23 values, functions of the public configuration and public information state only; tested):
- posterior over the hidden type after the history (4);
- P(probe resolves), P(false "solved"), P(H | "solved"), P(b1 resolves);
- expected side cost and expected T surcharge under the belief, and the active event hazard;
- expected remaining cost of the current query per strategy (b2 route, probe-first, b1-first, use, best). These are one-query closed forms (`factor_features` docstring), **not** Q*;
- amortized build value over the remaining horizon (remaining queries x (best no-structure cost - c_use) - C_build), remaining queries, built;
- candidate validity / exactness / trust;
- steps left.

| arm | options (on top of `--train-split b6_B0` or `b6_B2`) | what the policy reads |
|---|---|---|
| PUBLIC-RAW | (none) | the B0/B2 ProbeNet, public inputs (72 = 68 + 4 v3 entries) |
| SUPPLIED | `--inputs factors6 --arch fuse` | heads read tanh(W [z; phi]), phi = exact derived quantities (a localization ceiling; labelled supplied) |
| LEARNED | `--arch fuse --factor-mode learned --aux-weight 1` | an auxiliary MLP head predicts phi from the trunk z (MSE to the exact quantities); the heads read tanh(W [z; **sg**(phi_hat)]) |
| RAW-FUSE (capacity control) | `--arch fuse` | the same fusion layer with phi = 0 |

**Stop-gradient choice (LEARNED).** The policy consumes `phi_hat.detach()`, so the auxiliary head and trunk are shaped by the factor loss (and the trunk also by the policy loss through z). The policy loss cannot turn phi_hat into a free extra policy layer, and phi_hat remains an estimate of the named quantities. The fuse layer adds (hidden + 23) x hidden + hidden parameters to every fuse arm; LEARNED also adds the auxiliary MLP. RAW-FUSE isolates the capacity change.

## 7. Endpoints (`campaign06_bscore.py`) and the proposed primary

Every endpoint is per arm and **per seed**, with configuration-clustered 95% bootstrap CIs (1,000 resamples; octets resampled as whole sets) and seed-paired differences on the same configurations. Support < 20 is flagged.

Groups: each evaluation split, plus `:eligible`, `:all_any_relevant`, `:all_regret_relevant` and `:joint_flip` subsets of the held-out pools. The exact per-factor relevance of each held-out configuration is computed with `--hold-relevance`.

- **Held-out and historical pools** (1 world per configuration):
  - first-decision accuracy, overall and by unique-optimal class (probe / exact / gather / structure / terminal), incl. the uniquely-optimal first-probe rate;
  - **later-decision accuracy by visible context** in the model's own trajectory: query_first (later queries), after_probe_solved, after_probe_failed, after_b1_timeout, other;
  - gap regret, realized regret;
  - success and the success floor (>= .8 x pi* success on the same worlds).
- **Octets:** conditional accuracy on flip units (full member), all-members accuracy, near-miss accuracy, balanced = mean(full, near-miss), and non-flip accuracy; per decision type and pooled.
- **Interventions:** change | optimum change, false change, new action optimal.

**Proposed primary** (to be registered by root with thresholds from dev power, before any held-out evaluation, per B-ARMS). On S+C+E, B2 - B0 over 5 seed pairs:

1. The gap regret difference is < 0, with the pooled clustered CI excluding 0 and >= 4/5 pairs negative.
2. **Later-decision** accuracy (pooled contexts) improves.
3. **Not a global shift.** On the octets, B2 must not buy flip accuracy with near-miss / non-flip errors: balanced accuracy (flip-full, near-miss) and intervention false-change are non-inferior.
4. The first-probe and first-decision endpoints are secondary (first-decision content is small).

## 8. Launch (root; CPU only; 1 thread per job)

Run from a clean worktree at the Track B commit: `SHA=$(research/tools/campaign06_remote.py snapshot)`.

Shorthands:
- `PY=/home/brand/structured-latent-dynamics-campaign03/env/bin/python`
- `R=/home/brand/structured-latent-dynamics-campaign06/results`
- `T=research/tools/campaign04_probeworld_train.py`
- `E="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1"`
- `L=research/tools/campaign06_remote.py launch-cmd`

Expand the loops explicitly (`launch-cmd` runs without a shell). The label parts write into one directory and can run in parallel.

```
# 1. labels (parallel parts; see section 9 for CPU / RSS)
$L e06-tb-labels-train  $SHA --cpu-cap 5000 -- $E $PY $T labels --split-set b6 --parts train --n-train 384 --out $R/e06-tb-labels/labels
$L e06-tb-labels-eval   $SHA --cpu-cap 6000 -- $E $PY $T labels --split-set b6 --parts eval --n-eval 128 --n-hold 400 --n-hist 200 --out $R/e06-tb-labels/labels
$L e06-tb-labels-cf-sce $SHA --cpu-cap 5000 -- $E $PY $T labels --split-set b6 --parts cf --n-cf 320 --out $R/e06-tb-labels/labels
#    (U+C+E octets: run the cf part with --n-cf 160 into a second directory $R/e06-tb-labels/cf-uce/labels, or accept 320 for both)
#    optional subgroup facts: add --hold-relevance to the eval part (+~2x its CPU)

# 2. trainings (<= 4 concurrent; L1 recipe, final checkpoint)
for s in 30 31 32 33 34; do   # primary contrast: 5 seed pairs
  $L e06-tb-b0-s$s $SHA -- $E $PY $T train --labels $R/e06-tb-labels/labels --train-split b6_B0 --rung L1 --seed $s --out $R/e06-tb-b0-s$s/run
  $L e06-tb-b2-s$s $SHA -- $E $PY $T train --labels $R/e06-tb-labels/labels --train-split b6_B2 --rung L1 --seed $s --out $R/e06-tb-b2-s$s/run
done
for s in 30 31 32; do
  $L e06-tb-b1-s$s    $SHA -- $E $PY $T train --labels ... --train-split b6_B1    --rung L1 --seed $s --out $R/e06-tb-b1-s$s/run
  $L e06-tb-b3-s$s    $SHA -- $E $PY $T train --labels ... --train-split b6_B3    --rung L1 --seed $s --out $R/e06-tb-b3-s$s/run
  $L e06-tb-dose1-s$s $SHA -- ... --train-split b6_dose1 ...   (dose2, dose3 likewise)
  $L e06-tb-sup-s$s   $SHA -- ... --train-split b6_B0 --inputs factors6 --arch fuse ...
  $L e06-tb-lrn-s$s   $SHA -- ... --train-split b6_B0 --arch fuse --factor-mode learned --aux-weight 1 ...
  $L e06-tb-rawf-s$s  $SHA -- ... --train-split b6_B0 --arch fuse ...
done

# 3. evaluation of every run (after its training and the eval + cf label parts)
$L e06-tb-<arm>-eval-s$s $SHA -- $E $PY $T eval --labels $R/e06-tb-labels/labels --run $R/e06-tb-<arm>-s$s/run \
     --split-set b6 --episode-rows --cf --world-offset 0 --worlds 1
$L e06-tb-refs $SHA -- $E $PY $T references --labels $R/e06-tb-labels/labels --out $R/e06-tb-refs.json --split-set b6 --world-offset 0 --worlds 1

# 4. score (pure Python)
$PY research/tools/campaign06_bscore.py --ref B0 --arm B0 <5 runs> --arm B2 <5 runs> --arm B1 <3> --arm B3 <3> \
    --arm dose1 <3> --arm dose2 <3> --arm dose3 <3> --arm SUP <3> --arm LRN <3> --arm RAWF <3> --out $R/e06-tb-score.json --md $R/e06-tb-score.md
```

(Section 9 fills in the CPU estimates from the metered smoke.)
