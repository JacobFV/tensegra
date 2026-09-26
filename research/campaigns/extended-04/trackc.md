# Track C: learned metacognitive appraisal and control over a frozen depworld policy

**Status: BUILT, smoke-tested, not launched.** Written 2026-09-26 by the Track C builder (branch `campaign/e04-trackc`). It implements design.md Track C with **v2 revisions 5–8**, which supersede v1, and review F7–F12 and F16. Every number below comes from development smokes on dev seeds. None is a result.

Code:
- `src/tensegra/campaign04_telemetry.py`: telemetry d_t.
- `src/tensegra/campaign04_meta.py`: runner, agents, branching, labels and the appraisal model.
- `src/tensegra/campaign04_meta_train.py`: packing, training and registration.
- Tools: `research/tools/campaign04_c_{configs,labels,train,eval,analysis,inspect}.py`.
- Tests: `tests/test_campaign04_meta.py`.

## 1. Telemetry d_t (`c-telemetry-v1`)

d_t is an immutable, versioned `Telemetry(version, step, values)`, measured at decision t **before** the action is chosen. Its inputs are:
- the public `DepObservation`s;
- the frozen actor's distribution at the current observation (numbers computed under `no_grad`);
- the progress diagnostic v1 state (flags and masks);
- the agent's own intervention history;
- the compute units the deployment charged for actor forwards.

It never reads the hidden spec, the evaluator, the solver, an RNG, a label or an outcome. The layout is fixed, with fixed scaling (`LAYOUT`, 89 features):

| group | features |
|---|---|
| policy | p_top1, entropy/log n, top1 − top2 margin, log1p(n)/6, p(default choice) |
| budget | steps used/limit, steps remaining/limit, work remaining/work_0, travel remaining/travel_0, remaining steps/96 |
| cost | public cumulative cost ×10, compute units/100, calls/10, inspections/20 |
| diagnostic | last class one-hot (7 + none), per-class counts (log1p/4), flagged-state bit, masked keys/8, default ≠ greedy bit, no-progress run/6, latent stalls/4, repeat rejections/4 |
| feedback | last status one-hot (11), last reason one-hot (REASONS + none + other) |
| info_events | new-information flag, event now, events, any revocation, requirements version/5, requirements known, map known, fraction of items known |
| stage | selection committed, assignment committed, verified, pending items/categories, pending assignment, at destination, drafts/10, records/10, retrieved/10 |
| control | last intervention one-hot (default, sample, mask_top, stop, none), interventions/10 |

The **public cost** is rebuilt from public facts:

  steps·action + travel·travel + inspections·observation + (work_0 − remaining work)·work + compute·compute

It is tested equal to the environment's cost at every decision. It **excludes** the metacontrol charge, so charging or not charging the metacontroller never changes behaviour.

Removing a telemetry group:
- `group_mask` zeroes a group at evaluation (reliance);
- `MetaSpec(drop_groups=…)` zeroes it in both training and evaluation (necessity).

## 2. Deployment, interventions and the declared continuation

**D0 = greedy + R-mask** is the default deployment and the continuation of every label (v2 rev. 6). The runner's `r_mask` choice uses the same torch operations as `campaign04_deploy`. The runner is tested bit-identical to `deploy_episodes` for greedy, sampled, r_mask and r_sample: actions, utility and progress summaries.

The interventions u at a decision are:

| u | action |
|---|---|
| `default` | the D0 choice (no intervention) |
| `sample` | one action sampled from the **full** π (float64 softmax, inverse CDF, one uniform from the episode's intervention stream `sampling_rng_seed(world, 20260928)`) |
| `mask_top` | the highest-logit action other than the default's choice. R-masked keys are not excluded, so this is a pure "second choice". It is unavailable with fewer than 2 candidates. |
| `stop` | abstain (ends the episode) |

Each u is followed by D0. The controller re-decides at every step, so it is a **one-step rollout improvement over greedy + R-mask** (F9). This is its name in every report.

## 3. Counterfactual labels (`c-labels-v1`)

**Branching.** At a selected decision, `snapshot` clones three things with `clone_branch`:
- the environment, sharing the executor;
- the actor's recurrent row (None for the lightweight X1 actors);
- the progress tracker.

Each needed branch then applies its first action, charging the step-t actor forward exactly once as on the main line, and runs D0 to the end.

**Label definition.**
- ΔU(u) = U_final(branch) − U_t, where U_t is the utility **before** the step-t charges.
- This uses the same `DepWorkshop` utility as evaluation, so every cost lands once, in the environment.
- Metacontrol charges are not in labels: D0 has no metacontroller, and the forward is u-independent (F16), so c_meta(u) = 0 in the argmax.
- The solver is the persistent `BoundedSolver` behind the exact `SolverCache` (F7).

**Stochastic intervention.** For `sample`, K = 4 draws are taken from π with a separate label stream. Mean and variance are both available.

**Deduplication.** A draw equal to the default's action reuses the default branch. A draw equal to the mask_top or stop action reuses that branch. Duplicate draws share one branch.

**Use of labels (F8, v2 rev. 7).** Branch returns are single-world, hindsight outcomes. They are used **only as regression targets**:
- step level (D0 base episodes, every decision; the main line after t *is* D0):
  - success (BCE);
  - remaining cost (squared error);
  - ΔU(default) (squared error);
- point level:
  - ΔU(default), success and cost under the default branch (non-D0 episodes);
  - paired advantages ΔU(u) − ΔU(default), with a common world across u (squared error; `sample` per draw).

No class target is ever formed. Inputs are telemetry only. A test corrupts every target and shows the inputs and predictions are unchanged.

**State distribution (registered).**
- **Base kind per world.** It is chosen from a hash of the seed:
  - **d0** (greedy + R-mask) .50;
  - **eps** (D0, with probability .15 per decision a random `sample` or `mask_top`) .25;
  - **sampled** (T = 1 from π, recorded as `sample`) .25.
  - eps and sampled worlds cover intervention histories like those the controller creates.
- **Decision points.** Points are chosen by an online Bernoulli draw per public stratum, with at most 4 per episode:
  - **flagged** (R-mask holds a mask here, or the last step was no-progress): .5;
  - **low_margin** (top1 − top2 < .2): .12;
  - **other**: .04.
- **Weights.** Inclusion probabilities are recorded, and so are skipped points once the cap is reached. Selection depends only on public features plus independent randomness, so the regression target E[ΔU | I_t, u] is unbiased at every I_t. Only the loss weighting over states changes. Training is unweighted.
- **Default checks.** On d0 worlds the default branch is supplied by the main line. At 10% of points it is also branched as a check: 17/17 matches in the smokes, and every point in the tests.

**Privilege and ledger.** The environment copy is an oracle-simulation privilege of label generation only. It is charged to the offline label ledger (job receipts) and never to episode utility, and it is not available at evaluation.

**Seeds and worlds.**
- Label worlds: **150,000,000 + i**, the same worlds for every base (paired).
- Condition: iid_f0 for even i, iid_f2 for odd i (the actors' training mix); address namespace `e04c-labels`.
- `check_fresh` asserts disjointness from every extended-03 range, A1 (130M), A2 (140M), the ext-04 test seeds (2.15e9) and probeworld (4.1–4.7e9, 8e9).

## 4. Appraisal model m_t

- **Architecture.** `MetaAppraisal` is a one-layer GRU with 64 hidden units over d_t, updated at every decision (the automatic path).
- **Heads:**
  - P(verified success | D0);
  - expected remaining cost under D0;
  - Q̂(I_t, default);
  - advantages Q̂(I_t, u) − Q̂(I_t, default) for u ∈ {sample, mask_top, stop}.
- **Size (disclosed).** 30,150 parameters: GRU 3·(89·64 + 64·64 + 2·64) plus heads. The X1 actor has ≈ 2.35M, a ratio of about 1.3%.
- **Stop-gradient.** The actor trunk is **not** an input. Actor information enters only as telemetry numbers computed under `no_grad`, so no loss can reach actor parameters. The losses are proper scoring rules only (BCE and MSE); nothing rewards higher predictions.
- **Training.** Adam (lr 3e-3), batches of 64 episodes, 30 epochs, full sequences (≤ 96 steps). The dev split is 20% of label worlds by a hash of the seed.
- **Seeds.** 3 metacontroller seeds per base (0, 1, 2), each trained on that base's own labels, since Q depends on the base policy. Seeds differ in initialization and minibatch order. The criterion is split by base type (v2 rev. 8).

## 5. Controller and registered constants

The learned rule is

  u_t = argmax_u Q̂(u) − c_meta(u)   if max_u (Q̂(u) − Q̂(default)) > m,   else default,

with c_meta(u) = 0 (F16).

**Registration of m (per model).** m is registered on the **dev split of the label worlds**, never on evaluation worlds. The criterion is the mean realized one-step gain of the rule over default on the dev points: ΔU(u chosen) − ΔU(default), using the K-draw mean for sample. The grid is {0, .0025, .005, .01, .02, .03, .05, .1, .2, ∞}. Ties go to the larger m, and m = ∞ means "never intervene".

**Registration of τ (automatic threshold rule).** The rule is: `sample` when P̂(success) < τ. τ is registered the same way over {0, .05, …, .9}, with ties going to the smaller τ.

**Metacontroller charge (protocol-A1).** Each meta decision charges neural_work_per_forward × (meta params / actor params), about 0.0128 units, i.e. about 1.3e-6 utility per decision. It is charged **in the episode**. Rows also report utility without it. The diagnostic's and telemetry's CPU is not charged (protocol-A1). A secondary column converts it at the actor's measured units per CPU-second ("the same rate").

## 6. Arms (identical inputs, identical worlds)

| arm | behaviour |
|---|---|
| fixed:greedy / fixed:sampled (T=1, seed 20260927) / fixed:r_mask (= D0) / fixed:r_sample | A1 semantics |
| appraisal_only | predictions logged, behaviour D0. It is also the **control pathway disabled** arm: the same behaviour with the forward charged. Calibration only. |
| learned | the rule in §5 |
| random_matched | intervenes at the **learned arm's per-condition decision rate**, drawing u from the learned arm's intervention mix (per-world stream 20260929). It is run in the same job right after `learned`. |
| threshold | automatic rule: `sample` when P̂ < τ |

## 7. Causal tests (F11)

| test | implementation |
|---|---|
| remove a group: **reliance** | `zero:<group>`: the learned rule with the group zeroed at evaluation |
| remove a group: **necessity** | `model:drop-<group>`: a model retrained with the group zeroed, with its own registered m |
| shuffled-target training | `model:shuffled`: step targets permuted across all train positions and point targets across points. m is registered against the **true** dev targets. |
| shuffled telemetry at test | `shuffle_telemetry`: each episode receives another active episode's d_t (a batch roll at each step) |
| disable the control pathway | `appraisal_only` |
| clamp u | `clamp:<u>`: the learned rule decides *when*, and u is replaced by `<u>`. Together with `random_matched` (right rate, random timing), this separates "when" from "which". |
| predicted vs actual effect | `--branch-eval N`: on the first N worlds of `appraisal_only` and `learned`, up to 2 decisions per episode (Bernoulli .1) are cloned and every u is branched under D0. This covers **controller-induced states** (F9). |

The branch-evaluation metrics are:
- Q̂(default) MAE;
- advantage MAE and sign agreement for each u;
- predicted vs actual gain at the controller's interventions;
- ECE of P̂ against success under the default branch.

This is an offline ledger item and is charged.

**Calibration** (F11) is measured on appraisal_only, where behaviour after t is D0:
- step-level and first-decision ECE: 10 equal-width bins, only bins with ≥ 20 examples, count-weighted |mean prediction − frequency|;
- Q̂(default) MAE against the realized ΔU, with metacontrol charges after t added back.

No DAgger relabelling round is registered. If the budget permits, one may be added **before** any evaluation output is seen.

## 8. Bases, evaluation and criterion

**Bases** (frozen; paths and hashes from the A1 generator, remote `/home/brand/tensegra-campaign03/results/...`; hash verified on load):

| base | type | F10 stratum |
|---|---|---|
| p1-boot-x1-r0, r1, r2 | competent | competent |
| p1-rl-x1-r0, r2 | loop-prone | loop-prone |
| p1-rl-x1-r1 | loop-prone | **stable** (shown separately) |

**Evaluation worlds.**
- **160,000,000 + 100,000·i** for conditions iid_f0, iid_f2, events_train_kinds_p1 and foreign4 (256 per condition); namespace `e04c-eval`.
- Labels use only the IID mix, so events and foreign4 are transfer conditions.

**Criterion** (v2 rev. 8, on the IID group, per lineage, using the seed-mean learned utility per world, paired with the best fixed rule by IID mean utility):
- **Loop-prone bases:** learned − best fixed ≥ **.02**.
- **Competent bases:** learned − best fixed ≥ **−.01**, **and** cost or no-progress below D0.
- **Lineages:** the criterion needs **≥ 2/3 lineages per base type**.

`campaign04_c_analysis.py` also reports:
- every condition;
- per-seed paired differences;
- learned vs random_matched and vs threshold;
- causal arms vs learned;
- calibration;
- CPU by stage.

**Power (F10).** Learned and D0 trajectories coincide wherever the controller does not intervene, so the paired variance is small: 512 IID worlds resolve differences of about .01 when intervention rates are low. A1's per-world variances should be used to confirm this before root registers the protocol. The analysis prints paired SEs.

## 9. Smoke results (dev seeds 2.17e9+, non-protocol; descriptive only)

| smoke | measurement |
|---|---|
| Labels, p1-boot-x1-r0 (40 worlds; strata p .5/.25/.06) | 1.58 core-s per world, 3.3 points per world; 0.38 core-s per point |
| Labels, p1-rl-x1-r0 (96 dev worlds; strata .5/.12/.04) | 263.6 core-s (**2.75 core-s per world**, 3.6 points per world, **0.76 core-s per point**), ≈ 4.5 branches per point; default checks 14/14 |
| Actor forward (CPU, width 1024, 1 thread) | ≈ 3.2–3.6 ms per decision in batch, ≈ 60–80% of label CPU. The actor forward dominates. |
| Label content, rl-r0 (348 points; descriptive) | mean one-step gains over D0 are ≈ 0 for sample and mask_top (max +.26/+.10), −.71 for stop. Stop beats D0 at 22% of points (where D0 goes on to fail). The headroom is small and concentrated on predicting failure. |
| Train, 96 worlds | 2.1 s per model. Registered m = ∞ (never intervene), τ = .8; dev ECE .12. The data were too few to read anything. |
| Eval, rl-r0, iid_f0, 64 dev worlds | fixed:greedy .000 / −.122, sampled .844 / .695, r_mask .891 / .780, r_sample .891 / .782, appraisal_only = learned (m = ∞) .891 / .7795, threshold (τ .8) .891 / .770 with no-progress 10.3 → 6.2. Cost per episode: r_mask 0.16, greedy 0.40, sampled 0.27, meta arms +0.01 core-s (plus branch-eval). |
| Eval, boot-r0, 16 worlds | ≈ 0.07 core-s per episode. |

Before the `--dev-worlds` switch existed, one 16-world smoke (boot-r0, iid_f0, arms fixed + core + 5 causal) ran on the **first 16 registered evaluation worlds** (160,000,000–160,000,015). It used a throwaway model trained on 40 label worlds. No registered constant was derived from it (m and τ come from dev labels; sizes come from timing only). It is disclosed here; root may exclude those 16 worlds or accept it.

## 10. Budget (Track C ≈ 25k core-s; labels ≤ 12k; CPU only)

| item | plan | estimate (core-s) |
|---|---|---:|
| Labels | 768 worlds per base (24 chunks × 32), 6 bases: boots ≈ 1.3–1.6 per world, rl-r1 ≈ 1.5, rl-r0/r2 ≈ 2.75 | **≈ 8.7k** (cap 12k: chunks 16–23 are launched only if the label ledger is < 8k) |
| Training | per base: main + shuffled × seeds 0–2, drop:<8 groups> × seed 0 = 14 models × ≈ 15 s | ≈ 1.3k |
| Eval tier 1 | fixed(4) + learned/random_matched/threshold × 3 seeds + appraisal_only × seed 0, 4 conditions × 256; branch-eval 64 worlds per IID condition | ≈ 10k + 0.4k |
| Eval tier 2 (causal) | 3 clamp + 8 zero + shuffle_telemetry + 9 retrained, seed 0, iid_f0 + iid_f2 × 128 | ≈ 3.6k (shrink to 64 per condition if the ledger is tight) |
| **Total** | | **≈ 24k** |

GPU: none. If CPU is too slow in wall time, the actor forward is the only candidate for a GPU. That would change the evaluation device relative to the labels, so it would have to apply to labels, training states and evaluation alike. It is not proposed.

## 11. Root launch commands

All commands run from a source snapshot, under the job wrapper, with `CUDA_VISIBLE_DEVICES=` and 1 thread. `R=/home/brand/tensegra-campaign04/results`, and B ranges over p1-boot-x1-r0, p1-boot-x1-r1, p1-boot-x1-r2, p1-rl-x1-r0, p1-rl-x1-r1, p1-rl-x1-r2.

```
# labels: 3 jobs per base (8 chunks each; about 3-12 min wall per job); launch the c16-23 jobs last
python research/tools/campaign04_c_labels.py --base $B --chunk 0 1 2 3 4 5 6 7 --output $R/c-labels-$B
python research/tools/campaign04_c_labels.py --base $B --chunk 8 9 10 11 12 13 14 15 --output $R/c-labels-$B
python research/tools/campaign04_c_labels.py --base $B --chunk 16 17 18 19 20 21 22 23 --output $R/c-labels-$B
# training (after all chunks of the base)
python research/tools/campaign04_c_train.py --labels $R/c-labels-$B --base $B --variants main shuffled --seeds 0 1 2 --output $R/c-train-$B
python research/tools/campaign04_c_train.py --labels $R/c-labels-$B --base $B --variants drop:policy drop:budget drop:cost drop:diagnostic drop:feedback drop:info_events drop:stage drop:control --seeds 0 --output $R/c-train-$B
# evaluation tier 1: one job per base x condition (IID conditions add branch evaluation)
python research/tools/campaign04_c_eval.py --base $B --models $R/c-train-$B --seeds 0 1 2 --conditions iid_f0 --arms fixed core --single-seed-arms appraisal_only --examples 256 --branch-eval 64 --output $R/c-eval-$B-iid_f0
python research/tools/campaign04_c_eval.py --base $B --models $R/c-train-$B --seeds 0 1 2 --conditions iid_f2 --arms fixed core --single-seed-arms appraisal_only --examples 256 --branch-eval 64 --output $R/c-eval-$B-iid_f2
python research/tools/campaign04_c_eval.py --base $B --models $R/c-train-$B --seeds 0 1 2 --conditions events_train_kinds_p1 --arms fixed core --single-seed-arms appraisal_only --examples 256 --output $R/c-eval-$B-events
python research/tools/campaign04_c_eval.py --base $B --models $R/c-train-$B --seeds 0 1 2 --conditions foreign4 --arms fixed core --single-seed-arms appraisal_only --examples 256 --output $R/c-eval-$B-foreign4
# evaluation tier 2 (causal; seed 0; compared with the tier-1 learned rows on the common worlds)
python research/tools/campaign04_c_eval.py --base $B --models $R/c-train-$B --seeds 0 --conditions iid_f0 iid_f2 --arms causal retrained --examples 128 --output $R/c-eval-$B-causal
# analysis
python research/tools/campaign04_c_analysis.py --eval $R/c-eval-* --train $R/c-train-* --labels $R/c-labels-* --output $R/c-analysis.json
```

## 12. Tests (`tests/test_campaign04_meta.py`)

- Telemetry layout and group masks. Telemetry is **public-only** and identical for identical visible histories: two worlds with different hidden requirements give identical observations and identical d_t.
- The public cost equals the environment cost at every decision.
- Fixed modes are bit-identical to `deploy_episodes`: greedy, sampled, r_mask and r_sample give the same actions, utility and progress summary.
- **The controller's default path is exactly greedy + R-mask**, for both appraisal_only and learned with m = ∞. The **metacontrol charge is applied** (compute units + units × forwards; utility − charge × price), and with the charge off the episodes are identical.
- mask_top and stop semantics.
- **Clone-branch labels reproduce the main line when u = default**, bit for bit. **Costs are charged once**: ΔU(stop) is exactly one action plus one actor forward, and main-line compute units = decisions.
- Labels exist for every base kind; packing and merging work.
- **Label targets are never inputs**: corrupting every target leaves the inputs and predictions unchanged, and the shuffled-target control leaves the inputs intact.
- Training, registration and group drop; parameters < 40k and hidden ≤ 64.
- The labels → train → eval → analysis tools run end to end with a tiny stand-in actor, including no-overwrite, and appraisal_only equals fixed:r_mask up to the metacontrol charge.
