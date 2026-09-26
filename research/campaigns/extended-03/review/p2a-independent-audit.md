# P2a independent audit (raw-row reconstruction)

**Auditor.** An independent subagent. It did not write the P2a implementation, `campaign03_p2a_analysis.py`, `campaign03_p2a_collapse_audit.py` or the no-progress operationalization. It read the root's tools only after its own screening reconstruction was finished, and then only to explain discrepancies.

- **Branch:** `campaign/e03-p2a-audit`, from `campaign/extended-03` 64905883.
- **Data:** read-only copies from pro6000 (`p2a-screening`, `p2a-partd`, and the run state, protocol and development data), plus read-only torch checks run from `source-033cd135`.
- **Numbers:** [p2a-independent-audit.json](p2a-independent-audit.json).
- **CPU used:** about 0.8k core-s on pro6000 (CPU only, at most 3 processes) and about 0.2k core-s locally.

**Tools** (all in `research/tools/`; P1's shared helper aside, each is standard library only unless marked torch):

| Tool | What it does |
|---|---|
| `campaign03_p2a_audit_reconstruct.py` | Screening metrics, readings S1–S3 and seed/world integrity. It takes correct-reuse windows from the P1 auditor's `campaign03_p1_audit_reconstruct.py`. |
| `campaign03_p2a_audit_compare.py` | Compares the auditor's values with the root's, cell by cell. |
| `campaign03_p2a_audit_runs.py` | Development curves, the replay trace hash, the config diff and the deployment rule, all from raw development rows. |
| `campaign03_p2a_audit_torch.py` (torch) | Weight and optimizer identity; the auditor's own rollout loop, KL and gradient decomposition. |
| `campaign03_p2a_audit_curves.py`, `campaign03_p2a_audit_trcurves.py` (torch) | Non-tensor checkpoint fields and per-update training curves. |

## Summary

- **Reproduction:**
  - All readings reproduce: S0 true, S1 true, S2 false, and S3 as the root states it.
  - Every S1/S2 input matches the root's to at least 1e-12.
  - 627 values were compared (13 policy/mode rows × 4 conditions × 11 metrics, plus the group means). Only two differ by more than 0.001, and neither enters a reading.
- **Replay:**
  - C0's config is byte-identical to P1's.
  - Model **and optimizer** tensors are identical at attempts 0, 5, 12, 18 and 24–29.
  - Per-update training curves are identical once timing fields are removed.
  - The 30 development files are identical row by row once timing fields are removed.
- **C1:** the only config difference is the anchor. The anchor term is present and nonzero at 1,799 of 1,800 updates. The tranche KL is unchanged.
- **Deployment:** both decisions re-derive exactly (C0: attempt 24; C1: attempt 29).
- **Integrity:** clean.
- **Part D spot-check:** greedy results reproduce exactly on all 64 probe worlds. Sampled results agree within sampling noise.
- **Claims to weaken:**
  1. C1's "preservation" is, in policy terms, indistinguishable from a near-freeze at the bootstrap (§7).
  2. The Part D reading "critic dominates / clip fraction 1.00" overstates what those measurements show (§6).
  3. C1 shows early loop signs under sampling that S1 does not read (§7).

## 1. Screening reconstruction (IID group; auditor = root to 3 d.p. in every cell)

| policy/mode | success | utility | work/success | no-progress | idempotent | to cap |
|---|---|---|---|---|---|---|
| bootstrap/greedy | 0.949 | 0.872 | 137.0 | 0.037 | 0.036 | 0.023 |
| bootstrap/sampled | 0.896 | 0.815 | 163.4 | 0.013 | 0.002 | 0.014 |
| c0_final/greedy | 0.000 | −0.145 | n/a | 0.712 | 0.700 | 1.000 |
| c0_final/sampled | 0.908 | 0.799 | 234.4 | 0.080 | 0.033 | 0.090 |
| c0_deployed/greedy | 0.900 | 0.820 | 135.7 | 0.177 | 0.089 | 0.100 |
| c0_deployed/sampled | 0.896 | 0.792 | 210.7 | 0.066 | 0.015 | 0.094 |
| c1_final = c1_deployed / greedy | 0.949 | 0.864 | 166.8 | 0.032 | 0.029 | 0.051 |
| c1_final = c1_deployed / sampled | 0.916 | 0.827 | 181.9 | 0.031 | 0.005 | 0.066 |
| p1_rl (= c0_final on every row) | identical | | | | | |
| dep_reuse/greedy | 0.979 | 0.905 | 126.2 | 0 (by rule) | 0 | 0.000 |

**Readings** (the auditor's values):
- **S1:** success .94922 vs .94922, and no-progress .03188 vs .03686. **Holds.**
- **S2:** utility .8638 vs .8723 (needs ≥ .8823). Work per success 166.8 vs 137.0, a ratio of 1.218 (needs ≤ 0.9). **Fails.**
- **S3:**
  - C0: final .000 vs deployed .900. The result is a deployment rule, not stable learning.
  - C1: final = deployed = .949.

Correct reuse and invalid reuse (the P1 definitions) match the root's exactly in all cells.

**Discrepancies greater than 0.001.** Neither affects a reading.
1. **steps_to_cap in c0_deployed/sampled/events_train_kinds_p1:** auditor .0820, root .0781.
   - One episode was verified exactly at step 96. The root excludes verified episodes, which is the better reading; the auditor's first pass included it.
2. **no_progress_per_episode:** this is a count, not a rate. The largest difference is 0.012 per episode, in sampled cells.
   - The trace lacks the post-state of each episode's final action. The root re-simulates the public state from `history`; the auditor infers it only when the final action repeats an earlier (state, key) pair or re-retrieves a record.
   - The corresponding **rates differ by at most 0.00028**.

**Two definitional notes.**
- **Short-cycle vs no-progress.** The root's `short_cycle_rate` is sometimes slightly below its `no_progress_rate`, by at most 0.0003 in sampled cells. So the root does not count every idempotent repeat as a length-1 cycle, which the registered text requires ("an accepted action with no effect counts as a cycle of length 1"). The auditor's short-cycle rate equals its no-progress rate by construction. No reading uses the short-cycle rate alone.
- **Part D uses a different decision state.** It has no `depends_on` and no event breaker. On the same 64 probe worlds the r2 bootstrap's greedy no-progress rate is **.068 under Part D's definition and .031 under the registered one**. Part D's no-progress rates are therefore **not comparable** to the screening rates.

## 2. Replay (S0)

- **Configs.** `configs/campaign03/p2a-c0-x1-r2.json` is byte-identical to `p1-rl-x1-r2.json` (sha256 `39131390…e123`). Both runs' `protocol.json` files are identical too (sha256 `35fe60f1…`).
- **Weights and optimizer.** At attempts 0, 5, 12, 18, 24, 25, 26, 27, 28 and 29:
  - `torch.equal` holds on all 18 model tensors;
  - the AdamW state is identical.
- **Non-tensor checkpoint fields that differ:**
  - `config`: the new code adds the defaults `anchor_kl_weight: 0.0` and `anchor_checkpoint: null`;
  - `source_sha256`;
  - `data_hash`, which contains wall-clock time;
  - `resume_history`, which differs only in the checkpoint sha256 values.
- **Training curves.** The `curves` field (660–2,400 entries) is **identical after timing fields are removed**. That covers loss, gradient norm, objective parts and training utility for every update.
- **Development data.** The development success and utility curves are identical at all 30 attempts: …, .945, .922, .609, .133, .000, .000. The development jsonl rows are identical once timing is stripped, and the file sha256 values match `state.json`. The training seed intervals are identical.
- **Conclusion.** S0 is confirmed. The collapse is deterministic under this recipe and GPU nondeterminism is ruled out. On the new screening worlds, `p1_rl` and `c0_final` also produce identical metrics in every cell.

## 3. C1 integrity

- **Config.** The C1 config differs from C0 only in `train.anchor_kl_weight: 0.3` and `train.anchor_checkpoint`, which points to the x1-r2 bank at sha256 `74ff10a6…2fc9`. That sha256 is also in `p1-banks.json` and is the screening bootstrap. The same two fields are the only difference between the two runs' `protocol.json` files.
- **Anchor provenance in `state.json`:** `anchor.sha256_verified_at_initialization: true`, `replaced: never`. All 30 allocations log `training_timing.anchor.verified: true` with the same sha256.
- **The anchor is active.** `kl_to_anchor` appears in the objective parts of every RL update, and 1,799 of 1,800 are nonzero; the zero is the first update, when the policy equals the anchor.
  - Per-tranche means rise from .007 (attempt 0) to .028–.032 (attempts 25–29).
  - Weighted by 0.3, the term contributes about 0.002–0.01 to the loss.
- **The tranche KL is unchanged.** `kl_weight` is still 0.3, and `kl_to_round_start` is present at every update (per-tranche mean .006–.022; C0's is .011–.032).
- **What the anchor visibly does in training.** C0's policy entropy rises from .82 to 1.60 over the 30 tranches. C1's stays between .84 and .96. The two arms' critic losses are similar.

## 4. Deployment rule (re-derived from raw development rows)

**Bootstrap reference:** P1 x1-r2 bank at attempt 5. Its development success is .953125 and its development utility .8808375, on the same 128 seeds (hash `1347e8fd…`) as every C0/C1 development file.

**Thresholds:** success .933125 and utility .8608375.

| Arm | Qualifying attempts | Deployed (latest qualifying) | Registered decision |
|---|---|---|---|
| C0 | 0–5 and 7–24 (24 of 30). Attempt 6 fails on success (.930). Attempts 25–29 fail. | attempt 24, sha256 `9acf814e…` | matches `deployment-c0.json` |
| C1 | all 30 | attempt 29, sha256 `c3bb1bb1…` = final | matches `deployment-c1.json` |

- No attempt lies within float tolerance of a threshold.
- The screening config binds exactly these sha256 values (bootstrap `74ff10a6`, c0_final `e06ab865`, c0_deployed `9acf814e`, c1 `c3bb1bb1`, p1_rl `d320a7ca`). Each matches the allocation sha256 in its run's `state.json`.

## 5. Screening integrity

- **Seeds.** Seed ranges are 120,000,000 + 100,000·i + [0, 255] for i = 0..3: 1,024 distinct seeds, no gaps. They are disjoint from:
  - P1/P2a training seeds (3.0–3.45e9);
  - development seeds (3.900–3.902e9);
  - P1 sealed seeds (110.0–110.7M);
  - preflight calibration seeds (2.05e9);
  - Part D probe seeds (1.990e9).

  No config in `configs/` other than `p2a-screening.json` uses 120.0–120.3M.
- **Worlds.** Within each condition and mode, every policy file has the same seed order as `worlds.jsonl.gz`. Greedy and sampled modes use the same worlds.
- **Sampling.** Every sampled row records `policy_mode: sampled`, `sampling_seed` 20260926 and a per-world `sampling_rng_seed` (256 distinct per condition). These are identical across the five learned policies, so the sampling streams are paired across policies.
- **Remaining caveats:** the screening is not sealed, and `dep_reuse` has no trace; its zero loop rates are set by the registered rule, not measured.

## 6. Part D spot-check

The auditor used its own batched rollout loop on CPU over the full 64 Part D probe worlds. Sampled mode used 2 samples per world with the auditor's own torch seeds.

| r2 checkpoint | greedy success, auditor / Part D | greedy utility | KL(boot‖ck) on boot states | KL(boot‖ck) on own states (argmax disagreement) | sampled success, auditor / Part D |
|---|---|---|---|---|---|
| boot | .953 / .953 | .8749 / .8748 | — | — | — |
| a25 | .938 / .938 | .8587 / .859 | .2463 / .2463 | .411 (.213) / .411 (.213) | .898 / .914 |
| a26 | .625 / .625 | .5186 / .519 | .2433 / .2433 | .693 (.576) / .693 (.576) | .938 / .914 |
| a29 | .000 / .000 | −.1445 / −.145 | .3117 / .3117 | .960 (.747) / .960 (.747) | .898 / .922 |

- **Reading 1 (greedy collapse, sampled intact) and reading 2 (KL rises on own states only) reproduce.** The greedy results and KLs match exactly, including state counts (1,937 bootstrap states). The sampled results differ only by sampling noise (SE ≈ .025).
- **Gradient reading ("critic dominates the shared encoder; clip fraction 1.00").** The auditor computed its own actor / 0.5·critic / −.003·entropy / .3·KL gradients with `autograd.grad` on an 8-world sampled batch:

| Checkpoint | ‖critic‖/‖actor‖ on shared params | Total norm |
|---|---|---|
| bootstrap | 64 (Part D: 84) | 60 |
| r2 a25 | **1.04** (Part D: 1.6) | 2.4 |
| C1 a29 | 6.9 | 5.3 |

- **What the gradient check does not support:**
  1. **The 30–80× ratio holds only at the bootstrap.** Imitation never fit the value head, so its MSE is .51–.74. After that, Part D's own table gives ratios of 0.3–11. The critic is larger at 32 of 36 non-bootstrap checkpoints, but by 1.5–7× typically, not 30–80×. The ratio is below 1 at r2 attempts 19, 21 and 24, immediately before the collapse.
  2. **The stable lineage r1 shows the same ratios and the same clip fraction** (critic larger at 11 of 12, clip 1.00). The measurement does not separate collapsing from stable lineages, so it cannot localize the collapse.
  3. **"Clip fraction 1.00, so the direction of each shared step is mostly the critic's" is a non sequitur.**
     - Norm clipping rescales the whole gradient and does not change its direction. AdamW then normalizes per coordinate, which makes the step largely scale-invariant.
     - Over actual training, pre-clip norms exceed 1 at 99.9% of C0 updates and 99.1% of C1 updates (per-update curves; means 4–21). So clipping is always on, as Part D says, but the fact carries little information.
     - With the checkpoint's saved AdamW moments, the auditor measured the fresh-gradient part of the next step on shared parameters. At r2 a25 it aligns more with the actor-only step (cos .86) than with the critic-only step (.51). At C1 a29 it aligns with the critic (cos .95 vs .31).
     - Critic influence on the shared direction is therefore checkpoint- and batch-dependent. It is present in the non-collapsing C1 as well.
  - **Recommendation:** keep "critic-dominated shared updates" only as a weak candidate, and drop the clip-fraction argument.

## 7. Freeze vs preservation, and other interpretation risks

**KL from the bootstrap to the checkpoint** (64 probe worlds; states visited by the bootstrap's greedy rollouts / by the checkpoint's own greedy rollouts):

| Checkpoint | KL on boot states (argmax disagreement) | KL on own states (argmax disagreement) | Greedy success | Greedy no-progress | Sampled no-progress |
|---|---|---|---|---|---|
| C0 a24 (deployed) | .262 (.160) | .343 (.191) | .953 | .089 | .054 |
| C0 a25 | .246 (.159) | .411 (.213) | .938 | .113 | .061 |
| C0 a26 | .243 (.174) | .693 (.576) | .625 | .515 | .059 |
| **C1 a24** | **.019** (.084) | **.017** (.067) | .953 | .021 | .017 |
| **C1 a26** | **.037** (.106) | **.045** (.089) | .953 | .084 | .017 |
| **C1 a29 (final)** | **.014** (.097) | **.020** (.084) | .953 | .024 | .029 |
| bootstrap | 0 | 0 | .953 | .031 | — |

- **C1 stays within about 0.02–0.04 nats of the bootstrap across 1,800 updates**, 13–20× closer than C0 at a comparable attempt. That is about the drift of a single C0 tranche (C0's `kl_to_round_start` ≈ .01–.03 per tranche). Its policy entropy never grows (§3).
- **Behaviour is not literally frozen.** On screening iid_f0, 0 of 256 greedy action sequences equal the bootstrap's (16 of 256 on iid_f2, 86 of 256 on foreign4). The main change is solver-call budgeting:
  - greedy calls per episode rise from 3.6 to 5.7 on iid_f0;
  - the bootstrap's calls are 94% budget 128; C1 splits them roughly evenly across budgets 16, 128 and 1,024.
  - This is the source of the +22% work per success and the −.009 utility.
- **Successes are unchanged world by world.** From bootstrap to C1 there are 240/244/246/239 success→success, 14/10/8/16 failure→failure, and ≤ 1 world changing in each direction per condition.
- **Reading.** In policy terms C1 is indistinguishable from "the anchor held the policy near the bootstrap". Its deviations are small and cost-increasing, not improvements.
  - S1 is met trivially by the unmodified bootstrap: success .949 and no-progress .037, which is what the bootstrap row *is*.
  - So S1 holding shows that the anchor **prevents the C0 collapse over this horizon**. It does not show that RL can run safely and then improve.
  - The report should state preservation as "collapse prevented, at the cost of near-zero movement" and pair it with the S2 failure. It should not suggest RL is now doing useful learning.

**Other risks.**
1. **C1 shows early loop signs that S1 does not read.**
   - The C1 greedy step-to-cap rate is .051 vs the bootstrap's .023: C1's few failures run to the cap more often.
   - Under sampling, C1's no-progress rate is .031 vs .013 and its step-to-cap rate .066 vs .014, 2.4× and 4.7× the bootstrap's.
   - On probe worlds, C1 a26's greedy no-progress is .084, then .024 at a29. The measure is not monotone and is noisy at 64 worlds.
   - These are the C0 precursor signatures, at much lower levels. A longer horizon could still degrade.
2. **Single lineage (r2), single anchor weight (0.3), single seed.**
   - The anchor also suppresses the entropy growth seen in C0 (.82 → 1.60). "Bootstrap KL anchor" is not separated from "entropy control", or from "a stronger trust region of any kind".
   - That is acceptable for a screening result, but it belongs in P2b's design, for example an entropy-matched control.
3. **The screening is non-sealed and was run once.** The IID group has 512 worlds, and S1 is an exact tie (486 of 512 successes for both). The .02 margin is about 10 worlds. The no-progress clause holds with room to spare: C1 is .005 below the bootstrap, and the clause allows .02 above it.
4. **The C0 deployed checkpoint is not "preserved" on new worlds either.**
   - Its greedy IID success is .900, 0.049 below the bootstrap. That is below the rule's development margin, which passed at .945 on 128 development worlds.
   - Its no-progress rate is .177 (bootstrap .037) and its step-to-cap rate .100.
   - The registered 0.02 development margin on 128 worlds is too coarse to protect deployment. This strengthens S3 ("deployment rule, not stable learning").
5. **Utility definition.** The protocol text says "verified outcomes per total cost". The implementation, like P1, uses `outcome.utility` = success − cost per episode. The auditor used the same definition. The protocol wording should be aligned with it.

## 8. Claims to weaken or add

- **S1 wording.** Say "the anchor prevented the collapse over the P1 horizon and kept the policy within ~0.02 nats of the bootstrap", not "RL preserves competence". Note that the bootstrap alone meets S1.
- **Part D reading 4.** Drop "clip fraction 1.00 ⇒ critic direction". Qualify "30–80×" as bootstrap-only (value head unfit by imitation). Note that the ratio is similar in the stable r1 and sometimes below 1 just before the collapse. It does not localize the collapse.
- **Add** C1's elevated sampled-mode no-progress and step-to-cap rates as an open risk for P2b.
- **Add** that Part D's no-progress definition differs from the registered screening definition (bootstrap .068 vs .031 on the same worlds), so its rates are not comparable to the screening.
- **Everything else stands:** S0 (bit-exact, deterministic), S2 false, S3, the deployment decisions, and integrity.
