# Extended-07 P2 screen, P2 frontier and P3 consumer screen: independent adversarial audit

**What was audited:**
- the registry entries P2-SCREEN (with both addenda), P2-CONFIRM and P3-SCREEN;
- decisions.md from 17:33Z to 21:28Z;
- p2-frontier.md;
- the frontier-tool edits made by root.

The audit is based on `campaign/extended-07` b87aba96.

**Out of scope:** P3-REPLICATE was not touched. No `e07-p3r-*` output was listed or read.

**Method:**
- I wrote my own re-derivation, `review/p23_audit_rederive.py` (sha256 `866c60d7…`). It does not import `campaign07_frontier` or `campaign07_diagscore`. It is written from the registered definitions and reads the raw per-decision logs read-only:
  - `e07-p3-eval-{mix,exact}-s4?/diag-v1-{cf-SCE,B}-CONS3-*`;
  - `e07-p2-eval-s4?/…`;
  - `e07-p2-evalX-s4?/…`.
- It uses its own bootstrap draws: 20,000, RNG seed 424242. These differ from the campaign's seed-7 draws, so CI agreement is a genuine Monte Carlo check.
- **Output:** `research/results/campaign-07/p23-audit/audit-p23.json`, sha256 `d00afc0a…`. It is identical to the remote copy.
- **Other evidence used:**
  - `train_meta.json` of every 2×2, SEP and P3 consumer checkpoint (flattened and diffed across arms per seed);
  - `launch.json` and `occupancy.json` receipts;
  - GitHub push events, which give an independent clock for the git commits.

## Verdict summary

| # | item | verdict |
|---|---|---|
| 1a | P2 reference = S1R1 (pool .819 > S1R0 .811) | **CONFIRMED** (exact) |
| 1b | Every deployable candidate's flip gain < 0; no candidate qualifies | **CONFIRMED** (exact). Robust to the reference: against S1R0 the best candidate gains +.000. |
| 1c | Addenda 1 and 2 registered before any P2 eval output | **CONFIRMED**: pushed 17:40:18Z and ≤ 17:51:42Z; first P2 eval started 19:05:54Z. |
| 2 | 2×2 integrity: common init, split clips, keyed RNG, one code path | **CONFIRMED**. The snapshot difference (c7d22aff vs 07b7132a) is immaterial. |
| 3a | Primary J_sub +.046 [+.011, +.082] 5/5; all_x_q2notH +.011; q2_after_notH +.119; after_probe_failed −.117 | **CONFIRMED** (exact points; CIs agree to ±.003) |
| 3b | P3 metric fixed before any P3 eval output | **CONFIRMED**: 9200f3ca pushed 20:15:33Z, 0ff79c41 at 20:15:50Z; first P3 eval started 20:31:01Z. |
| 3c | Root's frontier-tool edit (refs, stratum, seed parsing) | **CONFIRMED** correct and made before data; it ran unchanged from source-0ff79c41 (sha verified). |
| 4a | mlp and bil parameter-matched, identically initialized, equal budget | **CONFIRMED** |
| 4b | bil pool gain .830 vs .812 | **Numbers CONFIRMED; not resolved and not composition-related** (see §4) |
| 5a | "No context-general composition gain" | **CONFIRMED** |
| 5b | "Interaction consumer shifts discrimination toward belief-update contexts" | **CORRECTION NEEDED**: it holds only under the mix contract, and it is partly a weak-baseline effect (see §5.1). |
| 5c | Per-type reading "improves in query-2 contexts, worsens after failed probes/timeouts" | **CORRECTION NEEDED**: only q2_after_notH and after_probe_failed are resolved (see §5.2). |
| 5d | "The P2 flip-gain rule was passable by bias alone" | **CORRECTION NEEDED** (minor): the rule was not bias-proof, but a pure bias shift lands at the guard, not clearly inside it (see §5.3). |
| 5e | Registry P2-SCREEN still carries the preliminary reading ("shift action-change propensity, not interaction discrimination") | **CORRECTION NEEDED**: the frontier analysis superseded it (see §5.4). |

## 1. P2 selection (re-derived from raw logs)

Scope: b6c SCE octets (320) and b6c pool (B protocol), seeds 40–44. EvalX lineage 35+i is paired with seed 40+i.

| model @own | pool | flip | near-miss | invariance | flip gain vs S1R1 | qualifies |
|---|---|---|---|---|---|---|
| S1R1-bank (reference) | .819 | .469 | .695 | .760 | – | – |
| S1R0-bank | .811 | .453 | .708 | .751 | −.016 | no |
| S0R0-bank | .808 | .372 | .847 | .767 | −.097 | no |
| S0R1-bank | .805 | .447 | .817 | .754 | −.022 | no |
| SEP-bank | .803 | .416 | .814 | .730 | −.053 | no |
| CONS-exact-pred | .821 | .453 | .776 | .743 | −.016 | no |
| CONS-mix-pred | .819 | .445 | .786 | .741 | −.023 | no |
| CONS-oof-pred | .821 | .414 | .814 | .759 | −.055 | no |
| CONS-exact-predLRN | .817 | .439 | .797 | .737 | −.030 | no |
| CONS-mix-predLRN | .813 | .444 | .803 | .735 | −.025 | no |
| CONS-oof-predLRN | .818 | .412 | .820 | .755 | −.056 | no |

- Every value matches the registry and score.json to 3 decimals.
- Every candidate passes the pool floor. Every candidate fails on flip gain, which is negative everywhere, so "no candidate qualifies" is exact.
- If S1R0 had been the reference, the best candidate (CONS-exact-pred) would gain +.000. The conclusion does not depend on the reference choice.
- The φ-condition figures in the registry match score.json `phi_readout`:
  - S0R1 own/zero flip .447/.441;
  - S1R1 .469/.475;
  - CONS-exact exact/zero flip .456/.445, pool .826/.807, invariance .750/.685.

**Registration order.** Timestamps are from GitHub push events (an independent clock) and the job receipts:
- The P2-SCREEN and P2-CONFIRM rule (c7d22aff) was pushed ≤ 17:34:53Z. The first 2×2 training started at 17:35:37Z.
- Addendum_1 (SEP, evalX; 07b7132a) was pushed at 17:40:18Z. The first SEP training started at 17:41:19Z, from that snapshot.
- Addendum_2 (φ conditions; 21ebb31f) was on origin by 17:51:42Z, the next push event, whose `before` is 21ebb31f.
- The first P2 eval (e07-p2-eval-s40) started at 19:05:54Z, from 21ab15a6, which contains both addenda.
- **Verdict:** both addenda came before any P2 eval output. **CONFIRMED.**

## 2. 2×2 integrity

I flattened `train_meta.json` and diffed it across S0R0/S1R0/S0R1/S1R1/SEP for each seed 40–44. Apart from CPU times, the only keys that differ among the four 2×2 arms are `fuse.shape`, `fuse.read` and `fuse.arm`. Everything else is identical:
- `p2.init_sha256` = `init_sha256_constructed`, the same per seed (e.g. s40 `71b28d7a…`);
- `params` 168,636;
- `clip_mode` split;
- `action_rng` counter;
- `init_from` = `e07-p2-init-s4i/init.pt`;
- `bank.sha256` `3914ecaa…`;
- `sampling_rng` 7540+i;
- `decisions_seen` (identical per seed, so the bank batch streams are identical);
- lr, batch and updates (4,000).

The launch commands differ only in `--shape/--read` or `--factor-mode sep`.

- **SEP:**
  - It differs as specified: `factor_mode sep`, 293,564 params.
  - Its init hash differs because the full state includes the extra predictor encoder. Its shared tensors come from the same `init.pt`, and it uses the same bank stream.
- **Snapshots:**
  - The 2×2 ran at c7d22aff and SEP at 07b7132a.
  - The c7d22aff→07b7132a diff of `campaign04_probeworld_train.py` is additive for `sep`.
  - The one change touching the 2×2 path, `AUX_PREFIXES` for optimizer group X, adds prefixes (`pinp.`, `pgru.`, `ptrunk.`) that do not exist in `sr` models. For the 2×2 it is a no-op.
  - All five arms were evaluated from the same snapshot, 21ab15a6.
  - **No snapshot difference matters.**
- **Minor naming detail:** the R0 arms' constant φ is zeros (`phi_const_mode zeros`, non-persistent buffer). This is as documented.

## 3. P3 frontier metric (re-derived)

### 3.1 Numbers

Contrast: bil-mix-pred@own − mlp-mix-pred@own, SCE, 320 octets. My draws use seed 424242; campaign values are in brackets where they differ.

| stratum | ΔJ_sub | ΔH_sub | ΔFA_sub | seeds + |
|---|---|---|---|---|
| all | **+.046 [+.011, +.082]** | +.003 | −.042 | 5/5 |
| all_x_q2notH | **+.011 [−.026, +.047]** (campaign [−.026, +.048]) | −.032 | −.043 | 4/5 |
| first | +.050 [−.031, +.124] | −.056 | −.106 | 4/5 |
| after_probe_failed | **−.117 [−.213, −.038]** | −.132 | −.016 | 0/5 |
| after_b1_timeout | −.082 [−.195, +.029] | −.110 | −.028 | 1/5 |
| q2_after_H | +.079 [−.028, +.175] | +.104 | +.026 | 4/5 |
| q2_after_notH | **+.119 [+.007, +.219]** | +.084 | −.035 | **4/5** |

The co-primary and guards also reproduce:
- ΔBA_nm +.032 [−.024, +.085];
- ΔFA_sub −.042;
- Δnear-miss +.031;
- pool .830 vs .812.

The screen-rule outcome reproduces:
- all non-stratum conditions pass;
- all_x_q2notH is .011 < .02, so the stratum condition fails;
- therefore the screen rule is not met. **CONFIRMED.**

Support, summed over the 320 octets for one seed:
- sub pairs: 1,283 optimum-changing and 6,046 unchanged;
- q2_after_notH: 390 / 663, which is 30% of all optimum-changing pairs;
- after_probe_failed: 171 / 1,053.

### 3.2 Timing and tool edits

- 9200f3ca fixed the metric, the rule and the tool edits. It was pushed at 20:15:33Z; 0ff79c41, which only changed wording, at 20:15:50Z.
- The P3 trainings started at 20:09:20Z. They do not depend on the metric, and the registration said so beforehand.
- The first P3 eval (e07-p3-eval-exact-s40) started at 20:31:01Z. Its output files are later.
- The P3 score (diagscore) started at 20:57Z; the frontier at 21:16Z (failed at startup) and 21:17Z (frontier2).
- **Verdict:** the metric was fixed before any P3 eval output. **CONFIRMED.**

The tool edits (diff 1b3e0aed→9200f3ca):
- **`--refs`:** replaces the hard-coded P2 references. Correct.
- **all_x_q2notH:** accumulated for every decision type except q2_after_notH, with the same counters as `all`. My independent stratum gives identical point values. Correct.
- **Seed parsing:** `-s([45])(\d)$` → the last digit. Correct for s40–44 and s50–54.
- **Latent risk:** a single run that mixes s4x and s5x directories would silently pair seed 4i with 5i by index. This is irrelevant to the audited runs.
- **Provenance:** `campaign07_frontier.py` is byte-identical at 9200f3ca, 0ff79c41 and remote `source-0ff79c41` (sha `61e3ecba…`), so the frontier2 job ran the registered code.

## 4. P3 matching

- **Architecture:**
  - Per seed, mlp, bil and gate share `cons_common_sha256`, and it equals the plain lin consumer's `p2.init_sha256` (e.g. s40 `ee6fdfb1…`).
  - The added output layers (mlp `l2`, bil `out` and `gate`) are zero-initialized in `ConsHead.__init__`. So every variant starts as exactly the lin function.
  - Active parameters: mlp 160,082 vs bil 160,194 (added: 14,408 vs 14,520).
- **Training budget and inputs:**
  - Identical bank sha, `sampling_rng` and `decisions_seen` (same batch stream), 4,000 updates, batch 64, lr default.
  - Same OOF file, so mix noise has the same σ. The noise is keyed by (noise_seed, seed, update, slot), so it is identical across architectures.
  - Same full-data predictor at eval.
  - The launch commands differ only in `--cons-arch`.
- **Snapshots:** mlp/bil/gate trained at b369fa9d; the reused lin consumers at c7d22aff. The diff is additive (ConsHead, b6d). The primary pair shares a snapshot.
- **Verdict:** **CONFIRMED.**

**Pool gain (.830 vs .812).** It does not change the verdict, but it should not be read as a benefit of the interaction structure:
- It is **not resolved**: +.018 [−.013, +.041] (P3 score, two-level), 4/5 seeds. Seed 44 reverses (−.039).
- Gap regret is unchanged (+.002).
- On common π* histories (protocol A) it is +.003 [−.021, +.023]. The pool difference therefore arises on each model's own free-running state distribution, not in per-decision quality on shared histories.
- The gate variant, which has no interaction terms, reaches .827, and lin reaches .819. mlp is simply the lowest-pool variant.
- Under the exact contract bil − mlp pool is also +.019, while ΔJ_sub there is −.014.
- **Conclusion:** the pool gain is unrelated to composition discrimination.

## 5. Interpretation

### 5.1 "Interaction consumer shifts discrimination toward belief-update contexts" holds only under the mix contract

These are secondary arms, all registered as "reported" (the exact contract, gate and lin). They are not in the P3-SCREEN result text.

| contrast (predicted inputs, @own) | ΔJ_sub all | all_x_q2notH | q2_after_notH | seeds + |
|---|---|---|---|---|
| bil − mlp, **mix** contract (primary) | +.046 [+.011, +.082] | +.011 | +.119 | 5/5 |
| bil − mlp, **exact** contract | **−.014 [−.070, +.046]** | −.018 | **−.010** | 1/5 |
| gate − mlp, mix | −.023 [−.067, +.026] | −.015 | −.008 | 1/5 |
| bil − gate, mix | +.068 [+.033, +.106] | +.026 | +.127 | 5/5 |
| mlp − lin, mix | +.014 [−.014, +.042] | −.001 | +.039 | 4/5 |

Arm levels (J_sub all / q2_after_notH):

| arm | mix | exact |
|---|---|---|
| mlp | .348 / .047 | .392 / .140 |
| bil | .394 / .166 | .378 / .130 |

What this shows:
- **Under mix, the gain is attributable to the bilinear products.** Gating alone gives nothing (gate − mlp −.023), and bil − gate is +.068, 5/5. The mlp baseline is not weaker than lin.
- **The effect is contract-specific.** Under the exact training contract, bil does not beat mlp pooled or in q2_after_notH.
- **The mlp trained on the exact contract matches bil-mix** on pooled J_sub (.392 vs .394) and comes close in q2_after_notH (.140 vs .166).
- **So the q2_after_notH advantage in the primary contrast is largely the mix-trained mlp's failure there** (.047), not a capability unique to explicit interaction structure.
- **Mechanism:**
  - The pooled gain is false-alarm suppression: ΔFA −.042, ΔH +.003.
  - Outside q2_after_notH, bil changes its action less overall (ΔH −.032, ΔFA −.043). That is a more conservative criterion, not better discrimination.

### 5.2 Per-type claims

Resolved (CI excludes 0):
- q2_after_notH +.119 [+.007, +.219]. This is **4/5 seeds, not 5/5**; seed 42 is −.071. The lower bound is .007.
- after_probe_failed −.117 (0/5).

Not resolved:
- q2_after_H +.079 [−.028, +.175];
- after_b1_timeout −.082 [−.195, +.029].

These five per-type contrasts are exploratory, with no multiplicity control. The registered stratum rule only uses all_x_q2notH.

### 5.3 "The registered P2 flip-gain rule was passable by bias alone"

p2-frontier.md §6 says the rule is "not bias-proof". Along the fitted trade-off (slope −1.77), a pure criterion shift that buys the +.03 flip minimum costs about −.053 near-miss. That sits *at* the −.05 guard (marginally outside it), not clearly inside. The substantive point stands:
- the rule contained no discrimination requirement;
- a bias shift could reach the boundary;
- a small discrimination gain plus bias could pass it.

"Passable by bias alone" slightly overstates this.

### 5.4 P2-SCREEN registry "Preliminary reading"

The line "interventions shift action-change propensity, not interaction discrimination" was superseded by p2-frontier.md. The frontier analysis found:
- composition discrimination J_sub varies between models: shaping raises it (S1R1 − S0R0 +.074 [+.010, +.143]);
- only the near-miss axis is bias-dominated.

decisions.md (20:15Z) records the correction, but the registry does not.

## 6. Exact replacement wording

**R1. registry.json P3-SCREEN `result`.** Replace the sentence starting "Reading: the explicit-interaction consumer improves discrimination…" through "…over a matched ordinary MLP consumer." with:

> "Reading (mix contract, the registered primary): the bil − mlp J_sub gain is false-alarm suppression (ΔFA −.042, ΔH +.003), carried by the bilinear products (bil − gate +.068 [+.033,+.106] 5/5; gate − mlp −.023), and is resolved only in q2_after_notH (+.119 [+.007,+.220], 4/5 seeds; positive) and after_probe_failed (−.117, 0/5; negative); q2_after_H (+.079) and after_b1_timeout (−.082) CIs include 0 (per-type contrasts exploratory). Outside q2_after_notH bil is more conservative (ΔH −.032, ΔFA −.043), not more discriminating. The pattern is contract-specific: under the exact training contract bil − mlp is −.014 [−.070,+.046] (1/5; q2_after_notH −.010), and mlp-exact-pred (J_sub .392, q2_after_notH .140) matches bil-mix-pred (.394/.166) — so the primary contrast's q2_after_notH advantage largely reflects the mix-trained mlp's weakness there (.047). No context-general composition gain over a matched ordinary MLP consumer under either contract. Pool +.018 [−.013,+.041] (4/5) is unresolved, absent on common pi* histories (+.003), shared by gate (.827), and present also under the exact contract where J_sub is not: not a composition effect."

**R2. registry.json P2-SCREEN `result`.** Replace "Preliminary reading: interventions shift action-change propensity, not interaction discrimination (analysis commissioned)." with:

> "Preliminary reading superseded by p2-frontier.md: on the near-miss axis the differences are bias (flip vs near-miss r = −.82, BA_nm .51–.60); composition discrimination J_sub does vary (S1R1 − S0R0 +.074 [+.010,+.143]) and no deployable candidate reaches S1R1 on it (−.037 to −.083)."

**R3. decisions.md 2026-09-27T21:28Z, bullet "The interaction consumer shifts discrimination toward the belief-update contexts. It is not a general composition gain."** Replace with:

> "Under the mix contract only, the interaction consumer shifts discrimination toward q2_after_notH (+.119, 4/5 seeds) and away from after_probe_failed (−.117, 0/5); other per-type differences are unresolved. Under the exact contract bil does not beat mlp (−.014, 1/5), and the exact-contract mlp matches bil-mix, so the q2_after_notH advantage mostly reflects the mix-trained mlp's weakness there. No general composition gain (independent audit p23)."

**R4. decisions.md 2026-09-27T20:15Z, bullet "The registered P2 flip-gain rule was passable by bias alone."** Replace with:

> "The registered P2 flip-gain rule was not bias-proof: along the observed trade-off a pure criterion shift reaching the +.03 flip minimum costs ≈ .053 near-miss, at the −.05 guard, and the rule had no discrimination requirement."

**R5. registry.json P3-SCREEN `result`**, the phrase "q2_after_notH +.119 [+.007,+.220]". Change it to "q2_after_notH +.119 [+.007,+.220] (4/5 seeds)".

**For P3-REPLICATE (a recommendation only; nothing was read):** its descriptive report should also state the exact-contract bil − mlp contrast. §5.1 shows the screen pattern does not hold under the exact contract.

## 7. Compute and provenance

- **Remote, metered:**
  - `e07-audit-p23-recompute` on pro6000: 34.4 core-s, 32 s wall, 1 thread, exit 0. Receipt `results/dev/e07-audit-p23-recompute-20260927T213311-1277861-process`.
  - Output only under `results/dev/e07-audit-p23/`.
- **Remote, unmetered read-only inspection:** ls, receipt/meta JSON reads and sha256. Estimated < 10 core-s.
- **Local:** JSON summarizing only, < 10 core-s.
- **Totals:** ≈ 55 core-s. No GB10 use. Nothing was killed or detached, nothing was written under campaign06, and no P3-REPLICATE output was read.
- **Reproduce:** `python review/p23_audit_rederive.py OUT.json` on pro6000, which reads `~/structured-latent-dynamics-campaign07/results/`.
