# Extended-06 Track A: independent audit (A-HS, A-SEL, A-CF-SMALL)

**Auditor:** independent agent, 2026-09-27. **Base:** `campaign/extended-06` @ 305324eb. **Branch:** `campaign/e06-a-audit`.

**Inputs read:**
- campaign.md, design.md (v2 items 7–11 and 13), registry.json (A-HS, A-SEL, A-CF-SMALL), decisions.md, portworld.md, seed-ranges.json;
- `src/tensegra/campaign06_portworld.py`, `research/tools/campaign06_portfolio.py`;
- `research/results/campaign-06/a-hs/*` and `research/results/campaign-06/a-cfsmall/*`.

**Raw data pulled (read-only) from pro6000:** the train0–2, select and confirm `records.jsonl.gz`, the three `fit.pkl`, and `e06-pw-acf-score`.
- Every hash matches fit.json and score.json: fit.pkl e0ff1ace…, 1bff3f53…, 92460494…; confirm records b90b21ba…, 7251f685…; per_instance.npz ec77ce6a… (the same as the staged copy).

**What was recomputed and how:**
- All numbers below were recomputed from `per_instance.npz`, or from the raw records plus the frozen fits.
- CIs use two bootstraps:
  - the tool's own `cluster_boot`;
  - an independently written episode-resampling bootstrap (2,000 draws, percentile).
- The two agree to ±.0002 on every bound.

## Summary of verdicts

| # | Check | Verdict |
|---|---|---|
| 1 | Recompute GA-1/GA-2 and A-CF-SMALL primaries and CIs | **CONFIRMED** (exact) |
| 2 | Leakage, population disjointness, seed ranges, winner's curse in simple-family choice | **CONFIRMED** (one note) |
| 3 | Charging consistency and sensitivity | **CORRECTION NEEDED**: the A-CF-SMALL pass depends on the charge convention |
| 4 | Oracle certification and uncertified handling | **CONFIRMED** |
| 5a | Strong baseline matches design v2 item 8 | **CONFIRMED** (one minor deviation, disclosed below) |
| 5b | The simple baselines are genuinely strong | **CORRECTION NEEDED**: a modestly tuned tree removes the 3/3 pass |
| 6a | Interpretation | **CORRECTION NEEDED**: "simple captures most" is supported, even understated; "learned margin real" is overstated; the "~80%" figure is wrong |
| 6b | No generator revision after the GA-1 failure | **CONFIRMED** consistent with the design (one registry-wording note) |

**Net.** The registered A-HS gate result (GA-1 fails, GA-2 passes) and the Track A closure are sound, and they are robust to every sensitivity analysis below.

The A-CF-SMALL "3/3 PASS" reproduces exactly as registered, but it is **not robust**:
- it fails at 2× selector charge;
- it reverses under a GBT inference charge counted by the same per-output convention as the logistic;
- it drops to 2/3 against a depth-≤3 tree whose min_leaf/bins are tuned on the select population.

The learned one-shot selector is statistically indistinguishable from a depth-3 tree fit in-sample on the confirm population. The finding should read as "no practically or robustly detectable learned-selector margin". It should not read as "a real but negligible margin".

---

## 1. Recomputation: CONFIRMED

**A-HS** (dev_gate: 640 episodes, 305,000,000–305,000,639; N = 2,891).
- Folds are constant within episodes. Fold sizes are 587/587/580/569/568.
- `best_simple_inner` equals the declared per-fold family (d2, d2, d3, d3, d3) row for row.

| quantity | reported | recomputed (tool boot / independent boot) |
|---|---|---|
| GA-1 one-shot | +.0057 [.0030, .0087] | +.0057 [.0030, .0087] / [.0031, .0087] |
| GA-1 sequential | +.0058 [.0032, .0088] | +.0058 [.0032, .0088] / [.0032, .0089] |
| GA-2 one-shot | +.0403 [.0345, .0463] | +.0403 [.0345, .0463] / [.0347, .0465] |
| GA-2 sequential | +.0404 [.0348, .0465] | +.0404 / [.0349, .0467] |
| best simple − best single | .0346 [.0279, .0418] | .0346 [.0280, .0413] |
| oracle − best simple | .0292 [.0255, .0331] | .0292 [.0255, .0332] |

- Every policy mean in headroom.md matches the npz column means to 4 decimals.
- GA-1 fails by a wide margin: the point estimate is .0057 against .02.
- This does not depend on charging. Removing all selector charges moves GA-1 by at most ≈ +.002, since the learned−tree charge differential is 897–1,797 units × E[c] ≈ 1.07e-6.

**A-CF-SMALL** (confirm: 640 episodes, 330,000,000–330,000,639; N = 2,852).
- Every per_instance.npz column checked was regenerated bit-for-bit (`np.allclose`) from the frozen `fit.pkl` artifacts applied to the raw confirm records with the tool's `cf_apply`. Columns checked: learned one-shot, learned sequential, tree_d2, tree_d3 and best simple, for all 3 lineages.

| lineage | one-shot − best simple (reported) | recomputed (independent boot) | sequential − best simple | GA-2 analogue (one-shot) |
|---|---|---|---|---|
| 0 | +.0032 [.0006, .0067] | +.0032 [.0006, .0066] | +.0034 [.0005, .0069] | +.0351 [.0294, .0406] |
| 1 | +.0025 [.0005, .0046] | +.0025 [.0005, .0047] | +.0032 [.0011, .0053] | +.0349 [.0293, .0406] |
| 2 | +.0043 [.0014, .0080] | +.0043 [.0012, .0078] | +.0052 [.0023, .0087] | +.0352 [.0295, .0408] |
| paired mean | +.0033 [.0016, .0055] | +.0033 [.0015, .0056] | +.0039 | |

- Best simple is tree_d3 in all three lineages; it matches its column exactly. The hidden-state oracle is .9826, which matches.

**Distributional note (not in the reports).** The margin is heavy-tailed.
- On the lineage-averaged paired difference, **353 of 640 episodes favour the tree and 287 favour the learned selector**.
- The top 5% of episodes (32) contribute 110% of the net gain.
- In A-HS, the learned one-shot is below best-simple on 66% of instances. Largely this is the fixed ≈ 900-unit charge differential applied whenever both pick an equivalent arm.
- So the learned selector is slightly worse on the typical instance, and ahead on average only through a few large wins.

## 2. Leakage and populations: CONFIRMED

**Seed ranges.** These are the seeds actually present in the records and npz files.

| population | seeds | registered sub-range |
|---|---|---|
| dev_gate | 305.000M + 0..639 | [305M, 310M) |
| train0 / train1 / train2 | 310.000M, 311.000M, 312.000M + 0..639 | [310M, 320M) |
| select | 320.000M + 0..639 | [320M, 330M) |
| confirm | 330.000M + 0..639 | [330M, 340M) |

- All populations are pairwise disjoint and inside their registered sub-ranges. They are disjoint from dev_builder [300M, 305M): builder dev used 300.004M+, the ACF smoke 303M+, and the GA smoke 304.99M.
- `seed-ranges.json` matches `pw.SEED_RANGES`.
- `cf_populations` and `cmd_cf_score` assert the containment and disjointness at run time.
- A remote search of every job command line found `--lo 304990000` as the only explicit `--lo`. No dev or smoke job touched 330M.

**Order of operations.**
1. A-CF-SMALL was registered in de28b9f7 (00:10:58 PDT = 07:10Z). The registry text of `primary` is unchanged between de28b9f7 and 305324eb.
2. Tooling followed at 07:25Z, and train/select records were written at 07:26–07:28Z.
3. The fits were frozen at 07:38Z.
4. The confirm records were created at 07:40Z. Scoring used `--expect-sha`.

**Selection.**
- **A-HS:** the best simple family is chosen on 4 inner episode folds of each outer training set. The GBT config is chosen on 2 inner folds. Everything is scored out of fold.
- **A-CF-SMALL:** every family is fit on train L only. The family, logistic λ and GBT config are chosen on the select population only. The confirm population is scored once. The menu top-5 and the tuned family constants come from train L.
- There is no winner's curse in the confirm comparison.

**Note.**
- The dev headroom estimate (+.020) came from the same dev_builder data on which the generator was iterated v1 → v2 → v3 specifically to raise headroom (portworld.md §3: .0016 → .0084 → .020).
- The drop to .006 on fresh gate instances is consistent with generator-level selection on dev.
- The registered protocol handled this correctly: the gate used fresh seeds. The dev figure should not be quoted as evidence of headroom.

## 3. Charging: CORRECTION NEEDED

**What is charged (verified in code).**

| family | charge |
|---|---|
| trees | c × (feature_work + depth) |
| logistic | c × (feature_work + F × A), where F ≈ 42 features × A ≈ 60 menu columns ≈ 2,520 multiply-adds |
| learned one-shot | c × (feature_work + 3 × rounds × depth) = +900 units |
| learned sequential | the chosen candidate's charge + 300 |
| single / cascade / strong / hand ("pick") | 0 extra; each carries its own charges inside its utility |

- Cascades charge root_work.
- Hand charges feature_work, root_work and the opt probe.
- Strong baseline charges its GR/RU/RUPD/BM stages. It reads only `hidden_frac` / `n_hidden`, an O(G + n) count that is uncharged and negligible.

So trees and logistic are charged, and the pick families carry their own charges, as claimed.

**Two inconsistencies were found.**

1. **The GBT inference charge omits the per-output work that the logistic is charged for.**
   - The logistic pays one unit per feature × output multiply-add.
   - Each learned one-shot inference evaluates 3 multi-output GBTs × 100 trees. Each tree is a depth-3 traversal plus an **A = 45-dimensional leaf-vector accumulate**. The GBT is then combined with prices over 45 arms.
   - Only the 3 × 100 × 3 = 900 comparisons are charged.
   - Counting the accumulates by the logistic's convention gives 3 × 100 × (3 + 45) = 14,400 units, i.e. +13,500 over the registered charge (+14,300 for sequential).
   - The convention is documented in portworld.md §1 ("GBT rounds × depth × models") but is not consistent across families.
2. **Minor, and against the simple family: trees double-charge some menu columns.**
   - Trees pay feature_work on top of hand columns, which already include feature_work.
   - They also pay the cons root bound twice on cons-mode cascade and hand columns.
   - The measured effect is .00006–.00014 per lineage. Correcting it lowers the margin by ≤ .0001: L0 +.0031 [.0005, .0066], L1 +.0025 [.0004, .0045], L2 +.0041 [.0012, .0079].

**Sensitivity.** Per lineage: one-shot − tree_d3 on confirm, episode-clustered 95% CI; P/F = LB > 0.

| charge scenario | L0 | L1 | L2 | pass |
|---|---|---|---|---|
| as registered | +.0032 [+.0006, +.0067] P | +.0025 [+.0005, +.0046] P | +.0043 [+.0014, +.0080] P | 3/3 |
| zero selector charge (all families) | +.0042 [+.0015, +.0076] P | +.0035 [+.0015, +.0055] P | +.0052 [+.0023, +.0089] P | 3/3 |
| 2× selector charge (all families) | +.0022 [−.0004, +.0058] F | +.0016 [−.0005, +.0036] F | +.0033 [+.0004, +.0070] P | **1/3** |
| GBT charged per output (logistic convention) | −.0113 [−.0145, −.0074] F | −.0119 [−.0145, −.0093] F | −.0102 [−.0136, −.0063] F | **0/3 (reversed)** |

- Sequential behaves the same way: 3/3 as registered and at zero charge, 1/3 at 2×, and 0/3 at −.010 to −.012 per output.
- **Break-even:** the mean one-shot margin vanishes if the learned selector's inference costs **2,400–4,000 more work units** than the tree's (L1/L0/L2 = 2,376 / 2,979 / 3,961). The registered differential is 897.

**Required wording.** Add to the registry A-CF-SMALL result or interpretation and to decisions.md (08:03Z):

> "The margin is conditional on the registered selector-charge convention (GBT inference = rounds × depth × models units). It passes 3/3 at zero charge but only 1/3 at 2× selector charge. It reverses (−.010 to −.012, 0/3) if the GBT's 45-output leaf accumulation is charged per output, as the logistic's multiply-adds are. The mean margin vanishes at ≈ 2,400–4,000 extra inference units per instance."

- This does **not** affect A-HS/GA-1. It fails under every scenario, including zero charge: GA-1 would be ≲ .008.

## 4. Oracle: CONFIRMED

- **A-HS:** 2,891 / 2,891 certified. No uncertified-episode lines.
- **A-CF-SMALL confirm:** 2,848 / 2,852 certified (.9986).
  - The 4 uncertified instances are in episodes 330,000,114 and 330,000,300.
  - They are **kept** and scored against the certified UB. Their UB vs incumbent values are 924/793, 988/732, 989/741 and 989/741, and their oracle U is .70–.82.
  - The scale compression is common to all policies and cannot bias the paired differences materially (4/2,852).
- The `uncertified_episodes_legacy` drop path in `cmd_evaluate` (a whole episode dropped on a RuntimeError) fired **0 times** in every population (train0–2, select, confirm, dev_gate).
- The "hidden-state oracle" is the per-instance best of the 45 menu arms. It is a menu ceiling, not the true optimum, and it is correctly labelled non-deployable.

## 5. Baselines

### 5a. Strong baseline vs design v2 item 8: CONFIRMED (one minor deviation)

Code: `strong_baseline`, grid of 540 variants tuned on train.

| design item 8 requirement | implementation | status |
|---|---|---|
| propagation + decomposition always on | RUPD family | yes |
| B&B warm-started from greedy | the RUPD incumbent is the better of the repaired cache and GR | yes |
| budget scaled to the compute price | b ≤ κ/c, read off the trace | yes |
| certified-gap stop | (UB − v)/v < γ·c·b | yes |
| node-count switch | if uncertified at b ≥ b_sw → BM16 and stop | yes |
| always verify/repair the cache first | RU always runs | yes |
| inspect iff hidden_frac × L > observation price | θ · hidden_frac · L > **o × n_hidden**, with θ ∈ {.25, 1, 4} tuned | **deviation** |

- **The deviation:** the right-hand side is the *total* inspection cost, not the per-parameter price. It is arguably more sensible, but it is not the literal rule and is undisclosed.
- **Suggested disclosure (portworld.md):** "the inspect rule compares θ·hidden_frac·L with the total observation cost o·n_hidden (θ tuned), not with o."
- **A structural weakness:** the strong baseline always pays GR + RU + RUPD preprocessing and has no "commit the verified cache" shortcut. Yet RV (verify-and-commit the cache) is the hindsight-best method on 48% of instances. This is why it sits at .944, below the trees at .956.
- It is not the binding baseline: its top-5 variants are in the tree menu, alongside the raw RV arms. The conclusion therefore rests on the trees, which is appropriate.

### 5b. Are the simple baselines genuinely strong? CORRECTION NEEDED

The registered trees use a **fixed** min_leaf = max(20, .03·N_train) = 87 and 16 bins, with no hyperparameter tuning. Only the family is chosen on select. The learned GBT, by contrast, gets its config chosen on select.

**Auditor's post-hoc sensitivity analysis (a valid protocol, but not registered):**
- Setup:
  - policy trees on train L's menu over the grid depth ∈ {2, 3, 4} × min_leaf ∈ {20, 40, 87, 150} × bins ∈ {16, 32};
  - the configuration is chosen on the **select** population only;
  - it is scored once on confirm with the registered charges.
- The grid was fixed before looking at any tuned-tree confirm number.

| lineage | registered tree_d3 (confirm) | tuned depth ≤ 3 (chosen cfg) | one-shot − tuned d≤3 | tuned depth ≤ 4 | one-shot − tuned d≤4 |
|---|---|---|---|---|---|
| 0 | .9558 | .9576 (d3, ml150, B16) | +.0014 [−.0005, +.0036] **F** | .9578 | +.0012 [−.0007, +.0033] F |
| 1 | .9563 | .9559 (d3, ml20, B16) | +.0030 [+.0007, +.0053] P | .9582 | +.0006 [−.0015, +.0028] F |
| 2 | .9549 | .9568 (d3, ml150, B32) | +.0023 [+.00002, +.0052] P (marginal) | .9568 | +.0023 [+.00002, +.0052] P |

- **In-sample bound.** A depth-3 policy tree fit **on the confirm population itself** scores .9589–.9592 (ml 87/20, 32 bins).
  - The lineage-averaged learned one-shot minus this tree is **+.0001 [−.0009, +.0010]** (ml 87) and **−.0002 [−.0012, +.0008]** (ml 20).
  - A depth-4 in-sample tree beats the learned selector by .002–.003.
- **Reading:** the learned selector's advantage is at most the out-of-sample estimation error of a depth-3 tree, not additional structure. Modest, select-validated tuning of the tree's min_leaf/bins cuts the mean margin from .0033 to ≈ .0022 and turns 3/3 into 2/3 (L0 fails).
- **Required wording:** add to the A-CF-SMALL result:

> "The registered depth-3 tree used a fixed min_leaf (87) and 16 bins. With min_leaf/bins chosen on the select population (audit, post hoc), one-shot − tree is +.0014 [−.0005, +.0036] / +.0030 [+.0007, +.0053] / +.0023 [+.00002, +.0052] (2/3). The learned one-shot equals a depth-3 tree fit in-sample on confirm (−.0002 [−.0012, +.0008])."

## 6. Interpretation and the no-revision decision

### 6a. Interpretation: CORRECTION NEEDED

**"A strong simple portfolio captures most public-information headroom": supported, and if anything understated.**

| quantity | A-HS | A-CF-SMALL |
|---|---|---|
| best simple − best single | .0346 | .0318 (lineage mean) |
| learned − best single | .0403 | .0351 |
| share of the learned estimate captured by the simple portfolio | **86%** | **91%** |

- The in-sample tree matches the learned selector (§5b).
- Caveat: the "public-information optimum" is estimated by one GBT family, so it is a lower bound.

**"~80% of even the hidden-state headroom" (decisions.md 07:10Z) is wrong.**
- The simple portfolio captures (best simple − best single)/(oracle − best single) = .0346/.0637 = **54%** in A-HS, and .0318/.0587 = **54%** on confirm.
- The other 46% is hidden-state information that no public selector here reaches. The learned estimate captures only 63% (A-HS) and 60% (confirm).
- Required replacement:

> "a strong simple portfolio (depth-2/3 trees over public features) captures ~54% of the hidden-state oracle's gain over the best single method, and ~86–91% of the gain that a cross-fitted learned public-information selector achieves."

**"A real but practically negligible learned-selector margin" (registry A-CF-SMALL interpretation, decisions.md 08:03Z) overstates "real".**
- The pass reproduces as registered, but it depends on the charge convention (§3) and on untuned simple-tree hyperparameters (§5b).
- It is carried by ~5% of episodes; the majority of episodes favour the tree.
- "Practically negligible" is correct: it is 1/6 of .02 and 12% of the oracle − best-simple gap (.0033/.0268; recomputed).
- Required replacement:

> "The registered A-CF-SMALL test passes 3/3 (+.0033 [.0016, .0055]), but the margin is not robust. It fails at 2× selector charge (1/3), reverses under per-output GBT inference charging (0/3), falls to 2/3 against a select-tuned depth-3 tree, and equals zero against a depth-3 tree fit in-sample. We read this as no practically or robustly detectable learned-selector advantage over a tuned depth-3 tree portfolio in portworld."

**"Sequential telemetry adds nothing over one-shot": CONFIRMED.**
- A-HS: +.0001.
- Confirm: +.0002 / +.0007 / +.0009 per lineage, and the same qualitative sensitivity.

### 6b. No generator revision after the GA-1 failure: CONFIRMED consistent

- Design v1 says the distribution **"may"** be changed once, for a principled reason. A revision is permitted, not required.
- The stated reason is sound:
  - oracle − best simple is only .0292 [.0255, .0331] on the gate;
  - GA-1 would need public information to capture > 2/3 of the hidden-state gap, while the learned estimate captures ≈ 20% of it (.0057/.0292).
- The generator had already been revised twice pre-gate (v1 → v2 → v3), each time to raise dev headroom. A further revision would compound generator-level selection.
- **Note:** registry A-HS `next.fail` reads "one principled generator revision then close Track A", which is phrased as a plan. The decision entry should say explicitly that this optional branch was **declined**, as a conservative deviation from the registry's stated fail path.
- Suggested addition to A-HS `decision`:

> "(declines the optional one-revision branch in next.fail; design v1 permits but does not require it)."

## Compute and disclosures

**Local:** about **320 core-s** in total.
- Recomputation from the npz: 12 s.
- Population matrices from raw records: 5 × ≈ 21 s.
- Charge and choice analysis: 2 × 33 s.
- Tuned-tree sensitivity: 134 s.

**Remote (pro6000):** read-only file operations only; no computation was run. That is directory listing, `grep` over job logs, and a tar read of ≈ 40 MB of result files, unmetered at a few core-s. No processes were started or killed. The GB10s were not used.

**Confirm population use:** the auditor re-scored the confirm population post hoc for these sensitivity analyses. This is audit use only; no registered result was changed.

**Not re-run:** the unit tests (pytest is not installed locally). The builders ran them metered on the remote.

**Audit scripts:** kept in the session scratchpad and not committed. Each computation is fully described above and uses only the tool's own functions (`cf_matrices`, `cf_apply`, `cf_charge`, `PolicyTree`, `cluster_boot`) plus an independent bootstrap.
