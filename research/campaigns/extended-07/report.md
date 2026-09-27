# Extended-07 report: from learned factors to compositional decisions

**Status: FINAL.** The campaign ran on the pro6000 (WSL2, CPU only) from 2026-09-27T16:02:54Z and closed early on 2026-09-27 (last job receipt 22:13:33Z), well inside its 12 h window (deadline 2026-09-28T04:02:54Z).

**Audits** (corrections applied throughout):
- [review/p1-independent-audit.md](review/p1-independent-audit.md) covers Phase 1.
- [review/p23-independent-audit.md](review/p23-independent-audit.md) covers Phase 2, the frontier analysis and P3.

**Sources:**
- Registry: [registry.json](registry.json). Every entry is labelled registered, adaptive or descriptive.
- Decisions: [decisions.md](decisions.md), with timestamps from `date` or git.
- Design: [design.md](design.md).
- Deliverables: [factor-contract.md](factor-contract.md) with its JSON, [gradient-flow.md](gradient-flow.md), [p1-diagnosis.md](p1-diagnosis.md), [p2-infra.md](p2-infra.md) and [p2-frontier.md](p2-frontier.md).

## Headline

| # | Question (brief §20) | Answer |
|---|---|---|
| 1 | Are the predicted factors accurate on the states where interactions matter? | **No, and error grows with composition.** Per-group normalized error (b6c, historical LRN, own histories): posterior/outcome .45, side/event costs ≈ 1.04 (no better than a constant), strategy costs .32, build/bookkeeping .10. Error rises from none (.32) → single (.36) → pair (.44) → full combination (.53); on flip units full − single is +.22 [+.19, +.25], 5/5 seeds (all units +.17). The predictions carry almost none of the 2nd/3rd-order interaction content of the cost coordinates (predicted-vs-exact slope .01–.21, R² < 0), and the not-H belief update is never learned. Flips are not specifically worse than invariances (+.04, unresolved). A separately trained predictor with its own encoder roughly halves the error on the same scorer (π* histories: G1 .41 → .21, G2 .67 → .34 for SEP; the consumers' full-data predictor .23 / .38) without changing the downstream conclusions below. |
| 2 | Does correcting them improve the SAME frozen policy? | **Barely.** Exact factors substituted into the frozen LRN change 9/640 flip actions (7 rescues, 0 harms; +.011 [.000, .029]), mostly within LRN's training support. The frozen LRN barely reads its factor channel: zeroing it changes .045 of actions vs .24 for SUP, and in-support substitutions (another member's own φ) confirm it (.013 vs .061). |
| 3 | Does auxiliary supervision help through trunk shaping, explicit factor reading, or both? | **Trunk shaping, not reading.** In the genuine S×R 2×2 (common init, separate optimizers and clips, keyed RNG, bank training), shaping raises composition discrimination J_sub (S1R1 − S0R0 +.074 [+.010, +.143], 4/5 seeds; shaping alone, S1R0 − S0R0, +.060 [−.000, +.125], 3/5), through hits rather than fewer false changes. Reading adds little: the φ conditions barely move flips for R1 arms (S1R1 own .469 vs zero .475). The historical SUP's flip advantage survives with its factor inputs zeroed (.514 vs .562); it is training-induced, and under zeroed inputs the remainder is bias. |
| 4 | Can a downstream consumer use exact and predicted factors consistently? | **Yes for invariance, no for composition.** Consumers trained to rely on factors (exact / out-of-fold / noise-mixed contracts) use them as a "don't change here" signal: own vs zero or mean inputs cut false changes by .05–.07 and raise J_sub by .02–.11 (5/5 seeds in 23 of 24 contrasts; historical SUP: .09–.13 and .13), and pool accuracy falls when the input is blanked (.826 → .807). Exact vs predicted inputs cost .01–.06 J_sub (acquisition; P3 mlp .04), essentially all of it in q2_after_notH (without that stratum the cost is within ±.01). No consumer's near-miss discrimination changes with its factor inputs. |
| 5 | Does an intervention improve true flips without becoming indiscriminately sensitive or insensitive? | **No.** On the near-miss axis every model lies on one flip/near-miss trade-off (r = −.82; balanced accuracy .51–.60), so "flip gains" there are bias. On the composition axis (J_sub) no deployable candidate reached the S1R1 reference in the Phase-2 screen (−.037 to −.083). The explicit-interaction consumer's screen gain (below) is mostly false-change suppression and did not replicate. |
| 6 | Did any structured consumer outperform a strong ordinary consumer? | **No.** Setup: the bilinear factor-group consumer vs a parameter-matched MLP consumer (160,194 vs 160,082 active parameters, identical init and training). **Screen (b6c):** J_sub +.046 [+.011, +.082] 5/5, but it fails the registered stratum condition: without q2_after_notH the gain is +.011. The gain is contract-specific (exact contract −.014) and consists of fewer false changes, not more hits. **Fresh configurations (b6d, descriptive):** +.018 [−.018, +.056] 3/5 (mix), −.010 (exact). What replicates, directionally, is that bil is more conservative (fewer false and fewer correct changes: Δbias_sub −.040 [−.090, +.017] mix, −.057 [−.126, +.004] exact, 4/5 seeds lower in each) and, under the mix contract, worse after failed probes (−.146 [−.255, −.049], 0/5; exact contract −.069 [−.193, +.057]). Structural attention was **not** run: its precondition in brief §16 (an established downstream interaction problem that strong ordinary consumers fail and structure could fix) was not met. |
| 7 | What generalized, and what remained supplied? | See [below](#supplied-learned-generalized). |

**Localization (the immediate milestone).** The learned semantic information is **distorted** (acquisition error grows with combination order; the interaction terms and the not-H update are missing). It is **not used** by the historical LRN (redundant channel). Where consumers *are* made to use it, it serves mainly to suppress changes on invariances. **The missing ingredient is discrimination of interaction-driven changes, and no consumer-side intervention tested here supplies it.**

Two further points follow from the evidence:
- **The gap is not in the ontology in any simple sense.** A pairwise-additive reference recovers J .92 on the same octets, so the discrimination is achievable from pairwise structure.
- **Shaping helps, and supervision alone does not deliver it.** Auxiliary shaping of the shared trunk is the only intervention that raised J_sub through more correct changes rather than fewer false ones, and only part of the way (S0R0 .316 → S1R1 .390, against a pairwise-additive reference of .919; shaping alone +.060, unresolved). Factor inputs and the bilinear consumer raised J_sub only by suppressing false changes.

## Phase 0: contracts (static plus mechanical)

- **Factor contract.** The factor target is a single 23-d vector, not "four derived quantities". It has four groups; only 11 coordinates are functionally independent given the configuration. **No hidden-state leak** (three mechanical checks).
  - The strategy-cost coordinates are myopic: 3–27 price units from Q*, and the formula-cheapest strategy is optimal at only 36–38% of decisions.
  - Defects:
    - `candidate_trust` double-counts under U (irrelevant to S+C+E);
    - omitted terms in `cost_probe_first` and the b1/use costs;
    - `p_probe_resolves` ≡ `belief_H`;
    - two coordinates constant on b6/b6c;
    - clip saturation of ≤ 1.1%;
    - `build_value` carries 32% of target variance under the equal-weight loss.
- **Gradient flow.** The historical LRN's policy reads `tanh(W[z; stopgrad(aux(z))])` through a fusion layer shared with the value head. The aux loss updates the encoder, GRU, trunk and aux head. "LRN − RAWF" therefore bundled:
  - trunk shaping;
  - prediction reading;
  - a different fusion init (RNG order);
  - dead (RAWF) vs live factor columns;
  - global clip and Adam coupling (aux ≈ 1.1% of the squared gradient norm);
  - action-RNG drift.

## Phase 1: frozen-checkpoint diagnosis

Summarized under Q1–Q3 above; details in [p1-diagnosis.md](p1-diagnosis.md), audit-corrected.

- **Reproduction.** The runner reproduces all historical extended-06 evaluations bit for bit (4 models × 8 lineages).
- **Flip population.** On b6c, the full combination's flips are 93% pair-inherited. Pairwise-additive Q recovers the full optimum in 91% (SCE) and 83% (UCE) of flip units, and only 9/128 SCE flip units differ from every pair. **The benchmark supports full-vs-single recombination, not irreducible third-order interaction.**
- **UCE octets** (first evaluation, descriptive): no SUP or LRN flip effect (LRN .624, SUP .614, RAWF .603).

## Phase 2: controlled mechanism screen (P2-SCREEN, registered; b6c used as validation, seeds 40–44)

- **Arms:** the S×R 2×2; SEP (a separately encoded predictor whose detached output the policy reads); consumers under the exact, out-of-fold and mix contracts, each with exact, predicted and historical-LRN inputs; φ ∈ {own, exact, zero, mean} for every factor-reading model. All arms were trained on a common history bank (5,376 episodes, training combinations only).
- **Registered selection outcome:**
  - Reference: S1R1 (pool .819, flip .469).
  - Deployable candidates: flip .41–.45, so **none qualifies**.
  - P2-CONFIRM (a repair confirmation) was therefore **not run**.
- **The frontier analysis** ([p2-frontier.md](p2-frontier.md); indices defined before results) showed that the flip-gain rule was not bias-proof. It re-expressed the screen along two axes:

| model (b6c, own inputs) | flip | near-miss | J_sub (composition discrimination) |
|---|---|---|---|
| S0R0 (no shaping, no reading) | .372 | .847 | .316 |
| S1R0 (shaping only) | .453 | .708 | .377 |
| S0R1 (reading only) | .447 | .817 | .349 |
| S1R1 (shaping and reading) | .469 | .695 | .390 (+.074 over S0R0) |
| SEP (separate predictor, read) | .416 | .814 | .307 |
| CONS-exact, predicted inputs | .453 | .776 | .353 |
| historical LRN | .405 | .749 | .307 |
| historical SUP (exact inputs) | .562 | .536 | .433 (+.043 over S1R1; CI includes 0; concentrated in q2_after_notH) |
| pairwise-additive reference (oracle) | .914 | — | .919 |

## P3: explicit interaction consumer (branch B)

- **Screen.**
  - **Metric:** fixed from the frontier analysis before any P3 output. Primary J_sub bil − mlp (mix contract, predicted inputs); co-primary balanced near-miss accuracy; guards on false changes, near-miss and pool accuracy; and a stratum condition excluding q2_after_notH.
  - **Result:** primary +.046 [+.011, +.082] 5/5, co-primary and guards pass. **The stratum condition fails (+.011), so there is no confirmation.**
  - **Audit:** the gain is mix-contract-specific. The exact-contract mlp already matches bil-mix, and the pooled gain is false-change suppression.
- **Descriptive fresh-configuration check (P3-REPLICATE, b6d, registered before data):** see Q6. The pooled gain does not replicate; the conservatism and the after-failed-probe loss do.

## Supplied, learned, generalized

- **Supplied:**
  - probeworld-v3 and the exact labels;
  - the 23-coordinate factor definitions (myopic; not Q*) and their exact values (SUP and exact-input evaluations; privileged);
  - the octet construction;
  - the fusion/consumer architectures, including the bilinear factor-group structure;
  - the history bank's behaviour sources.
- **Learned:**
  - factor predictors (from the trunk, or a separate encoder);
  - policies under bank imitation, with or without auxiliary shaping;
  - consumers under each input contract.
- **Generalized:**
  - Factor predictions transfer only partly to the unseen S+C+E combination: error grows with combination order, and interaction terms are missing.
  - Auxiliary shaping's J_sub gain was measured on the held-out family but not confirmed on fresh configurations.
  - The consumers' "don't change" use of factors (invariance) generalizes to S+C+E.
  - No learned component generalized **interaction-driven change discrimination** to the unseen combination.

## Prospective vs adaptive vs descriptive

- **Registered before data:** P1-DIAG (diagnostic; b6c was already inspected by extended-06); P2-SCREEN, including both addenda (before any P2 eval); P2-CONFIRM (not run by rule); P3-SCREEN (arms before training; metric before any P3 eval; the confirmation-rule wording was cleaned before eval); P3-REPLICATE (descriptive; expectations stated before b6d data).
- **Exploratory:** the P1 Q3/Q4/Q5 analyses; the frontier indices J_nm (added after first tables); all per-decision-type contrasts.
- **Fresh confirmation data:** only b6d (400 configurations plus 320 octets), used once, descriptively. No repair was confirmed; none qualified.

## Costs, integrity and machine state

- **CPU:** 43.7k / 86.4k core-s charged (remote 43.4k metered, of which dev 3.2k, plus 0.3k local declared), well within the ceiling and the 20% reserve.
  - The diagnostic tranche used ~3.8k against its 10.8k cap.
  - About 43k core-s (25k below the 80% working ceiling) were intentionally left unspent after both repair gates failed. The brief warned against spending budget on repeated speculative repairs.
- **GPU:** 0 device-seconds charged; every job ran with `CUDA_VISIBLE_DEVICES` empty. The legacy extended-06 occupancy convention would report 14.2k s; it is shown separately and not charged.
- **Concurrency:** 4 memory-heavy jobs by default, raised to 6 after profiling (≤ 2.3 GB per job; > 40 GiB headroom kept). Never above 8.
- **Incidents** (all in decisions.md; failed jobs receipted and charged):
  1. The launch helper did not shell-quote commands. This was fixed in campaign07_remote before any real job, and the same latent bug exists in the extended-06 helper.
  2. A concurrency-guard refusal went unretried by a runner, so s0r0-bank-s43 was never launched. It was launched by hand, and the runner was rewritten to retry.
  3. Runner wait-loops stalled twice (P2 eval-stage waiter ≈ 40 min, cause not established; P3-REPLICATE watcher ≈ 25 min after its evals had finished). Wall time only. An ssh status call without a timeout is suspected for the second; later waiters use `timeout 60`.
  4. The first P3 frontier job failed because `--mkdir` pre-created the tool's output dir (0.1 core-s); it was rerun into a subdirectory.
  5. A mistaken runner relaunch (piped into `head -0`) was stopped before it launched anything.
  6. Merge conflicts (seed-ranges, campaign07_p2, campaign07_diag) were resolved by hand, and marker-free status was verified before each commit.
- **Seeds:** every new range is registered in [seed-ranges.json](seed-ranges.json), nested only in its parent sub-range. Historical campaign06 artifacts were read-only; the launch helper refuses outputs outside campaign07.
- **Machine state at close:** **no e07 jobs running or queued**; `campaign07_remote.py status` shows no active units. The pro6000 keepalive task and linger are still active, and the user's unrelated workloads were not touched. The GB10s were not used.

## Next research decision

1. **The bottleneck is interaction-driven change discrimination, not factor reading.** Supervising myopic factor summaries and building bilinear consumers over them did not produce it. The one lever that raised it through more correct changes was **trunk shaping** by auxiliary supervision, and only partly (S1R1 − S0R0 +.074 [+.010, +.143]; shaping alone +.060, unresolved).
2. **Recommended next step (small):** test whether auxiliary targets that *are* interaction-sensitive improve J_sub through shaping. Candidates are pairwise Q deviations, which a pairwise-additive reference shows are sufficient (J .92), or exact action-value differences between the full and ablated configurations. Run it within the S×R 2×2 (S1R0 vs S0R0), with J_sub primary and the near-miss and stratum guards.
3. **Not recommended now:** structural attention (its preconditions were not met); more consumer variants over the current 23 factors; the extended-06 flip-gain metric. Use J_sub and BA_nm with a stratum check instead.
4. **Protocol lessons** (carried forward):
   - pre-register discrimination, not flip accuracy;
   - evaluate φ ∈ {own, exact, zero, mean} for any factor-reading model;
   - keep 5 lineages and two-level 20k-draw CIs with MC checks;
   - put timeouts on remote status calls.
