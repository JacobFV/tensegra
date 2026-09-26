# Extended-04 report: metacognitive control, rational backtracking, and the value of structure

**Status: DRAFT (pending the A2/C and F independent audits).** Campaign window: 2026-09-26T18:43Z to 2026-09-27T18:43Z, on the pro6000.

**Sources:**
- Design: [design.md](design.md) (v1 plus review-driven v2).
- Protocols: A1, A2, B1, B2, C1, F (including the F1b addendum).
- Decision log: [decisions.md](decisions.md).
- Supplied/learned ledger: [ledger-supplied-learned.md](ledger-supplied-learned.md).

## 1. Headline

| Brief deliverable (§24) | Outcome |
|---|---|
| 1. Improvement over imitation, or a sharply localized reason for its absence | **Localized absence** (A2). Every stabilized RL variant prevents the greedy collapse, but none beats the imitation bootstrap in any deployment mode. RL improves the *sampled* policy it optimizes (.794 → .816–.844), yet that stays below greedy deployment of the imitation policy (.857–.861). Rehearsal equals imitation-only. Entropy control prevents collapse but doubles work per success. Every RL arm spends more work. |
| 2. A benchmark where unsuccessful initial attempts are rational | **Built and validated** (probeworld): exact belief-conditioned values, labels that depend only on visible history, timeout = unknown, irreversible commits, reuse-priced structure. |
| 3. A controller that recognizes stagnation and changes strategy | **Supplied rule, confirmed** (F1b). R-mask over progress diagnostic v1 removes 74–86% of stagnant steps at no utility or success cost on fresh lineages and sealed worlds (after F1's mis-specified episode clause failed). It also rescues collapsed RL policies (A1: 0–8% → 87–92% success). *Learned* stagnation control added no utility over this rule (C-H1 failed). |
| 4. Calibrated predictions of metacognitive variables | **In-distribution yes, under shift no.** depworld appraisal ECE is .009–.026 on fresh worlds, and it predicts the effects of stop and mask-top (C-H4). Probeworld value heads are calibrated in-distribution but not on held-out generator regions (B-H6 failed; B2 shows it is generalization, not a continuation mismatch). The value of a sampled step is not predictable (ρ ≈ 0). |
| 5. Causal evidence that deliberate regulation beats automatic control and heuristics | **Not established.** Learned control did not beat the best fixed rule on loop-prone bases (only P1-RL r2: +.018 utility, −24% cost, below the .02 threshold; its gain vanished under shuffled targets). It was trivially default on competent bases. |
| 6. A measured response to computation prices and reuse | **Yes (probeworld, confirmed F2).** Structure-building tracks the reuse horizon and prices: deviation from optimal is .016–.045 on held-out prices and k. |
| 7. An audited account of supplied, learned and generalized | In progress: the ledger plus 4 independent audits (B1 complete; A2/C and F pending). |

## 2. Track B: rational probing, switching and the value of structure (strongest positive result)
- **B1 ladder** (3 seeds; audited):
  - Action-set supervision (L1) and above reach held-out regret of 1.8–4.1 per episode. Outcome-only actor-critic (L0) gets stuck at 25–39, on a "never probe" rule (propagate → exact_b1 → commit).
  - Only L0 → L1 is beyond noise; stage/dependency, switch and Q* supervision add nothing measurable.
- **F2 confirmation** (fresh seeds 3–5, fresh world draws):
  - L1/L4 score 1.5–2.6 vs L0's 39–82.
  - The ρ/k build response is confirmed.
  - Rational first-probing and justified switching are confirmed **pooled over held-out splits**.
- **Limit (replicated exactly):** on held-out *combinations of conditions*, the uniquely-optimal probe rate is .545 and the justified-switch ratio .51–.58. Transfer covers new price regions and reuse horizons, but not new condition compositions.
- **Calibration:** the value heads fail held-out calibration (.08–.26 R), both V and own-greedy-return v_own (B2). Over-prediction is worst at held-out k = 4.

## 3. Track A: deployment and improvement (depworld)
- **A1 deployment matrix:**
  - R-mask/R-sample recover collapsed P1-RL policies (greedy .00–.08 → .87–.92 success), though their utility (.77–.83) stays below the bootstrap's (.86–.87).
  - R-mask does no harm to competent policies.
  - Sampling is the worst mode for every competent policy.
- **A2 improvement screens (one lineage):** no promotion. The localized reason is in §1, row 1.
  - The P2a anchor effect is not "just entropy control": entropy control needs twice the work.
- **F1/F1b:** see §1, row 3.

## 4. Track C: learned appraisal and control
- **Setup:**
  - Telemetry: 89 features. Appraisal GRU: 30k parameters, charged as compute.
  - Labels are regression targets from branch evaluation (oracle simulation, training only).
  - Default and continuation: greedy + R-mask.
- **Results:**
  - Calibration is good (C-H4).
  - One-step intervention value over the default is near zero for competent bases and for two of three loop-prone bases.
  - Learned control did not beat fixed rules (C-H1/C-H3 failed).
- **Registered localization:** "calibrated appraisal, but no actionable headroom over simple public rules".

## 5. Integrity, accounting and deviations
- **Metering.** All remote work was metered (main jobs plus dev/test/audit via metered.sh). Local no-torch work is estimated (~0.7k core-s).
- **Spend:** CPU ≈ 116k / 172.8k core-s (reserve intact); GPU occupancy ≈ 9.1k / 43.2k s.
- **Overrun:** the Track C evaluation used 45.5k core-s against a 14k estimate.
- **Deviations and incidents (all logged):**
  - A pre-registration smoke touched 16 Track C evaluation worlds, so the range was moved to 161M.
  - 12 empty-SHA label launches (0 s, charged); the launch helper now refuses them.
  - The B-H6 cause was misattributed twice before B2 and the audit corrected it.
  - F1's episode-level clause was mis-specified; F1b was registered adaptively on fresh worlds.
  - Three decision timestamps were written ahead of wall-clock time and have been corrected.

## 6. What this changes, and next steps
*(to be completed after audits)*
