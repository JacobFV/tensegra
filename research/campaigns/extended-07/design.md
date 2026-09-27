# Extended-07 design (v1, 2026-09-27; phases follow the brief; later revisions are appended below)

## Question
Supplied decision quantities (SUP) improve interaction-sensitive behaviour on S+C+E flips, while learned estimates (LRN) do not. Where does the learned factor → decision pipeline fail? The candidates are:
- prediction of the quantities (acquisition);
- preservation of their meaning (semantics drift through the shared trunk);
- consumption (the fusion/policy path);
- learning of the downstream interaction;
- the quantities themselves (targets/ontology).

## Phase 0: contracts (static plus mechanical; no training)
**P0a. Factor contract.** Document all 23 factor coordinates, grouped by semantic function into the four thematic groups:
- posterior/outcome probabilities;
- side/event costs;
- one-query strategy-cost estimates;
- amortized build quantities and bookkeeping.

For each coordinate, record:
- its definition, range and normalization;
- its input dependencies;
- whether it is exact, approximate, myopic or policy-dependent;
- whether it is supplied at inference or used only as a target;
- whether clipping or normalization loses information.

Then audit numerical consistency (probabilities, units, clipping, missing values, dependencies among targets). Verify that every target can be reconstructed from the permitted public history. The closed-form strategy-cost targets are **not** long-horizon Q*.

**P0b. Gradient-flow audit.** Build a table of which losses update which modules (recurrent encoder, shared trunk, auxiliary predictor, fusion, policy head) in RAWF, LRN and SUP. Include mechanical tests of gradients, parameter counts, initialization differences, and coupling through clipping, normalization, optimizer state and RNG.

## Phase 1: frozen-checkpoint diagnostics (≤ 3 CPU core-h in total)
**Inputs** (read-only): the five historical LRN/RAWF/SUP lineages with seeds 35–39. LRN seeds 30–32 may be used as development only.
- **Populations:**
  - the fresh `b6c_hold_SCE` pool and its octets (already inspected by extended-06, so it is diagnostic, not confirmatory);
  - the `b6_hold_SCE` development pool;
  - the training distribution, used for support/coverage comparisons.
- **Runner:** read-only, with versioned output names; it never overwrites eval files. At each decision it logs:
  - identity: configuration/history identity, query index and context (query 2 after H vs not-H, initial decision, failed probe, timeout, later-query start);
  - action mask;
  - factors: exact factor targets and predicted factors;
  - a pre-fusion latent or checkpoint reference;
  - policy probabilities and the chosen action;
  - exact Q* and the eps-optimal set;
  - the gap;
  - any intervention applied.
- **Protocols:**
  - **(i) common-history:** replay identical valid public histories through each model, rebuilding each model's own recurrent state. Histories come from a fixed reference (π*) and from each learned policy; low-probability histories are flagged.
  - **(ii) free-running:** every policy acts for itself, and its own state distribution is recorded.

  Denominators are never mixed.
- **Prediction quality** is measured per coordinate and per group:
  - absolute and normalized error, calibration, sign/order errors, invalid probabilities;
  - error broken down by condition combination × context × optimal-action margin;
  - a flip / near-miss / invariance split.

  Tolerances come from semantics or from development data, and are fixed before any explanation is drawn.
- **Replacement on the same frozen LRN and state:** predicted vs exact values, dependency-consistent group corrections, and controlled perturbations at the observed error scale.
  - It reports immediate decision rescue or harm first; rollout-level replacement is named separately.
  - It also reports whether replaced values lie inside the consumer's training support.
  - Exact values are never fed into RAWF's untrained channel.

## Phase 2: adaptive, chosen by the Phase-1 evidence (registered before use)
- **Controlled consumers:** exact-trained, trained on out-of-fold predictions, and trained on an exact/noisy mixture. They are trained on allowed combinations only and evaluated with both predicted and exact inputs.
- **Genuine 2×2 of S (auxiliary shaping of the trunk) × R (policy reads the predictions):** S0R0, S1R0, S0R1, S1R1.
  - Arms share modules, dimensions and explicitly copied initialization.
  - Tests guard against the auxiliary loss leaking into the actor through clipping, normalization, the optimizer or RNG.
  - Training first uses a common valid-history bank; free-running evaluation comes after.
- **Branches A–E** (brief §15):
  - A: factor acquisition fails;
  - B: predictions are adequate but the consumer is ineffective;
  - C: semantics or representation are incomplete;
  - D: the gain comes mainly from auxiliary shaping;
  - E: effects are seed-unstable or negligible.
- **Conditional:** structural attention (brief §16), with equal-information ordinary consumers as the baselines.

## Confirmation and statistics
- **Registration:** fresh validation/confirmation ranges are registered before use. Fresh S+C+E configurations confirm a mechanism within that family only.
- **Lineages:** ≥ 5 paired initialization lineages.
- **Predeclared:**
  - primary interaction metric;
  - near-miss non-inferiority margin;
  - competence floor;
  - regret metric;
  - aggregation;
  - checkpoint;
  - interval method: two-level (seed × shared configuration, whole octets), with ≥ 20,000 draws, a pre-fixed seed and a Monte Carlo resolution check.
- **Reporting:** seed-level effects and support are reported alongside intervals.
