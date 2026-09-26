# Protocol B1: probeworld factorization-supervision ladder

Registered 2026-09-26T19:40Z, before any full-ladder run. Only the builder's 1-seed reduced smoke exists; it is not used for any choice below except confirming the defaults run. Spec: [probeworld.md](probeworld.md).

## Design

- **Rungs** L0–L4: outcome-only actor-critic, then + optimal-set imitation, + stage/dependency, + switch/case, + Q* regression.
- **Matched:** architecture (GRU, 128 hidden, 129,189 parameters), initialization, world stream and updates (4,000 × 64).
- **Seeds:** 3 per rung (0, 1, 2), 15 runs. The final checkpoint only; no selection.
- **Inputs are public only.** Labels appear only in training losses.
- **Evaluation:** 512 episodes per split (128 configs × 4 worlds) over dev, test_iid, heldout_price, heldout_k and heldout_comp. Splits are defined by generator parameters. Nothing is tuned on the held-out splits.
- **Reference rules** (fixed) and the exact optimum π* are evaluated on the same episodes.

## Primary metric

Utility regret per episode against V* (the exact belief-conditioned optimum), on the three held-out splits. Each split is reported separately, together with their equal-weighted mean. Absolute success and total cost are reported alongside.

## Hypotheses and decision rules (per-seed values always reported)

| ID | Claim | Supported if |
|---|---|---|
| **B-H1** (factorized supervision helps held-out) | L4 < L0 in held-out regret | In **3/3 seed pairs** L4 regret ≤ 0.8 × L0 regret on the held-out mean, **and** on each held-out split L4 ≤ L0 in ≥ 2/3 seeds. |
| **B-H2** (which factor matters) | Descriptive ladder | Adjacent-rung differences, with per-seed values. A rung "adds value" if it reduces held-out mean regret by ≥ 10% in ≥ 2/3 seeds relative to the previous rung. |
| **B-H3** (cost-sensitive structure use) | Build decisions track ρ | On heldout_k and heldout_price, the mean absolute difference between the model's P(build) and π*'s P(build) across the registered effective-ρ bins is ≤ .15, for the best rung, in ≥ 2/3 seeds. Reported for all rungs. |
| **B-H4** (no fail-first shortcut) | Rational probing, not always-probe | On episodes where probing is **not** ε-optimal as the first action (π* goes directly to exact or inspect), the model's first-action probe rate is ≤ .10. On episodes where probing is uniquely optimal, the probe rate is ≥ .80. Best rung, ≥ 2/3 seeds. |
| **B-H5** (justified switching) | Switches follow value | Among the model's within-query strategy switches, the fraction that are **unjustified** (not ε-optimal) is ≤ .10. Justified switches after case-(b) failed probes occur at ≥ .8× π*'s rate. Best rung, ≥ 2/3 seeds. |
| **B-H6** (calibrated value estimates) | Value head predicts outcomes | Mean absolute deviation between binned V̂ and realized return-to-go (10 quantile bins) is ≤ .05 × R on held-out splits, for rungs with Q/V supervision. L0's value head (a baseline only) is reported. |

- "Best rung" is fixed per hypothesis as **L4**, to avoid selection. Other rungs are descriptive.
- **Transfer floor:** a held-out claim also requires absolute success ≥ .8 × π*'s success on that split.
- These are synthetic-benchmark claims about supervision type. They are not claims about depworld or about deliberate metacontrol. Track C carries that.

## Budget

- ~6.5k core-s CPU-only (builder estimate): labels 0.4k, training 15 × ~0.4k, evaluation 0.5k.
- Run ≤ 8 training jobs concurrently (~2.5 GB RSS each).
