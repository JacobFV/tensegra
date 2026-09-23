# Stage 9: closed-candidate probability contracts

## Historical failure localization

The standard-library `research/tools/stage9-belief-archive.py` reconstructs all 192 primary Stage8 cells / 98,304 episode evaluations from archived probability arrays. This is **not rerun inference**. Protected validation has 5 L1 failures and 14 impossible-mass failures; recurrent has 11 and 24 respectively. Initial protected null mass spans .01729–.05678 across N8/N16 and seeds. Ordinary candidate probabilities are equal. Across every saved protected initial frame, `|L1 − 2 × null_mass| <= 1.01e-7`. The full archive table contains per-condition support, ambiguity, null-target counts, signed null errors, worst L1, source-file hashes and initial probability spread.

The benchmark is a deterministic, closed-candidate intersection problem. Empty evidence means a uniform prior over ordinary candidates, not null. Ambiguous supported candidates mean uncertainty. No compatible candidate means null. Type invalidity is not a separate learned target in this dataset and cannot be claimed as tested. Equal content under distinct IDs is still the same noiseless equality constraint, unlike independent noisy evidence.

## Frozen model mechanism

At an empty protected ledger the ordinary candidate scores are zero; the null head receives `(max_score, mean_score)=(0,0)` regardless of candidate count. Its finite null logit `a` gives `u=exp(a)/(N+exp(a))`, hence L1=`2u`. Candidate-count dependence is therefore explainable directly from one null logit, not an evidence-integration failure.

Frozen seed0 gradients are nonzero: the initial posterior-loss derivative with respect to the null output bias equals .05678 atN8 and .02922 atN16, matching null mass. Initial loss exceeds the irreducible uniform-target entropy by `−log(1−u)`. The frame-averaged training objective dilutes the initial-frame contribution by the number of frames; the compatibility auxiliary does not directly train null-head parameters. Later clean mean bias gradients are approximately .00009/.00011. The null head also serves contradictory evidence, so it is not an isolated prior parameter.

These local gradients establish residual optimization pressure and rule out a zero-gradient explanation at the frozen checkpoint. They do **not** reconstruct optimization history, prove why AdamW stopped at that value, or establish that more training would fix every case. No optimizer-state history is available. Exact zero probability requires an infinite softmax logit gap, but the inherited finite tolerance is expressible; this is not a proof of an impossible training target at the gate threshold.

## Minimal intervention and supplied contract

A1 preserves each frozen learned model. A2 replaces the output distribution only when a publicly maintained observation-ID set is empty, including full retraction. It supplies the exact uniform non-null prior. It does not inspect compatible support, private gold labels, or the winning proposal. All nonempty predictions remain unchanged. The same additional public set bookkeeping is supplied to both comparators; it is an architectural intervention, not a claim that the recurrent model learned a ledger or prior.

Six frozen Stage8 checkpoints provide three paired initialization seeds per comparator; no new optimizer presentations. Source/config freeze523ec15 precedes all new test evaluation. New validation/test namespaces9.1M/9.2M are distinct from Stage8 inspected data. Every cell has512 episodes. Original eight conditions remain; full retraction, distinct-ID equal content, candidate permutation and observation-ID renaming extend the contract. N32 is separately labeled additional-size OOD. No hyperparameters, confidence thresholds or checkpoint selection are fitted on the new test data.

The supplied prior is an engineered closed-world guarantee. A pass cannot establish open-world uncertainty, readiness correctness, learned prior inference, autonomous composition, or metric-programming benefit; this model contains no structural-attention intervention.

## Evaluation status

Frozen-checkpoint matrix running; all final per-seed results and gate decisions will be appended after completion and independent raw reconstruction. Original and expanded contract gates are separate. Every official gate includes the initial frame and full-retraction frames; supplied probabilities do not relax thresholds. No A3 calibration intervention or new training is currently justified or authorized. Composition and supervision withdrawal remain blocked.
