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

Frozen-checkpoint matrix complete: 864 cells / 442,368 paired-policy episode evaluations. Independent raw reconstruction verified all864 cells, source/checkpoint hashes, every gate and every paired prior intervention. Across1,787,904 paired frames, all nonempty posteriors are bitwise unchanged and all239,616 empty frames receive the exact supplied prior. These counts repeat the same events across model/prior comparisons; they are not independent unique constructions. Original and expanded contract gates are separate. Every official gate includes the initial frame and full-retraction frames; supplied probabilities do not relax thresholds. No A3 calibration intervention or new training is currently justified or authorized. Composition and supervision withdrawal remain blocked.

## Coverage and confounds

The original broad gate is conditional on the supplied candidate-local representation and deterministic observation model. There are no noisy likelihoods, unknown candidate-generating processes, open-world entities, or schema-invalid operation labels. Support membership on an ambiguous frame is not complete-proposal correctness; posterior fidelity and expected-correctness calibration remain separate. Frame-averaged initial-prior error is diluted by longer episodes, so long duplication can make this particular error numerically smaller without calibrating the initial belief.

Observation-ID encodings were introduced in Stage8 to match public information between models; training exercised only IDs0–4. The original Stage9 `distinct_equal` control uses new IDs100+ and therefore combines content redundancy with ID-range shift. The separately preregistered seen-ID localization compares one identical redundant event under ID4 versus100 and permutes IDs within0–3; those outcomes are diagnostic development data, not reopened test selection. No inference-time private support or winner is supplied in any non-oracle comparison.


## Completed frozen-checkpoint comparison

| Comparator / prior | Original validation cells failed, seeds0/1/2 (16 per seed) | Expanded failed (24 per seed) |
|---|---:|---:|
| Protected / learned |5 / 5 / 3|11 / 11 / 7|
| Protected / supplied empty prior |0 / 0 / 0|4 / 4 / 4|
| Recurrent / learned |8 / 7 / 7|15 / 13 / 11|
| Recurrent / supplied empty prior |4 / 3 / 3|10 / 9 / 7|

The protected supplied-prior arm passes the original public-contract matrix at all three seeds (`passed_restricted`). Its untouched test clean, contradiction, retraction-back-to-prior and long-repetition final support selections are100%; mean posterior L1 in these conditions is at most .00144. This is learned evidence matching plus supplied bookkeeping and supplied prior, not learned prior calibration.

Both expanded-contract gates fail. Protected failures are precisely distinct-ID equal-content and arbitrary observation-ID renaming; new-ID test final accuracy ranges .38–.98 for the former and .006–.066 for renaming. Since both controls introduce previously untrained ID bit patterns, this is a newly localized representation/generalization failure. It is not evidence against the exact ledger rules. The separately registered seen-ID experiment below disentangles this confound.

The recurrent model's remaining original-matrix failures concern reorder/repetition posterior fidelity. At N16 long repetition after prior repair, test impossible mass is .0517–.0647, but null mass is only .0000036–.0000189, with zero false-null argmax frames. Thus the residual error is predominantly probability on wrong ordinary candidates, not learned null inference after substantive evidence. Full retraction additionally exposes intermediate uncertainty redistribution: mean L1 .315–.408 despite an exact supplied final prior and low impossible mass. Final support selection alone would miss this failure.

Additional N32 conditions remain separately descriptive; all models still fail the full N32 matrix, principally the new-ID controls plus recurrent posterior defects. No additional candidate-count condition silently grants a pass.

Total frozen-matrix runtime501.48seconds plus1.10second profile; peak allocated CUDA1,712,598,016bytes, process RSS1,649,568KiB. Six existing1024/2048 checkpoints,16,867,395parameters each; zero new optimizer presentations. The prior policy has zero trainable parameters but supplies an active-ID set. CUDA allocation and RSS are distinct measures on the shared-memory GB10; neither is total device capacity. Frozen source copies, configuration/checkpoint hashes and compressed posterior arrays are retained under `research/results/stage9/beliefs/`.

The raw-ID coverage is especially narrow: IDs0–4 vary only the lowest three bits of the16-bit neural feature; higher bits are constant during Stage8 training. Renaming into256–65535 changes previously constant input coordinates. Exact ledger lookup should be invariant to such bijections, but the learned compatibility network need not be. This is a specific input-representation/exposure confound, not a claim that any recurrent system cannot track observations. The seen-ID diagnostic is necessary before attributing the `distinct_equal` failure to redundant-content accumulation alone.


## Preregistered observation-identity localization

The48-cell post-registered diagnostic uses fresh development events, not reopened validation/test selection. It changes only one redundant observation's ID (seen4 versus unseen100), or permutes the four clean IDs within0–3. All512events/cell and allthree paired checkpoints remain visible. Prior policy is supplied in the table; unrepaired raw results are also retained.

| Comparator / control | Final support accuracy seeds0/1/2 | Mean posterior L1 seeds0/1/2 |
|---|---|---|
| Protected / clean |100 / 100 / 100%|.0007 / .0003 / .0004|
| Protected / redundant seenID4 |50.78 / 48.24 / 100%|.6565 / .6904 / .3277|
| Protected / redundant unseenID100 |50.78 / 48.24 / 99.80%|.6565 / .6904 / .3668|
| Protected / permuted seenIDs0–3 |100 / 100 / 100%|.0007 / .0003 / .0004|
| Recurrent / redundant seenID4 |96.29 / 100 / 100%|.2641 / .0039 / .0039|
| Recurrent / redundant unseenID100 |87.11 / 100 / 100%|.4991 / .0409 / .0092|
| Recurrent / permuted seenIDs0–3 |91.99 / 86.33 / 90.43%|.0740 / .0957 / .1457|

This narrows the new failure further: unseen high ID bits are **not the sole cause**. SeenID4 was only used for contradictory/retracted primitive evidence during training, with observed primitive value1 or2, never0. Posthoc raw stratification confirms seeds0/1 fail every ID4+primitive0 case (0/252 and0/265) and succeed on every ID4+primitive1 case (260/260 and247/247). Seed2 gets the final choice correct but its primitive0 posterior trajectory has L1 .668 versus .000157 for primitive1. That stratification is diagnostic, not a new selected gate.

The observed hole is compositional data coverage over a public nuisance identity and content value. Protected bookkeeping is correct; learned compatibility is sensitive to a semantically irrelevant association. Conversely, recurrent within-range ID permutation fails despite all numeric IDs having been seen; training ties IDs0–3 to arrival positions even when evidence roles are reordered. These controls do not prove which internal shortcut is used, but they exclude an explanation based only on unseen numeric ID magnitude and isolate failures under identity-renaming semantics.

The additional model runtime was16.23seconds, no optimization. Five mechanical tests passed. All historical source/result bytes remain unchanged. No null-head A3 or broader model repair is justified by these failures: the original protected defect was a supplied-prior mismatch; expanded failures concern compatibility/identity invariance and recurrent posterior redistribution. A future narrowly controlled training augmentation could test those associations, but was not run here and does not change the failed expanded gate.


A separate deterministic data reconstruction counts9,216 unique main public constructions and1,536 ID-localization constructions, with zero overlap. Identity here means the exact nonce-key and candidate-record content with candidate order removed, not alpha-equivalence or new abstract program structure. The much larger reported episode-evaluation counts repeat these constructions across conditions, model arms and prior policies. Per-example construction hashes and generation keys are saved in `data-audit.json`; no optimizer examples were consumed.
