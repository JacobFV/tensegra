# Stage 10 belief role separation

## Outcome

Both minimal repairs pass every preregistered validation cell across three paired seeds: **90/90 required N8/N16 cells and 45/45 separately labeled N32 cells per arm**. Untouched test data independently satisfy the same numerical thresholds in all 135 cells per repair. The correlated-ID reference passes the original eight conditions but fails the expanded ID-stress contract in every seed.

This is a restricted learned-interface result. The ledger-only model has programmed renaming invariance; the randomized-ID model learns strong average robustness but retains rare large intermediate posterior changes. Neither result establishes open-world uncertainty, unrestricted recurrent belief competence, readiness/execution, or programmable attention.

## Frozen comparison and supplied contracts

[Design](stage10-belief-design.md), nine main configs, and actor sources were frozen at `bc18496` before main training/inference; actor source is the same as the development snapshot `8fd9831`. There was no main checkpoint selection or recipe extension. Seeds 20/21/22 each receive 1,000 updates × 32 fresh episodes. Initialization tensors and semantic training-stream hashes agree across arms within every seed. Only the declared ID input/training assignment differs.

All arms use the **protected** aggregator, with four candidate-local neural residual phases, workspace width **1024**, FF width 2048, and 16,867,395 allocated parameters. This does not retrain the historical generic recurrent comparator. No attention memory tokens or structural-attention intervention are added.

- `ledger_only`: public observation handles still address the exact ledger, while sixteen neural ID features are zeroed. Its 16,384 corresponding input weights are inactive. Exact probability invariance under consistent handle bijections is architectural; compatibility learning is not.
- `raw_randomized`: raw 16-bit handles remain neural inputs, but training applies an independent random bijection per episode. Repetition/retraction references preserve handle identity.
- `raw_correlated`: fresh training retains historical ID/content associations. It is a matched reference, not a reused Stage 8 checkpoint.

The exact empty-ledger prior is supplied during training and evaluation in **all arms**, including after full retraction. Finite masking logits avoid zero-target × negative-infinity loss failures. Ledger first-write idempotence and exact removal are programmed. Nonempty compatibility and null behavior remain learned. This shared prior means the new reference is not an exact replica of Stage 8's learned-prior training recipe.

The benchmark uses noiseless deterministic constraints. Distinct handles with identical content need not change compatible support; these experiments do not model independent noisy measurements accumulating likelihood.

## All seeds, validation and untouched test

There are 512 episodes per cell, fifteen conditions, three candidate counts, two splits, and nine trained models: **810 cells / 414,720 policy-episode evaluations**. Initial and intermediate frames remain included. Validation is the registered gate split; test threshold counts below are independent descriptive confirmation, not reopened selection. Ambiguous support membership is not unique-proposal certainty.

| Arm | Seed | Validation N8/N16 | Validation N32 | Test N8/N16 | Test N32 |
| --- | --- | --- | --- | --- | --- |
| ledger_only | 20 | 30/30 | 15/15 | 30/30 | 15/15 |
| ledger_only | 21 | 30/30 | 15/15 | 30/30 | 15/15 |
| ledger_only | 22 | 30/30 | 15/15 | 30/30 | 15/15 |
| raw_randomized | 20 | 30/30 | 15/15 | 30/30 | 15/15 |
| raw_randomized | 21 | 30/30 | 15/15 | 30/30 | 15/15 |
| raw_randomized | 22 | 30/30 | 15/15 | 30/30 | 15/15 |
| raw_correlated | 20 | 24/30 | 12/15 | 24/30 | 12/15 |
| raw_correlated | 21 | 26/30 | 13/15 | 26/30 | 13/15 |
| raw_correlated | 22 | 28/30 | 14/15 | 28/30 | 14/15 |

Each repair passes all original and expanded gates. The correlated reference passes 78/90 required and 39/45 N32 cells on each split; failures concern arbitrary ID renaming and some distinct-equal-content ID controls. All original eight conditions pass across all arms/seeds.

Across all test cells, ledger-only worst mean framewise L1 is **.002853**, worst impossible mass **.001426**, and minimum final support accuracy **511/512**. Randomized-ID worst L1 is **.004879**, impossible mass **.002440**, and final support accuracy **512/512** in every cell. These are finite evaluated supports, not universal guarantees. The exact renaming guarantee says changing opaque handles cannot change ledger-only predictions; it does not guarantee those predictions are correct.

Repeated evidence, contradiction, intermediate retraction, and full retraction are evaluated separately in raw metrics, including ordinary impossible mass and signed null error. Good aggregate scores therefore do not come from removing the initial frame or conflating ordinary probability redistribution with null. Empty-ledger correctness remains supplied, not learned.

## Renaming tails: average calibration is not exact invariance

Paired clean/renamed episodes have identical gold posterior targets. `invariance-tails.json.gz` retains all per-episode deltas, final outcomes, actual training-ID coverage, and 108 counterfactual cells. The following untouched-test values are the maximum individual probability change over all frames of each episode, under arbitrary consistent ID renaming. Each row has 512 paired episodes.

| Randomized seed | N | p95 | p99 | Maximum | Counts >.01 / >.05 / >.1 | Final argmax changes |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 8 | 0.006130 | 0.024292 | 0.664901 | 12 / 3 / 2 | 0 |
| 20 | 16 | 0.002640 | 0.060164 | 0.337879 | 13 / 7 / 4 | 0 |
| 20 | 32 | 0.000809 | 0.116616 | 0.499813 | 9 / 6 / 6 | 0 |
| 21 | 8 | 0.000131 | 0.000648 | 0.155624 | 1 / 1 / 1 | 0 |
| 21 | 16 | 0.000036 | 0.000163 | 0.155624 | 1 / 1 / 1 | 0 |
| 21 | 32 | 0.000006 | 0.000027 | 0.418894 | 3 / 3 / 2 | 0 |
| 22 | 8 | 0.000086 | 0.021002 | 0.497589 | 6 / 5 / 5 | 0 |
| 22 | 16 | 0.000011 | 0.006731 | 0.497344 | 5 / 5 / 4 | 0 |
| 22 | 32 | 0.000007 | 0.031767 | 0.635787 | 10 / 5 / 4 | 0 |

Ledger-only delta is exactly zero for every arbitrary and within-range bijection in every seed/split/size. Randomized-ID final selections do not change in the table, but intermediate confidence occasionally does. No new tail threshold was used to redefine the registered mean gates.

The largest randomized test example is seed 20, N8, episode 149, frame 3: clean mass on the sole compatible candidate is approximately 1; after renaming it is .3351, with two impossible candidates near .3324 each. The four replacement numeric handles were all encountered in training. Therefore numeric novelty alone cannot explain this example; joint handle/content coverage and learned nuisance sensitivity remain distinct possibilities. Seen numeric handles do not imply every handle/role/content combination was trained.

`invariance-failure-events.json` supplies public nonce keys, candidate records, event streams, frame events, and gold support for the largest three counterexamples per run. They are regenerated from archived event seeds without new model inference. Within-range permutations have their own complete tail rows and must not be collapsed into arbitrary-ID novelty.

## Exposure, pairing and provenance

Randomized training sees 56,253 / 56,262 / 56,333 unique numeric handles in seeds 20/21/22, versus five in correlated and ledger-only training. Handles 4 and 100 appear in all randomized runs, though not with every semantic role/value. Legacy condition `distinct_equal_unseen_id` denotes historical ID100; it is **not** a current unseen-identity claim. Full inventories and per-cell actual unseen-assignment counts are archived.

Every run sees 32,000 exact public constructions and 32,000 presentations. Those construction hashes identify nonce/candidate content, not alpha-distinct semantic algorithms. Paired arms share semantic episodes and initialization; the 288,000 presentations are not 288,000 independent semantic constructions. Repeated evaluation across conditions/arms is likewise paired, not independent support.

Main model runtime is **1,099.01 seconds**; profiles/development add **61.63 seconds**, totaling approximately **19.34 GPU minutes**, below the 45-minute ceiling. Peak allocated CUDA memory is **564,333,056 bytes**; peak process RSS **2,092,916 KiB**. These are distinct from total device capacity. Evaluation/compact export dominates much of elapsed runtime. Equal allocated parameters do not imply equal useful input features or equivalent attention cost.

Development used seed 101, 300 updates per arm, and separate namespaces. Both repair arms achieved 64/64 clean and renamed final selections, with exact ledger invariance; randomized maximum probability delta was already .1484. The reference achieved 64/64 clean but 3/64 renamed. This passed the prespecified continuation rule, not a main gate. Four mechanical tests verify bijections, unchanged ledger addressing, finite prior loss/backward, and nonempty behavior.

Raw probabilities, gold support, event hashes/seeds, all losses/curves, ID inventories, configs, source hashes, initial/final tensor hashes and checkpoint byte hashes are committed under `research/results/stage10/beliefs`. Immutable checkpoints remain at `gb10-direct:~/topoformer-stage10/beliefs/main/{arm}-{seed}/{initial.pt,model.pt}`. The independent audit regenerates raw metrics and counterfactual tail counts; its final disposition is recorded in the Stage 10 review.

## Interpretation and decision

**Programmed:** ledger identity semantics, exact empty prior, and ledger-only handle invariance. **Learned:** evidence/content compatibility and useful nonempty posterior behavior. **Calibration/generalization:** the registered full framewise mean contract passes in both repairs on fresh data and larger candidate sets; randomization does not eliminate rare nuisance-induced confidence changes.

The narrow conclusion is that opaque observation addresses should be separated from semantic compatibility, or exposed through sufficiently independent assignments during acquisition. This does not show that neural recurrence intrinsically cannot preserve structured state. It also does not repair the separate generic recurrent intermediate-retraction failure, establish open-world null semantics, or validate structural attention.

Composition, readiness/execution coupling, supervision withdrawal and additional experiments remain blocked. No new null head or architecture expansion was needed for this result.
