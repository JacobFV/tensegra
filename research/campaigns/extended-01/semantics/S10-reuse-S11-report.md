# S10 frozen policies on S11: prospective secondary analysis

Unchanged decoder `2085053`, prospectively registered reuse `8042d2f`, fixed S11 final and original S04 midpoint at **196,608 presentations**. Independent raw audit pending. No model/threshold selection or new policy fitting.

## Findings

S11 primary learned canonical DEV recovery remains **122/512 calibrated, 13/512 raw**. Supplied grammar/bookkeeping raises secondary recovery to **372/512 calibrated (72.66%) and 381/512 raw (74.41%)**. Both decoding modes were registered; the higher raw result is not used to replace the primary calibrated policy.

Combined rules repair 250 calibrated and 368 raw S11 graphs, with no previously exact S11 graph regressing. They still introduce 105 calibrated and 144 raw active edge errors on other events; supplied grammar is not error-free inference. Learned node presence/type/value/copy and valid structural associations remain necessary. The complementary inherited frequency-plus-grammar control recovers zero complete DEV graphs.

| Split | Mode | Snapshot | Baseline | Schema | Bookkeeping | Combined | Combined typed F1 | Combined ordered F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| train | raw | S04_mid | 0/128 | 0 | 1 | 4 | 0.94743 | 0.86752 |
| train | raw | S11 | 6/128 | 13 | 59 | 103 | 0.99840 | 0.99576 |
| train | calibrated | S04_mid | 0/128 | 0 | 11 | 40 | 0.98479 | 0.96740 |
| train | calibrated | S11 | 29/128 | 44 | 87 | 108 | 0.99864 | 0.99649 |
| development | raw | S04_mid | 0/512 | 0 | 3 | 25 | 0.94520 | 0.86809 |
| development | raw | S11 | 13/512 | 41 | 280 | 381 | 0.99405 | 0.99120 |
| development | calibrated | S04_mid | 0/512 | 1 | 53 | 183 | 0.98014 | 0.96199 |
| development | calibrated | S11 | 122/512 | 157 | 325 | 372 | 0.99401 | 0.99131 |

## Paired original-S04 → S11 outcomes

Gold target tensors and seed populations match exactly before pairing. These are the same events across snapshots/policies, not independent new samples.

| Split | Mode | Policy | Correct→correct | Correct→wrong | Wrong→correct | Wrong→wrong |
|---|---|---|---:|---:|---:|---:|
| development | calibrated | baseline | 0 | 0 | 122 | 390 |
| development | calibrated | bookkeeping | 48 | 5 | 277 | 182 |
| development | calibrated | combined | 165 | 18 | 207 | 122 |
| development | calibrated | schema | 1 | 0 | 156 | 355 |
| development | raw | baseline | 0 | 0 | 13 | 499 |
| development | raw | bookkeeping | 3 | 0 | 277 | 232 |
| development | raw | combined | 24 | 1 | 357 | 130 |
| development | raw | schema | 0 | 0 | 41 | 471 |
| train | calibrated | baseline | 0 | 0 | 29 | 99 |
| train | calibrated | bookkeeping | 10 | 1 | 77 | 40 |
| train | calibrated | combined | 36 | 4 | 72 | 16 |
| train | calibrated | schema | 0 | 0 | 44 | 84 |
| train | raw | baseline | 0 | 0 | 6 | 122 |
| train | raw | bookkeeping | 1 | 0 | 58 | 69 |
| train | raw | combined | 4 | 0 | 99 | 25 |
| train | raw | schema | 0 | 0 | 13 | 115 |

The combined calibrated difference is +189/512 graphs, but 18 originally correct graphs become wrong while 207 recover. A difference of marginals is not a per-example retention measure. Raw combined gains356 while one originally correct graph regresses.

## Interpretation and limits

This supports complementary effects from improved learned acquisition (the separate S11 LR-only intervention) and a supplied compiler output contract. It is not a pure learned semantic score, broad language understanding, unseen-motif/renderer transfer, a three-seed confirmation, or programmable-attention evidence. The known-family development set has been inspected; all results remain exploratory despite prospective policy reuse. Historical gates and primary S11 metrics are unchanged. S04 midpoint was selected by matched exposure before these outcomes, not retrospectively as a favorable endpoint.

No gold attributes/counts/roots enter the decoder. Exactly one predicted scope is required for supplied contains/declares; S05 all-matching copied-identity references remain unchanged. Raw tensor conflict/change counts include padded nodes, while metric errors use active-node conventions. Every output is reconstructible from immutable archive hashes and frozen policy, verified by complete edge/slot hashes.

CPU wall21.41seconds, maximumRSS612080KiB, zeroGPU. Target-paired analysis is a separate stdlib CPU step. No new training or architecture. Independent reconstruction requested; no confirmation authorized by this report alone.
