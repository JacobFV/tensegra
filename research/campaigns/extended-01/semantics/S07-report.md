# S07: occurrence supervision was partially learned, but did not improve complete graphs

The prespecified promotion gate fails. At 65,536 presentations the occurrence-supervised original actor recovers zero complete TRAIN diagnostic graphs (0/128) and zero complete DEV graphs (0/512), raw and calibrated. Its canonical copy accuracy is .93149 and calibrated ordered-edge F1 .78781, below the required .95/.90 thresholds and below the matched S01 reference (.95768/.85424). No extension or confirmation is triggered automatically.

| Presentations | DEV canonical copy | DEV typed F1 raw / calibrated | DEV ordered F1 raw / calibrated | Exact raw / calibrated |
|---:|---:|---:|---:|---:|
| 8,192 | .59820 | .26021 / .60817 | .30566 / .43642 | 0/512 / 0/512 |
| 32,768 | .83876 | .31274 / .68939 | .40213 / .65694 | 0/512 / 0/512 |
| 65,536 | .93149 | .47361 / .75343 | .59776 / .78781 | 0/512 / 0/512 |

Early canonical copying improves over S01, but that benefit does not persist to the endpoint or produce better calibrated graph recovery. The changed copy loss trains a different target; its numerical value is not directly comparable to the original first-identity cross entropy.

Final DEV presence is exact for 511/512 graphs, kinds 16,365/17,337 (78 complete sequences), canonical identity copying 12,058/12,972 (113 complete sequences), and finite values 4,365/4,365. TRAIN kind/copy complete sequences are 16/128 and 30/128. Replacing all edges with gold recovers 23/512 DEV graphs and 7/128 TRAIN graphs; other single-field or all-node-attribute replacements recover zero. These privileged ceilings are diagnostic, not acquired performance.

The separately frozen S05 identity-equality rule improves calibrated reference F1 .02783→.81315 and makes 23/512 reference sets exact, but full graphs remain zero. Gold reference replacement alone also yields zero complete graphs. Primary learned raw/calibrated outputs remain unchanged.

S07 differs from S01 only in privileged TRAIN identifier-copy positions: each ident node is supervised at its actual ordered occurrence; entity nodes retain first-identity positions. Deployment uses the same first-identity canonicalization, with no gold spans, new input channel, or S06 architecture. Common initial weights match exactly. The English mapping was independently audited; the existing identity contract was valid, not a label bug.

Source `04b44e9`, main config `2c576db`: width1,024, eight workspace rows, four blocks reused twice, 57,853,781 parameters; seed201, 8,192 unique graphs, eight visits each, 3,872,144 tokens. Optimizer time494.84398 s; full occupancy673.53936 s. Peak CUDA allocation1,225,695,232 bytes; process RSS3,800,772 KiB. Same learning rate, curriculum, batch8, sampled losses and TRAIN-only calibration as S01. Final checkpoint retained; no test or checkpoint selection.

All results are development evidence on two tree shapes at depth3, with diverse equality/reference patterns. [S08](S08-report.md) separately tests whether the new alignment target was acquired. The original occurrence proposition is not refuted merely because graph recovery fails; its elementary learned effect must be measured directly.

Raw scores/targets, calibration records, field/error counts, immutable checkpoint hashes and full receipts are under `research/results/campaign-01/semantics/s07-occurrence-n8192-dev201`; separate engineered-reference predictions are in `s07-occurrence-identity-diagnostic`. Independent reconstruction requested. Reserved confirmation predictions remain untouched.

Independent audit completed: reviewer `c538e8a` reconstructed S07 raw decisions/calibration and S08 lossless-logit statistics; `9196ec2` independently checked all3,840 exact-isomorphism rejections. No count discrepancies were found.
