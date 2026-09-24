# A08 frozen joint edge replacement

All nine frozen A06 models agree100% with exact execution on the supplied corrupt graph in every cell. Soft attention, hard gather, and keyed context therefore tie on this diagnostic. Their accuracy against the original graph's answers declines together as public edges are replaced.

| Shape | No replacement | 12.5% realized | 25% | 50% |
|---|---:|---:|---:|---:|
| N64/D8/K4 clean-answer accuracy | 100% | 38.77% | 18.07% | 13.09% |
| N128/D32/K8 clean-answer accuracy | 100% | 12.99% | 11.43% | 11.62% |

Every entry is identical across seeds601/602/603 and all three interfaces, with1024 paired fresh graphs per shape. The corrupt-graph exact executor has exactly the same clean-answer accuracies. Thus the measured error is sensitivity to changed graph facts despite faithful execution of the supplied graph. It is not evidence that models should reconstruct missing truth.

This intervention jointly removes and inserts edges by degree-preserving destination rotations within public attribute blocks. Requested10% realizes12.5%;25% and50% are exact. It preserves one neighbor perattribute, and does not test pure deletion or arbitrary free-degree graph noise. All successful A06 scales were fixed before this run: content16 for soft/gather, address16+content16 for context. Fraction0 is100% for all nine models. No unique soft-attention robustness advantage appears.

Sourcee789147; config file SHA256 `04b61a0acd05ece7cab53eb97759b4e9203a6f1011b3ff05b97d76b694a9e2e6`. All checkpoint file hashes and initial/final tensor hashes verified; zero optimizer updates. Full-process receipt206.166659217seconds persisted atomically before exit. Internal wall205.111456521seconds, CUDA peak437,634,560bytes. GPU verified free immediately after completion. Attention accounting throughA08/A09 is1272.213297805seconds including the separate recovered A08 profile's60-second conservative upper bound; the uncertain profile bound is not a measured duration.

Raw public topology/labels stored once, plus per-checkpoint predictions, routes, metrics and process receipt: `research/results/campaign-01/attention/a08`. Remote durable mirror: `~/topoformer-campaign01/attention/a08-receipted`. Independent72-cell raw audit requested. These are diagnostics, not new confirmation or model-selection results.
