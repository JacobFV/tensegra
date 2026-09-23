# Stage 2 methods and interpretation guide

The complete experiment preset is `configs/study.json`; configuration defaults are serialized in the result summary. The training source revision is `ee49edc`. The study was designed before observing its full-run results. This document describes the measurements without implying positive results.

## What changes from pilot 1

The process remains `x_next = 0.5*x + 0.5*tanh(W*x) + epsilon`. Uniform nonself weights sum to approximately 0.8 per active row. Sparse graphs now use edge probability `min(3/(N-1),1)`, preserving expected indegree as node count grows. Sparse 12 therefore has a slightly different graph distribution from pilot 1's fixed0.25 density. Do not treat the two pilots' percentage improvements as directly paired estimates.

The robot case is the same morphology-shaped tree abstraction. It has local synthetic couplings, not physical joint mechanics, actuator commands, rigid-body contacts, vision or language. Hard attention restricts each layer's reads to graph neighbors and self; multi-layer computation can propagate through paths. Finite soft bias permits direct off-graph reads, so its empirical usefulness does not establish hard physical locality.

Signed mechanisms retain the adjacency while assigning nonuniform positive/negative coefficients, normalized to absolute row sum below 0.8. This guarantees a transition infinity-norm Lipschitz bound at most 0.9. The signed family is used for fixed-process experiments; it does not demonstrate identifying arbitrary hidden coefficients on unseen process families.

## Training and pairing

Models use four observations per node, width 32, four heads, two attention blocks, shared tokenwise normalization/MLPs, and one scalar next-state output per node. AdamW trains for 600 steps, batch 32, learning rate 0.001. Checkpoints are 0, 25, 50, 100, 200, 300, 600. There are three seeded training replicates per reported condition. Sparse seeds vary topology as well as trajectories and initialization. Uniform robot seeds vary trajectories and initialization on the same morphology and coefficients; they are not independent morphology draws.

Variants within a case have identical common parameter tensors at initialization and identical sampled graph/window schedules. Hashes preserve this pairing. The graph-input model adds a bias-free history projection; learned variants add layer/head coefficient tensors; parameter counts disclose these differences. Node-identity supplementary models add the same learned 12-by-32 table across none/soft4/hard. All transfer variants remain identity-free.

The ordinary fixed-graph predictor is permutation equivariant and has no variable IDs. Thus the no-graph baseline cannot directly memorize a fixed arbitrary adjacency by variable name. Its comparison with a graph-informed model mixes available information/representability and learning difficulty. The supplementary identity-enabled sweep is a necessary qualification, although it does not guarantee that 600 steps exhaust the identity-aware baseline's capabilities.

## Dataset budgets

Fixed-process suites use separate trajectory seeds for train, validation and test. Training trajectory counts are 8, 16, 32, 64, 128, 256, nested prefixes of a common pool. Validation/test each contain 16 independent trajectories. Sequences have 40 observations after 32 burn-in updates, innovation standard deviation 0.01. Scalar normalization is fit on only the smallest eight-training-trajectory subset and shared across budgets. Larger budgets do not leak distribution statistics into smaller runs.

Counts are whole independent trajectories, not independent windows. The runner also reports trajectory-local training windows and optimizer examples. Within a count, schedules are exactly matched across model variants. Across counts, the random seed is shared but index ranges and realized schedules differ. There are no exact cross-count scheduling claims.

Transfer trains one model across 16 graphs, validates on four unseen graphs, and tests on eight unseen graphs at each of 12, 32, 64, 128 nodes. Training conditions are fixed 12 nodes or mixed 9/12/16 nodes. Each training graph supplies 128 trajectories; batches sample graphs uniformly and then windows within that graph. Normalization uses training graphs only. Graph hashes are checked for disjointness across splits. Model selection uses validation graphs only.

## Supplied topology

Finite biases are fixed at 1 or 4; hard masks permit supplied edges and self. Permuted controls relabel adjacency without relabeling state variables, and use matched finite bias strengths. Correct topology and permuted topology preserve the number of edges, including self edges.

Corruption changes the supplied graph, never the generating coefficients. Drop/add fractions 10/25/50% mean rounded fractions of original directed nonself edge count. Additions sample original nonedges and cap at available capacity. Mixed 25% drops true edges and adds original nonedges independently. Realized precision/recall and graph hashes are saved. On symmetric robot truth, corruption operates on directed reads and may break symmetry; it models incomplete dependency knowledge, not removal of a physical bidirectional link.

Retrained corruption models see that supplied corruption during training and validation. Clean-trained models are also evaluated with runtime corruptions, without retraining. Selecting a soft strength for this runtime comparison must use clean validation, not corrupted test performance.

Graph-input computes the supplied adjacency's normalized neighbor-history average and adds a learned projection to the node input. Attention remains unbiased. This is a strong graph-conditioned input/message-passing baseline. It is not an exhaustive comparison against serialized graph tokens; it receives graph information through explicit aggregation. Typed signed biases receive actual signed coefficient magnitudes, an additional information advantage over adjacency-only models. Typed comparisons use clean signed cases.

## Error, efficiency and uncertainty

One-step errors use at most 256 deterministic equally spaced windows per validation/test graph, identically selected across variants. Both raw and training-variance-normalized MSE are retained. Each graph's error averages its evaluated nodes/windows; multi-graph comparisons first average within a seed. Population SD across three seeds is descriptive variation, not a confidence interval. Graphs/windows are not counted as independent training replicates.

Recursive rollouts begin from each test trajectory's first observed history and predict ten successive states. Stochastic rollout compares with the actually generated noisy future. Deterministic rollout compares with the known noise-free transition iterated from that same history. The latter isolates error against a deterministic trajectory; due to nonlinearity, it is not the exact stochastic multi-step conditional expectation. The oracle's deterministic error is identically zero by construction; its stochastic rollout is a diagnostic rather than a proven optimal multistep predictor.

Validation thresholds are `oracle + f*(zero-oracle)` for f=0.10,0.25,0.50, specified before the study. A nonpositive reference gap invalidates the threshold. `s_epsilon` is the first observed checkpoint reaching a threshold at a particular sample budget. `n_epsilon` is the smallest tested trajectory count whose curve reaches it; this uses a grid and first crossing, not interpolation, sustained crossing or a guarantee of final convergence. Failure to cross is censored. Speedup ratios require matched seeds and attained thresholds; no infinite improvements are manufactured from failures.

Normalized validation-curve AUC is trapezoidal area over 0..600 divided by 600. It measures the whole observed optimization curve, including initialization; it is distinct from final error and from test performance. Early large errors can dominate this area.

Soft strengths are selected from validation across seeds within the same training condition. Test results never select lambda. Single-seed effects, fixed-strength controls and all candidate results remain available so selection cannot hide contrary cases. An oracle-gap reduction is descriptive and may be unstable if the unbiased model is already close to the oracle.

## Operational bounds and artifacts

Execution uses one CPU process with at most two Torch/OMP/OpenBLAS threads. Dense attention is used even for hard masks; no sparse speedup is implied. Evaluation chunks are at most 32 examples in the full preset. The cooperative 7200-second deadline is checked between training/evaluation operations; external `timeout 7500` imposes a whole-process bound. Partial rows remain labeled partial and are excluded from matched summaries.

JSONL rows are flushed incrementally and summary JSON is replaced atomically. Artifacts include the resolved configuration, clean source revision, environment, process-wide elapsed time and peak RSS, per-model times, initial/final/checkpoint parameter hashes, schedule hashes, graph/weight hashes, all split seeds, sample counts, normalization, per-graph metrics, curves and learned coefficients. Model weights themselves are not saved by this small study; hashes and deterministic seeds support replay, but are not reloadable checkpoints.
