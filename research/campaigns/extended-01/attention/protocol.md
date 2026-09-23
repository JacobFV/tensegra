# A01 — explicit routing, equivalent graph information

Status: development protocol; no confirmation selection yet. Workspace 1024.

## Question and supplied contract
Can additive adjacency compatibility bias improve finite-budget acquisition or transfer relative to strong graph-context and message-passing interfaces? This is an explicit-node-grounding routing experiment, not semantic induction or learned program planning.

A graph has three typed directed permutation relations. Each relation has exactly one successor and predecessor per node. Independent uniformly random payload classes (16) are attached to all nodes. Keys are fresh normalized Gaussian vectors; node ordering is random and has no stable identity meaning. An ordered relation sequence and start node are public. Terminal values are not marked. Additional nodes are ordinary distractors.

Every method executes D shared recurrent steps. The supplied reverse relation schedule computes values for every possible starting node, then reads the public start. This backward dynamic-programming schedule is a programmed algorithmic prior shared by every method. No gold intermediate active node or future payload enters forward inference. Explicit grounding is identity at each node row, so B=A. Per-step all-node value targets are privileged supervision, common to all arms. Exact pointer execution is only a programmed reference.

## Arms
All arms share payload embedding, width-1024 learned update and output classifier. Immutable 64-dimensional keys provide row identity. Routing is:
- `soft`: content QK/sqrt(d) + trainable typed λ A, initialized λ=4.
- `hard`: content scores masked to selected adjacency. Single-successor clean graphs make routing exact, but encoding/update/readout remain learned.
- `message`: adjacency-weighted aggregation followed by the same learned update. This is an intentionally strong message-passing baseline, including supplied relation schedule.
- `context`: edge information supplied as each source's successor key; learned normalized query/key matching retrieves payload states. No structural logit bias. The encoding is a lossless graph representation when keys are distinct; deriving successor keys from A is graph serialization, not future answer computation.
- `none`: no graph information; informational ablation only.

Content Q/K dimensions, allocated/active parameters, memory tokens, dense attention work and measured latency are reported. Equal parameter count is not equal FLOPs. Message routing need not perform unused content QK work. All approaches use the same learned local update. Structural scores are dense; no sparse-compute claim.

## Development and promotion
Development seed 101, fresh graph batches, N16 and depth1–4; three relations, exclude adjacent relation pair (2,2) during training to reserve a composition condition. AdamW, full direct intermediate supervision. Profile first; screen 500 updates/arm with batch16, learning rate3e-4, checkpoints0/100/250/500. Development validation256 fresh graphs/cell: N16,D4;N32,D4;N16,D8;N32,D8; forced unseen (2,2) composition. No test claims from screening. If losses improve, a predeclared extension up to2000 updates is allowed under campaign budget; otherwise diagnose before extension.

Confirmation only after screening: strongest context/MP plus soft and hard, three paired fresh initialization seeds, fixed exposure chosen from development, >=1024 fresh graphs/cell, source/config frozen, no confirmation-driven selection. Include no-graph context if affordable. Holdout sizes32/64 and depths8/16; joint shift64/16. Degree-preserving wrong topology, identity correspondence permutation,25% missing/spurious edges, zero λ and node permutation controls. Corruption targets remain clean-graph answers; null rows receive declared self fallback, a semantics alteration disclosed in corruption results.

Metrics: start value accuracy, all-node accuracy, complete suffix-value trajectory, structural attention mass, routing argmax correctness. Payload collisions mean value trajectory is not exact identity traversal; independently propagate attention-selected indices to score the induced full pointer path. Training curves/exposures, actual examples, parameter count, wall time, CUDA memory, throughput and time per correct solve. Seeds share evaluation events; report event uncertainty separately from seed variability.

## Literature / mechanisms
[Universal Transformers](https://arxiv.org/html/1807.03819v3), methods §2.1: shared recurrent self-attention and local transformations; this experiment uses supplied fixed D, no ACT. [Graphormer](https://arxiv.org/html/2106.05234v5), structural encoding methods: additive pairwise structural score bias; here direct typed adjacency, no shortest-path encoding. [MPNN](https://arxiv.org/html/1704.01212v2), message/update formalism: relation-conditioned neighbor aggregation with shared learned update. Formulations borrowed; no code copied and no novelty claim.

## Initial implementation decisions
- Context Q/K start at identity, cosine temperature1/8 and learn thereafter. This known-key identity prior makes context a competitive information-matched baseline.
- Hard and message routing are algebraically identical on clean single-successor graphs. This is mechanically tested; they are not independent evidence for two learned algorithms. Multiple-neighbor corruption separates learned hard attention from uniform message averaging.
- Context and message both normalize outgoing adjacency to unit mass before encoding/aggregation. Soft bias uses unnormalized binary adjacency; hard masks all supplied neighbors. These are interface differences, not different raw graph information.
- `zero_strength` is a frozen-model evaluation intervention. The separately trained `none` arm receives no graph, including no neighbor keys.
- Fullpath attention argmax is a routing diagnostic, not proof that those argmax edges causally explain the neural answer. Value trajectories have collisions; identity routes do not.
- Any exposure extension after500 is a separately registered exploration decision; the phrase “losses improve” above is not an automatic retrospective budget extension.
- Reviewer correction before profile: multi-neighbor context now queries each supplied neighbor address separately and averages their read distributions. It never compresses several addresses into one mean key. This is equivalent to original single-neighbor context on clean graphs and preserves the address list under spurious-edge corruption.
- Generated graph examples are counted; exact semantic graph deduplication is not performed. Distinct random presentations are not labeled empirically deduplicated canonical constructions.
- Profile acquired N16/D4 value accuracy by12updates. Before development, add checkpoints10/25/50 to0/100/250/500 so learning-curve resolution can measure early acquisition. These are development metrics only; fixed500 remains primary endpoint.
