# Stage 3 methods: latent grounding into runtime graphs

## Study question and claim boundary

Stage 3 tests whether a recurrent neural state can remain bound to runtime entities while following externally supplied typed relations through a graph. It isolates keyed entity grounding from language understanding. Each example supplies a fresh set of continuous random identity keys, an immutable shuffled key-value memory, a directed typed graph, a start key, and a sequence of relation instructions. The model predicts the categorical payload at the instructed destination.

One recurrent update is invoked for every supplied relation instruction. Evaluation at greater depth therefore gives the model proportionally more computation and an explicit execution schedule. The experiment does not test discovery of a program counter, autonomous stopping, latent plan induction, lexical name resolution, or natural-language interpretation. The primary model also begins with an identity-coordinate alignment prior. The experiment is designed to characterize when this binding-and-traversal mechanism works and how it fails, rather than to presume successful latent grounding.

The preregistered full configuration is [`configs/stage3.json`](../configs/stage3.json). It contains 13 variants, 3 seeds, and 400 optimizer steps per run, for 39 trained runs. The feasibility pilot contains six variants for one seed and 100 steps. A separately labeled keyed supplement contains three variants, three seeds, and 400 steps, adding nine runs. The main launch source is `2adc87c`; its exact-source gate passed 206 tests in 2.16 seconds. The keyed source is `27ca0a2`, gated by 211 tests in 2.18 seconds. Results are not included here; this document fixes their interpretation.

## Runtime graph task

### Entities, identity keys, and payload memory

Each example independently samples a graph with 16 nodes for training. Every node receives a normalized Gaussian identity key in 16 dimensions and an opaque integer metadata ID. The integer IDs are unique within a graph but are never used as scalar model features. Identity keys are sampled independently of graph topology and categorical payloads. Payloads are sampled independently from eight classes.

The memory contains one token for every entity plus four unrelated distractor tokens during training. An entity token concatenates its continuous identity key with a one-hot categorical payload and is padded to model width 32. Distractors receive independent normalized Gaussian keys and independent categorical values. Entity and distractor tokens are shuffled together separately in every example. There is no canonical node or token order, no positional clue linking tensor slot to graph slot, and no payload or topological preprocessing in the identity key.

Memory is immutable during traversal. The model updates a single query state, while token keys, values, entity keys, and adjacency remain fixed. The initial query state is the start entity's identity key padded to width 32. Intermediate gold nodes, destination identity, target payload, clean graph under corruption, and token-to-node diagnostic labels never enter the forward pass.

The use of fresh continuous keys prevents memorization of a fixed finite name vocabulary. It does not test lexical names, aliases, tokenization, or grounding from semantic descriptions. “Identity” in this study means equality or learned alignment in a continuous random key space.

### Random typed functions

There are three relation types. For each graph, relation, and source node, the generator samples exactly one destination uniformly from all nodes. Each typed adjacency is therefore a total directed function. Self-edges and cycles are permitted. Repeated nodes in a gold path are valid rather than deduplicated; the evaluation records the mean number of distinct visited nodes to disclose how often long instruction sequences revisit earlier states.

An instruction sequence determines exactly one path and payload. Training cycles through depths 1, 2, 3, and 4 in optimizer-step order. The pair of consecutive relation types `(0, 1)` is excluded at every position in training sequences. The held-out-composition condition inserts that pair at a random valid position. This tests a new consecutive composition of familiar relation types, not unseen relation symbols.

The clean answer and path are computed by exact functional traversal. Under corruption, a Bernoulli fraction of source/relation rows is independently selected and rewired to a different destination, preserving one destination per row. The supplied adjacency changes, while the clean path and payload target remain fixed. Thus corrupted-graph accuracy measures recovery of the original clean task from incomplete or contradictory structural information. At high corruption there may be insufficient public information to identify the clean answer; failure is not automatically a grounding or optimization defect. Realized corruption varies around the requested fraction, and a one-node graph cannot be rewired.

### Public model inputs and protected diagnostics

The runner passes an explicit allowlist to every model invocation:

- entity identity keys;
- supplied typed adjacency;
- opaque node IDs used by the runtime graph container;
- shuffled token keys and categorical values;
- start identity key; and
- relation instruction sequence.

Targets, token-to-node assignments, start-node indices, full path nodes, and clean adjacency are labels or diagnostics only. Training uses final-answer cross-entropy and no intermediate grounding supervision. Gold paths are read only during evaluation. A dedicated leakage review and tests verify this boundary.

## Grounding and induced graph bias

### Learned soft grounding

For a latent collection \(H\), entity keys \(E\), and role \(r\in\{q,k\}\), learned grounding uses separate bias-free projections for latent states and entity keys:

\[
S_r(H,E)=\frac{(H W^{latent}_r)(E W^{entity}_r)^\top}{\sqrt{d_g}\,\tau}.
\]

A separately learned scalar null logit is appended before the softmax. The resulting query and key distributions, \(P_q\) and \(P_k\), have \(N+1\) columns; the last is an unbound/null slot. Invalid padded nodes are masked. Null mass contributes no graph edge and the real-node mass is not renormalized.

For relation adjacency \(A_r\), the induced token-space route score is

\[
B_r=P_q A_r P_k^\top.
\]

The contraction preserves directedness, supports independent query and key assignments, and remains differentiable through both grounding roles. In half precision the contraction accumulates in float32; float64 inputs remain float64.

The default temperature is 0.05. The default learned models use one grounding module at every recurrent step. Query grounding is recomputed from the current query state, and key grounding is recomputed from the same immutable memory. Query and key projections are independent after initialization. The `frozen` ablation instead reuses both initial distributions for every step.

### Identity-coordinate initialization

The primary learned models contain an explicit keyed-binding prior. The first 16 query-state and memory coordinates contain the random identity key. Query/key latent projections begin as \(\sqrt{16}I\) on those coordinates, entity projections begin as \(I\), and both null logits begin at 0.65. Because entity keys are unit-normalized, the square-root scale cancels the kernel's \(1/\sqrt{d_g}\) factor at initialization. The initial real-entity score is consequently an identity-key dot product competing with a null threshold of 0.65, followed by temperature 0.05.

The `random_init` control removes this alignment: all four projections use ordinary random initialization and null logits start at zero. Thus it removes both the coordinate-matching prior and the initial null-threshold calibration; its null behavior is not an isolated test of projection initialization. It retains the same task, loss, architecture, and structural strength. A gap between `soft` and `random_init` measures dependence on the combined initialization prior; it cannot be described as evidence that the primary model discovered arbitrary identity binding unaided.

### Known cosine grounding is a different mechanism

The `known` and `graph_input` controls use fixed cosine matching between the identity prefix of the state or memory and normalized entity keys. They append a fixed cosine null score of 0.65 and apply the same fixed temperature 0.05. This mapping has no learned grounding projections and is invariant to the norm of the identity prefix.

Learned grounding instead uses raw dot products after independently trainable projections. It is sensitive to projected norms as well as direction. Even when its initialization approximates the same identity alignment, `known` is not simply the learned grounder with frozen weights. Comparisons must retain this distinction.

### Structural strength and domination risk

The default structural coefficient is initialized to 8 for every attention head and relation type. The induced route score is multiplied by that coefficient and added to ordinary content-attention logits. With temperature 0.05 and identity-aligned initialization, this can create a near-hard routing prior early in training. A successful default run would establish feasibility under a strong prior; it would not by itself show that weak structural information was learned from scratch.

The risk is visible in both directions. A large coefficient can dominate content scores when grounding is correct, and it can force confident wrong retrieval when grounding or supplied adjacency is wrong. Null probability and diffuse groundings attenuate the induced bias because null mass contributes no edge. The experiment records final strengths, grounding entropy, structural next-node mass, and corruption behavior so that answer accuracy is not interpreted without this mechanism context.

## Recurrent traversal model

The model uses width 32, four attention heads, and one shared recurrent attention/update block. At step \(t\), the externally supplied relation embedding is added to the normalized current query for content attention. The relation selects a typed adjacency slice, grounding induces the token-space route bias, and ordinary multihead attention retrieves from immutable memory. A learned sigmoid gate interpolates the query state toward the retrieved identity/payload vector:

\[
h_{t+1}=h_t+\sigma(g)(v_t-h_t)+\operatorname{MLP}(h_t+\sigma(g)(v_t-h_t)).
\]

The gate starts from scalar logit 2. The value projection and final MLP projection begin at zero. After the last instruction, a linear readout predicts one of eight payload classes. The block's ordinary query, key, value, update, MLP, and readout parameters are shared over all relation steps. The model has no gold intermediate-state loss.

This residual replacement mechanism is intended to carry the retrieved entity identity forward so it can be grounded again at the next step. Failures can therefore arise from at least four distinct sources: incorrect initial binding, drift after state update, failure to use the supplied relation, or incorrect payload readout. The evaluation metrics separate these cases.

## Experimental variants

All 13 variants receive the same public task inputs and relation schedule.

| Variant | Grounding and structural mechanism |
|---|---|
| `soft` | Primary identity-initialized learned soft grounding; temperature 0.05; typed per-head/per-relation strengths initialized to 8. |
| `soft_strength4` | Same as `soft`, with structural strengths initialized to 4; tests sensitivity to the potentially dominating default coefficient. |
| `known` | Fixed cosine identity matching with null threshold 0.65; learned typed structural strengths still begin at 8. This is a privileged known-alignment control. |
| `frozen` | Learned identity-initialized query/key distributions are computed once and reused for all routing steps; post-update grounding is recomputed only as a diagnostic probe. |
| `permuted` | Cyclically shifts the real-node query-grounding coordinates while leaving null and key grounding intact, creating a deliberately misaligned structural route. |
| `none` | Ordinary attention receives no graph bias. Fast inference skips unused grounding; diagnostic evaluation may compute grounding probes without using them for routing. |
| `graph_input` | Uses fixed cosine token-to-entity matching, pushes token memories through the selected graph, normalizes by induced outgoing mass, and presents the resulting neighbor messages as values to ordinary unbiased attention. It is a strong graph-conditioned message-passing baseline, not serialized edge tokens. |
| `hard` | Argmaxes learned identity-initialized query/key grounding, constructs a Boolean legal-route mask, and restricts attention to legal destinations. If no mapped legal destination exists, it falls back to ordinary attention to avoid an invalid row. Argmax routing is non-differentiable, so answer loss does not train the grounding projections. |
| `random_init` | Same learned soft mechanism as `soft`, but removes identity-coordinate initialization and starts projection weights randomly and null logits at zero. |
| `learned_temperature` | Same as `soft`, with each grounding temperature learned through a bounded log parameter rather than fixed at 0.05. |
| `shared_strength` | Same as `soft`, but shares one coefficient across heads for each relation instead of learning separate head coefficients. |
| `untyped` | Replaces the instructed typed relation with the union/max over relation adjacencies and uses relation-agnostic strengths. The instruction embedding remains available to content attention. |
| `period4` | Uses four separately trained grounding/strength slots for training depths 1–4 and cycles them at deeper steps. Ordinary computation parameters are copied from a canonical same-seed initialization. This tests per-step projections without introducing untrained parameters at extrapolation depths. |

### Supplemental keyed intervention

The keyed supplement adds a fixed content-attention term \(8\cos(h_t^{id},k_j)\), computed between the current state's identity prefix and each immutable token key. `none_keyed` uses this content term without structural routing. `soft_keyed` combines it with learned soft structural routing initialized at strength 16. `graph_input_keyed` applies it after fixed-cosine graph-message construction; the configured structural coefficient is inactive in graph-input mode because that mode supplies the graph through values rather than structural logits.

The intervention does not make the mechanisms identical. For direct soft routing, cosine content favors the current-identity token while graph bias favors the successor token. For graph input, the successor message is stored in the current/source token value, so both graph construction and content retrieval favor the source-token position. The supplement diagnoses this interaction and sensitivity to a stronger structural coefficient; it is not folded into the original 13-variant grid or described as an all-else-equal ranking of graph encodings.

The direct functional oracle and exact one-hot attention oracle are evaluation controls rather than trained variants. The direct oracle indexes the supplied adjacency exactly. The attention oracle binds keys by exact tensor equality and traverses shuffled memory with one-hot legal attention. Both are privileged algorithms and test task/routing algebra; their success is not neural grounding evidence. Under corrupted supplied adjacency they follow the corrupted structure, while gold targets remain clean, exposing the task's information limit.

## Training design and pairing

Each variant is trained for 400 AdamW steps with learning rate 0.001, weight decay zero, batch size 32, and gradient norm clipped to 1. Training therefore presents 12,800 freshly generated examples per run. Checkpoints are evaluated at steps 0, 25, 50, 100, 200, and 400. Checkpoint validation uses 128 examples at depth 4, 16 nodes, four distractors, clean adjacency, and the training composition split.

Training graph, identity key, payload, token order, distractor, start, and instruction randomness is regenerated from a deterministic seed at every optimizer step. Seeds 0, 1, and 2 define three training replicates. All variants within a seed receive the exact same batch seeds, depths, and data hashes. Training, checkpoint validation, final evaluation, and inference timing occupy disjoint seed namespaces. Evaluation datasets and schedules are also identical across variants within a seed.

Model construction begins from the same Torch seed. Ordinary parameters that exist with matching names and shapes are paired. Architectural variants intentionally differ in some parameters: `random_init` changes grounding initialization; `known` and `graph_input` do not use learned projections for routing; `learned_temperature`, `shared_strength`, `untyped`, and `period4` change parameterization. Per-parameter initial hashes disclose exactly which tensors match. For `period4`, ordinary recurrent parameters are explicitly copied from a canonical same-seed model so module-construction order does not break the intended pairing.

The three seeds provide descriptive replicates. Graphs and all example contents vary within and across seeds; batches or graph instances are not treated as independent experimental replicates. With three seeds, results will report individual paired effects and descriptive variation without formal significance claims.

## Evaluation conditions

Every final condition contains 128 examples evaluated in batches of 32. The grid changes one factor at a time from the training reference of depth 4, 16 nodes, four distractors, training composition, and clean adjacency:

- depth 1, 2, 4, 8, 16, or 32 at 16 nodes;
- node count 32 or 64 at depth 4 (the 16-node reference is already present);
- held-out consecutive relation composition `(0,1)` at depth 4 and 16 nodes;
- one declared joint extrapolation at depth 32 and 64 nodes;
- supplied-edge corruption 0.10, 0.25, or 0.50 at depth 4 and 16 nodes; and
- zero or 16 distractors at depth 4 and 16 nodes.

This produces 15 unique conditions per run. Depth and size curves are not a full Cartesian sweep; only the declared depth-32/size-64 condition changes both. Graphs, keys, paths, and token order are fresh in evaluation and common across variants through shared condition seed namespaces.

Depths beyond four invoke the recurrent block more times. The default projection period is one, so the same learned grounder is reused at every step. `period4` cycles its four trained slots modulo four at depth 8–32. No variant invokes previously untrained layer parameters during depth extrapolation.

## Metric semantics

### Task and privileged-oracle metrics

`task_accuracy` is final categorical payload accuracy and is the primary outcome. `oracle_task_accuracy` executes exact indexing on the supplied graph, then reads the destination payload. `attention_oracle_task_accuracy` performs exact identity binding and one-hot structural attention through shuffled memory. Their corresponding exact-path metrics require agreement at every node in the path. Under corruption, oracle disagreement with clean targets quantifies information loss in the supplied graph rather than learned-model error.

### Grounding trajectory metrics

Diagnostic record \(t\) is created before recurrent update \(t\). Accordingly:

- `grounding_accuracy` compares query-grounding argmax with clean `path_nodes[t]`;
- `grounding_entropy` is entropy over real entities plus null;
- `exact_pre_step_grounding` requires every pre-update grounding in the instruction sequence to be correct;
- `exact_path_completion` requires every pre-update grounding and every post-update probe, including the final destination, to be correct; and
- `key_grounding_accuracy` measures entity-token key grounding, averaged over non-distractor memory tokens.

The post-update `pq_after` probe uses the current grounding slot after the state update. In `frozen`, it is observational: routing still uses the distributions frozen at the first step. In `permuted`, the same wrong coordinate permutation is applied to the probe. These probes are excluded from plain inference timing.

### Attention and graph-use metrics

`exact_attention_path_completion` requires the mean-over-heads attention argmax to select a token belonging to the clean next node at every step. `clean_next_attention_mass` is total attention probability on tokens for that clean next node.

Those next-token attention metrics describe retrieval-style variants but are not mechanism-comparable for `graph_input`. Graph-input first places each supplied successor's message into the corresponding source token's value, so attention to the current/source token can correctly retrieve the next entity. Its next-node token attention can therefore be low even when its message-passing operation is correct. Final task accuracy and post-update state grounding remain applicable; graph-input attention mass must be interpreted according to its source-token message placement.

`pre_step_next_node_mass` is query-grounding probability placed directly on the next node before applying the graph. It is not expected to be high for a correctly grounded current state and is retained as a diagnostic against metric confusion. `structural_next_node_mass` instead pushes current query grounding through the supplied relation adjacency and measures mass on the clean next node. `relation_attention_mass` measures attention on the successor of the clean current node under the supplied graph. Under clean adjacency the structural and clean targets coincide; under corruption they can differ. This separation distinguishes following supplied structure from recovering the original clean answer.

### Null, distractor, and recurrent diagnostics

`null_mass` includes the query grounder's explicit null probability. `distractor_null_accuracy` is the fraction of unrelated memory tokens whose key grounding argmax is null and is reported as not applicable when a condition contains no distractors. Null capacity does not guarantee learned rejection, so both mass and accuracy are measured. In variants that do not use learned grounding for routing, such as `none`, learned-grounder null values collected during diagnostic evaluation are probes only and must not be described as part of the variant's operational mechanism.

All grounding, entropy, route-mass, attention-mass, and null metrics are also stored by recurrent step. Each run retains one complete first-example query-grounding distribution trace per condition, selected deterministically rather than by success. Mean distinct path nodes discloses cycles/revisits. Final per-head/per-relation structural coefficients and grounding temperatures are stored to diagnose domination, collapse, or compensating scale changes.

## Resource measurement and artifacts

Execution uses one CPU process with Torch, OpenMP, and BLAS threads limited to two. The configured cooperative wall limit is 7,200 seconds per run and is checked between optimizer steps. Evaluation batch size is capped at 32 in the full configuration.

Training time measures forward/backward optimization only. Total run time includes checkpoint validation, final diagnostics, both privileged oracles, and inference benchmarking. Plain inference timing uses diagnostics disabled, three warm-up calls, and ten measured repeats at batch size 32 and depth 4. This avoids charging unused grounding probes to `none` and avoids conflating diagnostic collection with deployed inference.

Peak RSS is the process-wide cumulative high-water mark reported by the operating system. Because all variants execute sequentially in one process, it is not an isolated per-model allocation and should not be subtracted or treated as precise incremental memory. Parameter counts, device, thread count, Torch version, training examples, timing, strengths, temperatures, initial/final hashes, schedule hash, and evaluation data hashes are retained.

Each completed run is appended and flushed to JSONL. Resume requires the exact resolved configuration hash and the same source-file hashes; duplicate run keys or mismatched provenance abort. Nonfinite loss or metrics abort rather than yielding a partial successful row. Analysis requires a complete variant-by-condition-by-seed grid, common provenance, matched schedule/evaluation hashes, and finite metrics before paired summaries are produced.

The final code, analyzer, render, and artifact state at source `85e6017` passed 215 tests in 2.17 seconds. The committed audit covers the six-run pilot, 39-run main family, nine-run keyed supplement, and 50 hashed artifacts.

## Interpretation limits

The experiment can distinguish answer failure from initial binding failure, recurrent drift, incorrect attention, relation misuse, distractor binding, and corrupted-information limits. It can test whether a strongly initialized differentiable binding survives longer recurrence and larger runtime graphs. It cannot establish a scaling law from six depths and three node sizes, and it cannot attribute every answer failure to grounding when readout and recurrent dynamics are also learned.

Success of `soft` must be read alongside `random_init`, `known`, `hard`, `graph_input`, `none`, and `permuted`. The identity-coordinate prior, temperature 0.05, strength 8, immutable keyed memory, explicit relation schedule, and residual identity-copy mechanism make this a routing feasibility study. Hard routing is not an end-to-end learned grounder because argmax blocks its grounding gradients. Graph-input and known grounding receive fixed cosine alignment. Corruption may remove the information required for the clean target. Long random-function paths can cycle, so depth does not guarantee the same number of distinct entities.

No result should be generalized to lexical names, natural-language entities, pretrained language models, or embodied systems. Those settings require learned aliases, semantic ambiguity, variable mentions, and distribution shifts that continuous random identity keys deliberately remove.
