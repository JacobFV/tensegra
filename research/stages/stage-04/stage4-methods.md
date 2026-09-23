# Stage 4 methods: stable binding

Stage 4 isolates whether an explicit identity register stabilizes latent binding during recurrent graph traversal. It extends the Stage 3 task and model without changing earlier experiments. The preregistered [design](stage4-design.md) distinguishes learned binding from initialized routing, and attention-based identity retrieval from explicitly programmed pointer propagation. No interpreter, language model, or autonomous execution scheduler is introduced.

## Task and information boundary

Each example contains a fresh directed graph with three relation types. Each node has exactly one outgoing edge per relation; cycles and repeated visits are allowed. Node identities are independent normalized 16-dimensional Gaussian keys. Opaque integer IDs, keys, topology, categorical values, start node, and memory order are randomized. Node embeddings contain neither values nor outgoing-edge encodings. Memory contains one identity/value item per entity plus four unrelated distractors. There are eight value classes, so chance answer accuracy is 12.5%; correct values do not uniquely identify correct destinations.

The query supplies the start key and a sequence of relation instructions. A shared recurrent block executes once per instruction. Thus deeper testing grants proportionally more computation using trained shared parameters; it tests generalization of repeated binding and traversal, not discovery of how many computation steps to run. Immutable memory persists throughout the trajectory.

Only entity keys, supplied adjacency, opaque IDs, shuffled memory keys/values, start key, and relation instructions enter the model. Gold intermediate nodes, token-to-node labels, clean graph diagnostics, and answers are removed by `model_inputs`. Auxiliary supervision reads gold labels only after the forward pass. Evaluation never teacher-forces an intermediate identity. Training and this stage's depth/size matrix exclude the relation bigram 0→1, following Stage 3's training composition distribution; the matrix is not an additional held-out-composition experiment. Corruption and distractor-count sweeps are not included in this matrix.

## Matcher, read isolation, and write protection

The recurrent residual has width 32: the first 16 coordinates carry identity, with remaining coordinates available for value/content computation. Separate query-side and key-side projections map latents and entity keys into grounding space. Every step recomputes grounding from the current state. A learned null score supplies an unbound category; null has no graph edges.

The raw-dot matcher retains Stage 3's scaled dot product. The cosine matcher normalizes the projected latent and projected entity vectors before matching. Both use temperature 0.05 in this study. Cosine normalization removes direct dependence on projected norms, but does not ensure that the learned transform preserves entity identity.

Three state-update mechanisms separate different hypotheses:

* **Mixed:** the learned attention-value transform, gated residual update, and MLP may all modify identity coordinates.
* **Attention write:** the identity register is overwritten with the head-averaged attention-weighted immutable memory keys. The value transform and MLP still update content, but their proposed identity changes are discarded. This protects the write target while retaining attention as the retrieval mechanism.
* **Pointer write:** compute `p_next = p_query A_relation`, then write `p_next @ entity_keys` directly into the identity register. Null mass writes zero identity; real mass is not renormalized. This bypasses token-attention retrieval for identity and supplies a substantially stronger algorithmic prior. Content still retrieves payloads through attention. Pointer writes continue to use the graph even at zero structural logit strength; that configuration is not an ordinary-attention control.

Read isolation is independent of write protection. Full-state grounding can use content coordinates; identity-only grounding zeros those coordinates before the grounding projections. Identity-only grounding does **not** isolate all computation: the ordinary attention query still depends on the full residual, and therefore content can change attention weights and the resulting attention-based identity write. The interference measurements describe this limited separation rather than claiming a perfectly encapsulated register.

Structural attention uses `P_query A_relation P_keyᵀ` with learned per-head/per-relation strengths, shared across recurrent steps. Strength values in variant names denote initialization, not fixed coefficients. The default is 4. The instruction embedding, content attention, value transform, residual gate, MLP, and readout remain trainable.

## Registered comparisons

The registry contains 23 variants, each run with seeds 0, 1, and 2:

| Family | Registry names | Purpose |
|---|---|---|
| Stage 3 reference | `stage3_soft4`, `stage3_soft8` | Raw-dot, mixed-state aligned initializations |
| Strong controls | `known`, `graph_input_keyed` | Fixed cosine binding, or graph-message injection with keyed retrieval |
| Aligned decomposition | `cosine_mixed`, `cosine_attention_full`, `cosine_attention_identity`, `cosine_pointer_identity`, `dot_attention_identity` | Matching normalization, write protection, read isolation, and explicit pointer transition |
| Cold initialization | `random_cosine_mixed`, `random_cosine_attention`, `random_cosine_pointer` | Random grounding projections with null score initialized to zero |
| Null-prior control | `random_cosine_attention_nullprior` | Same cold attention-write model with null score 0.65 |
| Supervised attention and pointer | `random_cosine_{attention,pointer}_{aux001,aux01,aux1,ground1,nullcycle}` | Ten variants separating supervision weights and mechanisms |

Aligned initialization maps the designated identity coordinates to the supplied entity-key basis. It is an explicit retrieval prior, not an emergent learned binding. Aligned/null-prior variants initialize null scores to 0.65; the cold families initialize them to zero independently of projection randomness. The `known` control uses fixed cosine matching with threshold 0.65 and strength initialized to 8. `graph_input_keyed` uses that same fixed matcher to inject successor messages into memory values and adds an identity-based content-attention bias of 8; its graph is not supplied through structural logits. These strong controls must not be presented as randomly initialized models discovering identity.

## Training and auxiliary losses

All variants receive 400 optimizer updates of 32 freshly generated examples, cycling training depths 1–4 on 16-node graphs. AdamW uses learning rate 0.001, zero weight decay, and gradient clipping at norm 1. Shared seeds pair data, depth schedules, and compatible initial parameters; hashes record the actual pairing. Final checkpoints, not matrix-selected or validation-selected checkpoints, determine reported test performance.

The objective is `task_CE + beta * ground + gamma * null + eta * cycle`. The default three auxiliary weights are zero. Grounding supervision averages three equally weighted roles: pre-update query node, post-update query destination, and real-memory key identity. Each role first averages over steps/examples. Null supervision is binary cross-entropy on memory-key null probability, balancing real-token and distractor groups rather than allowing the larger group to dominate. It does not directly supervise a query null state.

For `aux001`, `aux01`, and `aux1`, beta is respectively 0.01, 0.1, and 1, with gamma=eta=0.1. `ground1` uses (1,0,0); `nullcycle` uses (0,0.1,0.1). Combined with the unsupervised parent, these distinguish answer-only learning, grounding labels, and null/consistency losses, without treating the latter two as individually isolated ablations.

Cycle loss is KL divergence from the **detached predicted graph pushforward** (including residual null mass) to the post-update grounding. It is forward transition/rebinding consistency, not traversal of inverse edges. It cannot independently establish correct binding: uniformly diffuse or otherwise self-consistent wrong states can have low consistency loss. Gold paths do not construct its target.

## Evaluation and stability evidence

The final matrix crosses N={16,32,64,128} with D={4,8,16,32,64}, using 128 fresh examples per cell and evaluation batches no larger than 16. All 69 runs receive all 20 cells. Step-zero evaluation includes shallow validation and the D64/N128 anchor; it does not cover the entire initial matrix. Shallow curves are measured at steps 0,25,50,100,200,400. Initial/final anchor comparisons use paired evaluation samples; curve validation uses a separate sample stream.

Primary outcomes are answer accuracy and complete grounding paths. Complete paths require correct pre- and post-update query argmaxes at **every** hop, including the destination. Final destination accuracy, paths that recover after errors, and correct answers despite imperfect paths remain separate outcomes.

Temporal diagnostics use a canonical sequence comprising the initial pre-state followed by each post-state. Persistence, recovery, onset, first-error position, and survival therefore do not double-count each state as both a previous post-state and a subsequent pre-state. First-error position is zero-based; D+1 denotes no observed error. Conditional rates pool counts before division; missing denominators are null, not zero.

Recorded diagnostics include entropy, top-two margin, correct-node probability, next-node probability, relation-consistent attention, key/null grounding, query and identity norms, cosine/Euclidean identity drift, proposed MLP interference, write overrides, strengths, and distinct gold/predicted nodes. Per-example summaries retain joint associations with failure. Products of marginal step accuracies are descriptive independence references, **not** iid significance tests or proof of an attractor: heterogeneous examples, repeated nodes, reconvergence, and correlated difficulty can also produce persistent errors.

The preregistered adaptive-strength gate requires both task accuracy and complete-path accuracy ≥0.95 at D64/N128 **in each seed** for a fixed-initial-strength configuration. It is a diagnostic gate, not permission to tune on OOD cells. Adaptive crystallization and interpreter work remain separate follow-ups. All main architecture and hyperparameter choices precede matrix inspection.

Runs use the linked machine on CPU with two threads. Raw records retain source/config hashes, training losses, evaluation diagnostics, timing, process high-water RSS, and a small repeated inference benchmark. High-water RSS includes prior allocations in the process and is not isolated per-model memory. Reproduction starts from `configs/stage4.json`; exact source revisions, validation results, measured resources, and empirical conclusions belong in the results report rather than this methods specification.

## Gated adaptive follow-up

After the complete main study, only the aligned pointer family passed the gate. A separate six-run study uses `pointer_fixed4` and `pointer_adaptive4`, each at three seeds, the same 400-step training budget and full matrix. Both arms freeze all base structural strengths at 4 and retain the explicit pointer identity write. The adaptive arm multiplies content-attention structural bias by `(1-p_null) * (1-H(P_query)/log(N+1))`. Including real-entity mass prevents a confident null state from producing strong graph routing. This modulation does not gate the pointer transition, and therefore does not test confidence-controlled acquisition or release of binding itself.

The control is freshly trained with matched initial parameters, data and schedules; the main learned-strength model is not reused as its comparator. The attention-before-pointer eligibility preference was documented while partial main results were visible, a disclosed follow-up selection rule rather than part of the original preregistration. Configuration and source provenance are separate under `results/stage4/adaptive`. Saved-checkpoint auditing verifies the frozen strengths independently from behavioral confidence diagnostics.
