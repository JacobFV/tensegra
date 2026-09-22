# Stage 3 causal leakage and metric-alignment review

Scope: independently inspect data, model, and runner for target/intermediate leakage and grounding diagnostic semantics. This is not a general implementation review.

## Initial data/model inspection

- `TraversalTransformer.forward` reads only `entity_keys`, `adjacency`, `token_keys`, `token_values`, `relations`, `node_ids`, and `start_keys`. It does not access targets, gold path, clean adjacency, start-node index, or token-node index. Full current object memory and graph are available inputs by task definition.
- Entity keys are normalized random Gaussian vectors generated before graph and payload sampling. They contain no payload, topology, relation sequence, or answer. Categorical values are sampled independently. Graph and token slots are exchangeable and token memory is independently shuffled. Node IDs are opaque metadata, not scalar model features.
- Training and evaluation must use fresh seeds. Continuous identity keys generated per batch eliminate fixed-vocabulary/name memorization but do not establish natural-language alias binding.
- `composition='train'` excludes consecutive relation pair `(0,1)` at every position. `heldout` inserts this pair and requires depth >= 2. This is a composition split, not unseen relation types.
- Corruption modifies supplied adjacency only. `targets` and `path_nodes` remain tied to the clean graph. Therefore task accuracy under corruption measures recovery of the original process, not correctness on a changed target problem.
- The exact oracle accepts only adjacency, starts, and relation sequence. Gold paths are not inputs to its computation. Retrieving categorical values using the resulting destination is legitimate algorithmic control, not learned model evidence.

## Architectural claim boundaries

Identity-coordinate initialization, residual copying of retrieved key/value memory, immutable entity memory, and externally scheduled relation instructions are explicit priors. Recurrence gets one supplied relation per computation step. Deep traversal therefore does not demonstrate inferred execution order, unprompted stopping, or natural-language entity binding. A random-initialization comparison is needed to characterize dependence on these priors.

Query grounding in diagnostic item `t` is computed before updating the recurrent state, so its correct reference is `path_nodes[:, t]`. Attention at that item should retrieve `path_nodes[:, t+1]`. Key grounding refers to immutable memory identities, with distractors assigned to null slot N. Correct-next-node attention mass and supplied-graph relation-consistency are distinct, especially under graph corruption.

## Pending runner inspection

The runner was incomplete at initial inspection. Final metric/data split audit will be recorded here when implementation stabilizes.

## Runner findings communicated during implementation

Three metric-alignment issues were found in the initial evaluator and sent to the runner/model authors for correction before experiment launch:

1. Exact-path completion intersected only pre-step groundings, omitting the last destination. This would make depth-one completion trivial despite a wrong hop. Require post-update grounding and include final destination, or explicitly rename the incomplete metric.
2. Relation-attention mass was clean-next-token attention, which is not supplied-graph relation consistency when graph edges are corrupted. Preserve clean-next mass as a separate metric and calculate supplied-edge mass explicitly.
3. Next-node mass was the pre-step current-state grounding probability of the next node. Use a clearly named structural pushforward or post-update distribution instead.

The initial evaluator also assigned different graph seeds to corruption conditions. Paired corruption comparisons should use the same base evaluation seed, since the data generator changes only adjacency under corruption.

Additional interpretation limits: categorical answers admit chance collisions (1/classes); depth 32 on 16 nodes permits revisits rather than guaranteeing 32 distinct entities; and corruption may erase unrecoverable true edges rather than leave redundant inference cues.
