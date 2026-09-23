# Stage 2 core review — Tasks 1 and 2

Reviewed `stage2-design.md`, Tasks 1–2 of `plan-02.md`, the data/model implementation notes, `data-model-review.diff`, and the current implementations of the shared data, attention, and model layers. Per the review request, the already reported 42-test data run and 38-test model run were not repeated.

## Spec-compliance verdict: PASS

No blocking or major compliance defect was found in Tasks 1–2.

- **Directed graph semantics:** consistent. `Dynamics.weights[i,j]` means target row `i` reads source column `j`; `dynamics_step` implements this as `state @ weights.T`. The same row-query/column-key convention is used by attention bias/masks and by `adjacency @ x` in the graph-input baseline.
- **Corruption fairness:** `supply_graph` samples directed nonself supports, bases every requested count on the original true nonself edge count, uses original nonedges for additions, and therefore cannot restore a dropped edge in mixed corruption. Reusing `(truth, corruption, fraction, seed)` gives the identical corruption across variants. Additions saturate at the available original nonedges, and `graph_quality` records the realized result.
- **Input immutability:** graph corruption operates on a clone, including the CPU-input case where `detach().cpu()` may alias the source. Model paths do not mutate `x`, `graph`, or `weights`.
- **Signed contraction:** each nonempty signed row is normalized to absolute mass below `0.8`; empty rows remain zero. Together with the `0.5 * state + 0.5 * tanh(...)` transition this gives the documented infinity-norm bound of at most `0.9`. Signed and uniform mechanisms consume the same graph draw before signed magnitudes/signs are sampled, so topology is paired.
- **Zero-alpha behavior:** learned and typed variants add an exactly zero floating bias at initialization, so their forward values are exactly those of unbiased attention under matching shared parameters. Alpha remains in the score computation graph, allowing gradients at zero. Coefficients are unconstrained and represented per layer/head (and per relation for typed).
- **Graph-input baseline:** the supplied adjacency is used only to compute normalized neighbor histories before the shared transformer blocks; attention receives neither graph bias nor mask. The added bias-free projection plus the inherited input projection is algebraically a projection of concatenated self/neighbor histories. The implementation and notes correctly identify it as graph-conditioned message passing rather than graph serialization.
- **Batch and size equivariance:** `[N,N]`, `[1,N,N]`, and `[B,N,N]` relations are handled without mixing examples. All operations share parameters over nodes and commute with simultaneous node/history/graph permutation. No learned parameter depends on node count, so the same model can run at transfer sizes.

## Code-quality verdict: PASS WITH MINOR FOLLOW-UPS

The implementation is compact and reuses the pilot parameter names, which supports paired initialization and preserves the pilot API. Validation is generally clear, and the tests described in the supplied evidence directly exercise the main scientific contracts.

### Findings

1. **Low — `deterministic_future` has incomplete type/device validation.** [`src/topoformer/study_data.py:109`](../../../src/topoformer/study_data.py#L109)

   A non-tensor `history` raises an incidental `AttributeError`, and a history on a different device or with an incompatible dtype from `system.weights` fails inside matrix multiplication. The study is specified for CPU float32 execution, so this does not affect the planned runs. For a reusable public interface, validate that `history` is a floating tensor and either require matching device/dtype explicitly or create a local system tensor on the history device/dtype.

2. **Low — `degree=True` is accepted as numeric degree 1.** [`src/topoformer/study_data.py:18`](../../../src/topoformer/study_data.py#L18)

   Python booleans satisfy `isinstance(value, (int, float))`. Reject `bool` explicitly, as `_positive_integer` already does, so malformed configuration cannot silently select a different graph distribution.

3. **Low — signed normalization and support construction depend on private pilot helpers.** [`src/topoformer/study_data.py:7`](../../../src/topoformer/study_data.py#L7)

   Importing `_positive_integer` and `_weights_from_graph` couples the new public study module to private implementation details. This is not a current correctness issue, but either promote those helpers to shared public utilities or keep small local validation/normalization helpers if the pilot module is expected to evolve independently.

4. **Low — typed added edges carry no relation signal when original process weights are passed.** [`src/topoformer/study_model.py:71`](../../../src/topoformer/study_model.py#L71)

   The typed bias has only positive- and negative-weight channels. A supplied spurious edge whose process weight is zero is indistinguishable from an absent edge. The current Stage 2 corruption comparison is explicitly soft/hard, so this is not a Task 1–2 compliance failure. If later runner work combines typed models with add/mixed corruptions, define the coefficient assigned to a spurious supplied edge or add an adjacency-presence channel, and record that choice in the artifact configuration.

## Review conclusion

Tasks 1 and 2 are suitable to build the runner on. None of the findings changes the directed semantics, pairing, contraction claim, zero-alpha equivalence, information-baseline interpretation, or graph/size equivariance required by the Stage 2 design.
