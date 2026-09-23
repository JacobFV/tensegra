# Stage 3 architecture review

Independent review of `runtime_graph.py`, `grounding.py`, `traversal_model.py`, their targeted tests, and the public traversal-data contract. Reviewed during implementation; findings were sent to the owning agents before experiments. No result-affecting mathematical or leakage defect was found in the reviewed core.

## Verified contracts

- `induce_bias` slices away the null column and computes `[B,1,Tq,N] @ [B,R,N,N] @ [B,1,N,Tk]`. The first graph axis is the query/source entity, the second the key/destination. The operation supports different query/key sequence lengths and independent probability distributions. Neither null mass nor padded nodes acquire implicit edges.
- Query and key roles have four independent latent/entity projections and separate null logits. Temperature remains positive. The module is stateless; the caller decides when to recompute it. Tensor tests cover asymmetric directed edges, permutations, padding, empty graphs, float64 gradient checking, and lower-precision accumulation.
- The recurrent model re-grounds its current query residual at every instruction in learned-soft mode. It does not read target labels, token-node diagnostic mappings, gold path nodes, or clean adjacency. The graph carries random identity keys and adjacency, while payloads appear only in shuffled memory.
- Zero structural strength preserves ordinary attention in the soft control. The test checks exact model-output equality. Graph-slot and memory-token permutations preserve learned output within floating-point tolerance.
- Projection-period variants cycle trained projections; they do not introduce untrained layers at deeper evaluation. A period of four is only fully trained when training includes four-step paths.

## Review findings and disposition

1. Ordinary-attention mode initially computed and discarded grounding and induced bias. This would understate the incremental inference cost of structural grounding. The implementation now skips these computations when `mode='none'` and diagnostics are disabled. Timings must use this diagnostic-free forward path. Diagnostic timings include deliberate probe work and must be labeled separately.
2. Constructor validation originally evaluated `width % heads` before checking positive heads. The owning agent moved positive-dimension validation first, avoiding an accidental division-by-zero exception.
3. A direct graph traversal algorithm proves the data oracle, but does not by itself verify one-hot induced-attention routing. The core owner is adding an independent one-hot `Pq A Pk^T` hard-attention oracle check. Its result is an algebraic mechanism control, not evidence of learned generalization.

## Interpretation constraints

The model is a single-query recurrent cross-attention machine over immutable memory, with externally scheduled relation instructions and a copy-friendly residual update. It is not a full evolving self-attention token stack, inferred execution scheduler, or programming-language interpreter. Depth-32 evaluation receives 32 recurrent steps.

The default grounding starts with an explicit shared identity-coordinate prior, a low temperature, and substantial positive structural strength. Random identity keys prevent fixed-name memorization, but successful binding under this initialization does not establish discovery of arbitrary semantic entity identity. Random-projection and weaker-bias ablations are needed to characterize that dependence.

The `known` control uses fixed normalized cosine matching; learned grounding uses trainable dot products and therefore has different state-norm sensitivity. `graph_input` is a strong message-passing baseline with fixed cosine-grounded edges, ordinary attention keys, and graph-transformed values. It is not raw graph serialization. These asymmetric priors should remain visible in reports.

The hard control applies nondifferentiable argmax grounding. Without a separate auxiliary objective, task gradients do not train grounding projections. Content, value, residual, and readout parameters can still train. When no legal destination is mapped, it falls back to ordinary attention. It is consequently an initialized hard-routing control, not a learned discrete-grounding estimator.

`pq` records the actual pre-update distribution used to route the current step. The later `pq_after` probe re-grounds the resulting state for diagnosis; in frozen mode that probe is not the distribution subsequently used for routing. Exact routing-trajectory metrics must distinguish these quantities. Correct final class prediction alone does not certify exact path completion, especially because different destination entities can share the same categorical payload.

## Validation evidence

An independent invocation of the worker's remote CPU snapshot passed **27 targeted core/model tests in 0.78 seconds**, using two OpenMP/OpenBLAS threads. This is a targeted snapshot check, not a claim that the final integrated tree has already passed its full suite. Final source-pinned integration validation remains the coordinating agent's responsibility.
