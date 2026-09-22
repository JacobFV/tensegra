# Stage 4 binding model

`BindingTransformer` subclasses the existing traversal transformer and retains its parameter names and initialization, while independently controlling null logits. Stage 1–3 source files are unchanged. Forward uses only the seven public inputs, never target IDs, token-node correspondences or gold trajectories.

## Architectural contracts

- `matcher=dot` delegates to existing independent query/key and entity projections. `cosine` normalizes projected latents and projected entities before comparison; null is a learned scalar on the same pre-temperature cosine scale. Temperature remains independently fixed or learned through the original grounder.
- `identity_only=True` zeros content coordinates **at the grounding interface**. Content still contributes to ordinary attention queries and learned content updates; this does not claim full causal isolation of identity from content.
- `identity_update=mixed` retains Stage 3 gated retrieval plus MLP. With dot/full grounding and shared state dict it is bit-exact across all existing modes.
- `attention` writes identity from the mean of actual head attention distributions applied to raw immutable token keys. Learned V and MLP proposals cannot directly overwrite identity. In graph-input mode the keys written are the successor-message identity coordinates, matching that separate graph interface.
- `pointer` writes `(P_query_real A_selected) @ entity_keys`, without renormalizing away null mass. This is an explicitly programmed transition, still active with zero structural logit strength and mode `none`. Hard mode affects token attention; pointer identity remains a soft graph expectation. This control must never be described as an ordinary no-graph model.
- Zero graph strength matches ordinary attention exactly for mixed/attention writes. Pointer mode is deliberately excluded from that equivalence.
- `null_init` is independent of projection initialization. A zero cosine identity favors null over all real nodes. Pointer null contributes zero identity rather than choosing a real entity. Cosine rematching is scale invariant, so a small nonzero expected identity can subsequently rebind; null is not an absorbing state imposed by the architecture.

All original modes remain supported, including frozen initial grounding, known cosine, permuted query grounding, hard routing and graph-input. Shared or cyclic projections preserve the existing recurrent evaluation contract.

## Diagnostics

Each step returns differentiable `pq`, `pq_after` [B,1,N+1], `pk` [B,T,N+1], `attention` [B,H,1,T], `pnext` [B,1,N], `state_before/state_after` [B,1,width], `identity_write/identity_proposal/mlp_identity_delta` [B,1,key_dim], `projected_query_norm` [B,1], and original bias/strength/state fields. `pnext` contains only real-node mass; a functional adjacency preserves incoming real mass. Missing edges may lose additional mass, which diagnostics can attribute to the null complement. No diagnostic tensors are detached inside forward.

## Verification

On gb10-direct with two CPU threads and the existing environment: 12 focused tests passed (0.82 s initial run), covering exact Stage 3 compatibility, zero-strength equivalence, independent role gradients, cosine scale invariance, protected writes, null capacity, public-input-only execution and token/graph permutations. Local training and Torch installation were not used.
