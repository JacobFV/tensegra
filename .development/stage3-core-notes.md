# Stage 3 runtime graph and grounding core

## Contract

`RuntimeGraph` keeps external stable integer IDs separate from tensor slots. Batched embeddings are `[B,N,D]`; directed relation matrices are `[B,R,N,N]`, with source/query on the first node axis. An optional valid mask permits heterogeneous graph sizes in a padded batch. Only valid IDs must be unique. Padded nodes cannot have incident edges; their embeddings remain finite but are masked from grounding. The graph contains neither a special null ID nor implicit self edges.

`SoftGrounding` has four independent, bias-free projections: query/key latents and query/key entity embeddings. Scores are scaled dot products divided by temperature. A separate trainable scalar null score per role competes with real entities. Each output has `N+1` columns; the final column is null. Empty graphs and graphs with all slots masked ground entirely to null. A fixed temperature is a buffer; a learned temperature is log-parameterized, numerically bounded to exp([-12,12]). Neither identity initialization nor graph-aware feature construction is hidden in this generic module.

`induce_bias(pq, adjacency, pk)` returns `[B,R,Tq,Tk]`. Null probability contributes zero incoming/outgoing graph bias, without re-normalizing the remaining probability. The contraction supports rectangular latent collections, signed edge weights, independent query/key distributions, and gradients through both grounding and adjacency. Half precision contractions accumulate in float32; float64 remains float64.

Grounding is stateless: the caller must invoke it on every current residual state when recursive grounding is wanted. It does not enforce a layer schedule or cache entity assignments. Null capacity alone does not guarantee correct distractor rejection; that must be measured in task experiments.

## Validation

Tests cover exact directed one-hot contraction, null attenuation, invalid/padded graph contracts, distinct query/key mappings and nonzero gradients through every projection and learned temperature, numerical gradcheck of the induced bias, node-slot permutation equivariance, all-padded/empty graphs, and low-precision accumulation. Initial tests failed on the absent core modules; implementation was then added. Full remote suite at this change: **174 passed in 1.84 seconds** (includes concurrently added Stage 3 data tests). Remote execution uses the existing CPU Torch environment with OMP/BLAS threads limited to two. No local model allocations were used.
