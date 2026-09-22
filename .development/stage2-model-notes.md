# Stage 2 model implementation

`StudyPredictor` subclasses the pilot `GraphPredictor`, preserving every shared
state-dictionary key. Copy matching state keys between variants to pair common
initial parameters. Additional parameters exist only for graph-input projection
and learned coefficients; record their counts separately.

- `none`, `soft`, `hard` reproduce pilot behavior exactly under matched parameters.
- `graph_input`: average the full supplied adjacency's neighbor histories, project
  them through an extra bias-free linear layer and add to the node input embedding.
  Attention remains unbiased. Empty neighborhoods contribute zero. This baseline
  explicitly performs input message passing; it does not test serialization of
  graph tokens. Its projection receives graph information before all layers.
- `learned`: unconstrained alpha[layer, head], exactly zero initially. Multiply
  adjacency by alpha before calling scalar-strength attention with strength 1.
- `typed`: unconstrained alpha[layer, head, relation], exactly zero initially.
  Two channels contain positive and negative weight magnitudes, masked to supplied
  edges. It receives sign and magnitude information beyond adjacency-only models.
  No assumed sign or depth trend is enforced on learned coefficients.

Graphs and typed weights accept [N,N], [1,N,N], or [B,N,N]. Row i reads column j.
All variants preserve node-permutation equivariance, and no parameters depend on
node count. The inherited hard-mask conversion always allows self attention.
`structure_coefficients()` returns detached Python lists (empty for fixed modes).

Remote-only TDD: initial test collection failed with ModuleNotFoundError for the
new module. After implementation and singleton/pilot parity coverage, the targeted
study-model, pilot-model, and attention suites pass: **38 passed in 0.80 s** on
`gb10-direct`, isolated `~/topoformer-stage2-model`, existing CPU venv, OMP/BLAS
threads 2. Tests cover exact zero-alpha equivalence and nonzero coefficient
gradients, batched independence, permutation equivariance, actual graph-input
influence with unbiased attention, typed signed magnitudes and graph masking,
shape/type checks, and exact fixed-mode parity with pilot predictions.
