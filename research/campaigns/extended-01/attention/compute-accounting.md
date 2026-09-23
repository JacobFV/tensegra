# A03 compute accounting

Allocated parameter equality does not imply equivalent work. Forward CUDA timings in each final cell are the primary measured comparison. They bracket the model call, excluding data generation, scoring and artifact export. Full process occupancy includes those costs and startup. Report both; per-correct-example forward cost is undefined when no example is solved.

For a batch of B graphs, N nodes, D supplied relation steps, width W=1024 and key width K=64, the dominant dense matrix multiplication counts below are multiply-accumulates (MACs), not measured hardware FLOPs. They omit normalization, activation, softmax, embedding lookup, readout and other elementwise operations, so they are explanatory approximations rather than a complete profiler.

- Every arm's two-layer local update costs about `2 B D N W²` MACs.
- Soft, hard and no-graph attention additionally perform two W×W projections (`2 B D N W²`) and score/value products (`2 B D N² W`). Hard routing does not skip these dense products in this implementation.
- Message passing adds a dense adjacency/value product (`B D N² W`). The functional graph could permit an exact indexed gather, but that optimization is not implemented or claimed here.
- Structured context additionally projects two K-dimensional key maps (`2 B D N K²`), computes their compatibility (`B D N² K`), composes adjacency with address reads (`B D N³`), and applies reads to values (`B D N² W`). Key maps are recomputed at each step in the frozen implementation.

Thus the strong message baseline is also computationally cheaper than soft attention in the dominant W² terms. Identical final task accuracy would not constitute a soft-attention efficiency advantage. Dense structural bias introduces no sparse-compute guarantee.

Parameter reports distinguish allocated tensors, autograd participation, and nonzero gradients at the final update. None measures the quantity of learned computation. On clean single-successor rows hard attention and normalized message passing have the same routing operator; zero Q/K gradients are expected for hard singleton softmax.
