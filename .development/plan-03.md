# Stage 3 execution record

- [x] Isolate the new stage; preserve original explicit-node predictors and experiments.
- [x] Implement batched runtime graphs, directed relations, padding and stable IDs.
- [x] Implement separate differentiable query/key grounding with null bindings and temperatures.
- [x] Verify induced bias, gradients, permutation equivariance and zero-strength equivalence.
- [x] Generate shuffled keyed traversal tasks with held-out graphs, compositions and sizes.
- [x] Verify exact one-hot attention traversal through depth 32 independently of direct graph execution.
- [x] Implement recurrent query grounding, known/frozen/permuted/hard/no-graph/message-passing controls.
- [x] Run the IID-only feasibility pilot, then the fixed 39-run main study.
- [x] Add and execute a separately labeled nine-run content-prior comparison after fairness review.
- [x] Preserve raw metrics, source hashes, configurations and environment details.
- [x] Generate seed-level summaries and exported scientific figures.
- [ ] Complete independent report/artifact review, integrate and push.

The empirical target remains open: identity-initialized traversal partially succeeds,
but cold grounding and simultaneous depth/size generalization fail, and the stronger
message-passing control prevents a metric-specific superiority claim. Interpreter and pretrained-model extensions remain outside this stage.
