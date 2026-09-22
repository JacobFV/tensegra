# Stage 5 neural boundary

`RuntimeBindingModel` accepts only public surface clauses, noisy reference vectors,
shuffled candidate descriptors (keys, values, types, mask), initial typed edges,
step masks and output style/comparison cues. Word position is encoded within a
clause; candidate position is never encoded. A one-layer small transformer pools
each clause, adds a learned reference projection, and predicts independent
operation and cosine candidate scores. The two cosine projections start randomly;
there is no fixed identity matcher. A learned final null candidate permits unbound
lowering. Runtime execution is external: this module cannot write protected IDs,
frames or types. Argmax/exact execution creates no pathwise task gradient.

The local confidence score combines normalized entropy and top-two margin for
both heads, multiplied by non-null probability. It is not a calibrated correctness
probability. Schema/type checking and threshold selection belong to the external
executor/evaluator. No single symbolic/neural switch is implemented.

Learned lifting takes the raw runtime scalar divided by `result_scale`, an already
normalized public comparison scalar and a learned style embedding. A small MLP
predicts output classes. It does not implement sign/comparison/text rules itself.
The default 129 classes cover integers -64..64; style semantics are data-owned.

Neural controls share the clause/lowering modules and unroll one GRU update per
public clause. `none` attends to initial candidate keys/types/values without using
edges. `graph_data` adds typed message-passing features; `soft` adds a learned
relation mixture of soft grounded graph rows to attention logits;
`protected_learned` maintains a separate learned register whose update is a GRU,
not an exact pointer operation. All controls receive the public comparison cue.
These are deliberately small matched recurrent controls, not claims of optimal
neural architectures. Runtime mode returns no neural answer prediction until the
caller supplies an executed scalar to `lift`.

Validation: 10 remote CPU tests passed (0.81 s), covering independent lowering and
lifting gradients; permutation equivariance in all five modes; exact zero-strength
soft/ordinary equivalence; confidence/null behavior; no gold dependence; empty
padding/null-only candidates; and comparison-cue observability.
