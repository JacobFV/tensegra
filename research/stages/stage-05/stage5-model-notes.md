# Stage 5 neural boundary

`RuntimeBindingModel` accepts only public surface clauses, noisy reference vectors,
shuffled candidate descriptors (keys, values, types, mask), initial typed edges, slot-index/current-frame/root-frame payloads,
step masks and output style/comparison cues. Word position is encoded within a
clause; candidate position is never encoded. A one-layer small transformer pools
each clause, adds a learned reference projection, and predicts independent
operation and cosine candidate scores. The two cosine projections start randomly;
there is no fixed identity matcher. Candidate key projections also consume learned
type/value/payload projections; small random type embeddings avoid overwhelming
unit lexical keys at initialization. Every neural control consumes the same
payloads, so scope and index information is not privileged to the executor. A learned final null candidate permits unbound
lowering. Runtime execution is external: this module cannot write protected IDs,
frames or types. Argmax/exact execution creates no pathwise task gradient.

The local confidence score combines normalized entropy and top-two margin for
both heads, multiplied by non-null probability. Binding probabilities first aggregate
by public semantic selector groups (kind/payload equivalences, not gold masks).
Entropy normalizes by each example’s valid group count plus null, independent of
batch padding. It is not a calibrated correctness
probability. Schema/type checking and threshold selection belong to the external
executor/evaluator. No single symbolic/neural switch is implemented.

Learned lifting takes the raw runtime scalar divided by `result_scale`, an already
normalized public comparison scalar and a learned style embedding. A small MLP
predicts output classes. It does not implement sign/comparison/text rules itself.
The default 129 classes cover integers -64..64; style semantics are data-owned.

Neural controls share the clause/lowering modules and unroll one GRU update per
public clause. `none` is ordinary graph-as-token attention: directed edge records encode ordered
source/destination features and relation type as additional memory tokens. All
neural controls receive these same edge tokens, materialized sparsely from existing
edges. `graph_data` adds typed message-passing features; `soft` adds a learned
relation mixture of soft grounded graph rows to attention logits;
`protected_learned` maintains a separate learned register whose update is a GRU,
not an exact pointer operation. All controls receive the public comparison cue.
These are deliberately small matched recurrent controls, not claims of optimal
neural architectures. Runtime mode returns no neural answer prediction until the
caller supplies an executed scalar to `lift`.

Validation: 19 remote CPU tests passed, covering independent lowering and
lifting gradients; permutation equivariance in all five modes; exact zero-strength
soft/ordinary equivalence; confidence/null behavior; no gold dependence; empty
padding/null-only candidates; comparison-cue observability, and scope/index payload sensitivity in all modes.

Follow-up tests cover directed/typed edge observability and gradients in ordinary
attention, sparse edge padding, semantic-group confidence and padding invariance.

## Pre-main lifting representation fix

The shallow oracle pilot attained exact execution but its scalar-only learned
lifter remained weak. Before the main run, lifting now augments the normalized
raw scalar/comparison and style embedding with fixed Gaussian RBF features
centered at every integer in the bounded result vocabulary (-64..64), width .75.
This is an explicit numerical representation prior, shared in every model's
parameter layout. It is differentiable, does not clip out-of-range values, and
contains no output-class, sign, comparison or report rule. The final MLP must
still learn output mapping. This evaluates bounded report labels, not free prose.
A test verifies neighboring-value discrimination, finite scalar gradients,
nontrainable centers and complete dependence on learned output parameters.

## Separately labeled semantic-read controls

After the main experiment source freeze, optional `selector_read=True` adds the
predicted selector-weighted node-memory vector to ordinary attended memory. The
predicted operation distribution also combines the existing operation-head weight
rows into the recurrent instruction. Graph-data mode reads its message-enriched
node memory. Every resulting update remains learned; no pointer transition or
interpreter primitive runs. This tests whether the original neural controls were
limited by lacking a direct read from their supervised lowering head.

The default is false and introduces no parameters or random draws. The separately
labeled privileged control may pass `lowering_override=(operation_probs,
selector_probs)` explicitly, outside public inputs. Shapes, distributions and
padding are validated. Prediction logits remain unchanged for auxiliary metrics;
returned diagnostics distinguish used probabilities and override status. This
privilege must be disclosed and must never be described as learned binding.

Validation: 22 model tests pass remotely. An independent isolated import of
`runtime_model.py` from frozen main source `2e95` compared identical seeded
initialization and every tensor output in all five default modes: bitexact.
No main training checkout or artifacts were changed by this source comparison.
