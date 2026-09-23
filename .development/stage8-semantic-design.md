# Stage 8 independent semantic acquisition

This track is separate from runtime, belief updates, readiness, and returns. Its
standard workspace width is **1024**, with four distinct attention/MLP phases
reused for two microsteps. Tests explicitly use width 16; they are contract tests,
not scientific experiments. No training has been authorized or run at this point.

## Independent axes and curriculum

Candidate configurations cross unique training graphs N={1k,10k,100k} with
optimizer presentations E={10k,100k,1M}. Resource authorization follows actual
1024-wide device timing; the grid is configurable, not a claim all cells ran.
A presentation is one graph rendered in one language. English/Spanish alternate
for every graph, independent of N. At E presentations, at most E/2 distinct graphs
have been consumed. Curves log unique graphs actually seen, tokens, presentations,
renderer counts and available corpus size separately. Evaluation uses the same 64 fresh semantic keys across N, excluding previously
inspected evaluation keys; training sets are nested disjoint prefixes. The final
main freeze uses balanced lesson presentations to hold family mixture fixed.

The first 1000 presentations supervise node presence/types, then values and visible
identity copying enter, and sampled typed/ordered edges enter at 2000. Every loss
is logged independently. No lexical-hash target is used. Whole-graph equality is
an evaluation metric, not the only learning target. Curriculum boundaries are
prespecified exposures and do not certify prior component competence.

Training scores all gold-positive edges plus up to 128 sampled negative node
pairs, avoiding materializing pair×relation×latent tensors. Sampling uses labels
only to select loss queries; it does not change node states. Dense edge scores
are computed for evaluation. A width 128 edge compatibility head is not a narrow
workspace: all four recurrent phases, node states and attention operate at 1024.

## Controls and priors

Paired no-input controls receive identical supervision/exposures/model capacity.
They have no textual features, but retain the public copy-candidate count and
text-based pointer normalization used by the shared evaluator. This is explicitly
a no-text-feature control, not a claim that absolutely no input metadata remains.
The frequency baseline sees both language labels from every training graph;
its label budget is reported independently from optimizer exposure.

The 1k/10k comparison uses one common 10k training-only vocabulary pool, excluding
the reserved heldout prefix. This keeps categorical value-head capacity equal.
Larger 100k configurations use a 100k pool; to compare their capacity exactly with
smallerN, rerun smaller arms with vocab_pool_count=100000. The smaller arms then
receive a declared categorical ontology prior from unoptimized training graphs,
not heldout examples or target answers. Source/compiler/vendor hashes and this
pool size are recorded.

Inputs never include gold graph size, spans, nodes or edges. Decoder capacity 128,
canonical compiler slot order, relation/type ontology and visible-copy targets
are supplied architectural/supervision priors. Copy supervision uses pinned
renderer forms. Evaluation includes English, Spanish, heldout symbols renderer,
and consistently renamed lexicon. Exact graph success means canonical ordered
representation equality; arbitrary graph isomorphism is not claimed.

This differs from Stage 7 in width, actor simplification, curriculum and sparse
loss sampling. It cannot isolate a single cause of improvement over Stage 7.
Within Stage 8 the N×E axes hold those choices fixed. No composition, annealing,
or semantic competence claim follows merely from constructing this runner.
