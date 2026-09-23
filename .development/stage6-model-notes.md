# Stage 6 recurrent model

`thinking.py` supplies `ThinkingConfig` and `ThinkingModel`. Defaults are width 32,
four heads (two free, two structural), eight distributed workspace rows, two
candidate queries, and four distinct recurrent blocks reused on every call.
There is no latent history, cache, or externally mutable recurrent state in the
module. Caller passes current workspace, public context and current value memory.

Each block performs content workspace attention, cross-attention to context plus
current memory, and an MLP. Query and key groundings use independent normalized
projections. Structural head bias is Pq A Pk transpose; free heads receive zero
bias. A is predicted from current memory by default. Any explicit graph argument
must be observable, never the hidden target. Zero strength gives ordinary
attention. The supplied feature interface is a prior, not language induction.

Candidates softly pool workspace rows; routes overlap. Heads predict operation,
ordered argument pointers with null, local readiness, and clause identity pointers.
Identity memory may contain all currently declared public clauses independently
of currently available runtime values. This avoids assigning write IDs from an
oracle execution schedule. Routes are also the additive event reinjection weights.

Event tensors contain scalar values, type IDs (integer 0, float 1, boolean 2),
operation IDs, ordered public argument features, and optional public provenance
features. Type, operation and scalar roundtrip heads are auxiliary diagnostics.
Zero event messages or a zero event mask preserve all workspace rows exactly.
Final output only reads pooled workspace. The driver must enforce at least one
recurrent update after a returned event; this stateless model does not track that
external lifecycle. Emission has a learned probe, smooth progress bias, hard
minimum/maximum bounds, and differentiable ponder cost. Token time is explicit and
separate from microstep time. Initial output is an answer label, not a general
language model.

Verification: remote CPU only, gb10-direct, OMP/MKL two threads, dedicated scratch
`~/topoformer-stage6-model`, interpreter `~/topoformer-pilot/.venv/bin/python`.
The first missing-module test failed before implementation; the clause identity
contract failed before that behavior was added. Nine model tests pass, covering
state reuse/no history, zero-strength equivalence, free-head isolation, asymmetric
grounding, overlap, gradient paths, additive/drop events, hard halting, public
causality, null-only memory, identity pointers, and event roundtrip gradients.
Full isolated baseline-plus-owned-files suite: **380 passed**. The first full run
had two provenance test failures because a git archive has no `.git`; initializing
and committing the disposable snapshot resolved that environment issue. No
training experiment or changes to Stage 1–5 sources were performed.

Pre-freeze hardening: topology now uses a learned memory cross-attention refresh
from public context and current workspace before edge prediction. Changing only
public context changes the predicted graph. The predictor returns typed adjacency
`[B,R,N,N]` with `n_relations=2` by default (ordered argument slots). Structural
geometry retains each relation and learns independent head-by-relation strengths;
its weighted mean preserves scale for the backward-compatible 3D graph input,
which is explicitly broadcast to all relation channels. Query and key grounding
now each include a learned final null column, excluded from induced graph edges.
All-masked and truly empty node memories ground entirely to null. Three new
regressions failed before these changes; all twelve model tests now pass, and the
full isolated baseline-plus-owned-files remote suite reports **383 passed**.
Runner notified of both changed graph and grounding tensor shapes before pilots.

Pre-main halting correction: removed hard-budget-normalized progress from the
emission prior. The probe now uses
`sigmoid(probe - alpha*softplus(soft_min-t) + beta*softplus(t-soft_max))`,
with independent positive configuration fields `soft_min_microsteps=2`,
`soft_max_microsteps=10`, `emit_alpha=.5`, and `emit_beta=.05`. Hard minimum and
maximum only clamp the probability. Thus hard caps 12 and 80 produce exactly the
same non-clamped emission probability for identical current state/context at
microstep 3. This fixes a train/evaluation budget confound before the controlled
main experiment, without adding weights or changing workspace/language outputs.
Both new regression tests failed against the previous implementation; fourteen
model tests and the full isolated remote baseline suite now pass: **385 passed**.
