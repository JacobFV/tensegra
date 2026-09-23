# Stage 6 semantic compiler and pinned TCN subset

The vendored source is `typed-crystallization-networks` commit
`018c9ce9286fa4961292fe1e024b49fcd7e2dd7f`, under
`generators/language/engine`. Its MIT LICENSE is retained. `MANIFEST.json`
records each original and vendored SHA256, including explicitly modified root
initializer, lesson initializer, and restricted registry. Only three lesson
modules ship; the shared grammar/language dependency closure ships intact
(excluding caches and unused non-source files). There are no sibling imports,
symlinks, installed legacy dependencies, or production path manipulation.

`compile_term(TermJSON)` creates immutable typed nodes and adjacency edges.
Canonical occurrence paths are stable under record dictionary insertion order;
argument and list-item edges carry ordered integer slots. Identifier occurrences
refer to shared entities within an explicit construction scope. `bind` relations
have binding and binding-scope edges. Each node records structural provenance.
Named App/record fields follow TCN's lexical field order. Nested terms are
compiler input only; the canonical result is a flat node/edge graph. The compiler
handles every TCN TermJSON constructor and rejects unknown constructors.

Scope support currently means one explicit construction scope, not inference of
arbitrary lexical binders from arbitrary predicate names. Identifier nodes do
not presume that uppercase always means a variable. This is a structural
semantic compiler, not a general executable TCN rewrite system. TCN generator
solutions provide supervision only; no generic exact TCN rewrite trace or learned
symbolic executor is claimed. The separate controlled arithmetic runtime owns
exact primitive execution.

`build_tcn_example(lesson, seed, languages=..., difficulty=...)` calls the pinned
lesson's `build` exactly once in fixed English generation context and renders
that same immutable Term through English, Spanish, and symbols renderers. It
never regenerates with a language-dependent seed. Default TCN hardening remains
active, including unification distractors. The tested subset uses literal name
and object-ID answers, preserved in options; renderings retain those identities.
Actual text hashes and distinct-text counts are recorded, not assumed from
language labels. `build_tcn_corpus` returns a deterministic balanced lesson cycle.

Actor input is a separate `TCNSurface(language, text, options)` object. Graph,
answer, hidden generator metadata and audit IDs/seeds/hashes do not occur on that
object. `TCNPrivileged` and `TCNExample.audit` are training/evaluation data and
must never be passed wholesale into actor inference. Audit includes source
commit and manifest hash, compiler/adapter hashes and versions, semantic graph
hash and each surface text hash. Changing hidden fields cannot mutate public
surface objects.

Local validation used stdlib unittest only, supplying a namespace package in the
test launcher to avoid the existing eager `topoformer.__init__` Torch import.
This is a test-launch detail, not source behavior. Two semantic graph tests and
two TCN integration tests pass, covering all term constructors, ordered bindings,
identity, stable compilation, deterministic generation, distinct actual surfaces,
public field separation, and vendored source integrity. Full project/Torch tests
must run on the remote CPU host through the root coordinator.

Remote installed-wheel verification also passed: `pip wheel --no-deps
--no-build-isolation`, `pip install --no-deps --target installed`, then the four
focused tests with `PYTHONPATH=installed` on `gb10-direct` using the existing
`~/topoformer-pilot/.venv`. This validates normal package imports and inclusion
of the license, manifest and all grammar JSON resources. The build snapshot is
in `~/topoformer-pilot/stage6-semantic-check`; full-suite validation remains with
the root coordinator.
