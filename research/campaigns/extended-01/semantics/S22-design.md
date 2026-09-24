# S22: frozen semantic decoding boundary diagnostics

Registered after S21's complete paired development result, before any S22 inference. S21 acquires its new motifs and retains old motifs but both arms remain 0/512 on omitted3x4; its conditional confirmation is ineligible. This is a new exploratory diagnostic, not an extension or a replacement gate. Independent S21 raw auditing/localization continues.

## Question and supplied boundaries

Can the frozen S21 final learners generate correct identities and relations when supplied only the correct node count, the typed node skeleton, or the complete node prefix? This distinguishes early layout errors from conditional field/relation failures. None is a deployable public-text result. Gold count, kinds, and node records are explicitly privileged inputs at named boundaries, never concealed as learned predictions.

Use both unchanged final4096 S21 checkpoints, original and broad, width1024. No optimizer or new training. Evaluate exactly the same seven DEV512 cells, in the same order and batch32; no sealed confirmation access. Preserve S21's unmodified public-only outputs as the reference. Three separately named policies:

1. `oracle_node_count`: supply exact N; force NODE tag for first N output records, predict all node fields. Thereafter NODE is disallowed and the learned EDGE/EOS scores choose the tag. Edge fields and termination are learned, bounded by the original160-record cap.
2. `oracle_node_kinds`: the same count/tag rule, plus supply each canonical node kind; value/copy fields remain predicted. Use the forced kind to choose which value/copy channel is active. All edge fields and termination remain learned.
3. `oracle_node_prefix`: supply all canonical NODE records as the prefix; model receives them sequentially through the unchanged cache. Afterward generate edges/EOS using the same no-more-NODE constraint. Identities, values and all node semantics are privileged here; only conditional relation generation is measured.

The encoder still receives the unchanged public text only; diagnostic metadata is isolated in the decode controller. Do not expose gold edge counts, edge records, slots, answers, or future hidden states. NODE ordering is the supplied canonical convention. Record-level masks protect only the named boundary. Keep malformed records and the original strict codec; do not repair graph edges after generation.

## Metrics and controls

Primary: exact complete graph counts for each arm/policy/cell, alongside the untouched S21 public-only reference. Report node presence/kind/value/copy, edges/ordered slots, validity and invalid reasons, exact nodes and conditional edge recovery, paired corrections/regressions and deterministic failure examples. Count forced-tag/kind replacements and expose the privilege each arm receives. Conditional-edge scores under gold nodes are an oracle-boundary diagnostic, not evidence that public text acquired those nodes. Do not multiply component marginals or reinterpret oracle accuracy as a historical gate pass.

Mechanical tests must establish no-op decoding parity with unchanged greedy decoding, mask direction and exact prefix/cache behavior, input/output type contracts, retained invalid outputs, unmodified checkpoint hashes, and no gold edge access. Labeled small fixtures are allowed only for these tests. Independently replay all compact raw predictions and targets. The actual main config/source and descriptive analysis rules must be frozen before full inference.

## Budget and decisions

Planning envelope at most1200 GPU seconds including a representative profile and all six policy/backbone evaluations/export. Freeze a modest profile first, measure longest public inputs and160-step decoding, then set a full-process main cap within this envelope. GPU scheduling remains serial under root; no release is implied by this design. CPU auditing is separate.

No acquisition/promotion threshold is added. Report all cells and policies. If kind/count assistance recovers omitted composition, the early layout interface is a justified target for a new separately registered learned repair. If full node prefixes still fail, the relation-generation boundary also fails under the tested omitted combination. Even perfect oracle performance does not establish learned structural transfer or authorize autonomous composition. If localization yields no clear learnable repair, stop this branch rather than adding a speculative architecture. Historical results and S21 failed confirmation eligibility remain unchanged.
