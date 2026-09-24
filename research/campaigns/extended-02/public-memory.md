# Public relational memory v1

`campaign02_memory.encode_memory(observation, actions)` returns a Torch-independent `PublicMemory` with **281-dimensional rows**, sparse typed edges, action roots, candidate-to-row links, and capacity statistics. Both learned controller families receive the same memory. Ordinary readers have explicit node index, parent index, child ordinal, field identity, and reference identity in each row; relations are not available exclusively to a graph-bias arm.

The codec preserves the full **currently public** observation and candidate-action document. It never inspects hidden world state. JSON dictionaries/lists remain ordered trees. Scalar type and exact signed64 integer / finite binary64 float bytes remain distinct, with magnitude features alongside them. Fixed schema tags are a compact convenience: unknown strings and field keys fall back to ordered UTF-8 byte chunks without hashing or vocabulary truncation. Unknown facts (`null`) are distinguishable from observed empty lists. Ordered arguments and graph endpoints are retained by tree structure, not reduced to bag statistics.

Opaque item/problem/return handles are interned by public declaration/occurrence order, not spelling. Typed handle occurrences share explicit local identities and sparse reference-to-anchor edges. Candidate links include the complete action subtree and anchors of its references. Consistent renaming leaves all rows, sparse edges, and action links identical in the mechanical test. The table is rebuilt per public frame; local handle indices can shift when new declarations arrive. Persistent identity remains the simulator's exact handle contract, not an invariant promised for these per-frame numeric indices. A query must use current-frame bindings rather than treat an index as a timeless object identifier.

`decode_memory` reconstructs the alpha-normalized public document from the **row vectors alone**, without consulting raw JSON, metadata, or sparse edges. Mechanical tests additionally cast row values through binary32 and preserve exact scalar-byte reconstruction (including negative zero). Neural embeddings of those rows are not guaranteed lossless; this is an input-contract property, not model acquisition.

Opaque references follow explicit workshop reference fields and maps, including ordered identity lists. Public literal strings retain their text; quoted known handles in KeyError diagnostics are normalized. This is a versioned workshop schema codec, not a claim to automatically infer reference semantics for every possible new language/runtime. New reference-bearing schema fields need explicit tests and versioning.

## Capacity and cost

Default limits:8192 nodes,512 handles,depth64,4096 UTF-8 bytes per string. Limits fail with `MemoryCapacityError` rather than truncating. Unsupported live examples must remain in the evaluation denominator as unsupported/failures; a corpus generator must not silently drop them. The codec has no padding or fixed node-count assumptions. Consumers supply masks for actual rows and may batch by size. Dense relation matrices are unnecessary.

CPU profile:88 snapshots,approximately0.81 process-CPU seconds, no Torch/GPU or real solver. A2×3 public-teacher trace reached285 rows. A3×4 no-executor trace reached504. A deliberately large8×8-item repeated-call **mechanical** trace reached5290 supported rows and exceeded8192 after four repeated full-snapshot result records. The stress executor returns only `unknown`, consumes no simulated solver work, and certifies nothing; its purpose is shape growth. It is not a solvability or policy benchmark. Full receipts are in `diagnostics/public-memory-profile.json`.

The higher cap is not evidence that all supported trajectories fit GPU memory at a desired batch size. Root must profile both encoder/token costs and neural forwards before choosing experimental batch/capacity settings. Any smaller deployed cap changes the supported contract and must be explicit.

## Mechanical checks and interpretation

Five CPU-only tests pass: exact normalized reconstruction (unknown tags/UTF8/scalars/order), float32 transport, opaque-renaming invariance, action binding links, unknown-versus-empty distinction, explicit overflows, and repaired historical counterexample distinguishability. The two formerly identical-input pairs now have distinct memory rows. No greedy solution, shortest-path answer, feasibility label, preferred action, or gold mask was added.

This repairs an observable-input omission. It establishes neither a learned resource policy nor a structural-attention advantage. Subsequent ordinary/keyed, message-passing, and graph-bias consumers must receive equivalent memory and candidate information.

## v2: stable typed address ranges before consumer freeze

The initial v1 per-frame joint numbering could shift existing record indices when a new problem was declared. **v2 removes this unnecessary drift without changing281-dimensional row width.** Item, problem, record, and other identities occupy distinct ranges of `max_handles` slots, using append-only public declaration order within each range. At default limits the ranges start at0,512,1024,1536. The overall number of declared handles still cannot exceed512; this is address spacing, not permission for2048 live identities. The maximum configured handle capacity is16383 so all four ranges remain representable with the fixed binary index scale.

Existing return coordinates now remain unchanged when a new problem is declared, and a fresh problem candidate receives the same identity when the world creates that problem. This guarantee assumes the workshop's append-only declaration/order contract. A future environment deleting/reordering declarations would need persistent public interning or a different declared reference contract.

Handle roles are explicit: the same spelling can denote an item, a problem, and a result without conflation. Reference-bearing action fields are interpreted through the public action kind and schema, not inferred from numeric spelling. `map` remains the literal inspection keyword; an item named exactly `map` is rejected because the existing world's inspection API would already make it ambiguous. A diagnostic string quoting a spelling shared across namespaces is rejected if its role cannot be uniquely resolved rather than silently normalizing it incorrectly.

Seven CPU tests pass, adding stable-existing-return/fresh-problem binding and same-spelling namespace-collision fixtures. The v1 profile is preserved unchanged; `diagnostics/public-memory-v2-profile.json` records the refreshed source-bound shape profile. No learned policy or live experiment was changed during this mechanical correction.
