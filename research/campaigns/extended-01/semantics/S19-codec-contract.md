# S19 typed-record codec contract

The codec preserves the six historical scored targets, not the entire canonical graph dataclass (node IDs/provenance/compiler metadata are excluded). It is a supervised target serialization; deployment accepts only generated records and public input lengths/vocabulary capacities. It contains no public-language parser or symbolic inference engine.

Records are five integers. Tags: BOS=0, NODE=1, EDGE=2, EOS=3, PAD=4. NODE payload is `(kind, finite-value-class, public-copy-position, -1)`; exactly one of value/copy applies according to kind. EDGE payload is `(source-node-position, target-node-position, relation, slot)` with slot=-1 unordered. Non-applicable fields are -1. Nodes retain canonical target order; edges sort lexicographically by source/target/relation/slot. EOS is mandatory. BOS is only previous-input state; PAD is permitted only after EOS. The 160 record budget includes EOS and excludes BOS. Maximum nodes128; slots0–31. Previous-input targets are BOS plus strictly previous records.

Training identity targets retain the inherited English alias mapping and first visible occurrence. At deployment, copying an identical public token at another position can be canonicalized to its first identical occurrence without gold labels. No alias substitution is inferred at deployment. Equality and ordered references are preserved.

Malformed records, unsupported tags/fields, references to absent nodes, record/node/slot overflow, duplicate identical edges, unsorted edges, NODE after EDGE, or absent EOS raise `CodecError` and **must count as exact graph failure**. Callers must not substitute a partially decoded graph and let historical metrics mask syntax errors. The returned presence vector derives solely from generated NODE count and pads to128; no target count is accepted.

The record list supports multiple relations and slots on the same node pair without overwriting. The historical tensor adapter explicitly rejects differing slot labels on the same pair, including ordered/unordered conflicts. This restriction belongs to the old metric representation, not the record representation. Exact duplicate edge records are rejected rather than silently merged.

## Mechanical audit

CPU only, CUDA hidden and2threads. All4096 S15 mixed TRAIN constructions were opened; no development or confirmation rows were opened. Cache SHA256 `8ddbd15bea881c13bfd24554ada9293082fcef7b616f6eeb3dad78a4bfa662d2`. All presence/kind/value/copy/edge/slot tensors exactly equal the inherited compact-target implementation.123501 nodes and276077 edges; every copy target equals first-identical-token canonicalization. Record stream SHA256 `a153c648f1ff7817d6d1811cb5ef840acc6d0dd09f02698aac4a9518ff96c022`.

Actual record counts are **82–123 including EOS**, correcting preliminary84–125 estimates. Histogram:82:124,84:841,86:897,88:186,100:192,102:568,104:264,119:55,121:486,123:483. Max160 therefore covers all audited TRAIN targets. This does not establish model acquisition or all possible future generator support.

Initial audit wall time8.325042728s;9fixture tests passed in0.003s. Reproducible full audit is the opt-in test `TrainCacheAudit`: set `S19_TRAIN_CACHE` to the exact train_mixed cache and `S19_VOCAB_AUDIT` to data/s01/audit.json, then run `PYTHONPATH=src python -m unittest discover -s tests -p test_campaign_semantics_s19_codec.py`. Tests independently compare inherited target tensors and pin stream/cache hashes. Ordinary test execution skips the external-cache audit. Fixtures cover ordered reversal, shared identities, multirelations/multislots, invalid grammar/references/capacities, and causal previous-record inputs.

Reproducible opt-in rerun:10 tests passed in6.923s, including full TRAIN loop6.916652337s. Combined measured audit/test body wall time15.251695066s (imports, export and SSH overhead excluded; no GPU process).
