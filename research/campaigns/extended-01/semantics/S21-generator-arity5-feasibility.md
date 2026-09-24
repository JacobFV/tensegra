# Existing arity-five support: additive bounded probe

**Yes: unchanged `ctx.at(2,5)` supports arity5 at difficulty1.0.** No generator extension, extra identity vocabulary, relation type, or semantic operation is needed. Together with arities2–4 and facts3/4, this supplies eight occurrence-tree motifs in principle, hence seven TRAIN motifs with3×4 held out. The required512 alpha-distinct examples per TRAIN cell is not established by this probe.

| Cell | Attempts | Selected draws | Alpha-distinct | Lexically distinct | Public tokens | Nodes | Records including EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5×3 | 256 | 142 | 141 | 142 | 62 | 35–36 | 118–120 |
| 5×4 | 256 | 122 | 122 | 122 | 74 | 41–42 | 141–143 |

All512 independent graph-based answer checks passed: exactly one fact agrees with the pattern's bound ordered arguments and gives the generator's answer/option. All264 selected draws passed visible-copy and the unchanged strict128-node/160-record codec. No conflicting target for identical public text occurred. Maximum ordered slot is4, within existing32-slot support. Finite nonlexical values remain `parent`, `unify`, and null; lexical values remain the existing six names and one of A–E.

For the largest5×4 shape, the generator has34 term occurrences, at most seven entity identities and one scope node:42 nodes. Edges comprise34 contains,33 occurrence-tree,26 refers_to and at most7 declares:100 edges. Thus maximum records are42+100+EOS=143, not merely a sampled capacity estimate.

Each cell adds one tree motif. Alpha-distinctness is variation in equality/sharing and ordered identities, not122 different tree shapes. Existing2×3 saturation concerns remain unchanged.

The public token lengths62 and74 do **not** overlap any existing2..4-arity lengths30,34,46,54,64. The sole equal-length motif pair remains3×4/4×3 at54. Broader motifs may support factor learning, but this addition alone does not give a second TRAIN layout at54 or isolate a causal length shortcut.

Reproduce with `python3 research/campaigns/extended-01/semantics/S21-generator-feasibility.py --arity5-only`. Output is separate `research/results/campaign-01/semantics/s21-generator-arity5-feasibility.json`; prior result bytes remain unchanged. Original default probe semantics remain unchanged. Fresh namespaces921503000 and921504000,256 attempts each. CPU wall time **0.448970s**. No models, GPU, campaign dataset construction, confirmation access, or allocation.
