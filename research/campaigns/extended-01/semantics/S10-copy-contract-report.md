# Surface-local copy identity: no hidden bilingual decoder

**Answer:** the existing factorized output can represent an English/Spanish instance using visible copy positions and equality. It does **not** export the hidden English spelling of an entity. The historical “exact canonical” metric is exact canonical node/edge/slot structure plus **surface-local** identity pointers, not byte-exact reconstruction of the compiler's English-named JSON graph.

Gold target construction is privileged: `identifier_forms(canonical_name, language)` finds visible renderer forms, then selects the first matching token index. For ident/entity nodes, the finite `value` target is **-1**, so hidden canonical identity strings are not output classes. At inference, `semantic_scaling.decode` only selects a copy position and folds identical **public token strings** to their first occurrence. It never calls the renderer, dictionary, or hidden-name map. Metrics compare that pointer with the renderer-specific gold pointer; supplied S10 `refers_to` compares predicted pointer equality only.

## Same graph, different visible spelling

Pinned generator fixture seed900100001, canonical alias `alice → red`; graph/ordering is identical across views. Representative clauses:

* English: `parent: erin, erin, red and erin`
* Spanish: `es padre de: erin, erin, rojo y erin`

| Canonical node | Kind | Hidden label (supervision only) | English copy target | Spanish copy target | Value target |
|---|---|---|---|---|---:|
| 7 | ident | red | token11 = `red` | token13 = `rojo` | -1 |
| 8 | entity | red | token11 = `red` | token13 = `rojo` | -1 |

All other occurrences of this entity target the same first visible position within their own surface. Non-copy target tensor hashes match across languages. A mechanical oracle-logit fixture achieves exact metrics in both languages with `identifier_forms` patched to raise if the decoder attempts an inference-time lookup. This establishes target/decoder realizability, **not learned translation or graph acquisition**. The fixed operation labels `parent`/`unify` remain supervised ontology classes; recognizing Spanish expressions for them must be learned from training, not supplied by the decoder.

No deployed graph exporter currently promises that Spanish token13 should be named English `red`. Literal English-name export or persistent identity shared across separate surface inputs would require a separately declared learned/public binding contract. Do not add a hidden dictionary at inference to claim that capability. Cross-surface evaluation must not compare raw token indices directly; use per-surface exact scores or an explicitly privileged scoring alignment.

## A genuine future-cache rejection case

Canonical injectivity is insufficient. The accepted canonical renaming `alice → red`, `carol → rojo` contains distinct names, but Spanish renders **both as `rojo`**. The current target helper assigns both identities the same first copy position. Its oracle exact metric is still1.0: gold copies have already collapsed, while compiler entity/reference distinctions persist. That number does not certify identifiable semantic targets.

A new additive dataset-only validator therefore **rejects** overlapping visible token supports for different canonical identities, uncopyable identities, inconsistent pointer/value targets, and identical actor text with different representable graph targets. It never drops, remaps, masks, or repairs a failing example silently. It is not yet connected to a main cache or actor; historical data and the frozen S10 decoder remain unchanged. New multilingual cache construction must stop and report rejection evidence before training if such cases occur.

The validator's public-text collision signature uses surface-local pointers, not English alias strings; two hidden labels that differ only in unrequested spelling are not a new target distinction. But two distinct entities merging within one example is rejected before target metric evaluation. Morphologically differing forms of one identity are allowed as supervision; inferring their equality is a learned task, not a guarantee of the literal-token decoder.

## Evidence and limits

Three fixed fixtures × two languages are retained, including original names, translated alias, and collision. Initial `cat/gato` probes are preserved: `cat` is not translated by this pinned vocabulary, so those probes were inconclusive for translation and were replaced by the documented known-vocabulary `red/rojo` fixture. No training or reserved data was used. Three new CPU tests pass; complete exact output and collision records plus version hashes are in `s10-copy-contract`. This is a small contract audit, not a full corpus audit. The proposed S13 cache must validate every actual surface independently before bulk training.
