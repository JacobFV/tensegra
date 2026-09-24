# Pinned unification structural-transfer feasibility (CPU only)

The existing pinned generator supports genuinely unseen **ordered tree motifs through predicate arity**, with unchanged vocabulary and actor capacity. It does **not** support increased structural depth: the hardened constructor always makes a record containing a list of flat predicates, one flat pattern predicate, and a flat query, so maximum structural depth is always3. `difficulty` only sets `round(2 + 3*difficulty)`; no seed can add nesting. Historical difficulty0.5 selects arity4 (Python rounds3.5 to4). The two historical motifs are the three-fact and four-fact cases.

A bounded64-fixture audit (16seeds per arity, English only, no model imports/forwards or new corpus) confirms:

| Predicate arity | Fixture motifs | Unseen vs historical TRAIN | Nodes | Tokens | Max slot | Depth |
|---|---:|---|---|---|---:|---:|
| 2 | 2 | Both | 21–27 | 30–34 | 3 | 3 |
| 3 | 2 | Both | 25–32 | 46–54 | 3 | 3 |
| 4 | 2 | Neither | 30–37 | 54–64 | 3 | 3 |
| 5 | 2 | Both | 35–42 | 62–74 | 4 | 3 |

All fixtures fit128nodes, the unchanged `<unknown>`, `"parent"`, `"unify"`, `null` value-class order, visible English identity copying and unambiguous single-slot targets below32. Exact tree fingerprints use the existing diversity-audit definition: omit scope/entity bookkeeping, retain node kinds, ordered structural edges and slots. Pinned vendor manifest passes. Per-fixture graph/surface/shape hashes and audit source hashes are archived. This is feasibility, not a full split/collision/feature audit or model result.

Arity3 is the least confounded prospective first structural diagnostic: it retains the same n-ary English rendering family and already trained slot support, while changing ordered motif. It is smaller than historical arity4, so it tests transfer across arity, not increased complexity. Arity5 is a useful separate extrapolation condition but introduces argument slot4 and longer sequences. Arity2 also switches English rendering from `parent: ...` enumeration to the binary infix `alice parent bob` pattern, so it changes syntax as well as motif. Report these distinctions explicitly; a single pooled accuracy would obscure them. Generate/split by construction before surface rendering and freeze independently fresh DEV/confirmation supports only after separate authorization. No existing S12/S13 cache or recipe is changed by this note.

Depth transfer needs a **new versioned constructor/adapter**, not a hidden difficulty setting or vendor modification. Reusing `parent`/`unify` heads could preserve value vocabulary, but nested-term matching, answers, rendering, copyability and the single-scope reference contract require explicit validation. Hold renderer/language, vocabulary, task contract, training exposure and architecture fixed when feasible; freeze any irreducible semantic change. No such extension is implemented here. S13 remains paired renderer acquisition on its existing two motifs.

Reproduce the bounded audit from repo root with `python3 research/campaigns/extended-01/semantics/S14-motif-feasibility.py`; it uses standard-library data/compiler imports and deliberately bypasses the package initializer that imports attention/torch. No actor is instantiated.
