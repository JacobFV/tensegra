# S21 supplementary strict permutation-equivalence diagnostic

**None of the valid held-out predictions can be repaired by node renumbering under the prescribed strict attribute contract.** This conclusion now comes from a separate permutation-invariant necessary-condition check, not from canonical-position mismatch alone.

| Final arm | Canonical exact | Strict permutation-equivalent | Valid non-isomorphic | Invalid output | Unresolved |
|---|---:|---:|---:|---:|---:|
|Original|0|0|395|117|0|
|Broad|0|0|403|109|0|

Each arm has512 events; they share the same512 examples and are not1,024 independent sampled constructions. All798 valid outputs already fail equality of the **multiset of complete node attributes**: node kind together with finite JSON value or exact copied public identity. Node permutation cannot change that multiset. This diagnostic does not attribute the mismatch exclusively to types, counts or identity values; these were tested jointly.

No actual case needed the graph matcher after this necessary-condition rejection. The implementation nevertheless preserves directed typed edges, ordered slots and parallel-edge multiplicity and was prospectively tested with a same-count/non-isomorphic fixture that reaches the matcher. Fifteen synthetic fixtures cover harmless node renumbering, identical public-token occurrences, changed lexical identity/type/scalar value, reversed direction, relation changes, argument-slot swaps, multiplicity, and explicit unresolved timeouts.

The226 malformed outputs remain **invalid-output failures**, not asserted non-isomorphic decoded graphs. Original invalid reasons are115 invalid references and2 pair-slot conflicts; broad reasons are102 invalid references,5 pair-slot conflicts and2 duplicate edges. No invalid prefix was repaired or assigned partial equivalence credit.

## Contract and limitations

This compares the historical **scored graph contract**, preserving all scored node/edge distinctions. Copy identities are public token strings, so selecting a different occurrence of the identical token is harmless, consistent with existing copy canonicalization. Different names cannot be alpha-renamed into equivalence. Types, finite values, directed relations, argument slots and multiplicity cannot be discarded. Compiler path/hash IDs and provenance, absent from the historical scored tensor target, were not introduced as extra requirements.

This supplementary audit does not revise canonical exact scores or the failed held-out promotion criterion. It rules out pure node renumbering as a repair for these798 valid outputs under this contract; it does not prove a universal limitation of the architecture, identify the learned shortcut causally, or establish arbitrary-language semantic equivalence.

## Provenance and compute

Root protocol082b1535 authorized this diagnostic before isomorphism outcomes. Implementation6563de69 and scalar fixture990de467 were committed before actual evaluation. Independent reviewer cleared the matching contract and reran all15 fixtures before execution. Installed NetworkX3.6.1 was used without installation or environment changes.

The process reused the completed independent S21 raw/codec validity audit; it does not claim a second independent codec-validity audit. Both fixed endpoints, public identities, paired target equality, artifact hashes, frozen config5a13908e and successful closed receipt were checked before comparison. Raw results retain exact source/input hashes and per-event status/reason.

Actual outer CPU wall time **11.99s**, exit0, maximum RSS3,352,624KiB. No GPU or model inference, confirmation access, failed attempts, or timeouts. The120s outer bound and105s internal deadline were not reached; no cases were unresolved. Independent counter verification remains separate from this producer report.
