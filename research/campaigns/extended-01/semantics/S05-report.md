# S05: learned identity copying plus programmed reference edges

The scoped compiler contract allows reference edges to be reconstructed much more reliably from the model's predicted node attributes than from its learned relation logits. This is a posthoc engineering diagnostic on inspected S03 development predictions, not learned coreference or an independent confirmation.

The exact rule uses only predicted presence, predicted `ident`/`entity` types and equality of predicted public copy identities. Slots and every other edge/attribute remain unchanged. No gold node, identity, presence, relation or tie-breaker enters that rule. Multiple predicted matching entities retain multiple edges, exposing errors rather than selecting a hidden answer. The unchanged compiler's construction-scoped identity table and visible-copy equivalence were audited across8,704TRAIN+DEV constructions/165,704reference edges with zero counterexamples. This does not extend to lexical shadowing, distinct scopes, pronouns, aliases with different spellings or new renderer families.

| S03 DEV512 policy | Refers-to F1 | Exact reference sets | Complete graphs |
|---|---:|---:|---:|
| Learned raw edges | .45049 |0/512|0/512|
| Learned TRAIN-calibrated edges | .67762 |0/512|0/512|
| Programmed refs + other raw edges | .95518 |280/512|0/512|
| Programmed refs + other calibrated edges | .95518 |280/512|27/512|
| Gold refs + other calibrated predictions (privileged) |1.0|512/512|27/512|

TRAIN128 programmed referenceF1=.96851, exactsets74/128; calibrated completegraphs6/128 versus learned0. The programmed and gold-reference whole-graph scores matching does not mean the reference interface is perfect: many graphs still have incorrect predicted types/identities or other relations. Exact graph success already requires all copied identities and types correct, so the validated equality contract naturally matches the gold relation on that successful subset. The primary learned contribution is identity/type acquisition; equality-reference construction is supplied.

There is no additional training or model selection. Policy was fixed before applying it to predictions, but selected after inspecting the corpus/earlier failures; thus the result remains a development diagnostic. CPU application15.704s (including contract audit), no GPU inference. Source4d98520 and complete config/dependency/input hashes, per-example outputs, zero/multiple-match counts and aggregate metrics are saved under `research/results/campaign-01/semantics/s05-identity-contract`. An earlier standalone contract audit took2.032s. Independent reconstruction pending.

This localizes a substantial avoidable cost of asking a learned edge head to regenerate deterministic identity bookkeeping. It does not justify declaring semantic graph induction solved, changing S04 training retrospectively, or claiming programmable attention was tested here.
