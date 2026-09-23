# Stage 9 semantic failure localization

Status: **diagnostic_only**. No Stage 9 public-text training has run.

## Historical numerical reconstruction

The independent Python archive reader reconstructs3,072 final semantic-arm
prediction rows:2corpora ×3seeds ×4surfaces ×64graphs ×2decoders. This is a
numerical reconstruction of archived predictions, **not rerunning inference**.
Every complete-graph, typed-edge true-positive/prediction and ordered-edge
true-positive count matches its saved metric. Historical test outcomes are used
for diagnosis only.

Across this paired/repeated population (not3,072independent constructions):

| Entire component correct | Count /3,072 |
|---|---:|
| Node presence |2,056|
| Node types |20|
| Non-identity categorical values |2,046|
| Visible identity copies |0|
| Typed edge set |0|
| Slots on gold edges |52|
| Complete graph |0|

Each single-component logical oracle replacement leaves zero complete graphs.
Thus correcting slots alone cannot make the *frozen predictions* exact; copying,
node typing and edge prediction also fail. This is not evidence that decoder
repair cannot alter joint acquisition after training. Logical replacement of
presence holds already-masked edges fixed; it is a remaining-error diagnostic,
not a claim about changed inference. Per-seed/renderer/lesson counts are preserved
in `research/results/stage9/semantic-contracts/archived-final-diagnosis.json`.

## Objective realizability

For binary class margin `m(i,j)=u_i+v_j`, diagonal and cross-pair sums are exactly
equal. Two positive diagonal margins and two negative cross margins contradict
this equality. A bilinear interaction can explicitly construct these margins.
Mechanical width4 fixtures verify both statements; the experimental standard
remains1024. The original slot loss samples nonedges as no-slot. A fixture shows
that real unordered edges must still receive no-slot supervision in the
edge-conditional alternative, while nonexistent edges receive no slot loss.
The edge-existence head remains independent and can suppress cross pairs at
inference. The theorem concerns the objective/head pair, not universal final
graph impossibility.

## Public observability and support

Read-only regeneration of **all10,128 archived descriptors /30,384surfaces** found:

- No observed identical public text mapping to incompatible alpha-normalized
  canonical graph targets.
- No observed lexical64-bit hash collisions, uncopyable required identities,
  unknown nonidentity values, node/slot overflow, or conflicting multiedge slots.
- Maximum37nodes,85edges,slot3,74tokens; actor encoding does not truncate tokens.
- Four variable-binding,144set-operation and9,980unification graphs in the full
  archived pool. The selected10ktraining pool includes4binding and120set graphs.
  Heldout64has24set/40unification and zero binding constructions.

An initial prefix-only audit was expanded after independent review caught the
extra64reserved corpus records. The committed audit now covers the entire DB.
These negative findings are finite-support checks, not a proof that every future
surface determines every compiler target. Canonical traversal and compiler-added
scope/identity nodes remain supplied label conventions. Alpha-renaming creates
surface diversity, not additional semantic construction diversity.

## Known confounds and missing coverage

- The public actor has a fixed canonical output ordering and a privileged
  compiler ontology; this is not unsupervised language acquisition.
- Stage8balanced edge/slot training does not justify treating zero logits as
  calibrated natural-prevalence probabilities. Per-relation train-only empirical
  calibration is retained; no guessed analytical constant is introduced.
- No fresh variable-binding structural transfer population exists in the frozen
  evaluation. Set-operation coverage is small and finite.
- Sixteen-bit/64-bit naming statistics, copying, canonical graph recovery and
  denotational equivalence are different targets; the current exact metric is
  compiler aligned.
- No Stage9experiment here tests programmable attention. These are supporting
  semantic interfaces; no composition or supervision withdrawal is permitted.
