# Stage 9 semantic contracts: complete acquisition remains blocked

**Outcome:** the slot objective mismatch is real and repairable in a mathematical
fixture, but every learned run fails the prespecified zero-logit edge decoder. A later
TRAIN-only diagnostic recovers every development graph using relation-specific
thresholds. A separately frozen three-seed confirmation then fits8/8 graphs in
two seeds and 6/8 in one, failing the calibrated all-seed criterion. This separates
score acquisition, decoder calibration and residual seed robustness. Public-text training and fresh/renderer transfer
therefore remain **blocked**. No larger semantic sweep ran.

## What was reproduced before new learning

The [localization report](stage9-semantic-localization.md) maps historical source
and reconstructs3,072final Stage 8prediction rows independently. The separate
Stage 9review reconstructs all 18,944historical rows. These are archived metric
reconstructions, not new inference. Complete recovery remains zero, with complete
copy and edge components also zero. No single corrected component makes the
frozen outputs exact.

The entire frozen10128-construction database was regenerated read-only across
30384surfaces. No observed incompatible same-text target, lexical hash collision,
uncopyable identity, unknown value, capacity overflow or conflicting pair-slot
label was found. These checks cover finite support; they do not prove general
identifiability. The heldout pool contains no variable-binding constructions.
[Machine-readable audit](../../results/stage9/semantic-contracts/observability.json)
records support and split counts.

## Mathematical head/objective contract

An additive source/target class margin obeys
`m(a,c)+m(b,d)=m(a,d)+m(b,c)`. It cannot make both diagonal margins positive and
both cross margins negative. The Stage 8all-sampled-pairs slot loss can demand
exactly this pattern. A bilinear interaction has an explicit exact construction.
Four CPU tests verify the identity/construction, preservation of unordered-real-
edge supervision, exclusion of nonedges under conditional loss, and fail-fast
handling of incompatible multiedge slot labels. Width4is an explicitly
mathematical fixture; primary node representation width remains1024.

This does not make final graph decoding universally impossible: the separate
edge-existence head can suppress cross pairs. Indeed, every learned diagnostic
below reconstructs every **real-edge slot** correctly despite failing whole graphs.

## Privileged fixed-node acquisition

Eight fixed graphs receive distinct graph/node one-hot codes padded to1024.
Node presence, type, value and copy identities are supplied exactly. Learned
heads predict full directed relation existence and ordered slots. Gold edges
select conditional training loss and score metrics; deployment uses predicted
edge logits at the prespecified zero threshold. This is privileged decoder
memorization, not semantic induction or a recurrent-workspace experiment.

Each of four arms trains300 updates on all 8 graphs, for2400graph presentations,
with development seeds 101/102. There are no heldout generalization claims.
Source, config, data/compiler, initial/final model and checkpoint hashes are
saved, as are losses and complete predicted edges/slots. Historical source bytes
are archived under the result directory. No threshold or exposure was changed
following outcomes.

The first diagnostic accidentally simplified the historical edge projections to
bias-free linear maps. We retained that result and froze a separate faithful
check restoring the Stage 8affine edge projections, with all other settings and
budgets unchanged. Both outcomes are below.

| Head / objective | Bias-free total loss (101/102) | Affine total loss (101/102) | Exact graphs, each seed |
|---|---|---|---|
| Additive / all pairs |.2786/.2772|.2781/.2756|0/8|
| Additive / edge conditional |.0826/.0814|.0820/.0796|0/8|
| Interaction / all pairs |.0124/.0116|.0120/.0098|0/8|
| Interaction / edge conditional |.0110/.0101|.0105/.0083|0/8|

All 16 runs have **zero slot errors on real edges**. The bias-free reference has
614/495edge errors in seeds 101/102; the faithful affine check has445/354. Within
one seed these counts are identical in all four arms because the independently
parameterized edge head receives the same initialization and objective.
Thus lowering the slot objective does not improve complete graph acquisition
under the registered raw decoder. It would be incorrect to call component-loss improvement
semantic competence.

Frozen CPU inference on the bias-free checkpoints localizes all errors to false
positive edges, with zero false negatives. Positive/negative score ranges overlap
in five of eight graphs per seed, so a global threshold shift does not trivially
repair those outputs. This inference diagnosis does not establish the unique
cause of failed optimization.

A separate explicitly privileged construction sets edge weights from gold
adjacency while retaining frozen learned slots. It yields8/8 complete graphs in
both seeds. The same affine/bilinear head can therefore represent this finite
fixture. These programmed weights are a capacity ceiling; they neither count
as acquired graph reconstruction nor override the failed learning gate.

## Resources and provenance

The faithful additive decoder has1,904,450parameters; interaction has6,360,898.
Bias-free variants have1,902,658/6,359,106. Pair interaction rank 128is an output
head rank, not a smaller experimental workspace. The oracle diagnostic has
node features of width 1024 and no learned recurrent workspace or memory tokens.
Attention cost is not compared here.

Both bounded development studies together consumed38,400graph presentations and
4,800optimizer updates. Recorded model-run times total90.19seconds; the profile
adds1.07seconds. Filesystem start/final-results timestamps bound the two run
processes at approximately51/52seconds, including JSON/checkpoint overhead.
Peak CUDA allocation392MB; process maximum RSS about1.97millionKiB is a separate
quantity, not device capacity. Existing GB10/PyTorch2.14CUDA environment was reused.
No core dependencies changed. Compressed JSON predictions retain exact integer
outputs and hashes; full checkpoints remain under
`gb10-direct:~/topoformer-stage9-semantics/results/`.

## Gate decisions and interpretation

- Mathematical realizability: **passed_restricted**, constructive mechanical
  proof for the interaction fixture. This is not learned semantic acquisition.
- Fixed-set complete learned acquisition: **failed**, all eight affine runs and
  all eight original diagnostic runs fail the prespecified exact-fit condition.
- Fresh known-renderer generalization: **blocked**.
- Heldout renderer/lexicon/structure transfer: **blocked**; missing variable-binding
  support remains explicit.
- Autonomous composition and supervision withdrawal: **blocked**.

The bounded result identifies a slot objective mismatch and a raw edge-decoding
failure. The posthoc relation-wise diagnostic below establishes that the fixed
training graphs are separable in the learned scores, so the raw failure must not
be summarized as an absence of learned edge information. Fresh calibration and
generalization remain untested; no further training or speculative sweep ran. It does not support claims that
neural recurrence is intrinsically incapable of structured state. This entire
track concerns supporting semantic interfaces, not programmable attention.

Independent review reconstructed all 512 recorded fixed-node rows across16 runs,
verified identical paired initialization, recomputed edge/slot/complete-graph
counts, and rehashed all 16 durable checkpoints. See
[raw-metric audit](../../results/stage9/audits/semantic-fixed-node.json) and
[checkpoint audit](../../results/stage9/audits/semantic-fixed-node-checkpoints.json).
The8 fixed constructions comprise3 binding,3 unification and2 set examples; repeated
optimizer presentations and both seeds do not increase that semantic support.


## Posthoc frozen-score calibration: an important localization correction

A final read-only check uses the **affine** checkpoints and only their eight
TRAIN graphs. For each relation independently, it sorts all training edge logits
and chooses the boundary minimizing pair-classification errors (ties choose the
lowest threshold). This is an explicitly posthoc diagnostic, not the frozen raw
decoder and not a retrospective gate pass. No test labels or new optimizer steps
are used. Every relation with positive training examples is strictly separable
within each seed; the one absent relation has no positive support and establishes
no positive-edge competence.

| Development seed | Raw exact graphs | Per-relation TRAIN-calibrated exact | Raw edge errors | Calibrated errors |
|---|---:|---:|---:|---:|
|101|0/8|8/8|445|0|
|102|0/8|8/8|354|0|

The independently parameterized edge heads and true-edge slot outputs agree
across all four arms within a seed, so this ceiling applies to each arm. This is
complete fitting of a tiny privileged training set with additional calibrated
readout parameters, not fresh semantic induction. The global positive/negative
overlap reported earlier **does not imply overlap within each relation**. The
learned edge scores contain sufficient information to separate this fixed set;
raw thresholding was an important remaining contract mismatch.

### Actual loss weights and analytical correction

For each complete c2 graph, the exact objective averages positive binary edge
entries and negative entries separately, then averages their two losses. Thus
`w+ = 1/(2 N+)`, `w- = 1/(2 N-)`, where entries include every node pair and every
relation. At an ideal weighted population-risk optimum,

`q = logit(p) + log(w+/w-)`, hence

`logit(p) = q + log(N+/N-)`.

There is **no guessed universal constant**: per-graph counts, weights and
corrections are logged. Correcting these finite trained logits with the actual
gold graph prevalence leaves156/168edge errors and0/8 complete graphs in seeds
101/102. The population identity does not guarantee calibration of a finite
trained model. Gold prevalence is unavailable at deployment, making this an
explicitly privileged analytical diagnostic.

Stage 8's sampled objective has a further selection distinction. If a graph has
`P` positive node pairs, `M` entirely negative pairs, `K=min(128,M)` sampled
negative pairs and `R` relation types, its selected negative-entry count is
`R(P+K)-N+`. Negative relation entries on positive pairs are always included;
entirely negative pairs have inclusion probability `K/M`. Consequently effective
negative weights differ by pair stratum. The corresponding risk correction must
include these inclusion probabilities as well as class weights; neither the c2
constant nor one guessed global shift can be transplanted to Stage 8. These strata
and prevalence involve gold topology, so no deployable analytical correction is
claimed. The empirical TRAIN-only relation decoder above is reported separately.

Full per-relation min-positive/max-negative scores, thresholds, class support,
per-graph signed-count corrections and exact outcomes are retained in
[calibration records](../../results/stage9/semantic-contracts/relation-calibration.json).
No new training, public c3 run or composition was authorized by this diagnosis.


## Prospective calibrated-contract confirmation

The posthoc result motivated one separately preregistered contract, source
`30e36ed`, before any new outcomes: four head/objective arms, three new paired
initializations10/11/12,8 fixed graphs generated from seed9001000, the same
width 1024/rank 128/300 updates, and the identical final-TRAIN threshold rule.
No threshold fitting at intermediate checkpoints, per-graph thresholds or
additional exposure was allowed. Independent review approved the policy before
launch. The new restricted gate required8/8exact graphs in every one of12 runs.

| Initialization | Raw complete graphs, every arm | TRAIN-calibrated complete graphs, every arm | Remaining edge errors |
|---|---:|---:|---:|
|10|0/8|6/8|5|
|11|0/8|8/8|0|
|12|0/8|8/8|0|

All 12 runs again have perfect slots on real edges. Seed10's remaining errors are
all `argument` relation entries, four on construction9001004and one on9001007.
The complete-graph result is identical across the four arms within each seed;
changing the slot head/objective does not resolve that independent edge-score
error. **The prospective calibrated gate fails.** Public c3 remains blocked;
there is no subsequent optimizer or corpus expansion.

This confirms that calibrated readout can turn much of the raw failure into
complete fixed-set reconstruction, but it does not establish robust acquisition
across every prescribed seed. The8 graphs are a new fixed fixture, not a heldout
semantic test: one overlaps the prior c2fixture and five overlap the Stage 8pool
under alpha-normalized graph equality. They comprise2 binding,3 set and3 unification
constructions. No512-example generalization claim is made from tiny fixed support.

The confirmation adds28,800graph presentations,3,600updates and68.37seconds of
recorded model time. [Per-seed results and source/config/checkpoint hashes](../../results/stage9/semantic-contracts/calibrated-confirmation-summary.json)
retain every arm. Raw predictions, threshold records and targets are compressed
losslessly. The historical raw gate and earlier failures remain unchanged.

Independent confirmation review reconstructed 480raw/calibrated rows and replayed
all 12 frozen checkpoints on CPU. An independent Python threshold sweep matched
all 156relation counts and 96calibrated graph outcomes; threshold differences were
at most 4.77e-7from floating-point inference. All 12 checkpoint hashes matched.
[Confirmation audit](../../results/stage9/audits/semantic-calibrated-confirmation.json)
and [independent calibration replay](../../results/stage9/audits/semantic-calibration-replay.json)
confirm the failed all-seed criterion without changing the registered gate.
