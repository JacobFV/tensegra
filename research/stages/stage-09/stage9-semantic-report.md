# Stage 9 semantic contracts: complete acquisition remains blocked

**Outcome:** the slot objective mismatch is real and repairable in a mathematical
fixture, but no tested learned decoder reaches complete fixed-set acquisition
within the declared budget. Public-text training and fresh/renderer transfer
therefore remain **blocked**. No larger semantic sweep ran.

## What was reproduced before new learning

The [localization report](stage9-semantic-localization.md) maps historical source
and reconstructs3,072final Stage8prediction rows independently. The separate
Stage9review reconstructs all18,944historical rows. These are archived metric
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
both cross margins negative. The Stage8all-sampled-pairs slot loss can demand
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

Each of four arms trains300updates on all8graphs, for2400graph presentations,
with development seeds101/102. There are no heldout generalization claims.
Source, config, data/compiler, initial/final model and checkpoint hashes are
saved, as are losses and complete predicted edges/slots. Historical source bytes
are archived under the result directory. No threshold or exposure was changed
following outcomes.

The first diagnostic accidentally simplified the historical edge projections to
bias-free linear maps. We retained that result and froze a separate faithful
check restoring the Stage8affine edge projections, with all other settings and
budgets unchanged. Both outcomes are below.

| Head / objective | Bias-free total loss (101/102) | Affine total loss (101/102) | Exact graphs, each seed |
|---|---|---|---|
| Additive / all pairs |.2786/.2772|.2781/.2756|0/8|
| Additive / edge conditional |.0826/.0814|.0820/.0796|0/8|
| Interaction / all pairs |.0124/.0116|.0120/.0098|0/8|
| Interaction / edge conditional |.0110/.0101|.0105/.0083|0/8|

All16runs have **zero slot errors on real edges**. The bias-free reference has
614/495edge errors in seeds101/102; the faithful affine check has445/354. Within
one seed these counts are identical in all four arms because the independently
parameterized edge head receives the same initialization and objective.
Thus lowering the slot objective does not improve complete graph acquisition
in this diagnostic. It would be incorrect to call component-loss improvement
semantic competence.

Frozen CPU inference on the bias-free checkpoints localizes all errors to false
positive edges, with zero false negatives. Positive/negative score ranges overlap
in five of eight graphs per seed, so a global threshold shift does not trivially
repair those outputs. This inference diagnosis does not establish the unique
cause of failed optimization.

A separate explicitly privileged construction sets edge weights from gold
adjacency while retaining frozen learned slots. It yields8/8complete graphs in
both seeds. The same affine/bilinear head can therefore represent this finite
fixture. These programmed weights are a capacity ceiling; they neither count
as acquired graph reconstruction nor override the failed learning gate.

## Resources and provenance

The faithful additive decoder has1,904,450parameters; interaction has6,360,898.
Bias-free variants have1,902,658/6,359,106. Pair interaction rank128is an output
head rank, not a smaller experimental workspace. The oracle diagnostic has
node features of width1024 and no learned recurrent workspace or memory tokens.
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

The bounded result identifies a slot objective mismatch and a separate learned
edge-acquisition bottleneck. Available evidence does not isolate whether another
optimizer, exposure budget, objective weighting or conditioning would solve the
latter; no further speculative sweep was run. It does not support claims that
neural recurrence is intrinsically incapable of structured state. This entire
track concerns supporting semantic interfaces, not programmable attention.
