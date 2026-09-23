# S09 proposal: tie predicted copying to the contextual node read

## Question and evidence

S08 shows that a copy output can become substantially occurrence-aligned without acquiring the corresponding ordered edge. The current model predicts copies from node states, but those pointer probabilities do not control the information read into the states consumed by graph heads. S06 changed the entire workspace into token states and underperformed; it did not preserve the original workspace while adding a pointer-conditioned contextual read.

The hypothesis is narrower than “more text attention helps”: making the predicted pointer itself retrieve contextual text into a retained workspace node may improve the learned surface-to-graph interface. No source graph is supplied, so this is not an experiment validating structural attention or programmable graph bias.

## Method and one intervention

Preserve the complete original eight-row workspace computation and canonical learned node queries. In a parallel stream, apply the same four existing blocks, twice, to projected public-token states. Weights are shared; no additional learned layers or parameters are added. Padded keys are masked and padded contextual residuals zeroed after each block.

Let `h` be the original workspace and `T` the contextual public-token states. The unchanged workspace decoder yields `n0`. The unchanged copy head yields logits `c(n0,text)`; masked `softmax(c)` produces the predicted pointer `p`. The new node state is:

```
r = W_out [p W_value T]
n = n0 + 0.5 r
```

`W_value` and `W_out` are the existing node decoder's value and output maps, reused to keep the contextual read in its existing value basis. The output bias is already present in `n0` and is not added twice. Scale .5 is fixed before outcomes and is not swept. Kind, scalar value, relation and ordered-slot heads consume `n`; the reported copy logits remain `c`. Thus the same learned pointer receives copy supervision and conditions the contextual read. Edge losses can train the pointer through that read.

The public-input and output contracts remain historical: all public tokens and positional features, 128 canonical query slots, predicted presence, predicted node kinds/values/copies/edges/slots. No gold node identities, spans, presence, edge positions or graph structure enter forward. Gold sampled pairs only select supervised output rows after latent computation. No occurrence-based candidate list or supplied syntactic segmentation is introduced; ordinary tokenization is unchanged.

Original first-identity copy targets remain unchanged. Only `ident` and `entity` nodes receive copy labels. Record/list/predicate/scope queries still retain their full free workspace residual and can learn contextual reads through their supervised heads; they are not forced to reconstruct their semantics solely from an identity pointer. This read is soft, not an exact symbolic write or programmed reference edge. S05 remains a separate engineered diagnostic if later applied.

Setting `context_read=False` dispatches exactly to the historical actor, with bitwise identical forward and gradients under common initialization. This is a mechanical reference, not evidence of acquisition. The proposed first main comparison is against the frozen matched S01-N8192 arm. It changes only the enabled contextual retrieval path, keeping the original first-copy objective; S07's occurrence objective is not combined.

## Primary methods inspected

[Dozat and Manning (2017)](https://arxiv.org/html/1611.01734v3) score directed dependencies from contextual word representations with role-specific projections and biaffine interactions. [Their semantic dependency extension (2018)](https://aclanthology.org/P18-2077.pdf) predicts graph edges and labels using contextual token states rather than requiring a single dependency tree. These motivate accessible, contextual node evidence; their word-aligned nodes are a stronger supplied correspondence than our learned canonical queries. We do not transplant their MST constraints, POS inputs, pretrained embeddings, or reported task claims. Existing topoformer edge/slot heads remain unchanged to avoid mixing a head replacement into this intervention.

The recurrent reuse follows the current implementation, not a claim that recurrence learns an algorithm. [Universal Transformers](https://arxiv.org/html/1807.03819v3) provide related per-position recurrent refinement, but no adaptive halting or theoretical universality claim is tested here.

## Considered representation contract

One candidate per public word/occurrence is a legitimate alternative parsing interface when tokenization and order are explicitly supplied. It provides candidate segmentation/alignment, not gold relation edges; non-token entities and scope/record nodes would still need a declared treatment. It would substantially simplify canonical node allocation compared with the current free query slots, so it cannot be presented as the same inference problem with only a better neural matcher. Using gold spans to create those candidates at inference would violate the public-input contract.

S09 deliberately retains the current canonical-query interface to isolate a predicted retrieval path. A future public occurrence-candidate study would be separately versioned, with its supplied segmentation/order and canonicalization algorithm disclosed. It is not silently folded into this experiment.

## Bounded acquisition and development plan

Use the existing 8,192 alpha-distinct English unification TRAIN constructions, seed201, width1,024, batch8, AdamW1e-4, same curriculum and pair sampling, and 65,536 presentations. No additional eight-example memorization run. The fixed first128 TRAIN constructions form the acquisition diagnostic panel throughout this real-diversity training, not a separately optimized or exhaustively fit corpus. Its exact full-graph outcomes are reported alongside fresh DEV512; this distinguishes observed fit on a fixed training panel from generalization without diverting another run to tiny-set memorization.

Freeze checkpoints at0/8,192/32,768/65,536 presentations. Preserve raw and TRAIN-calibrated outputs, complete graphs, presence/type/value/copy counts, typed and ordered edges, and component ceilings. Same first128 TRAIN calibration data and threshold rule as S01; never fit DEV thresholds. No endpoint selection. The 1,024 reserved confirmation constructions remain untouched.

Development promotion requires at least10% complete calibrated DEV graphs (52/512), at least five percentage points above S01, canonical copying >=.95 and ordered-edge F1 >.90. At least10% complete graphs on the fixed TRAIN128 panel is also required. These are search criteria for a promising branch, not a broad competence gate. If they fail, retain the result and localize errors; do not automatically extend exposure or launch confirmation. Nonfinite losses, failed provenance/replay contracts or the external time cap stop the run with an incomplete/failed record.

## Parameter and compute contract

Exactly57,853,781 total/active parameters, as the original width1,024 actor: the context encoder and read projection reuse existing weights. The variant keeps eight workspace rows plus54/64 contextual token rows. Its extra block applications, activation storage and arithmetic are substantial despite parameter equality.

Analytic dense projection/attention-product MAC estimates, excluding unchanged graph heads and elementwise work, are1.60–1.77G per example for the original backbone, versus7.35–8.58G for the proposed backbone plus context read. This is an estimate, not measured FLOPs or equal compute. Source numbers are recorded in `s09-compute-estimate.json`.

Before any main release, run20 updates at actual width1,024/batch8 with16+16 mechanical evaluations, measuring full occupancy, optimizer time, CUDA allocated memory and RSS. Provisional main refuses execution. Initial forecast is1,000–1,350 s for the full experiment, with a requested cap1,500 s; freeze or revise the budget only after profile and coordinator approval. Do not silently reduce width or exposure.

If the path is promising, an extra-workspace-pass control is required before confirmation. Keep the original actor and objectives; use9 recurrent microsteps for padded token length54 and10 for64, the nearest dense-MAC match selected by the declared arithmetic formula before outcomes. These controls are within roughly−1.9%/+3.3% of the proposed backbone/read MAC count; actual runtime may differ substantially because small recurrent matrices launch more kernels. Report both measured time and estimated arithmetic, never call them exactly equal FLOPs or wall time. This conditional control requires its own source/profile/budget release and cannot be omitted from an attribution claim.

## Mechanical verification and scope

Width16 CPU fixtures are explicitly mechanical: common parameters and off-path forward/gradients are bitwise identical; variable-length padding is invariant; sampled/dense edges agree; downstream edge loss reaches the copy query. No primary model uses width16.

The corpus has only two ordered tree shapes at structural depth3, despite many equality/reference patterns. Any successful new-construction result remains limited to that support. Supplied canonical output conventions, dense privileged supervision and programmed identity-equality diagnostics must remain distinct from learned parsing and future confirmation.

## Measured profile (mechanical, not acquisition evidence)

The fixed20-update seed901 profile used width1,024 and batch8. Optimizer time was2.00330 seconds; full wrapper occupancy11.33561 seconds (inner process9.70415). Peak allocated CUDA memory1,278,865,408 bytes; maximum process RSS2,960,496KiB. All57,853,781 parameters are active. Projection for8,192 updates is approximately821 optimizer seconds plus170–250 seconds for full evaluation/serialization; proposed main cap1,500 seconds awaits coordinator approval. Small16-row profile evaluation is mechanical and is not a substitute for the main512-row development set. Receipt and manifest are archived under `research/results/campaign-01/semantics/s09-grounded-profile`.
