# Stage 8: full-width protected semantic interfaces

**In progress. Default latent width1024. No composition or supervision withdrawal.**

Stage8 tests learned evidence updates and return readout independently. It also separates procedural semantic corpus size from actual optimizer exposure. Baseline Stage1–7 files remain unchanged.

## Supplied mechanisms versus learned interfaces

- Belief: public typed joint candidates, role-bearing evidence, and observation IDs are supplied. Protected storage adds learned log-evidence contributions, makes repeated-ID writes idempotent, and retracts stored contributions exactly. Compatibility/posteriors remain learned. A full-width four-phase recurrent MLP is the comparator; it is not a widened Stage7 transformer. Exact support intersection is an oracle only.
- Return: correct typed return events are supplied. Mixed additive, disjoint concatenation, and facet-token encodings are compared with once-only and persistent access. Concat/facet share parameters and1024 informative coordinates; facet KV allocates six memory tokens. Mixed encoding differs in encoder capacity, disclosed separately. Semantic outputs come from the learned workspace.
- Semantic graphs: pinned procedural language/compiler, typed graph targets and direct curriculum supervision. Corpus cardinality, distinct graphs visited, optimizer presentations, tokens and renderer exposure are distinct logged quantities. No runtime execution.

## Acquisition observations (not competence)

Protected belief acquisition has strong final answers on clean/retraction/contradiction but fails the empty-prior posterior check. Generic recurrence struggled with retraction in the short acquisition probe; the longer frozen diagnostic subsequently acquired ordinary clean/retraction updates. Initial main ID-dependent controls are confounded because the generic encoder does not consume IDs; preserve them and use the separately frozen ID-matched followup for strict comparisons.

All six return arms memorize32fixed events atzero/one step. Fresh128-example joint reconstruction remains0–3/128, so this only establishes fitting capacity. The completed fresh-data main study is reported below; fixed-set acquisition is not its success criterion.

The width1024 semantic model improves node/type/identity recovery on eight training graphs, but overpredicts edges and has zero exact graph recovery. Raw thresholds and a separate train-only edge-threshold calibration will both be reported; no heldout threshold search is permitted.

## Reproducibility and gates

See [plan](stage8-plan.md), [decisions](stage8-decisions.md), [gate registry](stage8-gates.json), and [independent review](stage8-review.md). Every experiment freezes config/source before final evaluation. Timing/acquisition and main outcomes remain distinct. CUDAdependency freeze is under `results/stage8/cuda-requirements.txt`; GPU is remote NVIDIA GB10, PyTorch2.14.0+cu130. No local training or mutation of previous environments.

Final paired results, raw metric audits and integration status will be appended after the queued studies finish.

## Initial full-width diagnostic (ID-access caveat)

After1000updates, both protected and generic recurrent models reach roughly99.6–100% clean/retraction final answers. Long repeated evidence is less stable in generic recurrence, but ID-access differs in this diagnostic, so use the completed matched followup below for strict stress-test comparisons. Both models still fail broad posterior/prior/null gates. This does not support a general claim that generic latent recurrence cannot acquire evidence updates. It also does not isolate width as the sole cause of improvement relative Stage7, because the task/interface and training recipe differ.

## Completed return main (independently audited)

All 18 runs finished at width 1024; none passes the prespecified retention gate. With eight distractors, three-seed mean test joint reconstruction for factorized persistent memory is 89.19% at one step, 75.59% at 16 steps, and 61.98% at 32 steps. Concatenated persistent memory is 70.25%, 67.51%, and 67.90% respectively. Thus factorization improves short/intermediate-delay readout in this recipe but does not dominate at longer recurrence.

At 16 steps, factorized persistent scalar accuracy is 79.04% (MAE 0.243), type 99.54%, and operation 97.98%. Scalar accuracy uses the frozen exact discrete-value criterion; MAE is a diagnostic, not a replacement gate. Protected storage remains separate from learned readout. Downstream returned-fact use stays blocked. See the subsystem report for every seed, field, delay and intervention.

Independent audit reconstructed all field/joint counts across 1,008 rows and 516,096 example evaluations and confirmed all 18 failed gates. Frozen config and source hashes match.

## Completed observation-ID-matched belief primary

The [belief report](stage8-belief-report.md) now includes all six primary runs with identical public observation-ID features in both encoders. Protected final support accuracy is 100%; recurrent clean accuracy is 99.80–100% and eight-fold repeated-ID accuracy is 98.05–99.80%. Repeated-ID mean posterior L1 is .0011–.0071 protected versus .0685–.1274 recurrent. The ledger supplies idempotence and retraction; these are not learned bookkeeping rules.

Both strict gates fail: 14/48 protected and 24/48 recurrent validation cells. Protected failures are localized to initial null/prior probability; excluding the initial frame gives worst posterior L1 .00244, but this is a posthoc diagnosis, not a relaxed gate. Independent audit reconstructed 192 cells / 98,304 episode evaluations. Ordinary updates are learned in this candidate-local residual interface; this does not establish a general property of transformer recurrence.

## Interpretation and next diagnostics

Persistent return access tests repeated learned retrieval/readout of an intact record, not memory disappearance. The one-to-sixteen-step marginal decline is not a measured forgetting probability. Paired correct/wrong transitions and conditional non-value/value accuracy should be computed from matched examples. Before changing value encoding, isolate field encoding, mixed memory, initial retrieval and late workspace readout using disjoint probe training/evaluation sets. A supplied exact initial belief prior is a separate architectural-prior control, not learned calibration. These followups do not alter completed gates or the frozen semantic sweep.

Completed reports and raw evidence: [beliefs](stage8-belief-report.md), [returns](stage8-return-report.md), [independent audit](stage8-review.md). Semantic scaling remains in progress.
