# Stage 8: full-width protected semantic interfaces

**In progress. Default latent width1024. No composition or supervision withdrawal.**

Stage8 tests learned evidence updates and return readout independently. It also separates procedural semantic corpus size from actual optimizer exposure. Baseline Stage1–7 files remain unchanged.

## Supplied mechanisms versus learned interfaces

- Belief: public typed joint candidates, role-bearing evidence, and observation IDs are supplied. Protected storage adds learned log-evidence contributions, makes repeated-ID writes idempotent, and retracts stored contributions exactly. Compatibility/posteriors remain learned. A full-width four-phase recurrent MLP is the comparator; it is not a widened Stage7 transformer. Exact support intersection is an oracle only.
- Return: correct typed return events are supplied. Mixed additive, disjoint concatenation, and facet-token encodings are compared with once-only and persistent access. Concat/facet share parameters and1024 informative coordinates; facet KV allocates six memory tokens. Mixed encoding differs in encoder capacity, disclosed separately. Semantic outputs come from the learned workspace.
- Semantic graphs: pinned procedural language/compiler, typed graph targets and direct curriculum supervision. Corpus cardinality, distinct graphs visited, optimizer presentations, tokens and renderer exposure are distinct logged quantities. No runtime execution.

## Acquisition observations (not competence)

Protected belief acquisition has strong final answers on clean/retraction/contradiction but fails the empty-prior posterior check. Generic recurrence struggled with retraction in the short acquisition probe; the longer frozen diagnostic subsequently acquired ordinary clean/retraction updates. Initial main ID-dependent controls are confounded because the generic encoder does not consume IDs; preserve them and use the separately frozen ID-matched followup for strict comparisons.

All six return arms memorize32fixed events atzero/one step. Fresh128-example joint reconstruction remains0–3/128, so this only establishes fitting capacity. Fresh-data main studies are required.

The width1024 semantic model improves node/type/identity recovery on eight training graphs, but overpredicts edges and has zero exact graph recovery. Raw thresholds and a separate train-only edge-threshold calibration will both be reported; no heldout threshold search is permitted.

## Reproducibility and gates

See [plan](stage8-plan.md), [decisions](stage8-decisions.md), [gate registry](stage8-gates.json), and [independent review](stage8-review.md). Every experiment freezes config/source before final evaluation. Timing/acquisition and main outcomes remain distinct. CUDAdependency freeze is under `results/stage8/cuda-requirements.txt`; GPU is remote NVIDIA GB10, PyTorch2.14.0+cu130. No local training or mutation of previous environments.

Final paired results, raw metric audits and integration status will be appended after the queued studies finish.

## Initial full-width diagnostic (ID-access caveat)

After1000updates, both protected and generic recurrent models reach roughly99.6–100% clean/retraction final answers. Long repeated evidence is less stable in generic recurrence, but ID-access differs in this diagnostic, so strict stress-test attribution awaits the matched followup. Both models still fail broad posterior/prior/null gates. This does not support a general claim that generic latent recurrence cannot acquire evidence updates. It also does not isolate width as the sole cause of improvement relative Stage7, because the task/interface and training recipe differ.
