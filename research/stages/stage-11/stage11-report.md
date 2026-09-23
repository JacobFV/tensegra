# Stage 11: trained horizons and public-text acquisition

Stage 11 preserves the ID-separated belief reference, demonstrates complete
public-text acquisition on a tiny fixed set, and finds a consistent benefit from
longer-horizon return training. Neither fresh semantic transfer nor the complete
return-retention gate passes. Autonomous composition and supervision withdrawal
remain blocked.

Baseline: `e10b80a`. All experimental workspaces are width **1024**. No memory
system, numeric encoding, runtime primitive, or structural-attention mechanism
was added. [Registry](stage11-registry.json), [claim map](stage11-claim-map.md),
and [independent review](stage11-review.md) distinguish the independent tracks.

## Beliefs: reference frozen

The preferred restricted reference is the Stage 10 ID-separated protected
aggregator, retaining all three checkpoints and their provenance. Observation
handles still address ledger entries, but their spelling cannot affect semantic
matching. This programmed invariance does not guarantee correct beliefs, and it
does not repair the earlier unrestricted recurrent comparator. No belief
training or inference was repeated. Readiness and execution need a separate
confidence-conditioned evaluation. See the [reference manifest](stage11-belief-reference.md).

## Returns: training support helps, exact scalar access still limits the interface

Both arms continue the same archived Stage 9 backbones (initialization seeds
10/11/12), resetting AdamW identically. Each sees 32,000 fresh events over 1,000
updates, paired exactly between arms. Architecture and all six cross-entropy
losses remain unchanged. The short schedule is `0/1/2/4/2/4`; the wide schedule
is `0/1/2/4/8/16`. This changes coverage and recurrent compute: 69,280 versus
164,896 example-microsteps per arm, about **2.38×**. It is not FLOP matched.

The independent prerequisite diagnostic fits all 32 fixed events at every
covered delay in both arms. Its substantially worse fresh-event scores are
reported separately; fitting did not count as generalization.

The table below reports mean exact accuracy across the three backbones on the
same 512 test events with eight distractors. These are 512 distinct events,
reused across seeds, arms and delays, not 1,536 independent constructions.

| Interface | Scalar at 0 | Scalar at 16 | Scalar at 32 | Full joint at 16 | Full joint at 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen original | 83.07% | 73.50% | 66.02% | 70.57% | 58.59% |
| Frozen Stage 10 shared readout | 75.91% | 94.73% | 90.62% | 89.97% | 78.84% |
| Short continuation | 91.34% | 72.85% | 54.10% | 66.73% | 44.53% |
| Wide continuation | 92.90% | 88.02% | 83.20% | 86.85% | 81.25% |

Wide improves the late scalar and joint scores over short continuation in every
paired backbone. It also improves over the original frozen model at these late
delays. Short continuation itself degrades late performance, so the wide–short
gap must not be presented without the frozen references. The frozen shared
readout remains stronger on scalar accuracy; it has a different fitting history
and is a contextual reference, not an objective-matched causal arm.

For wide continuation, **every non-value field passes its inherited criterion
at 16 updates in all twelve seed × split × distractor cells**. Exact scalar
accuracy fails in all twelve. Thus the complete gate remains failed. At 32,
non-value joint accuracy is 97.66% for wide versus 80.27% for short; full joint
accuracy is still only 81.25%. Zero-step reconstruction also remains imperfect.

Sixteen is a trained horizon for wide, not extrapolation. Thirty-two is excluded
from both arms' training and selection. Fresh contexts and identities use the
same 33 half-unit scalar classes; this does not test unseen numeric labels or
numerical ranges. These observations support improved learned access under
broader recurrent training, not a proof of information destruction, a stable
universal readout, or a newly acquired symbolic execution policy.

See the [return report](stage11-return-report.md) for every seed, condition,
field, paired outcome and failure, and the [protocol](stage11-return-design.md)
for the frozen comparison.

## Semantics: acquisition passes; construction transfer fails

The unchanged public-text actor receives English text and its derived copy
inventory, with canonical graphs used only as privileged supervision. It fits
eight alpha-distinct unification constructions. The decoder uses edge-conditional
slot supervision; deployment predicts presence and edges without gold masks.

All three seeds (30/31/32) reach **8/8 exact complete graphs** at the prescribed
4,000-update endpoint, under both raw and training-calibrated decoding. The
matched frequency baseline gets 0/8. Each model saw 32,000 presentations but
only eight distinct constructions. This is complete fixed-set acquisition,
which may be memorization.

A separately preregistered frozen test then evaluates 512 alpha-disjoint
constructions, using unchanged weights, vocabulary and training thresholds.
All targets are supported and publicly observable under the checked contract;
there are no novel lexical tokens. **Every seed gets 0/512 exact graphs**, under
raw and calibrated decoding. Calibrated typed-edge F1 is .528/.570/.574 and copy
accuracy .291/.299/.315. Neither component score substitutes for complete graph
recovery. Gold replacement of any single component, or all node attributes
together, still leaves exact recovery at zero.

The model can acquire the eight complete text-to-graph targets, but transfer to
new constructions is not acquired from this tiny set. The failure cannot be
attributed only to an unsupported target vocabulary or one decision threshold.
It does not establish failure under adequate semantic diversity/exposure.
Further renderer/lexicon tests and additional training stop at the failed fresh
gate. See the [semantic report](stage11-semantic-text-report.md) and
[prospective fresh protocol](stage11-semantic-fresh-design.md).

## Verification, compute and next decision

Independent audits reconstruct the return matrix, semantic predictions,
training-only thresholds, component-replacement diagnostics, event pairing and
provenance. All **3,866 baseline tracked files remain byte-identical**. The full
regression snapshot `d8b6811` passes **567 tests and 6 subtests**; subsequent
reporting and archived-data audit changes do not alter the tested model or test
sources.

Recorded main return job time is 545.23 seconds. Semantic acquisition occupies
approximately 826.86 seconds including export, and frozen fresh evaluation
39.07 seconds. Together with development/profile work, measured experiment time
is approximately 24.4 minutes; a conservative **30-minute charge** remains well
below the 180-minute ceiling. These are job-time accounting figures, not GPU
kernel profiling or FLOP measurements. GB10 was used with existing environments;
no core dependency upgrade was made.

The next decision should retain the ID-separated belief reference, isolate the
remaining scalar consumer problem without discarding the improved non-value
interface, and test semantic diversity separately from repeated presentations
of eight examples. Those are recommendations, not newly launched experiments.
This stage contains no new `QK` versus `QK + B_G` comparison and provides no new
direct evidence for programmable attention. Its isolated interface results do
not authorize runtime composition, supervision withdrawal, or pretrained-model
integration.
