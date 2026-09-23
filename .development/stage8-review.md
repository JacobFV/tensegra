# Stage 8 independent review

Status: implementation review in progress. No scientific result or competence pass is established by this document yet.

## Prespecified review boundary

The standard latent width is **1024**. Smaller smoke-test widths are implementation checks, not the primary scientific configuration. Stage 1–7 source and artifacts must remain unchanged. No runtime composition or supervision withdrawal is authorized by an isolated acquisition probe.

The review separates architectural state protection, privileged supervision, learned evidence updates, exact oracle updates, and learned semantic readout. Exact storage integrity is not learned return retention; declining posterior entropy is not correctness.

## Required checks before main experiments

- Candidate-belief models receive the same public evidence and supplied hypothesis set. Private posterior/support targets may enter losses and evaluation only. Candidate permutation and randomized identity controls prevent fixed class indices from substituting for structured matching.
- Learned additive updates and unrestricted recurrence are compared against an explicitly labeled exact oracle. Duplicate evidence, contradictions and retractions have declared semantics. Replaying evidence must not silently count as independent evidence when the generator defines it as the same observation.
- Compressed versus factorized return encoding is separated from once-only versus persistent access. Total capacity, model parameters and compute are reported. Semantic heads decode learned state, without privileged exact field bypasses. Zero/one-step reconstruction precedes conclusions about long-delay retention.
- Corpus cardinality and actual optimizer exposure are distinct axes. Actual unique constructions, duplicate/saturation rates, renderer exposure and optimizer tokens are recorded. Heldout graph structure is not exposed through decoder capacity or privileged spans.
- Gate checks require every prespecified seed and condition, with empty or missing matrices failing. Validation selects gates/calibration; untouched test outcomes report generalization. Existing failed gates block composition.

## Findings and dispositions

Review pending new source. Stage 7 fixed-index progressive hypotheses motivate an explicit Stage 8 candidate-permutation test. This is a proposed control, not a claim that the Stage 7 failure was caused by that feature.

### Initial source review

- Return memory: public event fields enter learned encoders and workspace attention; semantic heads read workspace only. Compressed and six-token arms share field encoders and parameters. Both already factorize fields into disjoint coordinates, so this comparison isolates tokenized versus concatenated facets, not factorization versus a wholly undifferentiated encoder. Persistent access remains an independent intervention. Six-token memory uses more attention allocation despite equal informative coordinates.
- Belief prototype: candidate roles and observation IDs are public priors; exact ledger insertion/retraction is architecturally supplied, while compatibility is learned. The initial proposed width-1024/inner-128 MLP bottleneck and lack of attention were escalated for an explicit design decision. Random candidate records risk mostly immediate disambiguation; posterior support progression must be audited.
- Belief metrics: initial reliability code scored support membership as correctness. This incorrectly assigns accuracy one to any oracle-supported choice while oracle confidence is `1/K` under ambiguity. Requested calibration against realized latent truth or posterior-expected correctness, retaining support accuracy as a distinct metric. Main experiments require separate validation/test splits and posterior fidelity across frames.
- Semantic prototype: sampled edge pairs derive from gold only to choose supervised loss positions; this is privileged optimization, not actor conditioning. The no-input baseline still exposes copy inventory/length and must disclose that boundary. Corpus-dependent output vocabularies can change parameter count and need a shared vocabulary or explicit accounting.

These are preliminary findings before test completion or training, not final dispositions.

### Pre-timing dispositions

Return source `21d4382`: timing cleared. Pilot gate now requires at least 512 examples per prescribed condition; initialized heldout zero/one-step metrics and corrupted-event readback are recorded. A separate `mixed` additive arm has six full-width field encoders (6144 pre-sum coordinates at width1024), unlike the matched 1024-coordinate compressed/factorized pair. Its larger encoder parameter count must remain explicit.

Semantic source `93c9138`: timing cleared. Common training-only vocabulary pool keeps the 1k/10k categorical head identical; decoder capacity stays fixed. No-text-feature control deliberately retains copy inventory metadata, now disclosed. Gold-selected edge indices affect supervised score selection only. A small fixed-set graph acquisition probe is requested before attributing failures to corpus scaling.

Belief draft fixes reviewed: latent width1024 and feed-forward width2048; expected posterior correctness replaces support-membership calibration; validation/test seeds separate; gate requires framewise posterior L1 fidelity in addition to final support metrics. This is explicitly a candidate-local four-phase residual MLP comparison, not a recurrent attention transformer replication. Shared candidate alternatives make progressive ambiguity substantive. Candidate role selection, evidence-ID deduplication and retraction remain supplied priors. Final tests/source commit pending.

No main experiment budget, scientific gate pass, or composition clearance is implied by timing clearance.

Additional pre-outcome checks: return `8b0f3a0` aligns all shared backbone parameters across the mixed/concatenated/facet arms by constructing unequal encoders last; tests assert equality. A stdlib-only independent generator audit on 1024 episodes (seed200000) found mean posterior support sizes 8→4.465→2.688→1.310→1 for eight candidates and 16→8.442→4.688→1.431→1 for sixteen. The revised generator therefore contains genuine staged ambiguity rather than almost immediate identification.

## Acquisition artifact audit

Audited committed/local acquisition artifacts independently using Python's standard library. Return acquisition contains 18 heldout evaluation rows across six arms; every saved field, scalar-joint, identity-joint and full-joint count exactly matches its raw predictions and targets. All six arms fit the 32-example training set perfectly at zero/one update, but heldout joint reconstruction is only 0–3/128. This is fitting evidence, not competence. Every manifest retains `composition_allowed=false`.

Belief acquisition summaries exactly match the 14 validation cells per arm and saved distributions normalize. Both arms use width1024 and identical source hashes. Saved sample trajectories permit spot checks, not reconstruction of every aggregate; requested full compressed posterior exports for main-result auditing. Main source `1350eaf` aligns gate logic with the root registry: complete validation matrix, 512 episodes/cell, IID>.98 or moderate OOD>.95 final support accuracy, mean framewise posterior L1<.05 and impossible mass<.01. Empty-prior error remains an observed failure, not silently repaired after inspection.

Return `dc2f9b1` chunks evaluation without changing public observations. Required second-argument scoring excludes the unary null class; the 512 threshold refers to episodes per cell, and required-argument denominators are reported separately. Semantic `662a2ee` batches public text with masks; private graph labels still only select supervised loss queries. These instrumentation/throughput changes do not add a privileged actor input.
