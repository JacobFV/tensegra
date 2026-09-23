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

## Frozen-run interpretation restriction

The belief worker's self-audit identified a remaining observable-consumption asymmetry: observation IDs are present in both public dictionaries, but only the protected arm actually uses them (as ledger keys); the recurrent encoder consumes candidate/evidence/role/action features without IDs. Therefore duplicate/retraction comparisons are not strictly information-matched. Preserve the frozen run and report this limitation; a separately prespecified ID-feature-matched follow-up is required for a stronger claim on those controls. Clean and independent-order conditions do not require IDs and retain their narrower comparison. Exact deduplication/retraction remains an architectural capability, not learned behavior.

Semantic `49ede10` train-only threshold calibration was reviewed: first eight training graphs, both training renderers, fixed per-relation grid, dense decoded edges gated by predicted rather than gold presence, one frozen vector across heldout surfaces. Raw-zero and calibrated results are separate. This uses 16 extra privileged label presentations even at optimizer step zero; those are explicitly disclosed. No evaluation labels select thresholds.

## Original belief diagnostic: full raw audit

Independently reconstructed all 192 heldout cells (98,304 episode evaluations) from saved full posteriors: support accuracy, impossible mass, posterior L1, entropy, and every reliability bin match aggregate metrics within 3e-6. Source hashes match canonical `9745438` and every run manifest. Gate failure reasons independently match all 14 protected and 25 recurrent failed validation cells. Both gates fail; the protected arm's perfect final selections do not erase its prior/null posterior error. The original duplicate/retraction comparison remains metadata-asymmetric. The separately frozen `bac2950` follow-up supplies identical 16-bit observation-ID features to both encoders and retains equal capacities.

The independent raw reconstruction is reproducible with `scripts/audit_stage8_beliefs.py`; its output is recorded alongside diagnostic artifacts. Semantic acquisition's original 2304 raw prediction rows were also streamed independently: all typed-edge true-positive, prediction and gold counts match saved metrics exactly. This validates those counts, not semantic competence.

## Lossless semantic artifact repacking

The oversized semantic acquisition COO export was replaced by bit-packed boolean edge tensors without changing metrics. The converter asserts equality of every decoded tensor in all 2304 records. An independent standard-library decoder additionally compared the first 32 complete packed records against the durable original: every edge coordinate, other graph field and metadata field matched exactly. Local packed SHA256 `662bea01015df615c9748927cf1d48ce6402e7ee673b3fe8c03ce9952387a67d` matches the conversion manifest; size is 22,366,167 bytes versus 500,243,173. The original remains archived remotely with its SHA256. Final integration must omit the oversized unpublished blob from pushed history while retaining source provenance manifests.

## Return main: complete independent audit

All 1008 raw evaluation rows (516,096 example evaluations) independently reproduce every field, scalar/identity/full joint, and required-second-argument count. All 18 prespecified gates fail independently. The config hash and five source hashes match frozen `dc2f9b1`. A further 864 supplied-fact readback rows exactly reproduce field counts after independently applying the documented value/type/provenance corruptions. No missing or tiny cells pass.

For persistent memory with eight distractors, mean joint accuracy at 1/16/32 updates is 64.84/62.17/61.26% for mixed encoding, 70.25/67.51/67.90% for disjoint concatenation, and 89.19/75.59/61.98% for factorized tokens. Exact value accuracy and numerical error are distinct: factorized values reach 89.45% exact and MAE0.055 at one update, versus 79.04%/MAE0.243 at16 and 68.82%/MAE0.586 at32. Thus the short-delay improvement is real, while recurrent deterioration and the exact-retention gate failure remain. These data do not demonstrate downstream learned use or authorize composition.

## Semantic main freeze audit

Frozen `f81f33f`/source `26c8b5f` separates 1k/10k available constructions from 10,016/100k presentation checkpoints, three paired seeds and matched no-text-feature controls (1.2 million total optimizer presentations). Equal lesson sampling is fixed before outcomes, its lesson set is asserted across corpus sizes, and frequency controls use inverse lesson size. Lesson labels affect sampling only, not actor inputs. Actual distinct constructions and per-lesson exposures remain separately logged.

Fresh evaluation partitioning skips only previously inspected semantic keys when reserving heldout examples; these keys remain eligible for training. Fourteen prior-key collisions are excluded from the heldout prefix. No fresh variable-binding example remains in this reserved evaluation set, explicitly recorded as missing heldout coverage; that family's generalization cannot be claimed from these results. This is a known coverage limitation, not hidden outcome-based selection.

## Primary observation-ID-matched belief audit

All 192 cells and 98,304 episode evaluations independently reconstruct from full exported posteriors, including calibration, entropy, impossible mass and L1. Every run enables neural observation-ID features, and source hashes match frozen `bac2950`. Independent validation checks find 14/48 protected cells and 24/48 recurrent cells fail the unchanged gate. Both remain blocked.

The primary results materially narrow the claim: at 16 candidates, both arms achieve 100% final clean test selection. Under long duplicate evidence, protected final selection is 100% with mean impossible mass0.00126; recurrent selection is99.09% with mass0.05923. Empty-evidence null mass remains approximately0.0246/0.0251, preventing full posterior competence. Wide fresh-data recurrence therefore acquires ordinary updates; the measured protection advantage concerns posterior stability under repeated evidence, with exact deduplication supplied by architecture. It is not evidence that generic recurrence inherently cannot accumulate evidence.

## Historical-width and semantic-head audits

The width32 historical control independently passes raw reconstruction for all192 cells/98,304 episode evaluations. Source hashes match `bac2950`; unchanged gate checks reproduce25 protected and46 recurrent failed validation cells. This is a CPU control versus CUDA width1024 and does not establish matched hardware throughput.

The additive slot-head expressivity finding is mathematically valid: for two competing classes, the logit difference has form `u_i+v_j`; positive diagonal and negative cross-pair differences in a2×2 matrix are impossible because the diagonal and cross sums are identical. Gold witnesses demonstrate conflicting no-slot/ordered-slot targets in the supervised population. The separate edge mask can suppress the cross-pairs, so this is **not** a proof that exact masked graph decoding is impossible. No frozen main recipe changed.

A new independent standard-library bitmap auditor reconstructs node counts, typed/ordered edge counts, type accuracy, identity copying/equivalence and exact canonical semantic equivalence. All2304 acquisition rows pass every reconstructed metric, extending the earlier typed-edge-only audit.

First immutable semantic main shard audit: all3072 N1k seed0 rows across semantic/no-text-feature arms, three exposure checkpoints and raw/calibrated decoders reproduce every semantic metric without discrepancy. Remaining paired seeds and N10k remain pending; this partial audit is not a completed scaling result.
