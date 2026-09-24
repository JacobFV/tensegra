# S21: broader motif exposure, unchanged sequential learner

Prospective exploratory branch registered while the complete S20 confirmation remains running and unread. Motivation is the independently audited S19 heldout3×4 failure and familiar-layout prefix errors, not a S20 outcome. S20's recipe and decisions remain unchanged. CPU data preparation is authorized; no S21 GPU allocation yet.

## Hypothesis and comparison

Can exposure to more arity/fact combinations teach the existing sequential learner to recover an omitted combination? Compare the unchanged scratch S19 learner under the existing three-motif4096 corpus and a six-motif4096 corpus. One paired development initialization2101, identical width1024,4096updates/batch8, AdamW/loss/schedule/160-record capacity. This is a data-distribution intervention, not a new architecture or a test of programmed attention. Different token/record work must be measured despite matched presentations.

The broad corpus contains shared existing TRAIN examples:1024 of3×3,512 of4×3,512 of4×4; plus new examples:512 of2×4,768 of5×3,768 of5×4. Select each old subset by Python random.Random(210021 + lexicographic cell index).shuffle of its original S15 TRAIN row indices, without inspecting model outcomes. Both arms have4096 distinct constructions and32768 presentations; the baseline's additional old-motif examples are replaced by new motifs in the broad arm. Preserve exact construction, token, node, edge and record exposures, and the overlap inventory. Shuffle the resulting broad corpus with Python random.Random(210022) before the unchanged15115 training schedule.

Do not include2×3: the bounded32768-draw audit demonstrated only496 distinct examples in that cell. The unchanged generator has no two-fact mode. Arity5 is already supported, with at most42nodes/143records in these configurations. No generator semantics, vocabulary, public parser, output schema, or model capacity is extended. The omitted3×4 combination remains absent from both TRAIN corpora.

## Data and public contract

Use the existing audited English generator and canonical compiler. Reuse all four existing S15 DEV512 cells as explicitly inspected development populations. Add fresh512-example DEV cells for2×4,5×3,5×4. Generate and seal a separate fresh512-per-cell confirmation cache over all seven evaluated motifs; it is unavailable to model selection. Exclude all inherited and current training, development and earlier confirmation constructions under the same alpha-equivalence definition. Prior confirmation metadata may be used only for split exclusion, never model outcomes or recipe selection.

Generation uses deterministic counter ranges starting210000000 for TRAIN,211000000 for DEV,212000000 for reserved confirmation, with disjoint offsets100000 per lexicographically ordered cell. Cap attempts at20000 per cell per split; do not increase the bound to satisfy a desired result. A shortfall blocks that data condition and is reported. Validate exact answers independently, public-input target identifiability, copy visibility, no vocabulary overflow, maximum nodes/records, alpha uniqueness and complete split exclusion before freezing any model run. Corpus construction requires its own version/hash even though generator semantics remain unchanged.

## Evaluation and decisions

Evaluate fixed0/1024/2048/4096 checkpoints on the same existing actual-TRAIN128 panel for the baseline, and a deterministic broad-TRAIN128 panel selected proportionally to its declared cell counts, plus all seven DEV512 cells. The broad panel allocation is32/16/16/16/24/24 in3×3/4×3/4×4/2×4/5×3/5×4, using Python random.Random(210023 + lexicographic cell index).shuffle within cells. These are in-sample diagnostics, not generalization. No calibration or checkpoint selection; all categorical greedy outputs, malformed prefixes, teacher-forced field metrics and paired events are retained.

Primary development advancement requires at fixed4096: broad heldout3×4 complete>=52/512 and at least26 more complete graphs than the paired original corpus arm; broad accuracy on the three original known motifs no more than5 percentage points below that arm's macro accuracy; and broad>=256/512 in each newly trained motif. Report each component and all continuous effects even if the joint rule fails. The same512 events support paired differences, not independent repeated samples. This does not establish broad unseen-structure generalization.

No automatic exposure extension. If these criteria pass, register a fresh three-seed confirmation using the sealed new cache and the unchanged paired recipes. If they fail, preserve the outcome and distinguish familiar-layout persistence, general acquisition failure, or loss of old-motif competence using archived outputs before choosing anything else. This branch cannot alter S20's confirmation or historical gates.

## Budget and provenance

Planning envelope<=1800GPU seconds for profile and both development arms; actual caps require profiling full seven-cell evaluation, longest public inputs,160-step decoding, exports and checkpoint costs. Confirmation, if earned, needs a separate allocation. One serial GPU owner remains root; all CPU preparation can proceed while S20 runs. Freeze source, dataset, configs and analysis rules before launch. Preserve all seeds/failed attempts, independently audit metrics, and label supplied serialization versus learned semantic mapping explicitly.
