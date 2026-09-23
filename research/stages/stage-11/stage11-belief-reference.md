# Stage 11 preferred belief reference

The preferred reference is Stage 10's **`ledger_only` learned content matcher**, unchanged: observation handles address the protected ledger but are excluded from neural compatibility features. This freezes an existing restricted interface; it adds no training, inference, adapter, ensemble, or runtime capability.

The [machine-readable reference](stage11-belief-reference.json) pins the baseline `e10b80a`, Stage 10 source/config freeze `bc18496`, all five frozen source hashes, and the complete three-seed reference panel (20/21/22). It records exact config, metric, checkpoint-byte and checkpoint-tensor hashes. No best seed is selected. The original randomized-ID and correlated-ID comparison panels remain referenced, with their reports, raw counterfactual tails and independent audits.

## Address semantics

An observation handle is an **opaque ledger address**, used to recognize repeated delivery of the same observation and to retract that observation. Its spelling is not evidence about a candidate. A consistent handle bijection must preserve all duplicate and retraction references while leaving semantic content unchanged.

This rule does **not** mask semantic entity identity, operand identity, timestamps, semantic roles or event content. Those can carry information required by the task. If a future task makes chronology or a named entity semantically relevant, it must represent that information explicitly rather than smuggling it through an opaque delivery handle. Extending this contract would require a separately versioned interface, not silently reinterpreting the existing mask.

The inherited public domain is unsigned 16-bit handles, with a negative no-observation sentinel. `ContentEncoder` zeros only the sixteen appended handle features; the integer handles still reach exact ledger insertion/removal. Distinct equal-content observations in this benchmark are deterministic constraints, not independent noisy likelihood measurements.

## What the reference guarantees and what it learns

| Category | Frozen reference behavior | Limit |
| --- | --- | --- |
| Programmed invariant | Consistent handle renaming cannot change ledger-only probabilities; duplicate-ID first-write semantics and retraction are exact | Does not make a prediction correct or infer the semantics of a new domain |
| Supplied prior | Empty active ledger gives the specified uniform non-null prior, including after full retraction | Not learned prior calibration or open-world null detection |
| Learned interface | Content compatibility and nonempty candidate/null behavior | Finite supervised closed-candidate task; no generic recurrent workspace repair |
| Historical mean posterior contract | Every prescribed validation cell passes in all three seeds; untouched tests independently satisfy the same thresholds | Mean errors do not establish execution-time conditional precision or worst-case reliability |

The architecture remains a protected aggregator with four candidate-local neural residual phases, workspace width **1024**, FF width 2048, and **16,867,395 allocated parameters**. Its **16,384 handle-input weights are inactive**. No structural attention or attention-memory tokens are introduced. This reference cannot independently validate programmable attention.

Each seed trained for 1,000 updates of 32 examples. The historical validation result is **90/90 required N8/N16 cells and 45/45 additional N32 cells**, with 512 examples per cell; untouched test has matching counts, reported separately rather than reused for selection. Initial and intermediate frames remain included. These are existing Stage 10 observations, not a new Stage 11 evaluation.

## Preserve the comparison and its tails

The randomized-ID matcher also passes those mean posterior gates. It remains a useful learned-robustness comparator; it is not discarded because the preferred interface supplies invariance directly. Its maximum probability-coordinate changes under paired handle counterfactuals are approximately **.664901, .418894 and .663871** in seeds 20/21/22 across the archived matrices. Ledger-only changes are exactly zero. Mean success and rare confidence distortion must remain visible together.

Opaque-handle masking supplies the preferred reference's renaming guarantee. Learned content matching supplies its task performance. Those are distinct achievements. The frozen randomized/correlated comparisons and full failure events are linked through the JSON manifest, with the [Stage 10 report](../stage-10/stage10-belief-report.md) and [independent review](../stage-10/stage10-review.md) retaining detailed counts and qualifications.

## No execution authorization

This reference does not authorize readiness-driven execution, autonomous composition or supervision withdrawal. Before any later execution study, separately predeclare:

- correctness of the particular candidate structure and ordered binding being scored, not a global confidence label;
- conditional precision, execution coverage/refusal, and finite support counts at frozen calibration-selected thresholds;
- posterior/error tails, intermediate retraction and duplicate behavior, and handle counterfactuals across every required seed and fresh heldout condition;
- exact schema validity separately from learned semantic correctness, including confident wrong and valid-but-wrong candidates.

No product of concentration, persistence and schema validity automatically becomes a correctness probability. This document sets no new threshold after viewing historical tails and starts no such experiment.

## Verification scope

Stage 11 verification reads files only: local source/config/artifact hashes match their archived references, and checkpoint hashes match Stage 10's independent byte-audit record. Checkpoints remain at `gb10-direct:~/topoformer-stage10/beliefs/main/{arm}-{seed}/model.pt`; this stage does not reload them or freshly attest remote storage. The machine-readable manifest says exactly which hash is file-byte versus canonical JSON or parameter-tensor provenance.

All historical Stage 1–10 files remain unchanged. No model source modification was needed.
