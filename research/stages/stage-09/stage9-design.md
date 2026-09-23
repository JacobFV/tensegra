# Stage 9: representation contracts and failure localization

Baseline: `0f44e13e3adee7f2ed15d7b76178b20c0ba6aabb`. Dedicated branch `feat/representation-contracts`. Primary workspace width is 1024. Stage 1–8 source, reports, gates and result bytes are immutable. No autonomous composition, returned-fact-use training, runtime expansion, pretrained integration or supervision withdrawal.

## Questions and sequence

First reproduce archived metrics, construct a source-to-report map and failed-component table, and distinguish archived numerical reconstruction from fresh frozen-checkpoint inference. Historical test outcomes are diagnosis only. Then inspect three independent contracts:

1. Belief: closed candidate sets imply a uniform non-null prior when the public observation ledger is empty, uncertainty over consistent candidates when ambiguous, and null when none match. Schema validity is separate. Compare unchanged checkpoint inference with a public empty-ledger-prior override, including full retraction. This is a supplied prior, not learned calibration. Inspect loss and gradients before attributing the learned null defect to optimization.
2. Return: verify same-event pairing across delays, report all four correctness transitions for value/non-value joint/full joint, and instrument encoder, mixed memory, initial read and recurrent workspace. Frozen-checkpoint probes use independent data and matched probe capacity where practical. Locate scalar errors before choosing at most two repairs. Persistent readout degradation does not establish loss from protected storage.
3. Semantics: reproduce additive slot-margin incompatibility, audit sampled labels/multilabel pair collisions and public identifiability, then test the 2×2 head/objective comparison. Mathematical fixtures and privileged fixed-node complete acquisition precede public-text acquisition. Stop dependent scaling if the target is unidentifiable or the head cannot fit it.

## Selection, data, and interpretation

Development initialization seeds 101/102 are distinct from any new main seeds 10/11/12. Fresh Stage 9 data seeds use the 9,000,000 series, with separate train/probe/calibration/validation/test allocations recorded per experiment. These are namespace reservations, not a claim of semantic disjointness: canonical-key overlap checks are required for language graphs. Reusing Stage 8 trained checkpoint seeds 0/1/2 for frozen-checkpoint paired diagnostics is explicitly historical; fresh input validation/test sets are still required.

Every main comparison needs at least three paired initialization seeds and 512 examples per required cell where meaningful support exists. Tiny or missing semantic families cannot pass a generalization gate. Select architectures/decoders using development data only; freeze source/config/thresholds before new final tests. Preserve all failed recipes. Log actual widths, parameters, memory tokens, presentations, unique examples, runtime and resource measurements. Parameter equality is not FLOP equality.

Categories remain distinct: programmed guarantee, learned interface, calibration, generalization, composed competence. No diagnostic or restricted pass changes Stage 8's gates or authorizes composition. Exact scalar reconstruction remains primary; MAE, oracle component substitution and exact-copy ceilings are diagnostic only. Semantic edge F1 cannot stand in for whole-graph competence.

## Compute allocation before launch

Hard Stage 9 ceiling: **180 GPU-minutes**, including frozen-checkpoint inference/probes and main runs; brief explicitly labeled mechanical fixtures are tracked separately. Allocation caps: belief 15 minutes; return localization/repair 75 minutes; semantics realizability/acquisition 75 minutes; shared reserve 15 minutes. These are ceilings, not permission to launch factorial sweeps. Initial estimates: belief frozen inference under 10 minutes; return boundary/probe phase up to 25 minutes before repair selection; semantic mechanical/oracle diagnostics up to 15 minutes before public-text decisions. Representative profiling must precede every main budget freeze. Root serializes GPU jobs and records actual usage; no worker starts GPU training without a specific queue release.

Verified host at 2026-09-23 08:56 UTC: `promaxgb10-4dfb` via `gb10-direct`; NVIDIA GB10, driver 580.126.09, Torch 2.14.0+cu130 in existing `~/topoformer-stage8-cuda`, CUDA functional, no active GPU process. Home filesystem had 471 GB free; host RAM 119 GiB total / 115 GiB available. Unified-memory capacity is not a per-process CUDA allocation measurement. Record peak CUDA allocated bytes and process RSS independently during profiles. No dependency upgrade or local heavy training.

## Deliverables

Versioned modules/configs, diagnostic package and contract tests; three subsystem reports; independent raw/provenance audit; frozen registry and explicit gate decisions; final report answering the six user questions. Commit progress locally; final merge/push follows verification under existing authorization. All new records use `research/stages/stage-09`, `research/results/stage9`, and `research/tools`.

## First profiled release

Belief frozen-checkpoint profiling (source `857a57c`) used 64 N16 episodes with 33 frames: 0.374 seconds for paired prior-output evaluation; 1.10 seconds including gradient diagnostics, 928 MB peak CUDA allocation and 1.52 GB RSS. Checkpoint SHA matched. Authorized the frozen 3-seed × 2-model × 3-size × 12-condition × 2-split matrix at 512 episodes/cell, estimated 2–4 minutes including export. This is an architectural inference intervention on existing checkpoints, not newly trained calibration. No training is required merely to rename the historical initialization seeds.

## Post-freeze attribution followup: observation IDs

The first completed frozen checkpoint fixes the original matrix with the supplied prior but fails new large-ID conditions. Preserve the full original run. A separately named development-only followup will compare clean IDs0–3 plus a repeated-content observation at ID4 (seen training range) against an otherwise equivalent new ID100, and within-range ID permutation against out-of-range renaming. This is attribution of ID shift versus content redundancy, not main-recipe selection or a replacement test. Use fresh development data and log every checkpoint outcome. No null-head repair is authorized from this observation alone.

## Profiled semantic oracle development release

The four three-update head profiles took 0.458/0.065/0.284/0.264 seconds (first includes cold startup). Peak CUDA allocation: 170 MB additive / 392 MB interaction; head parameters 1.90M / 6.36M with width-1024 privileged node codes and interaction rank128. Authorize the frozen four-arm × two-development-seed × 300-update fixed-node comparison, estimated2–3minutes, hard cap5minutes. It runs after frozen return probes. This is closed-set privileged decoder acquisition, not public-text or fresh-structure success. Stop at the declared budget; C3 remains gated.
