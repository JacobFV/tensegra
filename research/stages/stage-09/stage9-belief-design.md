# Stage 9 belief contract diagnosis (pre-outcome)

Width 1024 / inner 2048 remains fixed. No readiness, runtime composition, or supervision withdrawal.

## Probability contract

Candidate proposals are a closed enumerated set. Observations are deterministic noiseless role-equality constraints; posterior targets are uniform on the intersection, or null if the intersection is empty. This is not a noisy Bayesian likelihood model. Distinct IDs with identical content add distinct ledger entries but imply the same constraint/posterior; independent noisy measurements are outside this generator. Same-ID insertion uses first-write semantics and is idempotent even if payload changes; retraction removes by ID and an absent-ID retraction is a no-op. Empty active evidence, including full retraction, requires the uniform non-null prior. Type invalidity is not represented by null.

## Source-to-report map

Stage8 `belief_state.py`: generator/public collator, four-phase candidate-local model, ledger and null head, posterior cross-entropy plus compatibility BCE. `belief_study.py`: evaluation/export/gates. Frozen primary source `bac2950`, archived manifests and raw probabilities: `research/results/stage8/belief-idmatched/{protected,recurrent}-{0,1,2}`. Durable frozen checkpoints: `gb10-direct:~/topoformer-stage8-artifacts/beliefs/idmatched/`. `stage8-belief-report.md` and gate registry report 14 protected / 24 recurrent failed validation cells. All historical bytes stay unchanged.

## Experiments and selection

A0: standard-library reconstruction of every archived primary posterior. Hypothesis: initial protected error equals twice null mass because candidate logits are equal. Historical test is diagnosis only. Report support/ambiguity/null counts, signed null error and worst L1. This is numerical reconstruction, not inference.

A0-checkpoint: load frozen checkpoint hashes, hook null inputs/logits and inspect posterior-only gradients on initial frame versus later frames. The initial null head receives (0,0). Verify gradient pressure, distinguish nonzero finite-budget residual from an expressivity impossibility. No optimizer update.

A1/A2: paired frozen-checkpoint intervention, not retraining. A1 preserves learned priors/null. A2 supplies exact uniform prior iff a publicly maintained active-ID set is empty; otherwise logits are bitwise unchanged. Apply same public bookkeeping to protected and recurrent comparators and disclose it. Both arms reuse each of three independently trained Stage8 initialization seeds; no selection of checkpoint or hyperparameters. Fresh validation seeds 910000+seed and untouched test seeds 920000+seed. 512 per cell, N8/N16 all eight original conditions plus full retraction, distinct-ID equal content, candidate permutation, observation-ID renaming. N32 is separately labeled size OOD. Targets privileged only for evaluation. No winner/support label enters intervention.

Inherited official thresholds and initial-frame inclusion unchanged. Report original-matrix gate and expanded-contract gate separately; all seeds and cells required, final support selection >.98 IID / >.95 moderate OOD, framewise mean L1 <.05 and impossible mass <.01. Additional N32 descriptive OOD does not erase original failures. Supplied-contract pass is not learned calibration. No A3 until these diagnostics indicate a remaining null problem; stop rather than speculative repairs.

## Budget and provenance

No new model training planned. Estimate <=10 GPU minutes frozen inference plus <1 minute gradient diagnostics, after root queue authorization. Profile one N16 long-duplicate cell before freeze. Chunk evaluation to 32; model parameters 16,867,395. Same checkpoint and paired event IDs across A1/A2; raw posterior/targets gzip, checkpoint/source/config/data hashes saved. Report runtime, allocated CUDA bytes and RSS separately. Archived standard-library processing is CPU-only. Main freeze precedes untouched test. No further configuration selection on Stage8 tests.

Root registry amendment before new inference: fresh evaluation namespace is 9M (validation 9100000, test 9200000). Existing seeds 0/1/2 are frozen historical diagnostics only. A new main, if needed/authorized, uses original training recipe at paired initialization seeds 10/11/12, then compares both prior policies on each same checkpoint. Track cap 15 GPU minutes. No outcome-based recipe change. Profile config uses historical protected0, N16 long repetition, 64 events (diagnostic only).

Frozen-checkpoint intervention approved after profile: profile ran in 1.10 seconds (0.374 evaluation), CUDA peak 927,998,976 bytes, RSS peak 1,515,544 KiB. Complete matrix estimated 2–4 minutes, within 15-minute track cap; zero new optimizer presentations. Six `stage9-belief-frozen-*` configs prescribe existing paired seeds0/1/2 × both comparators, all 12 conditions, N8/16/32, validation/test512. N32 is descriptive and separate from original gate. No new initialization seeds are claimed. No A3 justified before results. Existing checkpoint gradients show nonzero residual pressure: protected seed0 initial null output-bias derivative .05678 atN8 and .02922 atN16, exactly null mass; later clean mean derivatives are approximately .00009/.00011. This is a frozen local gradient, not a reconstruction of optimizer history or proof of a single optimization cause.

## Separately registered identity-shift localization

Initial completed frozen seeds revealed a new confound: `distinct_equal` inserts IDs100+ outside trained0..4. Preserve that result. Before this diagnostic's outcomes, compare exactly one redundant first-observation insertion with seenID4 versus unseenID100; payloads and target trajectories are identical. Also permute IDs within trained0..3 versus clean. Use 512 fresh development episodes9.3M+seed, N8, all six frozen models; no new training, no main-gate reclassification, expected <1 GPU minute. Root authorization required in serialized queue. This isolates ID novelty from one additional redundant deterministic constraint, but does not cover arbitrary multiple independent noisy observations.
