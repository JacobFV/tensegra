# Stage 9 independent review

Status: historical failure-localization audit in progress. Baseline is `0f44e13`. No new model training or inference has been performed by this audit. Existing Stage 8 gate decisions remain unchanged; no composition or supervision withdrawal is permitted.

## Source-to-report map

| Interface | Historical implementation | Frozen evidence | Report / gate boundary |
|---|---|---|---|
| Beliefs | `src/topoformer/belief_state.py`, `belief_study.py`; frozen `bac2950` | `research/results/stage8/belief-idmatched/{protected,recurrent}-{0,1,2}`: manifests, full posterior/target exports, metrics and checkpoints hashes | Stage 8 belief report; full initial-frame-inclusive posterior gate |
| Returns | `src/topoformer/return_memory.py`, `return_memory_study.py`; frozen `dc2f9b1` | `research/results/stage8/return-main`: 18 raw JSONL streams, config, manifest and checkpoint hashes | Stage 8 return report; workspace-only exact reconstruction |
| Semantics | `src/topoformer/semantic_curriculum.py` and Stage 6 compiler/base decoder; frozen `26c8b5f` | `research/results/stage8/semantic-main-n{1000,10000}`: compact raw tensor shards, manifests, corpus records, threshold records | Stage 8 semantic report; component learning distinct from complete acquisition/transfer |
| Independent metrics | `scripts/audit_stage8_{beliefs,returns,semantics}.py` | Standard-library reconstruction of saved probabilities/classes/bitmaps | Numerical reconstruction, **not** rerun inference |

Historical source snapshots and source-hash manifests take precedence over later working source. Stage 9 source must be versioned separately. The baseline preservation manifest records every baseline Git blob, including symlink bytes, and checks the working file against it.

## Recorded failed components and missing coverage

| Interface | Failed component | Evidence / limitations |
|---|---|---|
| Belief protected | Empty-frame null mass despite uniform non-null public prior | 14/48 validation cells fail; equal candidate logits imply posterior L1 = twice null mass. This explains the output error, not optimizer behavior. Nonempty update diagnosis must remain separate. |
| Belief recurrent | Initial calibration plus repeated-evidence posterior errors | 24/48 validation cells fail. Ordinary unique selections are near ceiling; no universal recurrence-incapability claim. |
| Returns | Exact scalar reconstruction, plus arm-dependent type/operation/identity errors | All 18 runs fail. Persistent storage is intact; learned repeated retrieval/readout is tested. Full per-field denominators must accompany means. |
| Semantics | Exact complete graph recovery and robust renderer/lexicon transfer | All exact graph scores zero; calibrated known-renderer edges improve. Additive slot objective mismatch does not prove final masked decoding impossible. |
| Semantic coverage | No fresh variable-binding evaluation in final heldout set | Four available canonical constructions; do not silently count absent family as passing transfer. Set-operation support only 120 at tested difficulty. |

Known confounds: width is not the only Stage 7→8 change; candidate-local residual MLP is not the historical shared attention workspace. Protected deduplication/retraction is supplied. Facet versus concat has equal parameters but six versus one attention-addressable tokens. Mixed has unequal encoder parameters. Half-unit scalar exactness is not MAE. Lifecycle interventions were not trained. Random distractors are not concurrent learned cognition. Gold training query selection is privileged supervision; it must not enter actor inputs. No-text controls retain copy inventory/length. Test thresholds cannot be reopened.

## Paired historical return analysis contract

The frozen runner regenerates `data(data_seed, eval_size, distractors)` at every clean delay. Thus identical split/data seed/distractor condition and row index identify the same underlying event. The audit additionally requires full target-array equality across delays. It computes correct→correct, correct→wrong, wrong→correct and wrong→wrong separately for scalar, all five non-value fields, and all six fields; no marginal independence assumption is used. Historical `identity_joint` includes the unary null label; the official required-argument gate separately excludes that label only from its denominator.

All Stage 8 heldout predictions are historical diagnostic data. They may localize failure but cannot select a new Stage 9 main recipe. New probe training/test partitions and frozen development selection are required before new test inference.

## Prelaunch review requirements

- Root registry must freeze hypotheses, observables, privileged labels, development/test split, selection rules, budget and stop rules before main runs. The three-GPU-hour total ceiling includes all new experiments, with mechanical exceptions recorded.
- Empty-ledger prior may use publicly maintainable observation state, including after full retraction. No private gold support or hidden winner may drive it. Original learned model retained.
- Frozen-checkpoint scalar probes must train on disjoint examples and report probe capacity/limitations. Exact-copy/oracle-read ceilings cannot pass workspace reconstruction.
- Slot factorial must retain edge-existence prediction and deploy using predicted edges. Gold edge selection is permitted only in training loss. Audit multiple labels per pair before choosing a single target.
- Missing cells, fewer than 512 required examples, failed seeds or missing families cannot pass. Fixture/fixed-set competence must be labeled separately from fresh generalization.

## First independent reconstruction results

All **1,747 baseline tracked files** match the Git blob bytes at `0f44e13`, including the compatibility symlink. Belief raw reconstruction passes all192 cells/98,304 episode evaluations. Return raw reconstruction passes all1,008 rows/516,096 episode evaluations and independently retains18 failed gates. These are numerical reconstructions from archived predictions; no frozen model inference is implied.

A separately authored paired-outcome implementation agrees with the return worker's table. For factorized persistent memory at eight distractors, test delays1→16, pooled initialization-seed replicates yield full-joint1134 correct→correct,236 correct→wrong,27 wrong→correct,139 wrong→wrong. Scalar counts are1186/188/28/134; non-value joint1463/68/0/5. These are1,536 seed-replicates of512 underlying events, not1,536 independent events. Both deterioration and recovery occur, so subtracting marginal accuracies is not a forgetting probability. Full per-run/split/distractor/pair/field counts and target digests are retained.

Reproduce with `python3 research/tools/audit_stage9_baseline.py --semantic`. The script imports only standard-library independent historical auditors, never the model metrics implementation. Semantic full-bitmap reconstruction is pending completion; existing historical audit is evidence but not represented as a new rerun.
