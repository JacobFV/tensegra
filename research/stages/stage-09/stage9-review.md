# Stage 9 independent review

Status: independent source, archived/raw metric, checkpoint and calibration replay audits complete; final repository regression is tracked by the root verification. Baseline is `0f44e13`. This review distinguishes archived numerical reconstruction from explicitly labeled CPU frozen-checkpoint inference. No training was performed by the audit. Existing Stage 8 gate decisions remain unchanged; no composition or supervision withdrawal is permitted.

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

## Diagnostic implementation review

The Stage9 `PriorContract` override consumes public candidate-validity masks and a set maintained from public observation IDs/actions/frame-validity only. It replaces logits only while that set is empty, including after full retraction, and leaves nonempty logits unchanged. This supplies an architectural prior equally across protected/recurrent comparators; it does not train null inference. The deterministic equality-constraint generator makes equal-content observations yield the same support even under distinct IDs. It does not model independent noisy likelihood measurements.

Frozen return capture matches the historical clean initial read and recurrent updates. Probe normalization is fit on training features only; regularization selection uses validation accuracy, with test excluded. The correct scalar facet is selected by a privileged diagnostic boundary. Categorical ridge and numerical ridge differ in output prior and parameter count; they are not interchangeable capacity-matched repairs. Disjoint seed/event hashes, full validation-grid counts, fitted weights/hashes and original checkpoint hashes were requested before final probe inference and added in `53f05b5`. No source-to-target label enters the frozen actor forward.

The semantic interaction scorer contains an explicit bilinear source-target term; its rank32 output scorer is distinct from the1024-wide workspace. Edge-conditional loss masks use gold edges only at training-loss selection, retaining unordered real edges and edge-existence prediction separately. Multislot pair conversion fails explicitly rather than overwriting. The first public-observability audit inspected a10,064-graph prefix while the archived database contained10,128 graphs; this scope discrepancy was flagged before final interpretation and the worker is rerunning the full database. Zero observed collisions in a finite pool cannot establish universal identifiability.

The full historical semantic rerun is now complete: **18,944 raw prediction rows** independently reconstruct all saved semantic metrics with zero discrepancies. This includes every final and earlier checkpoint, both corpus conditions, raw/calibrated decoding and baseline shards. It validates arithmetic/provenance consistency, not graph competence. The first failure-localization package is complete before new diagnostic fitting.

A first new frozen-belief partial audit reconstructs290 cells (two complete protected seeds plus the two-cell profile) with zero metric/calibration/gate discrepancies. This is an incomplete matrix and cannot pass a study gate. Additional-ID failures must distinguish equal-content behavior from unfamiliar numeric ID bits: distinct-equal uses IDs100+, while training IDs are0–4 (clean episodes use0–3). New-ID stress is legitimate OOD, but its failure alone does not isolate evidence-multiplicity semantics.

The isolated bounded scalar-grid diagnostic independently reproduces all72 train/test prediction rows with zero discrepancies. Numerical readout exactly recovers the premixing affine scalar code for all18 checkpoints, and the factorized scalar memory facet for all6 factorized checkpoints. This is full-grid recovery across fresh nuisance contexts, not heldout numerical classes or a complete workspace interface. Poor linear decoding of mixed/concatenated memory does not prove information loss; nuisance mixing and nonlinear recoverability remain possible. Fixed-nuisance pair distances test the actual affine/normalized encoding rather than asserting that normalization necessarily erases magnitude.

## Completed frozen belief intervention audit

All864 prescribed main cells independently reconstruct from saved posteriors/targets, including every frame's ambiguity/support/no-match counts, signed null error, Brier score, expected-correctness calibration and unchanged gate thresholds. The six-model × two-policy × two-split × three-size × twelve-condition matrix is complete with512 episodes per cell. The two64-example profiling cells remain excluded from main completeness.

All432 paired policy cells have identical events/targets and historical checkpoint SHA256. Across1,787,904 paired frames, every nonempty-ledger posterior is bitwise unchanged and every one of239,616 empty-ledger frames receives the exact public uniform non-null prior. This verifies the intervention's contract; it does not make that prior learned.

On fresh validation N8/N16, protected learned-prior passes35/48 original cells; supplied-prior passes48/48. Expanded validation is43/72 versus60/72: unfamiliar-ID conditions remain failures. Recurrent learned/supplied policies pass26/48 versus38/48 original,33/72 versus46/72 expanded. Thus the protected original-matrix defect is repaired by the supplied contract, while the recurrent comparator retains post-evidence errors and neither expanded interface passes. Initial frames are included throughout. All numbers are new frozen-checkpoint inference, distinct from historical archived reconstruction.

For clarity, the expanded N8/N16 protected supplied-prior result is **60 passed / 12 failed** cells; recurrent supplied-prior is **46 passed / 26 failed**. N32 is separate OOD and is not pooled into these counts. The residual recurrent long-duplicate failure is not primarily false null: after the initial frame, N16 test mean impossible mass is .06084/.05332/.06669 across seeds, while corresponding mean null probability is only .00000991/.00001950/.00000374 and false-null argmax count is zero. The remaining probability error is therefore chiefly on impossible ordinary candidates. These quantities independently reconstruct from the already audited raw frame probabilities; they argue against treating a new null head as the general repair.

## Frozen scalar boundary probes

Independent reconstruction passes all1,026 saved prediction rows and252 probe selections. Every exact/signed/absolute/half-unit count agrees; validation-grid argmax, selected validation count, identical selection across train/validation/test, and categorical/numerical coefficient counts agree. No test-selected regularization was found.

Factorized persistent test values remain exactly recoverable by the numerical ridge from premixing and scalar memory in all three seeds. At workspace16, separate categorical probes achieve93.36/97.66/95.70% versus77.54/82.42/78.71% for the frozen historical head. At workspace32 they reach93.16/96.09/94.53% versus66.02/75.59/66.99%. Thus much information remains accessible to a different readout despite degradation of the historical decoder. This does not prove no recurrence interference: no16-step categorical probe passes the98% exact-value threshold, and failed probes alone cannot establish information absence. Each boundary-specific probe trains on that boundary's representations, including16/32; its success is not learned readout extrapolation beyond its own trained delay. Numerical and categorical ridge favor different representation geometries and have different output parameter counts.

All252 saved ridge coefficient files were independently rehashed on the linked machine and match the selected-probe SHA256 records. Their durable location is `gb10-direct:~/topoformer-stage9/returns/frozen-probes/`. Coefficient identity validates provenance; it does not by itself reproduce the numerical solve.

The separately registered ID attribution diagnostic independently passes numerical audit on all48 policy cells and48 public-primitive strata. An independent standard-library regeneration of the public primitive values matches every stratum count/correctness/L1. All6 seen-ID4/unseen-ID100 pairs have identical seeds, posterior targets and empty-ledger trajectories; source construction changes the inserted ID only. Protected seeds0/1 fail all252/265 primitive-zero cases after redundant seen-ID4 insertion, while their260/247 primitive-one cases remain correct; seed2 succeeds on both251/261 strata. Training IDs include0–4, but ID4 primitive zero never occurs in the original contradiction/retraction curriculum. This is an ID–payload combination shift, not merely unseen numeric magnitude. No general posterior calibration or ID-invariant grounding claim follows, and no new ID repair was tuned on these outcomes.

## Fixed-node semantic acquisition audit

All512 graph/checkpoint rows across16 development runs (2seeds ×4head/objective cells × bias-free/affine edge controls) independently reproduce edge symmetric-difference counts, slot errors on true edges, exact graph outcomes and curve means. Same-head objective arms have identical initial state hashes. Exported union-of-predicted-and-gold pair slots agree with the true-edge scoring view, so deployment-edge choices are inspectable without gold masking. All16 durable checkpoint files independently rehash to their manifests.

Every final run remains0/8 exact complete graphs, despite perfect slot classification on true edges. Affine edge controls retain445/354 edge errors across the8graphs for seeds101/102, respectively; objectives/head choices share this edge failure. These are privileged fixed-node fitting failures within the declared budget, not surface generalization tests. The additive slot incompatibility concerns the all-pair component objective; perfect true-edge slots here further show why it does not establish final decoding impossibility. Public-text acquisition and transfer remain blocked.

## Paired scalar readout repair audit

All396 prescribed evaluation cells across fresh backbone seeds10/11/12 and three heads are present. Independent reconstruction verifies every field/joint/required-argument count, supplied corrupted-fact target/count, per-event pairing and unchanged non-value predictions. Validation-only ridge and CE checkpoint selections match their logged grids/curves. All9 runs fail the unchanged retention gate independently. All9 backbone/readout checkpoint files rehash correctly on the linked machine.

The pooled repair trains readout features only at0/1/2/4updates;16/32 are genuinely beyond its trained delays. It is separate from the earlier per-delay probe. At16updates/eight distractors, test scalar correct counts are373/389/381 of512 unchanged,440/457/473 ridge, and411/437/456 CE refit. Additional supervised readout fitting helps, with larger gains for ridge here, but neither clears98%. Other semantic fields are bitwise unchanged across heads and can still block full retention. Ridge uses8,192 supplied feature rows; CE additionally samples128,000 optimizer presentations from those rows. Different objectives/exposures mean this is not a pure loss-function comparison.

## Relation-specific decoder calibration localization

An independent **CPU frozen-checkpoint inference** reproduces every relation's positive/negative support and separation margin and all graph decisions in the training-only threshold diagnostic. Every supported relation in the affine fixed-node heads is strictly separable across the8training graphs. One training-fitted threshold per relation yields8/8 complete graphs for both development seeds, versus0/8 at raw zero. A pooled global-score overlap therefore did not establish per-relation nonseparability. The gold-prevalence analytical correction instead retains156/168 edge errors and0/8graphs; its population-risk identity does not guarantee finite-model calibration and uses privileged prevalence.

These inspected-training results localize a decoder contract failure and **do not retroactively pass the original raw gate**. Before new outcomes, review accepts a separately registered calibrated fixed-acquisition confirmation with3fresh seeds ×4head/objective arms, new8-graph fixed data, same300update budget and training-only minimum-error per-relation thresholds. Every arm/seed must reconstruct every fixed graph for that narrowly labeled gate. No fixed-set outcome establishes fresh structural/rendering transfer or overrides composition restrictions.

The isolated grid fixture has a separately preserved v2 contract correction: converting unary returns to binary add now chooses a distinct public argument key, preventing two argument positions from naming one identity while carrying different witness values. The original diagnostic remains archived. Independent v2 reconstruction again passes all72 rows; all18 premixing and6facet-memory exact-grid results remain528/528. Other mixed/concat linear counts change slightly and are reported from v2. This fixture-only change does not alter frozen boundary probes, paired readout main studies or Stage8 bytes.

## Prospective calibrated-contract confirmation

The separately frozen confirmation contains all12 prescribed seed/head/objective runs on the new8-graph fixed set. Independent raw reconstruction verifies480 graph/checkpoint rows, including raw and final calibrated predictions. Independent CPU replay reloads and rehashes all12 checkpoints and refits thresholds with a separately written grouped-score sweep. All156 per-relation support/error counts and96 calibrated graph outcomes match; CPU/GPU threshold differences are at most4.77e-7 without changing decisions.

Seeds11/12 fit8/8 calibrated graphs in every arm, but seed10 fits only6/8 with5 remaining edge errors in each arm. Raw decoding stays0/8 throughout. The prospective all-seed calibrated fixed-acquisition gate therefore **fails**; no aggregate mean or successful development seed rescues it. Public-text acquisition and fresh/heldout semantic transfer remain blocked. This narrower result supports a substantial decoder-calibration component while retaining an acquisition/score-separation failure on a fresh initialization; it does not establish that slot correction alone solves complete semantic acquisition.

## Final historical preservation

After all diagnostic implementations and result exports, all **1,747 files tracked at baseline `0f44e13`** remain byte-identical, including Stage1–8 source, artifacts, reports and gate registry. The complete final Git-blob manifest is saved independently from the initial preservation check. New work is additive under the Stage9 research/source/config/test/tool paths. No old result was replaced by a corrected fixture or later favorable decoder.

## Consolidated report and final verification

The final report percentages, per-seed gate outcomes, calibration qualifications and compute accounting agree with raw/audited evidence. Minor wording clarifies that reported return percentages use the eight-distractor test cell and deterministic repeated-content constraints are not noisy independent likelihood measurements. The registry now names exact frozen source commits matching archived SHA256 for the probe and both semantic development recipes; categorical versus numerical probes are not described as parameter-matched to each other. Root final verification reports **553 tests plus6subtests passed** in8.38seconds on an immutable `2fa21bd` snapshot, a clean diff check, all1,747 baseline blobs unchanged and no newly oversized artifact. No scientific blocker to publication remains; failed competence gates and composition blocks are retained.
