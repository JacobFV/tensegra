# Stage 4: stable binding substrate

**Report status: experiment execution in progress.** This document currently fixes the reporting structure; it makes no empirical claim from partially completed seeds. The main study is 23 variants × 3 seeds × 20 depth/size cells, with 400 training steps per run. [Methods](stage4-methods.md) specify architecture, initialization, supervision, data boundaries, and the preregistered gate.

## Main conclusion and gate

Pending all 69 runs, artifact validation, and independent review. The conclusion will identify separately what was functional at initialization, what training improved, whether randomly initialized binding was learned, and whether task accuracy coincided with correct trajectories. The D64/N128 gate requires both task accuracy and full-path completion ≥0.95 in each seed. No partial-seed result or mean over seeds can satisfy it.

## Matched architectural comparisons

The completed table will report task / complete-path accuracy at D4/N16, D64/N16, D4/N128, and D64/N128, with across-seed uncertainty available in the artifact report. It will cover raw-dot mixed states, cosine mixed states, attention-write protection with full-state versus identity-only grounding, explicit pointer writes, and strong known-matcher/keyed graph-input controls.

The interpretation will distinguish matcher normalization, protection of identity writes, and isolation of grounding reads. Attention-write models still let content influence ordinary attention. Pointer-write models explicitly compute `P_query A` before writing the expected entity key; their result cannot establish the superiority of structural attention logits. All structural-strength names denote initialization, with final learned values reported separately.

## Initialization versus learning

Paired step-zero and final evaluations at D4/N16 and D64/N128 will identify gains and regressions rather than attributing initialized performance to training. Shallow learning curves will provide optimization context without selecting checkpoints. The initial matrix has only these two anchors, so claims about an unmeasured initial stability boundary will be excluded.

A post-hoc checkpoint audit will report null logits, temperatures, and other already-trained binding parameters. This is an inspection of frozen checkpoints, not a new training or hyperparameter-selection experiment. Null-score changes will be interpreted relative to the independently controlled initial values: zero for cold runs and 0.65 for aligned/null-prior runs.

## Can supervision teach cold binding?

For both attention-write and pointer-write families, the completed comparison will show answer-only training, grounding weights beta=0.01/0.1/1 with fixed null/consistency weights, ground-only supervision, and null-plus-consistency supervision. Query, memory-key, and distractor-null accuracy will locate improvements or failure separately from answer accuracy.

Supervised runs will remain explicitly labeled. Their intermediate labels affect losses only; no gold identity is written into the recurrent state. The consistency loss targets a detached predicted forward graph transition and does not independently enforce correct identity. Low consistency loss with diffuse or incorrect grounding will not count as learned traversal.

## Depth × size stability boundary

The report will link task and complete-path heatmaps for N={16,32,64,128} and D={4,8,16,32,64}, with every cell based on three paired seeds and 128 examples per seed. It will describe one-axis and joint shifts separately and state where high answer accuracy survives despite path failures.

No architecture or hyperparameter will be chosen for another main run by inspecting these OOD cells. Any future adaptive-strength experiment must be separately labeled and meet the gate first. Failure at the corner will leave adaptive crystallization and the interpreter deferred.

## What happens after binding fails?

Canonical trajectory diagnostics use the initial grounding and one post-state per hop. The completed section will report first-error survival, error persistence/recovery, reconvergence, and the distinction between correct final destinations and uninterrupted paths. It will compare empirical complete paths with the product of per-step marginal accuracies only as a descriptive reference.

Per-example associations will relate failures to grounding margin, entropy, query/identity norm, drift, proposed MLP interference, write overrides, distinct visited nodes, and structural strength. These associations can locate failure signatures but cannot by themselves prove an attractor mechanism or causality. Repeated nodes, heterogeneous graph difficulty, and recovery remain alternative explanations.

## Reproducibility, limits, and next decision

Pending completed artifacts: raw compressed records at `results/stage4/main/metrics.jsonl.gz`, resolved configuration, source/config hashes, analysis, checkpoint audit, resource measurements, and verification results. Raw records will preserve paired schedules and evaluation hashes; aggregate values will use final checkpoints and mean/sample SD across seeds, not a confidence interval.

This study remains a small functional-graph task with independent opaque identity keys, explicit relation instructions, immutable memory, and externally scheduled recurrent computation. It does not test language-grounded entity discovery, changing object memory, aliases, scope, or interpreter execution. The final decision will name the limiting component—binding acquisition, state maintenance, null behavior, retrieval, task readout, or combined extrapolation—and distinguish direct observations from proposed explanations.
