# A04 development: fresh-code content selection is acquired

This is a one-seed exploratory acquisition result, not confirmation. Four width1024 models receive fresh continuous attribute codes and random typed graphs. Each source has four typed neighbors with distinct attributes, while each attribute also labels eight nodes globally. The relation and desired attribute are supplied instructions; their execution order is programmed. The learned boundary is query/key matching plus payload propagation, not planning or arbitrary semantic grounding.

All three graph interfaces—soft structural bias, exact-address neighbor attention, and the keyed neighborhood selector—reach100% payload accuracy on every clean development condition by25updates and retain it at1000updates. Conditions separately cover N32/D4, N64/D4, N32/D8, N64/D8, held-out adjacent relation composition and K8 neighborhoods. Each condition has256 fresh development examples. The no-graph ablation remains25.39% at N32/D4 and12.50% at N64/D8. No soft-attention superiority appears.

The learned query/key maps start random. Initial IID diagnostic path accuracy is4.69%; by25updates it is100%. Initial task accuracy is around6%. There is no oracle route supervision: all arms receive the same privileged suffix-value loss. At25updates this represents400 graph presentations but31,232 supervised node/depth targets, not merely400 scalar labels. The content code coordinates and graph-node identities are supplied; this is a narrow matching interface.

## Input interventions

On paired instruction swaps, every graph arm answers the new query correctly. Of256 pairs,173 reach a different terminal node and163 have a different payload answer. Accuracy is100% on both changed subsets. The difference between these support counts matters: first-hop changes can merge later and payload labels can collide.

Supplying the swapped instruction while retaining the original scoring target yields36.33% original-answer agreement but100% agreement with the actually supplied instruction. Removing the learned content score drops graph-arm task accuracy to12.5–12.9% and diagnostic path accuracy to0–0.39%. These controls support acquired content dependence, not a terminal-attribute shortcut. Complete comparisons and supplied targets are retained for independent reconstruction.

## Remaining mechanistic question

Correct-route argmax is not the same as sharp value transfer. Learned selected-neighbor mass is roughly.935 atK4 and.870 atK8 for soft/exact-gather models; keyed retrieval is slightly more diffuse. Longer recurrence may expose cumulative mixing even though the current depth8 task is solved. A fresh confirmation should separate N, depth and neighborhood count, retain the strong alternatives, and score payload/path/mass separately.

The keyed neighborhood selector uses an explicit hard supplied-address mask plus a known-key retrieval prior. It is not an unconstrained transformer parsing graph tokens. Exact address gather and hard masked neighbor attention are algebraically equivalent and count as one baseline.

## Provenance and decision

Source2ba70af; seed301; four matched1000-update runs,16,000 graph draws each; no canonical deduplication claim. Width1024,4,306,986 allocated parameters, AdamW learning rate.0003 and weight decay1e-4. The final loss is approximately1e-5 for each graph arm,1.85 for no graph. Extending training is not justified by these saturated curves.

Full development controller occupancy63.01seconds: soft16.82, exact gather13.98, keyed context17.14, no graph15.05. The preceding mechanical profile cost3.64seconds. Six CPU contract tests passed, including unique local matches, repeated global codes, direct reference paths, permutation behavior, gradients, exact-gather/mask equivalence and independent supplied-target reconstruction. Independent raw audit is requested; historical artifacts remain unchanged.

Next: narrow three-seed confirmation on fresh constructions and larger separate/joint shifts. No architecture changes or route auxiliary are justified by this development result. No composition or calibrated execution claim follows from it.

## Subsequent generator-only limitation audit

Before extending this benchmark to depth32, a CPU-only audit (128 new N128/K8 graphs, no learned model) found the mean number of distinct terminal identities falls from16 after one hop to6.17 after four,2.30 after sixteen and1.38 after thirty-two. Changing the first instruction changes the terminal entity in only12.5% of these depth32 episodes. Thus a longer supplied path is not automatically sustained identity-sensitive computation. A04's fresh-code acquisition result remains valid, but no deep identity-maintenance claim is made. The unrun A05 confirmation draft is superseded by a separately versioned block-permutation generator and a fresh development screen.
