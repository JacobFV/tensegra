# Stage 3 report: latent grounding into runtime graphs

## Executive result

Stage 3 produced a partial positive result under a strong keyed-identity prior and a clear failure without it. With structural strength 8, the identity-aligned model was already a perfect validation path grounder before training; optimization learned answer readout but partially damaged traversal. Strength 4 began with zero complete paths and improved them to 0.9271 ± 0.0274 on validation, providing the clearest acquired routing behavior under the identity prior. Its final task accuracy was 0.9661 ± 0.0090 at depth four and 0.8438 ± 0.0563 at depth 32 on 16-node graphs; complete grounding trajectories at depth 32 were 0.6719 ± 0.0590.

The result does not survive the hardest joint shift. At depth 32 and 64 nodes, task accuracy fell to 0.2448 ± 0.0586 for strength 8 and 0.1901 ± 0.0239 for strength 4, near the learned controls that did not traverse. Exact trajectory completion was 0.0234 ± 0.0341 and zero. The privileged fixed-cosine `known` control retained 0.7500 ± 0.1595 task accuracy and 0.6823 ± 0.1936 complete-path accuracy in the same condition. The task and routing algebra were valid: both exact graph traversal and exact one-hot attention achieved 1.0 on every clean condition.

Removing identity-coordinate initialization caused failure. `random_init` scored 0.2188 ± 0.0271 at depth four, with query grounding accuracy 0.1419 ± 0.0296 and no complete paths, even though key grounding reached 0.8062 ± 0.0390. It assigned every distractor incorrectly rather than learning the null channel. This asymmetry localizes much of the failure to recurrent query binding and null calibration, not an inability to map immutable memory keys in isolation.

The keyed supplement changes the graph-input interpretation. With a fixed cosine content bias of 8, graph input reaches 0.9974 ± 0.0045 at depth four and 0.9714 ± 0.0197 at depth 32, while direct `soft_keyed` reaches only 0.4740 ± 0.1890 and 0.3620 ± 0.0705. Both keyed mechanisms already have near-perfect validation paths before training; graph input preserves that routing while soft keyed damages it. The main run therefore does not establish metric-specific superiority of structural logit bias over graph-as-data. Both learned mechanisms still fail at the combined depth-32/64-node shift; the original fixed-cosine known-routing control remains best there at 0.7500 ± 0.1595.

The central conclusion is therefore narrow. A small recurrent model can preserve and use bindings over long unseen instruction sequences when continuous identity coordinates are already aligned, and strength 4 is more stable than the stronger default on depth extrapolation. It did not discover robust grounding from random projections, and learned binding did not generalize jointly to much larger graphs and long depth. The strong goal—general latent grounding suitable as evidence for interpreters or language models—was not met.

## Study scope and evidence

The full methods, input boundary, model equations, metric definitions, and interpretation limits are documented in [stage3-methods.md](stage3-methods.md). Briefly, every example contains fresh normalized Gaussian entity keys, independently sampled categorical payloads, a shuffled immutable memory, and three independently sampled directed functional relations. A supplied relation sequence externally schedules one recurrent update per hop. Training cycles through depths one to four; evaluation reaches depth 32 and 64 nodes. This is a continuous keyed-identity benchmark, not lexical name grounding, language understanding, inferred scheduling, or autonomous stopping.

The full run contains 39 trained models: 13 variants × 3 seeds, each trained for 400 AdamW steps on 12,800 freshly generated examples, or 499,200 training-example presentations in total. Each of 15 final conditions contains 128 examples per seed, giving 74,880 main final-evaluation examples across runs and conditions. The keyed supplement adds 9 runs, 115,200 training-example presentations, and 17,280 final-evaluation examples. The feasibility pilot contains 6 one-seed, 100-step runs (19,200 training-example presentations) and is not pooled with the main evidence. Values below are mean ± sample SD over the three seeded training replicates. Seed-level effects are preserved for the main comparisons. Three seeds support descriptive pairing, not formal significance or confidence claims.

The main raw artifact completed in 167.75 aggregate run-seconds with a process high-water RSS of 430.79 MiB. Every clean exact-routing and one-hot-attention oracle score was 1.0. Training schedules and final evaluation data hashes match across variants within seed, all recorded metrics are finite, and the 39-row artifact is complete.

## Main accuracy results

The following table reports final payload accuracy. `Dk` varies instruction depth at 16 nodes; `N32` and `N64` vary graph size at depth four; `D32/N64` is the declared joint shift. Uniform random class guessing is 0.125 for eight payload classes. That is not the optimal graph-free baseline: visible memory class frequencies, graph cycles, and learned recurrent/content regularities can support higher accuracy, as the 0.20–0.25 no-structure results demonstrate.

| Variant | D1 | D4 | D8 | D16 | D32 | N32 | N64 | D32/N64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| soft, strength 8 | 1.000 ± 0.000 | 0.867 ± 0.090 | 0.745 ± 0.093 | 0.680 ± 0.170 | 0.677 ± 0.138 | 0.820 ± 0.107 | 0.659 ± 0.138 | 0.245 ± 0.059 |
| soft, strength 4 | 0.997 ± 0.005 | 0.966 ± 0.009 | 0.904 ± 0.009 | 0.878 ± 0.020 | 0.844 ± 0.056 | 0.794 ± 0.020 | 0.401 ± 0.073 | 0.190 ± 0.024 |
| known cosine | 0.997 ± 0.005 | 0.997 ± 0.005 | 0.997 ± 0.005 | 0.995 ± 0.005 | 0.974 ± 0.025 | 0.995 ± 0.009 | 0.971 ± 0.030 | 0.750 ± 0.160 |
| learned temperature | 1.000 ± 0.000 | 0.865 ± 0.122 | 0.797 ± 0.136 | 0.742 ± 0.165 | 0.727 ± 0.169 | 0.862 ± 0.114 | 0.714 ± 0.142 | 0.242 ± 0.095 |
| shared strength | 1.000 ± 0.000 | 0.839 ± 0.100 | 0.805 ± 0.102 | 0.776 ± 0.107 | 0.711 ± 0.180 | 0.812 ± 0.118 | 0.729 ± 0.063 | 0.250 ± 0.055 |
| period 4 | 1.000 ± 0.000 | 0.745 ± 0.132 | 0.656 ± 0.115 | 0.589 ± 0.118 | 0.503 ± 0.076 | 0.677 ± 0.163 | 0.576 ± 0.076 | 0.180 ± 0.028 |
| random initialization | 0.250 ± 0.041 | 0.219 ± 0.027 | 0.250 ± 0.047 | 0.208 ± 0.025 | 0.240 ± 0.074 | 0.190 ± 0.055 | 0.164 ± 0.023 | 0.206 ± 0.025 |
| hard | 0.997 ± 0.005 | 0.245 ± 0.016 | 0.245 ± 0.027 | 0.232 ± 0.033 | 0.279 ± 0.033 | 0.211 ± 0.028 | 0.161 ± 0.037 | 0.148 ± 0.021 |
| frozen | 0.971 ± 0.016 | 0.206 ± 0.024 | 0.250 ± 0.028 | 0.206 ± 0.047 | 0.206 ± 0.052 | 0.208 ± 0.035 | 0.138 ± 0.035 | 0.143 ± 0.050 |
| untyped | 0.490 ± 0.030 | 0.195 ± 0.034 | 0.185 ± 0.005 | 0.206 ± 0.039 | 0.232 ± 0.023 | 0.156 ± 0.014 | 0.169 ± 0.024 | 0.128 ± 0.012 |
| graph input | 0.695 ± 0.121 | 0.307 ± 0.024 | 0.357 ± 0.035 | 0.320 ± 0.056 | 0.302 ± 0.009 | 0.245 ± 0.047 | 0.203 ± 0.047 | 0.221 ± 0.016 |
| no structure | 0.229 ± 0.032 | 0.201 ± 0.005 | 0.229 ± 0.066 | 0.206 ± 0.016 | 0.255 ± 0.052 | 0.203 ± 0.043 | 0.167 ± 0.039 | 0.203 ± 0.021 |
| permuted grounding | 0.190 ± 0.012 | 0.219 ± 0.041 | 0.234 ± 0.043 | 0.232 ± 0.012 | 0.201 ± 0.032 | 0.190 ± 0.059 | 0.151 ± 0.023 | 0.128 ± 0.018 |

The identity-initialized soft models are separated from `none` and `permuted`, so their clean gains require correctly aligned runtime structure. The learned-temperature and shared-strength ablations behave broadly like strength 8. The period-4 grounder is worse, showing no benefit from assigning distinct learned projections to the four trained step positions and cycling them later. Within this architecture and 400-step budget, replacing typed relation selection with union adjacency loses valuable routing information beyond one hop.

The hard result is a mechanistic failure, not evidence against exact routing. Hard attention is nearly perfect at depth one, then collapses at depth two and beyond. Its grounding uses a non-differentiable argmax, so final-answer loss cannot repair grounding projections after recurrent state update. The exact one-hot attention oracle uses correct bindings and remains perfect. The comparison shows that hard routing is effective only while its initialized binding is still correct.

The fixed cosine `known` result is the strongest, but it is privileged. It normalizes state/key identity prefixes and applies the known matching rule; learned grounding uses independently trained raw-dot-product projections and is norm-sensitive. Known-cosine success demonstrates that the recurrent architecture and immutable memory can execute the task when binding is supplied reliably. It does not demonstrate that the learned model found that invariant.

## Seed variability and the strength tradeoff

The key task-accuracy triplets, in seed order 0/1/2, are:

| Variant/condition | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| soft 8, D4 | 0.7969 | 0.9688 | 0.8359 |
| soft 8, D32 | 0.5938 | 0.8359 | 0.6016 |
| soft 8, N64 | 0.5312 | 0.8047 | 0.6406 |
| soft 8, D32/N64 | 0.2109 | 0.3125 | 0.2109 |
| soft 4, D4 | 0.9609 | 0.9609 | 0.9766 |
| soft 4, D32 | 0.8281 | 0.7969 | 0.9062 |
| soft 4, N64 | 0.3203 | 0.4219 | 0.4609 |
| soft 4, D32/N64 | 0.1953 | 0.1641 | 0.2109 |
| known, D32/N64 | 0.6953 | 0.9297 | 0.6250 |
| random init, D4 | 0.2500 | 0.2031 | 0.2031 |

Strength 4 is clearly more reliable across depth, while strength 8 is better under size-only extrapolation. At 64 nodes and depth four, strength 8 scores 0.6589 versus 0.4010 for strength 4. At depth 32 and 16 nodes, the order reverses: 0.6771 versus 0.8438. The likely mechanism is a balance between correcting diffuse grounding on larger entity sets and amplifying a slightly wrong route over many recurrent updates. This is an inference from the joint metric pattern, not a direct causal measurement of logits. The D32/N64 collapse of both strengths shows that tuning this scalar alone does not solve the interaction.

The default coefficient remains near its initialization after training; coefficients do not shrink away from the strong prior. Strength-8 runs finish near 8, and strength-4 coefficients finish roughly 4.05–4.21 in the inspected seed. Learned temperature also changes modestly from 0.05 (for example, 0.0566 in seed 0), rather than discovering a qualitatively different binding regime.

## Grounding trajectories and binding failure

Task accuracy can overstate exact traversal because nodes may share one of eight payload classes. The trajectory metrics make the distinction explicit.

| Variant | Condition | Query grounding | Complete grounding path | Complete attention path | Structural next-node mass |
|---|---|---:|---:|---:|---:|
| soft 8 | D4 | 0.910 ± 0.050 | 0.740 ± 0.118 | 0.826 ± 0.092 | 0.903 ± 0.053 |
| soft 8 | D32 | 0.665 ± 0.156 | 0.357 ± 0.246 | 0.383 ± 0.244 | 0.675 ± 0.146 |
| soft 8 | N64 | 0.798 ± 0.095 | 0.523 ± 0.173 | 0.625 ± 0.169 | 0.784 ± 0.095 |
| soft 8 | D32/N64 | 0.255 ± 0.113 | 0.023 ± 0.034 | 0.023 ± 0.034 | 0.257 ± 0.110 |
| soft 4 | D4 | 0.979 ± 0.011 | 0.932 ± 0.005 | 0.958 ± 0.016 | 0.975 ± 0.008 |
| soft 4 | D32 | 0.849 ± 0.045 | 0.672 ± 0.059 | 0.682 ± 0.048 | 0.852 ± 0.042 |
| soft 4 | N64 | 0.667 ± 0.052 | 0.177 ± 0.058 | 0.336 ± 0.087 | 0.651 ± 0.046 |
| soft 4 | D32/N64 | 0.120 ± 0.014 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.130 ± 0.012 |
| known | D32 | 0.976 ± 0.017 | 0.938 ± 0.036 | 0.961 ± 0.023 | 0.940 ± 0.028 |
| known | D32/N64 | 0.840 ± 0.099 | 0.682 ± 0.194 | 0.701 ± 0.175 | 0.795 ± 0.105 |
| random init | D4 | 0.142 ± 0.030 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.132 ± 0.006 |
| hard | D4 | 0.370 ± 0.011 | 0.016 ± 0.008 | 0.052 ± 0.030 | 0.404 ± 0.020 |
| frozen | D4 | 0.235 ± 0.043 | 0.000 ± 0.000 | 0.026 ± 0.012 | 0.238 ± 0.038 |
| untyped | D4 | 0.300 ± 0.008 | 0.003 ± 0.005 | 0.000 ± 0.000 | 0.334 ± 0.023 |

At D32, strength-4 task accuracy is 0.8438 while exact complete grounding is 0.6719. The 0.172 gap can arise when wrong destinations share the target payload, when an incorrect intermediate hop later reconverges on the correct destination, or when grounding probes and the learned readout disagree. The complete-path metric is stringent: one incorrect pre- or post-update binding makes the whole example fail even if later state recovers. Its decline is therefore expected to compound with depth, but the fixed-cosine control shows that near-perfect D32 trajectories are feasible in the same recurrent architecture.

The soft-8 failures are often confident wrong bindings rather than diffuse uncertainty. At D32/N16, mean query grounding accuracy is 0.665 while entropy is about 0.129, far below the maximum \(\log 17=2.83\). At D32/N64, accuracy is 0.255 and entropy about 0.366 versus maximum \(\log 65=4.17\). Projected-norm drift or recurrent update error are plausible contributors, but the metrics establish only low-entropy misbinding, not its cause.

Frozen grounding demonstrates why recurrence must update binding. It is 0.9714 at one hop and 0.2057 at four. Reusing the start assignment routes later relations from the wrong node. The typed/untyped contrast likewise identifies structural use: relation-agnostic union adjacency gives 0.4896 at one step but falls to 0.1953 at four.

### Random initialization

At D4, random initialization has task accuracy 0.2188 ± 0.0271, query grounding 0.1419 ± 0.0296, key grounding 0.8062 ± 0.0390, exact path zero, distractor-null accuracy exactly zero, and mean query null mass 0.0061 ± 0.0005. Its per-seed task accuracies are 0.2500, 0.2031, and 0.2031; no replicate discovers the intended traversal.

The model can learn much of the immutable key-to-entity mapping while failing to maintain the changing query identity. Its D4 query-grounding entropy is about 2.531, so it has not simply collapsed every query onto one node. Key grounding benefits from repeatedly seeing all memory tokens in the answer-loss computation; the query must be correctly retrieved, copied into a recurrent residual, and rebound at each hop. The absence of distractor null predictions shows that answer-only loss did not teach the generic unbound category. This is a specific failure of the tested optimization and architecture, not proof that grounding from random coordinates is impossible.

This control removes two priors together: identity projections become random and null logits change from the calibrated 0.65 threshold to zero. Its null failure therefore cannot be attributed to projection initialization alone.

### Predetermined example trajectory

The generated artifact reports the first evaluation example for seed 0, fixed before success was known. On the D16 `soft` example, every pre-step grounded-node argmax matches the true node. Maximum probabilities are usually above 0.97; the weakest displayed point is step 8, where the correct node still wins with probability 0.7508 and entropy 0.6028. Next-token attention remains at least 0.9508 in the displayed path. The path revisits node 8, illustrating that instruction depth is not the number of distinct entities.

This example is illustrative rather than representative and must not replace aggregate exact-path statistics. Runtime node numbers are graph-local tensor indices, not stable names. The serialized example table records pre-update grounding and attention metadata but not the full `pq_after` distribution. Post-update destination binding is nevertheless included in the aggregate `exact_path_completion` metric. No post-state claim should be inferred from the example table alone.

## Size-depth interaction

Changing one axis at a time gives an incomplete impression. Soft 8 remains 0.6771 at D32/N16 and 0.6589 at D4/N64, but falls to 0.2448 when both axes shift. Soft 4 is 0.8438 and 0.4010 on the separate axes, then 0.1901 jointly. Known cosine also declines from 0.9740 and 0.9714 to 0.7500, showing that the joint condition stresses recurrent state dynamics even when identity matching is fixed. Its smaller loss suggests that binding robustness contributes substantially, but known cosine also changes normalization and score geometry, so this is not a clean isolation of one mechanism.

The functional graphs contain cycles. Mean distinct path nodes at D32/N16 are only 11.5677 ± 0.1641 despite 33 path positions, while D32/N64 visits 23.3073 ± 0.3157 distinct nodes. The larger graph therefore reduces revisitation and demands successful rebinding across more unique identities. The joint failure cannot be described as a pure depth scaling law because graph size changes the path-collision process as well as the grounding competition.

The experiment provides six depths and three sizes, with only one joint large condition. It supports an observed interaction, not an asymptotic scaling claim.

## Held-out composition and distractors

The held-out `(0,1)` relation pair causes little additional loss relative to the matching D4 condition. Soft 8 changes from 0.8672 ± 0.0901 to 0.8516 ± 0.0921; soft 4 changes from 0.9661 ± 0.0090 to 0.9688 ± 0.0135; known remains 0.9974. Because the schedule is supplied and the same typed modules are reused, this demonstrates execution of a new local instruction pair, not induction of a new relation or program.

Distractor count is not monotonically harmful in these 128-example evaluations:

| Variant | 0 distractors | 4 distractors | 16 distractors |
|---|---:|---:|---:|
| soft 8 | 0.8125 ± 0.1094 | 0.8672 ± 0.0901 | 0.8854 ± 0.0631 |
| soft 4 | 0.9714 ± 0.0119 | 0.9661 ± 0.0090 | 0.9115 ± 0.0163 |
| known | 0.9974 ± 0.0045 | 0.9974 ± 0.0045 | 0.9948 ± 0.0045 |
| random init | 0.2318 ± 0.0180 | 0.2188 ± 0.0271 | 0.2448 ± 0.0119 |
| graph input | 0.2839 ± 0.0325 | 0.3073 ± 0.0239 | 0.3411 ± 0.0502 |

The soft-4 decline at 16 distractors is real in all seeds at the mean level, while soft 8 and graph input fluctuate upward. Because each distractor condition uses its own fresh evaluation dataset rather than adding distractors to identical examples, these are condition-level comparisons, not paired token-addition effects. Known cosine's stability shows the memory can reject extra random keys under its fixed matcher. Random initialization has distractor-null accuracy zero at both four and sixteen distractors, so its superficially flat task curve is not evidence of learned rejection.

## Graph corruption and information limits

Corruption rewires the graph supplied to every model without changing the clean answer. The exact oracle reports the clean-answer accuracy obtained by exact traversal of that corrupted supplied graph: 0.7682 ± 0.0239 at 10%, 0.4635 ± 0.0045 at 25%, and 0.2396 ± 0.0180 at 50%. Its exact clean-path rates are 0.6901 ± 0.0369, 0.3359 ± 0.0341, and 0.0755 ± 0.0239. This is a diagnostic, not a proven Bayes-optimal or information-theoretic ceiling; class-frequency regularities or learned priors can exceed its answer accuracy without recovering the clean path.

| Variant | Clean | Corruption 10% | Corruption 25% | Corruption 50% |
|---|---:|---:|---:|---:|
| direct/attention oracle | 1.000 ± 0.000 | 0.768 ± 0.024 | 0.464 ± 0.005 | 0.240 ± 0.018 |
| soft 8 | 0.867 ± 0.090 | 0.680 ± 0.055 | 0.440 ± 0.025 | 0.263 ± 0.035 |
| soft 4 | 0.966 ± 0.009 | 0.745 ± 0.039 | 0.456 ± 0.005 | 0.242 ± 0.021 |
| known cosine | 0.997 ± 0.005 | 0.768 ± 0.024 | 0.464 ± 0.005 | 0.240 ± 0.018 |
| learned temperature | 0.865 ± 0.122 | 0.672 ± 0.087 | 0.430 ± 0.023 | 0.245 ± 0.009 |
| shared strength | 0.839 ± 0.100 | 0.648 ± 0.075 | 0.427 ± 0.020 | 0.237 ± 0.020 |
| period 4 | 0.745 ± 0.132 | 0.602 ± 0.082 | 0.380 ± 0.080 | 0.232 ± 0.016 |
| random initialization | 0.219 ± 0.027 | 0.219 ± 0.041 | 0.224 ± 0.032 | 0.208 ± 0.005 |
| hard | 0.245 ± 0.016 | 0.242 ± 0.043 | 0.240 ± 0.053 | 0.206 ± 0.024 |
| frozen | 0.206 ± 0.024 | 0.201 ± 0.012 | 0.180 ± 0.021 | 0.195 ± 0.008 |
| untyped | 0.195 ± 0.034 | 0.190 ± 0.016 | 0.208 ± 0.024 | 0.195 ± 0.023 |
| graph input | 0.307 ± 0.024 | 0.320 ± 0.000 | 0.292 ± 0.012 | 0.260 ± 0.020 |
| no structure | 0.201 ± 0.005 | 0.201 ± 0.005 | 0.201 ± 0.005 | 0.201 ± 0.005 |
| permuted | 0.219 ± 0.041 | 0.219 ± 0.051 | 0.232 ± 0.032 | 0.188 ± 0.023 |

Known cosine exactly tracks the direct oracle means at every corruption level, as expected for a model that reliably follows the supplied graph. Soft 4 nearly does so; at 10% it obtains 0.7448 and at 25% 0.4557. At 50%, several learned models slightly exceed the direct-oracle mean, but the small margins and shared categorical payloads allow correct answers after wrong routes; this is not recovery of clean topology. The no-structure model stays constant because corruption never enters its computation.

Complete-path and attention-path rates decline faster than answer accuracy. At 25% corruption, soft-4 task accuracy is 0.4557 while exact grounding-path accuracy is 0.3073. At 50%, they are 0.2422 and 0.0729. These gaps reinforce that payload accuracy cannot certify clean traversal.

## Graph-input control and the alignment confound

Graph input is intentionally strong in one sense: fixed cosine identity matching maps token memories into graph slots before relation-specific neighbor aggregation. It is weak in the observed task result, reaching 0.3073 at D4 and 0.3021 at D32, but this does not isolate graph encoding from alignment. Its fixed cosine matcher is a privileged identity prior, while its content-attention query does not receive the same explicit common cosine score used by direct structural routing.

Its attention diagnostics are also not mechanism-comparable. The graph-input operator places a successor message in the source token's value, so attending the current/source token can retrieve the next state correctly. Low next-node-token attention and zero “exact attention path” therefore do not establish failed message passing. Task accuracy and post-update grounding remain applicable.

The primary main-run comparison consequently leaves a confound: `known` combines fixed cosine identity alignment with a strong structural logit route, whereas graph input combines fixed cosine graph aggregation with learned ordinary content retrieval. The keyed supplement below adds a fixed cosine content bias to expose this difference. It does not make all mechanisms or effective priors identical, because the same content bias favors different token roles in the two architectures.

## Training dynamics and resource cost

Checkpoint validation reveals that initialization, rather than final accuracy alone, determines what was learned. Values are answer accuracy / query-grounding accuracy / complete grounding trajectory.

| Variant | Step 0 | Step 25 | Step 50 | Step 100 | Step 200 | Step 400 |
|---|---|---|---|---|---|---|
| soft 8 | .102 / 1.000 / 1.000 | .828 / .938 / .846 | .747 / .837 / .589 | .568 / .678 / .302 | .565 / .680 / .339 | .839 / .896 / .766 |
| soft 4 | .104 / .383 / .000 | .297 / .574 / .104 | .662 / .830 / .544 | .596 / .742 / .401 | .688 / .789 / .497 | .951 / .970 / .927 |
| known | .102 / 1.000 / 1.000 | .841 / .962 / .878 | .729 / .773 / .414 | .865 / .891 / .693 | .979 / .982 / .927 | .997 / .997 / .987 |
| hard | .102 / 1.000 / 1.000 | .802 / .915 / .802 | .456 / .577 / .188 | .318 / .449 / .078 | .315 / .437 / .068 | .300 / .395 / .042 |

Soft 8 and known start with functional routing but random answer readouts. Soft 8 damages perfect paths, bottoms near step 100, and only partly recovers. Known also dips before nearly recovering its fixed-matcher path. Hard deteriorates monotonically after step 25 because the non-differentiable grounder cannot be repaired by answer loss. Soft 4 is different: its weaker initial route yields zero complete paths, and optimization acquires a 0.927 path rate by step 400. The 400-step endpoint was fixed in advance; no checkpoint was selected by final test performance, and the table does not define an early-stopping policy.

Other final validation answer accuracies are learned temperature 0.8958 ± 0.0828, shared strength 0.9141 ± 0.0234, period 4 0.8021 ± 0.1138, graph input 0.3099 ± 0.0743, random initialization 0.2292 ± 0.0828, and none 0.2266 ± 0.0435. The random, graph-input-main, frozen, untyped, none, and permuted configurations do not enter a successful final traversal regime within 400 steps; hard instead begins there and loses it.

Resource figures are implementation measurements, not isolated hardware claims. Total seconds include checkpoint validation, full diagnostics, and oracles; training seconds cover optimization; plain inference uses diagnostics disabled, batch 32, depth four, three warmups and ten repeats. Peak RSS is a cumulative process high-water mark.

| Variant | Parameters | Total seconds | Training seconds | Plain batch inference, ms |
|---|---:|---:|---:|---:|
| soft 8 | 9,303 | 4.740 ± 1.137 | 2.871 ± 0.820 | 3.382 ± 0.237 |
| soft 4 | 9,303 | 4.057 ± 0.560 | 2.310 ± 0.343 | 3.643 ± 0.416 |
| known | 9,303 | 3.276 ± 0.322 | 1.808 ± 0.286 | 3.329 ± 0.498 |
| learned temperature | 9,304 | 4.294 ± 0.932 | 2.484 ± 0.624 | 5.367 ± 2.358 |
| shared strength | 9,294 | 4.075 ± 0.774 | 2.342 ± 0.509 | 3.819 ± 0.873 |
| period 4 | 13,953 | 4.798 ± 1.574 | 2.795 ± 1.147 | 4.556 ± 0.926 |
| random initialization | 9,303 | 4.259 ± 1.028 | 2.470 ± 0.678 | 4.891 ± 2.384 |
| hard | 9,303 | 5.008 ± 1.804 | 2.894 ± 1.481 | 3.953 ± 0.148 |
| frozen | 9,303 | 3.692 ± 0.746 | 2.113 ± 0.516 | 2.566 ± 0.039 |
| untyped | 9,295 | 4.735 ± 1.351 | 2.864 ± 0.908 | 3.948 ± 0.754 |
| graph input | 9,303 | 5.136 ± 1.652 | 2.961 ± 1.168 | 3.581 ± 0.455 |
| no structure | 9,303 | 3.861 ± 1.252 | 1.827 ± 0.811 | 3.351 ± 1.940 |
| permuted | 9,303 | 3.987 ± 0.263 | 2.180 ± 0.040 | 4.259 ± 1.059 |

The timing SD is large relative to some mean differences, the whole study runs in one CPU process, and several variants compute different diagnostic paths. These numbers do not establish a speed advantage. Dense ordinary attention remains in every learned model.

## Keyed supplemental controls

The supplemental run adds a fixed content-attention bias of \(8\cos(h_t,k_j)\), computed from the current state's identity prefix and each token key, to three variants. `none_keyed` receives only that bias. `soft_keyed` combines it with learned soft structural routing initialized at strength 16. `graph_input_keyed` uses the same content bias after fixed-cosine graph message construction; its configured structural strength is not an active logit route in graph-input mode. The supplement contains nine separately trained runs (three variants × three seeds), uses the same 400-step schedule and evaluation grid, and was executed from source `27ca0a2` after an exact-source gate of 211 tests in 2.18 seconds.

The common cosine-content term has architecture-specific effects. In direct soft routing, the structural bias favors the relation successor token while the content term favors the current-identity token, so the two can compete. In graph input, the successor message has already been placed in the current/source token value, so favoring that source token is the desired retrieval rule. The supplement therefore tests a specific alignment intervention; it does not equalize every effective prior or parameter path.

| Variant | D1 | D4 | D8 | D16 | D32 | N32 | N64 | D32/N64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| soft keyed, structural 16 | 1.000 ± 0.000 | 0.474 ± 0.189 | 0.401 ± 0.078 | 0.370 ± 0.070 | 0.362 ± 0.070 | 0.432 ± 0.140 | 0.305 ± 0.083 | 0.198 ± 0.016 |
| graph input keyed | 0.997 ± 0.005 | 0.997 ± 0.005 | 0.995 ± 0.009 | 0.979 ± 0.016 | 0.971 ± 0.020 | 0.992 ± 0.008 | 0.922 ± 0.051 | 0.268 ± 0.092 |
| no structure keyed | 0.188 ± 0.014 | 0.172 ± 0.031 | 0.203 ± 0.034 | 0.164 ± 0.059 | 0.148 ± 0.028 | 0.128 ± 0.039 | 0.109 ± 0.021 | 0.143 ± 0.012 |

Graph input keyed nearly solves the one-axis conditions: 0.9974 ± 0.0045 at D4, 0.9714 ± 0.0197 at D32, and 0.9219 ± 0.0512 at N64. It begins with validation query/path grounding 1.0/1.0 and ends with complete-path accuracy 0.9505, so this is preservation plus learned answer readout under initialized routing, not discovery of a binding rule. The result reverses the main run's apparent advantage for soft structural bias over graph input. That main difference was not mechanism-specific superiority; it depended on which architecture had an effective identity-aligned retrieval prior.

Trajectory diagnostics qualify the high task scores. Graph input keyed complete grounding is 0.9245 ± 0.0592 at D4, 0.7083 ± 0.1683 at D32, and 0.6432 ± 0.1340 at N64. Its exact next-token attention path is near zero by construction because it retrieves successor messages from source-token positions; that diagnostic remains mechanism-inapplicable. On D32/N64, task accuracy falls to 0.2682 ± 0.0916 and complete grounding to 0.0208 ± 0.0239. The supplement fixes the one-axis graph-input failure but not the size-depth interaction.

Soft keyed is a negative result. Raising structural strength to 16 while also favoring the current identity yields D4 task accuracy 0.4740 ± 0.1890, D32 0.3620 ± 0.0705, N64 0.3047 ± 0.0827, and D32/N64 0.1979 ± 0.0163. Validation complete-path accuracy starts at 0.9948 but falls to 0.2839, the strongest direct evidence that optimization damages its initialized route. The intervention changes content and structural biases together, so the outcome is consistent with interference but does not isolate which change causes it.

Under corruption, graph input keyed exactly tracks the direct supplied-graph traversal reference means: 0.7682 ± 0.0239, 0.4635 ± 0.0045, and 0.2396 ± 0.0180 at 10%, 25%, and 50%. This confirms close adherence to the supplied graph without treating that reference as a Bayes-optimal ceiling. Soft keyed scores 0.4141 ± 0.1213, 0.3385 ± 0.0430, and 0.2370 ± 0.0119; no-structure keyed remains 0.1719 ± 0.0312 throughout.

The supplement used 34.76 aggregate run-seconds with peak process RSS 420.58 MiB. Mean total/training/plain-batch-inference times were 5.073 ± 1.264 s / 3.082 ± 1.003 s / 4.511 ± 1.815 ms for soft keyed; 3.593 ± 0.210 s / 1.914 ± 0.173 s / 3.626 ± 0.421 ms for graph input keyed; and 2.921 ± 0.013 s / 1.311 ± 0.011 s / 2.097 ± 0.140 ms for no-structure keyed.

The original fixed-cosine `known` control remains the best joint-shift model at 0.7500 ± 0.1595 D32/N64 accuracy. Even the strongly keyed graph-input supplement reaches only 0.2682 there. Fixed identity matching alone therefore does not guarantee robust learned recurrent traversal under the combined shift.

## Interpretation and next decision

The observations support three positive statements. First, the recurrent update and induced-bias mechanism can execute unseen runtime graphs when identity binding is known or strongly initialized. Second, identity-initialized soft grounding can extrapolate far beyond the trained depth on 16-node graphs, especially at strength 4. Third, typed relation selection, recurrent rebinding, and aligned graph structure each matter: untyped, frozen, and permuted controls fail.

They also expose the limiting failures. Randomly initialized grounding does not learn query binding or null rejection from answer-only supervision. Hard routing loses the gradient needed to repair recurrent binding. Size and depth interact destructively. Task accuracy can remain high after exact trajectory accuracy has deteriorated. The fixed-cosine control remains substantially stronger than all learned grounders in the joint large/deep condition.

This is partial evidence for a keyed routing module, not broad evidence for a latent interpreter. The identity-coordinate initialization, immutable key-value memory, externally supplied relation schedule, total-function graph, and small categorical answer space simplify away the hardest semantic grounding questions. Continuous keys do not model lexical names. Deeper evaluation supplies more recurrent compute. Three seeds and one joint size-depth point do not establish a scaling law.

The strong milestone is not met. Interpreter, pretrained-language-model, or embodied follow-ups should remain deferred. The keyed supplement resolves the main graph-input weakness on one-axis tests but also shows that a common cosine term interacts differently with source-message and successor-token mechanisms, and it leaves the joint shift unsolved. A focused next experiment is justified only if it targets the observed failure directly: explicitly mechanism-matched identity priors, stronger null supervision or an explicitly labeled auxiliary-grounding study, and a broader joint depth × size grid. Any auxiliary path supervision must be separated from the primary answer-only result.

## Artifacts

Main raw results: [metrics.jsonl](results/stage3/main/metrics.jsonl) and [resolved config](results/stage3/main/config.json). The complete generated main numerical appendix is [analysis/report.md](results/stage3/main/analysis/report.md), with machine-readable aggregates in [analysis/summary.json](results/stage3/main/analysis/summary.json).

Supplemental raw results: [keyed metrics.jsonl](results/stage3/keyed/metrics.jsonl) and [keyed config](results/stage3/keyed/config.json). Its complete generated appendix is [keyed analysis/report.md](results/stage3/keyed/analysis/report.md), with aggregates in [keyed analysis/summary.json](results/stage3/keyed/analysis/summary.json).

Integrity and environment: [audit.json](results/stage3/audit.json) records 50 artifact hashes and family-level invariants; [environment-stage3.txt](environment-stage3.txt) records the remote CPU environment. Final source `85e6017` passed 215 tests in 2.17 seconds.

Primary figures: [main training curves](results/stage3/main/analysis/figures/training-curves-primary.svg), [main depth panel A](results/stage3/main/analysis/figures/depth-f814afda.svg), [main depth panel B](results/stage3/main/analysis/figures/depth-9c4cee73.svg), [main node-count panel](results/stage3/main/analysis/figures/nodes-171c914e.svg), and [main corruption](results/stage3/main/analysis/figures/corruption-55ff183a.svg). Supplemental figures: [keyed training curves](results/stage3/keyed/analysis/figures/training-curves-keyed.svg), [keyed depth panel A](results/stage3/keyed/analysis/figures/depth-f814afda.svg), [keyed depth panel B](results/stage3/keyed/analysis/figures/depth-9c4cee73.svg), and [keyed node-count panel](results/stage3/keyed/analysis/figures/nodes-171c914e.svg).
