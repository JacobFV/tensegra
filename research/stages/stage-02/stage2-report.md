# Stage 2 report: what programmed topology changes in small dynamical predictors

## Scope and status

Stage 2 asks whether supplying a dependency graph changes how much data or optimization a small transformer needs, how sensitive that benefit is to graph errors, and whether it transfers to unseen graphs and larger systems. The study deliberately remains below the vision-language-action level: every task is a synthetic multivariate dynamical process with observed scalar node histories. The robot domain is a fixed morphology-shaped graph with synthetic local couplings. It is neither a physical robot experiment nor a set of independent morphology draws.

The training source revision is `ee49edc`. The preregistered full preset contains 570 runs across six suites: 216 identity-free efficiency runs, 108 identity-enabled efficiency controls, 150 corruption runs, 30 transfer runs, 42 heterogeneous-mechanism runs, and 24 learned-strength runs. Three seeded replicates are used throughout. This report treats those seeds as descriptive replicates and reports population standard deviations and individual paired effects. With only three seeds, no result is described as formally statistically significant.

All 570 planned runs completed 600 optimizer steps in 1,094.46 seconds (18.24 minutes), with process peak RSS 462.88 MiB. Numerical claims below were derived independently from the explicit raw fields and then checked against the source-bound analysis artifacts. The JSONL and summary contain the same 570 rows, every metric is finite, transfer graph splits are disjoint, and initialization, schedule, checkpoint, final-state, and threshold-crossing audits pass.

## Main result

Correct topology consistently improves finite-budget one-step prediction in the 128-trajectory fixed-graph comparison and across the unseen-graph transfer suite, and the effect depends on edge alignment rather than bias magnitude alone. At 128 trajectories, strength-4 bias reduces normalized one-step error in every seed by 3.0–6.9% on sparse graphs and 8.2–11.9% on the robot graph; matched strength-4 permutation is flat or worse. Adding node identities does not remove the effect. The strict 10% oracle-gap target is crossed much earlier with correct topology, but the smallest-sample crossings are transient: every eight-trajectory soft4/hard run rises above that target again by step 600. Sparse eight-trajectory terminal test error reverses the large-budget ranking: `none` is 0.7181, versus 0.7591 for `soft4` and 0.7854 for `hard`. The strongest claim is therefore faster early optimization at the larger tested budgets, with a conditional terminal-data advantage, rather than a universal sample-complexity result.

Topology transfers to unseen graph identities and sizes. From training at 12 nodes, soft4, hard, and graph-as-data beat `none` at 12, 32, 64, and 128 nodes in every seed; hard and graph-as-data retain about a 7–8% one-step advantage at 128. Moderate graph corruption degrades performance gradually when present during training, while severe deployment-only corruption can erase the clean advantage. Signed coefficient types help beyond adjacency, especially on robot, but receive privileged sign and magnitude information. Learned per-head strengths improve one-step prediction yet fail badly in recursive rollout, with deterministic errors more than three times the unbiased baseline. One-step gains are therefore not sufficient evidence of stable dynamics.

## Experimental system

### Dynamics and graph families

Every trajectory follows

\[
x_{t+1}=0.5x_t+0.5\tanh(Wx_t)+\epsilon_t,
\]

where \(\epsilon_t\) is independent Gaussian innovation noise with standard deviation 0.01. Each retained sequence contains 40 observations after 32 burn-in updates. Uniform mechanisms assign equal nonself weights whose active-row sum is approximately 0.8. Sparse graphs use edge probability \(\min(3/(N-1),1)\), holding expected indegree near three as node count increases. The robot case uses one fixed 12-node morphology and coefficient matrix; its three seeds vary trajectories and initialization, while sparse seeds also vary graph topology.

The heterogeneous family keeps the same adjacency but uses nonuniform positive and negative coefficients, normalized to absolute row sum below 0.8. This gives a sufficient contraction bound: the transition has infinity-norm Lipschitz constant at most 0.9. The signed experiment tests whether a model benefits from receiving actual typed coefficient information in addition to adjacency. It does not test recovery of arbitrary hidden coefficients from adjacency alone.

### Models and graph information

All predictors consume four observations per node and use width 32, four attention heads, two attention blocks, tokenwise normalization and MLPs, and one scalar next-state output per node. The base model has no node identifier. Its permutation equivariance is useful for transfer, but on a fixed asymmetric graph it also means that an unbiased model cannot directly memorize arbitrary variable-specific adjacency by name.

The identity-free efficiency suite compares unbiased attention (`none`), finite correct-graph biases of 1 and 4 (`soft1`, `soft4`), a hard graph mask (`hard`), and matched-strength permuted-graph controls (`permuted1`, `permuted4`). A permuted control relabels the adjacency without relabeling the state variables, preserving edge count and bias strength while breaking the graph-state correspondence. The supplementary identity suite gives `none`, `soft4`, and `hard` the same learned 12-by-32 node-embedding table. This makes representation of a fixed named adjacency possible, although 600 optimizer steps do not prove that the unbiased control has exhausted that capacity.

The graph-as-data baseline computes a permutation-equivariant normalized average of neighbor histories, projects it, and adds it to each node input while leaving attention unbiased. This is an explicit graph-conditioned input/message-passing baseline, not a graph-token serialization baseline. The signed typed-bias model receives separate positive and negative channels and their magnitudes. Learned-strength models replace a fixed scalar strength with an unconstrained coefficient for every layer and head, initialized exactly at zero.

Hard masks restrict reads but do not make execution sparse: all variants use dense attention kernels. Runtime and memory comparisons therefore measure this implementation, not a sparse-attention speedup.

### Training, data budgets, and pairing

AdamW trains for 600 steps with batch size 32 and learning rate 0.001. Validation checkpoints occur at steps 0, 25, 50, 100, 200, 300, and 600. Fixed-process efficiency training sets contain nested prefixes of 8, 16, 32, 64, 128, or 256 independent trajectories. Validation and test each contain 16 separate trajectories. Scalar normalization is fit once on the eight-trajectory training subset and reused at every budget, so a small-budget run never obtains statistics from the additional trajectories.

Within a seed, case, and trajectory count, variants share common parameter tensors at initialization and the exact graph/window schedule. Across counts, trajectory sets are nested and random seeds are shared, but index ranges and realized schedules differ; cross-count comparisons are therefore not exact batch-for-batch pairings. Initial, checkpoint, and final parameter hashes, schedule hashes, graph hashes, weight hashes, split seeds, training windows, optimizer examples, elapsed time, and peak memory are retained.

For transfer, one identity-free model per seed trains on 16 graphs and validates on four unseen graphs. Test evaluation uses eight further unseen graphs at 12, 32, 64, and 128 nodes. One condition trains only at 12 nodes; another mixes 9, 12, and 16 nodes. Each training graph supplies 128 trajectories and batches sample graphs uniformly before sampling windows. Graph hashes enforce disjoint train, validation, and test identities. The design compares unbiased, correct soft, hard, permuted, and graph-as-data models.

### Corruption conditions

Graph corruption changes only the graph supplied to the predictor; the generating matrix remains unchanged. Drop and add conditions alter rounded fractions of the original directed nonself edge count at 10%, 25%, and 50%. The mixed stress condition independently drops and adds 25%. Self edges are retained. Each result records realized precision, recall, correct edges, dropped edges, added edges, and the supplied-graph hash. On the symmetric robot truth, directed corruption can break symmetry, so this manipulation represents incomplete dependency knowledge rather than removal of a bidirectional physical linkage.

Retrained-corruption models see the same corrupted supplied graph during training and validation. A separate runtime-corruption evaluation applies corrupted graphs to clean-trained models without retraining. These answer different questions: whether a model can adapt to consistently imperfect structure, and whether a trained model survives a graph error introduced only at deployment.

## Measurements

One-step error is evaluated on at most 256 deterministic, equally spaced windows per graph. Recursive rollout starts from the same observed four-step history and predicts ten future states. Stochastic rollout compares predictions with the realized noisy future. Deterministic rollout instead iterates the known noise-free transition from that observed history. It isolates error against one deterministic continuation, but because the transition is nonlinear it is not the exact multistep conditional expectation of the stochastic process. The deterministic oracle error is zero by construction; the stochastic oracle is a diagnostic and is not asserted to be the optimal stochastic predictor.

All primary error fields are retained both as raw MSE and divided by the training variance. Multi-graph conditions average within each seed before the three-seed mean and population SD are computed. Windows, nodes, and graphs are not treated as independent training replicates.

Optimization-curve AUC is the trapezoidal integral of normalized validation MSE from step 0 through 600, divided by 600. It incorporates initialization and the whole observed learning curve; it is neither final validation error nor test error, and a high initial error can dominate it.

When a single finite soft strength is needed for a primary comparison, it is selected within each training condition by the across-seed mean validation error at step 600. Test results never choose the strength. Both strength-1 and strength-4 sensitivity results remain in the tables. Runtime-corruption evaluation inherits the strength selected on the corresponding clean-trained validation condition rather than reselecting on corrupted test data. Strength 4 is selected for both clean-trained corruption conditions and hence for both runtime-corruption series. Among retrained corruption conditions, strength 4 is selected everywhere except sparse drop-50% and sparse mixed-25%, where strength 1 is primary.

The three validation targets were fixed before the full study:

\[
T_f=E_{oracle}+f(E_{zero}-E_{oracle}),\qquad f\in\{0.10,0.25,0.50\}.
\]

For a given trajectory budget, optimizer efficiency \(s_\epsilon\) is the first observed checkpoint at or below the target. Sample efficiency \(n_\epsilon\) is the smallest tested trajectory count whose curve reaches it. A missing crossing through 600 steps is reported as unreached, not converted into an infinite speedup. These are checkpoint-grid first crossings: they do not interpolate between checkpoints, require a sustained crossing, or establish asymptotic convergence.

## Efficiency results

### Identity-free fixed-graph sweep

The 128-trajectory slice shows a consistent one-step and whole-curve advantage for correct topology. Values below are mean ± population SD across the three seeds. “One-step effect” is each seed's percentage reduction in normalized test one-step MSE relative to its paired `none` run; listing all three prevents the mean from hiding a contrary replicate.

| Domain | Mode | Test one-step | Deterministic rollout | Stochastic rollout | Validation AUC | One-step effect by seed |
|---|---:|---:|---:|---:|---:|---:|
| sparse | none | 0.6266 ± 0.0086 | 0.0474 ± 0.0257 | 0.9583 ± 0.0642 | 0.6299 ± 0.0050 | reference |
| sparse | soft1 | 0.6020 ± 0.0080 | 0.0582 ± 0.0254 | 0.9695 ± 0.0728 | 0.6077 ± 0.0085 | +2.8%, +5.1%, +3.9% |
| sparse | permuted1 | 0.6274 ± 0.0094 | 0.0495 ± 0.0267 | 0.9564 ± 0.0721 | 0.6307 ± 0.0063 | −0.0%, +0.0%, −0.3% |
| sparse | soft4 | 0.5940 ± 0.0081 | 0.0379 ± 0.0091 | 0.9484 ± 0.0579 | 0.5983 ± 0.0123 | +3.0%, +6.9%, +5.7% |
| sparse | permuted4 | 0.6330 ± 0.0095 | 0.0513 ± 0.0233 | 0.9451 ± 0.0574 | 0.6372 ± 0.0071 | −0.8%, −1.0%, −1.2% |
| sparse | hard | 0.5940 ± 0.0082 | 0.0378 ± 0.0048 | 0.9465 ± 0.0516 | 0.5981 ± 0.0129 | +2.9%, +7.0%, +5.8% |
| robot | none | 0.5600 ± 0.0332 | 0.0998 ± 0.0349 | 0.9430 ± 0.0576 | 0.5613 ± 0.0384 | reference |
| robot | soft1 | 0.5177 ± 0.0349 | 0.0903 ± 0.0186 | 0.8965 ± 0.0405 | 0.5254 ± 0.0306 | +9.5%, +6.5%, +6.7% |
| robot | permuted1 | 0.5594 ± 0.0325 | 0.0904 ± 0.0316 | 0.9281 ± 0.0617 | 0.5614 ± 0.0365 | −0.1%, +0.1%, +0.3% |
| robot | soft4 | 0.5050 ± 0.0324 | 0.0747 ± 0.0506 | 0.9007 ± 0.0633 | 0.5126 ± 0.0307 | +11.9%, +8.2%, +9.4% |
| robot | permuted4 | 0.5596 ± 0.0327 | 0.0732 ± 0.0145 | 0.8899 ± 0.0463 | 0.5658 ± 0.0368 | +0.4%, −0.4%, +0.2% |
| robot | hard | 0.5042 ± 0.0317 | 0.0725 ± 0.0440 | 0.8933 ± 0.0492 | 0.5123 ± 0.0307 | +11.9%, +8.3%, +9.7% |

The one-step result cleanly distinguishes correct edge placement from generic bias strength: both permuted controls remain around the unbiased result, whereas correct soft and hard topology improve all three seed-level one-step comparisons. The rollout picture is less uniform. Sparse `soft1`, for example, improves one-step error and AUC but has a worse mean deterministic and stochastic rollout error than `none`. On robot, `permuted4` has no one-step benefit yet a low mean rollout error. Those negative and discordant results rule out treating a one-step gain as automatic evidence of better recursive dynamics.

Sample-threshold results are compactly reported as `trajectories@first step`; `U` means the target was never reached at any tested count through step 600. Each cell gives seeds 0/1/2. The columns correspond to the 10%, 25%, and 50% remaining oracle-to-zero gap targets.

| Domain | Mode | 10% target | 25% target | 50% target |
|---|---|---|---|---|
| sparse | none | 16@300 / 16@200 / 16@200 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| sparse | soft1 | 8@50 / 8@50 / 8@25 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| sparse | permuted1 | 16@300 / 16@200 / 64@200 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| sparse | soft4 | 8@50 / 8@50 / 8@25 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| sparse | permuted4 | 128@600 / 64@600 / 128@600 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| sparse | hard | 8@50 / 8@50 / 8@25 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| robot | none | 32@600 / U / U | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| robot | soft1 | 8@200 / 8@50 / 8@50 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| robot | permuted1 | U / U / U | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| robot | soft4 | 8@50 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| robot | permuted4 | U / U / U | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |
| robot | hard | 8@50 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 | 8@25 / 8@25 / 8@25 |

The loose 25% and 50% targets saturate at the smallest sample budget and first post-initialization checkpoint for every variant, so they show no measured savings at the available resolution. The preregistered 10% target distinguishes first crossings. Correct `soft4` and `hard` cross it with eight trajectories in every seed and domain; sparse `none` first crosses with 16 trajectories, while robot `none` is censored in two seeds. The matched strength-4 permutation is especially revealing: it is substantially slower than `none` on sparse and never crosses the robot 10% target.

That apparent eight-trajectory sample advantage is transient. Every eight-trajectory `soft4` and `hard` run—both domains, both efficiency suites, all seeds—is back above the 10% target at step 600. Because the preregistered definition uses first crossing rather than sustained or terminal accuracy, `n_epsilon=8` is a validation first-crossing result, not an evaluated early-stopping policy, and should not be read as stable sample complexity. As a supplemental terminal-budget check, the smallest count at or below the 10% target specifically at step 600 is:

| Suite/domain | none, seeds 0/1/2 | soft4, seeds 0/1/2 | hard, seeds 0/1/2 |
|---|---|---|---|
| identity-free sparse | 32 / 32 / 64 | 16 / 16 / 16 | 16 / 16 / 16 |
| identity-free robot | 32 / U / U | 16 / 16 / 16 | 16 / 16 / 16 |
| identity-enabled sparse | 32 / 32 / 64 | 16 / 16 / 32 | 32 / 16 / 32 |
| identity-enabled robot | U / U / U | 16 / 16 / 16 | 16 / 16 / 16 |

This terminal check still favors correct topology over `none`, especially on robot, but by a smaller factor and without the universal eight-trajectory claim. Terminal 25% and 50% counts also move above eight in many cells, confirming that the early first-crossing grid was optimistic; those targets remain non-discriminating at their first crossing.

At a fixed budget of 128 trajectories, the 10% optimizer-threshold steps are sparse `soft4` 25/50/25 versus `none` 300/200/100, and robot `soft4` 25/50/25 versus `none` 300/U/U. These are seed triplets for one fixed target, not averages and not the three target levels.

This comparison is evidence that correct topology changes finite-budget learning behavior in this model. In the identity-free fixed-graph setting it cannot be interpreted purely as an optimization effect, because the unbiased equivariant model lacks variable identities needed to encode an arbitrary fixed asymmetric graph directly. The identity control below narrows that interpretation.

### Fixed variable-identity control

The identity-enabled 128-trajectory results closely reproduce the main comparison.

| Domain | Mode | Test one-step | Deterministic rollout | Stochastic rollout | Validation AUC | One-step effect by seed |
|---|---:|---:|---:|---:|---:|---:|
| sparse | none | 0.6261 ± 0.0085 | 0.0484 ± 0.0289 | 0.9612 ± 0.0700 | 0.6299 ± 0.0053 | reference |
| sparse | soft4 | 0.5917 ± 0.0083 | 0.0362 ± 0.0118 | 0.9473 ± 0.0620 | 0.5976 ± 0.0117 | +3.9%, +7.1%, +5.5% |
| sparse | hard | 0.5912 ± 0.0084 | 0.0355 ± 0.0107 | 0.9462 ± 0.0590 | 0.5973 ± 0.0122 | +4.0%, +7.2%, +5.6% |
| robot | none | 0.5596 ± 0.0337 | 0.0976 ± 0.0300 | 0.9406 ± 0.0515 | 0.5615 ± 0.0383 | reference |
| robot | soft4 | 0.5044 ± 0.0335 | 0.0676 ± 0.0424 | 0.8944 ± 0.0501 | 0.5124 ± 0.0305 | +12.1%, +8.3%, +9.3% |
| robot | hard | 0.5036 ± 0.0328 | 0.0672 ± 0.0385 | 0.8902 ± 0.0403 | 0.5120 ± 0.0305 | +12.1%, +8.4%, +9.6% |

For the 10% sample target, identity-enabled sparse `none` first reaches at 32@100, 16@200, and 32@100 across seeds; `soft4` and `hard` require 8@50, 8@50, and 8@25. Robot `none` is 128@300, U, and U, while both graph-informed variants are 8@50, 8@25, and 8@25. At a fixed 128 trajectories, the 10% optimizer steps are sparse `soft4` 25/50/25 versus `none` 300/200/50, and robot `soft4` 25/50/25 versus `none` 300/U/U. The 25% and 50% targets again saturate at 8@25 for all seeds and variants.

Thus the prominent early-optimization advantage is not removed merely by adding equal-capacity learned node identities. Its strict-target eight-trajectory crossing is transient, as described above. This still does not prove that topology has an irreducible advantage after unlimited optimization: the identity table makes representation possible, but the experiment stops at 600 steps.

### What the permuted controls test

The final tables compare `soft1` only with `permuted1` and `soft4` only with `permuted4`. This matched-strength comparison is essential because finite attention biases can alter scale and optimization even when their edge placement is wrong. A correct-versus-permuted difference isolates alignment of the supplied edge pattern more closely than correct-versus-none, although it remains a descriptive intervention within this architecture.

## Robustness to incomplete or spurious graphs

The realized graph quality closely follows the requested directed-edge fractions. Sparse drop recall is 0.893 ± 0.004, 0.750 ± 0.005, and 0.500 ± 0.011 at 10%, 25%, and 50%, with precision 1. Sparse add precision is 0.903 ± 0.003, 0.800 ± 0.003, and 0.667 ± 0.005, with recall 1. Mixed sparse precision and recall are both 0.750 ± 0.005. The fixed robot graph gives drop recalls 0.909, 0.727, and 0.500; add precisions 0.917, 0.786, and 0.667; and mixed precision/recall 0.727. Its zero SD reflects the same morphology, not extra certainty.

The tables report normalized one-step MSE as mean ± population SD. Parentheses contain the range of the three paired seed-level changes from that mode's clean error. Positive changes are degradations. Retrained models see the corruption throughout fitting; runtime models are clean-trained and receive it only at evaluation.

| Domain | Corruption | soft1 retrained | soft4 retrained | hard retrained |
|---|---|---:|---:|---:|
| sparse | clean | 0.6020 ± 0.0080 | 0.5940 ± 0.0081 | 0.5940 ± 0.0082 |
| sparse | drop 10% | 0.6025 ± 0.0061 (−0.0017…+0.0048) | 0.5948 ± 0.0050 (−0.0019…+0.0053) | 0.5948 ± 0.0051 (−0.0018…+0.0054) |
| sparse | drop 25% | 0.6118 ± 0.0057 (+0.0071…+0.0148) | 0.6094 ± 0.0032 (+0.0095…+0.0241) | 0.6123 ± 0.0032 (+0.0110…+0.0278) |
| sparse | drop 50% | 0.6179 ± 0.0085 (+0.0091…+0.0224) | 0.6167 ± 0.0079 (+0.0092…+0.0336) | 0.6216 ± 0.0082 (+0.0120…+0.0410) |
| sparse | add 10% | 0.6064 ± 0.0096 (+0.0012…+0.0062) | 0.5996 ± 0.0107 (+0.0022…+0.0089) | 0.5994 ± 0.0107 (+0.0023…+0.0090) |
| sparse | add 25% | 0.6097 ± 0.0087 (+0.0056…+0.0097) | 0.6047 ± 0.0101 (+0.0082…+0.0136) | 0.6045 ± 0.0101 (+0.0082…+0.0138) |
| sparse | add 50% | 0.6138 ± 0.0068 (+0.0104…+0.0133) | 0.6101 ± 0.0075 (+0.0136…+0.0175) | 0.6100 ± 0.0075 (+0.0130…+0.0176) |
| sparse | mixed 25% | 0.6170 ± 0.0053 (+0.0115…+0.0214) | 0.6158 ± 0.0050 (+0.0140…+0.0297) | 0.6172 ± 0.0054 (+0.0152…+0.0304) |
| robot | clean | 0.5177 ± 0.0349 | 0.5050 ± 0.0324 | 0.5042 ± 0.0317 |
| robot | drop 10% | 0.5230 ± 0.0345 (+0.0046…+0.0062) | 0.5129 ± 0.0325 (+0.0070…+0.0089) | 0.5126 ± 0.0321 (+0.0073…+0.0092) |
| robot | drop 25% | 0.5305 ± 0.0323 (+0.0094…+0.0158) | 0.5235 ± 0.0307 (+0.0165…+0.0214) | 0.5243 ± 0.0302 (+0.0183…+0.0225) |
| robot | drop 50% | 0.5375 ± 0.0299 (+0.0133…+0.0290) | 0.5326 ± 0.0292 (+0.0233…+0.0352) | 0.5345 ± 0.0298 (+0.0260…+0.0363) |
| robot | add 10% | 0.5210 ± 0.0338 (+0.0014…+0.0054) | 0.5103 ± 0.0318 (+0.0037…+0.0079) | 0.5095 ± 0.0313 (+0.0038…+0.0080) |
| robot | add 25% | 0.5283 ± 0.0334 (+0.0086…+0.0125) | 0.5197 ± 0.0313 (+0.0133…+0.0167) | 0.5189 ± 0.0311 (+0.0140…+0.0159) |
| robot | add 50% | 0.5322 ± 0.0313 (+0.0096…+0.0190) | 0.5252 ± 0.0283 (+0.0145…+0.0242) | 0.5246 ± 0.0279 (+0.0150…+0.0237) |
| robot | mixed 25% | 0.5404 ± 0.0339 (+0.0212…+0.0253) | 0.5372 ± 0.0326 (+0.0299…+0.0338) | 0.5377 ± 0.0325 (+0.0314…+0.0349) |

Retraining makes degradation gradual, and even the severe conditions generally remain better in absolute one-step error than the clean unbiased reference (0.6266 sparse and 0.5600 robot). It does not make corruption free: all three robot seeds degrade monotonically with severity, and sparse 25–50% changes produce clear positive effects. Soft1 is usually less sensitive in paired change but also begins from a weaker clean result.

The validation-selected sparse severe-corruption comparisons are correspondingly `soft1` 0.6179 versus hard 0.6216 for retrained drop-50%, and `soft1` 0.6170 versus hard 0.6172 for retrained mixed-25%. The latter is effectively tied at the displayed precision. The `soft4` entries remain useful fixed-strength sensitivity checks but are not the primary selected soft result in those two rows.

Runtime corruption is harsher, especially for missing edges. At 50% drop, sparse `soft4` rises to 0.6283 ± 0.0124 and `hard` to 0.6363 ± 0.0149, eliminating their clean one-step advantage over `none`; retrained counterparts are 0.6167 and 0.6216. Robot runtime 50% drop gives 0.5373 ± 0.0274 (`soft4`) and 0.5453 ± 0.0276 (`hard`), versus retrained 0.5326 and 0.5345. Mixed runtime corruption is most damaging on robot: `soft4` 0.5469 ± 0.0335 and `hard` 0.5511 ± 0.0328, with every paired seed degradation between +0.0379 and +0.0518. Soft1 is less brittle there at 0.5442 ± 0.0362, but its advantage over the clean unbiased robot result is small.

Rollouts do not simply track one-step degradation. Retrained sparse hard attention under 50% drop has worse one-step error than clean hard (0.6216 versus 0.5940) but lower stochastic rollout error (0.9253 versus 0.9465). Retrained robot hard under 50% drop similarly has stochastic rollout 0.8625 versus 0.8933 clean. Conversely, runtime corruption usually increases rollout error: robot hard mixed-25 reaches deterministic/stochastic 0.1243/0.9576 versus 0.0725/0.8933 clean. This variability is why robustness conclusions rest on the paired one-step degradation and preserve both rollout definitions as separate diagnostics.

## Transfer to unseen graphs and sizes

All transfer test graphs are disjoint from training and validation. The table reports normalized one-step MSE; parentheses are the range of the three paired percentage reductions from `none`. Negative values indicate that the variant is worse. Each seed-level effect has already averaged the eight test graphs at that size.

| Training sizes | Test nodes | none | permuted4 | soft4 | hard | graph-as-data |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 0.5720 ± 0.0050 | 0.5791 ± 0.0063 (−1.6…−0.9%) | 0.5405 ± 0.0059 (+5.2…+6.0%) | 0.5412 ± 0.0062 (+5.1…+6.0%) | 0.5414 ± 0.0067 (+4.8…+6.2%) |
| 12 | 32 | 0.5747 ± 0.0066 | 0.5780 ± 0.0061 (−0.8…−0.4%) | 0.5365 ± 0.0052 (+6.5…+6.9%) | 0.5385 ± 0.0053 (+6.1…+6.5%) | 0.5393 ± 0.0049 (+5.7…+6.4%) |
| 12 | 64 | 0.5798 ± 0.0045 | 0.5810 ± 0.0046 (−0.2…−0.2%) | 0.5390 ± 0.0048 (+6.9…+7.2%) | 0.5383 ± 0.0053 (+6.9…+7.4%) | 0.5392 ± 0.0049 (+6.9…+7.2%) |
| 12 | 128 | 0.5834 ± 0.0070 | 0.5833 ± 0.0066 (−0.1…+0.1%) | 0.5465 ± 0.0063 (+6.0…+6.5%) | 0.5372 ± 0.0070 (+7.8…+8.1%) | 0.5382 ± 0.0068 (+7.3…+8.1%) |
| 9/12/16 | 12 | 0.5736 ± 0.0069 | 0.5815 ± 0.0055 (−1.9…−1.0%) | 0.5439 ± 0.0035 (+4.5…+5.9%) | 0.5450 ± 0.0029 (+4.3…+5.9%) | 0.5445 ± 0.0036 (+4.5…+5.8%) |
| 9/12/16 | 32 | 0.5757 ± 0.0088 | 0.5802 ± 0.0075 (−1.0…−0.5%) | 0.5388 ± 0.0079 (+6.2…+6.5%) | 0.5415 ± 0.0055 (+5.6…+6.6%) | 0.5409 ± 0.0074 (+5.4…+6.5%) |
| 9/12/16 | 64 | 0.5811 ± 0.0082 | 0.5832 ± 0.0072 (−0.6…−0.1%) | 0.5409 ± 0.0072 (+6.4…+7.4%) | 0.5408 ± 0.0034 (+6.0…+7.9%) | 0.5404 ± 0.0064 (+6.7…+7.4%) |
| 9/12/16 | 128 | 0.5846 ± 0.0070 | 0.5851 ± 0.0062 (−0.2…+0.1%) | 0.5480 ± 0.0081 (+5.9…+6.9%) | 0.5403 ± 0.0021 (+6.8…+8.7%) | 0.5398 ± 0.0047 (+7.0…+8.2%) |

Correct graph information transfers: every correct-topology and graph-as-data seed beats its paired unbiased model at every size, while the matched-strength permutation is worse or effectively equal. Fixed-12 training is at least as good as mixed-size training in most cells, so the study finds no general benefit from mixing 9/12/16-node graphs. It does find transfer from 12-node training to 128-node evaluation. At 128, hard and graph-as-data retain about 7–8% paired one-step reductions under both training regimes.

Soft4 weakens relative to hard at 128 nodes: fixed-12 errors are 0.5465 versus 0.5372, and mixed-size errors are 0.5480 versus 0.5403. This pattern is consistent with a finite-bias dilution hypothesis. With equal content logits, expected degree three plus self, and bias four, graph-neighbor attention mass would be approximately \(4e^4/(4e^4+N-4)\): about 0.965 at 12 nodes but 0.638 at 128. Actual content scores are neither equal nor measured here, so this calculation is a mechanism hypothesis, not an attention measurement. Hard masks and normalized neighbor aggregation do not have that particular dilution.

Transfer rollouts are mixed. With fixed-12 training at 128 nodes, `none`, `soft4`, `hard`, and graph-as-data deterministic/stochastic means are respectively 0.0694/0.8553, 0.0322/0.8165, 0.0238/0.8074, and 0.0677/0.8480. Thus hard attention improves both rollout measures, while graph-as-data's strong one-step result does not translate into comparable recursive improvement. Mixed-size training generally has noisier and weaker rollout results; at 128, the same four pairs are 0.0819/0.8597, 0.0761/0.8546, 0.0717/0.8509, and 0.0749/0.8524.

The graph-as-data model has 17,217 parameters versus 17,089 for the attention-bias variants, owing to its history projection. Its result demonstrates that explicit neighbor aggregation is competitive for one-step transfer. It does not rank all possible graph serialization or message-passing designs.

## Heterogeneous and signed mechanisms

The signed, nonuniform family yields higher normalized one-step errors than the uniform-weight efficiency family. Because the process and its oracle/noise floor differ, this cross-family observation does not establish that one task is intrinsically harder. Results remain descriptive means ± population SD.

| Domain | Mode | Test one-step | Deterministic rollout | Stochastic rollout | Information supplied |
|---|---|---:|---:|---:|---|
| sparse | none | 0.7270 ± 0.0271 | 0.0413 ± 0.0089 | 0.9731 ± 0.0220 | none |
| sparse | permuted4 | 0.7277 ± 0.0271 | 0.0458 ± 0.0186 | 0.9741 ± 0.0098 | misaligned adjacency |
| sparse | soft1 | 0.7208 ± 0.0276 | 0.0348 ± 0.0045 | 0.9657 ± 0.0263 | adjacency |
| sparse | soft4 | 0.7149 ± 0.0265 | 0.0328 ± 0.0101 | 0.9610 ± 0.0158 | adjacency |
| sparse | hard | 0.7148 ± 0.0262 | 0.0334 ± 0.0118 | 0.9618 ± 0.0141 | adjacency |
| sparse | graph-as-data | 0.7217 ± 0.0259 | 0.0447 ± 0.0165 | 0.9747 ± 0.0141 | adjacency |
| sparse | typed | 0.7026 ± 0.0185 | 0.0281 ± 0.0097 | 0.9469 ± 0.0121 | signed magnitudes |
| robot | none | 0.6338 ± 0.0574 | 0.0532 ± 0.0056 | 0.9123 ± 0.0031 | none |
| robot | permuted4 | 0.6335 ± 0.0579 | 0.0521 ± 0.0071 | 0.9058 ± 0.0163 | misaligned adjacency |
| robot | soft1 | 0.6216 ± 0.0562 | 0.0514 ± 0.0048 | 0.9118 ± 0.0015 | adjacency |
| robot | soft4 | 0.6106 ± 0.0586 | 0.0456 ± 0.0093 | 0.9070 ± 0.0043 | adjacency |
| robot | hard | 0.6102 ± 0.0587 | 0.0455 ± 0.0101 | 0.9094 ± 0.0078 | adjacency |
| robot | graph-as-data | 0.6207 ± 0.0578 | 0.0508 ± 0.0115 | 0.9055 ± 0.0193 | adjacency |
| robot | typed | 0.5738 ± 0.0446 | 0.0316 ± 0.0109 | 0.9042 ± 0.0254 | signed magnitudes |

Correct adjacency alone gives modest one-step reductions: about 1.7% for sparse `soft4` and 3.7% for robot `soft4` relative to `none`, while `permuted4` gives none. Typed signed biases reduce one-step error by about 3.3% sparse and 9.5% robot relative to `none`, and improve deterministic rollout in both domains. This is the clearest benefit in the suite, but it is also privileged: typed bias receives the true coefficient signs and magnitudes. The incremental typed-over-adjacency result measures the value of that extra information. It does not show that topology identifies arbitrary weights. Stochastic rollout changes are smaller, especially on robot, where typed 0.9042 is close to several adjacency-only conditions.

## Learned layer/head strengths

Learning eight unconstrained coefficients from zero produces an intermediate one-step result and a severe rollout failure.

| Domain | Mode | Test one-step | Deterministic rollout | Stochastic rollout |
|---|---|---:|---:|---:|
| sparse | none | 0.6266 ± 0.0086 | 0.0474 ± 0.0257 | 0.9583 ± 0.0642 |
| sparse | learned | 0.6107 ± 0.0111 | 0.1744 ± 0.1746 | 1.0760 ± 0.2517 |
| sparse | soft4 | 0.5940 ± 0.0081 | 0.0379 ± 0.0091 | 0.9484 ± 0.0579 |
| sparse | hard | 0.5940 ± 0.0082 | 0.0378 ± 0.0048 | 0.9465 ± 0.0516 |
| robot | none | 0.5600 ± 0.0332 | 0.0998 ± 0.0349 | 0.9430 ± 0.0576 |
| robot | learned | 0.5281 ± 0.0390 | 0.3195 ± 0.2739 | 1.1157 ± 0.2532 |
| robot | soft4 | 0.5050 ± 0.0324 | 0.0747 ± 0.0506 | 0.9007 ± 0.0633 |
| robot | hard | 0.5042 ± 0.0317 | 0.0725 ± 0.0440 | 0.8933 ± 0.0492 |

The learned model improves one-step error over `none` by 0.0159 sparse and 0.0319 robot, but remains worse than fixed `soft4` and `hard`. More consequentially, its deterministic rollout error is 3.7 times the sparse `none` error and 3.2 times the robot `none` error; stochastic rollout is also worse. This is a strong negative result: validation of one-step prediction did not protect recursive stability.

The final alpha matrices (rows are layers, columns are heads) are:

| Domain, seed | Layer 1 | Layer 2 |
|---|---|---|
| sparse, 0 | [0.750, 0.358, −0.071, 0.572] | [0.329, 0.522, 0.328, 0.170] |
| sparse, 1 | [0.389, −0.568, 0.497, 0.800] | [0.528, 0.469, 0.363, 0.606] |
| sparse, 2 | [−0.003, 0.577, 0.640, 0.710] | [0.563, 0.626, 0.635, 0.415] |
| robot, 0 | [0.810, 0.841, −0.293, 0.495] | [0.318, 0.528, 0.307, −0.232] |
| robot, 1 | [0.312, −0.263, 0.225, 0.838] | [0.493, 0.412, −0.572, 0.513] |
| robot, 2 | [−0.357, 0.606, 0.732, 0.799] | [−0.102, 0.678, 0.647, 0.133] |

Coefficients span approximately −0.57 to +0.84. Some heads become strongly positive, while a different subset is negative in each seed. There is no stable head identity, uniform sign, or simple early-layer/late-layer ordering across domains and seeds. Checkpoints show small mixed signs at step 25 and generally increasing magnitudes thereafter, but several trajectories change direction late—for example sparse seed 0 layer-1 head 3 returns negative, and robot seed 2 develops new negative heads by step 600. The measurements support differentiation among heads, but not a reproducible specialization map or a “crystallization” claim.

## Interpretation and milestone decision

The milestone for moving toward pretrained or Qwen experiments is intentionally demanding: graph information should improve more than one favorable fixed-graph metric. The complete result must show a coherent advantage in finite-data or finite-step learning, retain useful behavior under realistic graph errors, and transfer to disjoint graph identities and substantially larger node counts. Signed typed information and learned strengths are mechanistic probes, not substitutes for transfer and robustness.

The completed study partially meets the empirical milestone. Correct graph information advances strict threshold crossings, survives moderate retrained corruption, and improves one-step prediction on unseen graphs through 128 nodes in every seed. Runtime graph transfer is real, and matched permutation does not reproduce it. Hard masks and graph-as-data, however, are as strong as or stronger than soft bias in important transfer cells; the smallest-sample threshold result is transient; and topology does not win every rollout metric.

Large Qwen or VLA work should remain deferred. Severe runtime corruption can erase the advantage; mixed-size training adds no clear benefit; fixed soft bias weakens at 128 nodes in a pattern consistent with the dilution hypothesis; graph-as-data does not consistently improve rollout; learned alpha is recursively unstable; and privileged signed coefficients cannot be conflated with adjacency. The evidence can justify a small, predeclared lookup-style pretrained probe whose purpose is to test whether the finite-optimization effect survives a stronger representation. Such a probe should include matched permutation and graph-as-data controls, measure recursive behavior where applicable, and test runtime graph errors. The present result supports that bounded next measurement, not a general scale-up or embodied-learning claim.

## Reproducibility and limitations

The full run uses one CPU process with Torch, OMP, and OpenBLAS limited to two threads. The exact-source project test gate passed 126 tests in 1.55 seconds before execution. The cooperative 7200-second suite deadline is checked between operations and an external 7500-second timeout bounds the process. Incremental JSONL preserves completed rows if execution is interrupted.

The final exact-code gate, including report commit `ba11375`, passed 143 tests in 1.96 seconds. The study is small by design. Three seeds support paired descriptive comparisons but not formal significance claims. Sparse seeds combine variation in topology, data, and initialization; robot seeds are repeated training draws on one fixed morphology. Dense attention means latency does not demonstrate a computational advantage from sparse structure. Synthetic dynamics omit contacts, actuation, perception, language, morphology variation, and distribution shifts found in embodied learning. Model weights are not retained as reloadable checkpoints, although deterministic seeds and comprehensive hashes support replay. Finally, the tested grids—six sample counts, seven optimizer checkpoints, fixed widths and depths, and a few graph encodings—bound every conclusion.

## Artifacts

The immutable raw artifacts are [summary.json](../../results/stage2/summary.json) and [metrics.jsonl](../../results/stage2/metrics.jsonl); [audit.json](../../results/stage2/audit.json) records the independent integrity checks. The source-bound derived analysis is [analysis/summary.json](../../results/stage2/analysis/summary.json) with a compact generated [analysis report](../../results/stage2/analysis/report.md). Training used source `ee49edc`; the final analyzer and reporting revision is `6d404bc`.

Primary figures:

- Efficiency: [sparse](../../results/stage2/analysis/efficiency-sparse-efficiency.svg), [robot](../../results/stage2/analysis/efficiency-robot-efficiency.svg), [identity-enabled sparse](../../results/stage2/analysis/efficiency_identity-sparse-efficiency.svg), and [identity-enabled robot](../../results/stage2/analysis/efficiency_identity-robot-efficiency.svg).
- Corruption one-step robustness: [sparse](../../results/stage2/analysis/corruption-sparse-one_step.svg) and [robot](../../results/stage2/analysis/corruption-robot-one_step.svg).
- Transfer: [one-step](../../results/stage2/analysis/transfer-sparse-one_step.svg), [deterministic rollout](../../results/stage2/analysis/transfer-sparse-deterministic_rollout.svg), and [stochastic rollout](../../results/stage2/analysis/transfer-sparse-stochastic_rollout.svg).
- Signed information: [sparse one-step](../../results/stage2/analysis/heterogeneous-sparse-one_step.svg) and [robot one-step](../../results/stage2/analysis/heterogeneous-robot-one_step.svg).
- Learned strength: [sparse deterministic rollout](../../results/stage2/analysis/learned-sparse-deterministic_rollout.svg) and [robot deterministic rollout](../../results/stage2/analysis/learned-robot-deterministic_rollout.svg).
