# A13: shared exact addresses turn correct mean routes into execution

On this frozen checkpoint and paired fresh population, a common hard destination chosen from the original mean probabilities solves every event whose queried mean route is fully correct. Sharing the soft probability distribution alone does not repair deep execution and harms moderate-depth accuracy. This localizes a failure of the supplied soft transport interface despite largely correct aggregate addresses. It is a frozen kernel intervention, not learned address sharing, retraining, confirmation or an acquired-soft gate pass.

| Policy | IID task | Moderate task | Joint task | Joint complete all-node suffix |
|---|---:|---:|---:|---:|
| Unchanged | 505/512 (98.63%) | 479/512 (93.55%) | 86/512 (16.80%) | 0/512 |
| Shared mean probabilities | 505/512 (98.63%) | 417/512 (81.45%) | 89/512 (17.38%) | 0/512 |
| Shared argmax of mean probabilities | 511/512 (99.80%) | 512/512 (100%) | 499/512 (97.46%) | 471/512 (91.99%) |
| Privileged oracle common successor | 512/512 (100%) | 512/512 (100%) | 512/512 (100%) | 512/512 (100%) |

All three nonoracle policies have **exactly identical complete all-node mean-route arrays** at every shape. Their queried full-path correctness is511/512 IID,511/512 moderate and499/512 joint. This is equality of actual recorded routes, not merely equal marginal rates. The original soft record stage and residual/readout remain unchanged; shared soft and shared hard use the same original mean destination probabilities, never mean logits.

## Paired conditional evidence

The joint event contingency tables are:

| Policy | Route correct, task correct | Route correct, task wrong | Route wrong, task correct | Route wrong, task wrong |
|---|---:|---:|---:|---:|
| Unchanged | 84 | 415 | 2 | 11 |
| Shared soft | 87 | 412 | 2 | 11 |
| Shared hard | 499 | 0 | 0 | 13 |
| Privileged oracle | 512 | 0 | 0 | 0 |

Thus P(task correct | full queried mean route correct) is84/499 (16.83%) unchanged,87/499 (17.43%) shared soft and499/499 (100%) shared hard. These are observed joint counts, not marginal products. Shared hard also has100% conditional task accuracy on IID and moderate events. A wrong route can still produce the correct class: one moderate event is route-wrong/task-correct under shared hard, so final payload correctness must remain distinct from route identity.

Shared hard fixes6/33/415 prior errors and breaks0/0/2 correct answers on IID/moderate/joint. Shared soft fixes4/1/4 and breaks4/63/1. Retain the substantial moderate degradation under shared soft, not only its small joint gain. Complete suffix requires every node and reverse-step value to be correct, a stronger and different support from the queried path. Accordingly joint shared-hard complete suffix is471/512, or471/499 among query-route-correct events; other nodes may still have route errors.

## Head correctness on the same queried trajectories

The original destination heads have2580/4096 correct full head paths on IID,2048/4096 moderate and244/4096 joint. Each denominator is512 events×eight heads, with conjunction over the same queried path length4/8/32. These correlated head paths are not4096 independent graphs. The corresponding mean-route event denominators are512, with511/511/499 successes. On joint events, zero of512 have every original head's full path correct; no event has original destination heads unanimous at every queried step. Among the499 mean-route-correct events,241/3992 original head paths are correct.

Shared policies make addresses identical across heads by construction. Shared soft still mixes probabilities over nodes, whereas shared hard selects one full payload vector from the common model-selected node. Their large outcome difference, with identical aggregate routes, shows that common address argmax accuracy alone is insufficient under soft mixing. This intervention does not establish that every original head was intended as an individual symbolic pointer, nor does it retroactively explain every A12 all-node statistic by a queried-path error.

The privileged oracle common-successor intervention reaches100% task and complete all-node suffix at all shapes. The unchanged learned update/readout therefore supports these tested executions when given exact common routing. The oracle result is an intervention reference, not a learned capability, Bayes ceiling or universal guarantee. Shared-hard residual joint failures align with the13 observed wrong queried mean routes; this single development seed is not three-seed confirmation.

## Integrity and cost

Frozen source53bdfed, main config SHA256 `ec442effb25da9871da8e3932f3c079d40e974f94b66f9cd6feaf99a05be6f44`; A11 checkpoint SHA256 `f78aa380ddb2bb7e18234f73c5de751f5ebea2d457211ccc35591c62dd48609f`. Initial/final tensor hashes match; zero updates. Nonoracle label-isolation, exact unchanged parity and post-forward query-support contracts passed independent preflight. Main data241M/order242M is fresh and paired across all four policies; older A11/A12 endpoints remain unchanged.

Atomic full-process receipt105.608966729seconds; internal104.220010393seconds; CUDA peak1,082,438,144bytes. Timing includes instrumentation. GPU verified free before analysis. Cumulative attention accounting1643.730870210seconds includes A08 profile's explicitly uncertain60-second upper bound. No subsequent GPU job or policy is launched by this result.

Raw archive1f27021: `research/results/campaign-01/attention/a13-main/results`; paired analysis and all input hashes: `a13-main/paired-analysis.json`, generated by the prospectively prepared `research/tools/campaign_a13_analysis.py`. Independent twelve-cell main audit requested.
