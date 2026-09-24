# A14 fixed three-seed read-policy confirmation

A shared hard destination read passes the preregistered task thresholds in all three fresh seeds. The unchanged learned soft policy fails. This confirms a supplied read-policy intervention in the all-record model; it does not establish structural-attention superiority. All nine competitive A06 engineering references attain 100% task and complete suffix accuracy on the same fresh graphs.

## Frozen design and evidence

Source `67dc519`, protocol `a14-protocol.md`, and guarded frozen analysis `research/tools/campaign_a14_analysis.py` define the experiment. Seeds 1401/1402/1403 each received exactly 6000 updates and 96000 graph presentations. There was no endpoint or policy selection. Each condition contains the same 1024 events across seeds and policies; complete public-input and record-order hashes match. Conditions are IID N32/D4/K4, moderate N64/D8/K4, and joint N128/D32/K8. Analysis and input hashes are archived in `research/results/campaign-01/attention/a14/confirmation-analysis.json`.

The original objective is payload cross-entropy over batch × reverse step × node, trained at depths 1–4. It has **no routing supervision**, neither mean-head nor per-head. Shared-soft uses the mean of original destination probabilities for every value head; shared-hard uses the argmax of that same mean. Both preserve the original soft record stage. Both-hard separately hardens each head's record and destination choices. Oracle-common uses privileged correct successors and is only a routing-intervention ceiling. No inference policy changes model parameters.

## Fixed endpoint results

Every entry is correct tasks out of 1024. Columns within each condition are seeds 1401 / 1402 / 1403.

| Policy | IID | Moderate | Joint |
|---|---|---|---|
| Original | 1017 / 1016 / 1018 | 951 / 921 / 953 | 176 / 187 / 177 |
| Both hard | 1013 / 1010 / 1012 | 987 / 975 / 990 | 257 / 221 / 245 |
| Shared soft | 1017 / 1016 / 1016 | 880 / 799 / 849 | 177 / 183 / 179 |
| Shared hard | 1024 / 1024 / 1024 | 1022 / 1024 / 1023 | 1019 / 1007 / 1015 |
| Privileged oracle | 1024 / 1024 / 1024 | 1024 / 1024 / 1024 | 1024 / 1024 / 1024 |

Shared-hard joint gains over original are +82.324, +80.078, and +81.836 percentage points. Their mean is **+81.413 points**, with the preregistered shared-event bootstrap 95% interval **[79.492, 83.236] points**. All three gains are positive and the interval's lower bound exceeds zero: the replicated-effect rule passes. The 10000 bootstrap draws use the same event indices across seeds and policies; this is conditional uncertainty over the 1024 shared graphs, not 3072 independent graphs or a population-of-training-seeds interval. The three gains separately disclose seed variation.

Only shared-hard passes every seed's separate competence thresholds (98% IID, 95% moderate, 98% joint). Its mean joint accuracy is 3041/3072 = 98.991%, versus original 540/3072 = 17.578%. This is an engineering-policy pass, **not an acquired-soft gate pass**. A13's earlier 499/512 = 97.46% development result and every previous failed endpoint remain unchanged.

## Paired route and payload evidence

| Joint count | Seed 1401 | Seed 1402 | Seed 1403 |
|---|---:|---:|---:|
| Complete queried mean route correct /1024 | 1017 | 1006 | 1015 |
| Original task AND route correct | 175 | 183 | 176 |
| Shared-hard task AND route correct | 1017 | 1006 | 1015 |
| Original wrong → shared-hard right | 843 | 823 | 839 |
| Original right → shared-hard wrong | 0 | 3 | 1 |
| Shared-hard complete all-node suffix /1024 | 964 | 960 | 955 |
| Original complete all-node suffix /1024 | 0 | 0 | 0 |

P(task correct | complete queried mean route correct) is 1017/1017, 1006/1006, and 1015/1015 for shared-hard, versus 175/1017, 183/1006, and 176/1015 for original. The mechanistic conditional rule passes each seed. Pooled shared-hard support is 3038/3038, with empirical bootstrap interval [1,1] and no undefined draws. This zero-observed-failure interval is not a universal reliability guarantee. Shared-hard additionally answers 2/1/0 route-wrong events correctly; class correctness and exact route identity are distinct.

Original, shared-soft, and shared-hard have exactly identical all-node mean-route arrays in every seed and condition. The intervention changes payload transport while leaving that route statistic unchanged. On the queried oracle trajectory, original destination-head complete-path correctness is 454/8192, 471/8192, and 471/8192 head-event pairs in the joint condition, with zero events having all eight heads correct throughout. Heads are correlated; 8192 is a diagnostic support, not independent samples. Compare these full-path head diagnostics with the 1024-event mean-path supports above, not with all-node average head accuracy.

Common soft weights alone do not repair deep task behavior. Independent per-head hardening also remains weak. The paired interventions support common discrete address selection as an effective repair of this model's read interface under the frozen recipe. They do not prove a generic memory failure or a uniquely sufficient learned mechanism. Shared-hard's complete all-node suffix remains below 100%, and its task gate must not be substituted for an all-node suffix gate.

## Strong engineering references and compute disclosure

Each of soft, gather, and keyed-context references at A06 seeds 601/602/603 scores **1024/1024 task and 1024/1024 complete suffix in all three conditions**: all 27 cells. They use the same fresh public graphs, with exact checkpoint identities and public hashes checked. These are competitive supplied-neighborhood/keyed-context engineering references, with different training histories and frozen scale interventions; they are not unconstrained all-record or ordinary graph-text baselines. Their success precludes a unique structural-attention advantage claim here.

| Measurement | A14 all-record | A06 engineering references |
|---|---|---|
| Allocated parameters | 4,575,275 | 4,306,986 each |
| Gradient-participating parameters | 2,412,562 | soft 4,298,793; gather 4,298,769; context 4,306,962 |
| Training exposure per model | 6000 updates / 96000 graphs | 1000 updates / 16000 graphs |
| Training + monitoring seconds, A14 seeds | 83.278 / 69.729 / 64.756 | Historical A06 training; no retraining here |
| Of which development monitoring seconds | 1.713 / 1.839 / 1.779 | Not measured by this confirmation |
| Training + monitoring CUDA peak bytes | 508,828,160 each | Not measured by this confirmation |
| Instrumented inference CUDA peak bytes | 1,516,548,608 each | Different bare-forward scope below |

The three models' instrumented joint original forwards cost 47.567 / 47.686 / 47.406 seconds per 1024 events; shared-hard costs 47.709 / 47.741 / 47.527 seconds. Instrumentation includes full diagnostic tensors. Bare reference forwards, batch 16, average 3.154 seconds soft, 3.239 gather, and 3.436 context per 1024 joint events. Reference bare inference peaks are 232,671,744 / 225,004,032 / 234,244,608 bytes respectively. **These scopes differ; their quotient is not a normalized speed comparison.**

The separate profile measured bare all-record batch-16 largest-condition forwards on a fresh 100-update model: original 0.483–0.486 seconds, shared-hard 0.490–0.498, both-hard 0.636–0.639, three repeats each. Charged warmups were 0.511 / 0.493 / 0.655 seconds respectively; shared preparation including generation/tokenization/oracle preparation was 0.013882 seconds. Original/shared-hard bare peaks were 1,042,956,288 / 1,044,004,864 bytes. These are descriptive implementation measurements with different training history, not a controlled efficiency result. A compute-normalized learning claim still lacks matched training exposure/input interfaces, uninstrumented 6000-update-model timing, and an equal-budget acquisition comparison.

Atomic full-process occupancies are profile 22.198607951, seeds 355.515602229 / 341.759054285 / 336.333546519, references 51.444005635 seconds: **1107.250816619 seconds total**. Receipts and process logs are archived under `research/results/campaign-01/attention/a14`; checkpoint/state bytes remain durably remote with archived SHA inventories. Attention campaign accounting is 2750.981686829 seconds, including the older A08 profile's uncertain 60-second upper bound rather than measured occupancy. CPU validation/inference is separate. All A14 GPU work is complete; no follow-up run is launched.

## Closure and next question

Close this explicit-address read-localization branch: the fixed three-seed intervention is confirmed and competitive supplied-context references already solve the benchmark. A useful next question would replace supplied observation-to-node/edge bindings with learned grounding, evaluated against equal-observation baselines and privileged binding controls. Merely learning another version of the already explicit address mapping would repeat this benchmark. Such a branch needs its own prospective attribution test; none is implemented or authorized here. A shared hard read policy is also distinct from a graph-induced QK bias, which this confirmation does not isolate.

Independent aggregate audit is requested; the preregistered analysis loader has passed all frozen-contract and pairing checks.
