# S15: shape exposure improves components but fails complete acquisition

All three preregistered criteria fail at the fixed4096-update endpoint. Mixed training produces1/512 complete graphs on fresh examples of its new trained motif,0/512 on the held-out combination, and90/1024 on the known arity-four motifs versus276/1024 for control. Neither confirmation nor the one symmetric extension is eligible. No early checkpoint or policy is substituted.

| Global arity × facts | Interpretation | Control raw / calibrated | Mixed raw / calibrated | Mixed exact-copy graphs | Mixed mean copy | Mixed calibrated ordered F1 |
|---|---|---:|---:|---:|---:|---:|
|3 ×3|New trained motif acquisition|0 /0|1 /1|477|.9951|.9273|
|3 ×4|Untrained combination|0 /0|0 /0|0|.8128|.3019|
|4 ×3|Known motif retention|109 /231|7 /80|500|.9989|.9900|
|4 ×4|Known motif retention|3 /45|0 /10|453|.9951|.9749|

Each row has512DEV examples. Acquisition required at least52complete and a gain26; observed1and1. Retention required at least103/1024 and loss no worse than51; observed90 and−186. Recombination required52complete and gain26; observed0and0. The new motif's calibrated curve is0→0→0→1 at added0/1024/2048/4096updates. Mixed known-motif calibrated totals236→28→72→90 versus control236→228→252→276. This is partial recovery following interference, not a successful retention result.

The earliest demonstrated boundary is complete-graph acquisition on fresh constructions of the newly exposed motif, before a clean test of acquired-shape recombination. Strong copy and ordered-edge components on3×3 show substantial learning;1complete graph does not meet the registered competence criterion. These DEV results do not establish whether individual TRAIN graphs were memorized: calibration remains the historical English TRAIN128 and there is no new-shape full-graph TRAIN panel in this protocol. The untrained3×4cell still fails, but cannot isolate recombination independently of weak complete acquisition and retention.

Exact paired calibrated gains are+1,0,−151,−35graphs for3×3,3×4,4×3,4×4. Descriptive marginal paired-event95% bootstrap intervals are[0,.58594],[0,0],[−33.98438,−25],[−9.375,−4.29688]percentage points. The zero empirical interval in the all-failed cell is not evidence of a zero population uncertainty bound. These intervals condition on one fixed training pair and do not estimate training-seed uncertainty or change the decision rules. All raw/calibrated transitions, component precision/recall/F1 and prescribed curves are in `s15-paired-analysis.json`.

Both arms use the independently selected original S11 model+AdamW, identical4096-construction index schedule and32768presentations,8visits per graph, and16384common presentations with exact shared negative-pair stream. The new arm-independent presentation RNG prevents unlike shapes from perturbing later common-example sampling. Each endpoint uses one historical English TRAIN128 calibration policy across all four cells. The guarded analysis verifies frozen configs, source/cache/audit/calibration/parent provenance, full visits, finite logged losses and durable receipts; the parent checkpoint bytes are independently audited remotely. Confirmation remains untouched. The experiment is S14-informed, has2versus3training motifs and one untrained combination, all depth3; it is not broad structural transfer.

Control/mixed fulloccupancies486.056154539/484.141407571seconds, both exit0 within1100caps; optimizer248.4338/248.5729seconds. Token exposures1,933,312versus1,720,320; nodes1,105,664versus988,008; edges2,531,072versus2,208,616. Construction exposure is matched; tokens, graph sizes and arithmetic are not. The prepared conditional S09 contextual-read path is a possible new attributable intervention, not an automatic extension or a result established by this experiment.
