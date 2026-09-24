# S18: contextual reading yields partial gains but fails acquisition and retention

The fixed4096 endpoint fails all three registered criteria. Contextual reading improves trained3×3 calibrated complete graphs over both controls, but reaches43/512, below52, and its21-graph advantage over the original falls below26. Known-arity retention is84/1024, below103 (original119,10pass7); the35-graph loss versus original satisfies the loss≤51 condition but cannot repair the failed absolute criterion. All arms remain0/512 on heldout3×4. No extension, confirmation, checkpoint selection or retrospective S15/S17 promotion follows.

The pinned analysis7bc2b1fc passed all config/source/receipt/artifact/calibration/stream/event/target/policy guards on complete main62f5616d and reference4616963f. It reuses the immutable original S15 trajectory and S17 matched endpoint; early matched baseline curves come from the separately audited reference job. Main model/AdamW states and raw numerical outcomes are undergoing independent review; this report does not substitute for that audit.

| Endpoint,512 per cell | Policy | 3×3 | 3×4 | 4×3 | 4×4 |
|---|---|---:|---:|---:|---:|
| Original2pass | Raw |1|0|7|0|
| Original2pass | Historical TRAIN |1|0|80|10|
| Original2pass | Matched TRAIN |22|0|109|10|
| Contextual read | Raw |0|0|0|0|
| Contextual read | Historical TRAIN |7|0|79|15|
| Contextual read | Matched TRAIN |43|0|68|16|
| Workspace10pass | Raw |0|0|0|0|
| Workspace10pass | Historical TRAIN |0|0|2|0|
| Workspace10pass | Matched TRAIN |1|0|7|0|

Matched-policy curves, shown as3×3/3×4/4×3/4×4, retain the immediate architectural perturbation at checkpoint0:

| Added updates | Original | Context |10pass |
|---:|---|---|---|
|0|0/0/47/3|0/0/1/0|0/0/0/0|
|1024|0/0/35/6|0/0/29/4|0/0/0/0|
|2048|5/0/64/12|3/0/54/3|0/0/1/0|
|4096|22/0/109/10|43/0/68/16|1/0/7/0|

On trained3×3, context gains35and loses14original successes (8both-correct,455both-wrong), net+21/+4.10percentage points; the conditional event-bootstrap95% interval is[1.56,6.64]points. Against10pass it gains42and loses0 (1both-correct), net+8.20points,[5.86,10.55]. On4×3 it loses77and gains36against original, net−41/−8.01points,[−12.11,−4.10]. These descriptive intervals share event draws across policies/comparators and condition on one inspected parent lineage; they are not seed uncertainty and do not override the fixed gates.

## Acquisition versus calibration

The actualTRAIN128 panel is the same data used to fit thresholds, making its calibrated results optimistic in-sample diagnostics:

| Arm |3×3raw/cal of64|4×3raw/cal of32|4×4raw/cal of32|Total raw/cal of128|
|---|---:|---:|---:|---:|
| Original |1/5|1/7|0/2|2/14|
| Context |0/6|0/3|0/1|0/10|
|10pass |0/0|0/1|0/0|0/1|

Context still fits only6/64TRAIN3×3graphs despite exact presence64, kinds63, scalar values64, copying64 and slots64; exact edges are6. On DEV3×3, exact edges rise22→44and copy477→485relative to original, while kinds fall478→472and slots507→505. Context reduces reference-edge false negatives965→700but increases false positives468→501; argument errors worsen slightly. This is partial relation learning, not simply an unchanged model with a different cutoff, but raw complete graphs remain zero and the global calibration policy strongly affects reported exactness. The stronger inference of acquired graph construction is unsupported even on the optimistic TRAIN panel.

Heldout3×4is broader failure: context has exact presence505, kinds0, copying1, edges0and slots0. No recombination claim follows from trained-shape improvements. The10pass arm also remains poorly fitted; it changes recurrence depth on a2pass-trained parent, with a large checkpoint0 perturbation. Its failure rejects this declared compute allocation, not every stronger compute baseline or a general role for recurrence.

## Costs and next branch

Main outer1940.07seconds, conservative charge1940.08, exit0/GPUFREE. Context optimizer335.166seconds, calibration/TRAIN26.813, DEV265.864, export100.535;10pass optimizer494.294, calibration/TRAIN36.170, DEV403.050, export104.542. Each arm includes all four checkpoints and both policies; complete phase/serialization/receipt data are archived. Equal parameters did not mean equal time:10pass optimizer work cost about1.47times context. The main stayed far below4000hard cap; cap arithmetic must not be treated as a replication lower bound.

New S18 charged work totals2245.64seconds: profile86.53, failed reference63.93(retained), corrected reference155.10, main1940.08. Reused original training/evaluation484.141407571seconds and earlier S17 calibration work remain historical costs, not newly charged or free acquisition. Stage-matched independent parents701/702/703exist, so seed replication could be budgeted from actual measured costs; it is not ruled out by resource arithmetic. The scientific acquisition failure, rather than a presumed inability to afford it, is the reason not to spend confirmation/replication budget here.

Follow the prospectively stated poor-TRAIN branch in `S18-followup-decisions.md`: retain this negative result, perform a CPU-only archived TRAIN relation-score/ranking diagnostic if needed to specify the next hypothesis, and stop calibration/exposure tweaks to this workspace family. The smallest stronger public-text alternative worth separate preparation is a compact encoder plus autoregressive variable-length graph decoder with learned copying, testing whether predicting node creation/relations sequentially avoids fixed canonical query allocation. It is a different output/optimization interface, not a pure architectural causal claim or repetition of S06's token-state replacement under the same canonical decoder. Require public-only inference, no supplied parse/graph/alignment/arity, explicit invalid-output failures, meaningful TRAIN-versus-fresh-DEV acquisition, and a new source/profile/budget freeze. No such experiment, confirmation population or successor has been run or selected here.
