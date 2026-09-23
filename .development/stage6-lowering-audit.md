# Stage 6 independent lowering and execution audit

**Passed artifact integrity; autonomous semantic binding failed at this budget.** The standalone [audit script](audit-stage6-lowering.py) loads the generator and evaluation functions from frozen git source `c258785` without importing Torch or executing a model. Its [diagnostic JSON](stage6-lowering-independent-audit.json) is outside the canonical result archive. All three completed shards were audited through the merged manifest/raw gzip.

Coverage is exact: 780 evaluation rows, 11,520 training rows and 9,360 evaluation examples. Every training-data digest and every evaluation public/data digest matched frozen regeneration. All expected cells were present, with no duplicates. Final diagnostics contain 185 arm/condition/depth/intervention groups, each pooled across three seeds. Null rates retain zero denominators.

## What the metrics mean

A recognized ID belongs to a gold task transition; supplied dead destinations and unknown destinations are counted separately. IDs are selected from supplied operation slots, so high recognition is not evidence of learned semantic identity discovery. Primitive and binding accuracy are conditional on recognized identity. Binding means exact **ordered** argument IDs, independently of primitive choice. Full correctness requires identity, primitive and ordered arguments together.

Proposal counts include all logged predictions, including the final pass where no execution is attempted. Null and unavailable operands are explicit counts. Refused/deferred/conflicting/duplicate events are separate populations, and a valid refusal is never silently counted as wrong executed binding.

Committed arithmetic is independently replayed using public initial literals plus **actual event values**, with snapshot reads within each step. No gold intermediate values substitute into runtime memory. This tests whether an operation correctly computed its actual operands, distinct from whether those operands described the task. Complete transition-set equality ignores independent execution order; the stricter value-complete conditional additionally requires all committed values/types to match gold.

## Final primary findings

These pooled counts cover 15 arms × 11 final conditions × 36 seed/examples = 5,940 condition-examples. Conditions reuse some constructions, so these are audit populations, not independent statistical samples.

- 84,744 logged proposals: 78,559 recognized task IDs, 6,185 supplied dead IDs, no unknown IDs; 60,614 included null operands and 61,812 included unavailable operands (overlapping categories).
- Among recognized-ID proposals, 15,793 had the correct primitive and 24 had the correct ordered binding. Only 24/84,744 proposals were fully correct; this does not imply they executed.
- 1,215 transitions committed: 1,083 under fixed-compute and 132 under warm task-only. All 1,215/1,215 computed the correct exact result for their **actual valid operands**.
- Of those executions, 1,111 named recognized task destinations, and **0/1,111 had the correct ordered binding**. The remaining 104 targeted supplied dead destinations. No final primary example completed the correct transition set; task accuracy conditional on a complete correct trajectory is therefore **null**, not zero.

For the local arm at depths 4/8/16/32, recognized IDs were 132/144, 128/144, 142/144 and 144/144 logged proposals. Correct ordered binding was respectively 0/132, 0/128, 0/142 and 0/144. No transitions committed, so the local learned-vs-exact runtime-value comparison has no autonomous execution support.

## Privileged execution exposes a separate readout failure

Both oracle diagnostics supplied every correct transition and produced exact values. At depth 4 each diagnostic committed 324/324 correct transitions; at depth 32, 2,340/2,340. All 36 examples at each depth had complete correct trajectories. With minimal supplied unroll, joint task readout was only **1/36 at depth 4 and 1/36 at depth 32**. With the 80-step supplied unroll it was **0/36 at both depths**. These are privileged diagnostic results, not learned execution; they show that exact protected values do not guarantee correct neural readout.

The JSON includes six explicit public inputs with complete raw proposal/event traces: neural fixed, neural recurrent, protected learned, local, fixed compute and warm task-only. The first four illustrate absent execution/early output; the latter two include formally valid exact arithmetic over task-incorrect bindings. They are designated illustrative failures, not randomly sampled estimates.

No training or model inference was run for this audit. Completed main source and canonical archives remain unchanged.
