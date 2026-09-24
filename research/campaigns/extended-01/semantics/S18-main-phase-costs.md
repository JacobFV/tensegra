# S18 completed main: phase costs before outcome analysis

Both new arms completed4096updates/all four checkpoints/both policies, exit0 with no timeout or extension. Full outer1940.07seconds, conservative charge1940.08; wrapper1940.046164, child1936.785452, source/data preflight2.677933. GPU was free before archive/analysis. Peak allocated CUDA1,467,351,552bytes; max RSS6,222,460KiB. Main raw artifacts are archived under `research/results/campaign-01/semantics/s18-main-v2/s18-main-v1`; eight model+AdamW checkpoints remain read-only remotely with exact paths/bytes/hashes in `retained-checkpoints.json`. All example/query sequence hashes and eight-visit exposure checks pass. Each arm saw1,720,320tokens/988,008nodes/2,208,616edges.

| Phase seconds | Context |10pass control |
|---|---:|---:|
|Optimizer4096updates|335.166|494.294|
|Matched policy, all4curves, complete evaluation/export|207.937|268.027|
|Historical policy, all4curves, complete evaluation/export|186.633|277.281|
|Four checkpoint writes/hashes|4.157|5.951|
|Model/AdamW setup|0.947|1.125|

`phase-costs.json` provides every calibration/TRAIN-panel, DEV, export and state-hash term by policy. Optimizer milliseconds/update are81.828 and120.677. Per-DEV-example forward/decode/metrics/packing costs are17.410/15.044ms for context matched/historical and24.001/25.200ms for control. These include CPU work and are not isolated CUDA inference latency. All four calibration exports/prediction curves are included, not only an endpoint. Full outer accounting also includes CPU dataset/pair preparation, replay comparisons, other hashing and lifecycle work outside individual timers; do not treat optimizer or summed evaluation timers as total occupancy.

Both arms retain57,853,781parameters. Public-length/schedule MAC audit gives276,544,914,718,720(context) and287,118,920,253,440(control), ratio1.038236; counts exclude unchanged heads/backward/elementwise work and are not measured FLOPs. The10pass control's phase/depth-shift caveat remains; timing does not establish an optimized compute baseline or an efficacy/efficiency claim.

For later budget planning only, this actual paired main cost is32.33minutes. A **fresh original arm** in another seed cannot reuse seed201's trained baseline: historical S15 original mixed cost484.141408seconds covers its historical-policy curves, and matched-policy curves require additional work (the separately observed early-three reference cost155.09seconds plus S17 mixed endpoint46.72seconds provides a rough same-architecture basis). Thus a same-scope fresh three-arm seed is roughly1940+484+155+47≈2626seconds before any additional reserved-confirmation populations, seed variation or new audit exports. This is a planning comparison, not a new allocation or exact forecast; parent backbones may be reused only with their own frozen model/AdamW provenance, not to skip a fresh original/control continuation. Root alone decides whether outcome gates warrant replication and how to reserve its budget.

Separate charges remain: profile86.53, failed reference63.93, successful reference155.10, main1940.08seconds; total new S18 releases2245.64seconds. Inherited S15/S17 work is not charged twice. No outcome preview is included here; pinned paired analysis and independent raw audit follow the complete archive.
