# C04 mechanical profiles and prospective main allocation

All five serial phases from immutable executable source `8a92869` exited0, without timeout. Atomic remote receipts sum **21.218453886453062 GPU-process seconds** against420 allowed. GPUFREE and full process state were captured at2026-09-24T05:12:47Z before analysis. The first four phases used config SHA256 `1ce378c705a59cc11ebc6ec045e3066f717b959811dc2835ab392dfc9fb3a932`; timing used `d92905397274e2cb7e94368311d3d721ac8aac7c5e5616806eaf6251f2bd02c3`, adding only the three fixed16 endpoint paths/hashes. No accuracy-based choice entered timing.

| Phase | External seconds | Internal seconds |
|---|---:|---:|
|Hybrid and supplied-copy controls|6.557594823|5.575219009|
|N1 static|2.778078792|1.670895281|
|N1 roles|2.777742747|1.640547659|
|N2 rekey|5.559845667|4.352149200|
|Full-path inference timing|3.545191857|0.785799482 setup; cell timings separately retained|

Both N1 arms have identical initial-state and sample-index hashes; all three arms share the sample-index stream. Hybrid frozen hashes match before/after, as do timing interfaces. Mechanical populations have16 test events and16 updates, so their accuracy/gate flags establish no acquisition or sufficient causal support. Full summaries, original endpoint hashes, per-pass timing and an exhaustive raw artifact SHA256/size/mtime manifest are retained under `research/results/campaign-01/composition/c04-profile`. Full PT artifacts remain in corresponding GB10 `composition/c04-profile-PHASE` directories (roughly407MB total, mostly neural optimizer/checkpoints).

## Cost proposal before any confirmation outcome

Requested caps below apply identically to each of the three predeclared seeds. They require coordinator/reviewer freeze and separate scheduler release; no main phase is launched by this report.

| Phase per replicate | Expected seconds | Requested hard cap |
|---|---:|---:|
|Hybrid, supplied-copy, oracle and public/causal controls|200–450|900|
|N1 static4000|80–140|180|
|N1 roles4000|80–145|180|
|N2 rekey4000|280–400|600|
|Independent full-path inference benchmark|30–60|120|
|Total per replicate|670–1195|1980|
|All three replicates|2010–3585|5940|

Profile calibration0→16 artifact intervals are0.349000s static,0.350273s roles and1.358000s rekey. Scaling these inclusive intervals by250 gives87.25/87.57/339.50s, but includes first-interval initialization and tiny-population evaluation. Previous full C03 pair156.908s and C02 rekey286.884s provide the more representative anchors; larger C04 audit/export overhead motivates the stated ranges and caps. The new hybrid has two views and expanded oracle/control coverage. Its16-row profile has substantial setup and single-batch overhead, so blindly multiplying6.56s by256 is inappropriate. Previous C01 full hybrid117.99s plus expanded coverage motivates200–450s expected;900s conservatively replaces the draft600s ceiling without changing any scientific setting.

The sum of all12 measured timing cells is0.557586172s for one16-row pass. Scaling each by16 for256 rows and3 repeats gives26.764136s, before loading, warmup and larger-batch behavior. Batch64 measurement used a partial16-row batch, making that scaling conservative on batch count but not a guarantee of throughput. Requested120s cap retains substantial margin; main first256/warmup8/three-pass policy stays unchanged. Workspace batch1 delay16 alone accounts for18.683493s of this projection. No efficiency conclusion follows from profile timings.

Active training consists of nine new neural fits,4000×256 presentations each:9,216,000 total. Hybrid and supplied-copy perform no new fitting; their frozen acquisitions are still charged in the inherited ledger and executed inference cost. N1 inherits P01 initialization; N2 starts scratch. Workspace executes56,069,440 parameters (including55,854,360 backbone); supplied-copy executes181,255; N1 executes1,246,248; N2 executes17,028,136. Benchmark residency additionally includes all frozen controls and all three neural models, which is reported separately from executed counts. Repeated backbone execution is included in timed workspace paths.

`C04-inherited-ledger.json` preserves each302/303/304 P01 lineage and10/11/12 Stage9/Stage11/R04/R05 acquisition exposure and timing scopes. These historical runner costs overlap selected-head fitting and are not added again as new C04 occupancy or incorrectly summed as pure fitting. Failed development/search costs remain in the campaign ledger. Equal current presentations do not imply equal total training compute.

Main source model logic, three seeds, namespaces, endpoint/selection rules, gates and all retained arms remain unchanged. Timing checkpoint bindings will use each main fixed4000 endpoint by hash, regardless of measured scores. Proposed total incremental main ceiling5940s is separate from the already incurred21.218454s mechanical profile cost.
