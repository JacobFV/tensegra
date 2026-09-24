# Campaign scientific figures

Reproduce extraction from repository root:

```sh
python3 research/tools/campaign_figures.py --root . --extract-only --c04-audit research/campaigns/extended-01/review/C04-lineage-0-audit.json --c04-audit research/campaigns/extended-01/review/C04-lineage-1-audit.json --c04-audit research/campaigns/extended-01/review/C04-lineage-2-audit.json
```

Render in an existing matplotlib environment (no torch or GPU required):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 research/tools/campaign_figures.py --render-data research/campaigns/extended-01/figures/plotted-data.json.gz --output research/campaigns/extended-01/figures
```

`data-manifest.json` records row counts, source hashes, compressed/uncompressed table hashes and statuses. `plotted-data.json.gz` is deterministic gzip (mtime zero) over compact sorted JSON. It preserves exact numerators/denominators, source paths, audit status, return margin strata, paired contingency counts and condition mappings. All 20,686 rows are unique; 195 inputs are hashed. Return Wilson intervals remain in the table, not plotted. No confidence band pools fitted seeds, shared events or repeated delays.

- `semantic`: raw/calibrated learning, fresh S12 pairs and separate S10 supplied decoding including S11. S01 mechanical profiling excluded. S12 remains partial: seed701 audited; 702/703 pending.
- `attention`: A06 curve and final populations separated; supplied-target counts in table.
- `attention-controls`: A09/A10/A11 graph-record learning with distinct unconnected final markers; A08 original/supplied corruption targets separated.
- `attention-read-localization`: audited A12 four frozen policies/three shapes, paired fixes/regressions, and head diagnostic on explicitly different support from event-level path counts. Their numerical gap does not identify query-head failures.
- `attention-common-route`: audited A13 event-level task/query-route counts, actual conditional task counts and queried head paths on shared event support. Common hard is engineered and oracle privileged; one frozen checkpoint is not seed replication. All-node suffix outcomes remain separate in the data.
- `returns-composition`: restricted original-mixture R04 scalar recovery, R05 downstream use and changed-fact gates; C03 fixed development endpoint. These do not pass historical six-field reconstruction or establish balanced-tail competence.
- `composition-confirmation`: all three independently audited C04 lineages. Fixed4000 neural x markers are primary, selected + markers secondary. Workspace joint success, answer-only success, refusals, supplied-copy and changed-fact gates remain distinct. Supplied scheduling and finite numeric train/test overlap are explicit.
- `composition-accuracy-timing`: primary fixed4000 roles versus workspace delay16 and sole-return supplied-copy, with batch1/64 execution timing on log axes. Timing uses median recorded repeats, excludes setup and training, and does not establish a deployment speedup. Every raw timing repeat is retained.

C04 summaries require explicit audit receipts with exact `input_sha256` bindings. The optional `--c04-provisional` flag is only for explicitly authorized completed-worker previews, visibly marked pending audit; it is not used in this final C04 export. A10–A13 inputs likewise require audited bindings. No profile efficacy plots or model inference occur.

Validation: extraction, compilation, unique-row check and exact input hashes passed; all figure layouts inspected across incremental renders. Deterministic gzip round-trip and legacy-JSON losslessness verified. Rendering used existing matplotlib3.11.2 on `gb10-direct`, CUDA hidden and BLAS/OpenMP threads limited to two. Final C04 refresh: 3.958 seconds CPU wall (initial caption draft4.050 seconds). Previous provisional all-three refresh:3.851 seconds. `render-receipt.json` records current provenance. No dependencies installed/upgraded and no GPU work performed.
