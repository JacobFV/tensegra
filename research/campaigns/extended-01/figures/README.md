# Campaign figure snapshot

Reproduce extraction from repository root:

```sh
python3 research/tools/campaign_figures.py --root . --extract-only
```

Render with an existing matplotlib environment (no torch or GPU required):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 research/tools/campaign_figures.py --render-data research/campaigns/extended-01/figures/plotted-data.json.gz --output research/campaigns/extended-01/figures
```

`data-manifest.json` records row counts, source hashes, compressed/uncompressed table hashes and statuses. `plotted-data.json.gz` is deterministic gzip (mtime zero) over compact sorted JSON and retains exact numerator/denominator rows, source paths, SHA256 hashes, audited Wilson intervals where available, explicit pending statuses, and the attention condition-index mapping. The learning curves use archived manifest counts; confirmation, return, and composition panels use independent audit receipts. Rendering performs no model inference. Seed variation is shown by individual traces/dots; no confidence band pools seeds, shared events, or repeated delays. S01 mechanical profiling is excluded from the development curves.

- `semantic`: separate raw/calibrated learning, fresh paired S12 confirmation, and S10 supplied decoding. Only audited S12 seeds appear. Threshold comes from the audit receipt.
- `attention`: A06 curve population and final population are separate. Original-target scores appear in the final panel; supplied-target counts remain in the table. Condition order is recorded explicitly. No full attention tensor reconstruction is claimed.
- `attention-controls`: A09 graph-record learning and distinct final endpoints, plus A08 corruption with original/supplied targets on separate line styles.
- `returns-composition`: R04 scalar recovery and R05 use are separate; test/8-distractor views and changed-fact gates are shown. C03 uses the fixed endpoint, fresh clean/reversed development views, with answer and scalar metrics separate.

This is an intermediate snapshot, not a final campaign report. C04 confirmation and remaining S12 seeds need final refresh. A07 is represented by the replayed A09 prefix, not a separate plot. R05 signed decision-margin strata are retained in the machine-readable table. Consult the current experiment registry for execution status. S13 has no plotted model outcome. The return plots cover the restricted original mixture, not balanced tails or historical six-field reconstruction. C03 remains one development initialization with supplied scheduling; it is not three-seed confirmation.

Validation: stdlib extraction completed with 18,530 rows / 135 hashed files; integer attention counts checked; paired S12 population receipt asserted. Script compiled and all four figure layouts visually inspected. Matplotlib 3.11.2 rendered PNG/SVG/PDF on `gb10-direct` pilot venv with CUDA hidden and BLAS/OpenMP threads limited to two: 2.042 seconds wall time on current render (earlier drafts 1.506 and 1.531 seconds). No dependencies installed, upgraded, or GPU work performed.

Lossless table compaction checked against the previous JSON: all 18,530 rows equal, no duplicate complete rows, deterministic gzip round trip byte-identical. The table shrank from 5,981,884 to 121,726 bytes. Rendering accepts both legacy JSON and gzip inputs; extraction writes gzip plus the small manifest. Figures require no regeneration for this encoding change.
