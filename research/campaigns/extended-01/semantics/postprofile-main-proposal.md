# Measured post-profile main preparations — not launch authority

All four mechanical profiles completed with unchanged recipes and fixed populations. Root received GPUFREE/full occupancy before archive. No profile outcomes select a model, policy, checkpoint, population or training recipe. Source-bound main preparations retain the original S11parent/LR/S13exposure and both all-six diagnostic populations/thresholds.

| Profile | Full process | Guard preflight | Optimizer compute | Evaluation work |
|---|---:|---:|---:|---|
| S13 English |55.861390s|2.011768s|1.505836s/20updates|35.249815s for two bilingual checkpoints|
| S13 mixed |68.960136s|2.046210s|1.551793s/20updates|49.373152s for two bilingual checkpoints|
| S14 motif |20.162598s|2.919737s|None|Fixed8rows per all6endpoints|
| S12 rename |21.718716s|1.950634s|None|Fixed8rows per all6endpoints|

The S13profiles share exact initial model, inherited AdamWsteps, first batch, full construction/negative-pair sequence hashes, renderer phases and construction visits. English used160English presentations; mixed used85English/75Spanish under the prespecified phases, not a rebalanced draw. Same-checkpoint English TRAIN128thresholds equal the Spanish thresholds at both checkpoints. All profile artifacts remain archived; no profile outcome is used for promotion.

Proposed main caps for review are600seconds **each** for English/mixed, and420seconds **each** for all-six motif/rename. These are proposals, not allocations. S13GPU optimizer-time extrapolation gives308.4/317.8seconds for4096updates; four fixed bilingual evaluations extrapolate70.5/98.7seconds. CPU target generation, batch preparation, checkpoint I/O, initialization and full matrix hashing add overhead;600seconds covers those costs without altering exposure. A20-step profile is a limited timing sample, so actual main occupancy remains measured and timeout outcomes retained.

For S14/rename, a deliberately conservative envelope scales the entire six-endpoint profile loop (including repeated model-load overhead) by1024/8, then adds the measured fixed preflight/setup remainder: about357seconds for motif and400seconds for rename. Both proposed420second caps cover that envelope. This extrapolation overcounts fixed model-load costs; it does not inspect predictive quality or shorten a population. All6×1,024 main forwards remain required. Caps/timeouts never authorize a rerun or an outcome-selected subset.

New `campaign_semantics_main_freeze.py` creates immutable `*-main-frozen-v1.json` files from original prepared recipes, adding only source/prepared hashes, the complete already-fixed S12checkpoint/manifest matrix, proposed cap and explicit separate-release notice. The lexical main prepared copy adds `job=main`/status metadata to the original frozen all-six config; its evaluator at4a05aba is unchanged. S13trainer and S14evaluator are also unchanged. New `*_main_launch.py` wrappers share the audited v2 accounting pattern: started receipt before byte guards, whole-process cap subtracting preflight, failure receipts, idle/prefix guards and fsynced atomic receipts. Their source guards bind all dependencies and compare the six primary inputs exactly; no profile bytes or profile checkpoints become main initialization.

Main freezes: `campaign-s13-{english,mixed}-main-frozen-v1.json`, `campaign-s14-motif-main-frozen-v1.json`, `campaign-s12-renamed-main-frozen-v1.json`. The main output directories remain those of the prepared recipes. S13mains restart from immutable S11checkpoint3799ade5, never from20-step profile checkpoints. Neither new-main source/config staging nor GPUlaunch is performed in this preparation; root review/allocation/release is next.

Validation: CPU tests preserve every original recipe field, reject freeze overwrite/cap/source/matrix mutations, and exercise started/final receipts plus preflight subtraction/failure handling for all three new wrappers. Tests pass without any child process, model construction or CUDA access. Existing frozen-v2 profile sources/configs remain unchanged. Independent review is requested before launch.
