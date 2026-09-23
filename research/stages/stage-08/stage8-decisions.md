# Stage 8 implementation decisions

- The default is a genuine width1024 latent workspace and recurrent computation. A narrow internal bottleneck may not silently replace that capacity. Smaller unit-test dimensions are labeled tests, not experimental standards.
- Return controls distinguish mixed additive encoding, disjoint concatenated field encoding, and facet-token memory. The latter two can share parameters/informative coordinates while differing in allocated KV tokens; the mixed control may differ in encoder parameters, which must be reported. Field-specific query rows are an explicit decoding prior in every matched arm.
- Candidate order and nonce identities are randomized. Episode construction must retain multiple supported hypotheses until relevant primitive/order evidence arrives; unique destination keys must not trivially disambiguate the entire candidate.
- Exact observation-id ledgers, duplicate suppression and retraction semantics are supplied architectural mechanisms. Candidate/evidence compatibility and increments remain learned. Oracle support is supervision/evaluation only, never learned actor input.
- CUDA setup uses a new remote virtual environment (`~/topoformer-stage8-cuda`), preserving the earlier CPU environment. Official PyTorch CUDA13.0 wheels are used, following https://docs.pytorch.org/get-started/locally/ . Actual GPU forward/backward timing and device support must pass before experimental budgets are frozen.

## Hardware verified

Remote NVIDIA GB10, isolated PyTorch `2.14.0+cu130`, Python3.12. A1024-wide CUDA matrix forward/backward completed successfully. Dependency freeze: `results/stage8/cuda-requirements.txt`. Earlier CPU environments unchanged. Width1024 timing is authorized serially for beliefs, returns and semantics; no pilot result may count as a competence gate. All small-width tests are mechanical only.

## Pre-main acquisition budgets

CUDA timing validated width1024: belief arms16,851,011parameters (~404MB peak); returnfacet model55,854,360parameters (~1.18GB); semanticmodel57,861,981parameters (~1.22GB). These are timing probes, not competence evidence.

Authorize sequential independent acquisitions: belief two learned arms,256updates×batch32 fresh episodes,128validation only; return six encoder/access arms,300updates×32fixed examples,128validation only; semantic8distinctgraphs,1000presentations with node/identity/edge curriculum boundaries100/300, semantic/no-input arms. All use1024-wide computation. Fixed-set success does not authorize composition; no final test outcomes are used for recipe selection. Main budgets will be frozen after these timings/acquisition diagnostics.

## Belief main budget

Acquisition (256updates, one seed,128validation cases/cell) showed near-perfect protected final answers on clean/retraction/contradiction but incorrect empty-prior posterior; generic recurrence struggled with retraction. These are acquisition observations, not competence. Freeze the same architectures at1000updates×batch32, three paired seeds,512validation/test per condition, including empty/partial, duplicate/reordered, contradiction and retraction checks. Do not repair the empty-prior failure inside the main recipe. All-frame posterior fidelity and impossible-mass gates remain mandatory alongside final answers.

## Return main budget

Authorize fresh-data1000updates×batch32, three paired initializations, six encoder/access arms, genuine width1024. Train delays0/1/2/4; evaluate clean0/1/2/4/8/16/32, interventions16only,512examples/cell. Required second-operand identity excludes unary null cases. Use bounded evaluation minibatches and freeze config/source before final test. Exactstorage and neuralreadout remain separate.

## Observation-ID fairness correction

A source audit during the frozen belief main run found that protected ledger operations consume observation IDs while the generic recurrent encoder only consumes payload/role/action. Preserve this run as an architectural-plus-metadata-access comparison; clean/order conditions have a narrower interpretation. Do not claim a strictly matched update comparison on ID-dependent duplicate/retraction controls.

A separately frozen followup will expose the same stable public ID features to both learned encoders, test ID sensitivity and repeated-ID consistency, and repeat unchanged1000update/3seed/512condition budgets atwidth1024. No silent source substitution or pooling of recipes. GPU queue: currentbelief → semantic batchtiming → returnmain → ID-matched beliefmain → semantic exposure sweep.

## Artifact publication

The initial semantic COO prediction archive exceeded GitHub's100MB file limit; its push was rejected. The original is preserved remotely with a SHA256 manifest. Lossless bit packing reduced it to22MB, with every decoded tensor checked. A clean-tree progress snapshot is published on `feat/stage8-progress`; final main integration will exclude the superseded oversized blob's unpublished history while retaining source snapshots/hashes and all raw evidence. No Stage1–7 history is rewritten.

## Explicit historical width ablation

Before ID-matched1024 results are observed, freeze a separately labeled width32/FF64 historical control on exactly the same Stage8 ID-matched task, data,1000updates, seeds and gates. Default/main remains1024. Run remotely with two CPU threads after the primary1024 study; expected5–7minutes including full output. This ablation can test width within the new recipe; neither its result nor cross-stage differences may be used to retune the frozen1024 main.

## Semantic exposure main budget

Optimized batch32 BF16 timing at genuine width1024 measured approximately 4.1–4.4ms per presentation, peak CUDA1.29GB. Authorize corpus cardinalities1k/10k, three paired seeds, semantic/no-input arms, checkpoints at actual10016 and100000 presentations along each trajectory:1.2million total presentations. Estimated85minutes warm training plus preprocessing/evaluation, roughly95–110minutes total. Run remotely after the ID-matched belief primary; the historical width32 CPU control may overlap using two threads.

Before launch, freeze a balanced lesson presentation schedule shared across corpus sizes and log actual unique graphs, repetitions and per-lesson metrics. Alpha-deduplication leaves only four binding and120set graphs; a sequential corpus traversal would otherwise confound corpus size with lesson mixture. Fresh heldout seed800000 excludes previously inspected canonical evaluation keys; it has no fresh binding examples, so binding transfer cannot be claimed. Calibration uses training graphs only. Batch32 checkpoint10016 is reported literally rather than relabeled10000. No main-outcome recipe tuning or downstream composition.
