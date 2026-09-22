# Development journal

## 2026-09-22 — repository inspection and initial design

- User requested general structure-biased attention, time-series and morphology
  benchmarks, then interpreter-guided data/execution routing, scratch/pretrained
  comparisons, and scaling studies; record thoughts here and commit/push progress.
- Repository began empty with no commits; origin is JacobFV/topoformer on GitHub.
  Remote inspection returned no refs. No ancestor AGENTS.md was found in the
  inspected home/Documents hierarchy.
- Local inspection: 121 GiB RAM total, approximately 42 GiB available, and nearly
  all 15 GiB swap used. Python has NumPy but no Torch, pytest, or Transformers.
  No training or dependency installation performed locally.
- Batch-mode SSH to gb10-direct succeeded. Host promaxgb10-4dfb reports NVIDIA
  GB10, 119 GiB system RAM, about 115 GiB available. Device memory querying returns
  N/A; do not interpret that as zero memory or unlimited capacity.
- Wrote proposed design.md. Principal concerns: read-edge direction, cyclic
  dynamics versus time-unrolled DAGs, ordered call arguments, fresh invocation
  identity, latent association collapse, and leakage from oracle execution traces.
- Prior art establishes structural attention bias; no novelty or performance claims
  are supported yet. No product code, tests, or experiments exist at this point.
- Workflow status: architectural design review precedes implementation under the
  installed superpowers brainstorming skill. This design is prepared for review;
  it is not recorded as approved.

### Scope clarification and execution preference

- User supplied the broader 'programming the metric space' thesis, then emphasized
  that easy intermediate sequence and robot experiments must come first. Preserve
  both: the long-term hypothesis and the immediate, testable attention experiment.
- User explicitly requested subagents for implementation. Preserve that choice.
- Clarified that soft bias favors graph-local reads whereas hard masks enforce
  them. The robot proxy must compare both and test multi-layer receptive fields.
- Remote read-only inspection: Python 3.12.3, pip 24.0, NVIDIA driver 580.126.09,
  reported CUDA support 13.0, zero GPU utilization and no GPU processes at inspection.
  These are environment observations, not verified PyTorch/CUDA compatibility.
- User requested clean hierarchical design, reusable abstractions, minimal code,
  clear logging, and ongoing notes. Added these constraints to design and plan;
  separated training/evaluation from thin experiment orchestration and kept both
  benchmark domains on a shared implementation path.

### Implementation started after approval

- User approved plan-01 and selected subagent implementation. Work is isolated on
  `feat/structure-pilot` in `.worktrees/structure-pilot`; task agents own disjoint
  modules, and fresh reviews precede integration.
- Parent is creating `~/topoformer-pilot/.venv` on gb10-direct. No local Torch
  installation or training. Tests and pilots use at most two CPU threads initially.
- Ruling: retain early transient samples for the initial-state-shift evaluation.
  Applying a doubled initial condition before 32 warmup steps in a contractive
  system would nearly erase the shift. The shifted test must start with the
  shifted state and no warmup, and be described as a transient shift.
- Ruling: report a zero predictor and the known generator's conditional mean as
  diagnostic references alongside persistence. These are trivial baselines, and
  make noise-dominated stationary trajectories visible rather than mistaking a
  small absolute MSE for learned dependency structure. The oracle is explicitly
  privileged and is not part of the fair learned-model comparison.
- Ruling: use a shared nonlinear neighbor-average mechanism in the first pilot.
  A shared equivariant predictor with no per-node identity cannot in general
  recover arbitrary hidden signed coefficients from a short context. Varying
  topology first gives an identifiable easy experiment; heterogeneous generating
  functions require richer system-identification context in a later stage.
- Environment setup encountered a network read timeout downloading cuDNN after
  the default aarch64 Torch wheel selected CUDA dependencies. Switched to the
  official CPU wheel index for this small reference pilot; do not infer CUDA
  incompatibility from a download timeout. Installation source:
  https://docs.pytorch.org/get-started/previous-versions/ (CPU index instructions).
- Remote environment verified by imports and matrix multiplication: Python 3.12.3,
  Torch 2.14.0+cpu, NumPy 2.5.3, pytest 9.1.1. Package versions are captured in
  environment-pilot.txt. This pilot will measure remote CPU performance honestly;
  GPU adaptation and benchmarking remain separate.

### Attention core implemented

- Added reusable directed attention and graph-mode conversion. Remote CPU suite:
  20 tests passed in 0.71 s after review fixes (command: `PYTHONPATH=src
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q`).
- Review caught mixed float64 value precision loss and rank expansion from a
  singleton tensor strength. Both now have numerical/shape regression coverage.
  Masking is applied after structural bonuses; zero strength recovers ordinary
  attention. Rectangular query/key lengths and isolated-node hard masks are tested.
- CUDA correctness remains untested in this CPU environment. No GPU performance
  or compatibility conclusion is drawn from device-preserving code alone.

### Pilot comparison protocol (before results)

- Primary result: paired held-out one-step normalized MSE for true-graph soft
  attention versus absent structure, reported separately for sparse and robot
  domains. Select lambda from {1, 4} by mean validation MSE, never test MSE.
- Secondary results: hard graph masking, permuted graph bias, tokenwise MLP,
  persistence, recursive 10-step rollout, and transient initial-state shift.
  Report all three seeds and paired dispersion; these are pilot observations,
  not reliable population-level significance or scaling-law estimates.
- Keep zero and privileged generator-mean diagnostics visible to expose the
  stationary noise floor. No graph transfer or physical robot-control claim.

### Synthetic generators implemented

- Sparse directed and named robot-tree generators now share the same bounded
  nonlinear neighbor update. Explicit self recurrence is included in read support.
- Added seeded graph corruptions, trajectory-local history windows, independent
  random generators, and burn-in/initial-scale controls. A reversed undirected
  robot graph is flagged as a redundant control rather than a meaningful ablation.
- Remote verification: generator suite 27 passed; full suite 47 passed in 0.77 s.
  Command: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src
  .venv/bin/python -m pytest -q`. Commit: 876adce. Independent review pending.
- Review clarification: `initial_scale` changes the state before burn-in, which
  is the intended meaning. The transient-shift evaluation will explicitly set
  `burn_in=0`; shifting after warmup would be a different intervention. Retained
  the generator API and require an integration check of the runner's choice.

### Generator signal preflight

- Before learned-model training, sampled each domain at n=12, generator seeds
  0/1/2, trajectory seeds 1000/1001/1002, count=128, steps=40, default warmup/noise.
  Evaluated every consecutive state pair on remote CPU from revision 0638242.
- Persistence raw MSE ranged 1.393e-4 to 1.443e-4; privileged generator mean MSE
  ranged 1.002e-4 to 1.010e-4, matching the configured 1e-4 innovation variance.
  Zero prediction ranged 1.713e-4 to 2.288e-4. There is measurable predictive
  signal beyond persistence. This diagnostic is not a learned-model result and
  uses separate seeds from the final experiment's split policy.

### Predictor and runner verification / review

- Initial implementation 88145f4 passed 57 remote tests and a tiny smoke run.
  Parent also ran the CLI from the exact remote Git checkout with both domains,
  all five attention variants, two optimizer steps, and one seed: 18 total rows,
  identical graph-model initial hashes/batches, finite errors, correct revision.
- Review confirmed node-count independence, permutation equivariance, hard-mask
  locality and reachable positive controls, train-only normalization, and
  prediction-fed rollout semantics. It requested per-domain selection/paired
  reporting, complete baseline metadata, an explicit NumPy dependency, protection
  against overwriting existing results, and smaller orchestration helpers.
- Fixes are underway before full pilot training. CPU peak RSS will be labeled
  process-lifetime rather than misrepresented as independently reset per mode.
- First runner review fixes (f29fdf0) passed 63 tests. Parent repeated the clean
  remote CLI smoke across both domains/all five modes: 18 finite rows, paired
  initialization, and exact revision. Scoped review accepted all prior fixes,
  but found a new deadline-boundary issue: a completed mode could be followed by
  another after budget expiry. A focused regression/fix is in progress.

### Full pilot launched

- Task 3 approved after deadline-boundary regression fixes (acf9ce0); full remote
  suite 65 passed in 1.20 s. A fresh clean-checkout CLI smoke also completed.
- Launched unmodified configs/pilot.json from exact remote revision acf9ce0:
  sparse/robot domains, seeds 0/1/2, 300 optimizer steps, five graph modes plus
  tokenwise MLP and diagnostic references. All computation on remote CPU, two
  threads, max_wall_seconds=1800 and an outer 1900-second process timeout.
- Remote output: ~/topoformer-pilot/run/results/pilot-acf9ce0; progress log:
  ~/topoformer-pilot/pilot-acf9ce0.log. Approximately 115 GiB system RAM was
  available immediately before launch. No local model training performed.

### Pilot results and artifact audit

- All 54 rows completed; all 36 learned-model runs used 300 optimizer steps.
  Verified identical graph-model initial hashes and schedule hashes/seeds for each
  paired case; finite metrics; exact source revision acf9ce0. JSONL rows exactly
  match summary rows, and normalized/raw metrics use consistent per-case scales.
- Validation selected lambda=4 separately in both domains. Mean held-out normalized
  MSE: sparse absent 0.583298, selected soft 0.551450; robot absent 0.504851,
  selected soft 0.454132. All three seeds improved for each one-step comparison.
  Correct lambda=1 graphs also beat same-strength permuted graphs.
- Generalization is mixed: sparse normalized rollout 0.922631 -> 0.909979;
  robot 0.808069 -> 0.812898. Transient normalized MSE is very large and does
  not improve (sparse 218.258 -> 218.492; robot 266.374 -> 268.842).
  This is evidence for the simple one-step structural prior, not the end thesis.
- Peak process RSS: 366360 KiB (~357.8 MiB). Sum of measured learned-model run
  times: 32.286 seconds; this excludes some orchestration/setup and is not an
  independent end-to-end measurement. Torch 2.14.0+cpu; no GPU training.
- Artifacts (175611 bytes total):
  - pilot-summary.json SHA256 0eecd465e7201927feed10ab83997e79bd46d5ddee32cda1437bbfdca87fd30d
  - pilot-metrics.jsonl SHA256 6d8008b089edbb8f11288fb7491d963fc12f6e84cebfc77b8f221b5990b9a089
- A subagent is writing the scientific report from immutable artifacts; a final
  whole-branch review follows. No hyperparameter/budget changes after results.

### Final review and delivery

- Final whole-branch reviewer independently checked full code/docs plus artifact
  arithmetic and hashes. Found and closed: missing/incomplete unbiased baseline
  could prevent partial summaries; README environment commands; report topology,
  optimizer, and transient-metric wording. Source robustness fix: 55eb3f5.
- Final scoped review: Task 4 spec/quality pass, whole branch ready, no remaining
  Critical/Important/Minor findings. Full clean remote suite at 55eb3f5: 67 passed.
  README core API example also executed remotely, including a hard-mask assertion.
- The measured pilot remains sourced from acf9ce0. Later changes only repair
  missing-baseline aggregation and documentation; reviewer checked complete-pilot
  selection and pairs remain unchanged. Raw artifacts were not modified.
- All first-plan tasks are complete. Work is committed and pushed on
  feat/structure-pilot; integration into main is a separate final decision.
- Next research should remain easy and controlled: data-efficiency sweeps,
  heterogeneous mechanisms with sufficient context, and held-out graph/size
  tests with graph-as-input controls before interpreter-guided soft grounding.
