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
