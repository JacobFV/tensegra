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
