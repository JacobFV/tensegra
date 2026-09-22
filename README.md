# Topoformer structure pilot

This repository contains a controlled pilot for testing graph-biased attention on
synthetic sparse and robot-shaped dynamical systems. The first generators use a
shared, nonnegative, uniform-neighbor mechanism. That makes topology identifiable
to a shared permutation-equivariant predictor, but it does not test arbitrary
edge coefficients or physical rigid-body dynamics.

Run the tests in the prepared environment:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src python -m pytest
```

Run the paired pilot:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  python -m topoformer.experiment --config configs/pilot.json --output results/pilot
```

The runner validates configuration before allocating data, uses training-only
scalar normalization, clones one initial model state and one batch schedule across
graph ablations, and writes `metrics.jsonl` plus `summary.json`. It reports raw and
normalized one-step error, recursive rollout error, and a transient initial-state
shift using fresh trajectories at twice the initial scale with no burn-in. The
zero and persistence references expose noise-dominated results. The conditional
mean oracle receives privileged generator weights and is explicitly diagnostic,
not a fair learned baseline. Soft-bias strength is selected from validation error;
test errors are not used for selection. A wall-time limit retains completed rows
and marks the summary incomplete rather than implying a matched comparison.
