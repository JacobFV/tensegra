# Structure pilot report

## Scope and protocol

This pilot tests whether a supplied directed read graph improves prediction in two
synthetic dynamical-system generators. It is a small, paired experiment rather
than a generalization or scaling study. The `sparse` generator samples a sparse
dependency graph; the `robot` generator uses a fixed robot-shaped topology. Both
use the same positive, uniform-neighbor mechanism. Topology is fixed within each
domain/seed run: sparse topology varies across seeds, while robot topology is the
same across seeds. They therefore test whether this shared mechanism is easier to
learn when its graph is exposed, not arbitrary signed edge coefficients, new-graph
transfer, physical robot dynamics, or vision-language-action behavior.

Each domain used 12 nodes, four history steps, 128/32/32 independently generated
train/validation/test trajectories, three seeds, and training-only scalar
normalization. The graph predictor had width 32, four heads, two attention layers,
17,089 parameters, batch size 32, and 300 AdamW steps at learning rate 0.001. The
tokenwise MLP had 193 parameters. One-step test MSE, ten-step recursive rollout
MSE, and a transient initial-state-shift metric were recorded. That transient
metric is also ten-step recursive rollout MSE: its fresh trajectories start at
twice the usual initial-state scale with `burn_in=0`. It is a
transient scale shift on the same graph and mechanism, not new-graph transfer.

The five graph-predictor conditions were no structure, true-graph soft bias at
lambda 1 and 4, a node-permuted graph at lambda 1, and a true-graph dense hard
mask. For every domain/seed group these conditions shared the initial model-state
hash, batch-schedule seed and batch-schedule hash, data splits, optimizer, and
budget. The paired hashes in the raw rows verify those controls. The hard mask is
implemented by dense attention scores with disallowed entries masked out; it does
not establish sparse compute or a speed improvement.

The soft strength was selected separately for each domain by the mean validation
normalized MSE across seeds. Lambda 4 won for both domains. Test metrics were not
used to choose lambda. The zero predictor, persistence predictor, tokenwise MLP,
and conditional-mean oracle are calibration references. The oracle receives the
generator coefficients and noise-free conditional mean, so it is privileged and
is not a fair learned baseline.

## One-step results

Mean normalized test MSE across the three seeds is shown below; lower is better.
The parenthesized dispersion is the population standard deviation (`ddof=0`) over
the three recorded seeds.

| Condition | Sparse | Robot |
|---|---:|---:|
| Graph predictor, none | 0.583298 (0.040800) | 0.504851 (0.004890) |
| True soft, lambda 1 | 0.557239 (0.040269) | 0.464857 (0.005435) |
| True soft, lambda 4 (selected) | 0.551450 (0.045746) | 0.454132 (0.004716) |
| Permuted soft, lambda 1 | 0.584403 (0.040017) | 0.504787 (0.005056) |
| True dense hard mask | 0.552220 (0.047105) | 0.453652 (0.004952) |
| Tokenwise MLP | 0.607135 (0.035316) | 0.516567 (0.005368) |
| Zero predictor | 0.983255 (0.042505) | 1.007248 (0.013145) |
| Persistence | 0.753808 (0.046739) | 0.625681 (0.007298) |
| Privileged conditional-mean oracle | 0.538853 (0.038112) | 0.444623 (0.004282) |

Relative to the paired no-structure graph predictor, selected soft bias reduced
the mean normalized one-step MSE by 5.460% in `sparse` and 10.046% in `robot`.
Using `selected - none`, the paired differences were:

| Domain | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD |
|---|---:|---:|---:|---:|---:|
| Sparse | -0.030921 | -0.038997 | -0.025623 | -0.031847 | 0.005499 |
| Robot | -0.052805 | -0.050552 | -0.048800 | -0.050719 | 0.001639 |

All three paired differences favored selected soft bias in both domains. With only
three seeds, these are descriptive effect sizes rather than a precise uncertainty
estimate or evidence of a broadly reproducible advantage.

The matched-strength control isolates graph content from lambda. At lambda 1,
`true soft - permuted soft` normalized one-step MSE was -0.023694, -0.030242,
and -0.027555 for the sparse seeds (mean -0.027164; population SD 0.002688), and
-0.038305, -0.039767, and -0.041718 for robot (mean -0.039930; population SD
0.001398). The permuted condition was essentially level with no structure in the
domain means, while the correctly aligned graph improved them. This supports an
alignment-specific effect in these generators rather than an indiscriminate gain
from adding positive attention bias.

## Rollout and transient shift

Ten-step recursive rollout results were mixed. Mean normalized rollout MSE for
`none / selected soft / hard` was 0.922631 / 0.909979 / 0.906808 in sparse, but
0.808069 / 0.812898 / 0.812904 in robot. Thus the one-step improvements did not
consistently transfer to recursive rollout; selected soft was slightly better in
sparse and slightly worse in robot. For the matched lambda-1 control, mean
`true - permuted` rollout differences were -0.021848 (population SD 0.062292) in
sparse and +0.061129 (0.066405) in robot, with seed-level signs not uniformly
favorable.

The scale-2, zero-burn-in transient shift is severe for the graph predictors. Mean
normalized MSE for `none / selected soft / hard` was 218.258 / 218.492 / 218.351
in sparse and 266.374 / 268.842 / 268.946 in robot. Selected structural conditions
did not improve this metric. The tokenwise MLP was lower at 136.473 and 104.145,
respectively, but those values remain far above its in-distribution one-step error.
For the matched lambda-1 control, mean `true - permuted` shift differences were
+0.549 (population SD 0.773) in sparse and +1.510 (0.671) in robot. This shift
probes extrapolation from unusually large initial values before the dynamics settle;
it should not be described as morphology, graph, or mechanism transfer.

## Interpretation and next experiments

The clean result is narrow: correctly aligned structure improves normalized
one-step prediction for the shared positive-neighbor mechanisms in this pilot, and
a matched permuted graph removes the improvement. The hard mask reaches similar
one-step error, but because it is dense it says nothing about sparse-attention
throughput. Recursive and transient-shift results prevent a stronger claim about
robust long-horizon dynamics or distribution shift. These synthetic systems are
neither a physical robot evaluation nor a VLA evaluation.

The next experiments should remain simpler than an interpreter. First, increase
the paired seed count and add confidence intervals. Then vary signed and
nonuniform edge coefficients, graph sparsity, noise, and rollout horizon. Add
missing/noisy-edge and reversed-graph controls, and evaluate held-out graphs and
larger node counts as separately labeled transfer axes. A graph-as-input baseline
would test whether the benefit comes from access to topology or specifically from
injecting it into attention. For the robot proxy, vary chain/tree morphologies and
contact edges while retaining the caveat that graph hops are not physical
propagation delays. Measure sparse kernels only after implementing actual sparse
computation. These steps are needed before pretrained adaptation, scaling,
compositional-generalization, or interpreter claims.

## Reproducibility and resources

The committed raw artifacts are `.development/results/pilot-metrics.jsonl` and
`.development/results/pilot-summary.json`; the configuration is
`configs/pilot.json`. They record source revision
`acf9ce0d2077a7631aa8c3ed5865d271a0fe0292`, Python 3.12.3, PyTorch 2.14.0+cpu,
and CPU execution. All 54 rows completed: 36 learned rows reached 300 optimizer
steps, and 18 analytic/reference rows recorded zero optimizer steps. The graph
predictor has 17,089 parameters and the token MLP 193. Recorded process peak RSS
was 366,360 KiB (357.77 MiB); this is process-lifetime RSS and cannot be attributed
to one condition. CUDA peak memory is null because the run used CPU.

The sum of per-row recorded wall times is 32.291 seconds (32.286 seconds for learned
rows). This sum is useful for accounting, but it is not an independently measured
end-to-end experiment duration and should not be reported as one.

Recreate the environment and run tests as documented in the repository README.
To reproduce the pilot in a fresh output directory:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m topoformer.experiment \
  --config configs/pilot.json --output results/pilot-reproduction
```

The checked-in report can be audited without retraining by recalculating aggregates
from `.development/results/pilot-metrics.jsonl` and comparing selection, paired
differences, environment, and resource fields with
`.development/results/pilot-summary.json`.
