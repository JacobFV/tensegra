# Structure-biased sequence and robot pilot implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce working, tested structure-biased attention and reproducible pilot comparisons on synthetic sequences and joint graphs, before attempting runtime grounding.

**Architecture:** A dense attention reference accepts a directed read graph. A shared predictor processes one history token per variable/joint with tokenwise readouts. Seeded generators and a paired experiment runner compare absent, true, corrupted, and hard-masked structure.

**Tech Stack:** Python 3.12 on the remote host, PyTorch, NumPy, pytest; standard-library JSON/CSV for configuration and results.

**Spec:** `research/stages/stage-01/design.md`, first-milestone sections and the user's sequence/robot-first clarification.

## Global constraints

- An edge i -> j means that query i may preferentially READ key j.
- Structural bias never unmasks a causal or padded position.
- Target values are never included in the input.
- Keep initial weights, examples, optimizer, training budget, and seeds paired.
- Select lambda and other hyperparameters only on validation data.
- No large local allocations: inspection found local swap essentially exhausted.
- Run training on gb10-direct, one pilot job initially, in an isolated environment with timeouts and measured memory.
- No interpreter, soft grounding, pretrained adapter, or scaling-law claims in this implementation cycle.
- Clean hierarchical design, reusable abstractions, minimal code, and clear logging and development notes.
- Both domains share attention, predictor, training, and evaluation code; domain differences belong in graph/data generation.
- No plugin registry or abstract base-class framework. Use explicit functions and small modules with tensor-shape contracts.

## Review focus

- All-masked attention rows: reject with a clear error rather than silently emitting NaNs (Task 1).
- Directed graph accidentally transposed: asymmetric three-node examples must distinguish row/column orientation (Tasks 1, 2).
- Hard-mask locality bypassed elsewhere: distant perturbations cannot affect a target before enough layers (Task 3).
- Data leakage and inconsistent ablations: independent trajectory splits, training-only normalization, identical starting weights and batches (Tasks 2, 3).
- Empty or isolated neighborhoods: define self access for hard masks while preserving original dependency support (Tasks 1, 2).

## Ownership and dispatch

Use separate implementer subagents and reviewers as required by subagent-driven-development.
Each implementer owns only the files listed for its task. All share a repository;
instruct them not to revert other workers' changes. Dispatch dependent tasks only
after their contracts are implemented and reviewed. Keep remote jobs sequential.
The parent owns environment setup, coordination, integration, and the final report.

## Task 1: General directed attention reference

**Files:** `pyproject.toml`, `.gitignore`, `src/topoformer/__init__.py`, `src/topoformer/attention.py`, `tests/test_attention.py`.

**Interfaces:**

```python
def structural_attention(q, k, v, *, bias=None, strength=0.0, allowed=None):
    # q: [B,H,Q,D]; k: [B,H,K,D]; v: [B,H,K,Dv]
    # bias and allowed broadcast to [B,H,Q,K]; allowed is boolean.
    # Return (output [B,H,Q,Dv], weights [B,H,Q,K]).
    ...

def graph_structure(read_graph, mode):
    # read_graph: boolean [B,N,N]; mode: none, soft, hard
    # Return (bias [B,1,N,N], allowed [B,1,N,N] or None).
    # hard permits read_graph OR identity; none returns a zero bias.
    ...
```

- [x] Add failing tests for zero-bias equivalence, positive directed edge bonus,
  rectangular query/key shapes, broadcast shape rejection, nonfinite inputs,
  padding/causal masking, all-masked rows, isolated nodes, and gradients.
  Pin edge orientation with this test:

```python
q = torch.zeros(1, 1, 3, 2)
bias = torch.zeros(1, 1, 3, 3)
bias[0, 0, 0, 2] = 1
_, weights = structural_attention(q, q, q, bias=bias, strength=2.0)
assert weights[0, 0, 0, 2] > weights[0, 0, 0, 1]
torch.testing.assert_close(weights[0, 0, 2], torch.full((3,), 1/3))
```

- [x] Create the isolated remote environment, install the package's minimal
  dependencies, and record actual versions. Try CPU first for these small tests;
  this pilot uses CPU-only Torch. CUDA validation is deferred until a GPU run
  is needed, following the recorded environment ruling.
- [x] Run `python -m pytest tests/test_attention.py -q` and confirm the initial
  failure comes from the missing implementation.
- [x] Implement the score equation, broadcast validation, additive bias, masked
  softmax, and output contraction. Compute scores and softmax in float32 for low
  precision inputs; preserve float64 when inputs use it. Return outputs in the
  value dtype. Require compatible devices and floating query/key/value tensors.
  Use `masked_fill(~allowed, -torch.inf)` after bias addition. Reject rows with no
  legal key; validate finite scalar/tensor strengths. Do not expand logits into
  copies per graph relation in this initial single-relation reference.
- [x] Run the attention tests, obtain review, fix findings, then commit and push.

## Task 2: Reproducible sequence and morphology generators

**Files:** `src/topoformer/data.py`, `src/topoformer/graphs.py`, `tests/test_data.py`.

**Interfaces:**

```python
@dataclass
class Dynamics:
    weights: torch.Tensor  # [N,N], row reads column
    read_graph: torch.Tensor  # boolean [N,N], exact nonzero support

def make_dynamics(kind: str, n: int, seed: int) -> Dynamics: ...
def trajectories(system: Dynamics, *, count: int, steps: int,
                 seed: int, noise: float = 0.01) -> torch.Tensor:
    # Return [count,steps,N], CPU float32.
    ...
def windows(series: torch.Tensor, history: int):
    # Return x [examples,N,history], y [examples,N], preserving trajectories.
    ...
def corrupt_graph(graph: torch.Tensor, *, mode: str, seed: int):
    # mode reversed, permuted, dropped; return boolean same shape.
    ...
```

- [x] Write tests for reproducibility, invalid counts/history, no global RNG
  mutation, bounded finite trajectories, exact dependency support, isolated nodes,
  asymmetry, morphology connectivity, and disjoint seeded trajectories.
  Verify window alignment explicitly:

```python
series = torch.arange(10.0).reshape(1, 5, 2)
x, y = windows(series, history=2)
torch.testing.assert_close(x[0], series[0, :2].T)
torch.testing.assert_close(y[0], series[0, 2])
```

- [x] Run `python -m pytest tests/test_data.py -q` and confirm failures.
- [x] Implement seeded sparse directed nonlinear dynamics and a branching robot
  tree using local `torch.Generator` instances. For the robot, use a torso/root,
  left/right arm and leg branches and terminal hand/foot joints; additional nodes
  extend the branches. Require at least nine nodes for that named morphology.
  Record topology labels. Use `x_next = 0.5*x + 0.5*tanh(x @ W.T) + noise` with
  absolute row sums of W at most 0.8. Include the self term in read_graph.
  Warm up 32 steps, then store requested trajectories. State plainly that this is
  synthetic local coupling rather than a physical rigid-body simulator.
- [x] Build graph corruption without altering generator weights or datasets.
  Reversal may equal the original for undirected robot graphs, so flag redundant
  controls. Permuting both graph axes preserves its degree multiset but is not
  necessarily a degree-preserving rewiring per named node; label it accurately.
- [x] Test one-step dependency by perturbing a non-parent with noise disabled and
  asserting the target update is unchanged. Run tests, review, commit, and push.

## Task 3: Shared predictor and paired pilot runner

**Files:** `src/topoformer/model.py`, `src/topoformer/training.py`, `src/topoformer/evaluation.py`, `src/topoformer/experiment.py`, `tests/test_model.py`, `tests/test_experiment.py`, `configs/pilot.json`, `README.md`.

`model.py` owns only prediction; `training.py` owns the shared optimization loop;
`evaluation.py` owns one-step/rollout metrics; `experiment.py` validates configuration
and orchestrates paired runs and artifacts. Keep these as small direct functions,
not class hierarchies. The module hierarchy should clarify responsibility without
splitting every helper into its own file.

**Consumes:** `structural_attention`, `graph_structure`, and the generator interfaces above.

**Produces:**

```python
class GraphPredictor(torch.nn.Module):
    def __init__(self, history: int, width: int, heads: int, layers: int): ...
    def forward(self, x, graph, *, mode="none", strength=0.0):
        # x [B,N,history], graph [B,N,N] or [N,N]; output [B,N]
        ...

def run(config: dict, output_dir: str) -> dict:
    # Return manifest/results; write metrics.jsonl and summary.json.
    ...
```

- [x] Write failing tests for variable-count support, joint permutation
  equivariance, hard-mask receptive fields, zero-strength model equivalence,
  finite backward passes, invalid configurations, deterministic tiny training,
  and split/initialization pairing. Use a chain for locality: after one layer,
  changing node 3 cannot alter node 0; after two layers, node 3 still cannot
  alter node 0. Disable dropout; test with multiple inputs to avoid accidental
  equality hiding a communication path.
- [x] Run `python -m pytest tests/test_model.py tests/test_experiment.py -q` and
  confirm expected failures.
- [x] Implement a shared history projection, tokenwise LayerNorm, residual
  attention/MLP blocks, and tokenwise scalar readout. No global pooling or learned
  fixed-count node embedding. Receptive field grows at most one graph edge per
  layer under hard masking. History channels include only that node's past.
- [x] Implement CLI `python -m topoformer.experiment --config configs/pilot.json
  --output results/pilot`. Validate unknown keys, positive sizes, nonnegative
  noise, width divisibility, and supported modes before allocating datasets.
  Bound CPU threads at 2 and use no DataLoader subprocesses. Support explicit
  CPU/CUDA choice and a maximum wall-time setting that writes partial results.
- [x] For each system and seed, generate separate train/validation/test
  trajectories with recorded split seeds; compute a shared scalar mean/std from
  training observations only. Reuse the exact splits and batch index sequence
  across modes. Clone a single initialized model state for all modes. Initialize
  fresh optimizers. Train with AdamW and next-step MSE.
- [x] Initial config: kinds `sparse` and `robot`, 12 nodes, history 4, width 32,
  4 heads, 2 layers, batch size 32, 128/32/32 trajectories of 40 observed steps,
  300 optimizer steps, seeds 0/1/2, learning rate 0.001. Modes: none, true soft
  lambda 1, true soft lambda 4, permuted soft lambda 1, true hard mask. Check
  validation every 50 steps; pick soft lambda using mean validation performance,
  never test scores. Keep each mode's curves available. The primary comparison
  reports the selected soft setting against unbiased and hard-mask runs.
- [x] Include persistence and a tokenwise MLP with the same history input as
  calibration baselines. Evaluate one-step MSE and recursive 10-step rollout MSE
  on fresh held-out trajectories, updating predicted history after each step.
  Also evaluate the same fixed generator on held-out initial states scaled by 2;
  label this initial-state shift, not new-graph transfer.
- [x] Record validation/test errors, parameter counts, wall time, process peak
  resident memory, CUDA peak allocation when supported, seed, configuration,
  git revision, and software/device versions. Report every seed and paired
  differences with mean and standard deviation. No scaling-law fitting yet.
- [x] Emit concise progress logs at validation intervals: domain, seed, mode,
  step/total, train/validation loss, elapsed seconds, and artifact path at run end.
  Do not log every batch or dump tensors. Append journal notes at each verified
  milestone describing decisions, results, and unresolved questions.
- [x] Run the full suite and a tiny 2-step CPU integration configuration; review,
  fix findings, document runnable commands, commit, and push.

## Task 4: Remote pilot and honest report

**Files:** `research/results/pilot-summary.json`, `research/stages/stage-01/pilot-report.md`, `research/stages/stage-01/journal.md`.

**Owner:** Parent coordinates one remote experiment job at a time; analysis may
be delegated after immutable result files are available.

- [x] Inspect remote load and available memory immediately before launch. Sync
  the committed source into a dedicated remote working directory. Use an isolated
  environment and record its path; never modify unrelated user projects.
- [x] Run the 2-step smoke configuration, inspect finite losses and artifact
  contents, then run `configs/pilot.json` with a 30-minute wall-time limit.
  If interrupted or time-limited, report incomplete runs and retain checkpoints/
  metrics; do not compare incomplete budgets as if they were matched.
- [x] Retrieve small result artifacts, verify mode/seed coverage, and compute
  paired differences. Do not claim improved generalization merely because training
  loss is lower. Identify what a three-seed pilot cannot establish.
- [x] Write a report describing the generator, controls, actual errors, resources,
  limitations, and next experiment. Commit small summaries and report; leave raw
  large artifacts/checkpoints out of Git. Push and verify remote commit identity.

## Deferred stages, explicitly tracked

The broader design remains the roadmap. After this pilot: data-efficiency sweeps,
new-system contextual prediction and held-out graph/size transfer; graph-as-input
and message-passing controls; edge-noise sweeps and typed directed-distance biases;
then interpreter data/call graphs and soft grounding; pretrained adaptation; broad
compute/model/data scaling. The first pilot tests correctness and initial learning
behavior, not the entire research hypothesis.

## Plan review status

Self-review: interfaces are consistent; every first-pilot task has ownership,
validation, and a commit boundary. The broader design's unimplemented requirements
are listed under deferred stages rather than silently claimed covered.
User approved this plan and selected subagents. All four tasks are complete.
Final independent review passed with no outstanding findings; see pilot-report.md
and journal.md for results, validation, and the explicitly deferred next stages.
