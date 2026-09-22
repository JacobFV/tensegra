# Topoformer: programmable structure in attention

Status: user approved the first implementation plan and subagent execution. The
attention core, generators, and runner are implemented and reviewed. The first
three-seed pilot is complete; report/final review are in progress. Broader research
stages remain future work.

## Objective and sequence

The end goal is **programming the metric space**: make retrieval geometry a
programmable interface, and test whether it can improve compositional
generalization while preserving learned subsymbolic computation. State-dependent
soft grounding is central to that eventual hypothesis. Intermediate experiments
are independently important; do not skip them to start with interpreter execution.
Potential biases, relation-specific compatibility, hard communication topology,
and constraints on admissible representations are progressively stronger forms
of this interface. The last is a research direction, not a first-stage deliverable.

Test whether externally supplied symbolic topology improves learning efficiency,
generalization, and compute efficiency by biasing differentiable attention.
First establish a general attention interface and controlled synthetic time-series
experiments. Next test robot morphology and other graph tasks. Then introduce a
minimal interpreter whose data and execution structures guide decoder attention.
Study both training from scratch and adapting pretrained models, followed by
scaling experiments. A negative result is a valid research outcome.

The first implementation milestone is the attention core, reproducible time-series
benchmark, ablations, tests, and a measured pilot report. Later stages depend on
understanding this milestone, as requested; they are not implied completed by it.

## Approaches and recommendation

1. Additive directed logit bias: simple, inspectable, and exactly reducible to
   ordinary attention. Start here to isolate the effect of supplied structure.
2. Latent-to-node association followed by graph routing: allows arbitrary token
   latents to identify with runtime nodes and change their retrieval preferences
   across layers. Implement after the explicit-node benchmark is validated.
3. Relation-specific query/key transforms: richer geometry, but more parameters
   and harder attribution. Keep as a later ablation rather than the initial core.

Hard graph masking is a comparison, not the default: useful dependencies may be
missing from the supplied graph and content retrieval should remain possible.

## Attention contract

Use query-row, key-column convention throughout:

    logits[b,h,i,j] = dot(Q[b,h,i], K[b,h,j]) / sqrt(d)
                      + sum_r lambda[layer,h,r] * B[b,r,i,j]
                      + causal_and_padding_mask[b,1,i,j]

An edge i -> j means that query i may preferentially READ key j. This reverses
the usual input-to-output causal drawing: if x1 depends on x3, the read graph
contains x1 -> x3. Document and test this distinction.

Support rectangular query/key lengths, batched graphs, typed edges, fixed or
learned strengths, absent graphs, and optional attention diagnostics. Start with
a dense PyTorch implementation as the correctness reference. Validate shapes,
finite biases, and valid attention rows. Structural bias never unmasks a causal
or padded position. Missing structure and zero strength recover the same unbiased
model numerically. Use unconstrained zero-initialized gates when exact pretrained
equivalence is required; a positive-only gate cannot initialize exactly at zero
without special treatment.

Initially expose adjacency and bounded directed-distance bias providers. Unreachable
pairs receive zero structural bonus, not an accidental maximum-distance bonus.
Scope distance and object containment are different relation types.

## First benchmark: dynamical systems

Generate stable nonlinear lagged processes:

    x_i[t+1] = rho_i*x_i[t] + c_i*tanh(sum_j W[i,j]*x_j[t] + b_i)
               + noise_i[t]

Normalize recurrent coefficients to keep trajectories bounded and record every
generator parameter. Dependency support includes the self term when present.
Support arbitrary subsets, including self-dependencies, cycles, and empty sets.
This benchmark uses prior-time inputs: cycles in the variable graph are valid,
while the time-unrolled dependency graph is acyclic. Same-time algebraic cycles
require separate solving semantics and are outside the first generator.

Use tokens for observed variable/time pairs and target-variable query tokens.
Target values are never included in the input. Shared variable encoders avoid
making a fixed variable count part of the architecture. Encode variable identity
consistently within each episode; permute identities jointly with graph and data
to test equivariance and expose shortcuts. First evaluate one-step prediction,
then autoregressive rollouts that feed predictions rather than future truth.

Two separate settings are necessary:

- Fixed process: independently sampled trajectories split into train, validation,
  and test. This measures learning with known versus hidden dependencies.
- Family of processes: context trajectories identify each system; held-out graphs
  and coefficients evaluate transfer. Supply the same observable context to all
  models. Do not expect prediction of arbitrary unseen coefficients from graph
  topology alone.

Controls: absent structure, true structure, reversed structure, degree-preserving
rewired structure where feasible, missing/noisy edges, and hard masking. Keep
initial weights, examples, optimizer, training budget, and seeds paired. Add a
graph-as-input baseline to distinguish access to extra information from the value
of imposing that information through attention. Compare persistence, a small MLP,
and a graph message-passing predictor as calibration baselines.

Select lambda and other hyperparameters only on validation data. Report per-seed
MSE, normalized MSE with training-only normalization, rollout errors, learning
curves, time and examples to a predeclared error threshold, parameter counts,
wall time, and peak memory. Report failures to reach thresholds as censored, not
as fabricated crossing times. Start with at least three paired seeds for the
pilot; expand before strong claims. Attention edge mass is diagnostic, not proof
of correct reasoning.

Generalization axes: independent trajectories, coefficient shift, new graphs,
larger variable counts, longer rollout horizons, and changed sparsity. Label each
axis separately; avoid a single ambiguous 'OOD' score.

## Morphology and further domains

The same interface biases joint queries toward joint-state keys with lambda*E.
Begin with synthetic coupled-joint dynamics and held-out chain/tree morphologies.
This is a morphology-conditioned prediction proxy, not evidence of a working VLA.
A subsequent VLA integration must identify actual action/joint token spans and
evaluate language, vision, action prediction, and closed-loop performance.
Physical contact can create dependencies beyond rigid-body adjacency; test added
contact edges and missing edges rather than treating morphology as complete.

For the first robot proxy, explicitly compare soft bias and strict local masking.
A positive adjacency bonus does not prohibit a finger querying a toe. Strict masks
permit only self and neighboring joints within one attention layer; distant
information can travel through multiple layers. Keep feed-forward modules tokenwise
and avoid global pooling or cross-token normalization that would bypass that
constraint. Test perturbation locality through multiple layers. This is a controlled
local-interaction abstraction: real rigid-body dynamics can exhibit global coupling
through mechanical constraints, and graph hops are not physical propagation delays.

Other useful tasks: electrical-network dynamics, traffic-flow forecasting,
multi-agent communication, dependency-aware build scheduling, and spreadsheet
formula evaluation. The latter connects naturally to interpreter execution.

## Runtime graph and minimal interpreter

Use a small explicitly parsed language with literals, lexical let bindings,
records, arrays, field/index reads, pure functions, calls, and conditionals.
Evaluate a controlled grammar; never execute generated host-language code.
Enforce step, depth, and allocation limits. Extend to bounded recursion and
iteration only after pure expression semantics are tested.

Separate identities for bindings, values, scopes, function definitions, and call
instances. Each invocation receives a fresh ID even when its textual expression
is identical. A call has typed edges to its callee and ordered argument positions.
An unordered record representation alone loses repeated arguments and order:
subtract(a,b) must differ from subtract(b,a). Nested call arguments point to call
instances, not merely to function definitions. Return edges become available only
after execution produces the result.

Containment and versioned execution provenance can be DAGs. General mutable object
graphs can have cycles and aliases; do not claim all runtime graphs are DAGs.
Initially use immutable values and lexical scopes with explicit alias identity.
Add state versions if mutation is introduced later.

For car1.wheels[3] = wheel_003, represent car1 -> wheels collection -> element
slots -> wheel values with distinct field/index relations. Names may initialize
embeddings, but stable node IDs and lexical scope resolve identity. Test renamed
variables, repeated names in separate scopes, aliases, and unseen names.

## Differentiable latent routing

Let E contain runtime-node embeddings and A_r their typed read adjacency. At each
layer, compute separate query and key association distributions over graph nodes:

    Pq = softmax(project_q(Z) * project_node_q(E)^T / temperature)
    Pk = softmax(project_k(Z) * project_node_k(E)^T / temperature)
    B_r = Pq * A_r * Pk^T

This gives a directed differentiable bonus when a query identifies with a parent
and a key identifies with one of its children. Recomputing associations every layer
allows a latent that shifts from car1 to its wheels collection to favor wheel
elements on the next layer. This is a hypothesis to measure, not an automatic
guarantee of traversal or execution. Include a null association to avoid forcing
unrelated language tokens onto graph nodes. Compare frozen string initialization,
learned embeddings, and exact identity associations as an oracle control.

Start dense on small graphs; measure its memory and compute overhead. Factorized
or sparse routing can follow once dense equivalence is established. Directed,
typed retrieval need not correspond to a symmetric Euclidean distance metric.

## Leakage and algorithmic evaluation

Distinguish static syntax structure, currently available runtime state, and oracle
execution traces. Only the first two are permitted in deployable evaluation.
Graphs must not expose future return values, unexecuted branch outcomes, target
tokens, or future state. Charge interpreter work in end-to-end compute reporting.
Report oracle traces separately: giving solved traces may outsource the algorithm
rather than improve transformer computation.

Evaluate nested field lookup, expression evaluation, composition, list operations,
and later sorting/search against exact interpreter or algorithm outputs. Hold out
depth, length, value ranges, and program templates. Measure exact answers, execution
steps where applicable, and abstention/calibration only where the model actually
produces confidence estimates. Include ordinary transformer and algorithm baselines.

## Scratch training, pretrained adaptation, and scaling

Train small decoder transformers from scratch using identical corpora and paired
budgets. Separately adapt a pinned small pretrained decoder through an explicitly
supported attention adapter. Verify zero-bias logits and cached incremental decoding
against the original model before training. Compare bias-only adaptation, adapters,
and full fine-tuning with equivalent unbiased controls. Choose the checkpoint after
checking remote software compatibility and available resources.

Sweep model size, training examples/tokens, and graph complexity independently.
Record actual compute including graph construction/routing; equal training steps
alone are insufficient. Fit empirical power-law models only after a sufficiently
broad sweep with multiple seeds, report uncertainty and fit residuals, and avoid
claiming a new scaling law from a tiny pilot or extrapolating beyond observations.

## Implementation organization and validation

User requirement: clean hierarchical design, reusable abstractions, minimal code,
and clear logging and development notes. Organize by responsibility: attention
kernel, graph construction, data generation, predictor, training/evaluation, and
thin experiment orchestration. Both domains share the same predictor and training
path. Prefer small functions and explicit tensor contracts over extensible base
classes, plugin registries, or frameworks before a second real use requires them.
Emit concise structured progress records with domain, seed, mode, step, loss, and
elapsed time; keep scientific metrics machine-readable and decisions in this
directory. Validate at public boundaries without repeating checks in every helper.

Planned modules: attention core; graph/bias providers; synthetic datasets; prediction
models; experiment runner and aggregation; then interpreter, runtime routing, and
pretrained adapters. Keep experiments configured and seedable, write raw metrics
and manifests with git revision, environment, graph generator, and configuration.
Commit concise reports and small result files; exclude environments and checkpoints.

Required first-milestone checks: directed orientation, zero-bias equivalence,
positive edge bonus, masks, rectangular/batched shapes, gradient flow, permutation
equivariance, dependency support, reproducible disjoint trajectory splits, no target
leakage, and a small end-to-end training run. A successful milestone means reliable
comparison machinery and an honest report, whether or not bias improves error.

Use the local machine for lightweight edits/checks with bounded CPU threads. Run
training on gb10-direct, one pilot job initially, in an isolated environment with
timeouts and measured memory. Inspect active jobs and compatibility before launching.
No large local allocations: inspection found local swap essentially exhausted.

## Related work and research positioning

Graphormer already adds structural encodings to attention logits:
https://papers.neurips.cc/paper_files/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf

Shaw et al. describe relation-aware attention generalizable to graph-labeled inputs:
https://research.google/pubs/self-attention-with-relative-position-representations/

Therefore additive graph bias alone is not a novelty claim. The research questions
here concern runtime-dependent symbolic routing, latent association across layers,
execution semantics, controlled generalization, and the cost/benefit across training
regimes. A broader literature review is needed before making novelty claims.
