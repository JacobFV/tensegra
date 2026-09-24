# E05 repeated-budget failure: public-state diagnosis

Frozen source: `a9337b71`. This audit reads the final development trajectories
and recomputes only the public teacher on those archived observations. It does
not run a neural model, environment, or solver. Teacher decisions below are
counterfactual labels at **actor-visited states**, not training-label counts.

| Quantity | Lightweight | Recurrent |
|---|---:|---:|
| Episodes |128|128|
| Verified successes |57|46|
| Actor decisions |4,462|4,843|
| Calls, all with budget16 |1,962|2,491|
| Repeated calls to the same exact draft at the same budget |1,848|2,372|
| Distinct episodes containing these insufficient retries |66|86|
| States where the teacher instead chooses a larger call budget |1,261|1,321|

All repeated calls follow a timeout on that same unchanged instance, and none
allocates more work than the previous exhausted attempt. Repeated frames are
not independent examples. Some recurrent episodes with retries eventually
succeed, so the86 episodes must not be reported as86 failed episodes.

## What the input actually contains

For every observed escalation disagreement, the selected call and the teacher's
larger call have distinct neural candidate vectors. They differ at coordinate20,
the requested budget divided by4096. Coordinate52 correctly exposes the previous
call budget, also divided by4096. The public linked-record timeout flag and
call count are present, and observation state includes remaining work. Thus:

- budget16 is represented as0.00390625;
- budget128 is represented as0.03125;
- the previous failed budget is correctly represented as0.00390625;
- in the tighter states, remaining-budget alternatives112,96,80,64,48,32 are
  visibly distinct and the teacher selects them when appropriate.

Among escalation disagreements, **zero pairs have identical candidate features**.
Lightweight repeatedly chooses16 when the teacher chooses128 in1,221 states;
recurrent does so in1,274 states. The remaining40 and47 cases respectively
involve the smaller remaining-budget alternatives.

This is different from the separately proved incompatibility and downstream
edge collisions. No missing public relation is needed to distinguish these
specific same-instance16-versus-larger choices. This does not prove the full
teacher is realizable under the old encoder: those other collisions remain.
Nor does merely having unequal input vectors guarantee that optimization has
learned to use the distinction.

## Other counterfactual actions

When the actors choose a call, the teacher sometimes chooses a non-call action:
lightweight611 moves and4 abstentions; recurrent871 moves,127 subset builders,
100 item choices, and1 abstention. Those states involve more than choosing a
larger solver allocation and may depend on the already documented missing
relations or on earlier trajectory errors. Do not pool all deviations into a
single budget-classification failure.

## What is and is not known about acquisition

The candidate scorer receives budgets through an affine projection followed by
learned processing and layer normalization. Small numerical scaling may make
optimization harder, but it does not prove magnitude information is destroyed:
learned affine weights can rescale it. No activation, gradient, or decoder probe
was run here. Exact feature ties are ruled out for the observed escalation pairs;
normalization, weak gradients, class imbalance, and inadequate exposure remain
untested explanations rather than findings.

The inspected archives retain training curves and data hashes, not all teacher
training decisions. Actual training call-budget frequencies therefore cannot be
reconstructed from these development counterfactuals. A training-distribution
frequency claim requires exact teacher-stream replay or a separately declared
fresh training proxy, with generation and solver work charged. The long failure
loops greatly overrepresent escalation in the actor-state counts above.

A later loss-weighting comparison could isolate acquisition without changing the
old input contract: ordinary continued supervision versus explicitly weighted
call decisions, from the same checkpoint and on the same fresh stream. However,
weighting every call also increases the weight of initial budget16 labels. The
frequency of initial versus escalation targets should be measured before claiming
that weighting repairs class imbalance. No training intervention is implemented
or launched by this diagnostic.

Reproduce with `research/tools/campaign02_e05_budget_audit.py`. The compact receipt
binds frozen source bytes and both raw archives, stores exact frequencies and
examples, and records0.882 process-CPU seconds. Git subprocess CPU is not included
in that process-only number; no GPU, solver, or model inference was used.
