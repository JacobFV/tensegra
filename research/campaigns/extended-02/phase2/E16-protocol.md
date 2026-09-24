# E16: composition across primitives in the modular workshop (pre-registration)

## Question
Does a learned controller that acquires stage-level resource policies on some ordered compositions of three primitive families execute *unseen orderings and longer compositions* of the same primitives, without a supplied schedule?

## Environment (`workshop-modular-v1`, `src/topoformer/campaign02_modular.py`)
The public goal is an ordered tuple of distinct stages:
- **select**: choose one item per category under capacity/funds/incompatibility. Direct greedy commit, or a `constrained_subset` draft → constraints → budgeted call → retrieve → use.
- **route**: reach the destination. Greedy moves, or a `shortest_path` draft → call → use.
- **assign**: give each of 5 tasks a slot from its public domain, avoiding public forbidden value pairs (visible after `inspect roster`). Greedy slot choice + commit, or a `csp` draft → constraint → budgeted call → use.

Stage-completing actions are accepted only while their stage is current. Everything else (inspections, drafts, calls) is allowed at any time. Verification succeeds only when all stages are complete.

**Supplied:** exact solvers, typed statuses, draft builders, validators, and public m1 features. The features mark each candidate's stage and whether it is the current stage. This is a supplied stage-typing of actions, disclosed.

**Learned:** which action to take, when to compute, which budget to request, and when to verify.

Profiled leverage (assign 5×3): the no-tool greedy solves ~47% and the cheap-first teacher 100%. A 6×4 assignment region is budget-bound for every method and is reported separately (`hardassign_*`).

## Splits (fixed before training; `research/tools/campaign02_e16_config.py`)
- **Train mixture:** singles S, R, A and ordered pairs S→R, S→A, A→R (pairs weighted ×2). In training, route is never followed by another stage and select is never preceded by one.
- **Held-out ordered pairs:** R→S, A→S, R→A. These are new orders of familiar primitives, including stage positions never seen for route and select.
- **Held-out length:** all six 3-stage orders.
- **Budget/size shift:** hard assign (6 tasks × 4 slots), alone and after select.

## Training (per lineage, three fresh lineages)
1. Supervised bootstrap: 600 updates from the public `modular_cheap_first` teacher (privileged training supervision) on the train mixture.
2. Single-lineage actor-critic v2: the E08 single recipe (lr 3e-5, entropy .003, KL .3; 5 rounds × 6 slots × 60 updates = 1,800), on the same train mixture, decision cap 64.

The latest checkpoint is final. There is no selection on any held-out split.

## Evaluation
Frozen bootstrap and RL endpoints of all three lineages, plus references (`modular_cheap`, `modular_always_tool`, `modular_cheap_first`), on a sealed seed range (90,000,000+) with 256 worlds per condition.

## Endpoints
- **Primary: the composition gap.** Mean success on held-out pairs minus mean success on IID pairs, per lineage, for the RL endpoint.
- **Secondary:**
  - triple success;
  - stage-completion fraction;
  - utility relative to the teacher;
  - the solver/greedy mix per stage;
  - error localization for failures (stage-order rejections, premature verify, wrong-stage actions, budget failures).

## Decision rule
- **Compositional transfer supported:** in ≥2/3 lineages, held-out-pair success ≥ 0.9 × IID-pair success **and** triple success ≥ 0.8 × IID-pair success.
- **Failure:** held-out-pair success < 0.5 × IID in ≥2/3 lineages.
- **Otherwise:** partial.

The teacher solving all orders shows the orders are solvable with public information, but it is a supplied modular program. Its success is not evidence of learned composition.
