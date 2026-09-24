# E08 — population search versus matched non-evolutionary search (pre-registration)

Registered before any E08 main-run outcome exists. Design profiles `e08-profile-pbt` (actor-critic v1 collapse) and `e08-rlprofile-pbt` (actor-critic v2 stability) are development-only and are not evidence for the comparison.

## Question
Given the same initial bank, world mixture, development panel and number of RL updates, does population-based training (selection + inheritance + mutation) yield a finalist with higher verified utility on sealed fresh worlds than (a) independent multistart with best-development selection and (b) one conventional learner receiving all updates?

## Units and independence
- Three independent replicate processes r ∈ {0,1,2}. Each has its own initialization seeds, supervised bootstrap bank, training streams, development panel, mutation RNG and address namespace.
- Within a replicate, the three modes share the bank, streams, panel and six initial hyperparameter rows (paired design).
- Descendants within a PBT run are not independent seeds. The replicate is the unit for replication claims. Sealed episodes are paired across modes, and episode-level intervals describe within-replicate precision only.

## Shared bootstrap bank (privileged supervision)
Per replicate, six members (lightweight, width 1024, legacy interface, public features v2) are trained on 600 supervised updates each (batch 8) from the public `cheap_first_fallback_v2` teacher on the E05 mixture. The teacher is a supplied heuristic that sees only public observations, and it is labelled as privileged training supervision. Cost is charged to `e08-bank-r*`. Single uses only bank member 0, while PBT and multistart use all six; the bank's full cost is reported against every mode.

## Search phase (what is compared)
Actor-critic v2 has the following settings. Decision-mean policy gradient with per-batch standardized advantages. A KL(round-start policy ‖ current) trust region on on-policy states. Critic weight 0.5, batch 8 sampled episodes, reward = incremental evaluator utility including the modeled 1e-4/decision compute tariff. The optimizer is reset at bank import.
- Rounds × updates-per-slot are fixed in the configs (`e08-main-*`). Every round has six slots and six development evaluations of 128 fixed development worlds for every mode.
- PBT: after each non-final round, the two highest-utility members replace the two lowest (strictly greater only). Weights and AdamW state are inherited. lr, entropy and KL weights mutate by ×0.8/×1.2 (bounded). Recipients keep their own data streams and ledgers.
- Multistart: the six members train independently. There is no replacement.
- Single: bank member 0 with hyperparameter row 0 receives all six slots of every round.
- Finalist: maximum final-round development utility (slot-index tie break). Single uses its latest checkpoint. There is no historical cherry-pick.

## Frozen sealed evaluation (E09)
Finalists are frozen. Each is evaluated once on sealed seeds 70,000,000+ (never used by training, selection or profiling) using `research/tools/campaign02_e09_config.py`: six IID mixture components, seven transfer conditions, and paired interventions. Supplied references (cheap, always_tool, cheap_first, cheap_first_fallback_v2) run on identical worlds with the same modeled per-decision tariff.

## Endpoints
- **Primary:** mean verified utility over the six IID sealed components (equal weight, 256 worlds each) per finalist. PBT advantage is supported only if PBT > multistart in all three replicate pairs **and** the pooled paired episode-level mean difference has a 95% bootstrap interval excluding zero. Otherwise the claim is not supported. Ties or mixed signs are reported as such. The same rule applies to PBT vs single.
- **Secondary:** success and coverage, cost per success, solver calls/escalations, transfer-condition utility, development-to-sealed gap, and actual CPU/GPU/wall per mode (equal updates ≠ equal FLOPs; PBT copying and evaluation overhead are included).
- **Descriptive:** lineage (parents, mutations, replacements), hyperparameter trajectories, member diversity.

## Interpretation limits
Three replicates cannot establish a general evolutionary advantage. A null or negative result is reported as such and triggers reallocation, not a larger population. RL beating the bootstrap is a separate claim from PBT beating multistart. Utility gains that come from reduced cost are reported separately from success gains.
