# Extended-02 final report

**To:** Mother Agent
**Campaign:** topoformer extended-02, "evolving agents that learn when and how to use computation"
**Branch:** `campaign/extended-02` (origin). Historical baseline `65d44ae6` preserved.

**Status: closed at an evidential boundary.** No jobs are running on the GB10.

**Resources used:**

| Resource | Used | Ceiling |
|---|---:|---:|
| CPU | 44.4 core-h | 48 core-h |
| GPU (device occupancy, user-accepted convention) | 9.3 h | 12 h |

Failed and capped attempts are charged. Wall-clock time stayed within the 24-hour window.

## 1. Headline answer

> Can selection produce agents that obtain more verified problem-solving ability from the same computational resources, and does that ability transfer beyond the exact primitive combinations used during training?

**Selection: only in the narrow sense of allocating training depth.**
- **Copy-based evolution (PBT) never beat matched non-evolutionary search.** Pooled sealed utility vs independent multistart was −0.0097 [−0.012, −0.007] (E08). Adaptive curriculum (E12) and robustness-visible fitness (E13) did not help. E13 over-corrected into tool avoidance.
- **Successive halving helped.** Survivors inherit the eliminated members' training budget. It beat PBT and multistart in 3/3 replicates in point estimates (+0.015 / +0.005 utility, E15), and found the good hyperparameter configuration on its own. This is *efficient search under configuration uncertainty*, not an evolutionary-operator result. It did not produce the robust strategy its registered mechanism required, and it trailed a single learner that was handed the configuration.
- **No population design** (five tested) reliably produced the best behavioral strategy: try the direct path, then the solver, then escalate. Concentrated single-lineage training reached it in 5/7 exploratory and 2/3 fresh confirmation lineages (E14: partial).

**Transfer: yes for new resource conditions and new orders of known stage skills; not established beyond that.**
- **Resource conditions.** Direct-first controllers match the public teacher on unseen budget and price combinations and survive tool removal and corrupted returns (E09/E14).
- **New stage orders.** In a three-primitive modular world, controllers executed unseen stage orders and three-stage sequences (E16). The aggregate transfer criterion was met under the stated interpretation; conditionwise robustness remains incomplete.
- **What the stage-order result is.** This is **sequential composition of stage-level controller skills** under a public, environment-enforced stage order with supplied stage-typed features. It is not discovery of stage order, and not functional composition in which one solver's output parameterizes another's input. Functional composition remains untested.
- **A never-seen primitive** gave a large few-shot head start (20 updates: 0.67–0.98 vs 0.02–0.19 from scratch; unregistered, descriptive). That head start was not reliable at the registered 120-update endpoint (+0.02 / +0.46 / −0.19: partial), and one lineage lost retention.

## 2. Strongest mechanistic findings

1. **The return-binding shortcut: diagnosed, then repaired.**
   - **Diagnosis.** Controllers that were perfect on their training compositions failed whenever any irrelevant solver result was present. On the tested training pairs, success coincided exactly, world by world, with zero-distractor worlds (audited).
   - **Repair.** Training with wrong-type returns, with held-out orders still absent, raised assign→select from 0.57 to 0.94. Wrong-type return uses fell to zero in all audited lineages (E17).
   - **Reading.** The type-agreement feature was supplied; training made it load-bearing. The lesson: training must exercise an interface's discriminating conditions.
2. **Provenance was an interface limit, and the input fix works.**
   - **The limit.** An encoded-tensor counterfactual audit shows that m1 inputs are *bit-identical* for "own current return" vs "same-type foreign return" and vs "own stale-draft return". E19's collapse (0.35) is therefore not evidence about learnability.
   - **The fix.** m2 public provenance features, with exposure, gave 1.00 in 3/3 lineages (E20).
   - **Caveat.** m2 is close to an oracle given the current `prior_i` vs `problem_n` naming, and the sealed conditions share the training generator. E20 is IID robustness under a corrected input contract.
3. **Rejection perseveration is a memory gap.**
   - **The gap.** Under m1/m2, "rejected, then one other action" is encoded identically to "never tried". Only the m3 per-stage counters retain the history.
   - **The fix.** m3 removed rejected-commit loops: 126 → 0 failures over 22,272 sealed episodes per arm (E22).
   - **Caveats.** The evidence is concentrated in one lineage. A distinct `choose_item` loop remains in 5 of 22,272 episodes per arm. No feature version encodes the rejection *reason* (capacity vs funds).
4. **Architecture.** The lightweight controller beat the 12×-parameter recurrent workspace, which collapsed to never calling tools under RL (E11). This was a single seed per family.
5. **Input-contract repairs recur.** Phase 1's budget failure (57 → 119/128 with supplied relational features, E07), provenance (E20) and failure memory (E22) were each localized to missing encoded inputs. None of them is evidence of learned representation discovery.

## 3. Supplied vs learned

**Supplied:**
- exact solvers, typed statuses, validators and draft builders;
- the public cheap-first teacher, used as bootstrap supervision;
- relational, stage, provenance and failure-count features;
- the niche descriptor.

**Learned:**
- action and inspection choice;
- whether and when to compute;
- budget choice and escalation;
- return use;
- stage sequencing without a schedule.

**Absent:**
- autonomous planning of stage order;
- solver discovery;
- structural-attention advantage (untested);
- optimization-quality competence (the item objective is constant);
- confidence calibration.

## 4. Evidence quality and integrity

- **Independent audits.** Three independent raw-data audits (phase 2, phase 3, phase 3b) plus one external research audit. Every headline number was reproduced.
- **Corrections from the audits.** Wrong localization magnitudes were fixed. Two-reading gate outcomes are reported. A step-cap asymmetry in phase 2 is disclosed. A spec_hash drift was found; pairing is valid by seed plus verified content.
- **Sealed seeds.** Disjoint from over 1,500 training and development ranges.
- **Held-out integrity.** Held-out orders and primitives never appear in training mixtures.
- **Limits on independence.**
  - E17–E22 reuse sealed worlds after earlier results motivated them: these are paired interventions, not fresh confirmations.
  - RL lineage 0 shares its first 4,800 training worlds with bootstrap lineage 2.
  - The population experiments share three banks.
  - Gate aggregation was under-specified in E16/E17; both readings are reported.

## 5. Unresolved failure mechanisms

- The population-level loss of the direct-first mode under short-horizon selection. No tested selection design fixed it.
- Conditionwise composition robustness and lineage dependence.
- Provenance beyond the near-oracle naming.
- Old-but-still-valid result reuse.
- Rejection-reason blindness.
- The residual `choose_item` loop.
- The omitted semantic composition (RL01/RL02: 0/512; the pointer head produced duplicate edges instead).
- Functional (data-dependent) composition across primitives.

## 6. Recommended next steps (not started; ~3.5 CPU core-h remain)

The next steps follow the external audit's priorities:
1. **A frozen-policy provenance test on fresh worlds.** Use randomized draft-like names for foreign records, wrong-type controls, stale-inapplicable results and old-but-still-valid results, each validated independently. Report selection errors separately from final success.
2. **A deterministic identical-retry suppression diagnostic** on the E20-r2 lineage. This is a supplied progress monitor, used for localization only.
3. **Registered adaptation metrics** before any new adaptation study: area under the adaptation curve, updates to sustained competence, retention, and post-adaptation composition.
4. **A functional-composition task** (selected items → assignment domains → route timing constraints).
5. **Independent-bank, fresh-seed confirmation** of the depth-allocation result, run as a separately budgeted comparison.
6. **A versioned, content-based world comparison key** in the tooling.

## 7. Where to look

- **Reports:**
  - [campaign-synthesis.md](campaign-synthesis.md);
  - phase reports: [phase 1](campaign-report.md), [phase 2](campaign-report-phase2.md), [phase 3](campaign-report-phase3.md);
  - claim maps: [phase 2](claim-map-phase2.md), [phase 3](claim-map-phase3.md).
- **Audits:** [phase 2](review/phase2-independent-audit.md), [phase 3](review/phase3-independent-audit.md), [phase 3b](review/phase3b-independent-audit.md).
- **Encoded-input diagnostics:** [diagnostics/encoded-counterfactuals.md](diagnostics/encoded-counterfactuals.md).
- **Protocols:** `phase2/E08…E22`.
- **Records and handoff:** [decisions.md](decisions.md), [budget.json](budget.json), [handoff.md](handoff.md).
- **Raw episode rows and checkpoints:** `gb10-direct:~/topoformer-campaign02/results/`. Compact receipts, states and lineages are in `research/results/campaign-02/`.
