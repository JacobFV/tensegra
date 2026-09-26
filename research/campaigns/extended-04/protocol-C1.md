# Protocol C1: learned metacognitive control over frozen depworld policies

Registered 2026-09-26T20:30Z, before any Track C label, training or evaluation run on registered worlds. Spec: [trackc.md](trackc.md); design v2 revisions 5–8.

## Design (as built)
- **Bases (frozen).**
  - Loop-prone under greedy: P1 RL finals X1 r0–r2.
  - Competent: X1 bootstraps r0–r2.
- **Default and continuation:** greedy + R-mask. Interventions: sample-step, mask-top-step, stop. The controller is a one-step rollout improvement over the default.
- **Labels:** branch-evaluated regression targets only (ΔU; paired advantages; K = 4 for sample-step), on label worlds 150,000,000+.
- **Appraisal:** a 64-unit GRU (30,150 parameters) over telemetry (89 features, 8 groups). Actor features enter without gradients.
- **Registered settings:** the margin m and threshold τ are chosen on a 20% development split of the label worlds.
- **Evaluation worlds:** **161,000,000 + 100,000·i**, 256 per condition over iid_f0, iid_f2, events_train_kinds_p1 and foreign4. The IID group has 512 worlds.
- **Metacontroller seeds:** 3 per base. Lineage-level values are the mean over the 3 seeds, and per-seed values are reported.
- **Charging:** the metacontroller forward is charged in utility (with-charge is primary; without-charge is reported). For parity, the rule arms' diagnostic CPU is reported alongside.

## Arms
- Fixed: greedy, sampled, **greedy + R-mask (default)**, R-sample.
- Appraisal-only: control pathway disabled; behaviour equals the default.
- **Learned control.**
- **Random interventions at the learned controller's matched rate.**
- **Automatic threshold rule:** sample-step when predicted success < τ.

## Decision rules (IID group, per base lineage)

| ID | Claim | Supported if |
|---|---|---|
| C-H1 (loop-prone) | Learned control improves on the best fixed rule | In ≥ 2/3 P1-RL lineages, learned utility ≥ max(fixed arms) + .02, and success ≥ default − .02 |
| C-H2 (competent) | Learned control does no harm and trims waste | In ≥ 2/3 bootstrap lineages, learned utility ≥ default − .01, **and** total cost or no-progress episodes are below the default's |
| C-H3 (deliberate > automatic) | Deliberate regulation beats automatic and random control | Within each base type, learned utility > both matched-rate random **and** the threshold rule in ≥ 2/3 lineages |
| C-H4 (calibration) | Appraisal predicts outcomes | Reliability error of P(success) ≤ .05 on evaluation worlds (≥ 2/3 lineages per base type), and branch-evaluated predicted vs actual intervention effect is positively correlated (Spearman > 0 with a 95% bootstrap CI excluding 0) |
| C-H5 (causal use) | The learned gains depend on learned appraisal | The shuffled-target controller loses ≥ 50% of the learned-vs-default utility gain, and appraisal-only equals the default. Telemetry-group ablations are descriptive. |

- **Transfer check:** events_train_kinds_p1 and foreign4 are reported as secondary conditions, with the absolute floor success ≥ .8 × the default's.
- **Scope of any claim:** C-H1…C-H5 would establish a supplied-telemetry, learned-appraisal controller over a frozen policy. That does not include learned representations of self-state. Failure of C-H1/C-H2 with positive C-H4 would localize the result as "calibrated appraisal, but no actionable headroom over simple rules".

## Budget
~24k core-s (labels ≤ 12k; training 1.3k; evaluation ~14k), CPU-only. The third label chunk per base runs only if total label CPU is below 8k after two chunks.
