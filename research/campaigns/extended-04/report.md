# Extended-04 report: metacognitive control, rational backtracking, and the value of structure

**Status: FINAL** (2026-09-26T22:50Z; §2 composition figures and §6 horizon wording corrected after external review). Every promoted or confirmed reading was reproduced from raw rows by independent auditors ([B1](review/b1-independent-audit.md), [A2/C/A1](review/ac-independent-audit.md), [F/B2](review/f-independent-audit.md)), and their corrections are applied here. The campaign ran on the pro6000, starting 2026-09-26T18:43Z.

**Sources:**
- Design: [design.md](design.md), v1 plus the review-driven v2 ([review/design-review.md](review/design-review.md)).
- Protocols: [A1](protocol-A1.md), [A2](protocol-A2.md), [A3](protocol-A3.md), [B1](protocol-B1.md), [B2](protocol-B2.md), [C1](protocol-C1.md), [F](protocol-F.md), all registered before their runs. A3 and F1b are adaptive and labelled as such.
- Decisions: [decisions.md](decisions.md).
- Ledger: [ledger-supplied-learned.md](ledger-supplied-learned.md).

## 1. Deliverables (brief §24)

| # | Deliverable | Outcome |
|---|---|---|
| 1 | Improvement over imitation, or a sharply localized reason for its absence | **No deployable improvement; the reason is localized** (§3). On the X1-r2 lineage, every stabilized RL variant (anchor, rehearsal, critic warm-up, entropy control, deployment-aligned masked sampling) and every on-policy teacher-supervision variant (anchored and unanchored) prevents the known greedy collapse. None exceeds the imitation bootstrap's best deployment mode. |
| 2 | A benchmark where unsuccessful initial attempts are rational | **Built and validated** ([probeworld](probeworld.md)). Its features: exact belief-conditioned values; labels that depend only on visible history (0 conflicts over 9,786 histories); timeout = unknown; irreversible commits; reuse-priced structure. It covers the four attempt cases (a)–(d), with direct-success episodes in every split. |
| 3 | A controller that recognizes stagnation and changes strategy | **Supplied rule confirmed; learned controller no better.** R-mask over the public progress diagnostic v1 cut stagnant steps by 73–86% at no utility or success cost on 3 fresh lineages and sealed worlds (F1b, adaptively registered after F1's mis-specified episode-level clause failed; a replication, not a blind test). The step reduction is near-mechanical, because the rule acts on the diagnostic that scores it. The finding is that utility and success do not fall, while per-episode cost rises slightly (+.0003 to +.0015). The rule also rescues collapsed RL policies (A1: success .00–.08 → .87–.92). A *learned* metacontroller did not beat this rule and harmed two of three loop-prone bases (C-H1). |
| 4 | Calibrated predictions of a small useful set of metacognitive variables | **In-distribution only.** The depworld appraisal P(success) has ECE .009–.026 on fresh worlds, and it predicts that "stop" is harmful. The *within-intervention* predicted-vs-actual effect correlation is weak (ρ .18–.27), and the value of a sampled step is not predictable. The probeworld value head is calibrated in-distribution for L4 only. It fails on held-out reuse horizons and condition combinations, and passes on held-out prices in 2/3 seeds. An own-greedy-return linear readout (B2) is no better. |
| 5 | Causal evidence that deliberate regulation beats automatic control and simple heuristics | **Not established.** Learned control lost to the best fixed rule on P1-RL r0/r1 (−.017, −.023) and gained +.018 on r2, mostly by stopping early and below the registered .02. The r2 gain depends on trained appraisal (the shuffled-target controller collapses), but that is one lineage below threshold. On competent bases the controller was essentially the default, so C-H2/C-H3 passed degenerately. |
| 6 | A measured response to computation prices and structure reuse | **Yes, in probeworld (confirmed),** as a choice of *when to purchase a supplied build/use operation*. This is not discovery or construction of a new representation. The optimal build decision switches with ρ_k and reuse horizon k (exact labels). Trained policies track it within .016–.045 of the optimal build rate on held-out prices and horizons (B1 plus F2 on fresh seeds and outcome draws). |
| 7 | An audited account of supplied, learned and generalized | [ledger-supplied-learned.md](ledger-supplied-learned.md), plus four independent audits and an internal design review. Summary in §6. |

**It is acceptable to finish with an informative negative result, and deliverables 1 and 5 are negative.** No claim of learned metacognitive control rests on a descriptive probe.

## 2. Track B: rational probing, switching and the value of structure (strongest positive)

- **Ladder** (B1 on 3 seeds; F2 on 3 fresh seeds with fresh outcome draws of B1's held-out configurations; 512 episodes per split):

  | Supervision | B1 held-out regret | F2 held-out regret |
  |---|---:|---:|
  | Outcome-only actor-critic (L0) | 25–39 | 39–82 |
  | Action-set imitation (L1) | 2.6–3.5 | 1.9–2.6 |
  | L4 | 2.1–4.1 | 1.5–2.6 |

  For scale: V* ≈ 230–340 per episode, π* ≈ 0, fixed rules 99–127. L0 gets stuck on a "never probe" rule: propagate → exact_b1 → commit.
- **Which supervision matters.** Only L0 → L1 (action-set imitation) is beyond noise. Stage/dependency, switch/rollback and Q* supervision (L2–L4) add nothing measurable.
- **Rational probing and justified switching hold when held-out splits are pooled** (F2 confirmed):
  - probing when not optimal: ≤ .01;
  - probing when uniquely optimal: .83–.86;
  - unjustified switches: ≤ .05.
- **Composition limit (replicated, with different magnitudes).** On held-out *combinations of conditions*, the uniquely-optimal probe rate stays below the .80 threshold in all 6 seeds: .636/.636/.773 in B1 seeds 0–2 (B1 audit) and .545 (48/88) in F2 seeds 3–5. This is a **probing** limit: once a probe fails, switching is .90–.98 of π* (F2). *(Corrected 2026-09-26 after external review: an earlier version said ".545 in all 6 seeds".)* B1 and F2 are distinct experiments. F2 used new training seeds and new outcome draws on **the same 128 held-out configurations** as B1, so it is not new configuration-level transfer.
- **Supplied vs learned.** The environment, prior and exact labels are supplied; the labels are training targets only. Belief tracking, when to probe, switch or build, and cost sensitivity are learned from public inputs; leakage was checked on 2,525 steps.

## 3. Track A: deployment and the absence of improvement (depworld)

- **A1 deployment matrix** (fresh worlds, 7 frozen policies):
  - The recovery rules rescue collapsed P1-RL policies, though their utility (.77–.83) stays below the bootstrap's (.86–.87).
  - R-mask does no harm to competent policies.
  - Sampling is the worst mode for every competent policy (−.06 to −.07).
  - No learned policy reaches the dep_reuse teacher (.90–.91).
- **Improvement screens.** IID group, lineage X1-r2, fresh screening worlds; the registered promotion rule promoted nothing:

  | Arm | Greedy | R-mask | Sampled | Work/success |
  |---|---:|---:|---:|---:|
  | bootstrap | .857 | **.861** | .795 | 143 |
  | anchor (C1) | .851 | .851 | .828 | 179 |
  | + rehearsal | .849 | .848 | .844 | 155 |
  | imitation-only, anchored | .851 | .850 | .842 | 145 |
  | critic warm-up | .845 | .856 | .816 | 164 |
  | entropy control, no anchor | .806 | .817 | .829 | 334 |
  | masked-sampling RL | .859 | .858 | .824 | 164 |
  | A3 unanchored on-policy imitation (190M worlds; bootstrap there .873) | .858 | .857 | .845 | 131 |

- **Localized reason** (audit-corrected):
  - **(i) Collapse prevention.** Every variant prevents the greedy collapse, but none exceeds the bootstrap's best mode. The best, masked-sampling RL greedy, is +.002 ± .003 over bootstrap greedy.
  - **(ii) The sampled-mode gain is not RL's.** On-policy teacher supervision without RL gives the largest sampled gain (+.047 to +.050), and the RL arms give +.022 to +.035. What these procedures mainly do is **sharpen the policy**, closing the sampled–greedy gap. They do not move the argmax behaviour above imitation. Removing the anchor (A3) does not help.
  - **(iii) Anchor vs entropy.** Entropy control prevents collapse at 1.87× the anchored arm's work per success (CI 1.47–2.28), so the P2a anchor effect is not just entropy control.
  - **(iv) Cost.** Every RL-on arm spends more work per success.
  - **Conclusion.** With this reward, recipe, 1,800-update horizon and one lineage, neither actor-critic nor on-policy teacher supervision produced a deployable improvement over behaviour cloning. The ~.03–.04 gap to the dep_reuse teacher stays open. A plausible next lever is an objective that credits the *deployed* (argmax/recovery) behaviour, or a stronger or different teacher signal. Neither was tested here.

## 4. Track C: learned appraisal and control over frozen policies

- **Setup.**
  - Telemetry: 89 immutable public features.
  - Appraisal: a 30k-parameter GRU, charged as compute.
  - Labels: branch-evaluation regression targets (an oracle-simulation privilege used for training only).
  - Default and continuation: greedy + R-mask. The controller is a one-step rollout improvement over it.
  - Controls: random interventions at a matched rate, and an automatic threshold rule.
  - Causal tests: telemetry-group removal, shuffled targets, disabled pathway, clamped interventions.
- **Result.**
  - Calibration of outcome is good; intervention-effect prediction is weak within type.
  - There was little one-step headroom over the default (dev gains ≈ 0 except P1-RL r2).
  - Learned control did not beat fixed rules, and on two loop-prone bases it was worse.
- **Registered localization:** calibrated appraisal, but **no actionable headroom over simple public rules**.
- **Why this is informative for the brief.** The public stagnation rule already captures most of the recoverable value at this action granularity. A controller that only chooses among one-step deviations from greedy + R-mask has little to add. Useful learned regulation would need interventions with larger consequences, such as budget allocation, candidate switching or revision scope, or bases with more headroom.

## 5. Integrity, accounting and deviations

- **Spend (metered).** CPU 119.1k / 172.8k core-s; GPU occupancy 11.6k / 43.2k s. The 20% reserve was not needed and ~54k core-s is left unspent. Remote development, tests and audits were metered (5.5k core-s). Local no-torch work is estimated at ~1.4k core-s (probeworld builder, auditors).
- **Overrun.** The Track C evaluation used 45.5k core-s against a 14k estimate (3.25×), the largest single item.
- **Throughput.** The Phase A fast path is 1.7–2.0× faster and bit-identical to the reference (308 tests on both paths). The GPU is not the bottleneck.
- **Deviations and incidents** (all in decisions.md):
  - A pre-registration smoke touched 16 Track C evaluation worlds, so the range was moved from 160M to 161M.
  - 12 label launches went out with an empty snapshot SHA and never ran (charged). The launch helper now refuses a missing snapshot.
  - The B-H6 cause was misattributed twice (a "V* target", then a "continuation mismatch") before B2 and the audit localized it to held-out generalization.
  - F1's episode-level clause was mis-specified; F1b was registered adaptively on fresh worlds.
  - The A2 "sampled gain from RL" reading was withdrawn after the audit.
  - Decision timestamps from ~20:38Z onward were hand estimates that drifted. They are aligned to git commit times.
- **Worlds and seeds.** All seed ranges are disjoint and documented: 130/140/150/161/170/180/190M plus the probe/dev ranges. Sealed confirmation worlds (170M, 180M) were used once each.

## 6. Supplied / learned / generalized (summary)

- **Supplied:** worlds, solvers, validators, applicability relations, teachers, the progress diagnostic and recovery rules, telemetry, oracle labels (training only), and exact DP labels.
- **Learned:**
  - depworld action policies (imitation);
  - probeworld belief-dependent probing, switching and building;
  - depworld appraisal of success and of stop/mask-top effects.
- **Generalized:**
  - probeworld to a held-out price region and an **unseen intermediate** reuse horizon (k = 4, with training on k ∈ {1, 2, 8}; interpolation, not out-of-range extrapolation), but not to held-out condition combinations;
  - the depworld recovery rule's no-harm property to fresh lineages and sealed worlds;
  - appraisal calibration to fresh worlds of the training distribution, but not under generator shift.

## 7. Recommended next steps (not started)

1. **Improvement objective.** Test an objective that credits deployed behaviour, such as greedy/recovery-rule rollouts in the policy-improvement target or advantage estimated under the deployment procedure, against behaviour cloning. Use ≥ 3 lineages from the start.
2. **Metacontrol with headroom.** Move learned control to decisions with larger consequences: solver budget choice, candidate switching, revision scope, stopping. Use bases and conditions where fixed rules leave measurable headroom; estimate power from an A1-style matrix first.
3. **Composition.** The probeworld held-out-combination probing failure is the clearest generalization gap. Test training exposure to condition pairs versus structural inputs that factor the conditions.
4. **Calibration under shift.** The value heads over-predict at unseen reuse horizons. Test distribution-robust targets or explicit uncertainty inputs before using appraisal to drive control under shift.
5. **Budget planning.** Estimate evaluation costs from the actual per-arm, per-condition job count; the Track C 3.25× overrun came from arm multiplicity.
