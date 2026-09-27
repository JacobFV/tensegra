# Extended-05: decision-relevant metacognition and compositional computational control

**Status: COMPLETE**, adopted by the user 2026-09-27T01:26Z. **Report: [report.md](report.md).** Window: 2026-09-27T01:26:57Z to 2026-09-28T01:26:57Z. Design: [design.md](design.md). Registry: [registry.json](registry.json). Job plan: [jobplan.json](jobplan.json). Decisions: [decisions.md](decisions.md).

## Primary questions (from the user's brief)
1. Can **deployment-aware, cost-sensitive policy improvement** produce verified utility gains over competent imitation and simple recovery rules?
2. Can an agent generalize the **value of computational actions across unseen combinations** of familiar conditions, not only across new prices or single parameter values?

Metacognitive control is the mechanism under investigation, not an assumed capability.

**Two tracks:**
- **A:** deployed-policy improvement where intervention headroom has been measured.
- **B:** compositional value-of-computation generalization.

Calibration is a supporting question inside both. The structural-attention branch is conditional.

## Authority and limits
- **Autonomy.** Autonomous in-scope work.
- **New metered window:** 48 CPU core-h, 12 GPU-h, 24 h, ~20% reserve. These are ceilings, not targets. Extended-04's unused allowance is not carried over.
- **Host.** The pro6000 only, checked before use. The user has an unrelated workload there (an inpaint360gs env install); do not interfere.
- **No GB10s** without separate authorization.
- **Not in scope:** paid services, pretrained models, brain-derived architecture, population evolution.
- **Coordination.** One coordinator owns the queue.
- **Metering.** Everything is metered, including labels, audits and failed jobs. Local work is estimated.
- **Timestamps** come from git and receipts, never hand estimates.

## Headline the final report must answer
- Did we learn to choose better computations?
- Did that choice generalize across new combinations?
- Did the learned regulator add value beyond simple rules?

**Not substitutes:**
- more determinism, for better decisions;
- calibration, for control;
- buying a supplied operation, for discovering structure;
- a richer internal vocabulary, for demonstrated competence.
