# Extended-07: from learned factors to compositional decisions

**Status: COMPLETE** (closed 2026-09-27; see [report.md](report.md), [handoff.md](handoff.md)). Adopted from the user's brief on 2026-09-27T16:02:54Z (window start; deadline 2026-09-28T04:02:54Z, 12 h).
- **Ceilings:** 24 CPU core-h (86,400 core-s), 6 GPU-device-h, ~20% reserve.
- **Diagnostic tranche:** at most 3 CPU core-h, on frozen checkpoints, before any substantial new training.
- **Base:** main at 40154586 (extended-06 merged).

**Sources:** design [design.md](design.md), registry [registry.json](registry.json), decisions [decisions.md](decisions.md), job plan [jobplan.json](jobplan.json), budget [budget.json](budget.json), queue [queue.json](queue.json), seed ranges [seed-ranges.json](seed-ranges.json).

## Mission
Locate where the learned factor-to-decision pipeline fails. The candidates are:
- predicting the quantities;
- preserving their meaning;
- consuming them;
- learning the downstream interaction;
- the quantities themselves.

Then test the smallest intervention that the diagnosis justifies.
- **Strong result:** a learned intermediate representation transfers to an unseen combination, and a downstream computation uses it correctly on the decisions where the interacting factors matter.
- A **well-localized negative result** is also useful.

## Scope
- **Closed for this campaign:** portworld selection (Track A of extended-06), populations, teacher delegation, brain modeling, large pretrained models, new agentic environments.
- **Structural attention** is conditionally authorized. It requires the preconditions of brief §16: quantities available under a clear contract, a remaining downstream interaction problem, and strong ordinary alternatives already established.

## Empirical boundary inherited from extended-06
These findings are established:
- a small later-decision gain for LRN over RAWF, with fewer redundant inspections;
- a borderline total-regret gain;
- worse performance on counterfactual flips;
- better flips when the derived quantities are supplied (SUP).

These are **not** established:
- held-out factor accuracy;
- mediation through the factor-reading path;
- sufficiency of the representation;
- irreducible third-order interaction;
- a unique role for attention.

Historical flips mostly contrast the full combination against the single factors; flips relative to all pairs are generally absent.

The historical LRN already feeds **stop-gradient** predictions to the policy. The auxiliary loss still trains the shared trunk and encoder.

## Guards
- **Launch helper** (`campaign07_remote.py`). It refuses:
  - conflict markers;
  - a missing snapshot;
  - job names that don't start with `e07-`;
  - missing `--needs` inputs;
  - outputs outside the campaign07 results directory (historical roots are read-only);
  - a repeated job name;
  - more than 4 concurrent jobs by default, or 8 with profiling (never above 8);
  - launching when less than 16 GiB of memory is free.

  It also creates output directories before a job starts.
- **Procedure:**
  - one-job smoke before every new batch type;
  - staged artifacts must be hash-verified;
  - timestamps from receipts or git only;
  - every failed job is charged and recorded.
- **Statistics:**
  - ≥ 5 paired lineages for primary learned contrasts;
  - ≥ 20,000 bootstrap draws with a pre-fixed RNG seed;
  - two-level (seed × shared configuration) intervals that keep whole octets;
  - Monte Carlo resolution checked for borderline endpoints.
