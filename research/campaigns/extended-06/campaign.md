# Extended-06: computational strategy selection, structural composition, and decision-relevant value learning

**Status: ACTIVE.** Adopted from the user's brief on 2026-09-27T05:51Z. Window: 05:51:06Z to 17:51:06Z (12 h). Ceilings: 48 CPU core-h, 12 GPU-h, ~20% reserve.
- Design: [design.md](design.md). Registry: [registry.json](registry.json). Job plan: [jobplan.json](jobplan.json). Decisions: [decisions.md](decisions.md). Seed ranges: [seed-ranges.json](seed-ranges.json).

## Tracks
- **A: computational portfolio / algorithm selection (primary).** Can learned strategy selection beat the best single method *and* a tuned simple portfolio on fresh instances, including compute cost?
- **B: clean compositional generalization (primary).** Does structural-combination diversity at **matched data volume** transfer to an **untouched, interaction-heavy** pair or triple, with balanced counterfactuals ruling out a global action-rate shift? Are factorized representations (supplied vs learned) useful?
- **C: value/metacontrol (conditional).** Only where A or B exposes headroom that simple equally informed rules miss.
- **Optional late branches:** structural attention (only with a clean relational interface); latent microstep trajectories.
- **Population evolution:** deferred.

## Historical claims preserved (not to be weakened)
See the brief §3. Extended-05 findings stand as written in its corrected report (58b231d6).

## Guards
- Launch helper (`campaign06_remote.py`):
  - refuses a missing snapshot, conflict markers and non-`e06-` job names;
  - requires a one-job smoke before any batch;
  - requires staged artifacts to be hash-verified.
- Timestamps come from `date` or git only.
- Every failed launch is charged and recorded; no silent reruns.
