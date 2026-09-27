# Decisions (extended-05)

- 2026-09-27T01:26Z: **Brief adopted** (the user chose "adopt and start now").
  - New window 01:26:57Z → 2026-09-28T01:26:57Z; new root ~/structured-latent-dynamics-campaign05 with bin/job.py and metered.sh.
  - The pro6000 was checked: WSL up 6.5 h, keep-alive Running, GPU desktop-only. An unrelated user `uv pip install` (inpaint360gs env) is running and will not be disturbed.
  - The pro6000 directories were renamed by the user (tensegra-campaign0X → structured-latent-dynamics-campaign0X); tooling was repointed (7851b490).
  - Branch campaign/extended-05 from main 3a566812 (which includes the extended-04 report correction on the composition probe rates).
- 2026-09-27T01:29Z: Design v1, registry (7 entries) and machine-readable job plan (~88k core-s with 25% contingency; ~42k ungated) written. Internal review commissioned next. Timestamps for later entries come from git commits.
- 2026-09-27T01:44Z: Internal design review (3 BLOCKER, 11 MAJOR) adopted as **design v2**. Key changes: G1 counts no abstain gains and is a necessary condition only, plus a deployable cross-fitted single-deviation gate G1b ≥ .01; O(I) anchored to D's action kind; r3–r5 reserved for confirmation; the new probeworld split table (holds removed from train and dev, new seeds); exact-solver pool sizing (≥ 20 supporting configurations per cell); a B-HR headroom gate; BX3 modular arm; BO trained on the extended-04 split; learned-head evaluation priced at .35 core-s/episode. Both builders notified.
- 2026-09-27T02:06Z: **Track B tooling merged** (df55e56c; 39 tests; extended-04 paths bit-identical).
  - Split v2 audit: 18/18 pass.
  - B-LOC on 12 existing models: ~70% of L4 first errors on the challenge pairs are **value-estimate** errors (the Q-head argmax is also wrong); 57–65% occur at the first decision; the probing shortfall concentrates in U+E.
  - **The new holds are weak compositions:** ignoring one flag costs > ε in only 16% (U+C) and 12% (S+E) of configurations, vs 38% for U+E. The B-HR gate may therefore show no failure; that would be reported, not repaired.
  - BX2 is non-trivial but null at the first decision (belief = prior).
  - **The B-X non-inferiority regret tolerance is registered at 1.0 per episode** before any B-X run.
  - Launched: bx-labels and BO seeds 0–2.
