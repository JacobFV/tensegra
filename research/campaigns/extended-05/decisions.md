# Decisions (extended-05)

- 2026-09-27T01:26Z: **Brief adopted** (the user chose "adopt and start now").
  - New window 01:26:57Z → 2026-09-28T01:26:57Z; new root ~/structured-latent-dynamics-campaign05 with bin/job.py and metered.sh.
  - The pro6000 was checked: WSL up 6.5 h, keep-alive Running, GPU desktop-only. An unrelated user `uv pip install` (inpaint360gs env) is running and will not be disturbed.
  - The pro6000 directories were renamed by the user (tensegra-campaign0X → structured-latent-dynamics-campaign0X); tooling was repointed (7851b490).
  - Branch campaign/extended-05 from main 3a566812 (which includes the extended-04 report correction on the composition probe rates).
- 2026-09-27T01:29Z: Design v1, registry (7 entries) and machine-readable job plan (~88k core-s with 25% contingency; ~42k ungated) written. Internal review commissioned next. Timestamps for later entries come from git commits.
- 2026-09-27T01:44Z: Internal design review (3 BLOCKER, 11 MAJOR) adopted as **design v2**. Key changes: G1 counts no abstain gains and is a necessary condition only, plus a deployable cross-fitted single-deviation gate G1b ≥ .01; O(I) anchored to D's action kind; r3–r5 reserved for confirmation; the new probeworld split table (holds removed from train and dev, new seeds); exact-solver pool sizing (≥ 20 supporting configurations per cell); a B-HR headroom gate; BX3 modular arm; BO trained on the extended-04 split; learned-head evaluation priced at .35 core-s/episode. Both builders notified.
