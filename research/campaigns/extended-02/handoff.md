# Handoff (after phase 3)

Reports:
- **Phase 1:** campaign-report.md (unchanged).
- **Phase 2:** campaign-report-phase2.md and claim-map-phase2.md.
- **Phase 3:** campaign-report-phase3.md and claim-map-phase3.md.
- **Independent audits:** review/phase2-independent-audit.md and review/phase3-independent-audit.md.
- **Decision log:** decisions.md (phase-2 and phase-3 sections).

State at pause (2026-09-25 ~03:40Z):
- No campaign processes are running on gb10-direct.
- Budget used: CPU 41.1/48 core-h; GPU 7.7/12 device-h.
- Deadline: 2026-09-25T16:13Z.
- The branch is pushed to origin.

Infrastructure: remote gb10-direct; env /home/brandonin/topoformer-stage8-cuda/bin/python; root ~/topoformer-campaign02; source snapshots are named source-<sha>; jobs run via research/tools/campaign02_job.py. The ledger is research/tools/campaign02_phase2_ledger.py (GPU = device occupancy, accepted by the user). The modular world is src/topoformer/campaign02_modular.py; its conditions are built with research/tools/campaign02_e16_config.py.

If resumed (≤ ~7 CPU core-h remain), the most informative next steps are:
1. **Same-type (provenance) distractor returns.** E17 only tested wrong-type distractors. Provenance binding of an old same-type result for a different draft is untested.
2. **Halving with fresh banks and robustness-inclusive fitness.** This combines E15 (depth) with E13 (fitness visibility).
3. **Longer adaptation curves for the held-out primitive (E18), with retention monitoring.**
