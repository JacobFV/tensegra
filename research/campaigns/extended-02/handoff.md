# Handoff (after phase 3)

Reports:
- **Phase 1:** campaign-report.md (unchanged).
- **Phase 2:** campaign-report-phase2.md and claim-map-phase2.md.
- **Phase 3:** campaign-report-phase3.md and claim-map-phase3.md.
- **Independent audits:** review/phase2-independent-audit.md, review/phase3-independent-audit.md and review/phase3b-independent-audit.md.
- **Cross-phase synthesis:** campaign-synthesis.md.
- **Decision log:** decisions.md (phase-2 and phase-3 sections).

State at close (2026-09-25 ~05:30Z; E19–E22 and the phase-3b audit are included):
- No campaign processes are running on gb10-direct.
- Budget used: CPU 44.4/48 core-h; GPU 9.3/12 device-h.
- Deadline: 2026-09-25T16:13Z.
- The branch is pushed to origin.

Infrastructure: remote gb10-direct; env /home/brandonin/topoformer-stage8-cuda/bin/python; root ~/topoformer-campaign02; source snapshots are named source-<sha>; jobs run via research/tools/campaign02_job.py. The ledger is research/tools/campaign02_phase2_ledger.py (GPU = device occupancy, accepted by the user). The modular world is src/topoformer/campaign02_modular.py; its conditions are built with research/tools/campaign02_e16_config.py.

If resumed (≤ ~3.5 CPU core-h remain), the most informative next steps are:
1. **An m1 + same-type-distractor training arm,** to separate the input and data parts of the E20 repair.
2. **Distractors that are never accidentally goal-valid,** and provenance tests with non-trivially named drafts.
3. **The residual `choose_item` loop** (5/22,272 episodes per arm in E22).
4. **Independent-bank replication of the population mechanism** (E08/E12/E13/E15/E21 share three banks).
