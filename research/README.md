# Research record

Experiment designs, scientific reports, audits, and reproducibility artifacts for Topoformer. Model implementations remain in `src/`, runnable experiment configurations in `configs/`, and regression tests in `tests/`.

| Stage | Question | Report |
|---|---|---|
| 1 | Does aligned attention topology help synthetic prediction? | [Pilot](stages/stage-01/pilot-report.md) |
| 2 | Efficiency, corruption, and graph transfer | [Report](stages/stage-02/stage2-report.md) |
| 3 | Learned latent-to-graph grounding | [Report](stages/stage-03/stage3-report.md) |
| 4 | Stable protected binding | [Report](stages/stage-04/stage4-report.md) |
| 5 | Semantic lowering and exact runtime execution | [Report](stages/stage-05/stage5-report.md) |
| 6 | Recurrent workspace and local execution interfaces | [Report](stages/stage-06/stage6-report.md) |
| 7 | Isolated proposal, readiness, and return interfaces | [Report](stages/stage-07/stage7-report.md) |
| 8 | Protected beliefs and return access at width 1024 | [Report](stages/stage-08/stage8-report.md) |

## Campaigns

| Campaign | Question | Report |
|---|---|---|
| extended-01 | Semantic acquisition, returns, composition and attention interventions | [Report](campaigns/extended-01/campaign-report.md) |
| extended-02 | Evolving agents that learn when and how to use computation | [Synthesis](campaigns/extended-02/campaign-synthesis.md) · [Final report](campaigns/extended-02/final-report.md) · [Closeout](campaigns/extended-02/closeout.md) |
| extended-03 (P1 + P2a complete) | Managing a changing dependency graph of computations | [Proposal](campaigns/extended-03-proposal.md) · [P1 report](campaigns/extended-03/report-P1.md) · [P2a report](campaigns/extended-03/report-P2a.md) · [Status/handoff](campaigns/extended-03/handoff.md) |

## Layout

- `stages/stage-NN/`: reports, plans, methods, notes, reviews, and stage-specific fixtures.
- `results/`: raw metrics, frozen source/config snapshots, hashes, plots, and failure cases. Existing experiment subdirectories are preserved to keep artifacts reproducible.
- `tools/`: report generation, plotting, and audit helpers formerly mixed with notes. General audit entry points also remain in the repository's `scripts/` directory.
- `layout-migration.json`: old/new paths and before/after SHA256 values for this reorganization.

Raw result files, frozen configuration records, and archived source snapshots were moved without changing their contents. Historical manifests can therefore still contain `.development` paths. A `.development -> research` compatibility symlink preserves legacy result-directory access, and a fixture alias supports the frozen Stage 8 semantic configuration. New documentation uses `research/`. Historical root-level reports have moved to their stage directory; use the table above or the migration map to locate them.

The remote Stage 8 experiment continues from its original immutable checkout; new artifacts are copied into `research/results/` after they finish. Directory organization does not change an experiment's source identity or competence gate.
