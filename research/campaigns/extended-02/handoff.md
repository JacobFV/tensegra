# Handoff (phase 2)

Phase 1 closed at 49cece64 (campaign-report.md, unchanged). Phase 2: campaign-report-phase2.md, claim-map-phase2.md, decisions.md (phase-2 section), phase2/ (protocols, analyses, mechanism receipts), review/phase2-independent-audit.md.
Coordinator: root session. Remote gb10-direct; env /home/brandonin/topoformer-stage8-cuda/bin/python; root ~/topoformer-campaign02; source snapshots source-<sha>; jobs via research/tools/campaign02_job.py.
Ledger: research/tools/campaign02_phase2_ledger.py folds receipts into budget.json (GPU = device occupancy; per-process sum reported as upper bound).

Do not restart closed experiments. If resumed, the most informative next steps are:
1. Selection with longer evaluation horizons, or lineage-protected diversity (niching by behavioral mode), to test whether a population can find the combined direct-first+solver policy that single lineages reach.
2. New primitive semantics and call-graph motifs; the current workshop has a fixed two-solver workflow.
3. A graph-bias attention arm only after an input interface with an affordable cost exists (the memory interface was ~10x slower).
