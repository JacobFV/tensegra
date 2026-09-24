# Extended research campaign 01

Start with the [scientific report](campaign-report.md), [claim map](claim-map.md), and [figures](figures/README.md). The [evidence index](evidence-index.md) links raw results and independent audits for each claim.

| Question | Main evidence |
|---|---|
| Reliable scalar access and use | [R04](returns/R04-report.md), [R05](returns/R05-confirmation-report.md), [remaining tails](returns/R10-R11-closeout.md) |
| Bounded neural–symbolic composition | [C04](composition/C04-confirmation-report.md), including strong neural and supplied-copy controls |
| Programmable attention versus alternatives | [A14](attention/a14-report.md), with strong supplied-neighborhood references |
| Fresh public-text graph acquisition | [S20 three-lineage confirmation](semantics/S20-report.md) |
| Transfer after broader motif training | [S21](semantics/S21-report.md), [strict permutation diagnostic](semantics/S21-isomorphism-report.md) |
| Locating the remaining semantic boundary | [S22 oracle-node diagnostics](semantics/S22-report.md) |

The [experiment registry](experiment_registry.json) records hypotheses and decisions. [Budget](budget.json) and [queue](queue.json) track charged work and scheduling. [Closeout](closeout-decision.md) explains why scientific runs stopped before exhausting the resource ceiling. [Handoff](handoff.md) preserves exact operational checkpoints; older entries describe the state at their recorded time.

Versioned source is under `src/topoformer/campaign_*`; frozen configurations are under `configs/campaign-*`. Compact raw outputs are under `research/results/campaign-01`, and independent verification tools under `research/tools`. Every study retains its source/config/data/checkpoint hashes and any remote checkpoint inventory. Existing output paths are immutable: reproduction requires a separately versioned configuration and fresh output directory, with the original recipe and history retained.

Historical Stage1–11 files and gates are unchanged. Primary campaign workspace width is1024; smaller test fixtures are explicitly mechanical. Supplied symbolic operations, privileged diagnostic inputs and learned model behavior remain separate claims.
