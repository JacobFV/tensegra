# Handoff: extended-07 COMPLETE

**Read first: [report.md](report.md).** Decisions: [decisions.md](decisions.md). Audits: [review/](review/). Registry: [registry.json](registry.json).

## State
- **Closed.** The immediate milestone reached a localized negative result:
  - learned factor information is distorted (acquisition error grows with combination order);
  - the historical LRN does not use it (redundant channel);
  - consumers made to use it gain invariance/"don't change" behaviour, not interaction-driven change discrimination;
  - no consumer-side repair passed its registered gate or replicated.
- **No e07 jobs are running or queued.**
- **Budget:** CPU 43.7k / 86.4k core-s metered, plus ~0.3k local; GPU 0 charged (legacy occupancy is reported separately).

## pro6000
- **Root:** `~/structured-latent-dynamics-campaign07`:
  - `bin/job.py`, `bin/metered.sh`;
  - `results/`: receipts and all artifacts;
  - `source-<sha>` snapshots;
  - `agent-*` builder copies.
- **Historical inputs (read-only):** `~/structured-latent-dynamics-campaign06/results` (LRN/RAWF/SUP/B0 checkpoints; b6 / b6c / UCE labels).
- **Key artifacts** (in `results/`):
  - `e07-diag-*`: Phase-1 per-decision logs.
  - `e07-p2-bank/bank.pkl`: the history bank.
  - `e07-p2-*`: 2×2, SEP, predictors, OOF, consumers, evals.
  - `e07-p3-*`: interaction consumers and evals.
  - `e07-p2c-labels/labels`: the fresh b6d pool and octets. **They have now been used once, descriptively (P3-REPLICATE), so they are no longer fresh for confirmation.**
  - `e07-p3r-*`: the b6d evals.
- **Staged copies** (hash-checked) are in `research/results/campaign-07/`.
- **Keep-alive:** the task `TensegraWSLKeepalive` and linger remain active. Teardown commands are in ../extended-03/infrastructure.md.

## Tools
- **Remote and budget:**
  - `research/tools/campaign07_remote.py`: guards for `--needs`, `--mkdir`, concurrency/memory, output path and shell quoting.
  - `campaign07_ledger.py`: GPU charged only for `-gpu-` jobs.
- **Diagnostics:**
  - `campaign07_diag.py`: common-history and free-running protocols; interventions; φ conditions; b6/b6c/b6d populations; consumers.
  - `campaign07_diagscore.py`.
  - `campaign07_frontier.py`: J_sub / BA_nm / FA_sub; `--refs`; `all_x_q2notH` stratum.
  - `campaign07_p1analysis.py`.
- **Training and jobs:**
  - `campaign07_p2.py`: history bank, common init, OOF, job generators; stages P2-*, P3-*, `--confirm`.
  - `campaign04_probeworld_train.py`: `--shape`/`--read` 2×2; `--factor-mode sep`; `--phi-contract`; `--cons-arch lin|mlp|bil|gate`; `--split-set b6c|b6d`.
- **Contracts:**
  - `campaign07_factor_contract.py`;
  - `factor-contract.json` (23 coordinates, groups, dependencies);
  - `gradient-flow.md` (with tests).

## Operational lessons (new)
- **Put `timeout` on every remote status call used in wait loops.** Three runner stalls cost ~25–40 min of wall time each.
- **Runners must retry refused launches.** A concurrency refusal otherwise silently drops a job.
- **Don't pass `--mkdir` for a tool that creates its own output dir** (use a subdir).
- **Register discrimination metrics** (J_sub, BA_nm) with stratum checks; flip accuracy alone is passable by bias.
- **Evaluate φ ∈ {own, exact, zero, mean}** for any factor-reading model before attributing gains to reading.

## Next
See report.md, "Next research decision": interaction-sensitive auxiliary targets (pairwise Q deviations) tested through trunk shaping in the S×R 2×2, with J_sub primary.
