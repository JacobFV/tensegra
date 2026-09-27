# Handoff: extended-06 COMPLETE

**Read first: [report.md](report.md).** Decisions: [decisions.md](decisions.md). Audits: [review/](review/). Registry: [registry.json](registry.json).

## State
- **Closed** on 2026-09-27, before the 17:51:06Z deadline.
  - Track A reached the brief's stopping point: a strong simple portfolio captures the public-information headroom.
  - Track B answered both registered questions.
  - Track C was not triggered.
- **No jobs are running or queued** on the pro6000; `campaign06_remote.py status` shows no active `e06-` units.
- **Budget:** CPU 79.1k / 172.8k core-s metered (dev 8.8k), plus ~11.0k local or unmetered, disclosed. GPU: 17.4k s on a conservative wall-clock union; actual GPU use was zero.

## pro6000
- **Root:** `~/structured-latent-dynamics-campaign06`, containing:
  - `bin/job.py` and `bin/metered.sh`;
  - `results/`: all receipts and artifacts;
  - `source-<sha>` snapshots;
  - agent copies (`agent-*`).
- **Env:** `~/structured-latent-dynamics-campaign03/env` (torch 2.14 cu130; use `python -m pytest`).
- **Frozen artifacts** (in `results/`):
  - Track A fits: `e06-pw-acf-fit{0,1,2}` (sha256 in decisions.md).
  - Track B runs: `e06-tb-<arm>-s<seed>/run`, with seeds 30–34 (B0/B2), 30–32 (other arms) and 35–39 (LRN/RAWF/SUP/B0).
  - Track B labels: `e06-tb-labels`, `e06-tb-labels-uce`, `e06-tbc-labels`.
- **Staged copies** (hash-checked): `research/results/campaign-06/{a-hs, a-cfsmall, b-arms, b-factc, b-screen}`.
- **Keep-alive:** the task `TensegraWSLKeepalive` and linger are still active. Teardown commands are in ../extended-03/infrastructure.md.
- **Other workloads:** the user's unrelated jobs run on this host. Campaign jobs never touched them. The one OOM kill (33 concurrent evals) hit only a campaign job.

## Tools
- **Remote and ledger:** `research/tools/campaign06_remote.py` (snapshot/launch, guarded; `e06-` names only) and `campaign06_ledger.py`.
- **Track A:** `campaign06_portfolio.py` (`evaluate`, `headroom`, `cf-populations`, `cf-fit`, `cf-score`, `cf-plan`) and `src/tensegra/campaign06_portworld.py`.
- **Track B:**
  - `campaign04_probeworld_train.py`: split sets b6 and b6c; arms b6_B0–B3 and dose1–3; `--arch fuse`, `--inputs factors6`, `--factor-mode learned`.
  - `campaign06_bscore.py`: per-pair endpoints.
  - `campaign06_bprimary.py`: pooled registered rules; `--two-level`, `--group`.
  - `campaign06_bscreen.py`.
  - `src/tensegra/campaign06_probeworld.py`.

## Operational lessons (new this campaign)
- **Respect the concurrency plan.** Launching 33 evals at once caused an OOM kill. Keep Track B jobs at ≤ 8 and check `free -g` first.
- **Create output directories before a job writes into them.** A scorer lost 1,068 core-s at its final write.
- **Size label caps per family.** The UCE octets cost 1.5× the SCE octets per octet.
- **Never chain a commit after a command that can fail.** It produced the misnamed commit e8537d2d.
- **Register n_boot (≥ 10,000) for any CI-based rule.** B-FACT-C's Q1 was within Monte Carlo error at 1,000 draws.
- **Use ≥ 5 seed pairs and two-level (seed + configuration) CIs for probeworld arms.** Seed variance is large.
- Carried over: stage artifacts with hash checks; smoke one job before a batch; take timestamps from `date` or git.

## Next (from report "Unresolved")
1. **Track A:** find a strategy-selection family whose optimal method depends on inferable but non-obvious structure. In portworld, the residual gap is hidden-state information. Measure headroom over depth-3 trees first.
2. **Track B:**
   - Log held-out auxiliary-head accuracy.
   - Test stop-gradient heads, which separate representation shaping from reading the predictions.
   - Analyse head error × flip decisions, to find out why SUP's interaction information does not reach LRN.
3. Evaluate the built U+C+E octets. This needs a cf-only eval mode that doesn't overwrite eval files.
4. Keep extended-05's exposure effect (B-XC) as a volume-confounded observation. At matched volume it did not reappear on S+C+E.
