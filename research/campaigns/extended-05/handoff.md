# Handoff: extended-05 COMPLETE

**Read first: [report.md](report.md).** Decisions: [decisions.md](decisions.md). Audits: [review/](review/). Registry: [registry.json](registry.json).

## State
- Both primary questions reached an evidential boundary. No jobs are running.
- **Budget:** CPU 44.6k / 172.8k core-s metered (plus ~3.0k local, estimated); GPU 6.6k / 43.2k s. The rest is intentionally unspent.

## pro6000 (renamed dirs)
- **Roots:** `~/structured-latent-dynamics-campaign03` (env, historical), `...-campaign04`, `...-campaign05` (bin/job.py, bin/metered.sh, results, source-*).
- **Frozen artifacts:** controllers in results/e05-apit, results/e05-apic; labels in bx-labels, bxc-labels; branch labels in e05-hr/e05-hr2.
- **Keep-alive:** the task `TensegraWSLKeepalive` and linger are still active. Teardown commands are in ../extended-03/infrastructure.md.
- **Other workloads:** the user's unrelated jobs run on this host; campaign jobs never touched them.

## Tools
- `research/tools/campaign05_remote.py` (snapshot/launch, guarded) and `campaign05_ledger.py`.
- `campaign05_hr.py` (headroom), `campaign05_apit.py` (delegation controller: apit/acf/apic/acfc world kinds, cost), `campaign04_probeworld_train.py` (split sets b1/b5/b5c), `campaign05_bx_score.py`, `campaign05_bloc.py`.

## Operational lessons
- Snapshots exclude `research/results`, so stage artifacts explicitly and verify their hashes.
- Smoke one job before launching a batch.
- Never chain `commit -am` after a merge.
- Stamp times with `date` or git, never by hand.

## Next
See report.md "Next-decision recommendation".
