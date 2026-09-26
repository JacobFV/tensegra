# Handoff: extended-04 COMPLETE (2026-09-26T22:50Z)

**Read first: [report.md](report.md).** Decisions: [decisions.md](decisions.md). Audits: [review/](review/).

## State
- Every experiment is finished and every promoted or confirmed reading has been audited. No jobs are running on the pro6000.
- **Budget:** CPU 119.1k / 172.8k core-s metered (plus ~1.4k local, estimated); GPU 11.6k / 43.2k s. About 54k core-s is left unspent.
- **Extended-03's ledger is untouched.**

## Infrastructure left in place (pro6000)
- **Roots:** `~/tensegra-campaign04` (bin/job.py, bin/metered.sh, results/, source-*) and `~/tensegra-campaign03`.
- **Env:** `~/tensegra-campaign03/env`.
- **Keeping WSL alive:** systemd linger is on, and the Windows task `TensegraWSLKeepalive` runs. It had stopped once during the campaign, so check it with `schtasks /query` before long jobs.
- **Teardown**, if no further campaigns follow: `schtasks /delete /tn TensegraWSLKeepalive /f` and `loginctl disable-linger brand`. See ../extended-03/infrastructure.md.
- **Tools:**
  - `research/tools/campaign04_remote.py`: snapshot, launch, launch-eval, launch-cmd, status, fetch. It refuses a missing snapshot.
  - `research/tools/campaign04_ledger.py`: the ledger, built from all receipts.

## Next steps
See report.md §7:
- a deployment-crediting improvement objective;
- metacontrol on decisions that have headroom;
- the probeworld composition gap;
- calibration under shift;
- budget planning by job count.
