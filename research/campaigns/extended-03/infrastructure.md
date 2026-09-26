# Extended-03 execution host: pro6000 (from 2026-09-26)

Extended-03 runs on the **pro6000** workstation, not the GB10s:
- The peer GB10 (`gb10-direct`) is busy training another model.
- The local host GPU (Dell GB10) is busy as well.

Do not train on either GB10. `gb10-direct` may be used over ssh **only to copy files** (for example the extended-02 raw rows).

## Host

| Item | Value |
|---|---|
| ssh alias | `pro6000` (alias `supacomputa`), 100.75.41.88, user `brand` |
| OS | Windows 11 (build 26200). All work runs in **WSL2 Ubuntu** (kernel 6.6.87, WSL 2.6.3, systemd enabled) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB. It is shared with the Windows desktop (~11 GB used by desktop/VR apps at setup) |
| CPU / RAM | 24 logical cores / 62 GB available to WSL |
| Power | AC standby disabled (never sleeps) |

## Environment

- **Root:** `~/tensegra-campaign03` inside WSL (`/home/brand/tensegra-campaign03`).
- **Python env:** `~/tensegra-campaign03/env`, a uv venv. It matches the extended-02 GB10 env's key versions: Python 3.12.3, torch 2.14.0+cu130, numpy 2.5.3, pytest 9.1.1.
- **Source snapshots:** `source-<sha>`, immutable, built from the campaign worktree with `.git` and `research/results` excluded. Snapshots are pushed with a tar pipe because rsync is not available through the Windows ssh server:

  ```bash
  tar --exclude=.git --exclude=research/results -czf - . | \
    ssh pro6000 'wsl -d Ubuntu -- bash -c "mkdir -p ~/tensegra-campaign03/source-<sha> && tar -xzf - -C ~/tensegra-campaign03/source-<sha>"'
  ```

- **Results:** `~/tensegra-campaign03/results/<job>` and `<job>-process`, the job-wrapper receipts. Compact receipts are copied to `research/results/campaign-03/`.
- **Validation:** `tests/test_campaign0*.py` gives 173 passed in 23 s on CPU at `source-27c0de1d`, the same count as on the GB10 env.

## Launching jobs (root only)

The ssh login shell is Windows `cmd.exe`. It mangles nested quotes and interprets `|`, `>` and `&`, so:
- simple commands use `ssh -n pro6000 'wsl -d Ubuntu -- <cmd args>'`;
- anything else pipes a script on stdin: `ssh pro6000 'wsl -d Ubuntu -- bash -s' <<'EOF' ... EOF`;
- strip NULs from the output with `tr -d '\0'`.

**Detached jobs.** WSL terminates a distro, and kills `setsid`/`nohup` children, when the last `wsl.exe` session exits. This was verified: a test job died. The working setup has two parts:
1. **systemd user units with linger.** `loginctl enable-linger brand` is on. Jobs start through `~/tensegra-campaign03/bin/detach.sh <unit> <workdir> <cmd...>`, which calls `systemd-run --user --collect`. `systemd-run` does not inherit the caller's environment, so `detach.sh` sets the campaign-standard job environment explicitly: `PYTHONPATH=src`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. The first smoke run failed before that fix and is charged.
2. **Keep-alive task.** The Windows scheduled task `TensegraWSLKeepalive` runs `wsl.exe -d Ubuntu -- sleep infinity` at logon, which keeps the distro up. It was started manually on 2026-09-26. If WSL has been shut down, restart it with `ssh -n pro6000 'schtasks /run /tn TensegraWSLKeepalive'`.

Example (bootstrap):

```bash
ssh -n pro6000 'wsl -d Ubuntu -- /home/brand/tensegra-campaign03/bin/detach.sh <job> /home/brand/tensegra-campaign03/source-<sha> \
  /home/brand/tensegra-campaign03/env/bin/python research/tools/campaign02_job.py \
  --output /home/brand/tensegra-campaign03/results/<job>-process --wall-cap 3600 --cpu-cap 7200 -- \
  /home/brand/tensegra-campaign03/env/bin/python -m tensegra.campaign02_population \
  --config configs/campaign03/<job>.json --output /home/brand/tensegra-campaign03/results/<job>'
```

- **Status:** `systemctl --user is-active <job>`, or check `occupancy.json` in `<job>-process`.
- **Teardown after the campaign:** `schtasks /delete /tn TensegraWSLKeepalive /f` and `loginctl disable-linger brand`.

## Throughput (smoke)

`smoke-boot`, the X1-r0 bootstrap config cut to 6×5 supervised updates with 32 development examples, took 50.7 s wall and 51.2 CPU core-seconds. It is single-thread and CPU-bound, like the extended-02 jobs, so many jobs can run concurrently. The 24 cores allow the 12 P1 bootstraps at once.

## Accounting

The ledger is unchanged: job-wrapper receipts, with CPU as process-tree core-seconds and GPU as device occupancy (the union of GPU-job wall intervals, the convention the user accepted in extended-02). The pro6000 is faster per job, so the same GPU ceiling buys more work. The ceilings are **not** rescaled.

Different hardware and a different torch build mean extended-02 (GB10) numbers are not bit-comparable with extended-03 (pro6000). All extended-03 arms and references run on the pro6000.
