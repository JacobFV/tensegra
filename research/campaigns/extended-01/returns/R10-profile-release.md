# R10 profile release recipe (prepared, not launched)

Frozen training source `c0c5aca1`; profile driver `dd993efa`.
Remote immutable source: `/home/brandonin/topoformer-campaign-01/return-tails/r10-profile-source-c0c5aca1`.
Snapshot SHA256: `5e0d839c4199bfce67bb030e13c13e760eb76191f8d4e0e652fd5a55b891f287`.
Config SHA256: `b0068dd6910aea17548ed8a9a4ed964c4ba53879365e03997c963a67044a82f3`.
Driver SHA256: `2473180ce4e211c4509abb36d32dbf07394d2ebd081cccfef4c990746fde29bd`.
Checkpoint SHA256: `00e122c8cfe655dc9a8f801b48c5b5092db6caf8909479f2c3ac3910f0b7a077`.

Only the root GPU scheduler may release this command. It checks immutable output absence, every source hash, checkpoint hash and idle CUDA processes before execution. Full driver startup/checks, child training/inference/export and shutdown are measured by GNU time; timeout limits the whole driver to60seconds. The child manifest separately reports phase costs. Mechanical profile; no competence gate.

```sh
ssh gb10-direct 'cd /home/brandonin/topoformer-campaign-01/return-tails && test ! -e r10-profile && test ! -e r10-profile-driver.json && test ! -e r10-profile-outer.json && PYTHONPATH=/home/brandonin/topoformer-campaign-01/return-tails/r10-profile-source-c0c5aca1/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup /usr/bin/time -q -f '\''{"outer_seconds":%e,"exit_code":%x}'\'' -o r10-profile-outer.json timeout 60s /home/brandonin/topoformer-stage8-cuda/bin/python r10-profile-source-c0c5aca1/profile_driver.py /home/brandonin/topoformer-campaign-01/return-tails/r10-profile-source-c0c5aca1 /home/brandonin/topoformer-campaign-01/return-tails/r10-profile /home/brandonin/topoformer-campaign-01/return-tails/r10-profile-driver.json > r10-profile.log 2>&1 < /dev/null &'
```

If timeout/failure occurs, retain its receipt/log and charge outer occupancy. Do not overwrite or silently rerun. Root confirms GPU-free before releasing the next job. Main budget is not allocated until this profile completes.
