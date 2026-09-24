# R11 profile release recipe (prepared, not launched)

Frozen training source `a7f99f05`; profile driver `636f2557`.
Remote immutable source: `/home/brandonin/topoformer-campaign-01/return-tails/r11-profile-source-a7f99f05`.
Snapshot SHA256: `6d5ab3a64ad768d6c46bc52f363d20a2a61a25954dfe57fa0368ae6936bcf4d9`.
Config SHA256: `169ccf3b88398f0b701560372e732c86116d490d8a00bfc8a8cabc8565514b25`.
Driver SHA256: `3389aa9ac6ea0dd29a9b5d6c2caf42c296adac648021c43c3faf925f5ecb4c33`.
Checkpoint SHA256: `00e122c8cfe655dc9a8f801b48c5b5092db6caf8909479f2c3ac3910f0b7a077`.

Only the root GPU scheduler may release this command. It checks immutable output absence, every source hash, checkpoint hash and idle CUDA processes before execution. Full driver startup/checks, child training/inference/export and shutdown are measured by GNU time; timeout limits the whole driver to60seconds. The child manifest separately reports phase costs. Mechanical exact-replay profile only; no main competence decision. This60-second cap is proposed until root release.

```sh
ssh gb10-direct 'cd /home/brandonin/topoformer-campaign-01/return-tails && test ! -e r11-profile && test ! -e r11-profile-driver.json && test ! -e r11-profile-outer.json && PYTHONPATH=/home/brandonin/topoformer-campaign-01/return-tails/r11-profile-source-a7f99f05/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup /usr/bin/time -q -f '\''{"outer_seconds":%e,"exit_code":%x}'\'' -o r11-profile-outer.json timeout 60s /home/brandonin/topoformer-stage8-cuda/bin/python r11-profile-source-a7f99f05/run_driver.py /home/brandonin/topoformer-campaign-01/return-tails/r11-profile-source-a7f99f05 /home/brandonin/topoformer-campaign-01/return-tails/r11-profile /home/brandonin/topoformer-campaign-01/return-tails/r11-profile-driver.json > r11-profile.log 2>&1 < /dev/null &'
```

If timeout/failure occurs, retain its receipt/log and charge outer occupancy. Do not overwrite or silently rerun. Root confirms GPU-free before releasing the next job. Do not launch before independent source preflight and explicit root GPU release.
