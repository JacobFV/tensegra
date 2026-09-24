# R10 development release recipe (prepared, not launched)

Frozen training source `c0c5aca1`; profile driver `44020a10`.
Remote immutable source: `/home/brandonin/topoformer-campaign-01/return-tails/r10-main-source-c0c5aca1`.
Snapshot SHA256: `c2a955db17f76a7da7ee3a1b08d8511f245dc75e2860fb5273bc98e3ee29a9e4`.
Config SHA256: `803221f41eafe96bbede99071f10a4738fcb81d4bcbdc87bfc78b30101f4c027`.
Driver SHA256: `f066d1f3cab77ce3b08958e8b6aacac0925bbb79bf9c0dab080a454d4b67c1c8`.
Checkpoint SHA256: `00e122c8cfe655dc9a8f801b48c5b5092db6caf8909479f2c3ac3910f0b7a077`.

Only the root GPU scheduler may release this command. It checks immutable output absence, every source hash, checkpoint hash and idle CUDA processes before execution. Full driver startup/checks, child training/inference/export and shutdown are measured by GNU time; timeout limits the whole driver to600seconds. The child manifest separately reports phase costs. Single development run; fixed main gate evaluated only after full raw-metric validation. This600-second cap is proposed until root release.

```sh
ssh gb10-direct 'cd /home/brandonin/topoformer-campaign-01/return-tails && test ! -e r10-development && test ! -e r10-development-driver.json && test ! -e r10-development-outer.json && PYTHONPATH=/home/brandonin/topoformer-campaign-01/return-tails/r10-main-source-c0c5aca1/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup /usr/bin/time -q -f '\''{"outer_seconds":%e,"exit_code":%x}'\'' -o r10-development-outer.json timeout 600s /home/brandonin/topoformer-stage8-cuda/bin/python r10-main-source-c0c5aca1/run_driver.py /home/brandonin/topoformer-campaign-01/return-tails/r10-main-source-c0c5aca1 /home/brandonin/topoformer-campaign-01/return-tails/r10-development /home/brandonin/topoformer-campaign-01/return-tails/r10-development-driver.json > r10-development.log 2>&1 < /dev/null &'
```

If timeout/failure occurs, retain its receipt/log and charge outer occupancy. Do not overwrite or silently rerun. Root confirms GPU-free before releasing the next job. Do not launch before independent profile audit and explicit root GPU release.
