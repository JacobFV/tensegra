# R11 development release recipe (prepared, not launched)

Frozen training source `a7f99f05`; profile driver `636f2557`.
Remote immutable source: `/home/brandonin/topoformer-campaign-01/return-tails/r11-main-source-a7f99f05`.
Snapshot SHA256: `afc2c88f1dceb37674cf1aae9cf64fc43357e5171eca1f64e615785e46fdcd39`.
Config SHA256: `953e3a0996280b544e4457c034c17a0507cabf42c5a08507e17d9f17609505b3`.
Driver SHA256: `3389aa9ac6ea0dd29a9b5d6c2caf42c296adac648021c43c3faf925f5ecb4c33`.
Checkpoint SHA256: `00e122c8cfe655dc9a8f801b48c5b5092db6caf8909479f2c3ac3910f0b7a077`.

Only the root GPU scheduler may release this command. It checks immutable output absence, every source hash, checkpoint hash and idle CUDA processes before execution. Full driver startup/checks, child training/inference/export and shutdown are measured by GNU time; timeout limits the whole driver to180seconds. The child manifest separately reports phase costs. Single fixed-endpoint development study;180-second cap proposed until root release.

```sh
ssh gb10-direct 'cd /home/brandonin/topoformer-campaign-01/return-tails && test ! -e r11-development && test ! -e r11-development-driver.json && test ! -e r11-development-outer.json && PYTHONPATH=/home/brandonin/topoformer-campaign-01/return-tails/r11-main-source-a7f99f05/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup /usr/bin/time -q -f '\''{"outer_seconds":%e,"exit_code":%x}'\'' -o r11-development-outer.json timeout 180s /home/brandonin/topoformer-stage8-cuda/bin/python r11-main-source-a7f99f05/run_driver.py /home/brandonin/topoformer-campaign-01/return-tails/r11-main-source-a7f99f05 /home/brandonin/topoformer-campaign-01/return-tails/r11-development /home/brandonin/topoformer-campaign-01/return-tails/r11-development-driver.json > r11-development.log 2>&1 < /dev/null &'
```

If timeout/failure occurs, retain its receipt/log and charge outer occupancy. Do not overwrite or silently rerun. Root confirms GPU-free before releasing the next job. Do not launch before independent source preflight and explicit root GPU release.
