# S17 paired profile frozen and isolated; separate GPU release required

Independent source/selection review737118b0 covers scientific source65e43fa7. Root allocated90seconds for the one paired mechanical profile; no main budget exists. The only subsequent source delta99589a36 makes child PYTHONPATH resolve the launcher's own source tree. Guard/timeout fixtures still pass. The isolated snapshot is read-only and source/config/selection/checkpoint/cache guards pass; no model has run.

Frozen config SHA `f3e2e32dac363ea73cf0a6a60110e1a614bc6a4503d00b27195c66aefc8d0ea0`; launcher `823a3579da5b9abb385beba087640732a0fc34edd04cbba568834d0623eccbcb`. Snapshot `/home/brandonin/topoformer-campaign01-semantics/s17-source-v1`, manifest SHA `cca9c0b5722de7966d4175540bd5bbcd0906624e86c683e4de0963cbffb3db1a`. Full bindings/argv and GNU utility hashes are in `s17-profile-staging.json` and remote snapshot `launch-binding.json`.

Use the existing campaign GNU-time/GNU-timeout pattern: outer time includes wrapper interpreter imports and initial idle query; external90-second timeout bounds the child process group. The wrapper keeps its durable internal started/occupancy/failure receipt. Report outer elapsed with0.01second precision and conservatively charge the reported value plus0.01, alongside the higher-resolution inner receipt. On external timeout retain failures and all logs; do not retry automatically.

From `/home/brandonin/topoformer-campaign01-semantics`, after explicit root release, launch this exact argv detached with stdin closed and stdout/stderr at `s17-calibration-profile-v1.launch.out`:

```bash
/usr/bin/time -f 'elapsed_seconds=%e
exit_code=%x
maxrss_kib=%M' -o /home/brandonin/topoformer-campaign01-semantics/s17-calibration-profile-v1.outer.txt /usr/bin/timeout --signal=KILL 90s /home/brandonin/topoformer-stage8-cuda/bin/python -u /home/brandonin/topoformer-campaign01-semantics/s17-source-v1/src/topoformer/campaign_semantics_recalibrate_launch.py /home/brandonin/topoformer-campaign01-semantics/s17-source-v1/configs/campaign-s17-calibration-profile-frozen-v1.json --cap 90 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix /home/brandonin/topoformer-campaign01-semantics/s17-calibration-profile-v1
```

Record the detached process PID at `s17-calibration-profile-v1.launch.pid`. Prefix/output/outer receipt were verified unused at staging; repeat checks immediately before launch. Verify actual GPU idle inside the charged wrapper and source snapshot/config bindings before execution. Both frozen endpoints run serially,128actualTRAIN calibration and16DEV/cell each. No DEV fitting, threshold-policy selection, confirmation, architecture/training change or retrospective S15 gate change. Main cap remains pending the separately measured calibration/DEV/export phases.
