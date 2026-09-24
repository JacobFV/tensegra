# S17 paired main frozen; separate GPU release required

Root allocated300seconds after profile8675ef57. The main preserves both endpoints, actualTRAIN128 policies, all2048DEV, source65e43fa7 and launcher99589a36. Read-only source snapshot is `/home/brandonin/topoformer-campaign01-semantics/s17-main-source-v1`; all185 manifest files and unchanged source hashes pass CPU verification. Output and receipt prefixes are unused. Main is exploratory, not confirmation or retrospective S15 promotion.

Config SHA `95d7962bea362cccb793faf322e109fc4e98dd66ebde47b0ccbd8b94b01ee98b`; exact argv SHA `01853058b134a803dc6316ac3eaee8c98ca03ee2f72cef813d897728d6119e38`. Full bindings in `s17-main-staging.json`. Outer GNUtime includes imports/idlequery, GNUtimeout bounds300seconds; report outer+0.01second conservative charge and inner durable receipt. No GPU launched.

```bash
/usr/bin/time -f 'elapsed_seconds=%e
exit_code=%x
maxrss_kib=%M' -o /home/brandonin/topoformer-campaign01-semantics/s17-calibration-main-v1.outer.txt /usr/bin/timeout --signal=KILL 300s /home/brandonin/topoformer-stage8-cuda/bin/python -u /home/brandonin/topoformer-campaign01-semantics/s17-main-source-v1/src/topoformer/campaign_semantics_recalibrate_launch.py /home/brandonin/topoformer-campaign01-semantics/s17-main-source-v1/configs/campaign-s17-calibration-main-frozen-v1.json --cap 300 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix /home/brandonin/topoformer-campaign01-semantics/s17-calibration-main-v1
```

Launch detached only after root release, with cwd `/home/brandonin/topoformer-campaign01-semantics`, stdin closed, exclusive `s17-calibration-main-v1.launch.out` and PID receipt. Repeat snapshot/config/input/unusedprefix guards before launch.
