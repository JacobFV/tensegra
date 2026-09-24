# Parent702 prepared; no launch authority

Remote byte-only preflight passed: frozen parent source, dispatcher/wrapper, config, TRAIN and DEV cache match. No parent702 log, receipt or output directory exists. No model imported or forwarded. Config SHA `7baa144c53653a3d27c1b575be7f77c240e89b2247b3f0e3c0711e8ad07fa5ec`; source freeze `dbd3575`; cap1,400 seconds.

Root queue: A09, A08, then C04 short profile batch precede this job. Only a later explicit root release permits executing the following durable launch on `gb10-direct`:

```bash
cd /home/brandonin/topoformer-campaign01-semantics
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_confirmation_launch.py configs/campaign-s12-parent-702.json --cap 1400 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s12-parent-702 > s12-parent-702.launch.out 2>&1 < /dev/null &
echo $! > s12-parent-702.launch.pid
```

The immutable existing wrapper refuses occupied GPUs and existing prefixes, externally caps the process, and writes full wall-clock occupancy after exit. Poll the existing launch only; never duplicate it. Completion requires receipt exit0, success artifact and GPUFREE verification; report GPUFREE/full occupancy before analysis. Preserve all outcomes. Bind both702forks mechanically to the same completed parent bytes; each fork requires a separate root release at cap800. Seed703 then follows the same protocol. Seed701's below-gate result changes none of these jobs. No S13 profile/inference until the full S12 matrix completes and root separately releases it.
