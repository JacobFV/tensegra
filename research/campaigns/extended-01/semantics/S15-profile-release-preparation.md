# S15 v2 paired profiles frozen; individual GPU release still required

Independent preflight70a90416 and root allocation90seconds each bind the exact20-update/64DEV/full historical TRAIN128 mechanical profiles. Both remote freezes passed source/cache/audit/parent/calibration guards; output and prefix were unused. S13 mixed is still the GPU owner. No S15 job has started, and mains remain prepared/unallocated.

| Arm | Frozen config SHA256 | Cap |
|---|---|---:|
| control | e12d85cb12df72e321014735c217a5613747b41e2e99ed0046dce50e1accb6e7 |90|
| mixed | 2b4977ab6a049de65b16953d3916ed2ea391508418b8e5c0a4f9809cfae91f23 |90|

Trainer SHA bc7812834dffde6e0061b1590c51a2b70c06ddb57d84644c9b4fd396d2e49c36; wrapper8a7bee9f7cdcb20bd1110c46bd18ff4ec07cebce171a184e07b6c4335ab0c7ef; freezer aabf128649cf752231f4ffb64c87e49d34252e1b91784ceb45a7eb58c306da39. Full source bindings in `s15-profile-staging.json`. Sources are the independently reviewed34e27d86+790bdb75; no recipe delta during freeze.

From `gb10-direct:/home/brandonin/topoformer-campaign01-semantics`, after separately released idle/prefix checks, run only the released command:

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_shape_launch.py configs/campaign-s15-control-profile-frozen-v2.json --cap 90 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s15-control-profile-v2 > s15-control-profile-v2.launch.out 2>&1 < /dev/null &
echo $! > s15-control-profile-v2.launch.pid
```

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_shape_launch.py configs/campaign-s15-mixed-profile-frozen-v2.json --cap 90 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s15-mixed-profile-v2 > s15-mixed-profile-v2.launch.out 2>&1 < /dev/null &
echo $! > s15-mixed-profile-v2.launch.pid
```

Report full occupancy and actual GPUFREE before archiving, and do not launch the second automatically. Profile checkpoints are mechanical artifacts, never continuation parents. Main recipe and decision rules stay fixed; only measured paired costs inform future cap proposals.
