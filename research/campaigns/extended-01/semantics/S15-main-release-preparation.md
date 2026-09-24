# S15 v2 mains frozen at allocated800 seconds each

Root allocation4a037bf6 follows paired timing proposal70b85264. Independent paired-profile audit and each separate GPU release remain required. Both remote immutable freezes and all CPU source/cache/audit/parent/calibration guards pass; output/prefixes unused. No main started. Original S11model+AdamW parent3799ade5, LR1e-5,4096updates×8, fourcurves0/1024/2048/4096, full2048DEV and historicalTRAIN128, dataset/criteria/extension rules unchanged. Profile checkpoints are not parents.

Control config SHA a2e93451506a6c32c43ed7d61abf1b4721703e48578fc1fd9792b44924ce76f5; mixed86849e95b366a2c3adf24576f4d7a57beadcc993d260fd49d2108edede05a88d. Trainerbc7812834dffde6e0061b1590c51a2b70c06ddb57d84644c9b4fd396d2e49c36; wrapper8a7bee9f7cdcb20bd1110c46bd18ff4ec07cebce171a184e07b6c4335ab0c7ef. Full bindings in `s15-main-staging.json`.

From `gb10-direct:/home/brandonin/topoformer-campaign01-semantics`, run only the separately released job after actual idle/prefix guards:

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_shape_launch.py configs/campaign-s15-control-main-frozen-v2.json --cap 800 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s15-control-main-v2 > s15-control-main-v2.launch.out 2>&1 < /dev/null &
echo $! > s15-control-main-v2.launch.pid
```

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_shape_launch.py configs/campaign-s15-mixed-main-frozen-v2.json --cap 800 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s15-mixed-main-v2 > s15-mixed-main-v2.launch.out 2>&1 < /dev/null &
echo $! > s15-mixed-main-v2.launch.pid
```

No automatic successor, extension, confirmation or retry. Report full receipt and actual GPUFREE before archive/analysis. Preserve all fixed outputs and failures; no post-profile recipe adaptation.
