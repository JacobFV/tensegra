# S15 v3 mains frozen at revised1100 seconds each

Root revised the prospective cap to1100 before any main after reviewing combined evaluation timing uncertainty. Unused800-second v2 frozen files and preparation records are preserved as superseded. Independent paired-profile audit and each separate GPU release remain required. Both remote immutable freezes and all CPU source/cache/audit/parent/calibration guards pass; output/prefixes unused. No main started. Original S11model+AdamW parent3799ade5, LR1e-5,4096updates×8, fourcurves0/1024/2048/4096, full2048DEV and historicalTRAIN128, dataset/criteria/extension rules unchanged. Profile checkpoints are not parents.

Control config SHA 576d7e31c3b42395cb762c49c86fc11da7da62de949fe6e68a53a29108b7388c; mixed42216efd35879f66df125c2cacde8e334d78140917a7e75aac746ccf36ab672b. Trainerbc7812834dffde6e0061b1590c51a2b70c06ddb57d84644c9b4fd396d2e49c36; wrapper8a7bee9f7cdcb20bd1110c46bd18ff4ec07cebce171a184e07b6c4335ab0c7ef. Full bindings in `s15-main-staging-v3.json`.

From `gb10-direct:/home/brandonin/topoformer-campaign01-semantics`, run only the separately released job after actual idle/prefix guards:

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_shape_launch.py configs/campaign-s15-control-main-frozen-v3.json --cap 1100 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s15-control-main-v3 > s15-control-main-v3.launch.out 2>&1 < /dev/null &
echo $! > s15-control-main-v3.launch.pid
```

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_shape_launch.py configs/campaign-s15-mixed-main-frozen-v3.json --cap 1100 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s15-mixed-main-v3 > s15-mixed-main-v3.launch.out 2>&1 < /dev/null &
echo $! > s15-mixed-main-v3.launch.pid
```

No automatic successor, extension, confirmation or retry. Report full receipt and actual GPUFREE before archive/analysis. Preserve all fixed outputs and failures; no post-profile recipe adaptation.
