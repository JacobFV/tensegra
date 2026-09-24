# Optional all-six lexical rename: release preparation only

Keep the exact frozen evaluator at4a05aba and config population: all six constant/decay endpoints, the same1,024 semantic constructions, case-preserving unseen identifier spellings, and each model's existing primary English TRAIN thresholds. No new calibration, subset selection, policy fitting or checkpoint selection. CPU preflight currently verifies both701checkpoint bytes, primary outputs, source hashes and semantic identity order;702/703 endpoints are pending. A separate all-six preflight must pass before any rename model forward because the evaluator itself validates endpoints sequentially.

Cost estimate from completed primary endpoint evaluations: constant70114.843s and decay70117.376s include TRAIN128 calibration plus1,024confirmation examples. Six such units total89.059–104.257s; renamed inference drops calibration but incurs six model loads and checkpoint/source hashing. Plan roughly90–150full-process seconds, **not an allocation or measured rename profile**. Profile/allocate only after the complete S12 matrix. The frozen evaluator has no subset/profile flag; any smaller mechanical profile needs separately versioned code and a fixed prefix, with no policy selection. A full-evaluator cap can be set only by root after that decision.

New CPU-only readiness command (no actor imports/forwards):

```bash
python3 research/campaigns/extended-01/semantics/S12-rename-readiness.py configs/campaign-s12-renamed-inference.json
```

Ready requires `all_six_ready: true`. Stage the unchanged evaluator/config and new occupancy wrapper only when root chooses the later job. The current remote semantic checkout lacks evaluator/config; the cache already exists and matchesSHAa3f3bc5e. The preflight used a temporary config copy without changing the running S12source. Conditional durable launch after explicit release and a root-chosen cap:

```bash
cd /home/brandonin/topoformer-campaign01-semantics
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_confirmation_rename_launch.py configs/campaign-s12-renamed-inference.json --cap "$RENAME_CAP_SECONDS" --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s12-renamed-inference > s12-renamed-inference.launch.out 2>&1 < /dev/null &
echo $! > s12-renamed-inference.launch.pid
```

The auxiliary wrapper matches the existing bounded S12 wrapper, changing only the dispatched evaluator module; frozen evaluator/primary sources remain unchanged. Report full occupancy/GPUFREE before analysis. Compare each seed/arm's primary→renamed exact-recovery transitions and the within-event LR contrasts, preserving all raw and calibrated outputs. Lexical robustness is secondary and cannot replace primary competence or prove language/depth transfer.
