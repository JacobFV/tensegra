# Allocated mains staged; separate serial releases required

Root allocation c7838024 and independent preflight35f34b70 authorize the fixed budgets: S13English/mixed600s each and S14motif/rename420s each. They do not authorize starting a job without its separate root release. R10development remains the current GPUowner. No GPU/model import/forward was performed during this staging.

All26source/config files match frozen bindings; newly added main helper/wrappers/configs were copied without overwriting any differing existing source. All6primarycheckpoint/manifest bindings, fixedcache/S11parent/control hashes and unused output/prefix checks pass. Full source/config/wrapper hashes, actualhost, guard timings and primary bindings are in `postmatrix-main-staging.json`. Each main must repeat guards and check actual GPUidle at launch.

| Job | Cap | Config SHA | Wrapper SHA |
|---|---:|---|---|
| s13-english-main-v1 | 600 | `286e5b258f2e2b02d9a5605b94bf0d62aa0718a3d7f170ce0c93dacc22e002d9` | `a0fe8998576e23604a8c8c696eb4b1e6af1b5290fdead4b6adf42b36a4e27adb` |
| s13-mixed-main-v1 | 600 | `8edb4db2880813c865bfe264722fa00abab50abefa7a02a22bd076cdde621566` | `a0fe8998576e23604a8c8c696eb4b1e6af1b5290fdead4b6adf42b36a4e27adb` |
| s14-motif-main-v1 | 420 | `7d89a4a9bd78051058530670b24c3456bf868cbb81fc827d613e11fe6d48c0d7` | `912d5956856b2f5acd45b977ba9142ee979c2d4b49bce9733e8438d00608aad5` |
| s12-renamed-main-v1 | 420 | `e52b66b85115e692e35cb2b60748ec483c3744862eec479f538e8ef94b883da4` | `48d99792bcd743aee77069ec8f7b36ac3fc6e6f1bddd593c15f969a85be5531b` |

Run only the command for the separately released job, from `gb10-direct:/home/brandonin/topoformer-campaign01-semantics`. Commands are detached and have distinct immutable prefixes. Do not chain these as an automatic queue.

## s14-motif-main-v1

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_motif_main_launch.py configs/campaign-s14-motif-main-frozen-v1.json --cap 420 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s14-motif-main-v1 > s14-motif-main-v1.launch.out 2>&1 < /dev/null &
echo $! > s14-motif-main-v1.launch.pid
```

## s12-renamed-main-v1

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_confirmation_rename_main_launch.py configs/campaign-s12-renamed-main-frozen-v1.json --cap 420 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s12-renamed-main-v1 > s12-renamed-main-v1.launch.out 2>&1 < /dev/null &
echo $! > s12-renamed-main-v1.launch.pid
```

## s13-english-main-v1

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_multisurface_main_launch.py configs/campaign-s13-english-main-frozen-v1.json --cap 600 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s13-english-main-v1 > s13-english-main-v1.launch.out 2>&1 < /dev/null &
echo $! > s13-english-main-v1.launch.pid
```

## s13-mixed-main-v1

```bash
nohup /home/brandonin/topoformer-stage8-cuda/bin/python -u src/topoformer/campaign_semantics_multisurface_main_launch.py configs/campaign-s13-mixed-main-frozen-v1.json --cap 600 --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix s13-mixed-main-v1 > s13-mixed-main-v1.launch.out 2>&1 < /dev/null &
echo $! > s13-mixed-main-v1.launch.pid
```

Proposed serial order after current R10: S14motif, the other worker’s final A14seed, rename, S13English, independently released reference work, then S13mixed. Root owns that queue. Report full occupancy/GPUFREE before archive/analysis for every job; no automatic retries/extensions or policy changes. Main inputs and thresholds remain those frozen before profile outcomes.
