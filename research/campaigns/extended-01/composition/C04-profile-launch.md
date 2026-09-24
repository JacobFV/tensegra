# C04 bounded profile launch sheet

Prepared before GPU profile outcomes. Coordinator alone releases the slot. This sheet does not authorize launch. Main confirmation caps remain unset.

The clean composition worktree contains no abandoned uncommitted changes. Commit `09724b2` adds the previous worker's CPU neural smoke receipt to the ledger; it changes no executable source. The staged executable source is `8a92869` at `/home/brandonin/topoformer-campaign-01/composition/c04-source-8a92869` on `gb10-direct`. This supersedes `5847a5e` only to publish complete occupancy receipts atomically with file and directory fsync; model code and profile configuration are unchanged.

Local and staged SHA256 checks agree:

- `configs/campaign-c04-profile.json`: `1ce378c705a59cc11ebc6ec045e3066f717b959811dc2835ab392dfc9fb3a932`
- `src/topoformer/campaign_composition_confirm_launch.py`: `9f60532e8bfd2ed332bbfdda63d0d434b85d382544d31c76458bd2c0ea6031e1`

The existing CUDA Python is `/home/brandonin/topoformer-stage8-cuda/bin/python`. Profile-only namespaces are 569000001/2/3, with independent initialization1699 and view RNGs5690001xx/2xx/3xx. Requested caps are hybrid120, N1static60, N1roles60, N2rekey60, timing120 seconds: total420. Every phase uses an immutable output and external subprocess occupancy receipt, checks the GPU is idle, and aborts if it exceeds its cap. Stop on any failure and report the receipt; no automatic retry or increased cap.

After explicit coordinator release, run from the staged source directory with `PYTHONPATH=src`. For each released phase invoke:

```sh
/home/brandonin/topoformer-stage8-cuda/bin/python -m topoformer.campaign_composition_confirm_launch \
  --config configs/campaign-c04-profile.json \
  --output /home/brandonin/topoformer-campaign-01/composition/c04-profile-PHASE \
  --prefix /home/brandonin/topoformer-campaign-01/composition/c04-profile-PHASE \
  --python /home/brandonin/topoformer-stage8-cuda/bin/python \
  --phase PHASE --cap CAP
```

Substitute the four non-timing phase names and their registered caps above; execute serially. The timing config is derived only after all three neural profile phases finish successfully. Copy the unchanged profile config and add `neural_checkpoints` entries for `n1_static`, `n1_roles`, and `n2_rekey`, each containing the absolute `c04-profile-ARM/endpoint.pt` path and its file SHA256. Use each fixed16 endpoint, independent of accuracy or calibration selection. Record the derived config SHA256 before timing. Invoke the same wrapper with phase `timing`, cap120, the derived config, and output/prefix `c04-profile-timing`.

Timing measures the first16 mechanical test events, batches1/64, workspace delays0/16, warmup2 and one pass. Main registration still specifies first256 events, warmup8 and three passes; extrapolate timed cell costs individually, including setup, rather than scaling total profile occupancy blindly. Neural profiles have only16 updates and tiny data: use prior C02/C03 full training costs and measured fixed overhead alongside these profiles. Hybrid profile has16 rows versus main4096 (causal16 versus2048), so its startup fraction must also be separated when projecting main cost.

At each released batch boundary, report all external occupancy receipts plus `nvidia-smi --query-compute-apps=pid --format=csv,noheader` and full process state to the coordinator before result analysis. Retain failed/timed-out occupancy. Mechanical profile accuracy is software validation, not confirmation evidence. Freeze source/config hashes, measured estimates and main caps with the coordinator before any 560/561/562 model inference.
