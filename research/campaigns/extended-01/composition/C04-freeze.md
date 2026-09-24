# C04 prospective confirmation freeze

Coordinator adopted the measured allocation before any560/561/562 confirmation model outcome. Executable source commit `c34effbbfee2af554fa1726e87226a95c6b4fc92` is staged on GB10 at `/home/brandonin/topoformer-campaign-01/composition/c04-source-c34effb`. Model/evaluation implementations remain those profiled at8a92869; changes add only caps/status and a serial lineage launcher. This freeze adopts all scientific requirements in `C04-protocol-draft.md`, superseding its provisional cost/status paragraphs with this record and `C04-profile-report.md`. Three seeds1601/1602/1603 and inherited mappings302/303/304→10/11/12 are unchanged.

| Replicate | Config SHA256 |
|---|---|
|0|`dceac7b4e7ec147cf6575655f445ae1c7fb47fd1c108e9bc9188dae8c9eb0dd9`|
|1|`0bf93f3efbcfbba4ed74652ea88dda8d432ceb2fc328c3cbe4f9def046cb6003`|
|2|`197c12657b776adff66ec06c6a9259374f557afd6bbbc5e067bfa71aa4db01ef`|

Each released lineage runs five serial phases: hybrid900s, N1static180s, N1roles180s, N2rekey600s and timing120s. Total1980s per lineage;5940s for all three, expected2010–3585s. Main benchmark still uses first256 test rows, batch1/64, workspace delays0/16, warmup8, three passes. Primary neural endpoint remains4000; CLEAN-calibration-selected checkpoints remain secondary. Timing binds only fixed `endpoint.pt` hashes after training and never inspects scores to choose checkpoints.

The coordinator grants one explicit release per lineage. After release, from the staged source directory run the command below, replacing both `REPLICATE` occurrences with the released0/1/2:

```sh
PYTHONPATH=src /home/brandonin/topoformer-stage8-cuda/bin/python \
  -m topoformer.campaign_composition_confirm_batch \
  --config configs/campaign-c04-confirmation-REPLICATE.json \
  --prefix /home/brandonin/topoformer-campaign-01/composition/c04-confirmation-REPLICATE \
  --released-replicate REPLICATE
```

The launcher rejects lineage mismatches, nonregistered caps and preexisting outputs. It persists the start record, runs each bounded wrapper serially, stops on the first failed phase, and records phase receipts, summed process occupancy, GPU process listing and full process state in `.batch.json`. Timing config and start/batch accounting are persisted by temporary file, file fsync, atomic rename and directory fsync; each child wrapper independently persists its own receipt using the same durability procedure. Do not retry/extend/reseed on failure without a separately recorded prospective decision. The batch process itself performs no model inference, so charged occupancy remains the sum of the bounded child process receipts.

CPU-only mocked control-flow verification passed both all-five completion and failure at phase3: timing binds endpoint files, summed accounting is correct, failure stops subsequent phases and no temporary receipt remains. These checks ran no torch/CUDA inference. Historical source contract tests and mechanical GPU profiles remain the validation of unchanged model paths.

This freeze does not itself release any GPU slot. Root is sole scheduler; no main launch occurred while preparing it. Reviewer checks final source/config matching and resource accounting independently.
