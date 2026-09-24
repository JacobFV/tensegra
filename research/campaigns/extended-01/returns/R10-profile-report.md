# R10 mechanical profile and prospective main allocation

The released1024-wide mechanical profile completed successfully. Full process occupancy (GNU time outside the complete driver) was13.44seconds; child execution13.1923seconds. No main competence decision is computed from this reduced-support profile.

| Phase | Seconds | Main scaling consideration |
|---|---:|---|
| Training feature capture, including model setup |4.9765|2,048→65,536contexts; setup does not scale |
| Evaluation feature capture |3.3854|mixture counts increase up to32×; grid counts16× |
| Two consumer fits |.8760|100→3,600updates; larger feature matrices |
| Endpoint decoding |.2403|larger captured populations |
| Serialization and remaining child work |2.5226|larger lossless features/logits; included in budget |

Peak CUDA allocated492,246,016bytes; process peak RSS2,181,722,112bytes. These are process measurements, not total device capacity. The main run captures each64ktraining feature once and uses a16kprefix view; it does not export a duplicate16kfeature bank.

Estimated main process occupancy is320–450seconds; request a600-second hard ceiling before launch to include extraction, fitting, all evaluation and compression/export. This is one development backbone and two fixed arms, not a confirmation or learning-rate search. Phase-scaled estimates include a conservative32×training/cached-readout factor and36×fit-update factor. Main CUDA use is expected to remain below10GiB, substantially below verified GB10capacity; actual memory/runtime must be reported. These estimates are not an efficiency result.

No endpoint outcomes from the mechanical profile select the architecture, optimizer, main exposure or advancement rule. The profile does not contain adequate support for the main gate and the standalone main analyzer explicitly rejects its cell sizes and arm names.

Source`c0c5aca1`, snapshot/config/checkpoint hashes match the prospective release record. Frozen features, logits and both fitted heads remain remote at `/home/brandonin/topoformer-campaign-01/return-tails/r10-profile`; committed compact raw predictions/manifests and full/child receipts are under `research/results/campaign-01/returns/r10-profile`. Independent raw/provenance review is requested before main release.
