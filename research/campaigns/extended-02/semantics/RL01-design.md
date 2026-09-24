# RL01 — relation acquisition with a supplied node boundary

Status: prospective development diagnostic; not launched by this worker.

**Question.** Does allocating the existing teacher-forced objective solely to relation records and termination improve fresh ordered-relation acquisition, including an omitted structural combination, after the full correct node prefix is supplied?

This is independent of the primary agentic environment. It cannot authorize or block learned orchestration, and is not a structural-attention comparison.

## Matched intervention

Load two identical copies of the historical S21 broad endpoint (seed 2101, update 4096), reset AdamW in both, and use the same fresh construction stream. Actor, representation, codec, width 1024, eight attention heads, NODE capacity 128, record capacity 160, and slot capacity 32 remain unchanged. A new development seed controls minibatch ordering; there is only one inherited model lineage.

- `all_records`: unchanged historical eight-field objective.
- `relation_only`: zero kind/value/copy losses; type loss only at EDGE/EOS positions; source/target/role/slot losses at EDGE positions. **Retain the denominator eight**, so the edge-field terms are not silently increased by 8/5.

Both objectives see gold teacher-forced prefixes during training. The relation-only arm still computes the same sequence and can backpropagate through earlier NODE states. This is loss allocation, not removal of NODE computation or freezing node representations.

Proposed budget: 1024 updates × batch 8 per arm; constant learning rate 3e-5; AdamW reset, weight decay .01, gradient clip 1. Checkpoints 0/256/1024. Root must profile representative training and inference before freezing a <=900 GPU-second combined profile/main allocation. Source/config/input hashes and the resolved process budget are mandatory. If the estimate cannot fit, reject or separately register a smaller protocol before outcomes. No discretionary best-checkpoint selection: report every checkpoint and the fixed final endpoint.

## Parent provenance

Archived source: `configs/campaign-s22-main-frozen-v1.json`, `checkpoints.broad`.

- GB10 path resolved from S22 outer-command working root: `/home/brandonin/topoformer-campaign01-semantics/results/s21-main-v1/broad/model-u4096.pt`.
- File SHA256: `a5ad0f6d4fba9804a0ae3ed1ef8dd7b399ed2037dc89acd7b28c07f3bfc56af8`.
- Tensor state SHA256: `2d1267174e4f7fbd93ebd77da36bec0748e74f5c412cc0144b56cf64925d683e`.

These are archive-derived bindings, not assertions that the remote file is still present. The root must check existence/hash before scheduling. Runner enforces both file and tensor hashes and parent arm/update/seed metadata. Parent optimizer is deliberately not inherited.

## Data and observability

Reuse the unchanged `campaign_semantics_s21_data` generator, canonical alpha-equivalence function, independent answer validator, public-feature audit, finite reservation caps, and vendored TCN implementation.

TRAIN1024: 3×3:256; 4×3:128; 4×4:128; 2×4:128; 5×3:192; 5×4:192.

Fresh known-mixture DEV512: half those counts. Separate fresh omitted 3×4 DEV512. All new sets exclude the historical alpha inventory **plus all later S21 train/development/confirmation sets**, and each other. Root freezes a new seed base not used by earlier generation. Supply only alpha hashes from any earlier sealed population; never use its performance to select this recipe. Unknown unarchived historical ad-hoc data remains an exclusion limitation.

Generation is a separately root-scheduled CPU job capped at one core-hour, including collision/capacity/feature checks. No language renderer change. These are development populations, not a new replication or sealed confirmation claim.

## Readout boundaries and metrics

Public text is encoded by the unchanged actor. Evaluation additionally supplies exactly the correct canonical NODE prefix using historical `NodePrefix`/S22 decoding. The controller receives no edges, edge count, relation labels, slots, EOS time, or graph targets. It excludes NODE after the supplied prefix and chooses EDGE/EOS from the actor. Strict codec validation is unchanged; malformed generated graphs score zero complete credit.

Report separately:

1. Teacher-forced exact ordered EDGE tuple (tag/source/target/role/slot), and EOS accuracy. This is gold-prefix local prediction, not a free trajectory.
2. Free-decoded exact relation graph, slots, strict validity, and complete graph after supplied nodes; raw generated records and controller counts retained.
3. Per-field training losses, presentations, unique constructions visited, parameter count, allocated CUDA memory, RSS, process duration, and CPU seconds.

Store local argmax/gold tuples rather than full vocabulary logits; this supports exact independent reconstruction of reported accuracies but not confidence calibration. No calibrated probability or planning claim.

## Branch rule

Fixed-endpoint development signal: relation-only reaches >=52/512 omitted complete graphs and improves >=26/512 over all-records, while losing <=26/512 known graphs. Such a signal warrants a **new** independently registered confirmation with fresh constructions and lineages; it does not itself establish transfer. Report local-edge improvements even if complete graphs remain zero, without promotion. If neither arm acquires meaningful omitted relations, classify fixed-prefix relation transfer as unresolved; do not launch another architecture automatically. Historical gates remain unchanged.

## Reproducible commands (root scheduling only)

```
PYTHONPATH=src python -m topoformer.campaign02_relation source-bindings
PYTHONPATH=src python -m topoformer.campaign02_relation build-data <frozen-data-config.json>
PYTHONPATH=src python -m topoformer.campaign02_relation run <frozen-profile-config.json>
PYTHONPATH=src python -m topoformer.campaign02_relation run <frozen-main-config.json>
PYTHONPATH=src python -m pytest -q tests/test_campaign02_relation.py
```

The source-binding command emits the dictionary to insert into each configuration. Data config requires pinned exclusion inventory/sources, unique `first_seed`, `shuffle_seed`, output directory, and CPU cap. Runner config requires pinned `train`/`known`/`heldout` under `inputs`, pinned `parent`, fixed parent state hash, width1024, batch8, eval batch32, paired arm list, device, learning rate, update/checkpoint settings, seed/schedule seed, output directory and process cap. A profile may use a short update count and a public-length-selected `profile_examples` subset; main always evaluates all 512 per population.

## Prepared execution inputs and acquisition panel

`RL01-data-prepared.json` resolves and pins the historical inventory plus **all three** S21 split files (train_broad, development, confirmation). Read-only GB10 SHA256 checks on 2026-09-24 matched every declared exclusion input and the 752,293,177-byte parent checkpoint. The metadata inventory count is 41,638 historical alpha constructions; later S21 exclusions are added before reserving any new split. The metadata receipt's unknown unarchived-data caveat is retained in the new manifest.

A deterministic stratified TRAIN128 panel now accompanies every checkpoint: 3×3:32; 4×3:16; 4×4:16; 2×4:16; 5×3:24; 5×4:24, selected with fixed seed 20202201. It receives the same local/free metrics but is explicitly training-exposed, never a generalization score. The profile samples public-long examples from this panel too. This adds 768 decoded training examples across the paired main run.

Prepared source hashes bind this implementation but become authorized only after independent review. Root then copies/freezes the configs, fills runner data hashes from the generated manifest, and profiles. Suggested caps are profile120 + main780 seconds, **not a guarantee that 1024 updates fit**. If projected work exceeds the joint900-second allocation, stop before main and register a revised exposure rather than silently changing it. Example commands after source packaging:

```
PYTHONPATH=<snapshot>/src <CPU-python> -m topoformer.campaign02_relation build-data <snapshot>/research/campaigns/extended-02/semantics/RL01-data-prepared.json
PYTHONPATH=<snapshot>/src <CUDA-python> -m topoformer.campaign02_relation run <resolved-profile.json>
PYTHONPATH=<snapshot>/src <CUDA-python> -m topoformer.campaign02_relation run <resolved-main.json>
```

`RL01-profile-template.json` and `RL01-main-template.json` deliberately contain non-runnable dataset-hash placeholders until generation succeeds. No experiment was launched while resolving these paths.
