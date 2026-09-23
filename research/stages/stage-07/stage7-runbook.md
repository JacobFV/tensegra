# Stage 7 reproduction

Use a fresh checkout of the source commit recorded in each artifact manifest. Earlier-stage frozen checkouts are not reused for training. Install the project and test extra into an isolated environment; the remote pilot environment already contains Torch and pytest.

Run each study in a new output directory, with bounded CPU threads:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m topoformer.interface_study --config configs/stage7-ab-main.json --output results/ab
python -m topoformer.retention_study --config configs/stage7-c-main.json --output results/retention
python -m topoformer.halting_study --config configs/stage7-halt-main.json --output results/halting
python -m topoformer.semantic_scaling --config configs/stage7-semantic-1000.json
python -m topoformer.semantic_scaling --config configs/stage7-semantic-10000.json
```

Semantic study output directories are configured in their JSON files. Do not reuse an output directory with a different source/configuration. Initial acquisition recipes and later frozen main recipes are separate experiments. A run interrupted for timing is incomplete; it is not pooled into the restarted study.

Read subsystem reports alongside raw metrics. In particular, a perfect protected record is not evidence for a perfect neural readout, calibration on supplied synthetic factors is not calibration of learned proposals, and canonical graph reconstruction is not general semantic equivalence. The machine-readable gate registry is the authority on which dependent studies remain blocked.

Full dense semantic predictions and larger remote checkpoints have explicit path/size/SHA256 manifests; committed compact metrics and deterministic failure samples permit inspection without loading those files. Exact source snapshots accompany recipes changed during acquisition. Regeneration must use the matching compiler, renderer, source, seed, and config hashes.
