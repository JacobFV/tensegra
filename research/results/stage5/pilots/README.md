# Pre-main shallow lifting pilots

Two seed-0 oracle-lowering runs use identical resolved configuration, training schedule and evaluation data. Training is 400 updates; all evaluations are shallow (four post-resolution operations, eight distractor bindings). These pilots assess the lifting boundary before main source/config freeze and are not OOD model selection.

The first scalar-input MLP reaches 0.375 validation output accuracy. Adding fixed bounded scalar RBF features with a learned output MLP reaches 0.6875 validation and 0.703125 independent shallow evaluation accuracy. Exact interpreter execution is perfect in both. The RBF prior improves optimization but does not eliminate lifting error.

The isolated remote overlay has a snapshot git revision and dirty source state; the raw rows contain SHA-256 hashes for every source file. **Only `runtime_model.py` differs between these runs.** Their config and training-schedule hashes match exactly. `source.tar.gz` archives the precise overlay source/config for reproducibility. The main runner subsequently corrects confidence intervention counting, clarifies timing labels and decouples template choice from depth; these pilot runner versions intentionally remain identical.

`manifest.json` hashes the compressed raw metrics, source snapshots and resolved configurations. These are small one-seed development pilots, separate from the frozen main experiment.
