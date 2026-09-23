# S08: frozen occurrence-alignment acquisition diagnostic

Replay only the declared final S01-N8192 and S07 checkpoints on the same TRAIN128/DEV512. No training, checkpoint selection, threshold fitting, or confirmation predictions. Actor inputs remain public text; gold occurrence spans enter scoring only. Require raw decoded predictions to reproduce each arm's immutable archive exactly before interpreting the diagnostic.

Save lossless original copy logits, token identity equivalence, raw argmax, probability mass at first-identity and actual occurrence positions, total identity probability, entropy, and canonical identity correctness. Report identifier occurrences separately from entity nodes. Associate occurrence correctness with correctness of required incoming ordered edges descriptively; this is not a causal intervention or an independence claim.

This measures whether S07 acquires its changed target, and whether that co-occurs with ordered-edge success. It cannot conclude that failed linear retrieval means information is absent. It does not judge reference semantics from entropy alone. Existing TRAIN-calibrated thresholds remain frozen for edge association; canonical metrics remain historical.

Width1024, original actor, 1,280 total frozen forwards. Requested full occupancy cap60 seconds, subject to root queue release. Dataset/checkpoint/source hashes are checked or recorded. Checkpoint hashes are frozen only after the declared S07 endpoint is complete; no endpoint alternatives are selected. CPU scalar fixture distinguishes raw occurrence accuracy from identity-equivalent copying.
