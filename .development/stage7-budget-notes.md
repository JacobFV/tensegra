# Stage 7 acquisition and budget decisions

The study uses isolated remote CPU processes with two Torch/BLAS threads each. Local work is source editing, inspection, and artifact transfer. Independent tracks may run concurrently; no more than four acquisition jobs (eight threads total) are authorized. Frozen earlier-stage experiment directories are not modified.

Acquisition is distinct from competence validation. Fixed-set memorization is an optimization diagnostic and cannot authorize composition. Main studies require every prespecified seed and validation condition; heldout test data is assessed only after configuration freezing. No missing or empty condition may pass a gate.

## Decisions before main outcomes

- A/B: after a 1,000-update proposal acquisition succeeded on seed 0, freeze the same optimizer budget across seeds 0/1/2, with 1,024 examples per evaluation condition. Readiness retains the initial linear-factor calibrator as a recorded control, including its low-coverage acquisition failure. Supplied-posterior calibration cannot certify readiness of actual learned proposals.
- Return retention: after a fixed-set acquisition reached complete semantic reconstruction, authorize a 1,000-update, three-seed, three-arm study using newly generated training examples. Evaluate at least 512 examples per seed/condition, delay 0/1/2/4/8/16/32, and independent validation/test seeds. Retention objectives only; returned-fact use stays blocked pending the main retention gate.
- Halting: acquisition remains separate from the main budget. Public certification is an architectural observation, not learned certainty. Role, name, and certification marginals are balanced; only their conjunction identifies usable evidence.
- Semantic graphs: the first 200-update acquisition did not recover ordered edges. A supervised-loss imbalance diluted positive slot gradients across padded pairs. A revised 2,000-update small-set diagnostic tests a balanced loss before any large scaling run. Preserve the failed recipe and its metrics. Keep evaluation constructions fixed across cardinalities and report actual distinct graph count and optimizer exposure.

All success statements must distinguish learned readout from exact storage, typed-instruction retrieval from language parsing, and synthetic confidence calibration from learned candidate correctness. Acquisition-driven recipe revisions are reported as revisions, not as prespecified confirmations.
