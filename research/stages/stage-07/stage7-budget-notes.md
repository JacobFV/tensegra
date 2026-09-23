# Stage 7 acquisition and budget decisions

The study uses isolated remote CPU processes with two Torch/BLAS threads each. Local work is source editing, inspection, and artifact transfer. Independent tracks may run concurrently; no more than four acquisition jobs (eight threads total) are authorized. Frozen earlier-stage experiment directories are not modified.

Acquisition is distinct from competence validation. Fixed-set memorization is an optimization diagnostic and cannot authorize composition. Main studies require every prespecified seed and validation condition; heldout test data is assessed only after configuration freezing. No missing or empty condition may pass a gate.

## Decisions before main outcomes

- A/B: after a 1,000-update proposal acquisition succeeded on seed 0, freeze the same optimizer budget across seeds 0/1/2, with 1,024 examples per evaluation condition. Readiness retains the initial linear-factor calibrator as a recorded control, including its low-coverage acquisition failure. Supplied-posterior calibration cannot certify readiness of actual learned proposals.
- Return retention: after a fixed-set acquisition reached complete semantic reconstruction, authorize a 1,000-update, three-seed, three-arm study using newly generated training examples. Evaluate at least 512 examples per seed/condition, delay 0/1/2/4/8/16/32, and independent validation/test seeds. Retention objectives only; returned-fact use stays blocked pending the main retention gate.
- Halting: acquisition remains separate from the main budget. Public certification is an architectural observation, not learned certainty. Role, name, and certification marginals are balanced; only their conjunction identifies usable evidence.
- Semantic graphs: the first 200-update acquisition did not recover ordered edges. A supervised-loss imbalance diluted positive slot gradients across padded pairs. A revised 2,000-update small-set diagnostic tests a balanced loss before any large scaling run. Preserve the failed recipe and its metrics. Keep evaluation constructions fixed across cardinalities and report actual distinct graph count and optimizer exposure.

All success statements must distinguish learned readout from exact storage, typed-instruction retrieval from language parsing, and synthetic confidence calibration from learned candidate correctness. Acquisition-driven recipe revisions are reported as revisions, not as prespecified confirmations.

## Timing-only retention amendment

Before inspecting main evaluation outcomes, the first run's timing projected approximately 24 minutes for all interventions at every delay. The revised frozen schedule retains clean retention at every delay and all nine interventions at delay 16, including persistent-memory recovery. Cohorts, seeds, 512-example denominators, and competence thresholds are unchanged. The truncated run is marked incomplete and the amended run starts in a fresh output directory.

## Additional independent budgets

Halting main: three seeds, width 32, 2,000 updates, batch 32, 2,048 training examples, 512 examples per validation/test condition. The exact public evidence rule is an explicitly architectural upper bound, separate from learned output at oracle timing.

Semantic scaling diagnostic: requested 1,000 and 10,000 distinct constructions, three seeds, 1,000 updates each, semantic and no-input controls, fixed 64-example heldout constructions and checkpoints 0/100/500/1000. This finite exposure budget cannot establish a million-example scaling law. Corpus feasibility and actual unique optimizer exposure must be reported. No semantic/runtime coupling is authorized.

## Corrected renderer-matched semantic diagnostic

The first completed cardinality sweep used an epoch-dependent renderer schedule: at the same 2,000 optimizer-example budget, N=1,000 exposed English and Spanish while N=10,000 exposed English only. Those results are archived as confounded diagnostics. The corrected repeat changes only the renderer schedule to optimizer-step parity, preserving architecture, loss, budgets, seeds, heldout graphs, and cardinalities. Each arm receives exactly 1,000 English and 1,000 Spanish presentations. The source/configuration and schedule regression were committed before restart (`d4056f7`). Actual unique construction exposure still differs (1,000 versus 2,000), and neither run establishes full convergence or million-example scaling.

## Integrated verification

An immutable checkout at `0694ec5` passed **511 tests and 6 subtests** remotely in 7.26 seconds with two CPU threads. All differences against Stage6 baseline `04263b2` are additions; existing Stage1–6 files are unchanged. Remaining semantic result collection and report-only changes do not alter the verified model source.
