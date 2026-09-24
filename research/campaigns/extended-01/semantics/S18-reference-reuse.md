# S18 context-off baseline reuse audit

The immutable S15 mixed trajectory supplies the proposed S18 context-off baseline, conditional on the new runner preserving the planned training contract. Plan77d2075d and composition-worker intent match S15's original S11 model+AdamW parent3799ade5, mixed4096 cache8ddbd15b, LR1e-5, batch8,128negative pairs, gradient clip1, curriculum at cumulative update×8, schedule seed15115 and negative-query seed150150000+presentation. The four saved endpoints are cumulative updates24576/25600/26624/28672 (additional0/1024/2048/4096). Full hashes, immutable config, source map, exposure counts and historical raw/calibrated curves are in `S18-reference-baseline.json`. New runner implementation still requires its own review; this audit does not claim uninspected runner equivalence.

Context-off S18 dispatches through the unchanged original actor path. Independent mechanical review already passed small numeric forward/gradient/AdamW next-step parity and all98 original parameter/optimizer mappings, with57,853,781parameters. These checks do not constitute a full-width numeric training rerun. Neither contextual-read nor10-microstep-control trajectory can be reused from S15; only the original two-microstep mixed-data arm can.

Scientific evaluation difference: S15 used historical old-shape TRAIN128 calibration, whereas S18 prospectively uses S17's selected actual mixedTRAIN128 and retains historical/raw references. The unchanged evaluation function runs eval/no_grad then restores train mode, with no optimizer updates or random sampling. Training schedule and negative queries use explicit independent generators; the actor has no dropout. Thus changing calibration or adding fixed evaluation panels does not change the intended training trajectory. Reusing saved checkpoints avoids demanding fresh CUDA bitwise replay and does not erase the historical cost: **484.14140757126734seconds whole-process**, including **248.572851seconds optimizer work**, plus32768presentations/1720320tokens/988008nodes/2208616edges. Additional evaluation costs remain separately chargeable.

S15 has raw and historical-policy DEV curves at all four checkpoints. S17 already supplies final matched-policy DEV and all128actualTRAIN rows/metrics, so final baseline evaluation can be reused without model calls. Only the first three checkpoints lack matched-population calibration/panels. If the chosen acquisition panel is exactly these128calibration examples, no additional final TRAIN inference is missing. A different independent TRAIN panel would be new evaluation and must be frozen separately.

CPU extraction of the existing S17 final TRAIN128 yields:

| TRAIN cell | Examples | Raw complete | Matched-calibrated complete | Presence / kinds / values / copy / slots exact |
|---|---:|---:|---:|---|
| 3x3 | 64 | 1 | 5 | 64 / 64 / 64 / 63 / 64 |
| 4x3 | 32 | 1 | 7 | 32 / 32 / 32 / 32 / 32 |
| 4x4 | 32 | 0 | 2 | 32 / 32 / 32 / 32 / 32 |

Exact edges equal complete graphs in every displayed cell/policy. Both policies' full component counts are retained in JSON. These labels fitted the relation thresholds, making this an **optimistic calibration-overlap fit panel**, not independent TRAIN generalization;3x4has no TRAIN examples. Even this panel retains edge failures. No forward pass, new fitting, generation, checkpoint modification or GPU operation occurred in this audit.

Reuse is a retrospective comparison against inspected development, not a fresh independent baseline or confirmation. Original S15 failed gates and S17 exploratory status remain unchanged. Historical artifacts remain immutable; source/actor/runner/config ownership stays with the composition worker. `S18-reference-extract.py` reproduces this note's counts from SHA-guarded archives and checks them against saved metrics. S17 independent full audit53f4175c also passed all raw records, cutoffs, components and transitions.
