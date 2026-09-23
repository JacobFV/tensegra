# Stage 11: public-text graph acquisition

All three final checkpoints recover all eight complete canonical graphs, with both raw and TRAIN-calibrated decoding. This passes the narrowly scoped fixed-set acquisition gate. Independent raw-metric/provenance review confirms the acquisition result. The subsequent frozen-actor known-renderer test fails: every seed recovers 0/512 fresh complete graphs.

## Scope and supplied information

The actor receives English text and the copy inventory derived from that text. Gold graphs are privileged supervision, never actor inputs. Canonical compiler node ordering, ontology, and the training-derived finite categorical vocabulary are supplied label conventions. The experiment fits eight alpha-distinct unification constructions from the unchanged TCN generator. It does not measure language understanding or broad semantic generalization.

The unchanged semantic curriculum actor has 57,853,781 parameters, width 1024, eight workspace rows, four distinct blocks reused twice, and 128 fixed node queries. Public inputs contain at most 64 tokens and targets at most 37 nodes, with ordered slots through 3. The sampled edge objective is preserved; slots are supervised only on actual edges, including unordered real edges. There is no structural attention or runtime integration in this track, so results concern a supporting semantic interface rather than programmable attention itself.

Three initialization seeds (30, 31, 32) each receive 4,000 optimizer updates of the same eight examples: 32,000 presentations per seed, 96,000 total, but only eight distinct constructions. The inherited staged loss curriculum introduces copy/value supervision after 1,000 presentations and edge/slot supervision after 2,000. The final checkpoint alone determines advancement. Raw zero-logit edge decoding and separately labeled TRAIN-calibrated decoding are both reported.

## Observability and controls

The labels-only audit found no incompatible identical public inputs, observed lexical hash collisions, copy-target failures, unknown finite labels, node/slot overflow, or token truncation in the eight-example set. This is finite support, not a universal identifiability guarantee. Gold sampled pairs affect output loss rows only; a mechanical test verifies that they do not change actor node or copy outputs.

The matched frequency baseline uses coordinatewise modes from exactly the same eight English targets and no text features. It is not the historical bilingual fitter. Its exact graph accuracy is reported alongside learned results. Predicted presence and edges control deployment; gold masks are not used to restore missing nodes or edges. Train-only relation thresholds are frozen before any fresh evaluation.

## Resource and provenance

Source was frozen at `5a80a4a` before main training. The immutable remote archive and per-module hashes preserve actor, compiler, generator, and evaluation dependencies. The profile's first attempt stopped before any optimizer update because a public copy decoder expected CPU output indices; the wrapper now explicitly transfers inference outputs to CPU, without changing historical code. That failure and the successful profile receipt are retained.

The successful 20-update profile estimated 16–18 minutes for the prescribed matrix; the main hard cap is 20 minutes. Checkpoints remain immutable on the configured GB10 machine. Compressed predictions, targets, loss curves, per-relation calibration records, source/config/data/checkpoint hashes, GPU allocation and process RSS are archived separately from those checkpoints.

## Advancement policy

The restricted acquisition gate requires all three final checkpoints to decode all eight complete canonical graphs correctly. Component F1 or training loss cannot substitute. Any successful fixed-set result permits only a separately declared frozen-actor fresh-construction evaluation. It does not authorize broader training, runtime composition, or supervision withdrawal.

## Observed acquisition

| Seed | Exact graphs at 0 / 125 / 250 / 1000 updates | At 2000 | Final 4000 | Training seconds | Run wall seconds |
|---|---|---|---|---|---|
| 30 | 0 / 0 / 0 / 0 of 8 | 8/8 | 8/8 | 258.00 | 267.01 |
| 31 | 0 / 0 / 0 / 0 of 8 | 8/8 | 8/8 | 260.82 | 270.21 |
| 32 | 0 / 0 / 0 / 0 of 8 | 8/8 | 8/8 | 259.75 | 268.77 |

Raw and calibrated exact counts are identical at these checkpoints. At 1000 updates, type and visible identity copying are already perfect but complete graphs remain wrong; calibrated typed-edge F1 is .8690 / .9295 / .9650. Later edge acquisition closes the complete-graph gap. These checkpoint observations diagnose this fixed recipe; they do not select an earlier endpoint or isolate a causal effect of curriculum, width, or objective repair.

At the final checkpoint every required node attribute, copied identity, typed edge, and ordered slot is correct in all 24 seed-graph evaluations. Per-graph/per-relation error exports therefore contain zero final errors, and each privileged single-component replacement ceiling is trivially 8/8. The matched frequency baseline is 0/8 complete graphs. Final calibration is unnecessary for this fixed-set success because raw decoding also passes.

Measured optimizer time totals 778.58 seconds (12.98 minutes); per-seed wall intervals total 806.00 seconds (13.43 minutes), excluding inter-seed gzip serialization and process setup/teardown. Peak allocated CUDA memory is 1,226,426,880 bytes. Process high-water RSS rises to 2,969,304 KiB; it is not GPU allocation or total device capacity. All training ran within the 20-minute process timeout and exited normally.

## Interpretation

This establishes acquisition of complete public-text-to-canonical-graph outputs for eight supervised English unification examples. It removes the previous fixed-set acquisition block in this narrow family. It does not establish fresh construction transfer, variable renaming, heldout rendering, semantic trajectory learning, or runtime competence. The model can still memorize these eight surfaces. The separately preregistered frozen transfer evaluation requires >95% complete graphs in every seed over 512 alpha-distinct constructions, using unchanged vocabulary and final TRAIN thresholds. That evaluation fails as detailed below; further heldout-family advancement remains blocked.

## Frozen fresh-construction result

Before any fresh prediction, the protocol fixed a >95% complete-graph gate in every seed, a 512-example alpha-disjoint set, unchanged actor/checkpoints/vocabulary, and each seed's final TRAIN8 thresholds. All 512 first generated candidates were eligible: no train-alpha overlap, duplicate alpha key, unknown non-copy category, noncopyable target, capacity overflow, or observed public/hash collision. They use the same English renderer and have no lexical-token novelty relative to the eight training surfaces. This is therefore a clean narrow test of new constructions within familiar renderer/token and output support, not unseen-language or arbitrary-notation transfer. Alpha disjointness is asserted relative to Stage 11 training, not every historically inspected construction.

| Seed | Complete raw / calibrated | Type accuracy | Exact copy accuracy | Raw typed-edge F1 | Calibrated typed-edge F1 | Raw ordered-edge F1 |
|---|---|---|---|---|---|---|
| 30 | 0/512 / 0/512 | .7064 | .2909 | .5477 | .5284 | .3606 |
| 31 | 0/512 / 0/512 | .7192 | .2991 | .5732 | .5703 | .4182 |
| 32 | 0/512 / 0/512 | .7193 | .3153 | .5775 | .5738 | .4211 |

Each seed has 17,287 gold nodes, 39,550 typed edges and 13,051 ordered edges across these 512 graphs. Node-presence F1 remains .9761 / .9870 / .9793, which is insufficient evidence for semantic recovery. The compact summary includes precision/recall, graph-size support, entity-equivalence scores, conditional slot counts, and all privileged replacement ceilings. The lossless localization archive contains per-graph/per-relation false positives and false negatives.

For raw decoding, slot labels are correct on 19,628/20,118, 19,627/20,052, and 20,462/21,074 correctly predicted edge pairs. These conditional denominators exclude missed and spurious edges; the much lower ordered-edge F1 includes those failures. Correcting any single predicted component with gold—presence, type, categorical value, copy, edge set, or slots—still produces 0/512 complete graphs for every seed. Even replacing all node attributes together leaves complete recovery at zero. These privileged diagnostic ceilings establish that more than one output component fails; they do not prove which component can be learned or authorize a gold-conditioned inference path.

No new threshold was fitted on fresh labels, and no optimizer step or checkpoint selection occurred. The known-renderer transfer gate is **failed**. All dependent alpha-renaming/heldout-renderer studies and additional training stop here. This does not show failure under adequate broader exposure: the model saw only eight unique constructions, despite 32,000 presentations per seed.

## Timing and reproducibility receipts

The fresh actor inference totals 18.61 seconds; the prelaunch-provenance-to-final-artifact timestamp interval is 39.07 seconds including setup and repeated compressed export. The corresponding main interval is 826.86 seconds (13.78 minutes), versus 778.58 seconds of measured optimizer execution. These file timestamp intervals are process-occupancy proxies, not profiler-derived GPU kernel times. Conservative compute charges are 14 minutes for main and 1 minute for fresh evaluation, plus the separately retained brief profile. No further semantic GPU work is authorized or running.

Reconstruction commands:

```bash
PYTHONPATH=src python research/tools/stage11-semantic-text-localize.py research/results/stage11/semantic-text/fresh.json.gz research/results/stage11/semantic-text/fresh-localization.json.gz
python research/tools/stage11-semantic-text-summary.py research/results/stage11/semantic-text/fresh.json.gz research/results/stage11/semantic-text/fresh-localization.json.gz research/results/stage11/semantic-text/fresh-summary.json
```

The source-to-result chain is acquisition source `5a80a4a`, frozen-transfer source `7936417`, per-file hashes in the raw archives, and SHA-verified immutable final checkpoints. Subsequent edits concern archived analysis/reporting only. The independent review reconstructs metrics and threshold provenance separately from the runner.
