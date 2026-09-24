# S22 complete frozen-boundary diagnostic

All six conditions completed and the pinned full-matrix analysis passed. On omitted 3x4, every policy/backbone remains **0/512 complete graphs and 0/512 exact relations**, including the full gold NODE prefix. Correcting the early node interface alone is insufficient for these frozen models. This diagnostic does not change the failed S21 development gate or permit its conditional confirmation.

## Scope and cost

Both immutable S21 endpoints, seven inspected DEV cells of 512 events, three predeclared privileged policies, unchanged strict codec, no optimizer, no threshold fitting, and no confirmation access. Public text remains unchanged; supplied count, kinds, and full node semantics are explicit oracle inputs. Main charged 372.98 seconds; profile 34.29; total **407.27 GPU seconds**. The whole-process main cap was 1160 seconds. Independent raw replay has passed; aggregate comparison review is separate.

## Complete per-cell results

Each entry is **exact nodes / exact relations / complete graph**, out of 512. All component scores receive zero credit for an invalid final graph. In the full-prefix condition, the correct NODE records were supplied on all 512 events: lower scored exact-node counts measure final validity, not incorrect supplied nodes. Exact relations require both edge and ordered-slot equality.

| Backbone / policy | 2x4 | 3x3 | 3x4 heldout | 4x3 | 4x4 | 5x3 | 5x4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original / count | 0/0/0 | 510/507/507 | 0/0/0 | 510/507/507 | 489/456/455 | 0/0/0 | 0/0/0 |
| original / kinds | 0/0/0 | 510/507/507 | 0/0/0 | 510/507/507 | 489/457/455 | 0/0/0 | 0/0/0 |
| original / prefix | 81/0/0 | 510/507/507 | 421/0/0 | 511/508/508 | 510/474/474 | 227/0/0 | 452/0/0 |
| broad / count | 510/504/504 | 511/505/505 | 0/0/0 | 508/502/502 | 471/436/436 | 507/504/504 | 473/437/437 |
| broad / kinds | 510/504/504 | 511/505/505 | 3/0/0 | 509/503/503 | 472/437/437 | 507/504/504 | 474/437/437 |
| broad / prefix | 511/504/504 | 511/505/505 | 311/0/0 | 512/506/506 | 508/470/470 | 510/507/507 | 510/472/472 |

Public-only complete counts in the same cell order: original **0, 507, 0, 507, 455, 0, 0**; broad **504, 505, 0, 502, 436, 504, 437**.

## Paired counterfactuals

These are within-event interventions on frozen models, not comparisons between independently learned repairs. The JSON retains all 84 policy/comparator/cell tables and 10,000-draw descriptive bootstrap intervals (seed 22023), using shared draws over sorted semantic IDs within each cell. Intervals are conditional on these fixed models and inspected events.

| Backbone / cell | Prefix vs public gains | Losses | Net | 95% conditional delta, pp |
|---|---:|---:|---:|---:|
| original / 2x4 | 0 | 0 | 0 | [0.000, 0.000] |
| original / 3x3 | 0 | 0 | 0 | [0.000, 0.000] |
| original / 3x4 | 0 | 0 | 0 | [0.000, 0.000] |
| original / 4x3 | 1 | 0 | 1 | [0.000, 0.586] |
| original / 4x4 | 19 | 0 | 19 | [2.148, 5.469] |
| original / 5x3 | 0 | 0 | 0 | [0.000, 0.000] |
| original / 5x4 | 0 | 0 | 0 | [0.000, 0.000] |
| broad / 2x4 | 0 | 0 | 0 | [0.000, 0.000] |
| broad / 3x3 | 0 | 0 | 0 | [0.000, 0.000] |
| broad / 3x4 | 0 | 0 | 0 | [0.000, 0.000] |
| broad / 4x3 | 4 | 0 | 4 | [0.195, 1.562] |
| broad / 4x4 | 34 | 0 | 34 | [4.492, 8.984] |
| broad / 5x3 | 3 | 0 | 3 | [0.000, 1.367] |
| broad / 5x4 | 35 | 0 | 35 | [4.688, 9.180] |

Count assistance changes no complete counts in any cell. Kind assistance adds one complete graph each to broad 4x3 and 4x4; other complete counts are unchanged. Full prefixes improve some trained cells but recover none of the original model’s untrained 2x4, 5x3, or 5x4 populations, nor either model’s omitted 3x4. Every heldout paired transition is 512 failures remaining failures; empirical delta intervals [0,0] do not establish population equivalence or a capacity theorem.

## Heldout boundary

| Backbone / policy | Valid | Exact value | Exact copy | Typed-edge macro F1 | Ordered-edge macro F1 |
|---|---:|---:|---:|---:|---:|
| original / count | 452 | 0 | 0 | 0.5194 | 0.2439 |
| original / kinds | 372 | 195 | 0 | 0.4469 | 0.2122 |
| original / prefix | 421 | 421 | 421 | 0.5580 | 0.2463 |
| broad / count | 346 | 0 | 0 | 0.3208 | 0.1510 |
| broad / kinds | 299 | 267 | 3 | 0.3583 | 0.1725 |
| broad / prefix | 311 | 311 | 311 | 0.4094 | 0.1872 |

With supplied kinds, original has zero exact-copy graphs and broad has three, despite 195 and 267 exact-value graphs respectively. Full prefixes remove that node-field uncertainty by construction, yet relation exactness remains zero. Partial relation overlap survives; this is not evidence of absent semantic information. Under full prefixes, invalid outputs include original 87 dangling-reference, 3 duplicate-edge and 1 multiple-pair-slot cases; broad has 192, 4 and 5 respectively. Valid outputs also fail relation conjunction: 421 original and 311 broad valid heldout graphs remain inexact.

A deterministic example is the lexicographically first heldout semantic ID, `0075944ca6aca5a658ac70218ad3e34bf5cd1e42eddd1ea7e6d8042947814d04`; it fails under every policy/backbone. Full raw records, supplied-boundary counters and packed predictions/targets remain in the immutable archive for inspection.

## Decision

Stop this frozen diagnostic branch. Do not run the ineligible S21 confirmation, a calibration/exposure sweep, or a learned node-only repair on the assumption that it will resolve transfer. The declared strongest intervention leaves the relation-generation boundary unresolved. Any future experiment needs a separately registered, discriminating acquisition-versus-combination question about learned relation generation; S22 alone identifies no demonstrated repair to promote. Preserve known-motif acquisition, partial relation learning, and the zero heldout result together.

## Bindings

Main config `911a70787dfd31bd1225c04cc2d7c4c2a8b9d721396c12a754d47f1cacfb910e`; manifest `16db751e339d6c19a739c8adaf007bbc11ea7d6579af771c0131d52edd86fc7d`; analysis source `7ad77fa0b671238382d994f44b3edcdbe2f2cec05eff9cb77d650c7939f63ff3`. Archive commit `d3ad4396`, scientific source `7a4c2923`, prospective pinned analysis root `83cf2a22`. Analysis JSON artifact SHA `b0ad65aa10359e729570321762a4744e00cc3f5843f176680469cc0e938584b0`. Full machine-readable summaries and artifact SHA inventory: `research/results/campaign-01/semantics/s22-analysis.json`.
