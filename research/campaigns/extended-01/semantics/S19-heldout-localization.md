# S19 held-out composition: generated-prefix localization

Scope: posthoc diagnosis of the fixed final development endpoint, not recipe selection, confirmation, or a causal intervention. S20 was already frozen. The analysis uses every generated record prefix, including invalid sequences; no model or oracle forward was run. No historical metric is changed.

## Equal public length, different required structure

Both held-out **3×4** and trained-shape **4×3** cells have 512 examples and exactly 54 public tokens per example. Here the first coordinate is predicate arity and the second is fact count. Final complete canonical recovery is **0/512 versus 507/512**.

| Diagnostic | Held-out 3×4 | Trained-shape 4×3 |
|---|---:|---:|
| Valid complete serialization | 391 | 511 |
| EOS appears in retained generated prefix | 399 | 511 |
| Correct generated node count | 313 | 512 |
| Exact entire node-kind sequence | 0 | 512 |
| All canonical node records correct | 0 | 511 |
| Entity-removed kind sequence equals trained 4×3 template | 348 | 512 |
| Generated predicate argument counts `[4,4,4,4,1]` | 328 | 510 |

Removing entity nodes here is solely a structural diagnostic: entity allocation varies with repeated identities. They remain required by every actual score. The correct held-out skeleton contains five arity-three parent predicates plus the unary query; the trained 4×3 skeleton has four arity-four parent predicates plus that query. Generated outgoing argument-edge counts support the same pattern as the node-kind skeleton; this is not merely a total-node-count collision.

For held-out examples, the first canonical node-record discrepancy occurs at zero-based index **8 (17), 9 (153), or 10 (342)**. In **507/512**, the generated node is `ident` where the gold node is the next `pred(parent)`: it continues an argument list rather than beginning the next predicate. The other five first discrepancies are copy-only. No held-out example has the full correct kind sequence.

The entity-removed generated skeleton equals trained 4×3 in 348 cases, trained 4×4 in13, and neither registered template in151. Its edit distance is strictly closer to 4×3 than the true 3×4 template in **483/512**, tied in28, and closer to 3×4 in1. Full kind sequences, with entity allocation retained, occur among actual TRAIN 4×3 sequences in171 held-out cases; this stricter finite-template membership is neither necessary nor sufficient for semantic correctness.

## Where edge-only localization is possible

Held-out3×4 has **zero correct full node-record prefixes**, so there is no clean observed subgroup in which correct node construction isolates a downstream edge-only failure. Do not describe this cell as merely an edge threshold or slot-head problem.

For4×3, 511 have fully correct nodes. Four of those fail the final graph: one is invalid, three valid outputs have edge errors, and one of the three also has a slot error. One additional example has a copy-only node error. This separates a mostly acquired trained-shape interface from the held-out structural failure.

Invalid held-out outputs are108 `invalid_generated_node_reference`,5 `duplicate_edge`, and8 historical-metric pair-slot conflicts. These are retained failures, not repaired prefixes. Missing EOS and invalid reasons are not interchangeable: the generator can terminate on an invalid record before emitting EOS; eight pair-slot conflicts still contain EOS. The trained-shape cell has one invalid-node-reference case.

## Interpretation

Predicting a familiar motif layout is a plausible description of much of this held-out failure: the wrong fourth argument appears at the first predicate boundary, and the resulting node and edge signatures frequently match4×3. This is **not proof that input length causes motif selection**. Length is confounded with other learned surface cues; no counterfactual length intervention was performed. It also does not prove a single mechanism explains all151 non-template outputs.

These are canonical-position measurements, not graph-isomorphism claims. However, the different generated predicate counts/arities show substantive structural discrepancies beyond a mere arbitrary permutation of equivalent node indices. No model-selection threshold, semantic equivalence relaxation, or S20 configuration was altered.

## Reproduction and cost

Run `python3 research/campaigns/extended-01/semantics/S19-heldout-localization.py` from the repository root. The script is stdlib-only. It hashes the final raw DEV artifact, public DEV cache, and actual mixed TRAIN cache. Aggregates live in `research/results/campaign-01/semantics/s19-heldout-localization.json`; compact per-event diagnostics are in the adjacent gzip artifact. It never opens confirmation data.

The final JSON records actual CPU wall time including source loading and artifact export. An earlier exploratory script pass cost4.619309s; final rerun time is recorded in the JSON. Filesystem/worktree creation and interactive source inspection are separate overhead, not GPU occupancy. No training or inference was performed.
