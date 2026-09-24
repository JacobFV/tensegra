# Frozen mixed-teacher acquisition: public feature insufficiency

Source: `a9337b71` (full hash and world/reference byte hashes in raw receipt). This is a CPU-only constructive diagnostic; the live600-update experiment is unchanged. Two legal, feasible workshop pairs have **identical complete27-dimensional observation vectors and every row of their67-dimensional candidate matrices**, yet `cheap_first_fallback_v2` chooses different action **kinds**. Candidate catalogues also match. This proves the supplied neural encoding cannot reproduce this teacher everywhere. It does not measure collision prevalence or explain a particular fraction of its development failures.

| Pair | Public semantic difference omitted by encoding | Teacher left | Teacher right |
|---|---|---|---|
| Nonpending incompatibility | One conflict between items `a` and `c`, before any item is pending | `choose_item(a)` | `start_subset` |
| Downstream route weight | Edge1→2 costs1 versus4; current outgoing edges unchanged | `move(1)` | `build_route` |

The first pair contains two categories with two alternatives each. Capacity and funds are4. Items `(category,weight,price)` are `a=(0,1,1)`, `b=(0,2,2)`, `c=(1,1,1)`, `d=(1,7,7)`. The left world permits greedy `a,c`; the right adds only conflict `a,c`, still permits `b,c`, but the supplied greedy teacher declines its irreversible first choice and invokes subset search. The feature encoder reports compatibility **with pending items only**; pending is empty, so the conflict disappears from every candidate vector. Inspection histories are also indistinguishable under these features until this decision.

The second pair has edges0→1=1 and0→2=3 in both worlds, and1→2=1 versus4. Both have a feasible route within budget. Travel price is.002. The teacher's public greedy path costs2 versus5. Its possible savings relative to a one-edge lower bound are.002 versus.008, bracketing the same.00403 estimated tool overhead. Neither downstream edge weight reaches any encoded input: movement candidates expose only current outgoing edges; observation memory exposes counts/statuses, not the map payload. A recurrent policy with the same encoded history cannot recover the missing weight simply by repeating computation.

There is also a separate **target-tie issue**. Two still-uninspected items in one category have identical candidate features. The deterministic teacher chooses the first inventory entry. Both neural scorers produce identical logits for such identical rows; the selected member of a size-k exact tie has probability at most1/k, giving cross-entropy at least log(k). Argmax-first may nevertheless reproduce the teacher order and solve the task. This is an imitation-loss floor, not necessarily an environmental capability failure. Lexical greedy ties can additionally use irrelevant handle spelling. These ties must not be confused with the two genuine action-kind collisions above.

## Verification and limits

`a933-feature-collisions.json` stores complete paired public observations, complete action catalogues, feature vectors, SHA256 hashes, and teacher actions. The separate `--verify` invocation reconstructs vectors/actions from **archived public observations only**, checks both collisions, independently checks feasible item assignments, and independently computes the route cost inequality. It imports the frozen historical world/reference implementations extracted by git into a temporary package. No model, Torch, solver process, GPU, or external data is used. Generation consumed0.0120 local process-CPU seconds; the verification receipt records its separate cost. Git subprocess CPU is minor and not included in that process-only figure. These are constructed fixtures, not empirical estimates from the training distribution.

Reproduce from repository root:

```sh
python3 research/tools/campaign02_feature_collision.py --verify research/campaigns/extended-02/diagnostics/a933-feature-collisions.json
```

Task completion is not proved impossible. For example, an always-tool policy could succeed in both paired worlds. What is impossible is perfect imitation of these differing teacher labels, or reproducing this specific cost-sensitive distinction, from the identical feature inputs. Public JSON is available to the teacher but **does not enter either neural forward path**. Calling those models equal-information comparators to the full-JSON reference requires this qualification.

## Smallest justified next interface, after frozen600 completion

Keep the existing observation/action summaries, but provide a separately versioned **public relational memory** shared by both learned families. Preserve every currently observable item attribute, known directed edge and weight, incompatibility pair, draft/constraint membership, result status/payload/provenance, and state-version dependency. Use typed rows with explicit ordered endpoint/reference links and protected local handle tables; do not turn opaque handle spelling into semantic features. Unknown facts stay absent with explicit observation masks. Bounded one-hot/local reference indices can make this input lossless within a declared world capacity; a neural embedding of those rows is not itself guaranteed lossless.

No greedy-completion flag, shortest-path distance, feasible-subset label, gold action, or solved constraint feature should be supplied. Public syntactic masks remain the same. Both lightweight and recurrent actors must receive the same fact rows: a lightweight candidate-query/context or message-passing reader is a strong comparator; the recurrent workspace can attend to the same memory. The current pointwise lightweight scorer also lacks access to other candidates when ranking one action, so merely storing extra JSON beside it does not repair its forward interface.

This provides a specific later attention question: ordinary keyed reads versus message passing versus relation-induced bias, all using the identical public fact graph and explicit address correspondence. Any improvement over the lossy summary would first be an **input-contract repair**, not evidence for bias superiority. Fit complete-evidence action distinctions on heldout counterfactual fixtures before a large population run; report semantic action classes separately from arbitrary exact-handle/order ties. The live600 curve remains an empirical result under its original representation and should not be retrospectively replaced.
