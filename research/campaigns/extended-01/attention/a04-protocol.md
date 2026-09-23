# A04 development: learned content choice among typed neighbors

Registered before any A04 model outcome. This promotes the [content-selector design](content-selector-proposal.md) after A03 confirmation. It is exploratory acquisition, not confirmation or learned program planning.

## Hypothesis and observable contract

Soft structural compatibility can combine learned matching of fresh content codes with supplied graph edges. The main question is whether it offers any useful advantage over expressive neighbor attention or strong structured address retrieval. All receive the same graph, node keys, node attributes, payloads, start node and ordered `(relation, desired attribute)` sequence. No gold route or future suffix enters forward inference.

Each graph has three directed relations and exactly K outgoing neighbors per node/relation, one from each attribute group. Nodes are randomly assigned to groups; each group shares a fresh normalized64-dimensional code. N32/K4 leaves eight nodes per code, so the final instruction code does not reveal the destination. Payloads are independent16-class labels. A programmed reverse schedule and privileged all-node suffix-value targets are common supplied priors. The neural model learns matching through the value loss alone; there is no route auxiliary initially.

The workspace is1024-dimensional with8heads. Separate learned query/key transformations start at random and consume public attribute vectors. Normalized dot products have trainable initial scale8; structural strengths initialize8 per relation/head. This is an explicitly disclosed content-scoring geometry, not a learned symbolic variable binding mechanism.

## Arms and controls

- Soft adjacency logit bias with learned content score.
- Exact address gather plus the same learned content selector. This is algebraically equivalent to masked neighbor attention/message passing; count it as one strong comparator, not independent methods.
- Keyed neighborhood selector: identity-initialized cosine address retrieval obtains each neighbor's attribute/value, then the same learned content scorer selects among supplied addresses. This carries the same known-key prior as A03. It retains a hard supplied-address neighborhood mask and is not an unconstrained transformer parsing graph tokens. Its extra retrieval can fail; the exact-gather comparator prevents a false advantage caused merely by weakening address access.
- No-graph content attention: informational ablation, not equal-information primary control.

Paired instruction swaps keep graph/payloads fixed and recompute correct targets. Report swap accuracy separately on cases whose exact terminal node changes and cases whose payload answer changes; paths can merge and labels can collide. A wrong-instruction intervention retains original targets and separately scores agreement with the actually supplied instruction. Content-null evaluation removes compatibility scores, testing reliance on acquired selection. Consistent node permutation is restored before scoring. Verify unique local matching, repeated global codes and argument/relation order mechanically. Longer depth, larger N, larger K and reserved adjacent relation composition(2,2) remain distinct conditions.

## Acquisition budget and selection

Development seed301, fresh procedural training seeds61,000,000+step, evaluation62,000,000+group×100,000+offset. Batch16,1,000updates, depths cycling1–4 at N32/K4, AdamW learning rate.0003 and weight decay1e-4, preserving A03. Record steps0/10/25/50/100/250/500/1000 on256 held-out examples per cell. Every arm uses identical training/evaluation draws and common initial tensors where parameter roles coincide. Training never includes adjacent relation pair(2,2).

A mechanical resource profile first uses seed399,12updates and one64-example N64/D8/K8 cell. Its outcomes cannot select a scientific recipe. Root must release the GPU slot and approve the measured occupancy estimate before four-arm development. Preliminary development cap is300seconds including inference/export; revise prospectively from the profile if necessary. No confirmation data exists yet.

Primary development metric is complete start-payload accuracy at N32/D4; exact mean-head argmax path and selected-neighbor mass distinguish routing from payload errors. If no strong arm reaches95%, inspect acquisition rather than proceed to large OOD claims. If an arm is improving by at least3percentage points between500 and1000 or training loss falls at least20% over that interval, a newly registered exposure extension may be considered; it is not automatically authorized or a retrospective pass. Otherwise prefer a discriminating diagnostic to blind runtime multiplication. Promotion to confirmation requires an informative acquired mechanism and a new frozen three-seed protocol with fresh constructions.

## Provenance, limitations and next branch

Store configs/source/initial/checkpoint hashes, all seed outcomes, compact predictions/targets/diagnostic routes, loss curves, allocated parameters, CUDA peak allocation, process RSS and forward/process runtimes. Procedural draws are counted separately from deduplicated canonical graphs, which are not currently measured. The source remains versioned independently of A03.

Exact gather and schedule are programmed guarantees. Content matching and payload processing are learned. Soft/hard performance alone does not imply learned planning or language induction. Corrupted graphs may remove essential information; do not add a recovery claim without a publicly identifiable task. After acquisition, evaluate fair graph corruption and learned-grounding extensions only when they answer a new question. No autonomous composition follows from this development study.

## Profile receipt and development allocation

The frozen2ba70af profile completed in3.64seconds full process occupancy,1.485seconds inside the runner. The single64-case N64/D8/K8 forward took.0343seconds. Peak CUDA allocation272,366,080bytes; process peak RSS2,009,552KiB;4,306,986 allocated parameters. This mechanical seed's task outcomes are not selection evidence.

Using this condition cost together with A03's measured1000-update-equivalent training costs, the four-arm study is projected to require roughly120–200seconds, including80 evaluation cells per arm and exports. Startup dominates the12-update profile, so linear extrapolation of its total wall time is not a useful training estimate. Retain the prospective300second controller ceiling. If reached, preserve partial artifacts and register a completion decision without changing any selected outcome or dropping an arm. Root must separately release development; profile completion is not a main-run authorization.
