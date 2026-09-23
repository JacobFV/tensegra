# A06: concentration and content selection confirmation

Prospective confirmation selected after A04/A05-v2/A05b development. Historical v1 A05 confirmation was superseded before execution; its coalescing generator is not used. No further scale search is permitted. Source mechanisms remain `6704bd8`; this registry and configs are frozen before final outcomes.

## Hypotheses and scope

1. Value-supervised dynamic content selection acquires fresh code matching under supplied graph grounding and a supplied reverse relation/code schedule.
2. Finite soft averaging can impair deep value propagation despite correct mean-head routing maxima; increasing content-score concentration can improve the same frozen learned model.
3. Keyed neighborhood context has a second address-concentration boundary. Matched sharpening may eliminate any apparent graph-bias advantage. Strong baselines remain primary, including when they dominate.

This does not test learned planning, arbitrary latent grounding, public-text induction, or an ordinary unconstrained transformer parsing graph tokens. Exact-gather and masked learned neighbor attention are algebraically equivalent here and count as one baseline. Context uses supplied neighbor addresses and identity-initialized address projections. All methods except no-graph receive identical graph information. The blockwise bijection family is supplied and preserves N/K distinct identities beyond the first step; it is not a learned guarantee.

## Frozen design

Initialization seeds601/602/603, each four paired arms: soft graph bias initialized8 and trained, exact-gather/hard neighborhood attention, keyed neighborhood context, no-graph information ablation. Workspace1024,8heads, unchanged AdamW .0003/weight decay .0001,1000updates,batch16, fresh N32/K4 graphs and D1–4 cycling. Labels supervise all-node suffix values, never attention routes. Node keys and repeated attribute codes are fresh continuous64-dimensional public features. Relations are typed/directed; ordered relation/code sequence is supplied. No answer or future state enters forward.

Training seeds141,000,000+initialization_seed×100,000+step. Monitoring uses142,000,000,256events on IID and N64/D8 only. Final confirmation uses151,000,000+condition data_group×100,000+batch offset,1024events/cell, shared across arms/seeds and interventions. Semantic constructions are generated independently from development; shared evaluations are paired, not independent new samples per seed. Training, monitoring and final namespaces are disjoint. No checkpoint or threshold selection: final1000-update checkpoint only. Curves cannot change this protocol.

Nine unchanged-scale cells separate IID, N64, D8, N64/D8, reserved relation composition, K8, D32 alone, N128 alone, and joint N128/D32/K8. IID paired controls swap the first instruction, supply the wrong instruction, zero content scores, consistently permute node coordinates, use degree-preserving wrong topology, permute identity correspondence, and set structural strength to zero. Zero strength affects only the soft-bias arm; its no-op in other arms is labeled, not a graph-removal intervention. Fixed-K corruption preserves regular support; missing/spurious edges are excluded rather than padded into false neighbors. Wrong graph controls measure reliance/information loss, not recoverability of unobserved edges.

Frozen content-scale16 interventions use identical IID, D32-only and joint-stress events in every arm. Context additionally receives both content16 and address16 on those same cells. Original learned scales always remain reported. These are supplied inference policies, not learned sharpness or changes to graph strength. Address sharpening changes retrieved attributes and payloads. No optimizer update accompanies any intervention.19cells per non-context arm,22context cells; final outcomes remain paired.

## Primary analyses

Exact final task accuracy and complete suffix-value trajectory on fresh joint-stress graphs: within-arm paired sharpening effects; soft versus exact gather and strongest keyed context. Keep all three seed outcomes and paired differences. A restricted acquisition result requires IID exact task≥98% each seed; deep competence requires joint-stress≥98% each seed under its named scale policy. These thresholds do not pass historical return/semantic gates or authorize composition. A sharpening benefit is supported by a positive paired event-bootstrap95% interval with all three seed differences positive; compare the selected intervention with its own unchanged arm. An attention-specific advantage requires improvement over the strongest equal-information comparator, not merely no-graph.

Secondary: all-node values, complete suffix, mean-head argmax path (diagnostic only), selected/correct-edge mass, supplied-topology consistency, task conditional on an instruction swap changing terminal identity/answer, intervention original versus supplied-target agreement, learning curves, hardware latency and memory. Report node-count and depth shifts separately. Actual cost per correct example and per1024forward events accompany allocated/autograd/nonzero-gradient parameter counts; equal allocation does not imply equal compute. Dense attention/gather value products imply no sparse speedup claim.

Event bootstrap respects shared examples across arms/seeds; seed variation is reported separately. Report rare errors and supports without one-in-a-thousand reliability claims. No uninspected heldout result may tune another scale. A failure is retained; follow-up would be a new exploratory protocol and fresh confirmation.

## Budget, stopping and provenance

Existing largest-shape profile and v2 development suggest150–250seconds per four-arm seed including expanded final evaluation/export. Request hard300seconds per seed,900total, released separately by coordinator. Do not launch automatically. No new model mechanism requires a redundant GPU profile. Record actual startup/train/eval/export occupancy, CUDA peak allocation and process RSS. If a process exceeds cap, retain partial output and failed status rather than silently extending. Preserve compact predictions/targets, event coordinates, hashes, immutable remote checkpoints, config/source receipts, and independently reconstruct metrics. No autonomous composition or supervision withdrawal follows automatically.
