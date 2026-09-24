# A12 frozen localization of the two reads

Prospective inference-only diagnostic on A11's frozen6000-update checkpoint. No further training is authorized by this diagnostic. IID was acquired, but N64/D8 missed the gate and deep task/suffix accuracy remained poor despite high mean-route argmax accuracy. The question is where mixing occurs, not which intervention wins.

## Four fixed policies

1. Unchanged learned soft record read and soft destination-node read.
2. Record hardening only: replace each head's record distribution with one-hot at that head's own learned-score argmax; compute the destination read as usual from the resulting key.
3. Destination hardening only: retain the learned soft record read; replace each head's destination-node distribution with one-hot at that head's own score argmax.
4. Both hardenings.

These are supplied-kernel interventions. They add hard selection not acquired by the soft model and cannot retrospectively satisfy A11's acquired-soft competence gate. No oracle, target, adjacency mask, or exact public-neighbor selector chooses the hardened index. Ties follow the same deterministic PyTorch argmax. Both stages remain per-head, not an argmax of their mean. No scale sweep, threshold search, retraining, or condition-specific policy selection.

Per-head hardening can still combine payload blocks from different nodes across heads. Export the fraction of nodes where all heads choose the same record destination and the fraction where all choose the same destination-read node. These event/step fractions are repeated across the head axis for uniform archival shape. Do not interpret either hardened policy as a single global pointer unless the observed head agreement supports that statement.

## Instrumentation and comparisons

Source inspection shows record weights mix destination identity keys, then a normalized learned projection of that mixture drives a second softmax over node payloads. The second read mixes current per-head payloads before the unchanged residual local update/readout. Instrument both distributions' pre-hardening target/max probability masses, used target masses, and per-head argmax correctness. Export destination-key mixture MSE relative to the selected record's key and relative to the symbolic successor key. Export payload mixture MSE relative to each head's own destination argmax payload and relative to its symbolic successor's current payload, plus selected-payload energy. These are local mixing errors against the current hidden state, not error against an imagined ground-truth hidden representation. Readout exact task, all-node, complete suffix and mean diagnostic route metrics remain separately reported.

Preserve observations for every event, reverse execution step and head, averaging over nodes only. Also preserve summed payload error conditional on correct destination argmax and its correct-node count, permitting a properly weighted conditional error. Never average zero-support conditional ratios. Compute relative payload distortion as total MSE divided by total selected-payload energy with explicit zero-energy handling; raw numerator/denominator remain archived. Gold routing is posthoc instrumentation only. Changing posthoc oracle successors must leave all output logits/routes exactly unchanged.

Compare each intervention with unchanged on the same events: fixed/rebroken task counts, complete suffix, route correctness, read masses and key/payload distortion by execution depth. Hardening can change downstream scores/hidden states, so effects are total consequences of that intervention, not additive or independent causal components. Argmax path correctness alone remains insufficient evidence of successful value transport.

## Population and frozen identity

Main512 fresh graphs for each shape N32/D4/K4, N64/D8/K4, N128/D32/K8. Data namespace221M+shape×100k+offset; record-order namespace222M+shape×100k+offset. All four policies share identical public graphs, shuffled records and labels. Fixed batch16. Record order uses a separate explicit generator and never depends on policy/output. Preserve shared gold/successor/start/relation/value arrays once and per-policy predictions/routes/metrics/observations separately; uint8 IDs fit the declaredN≤128, floats retain float32. No redundant public arrays across policies.

Checkpoint SHA256 `f78aa380ddb2bb7e18234f73c5de751f5ebea2d457211ccc35591c62dd48609f`; initial/final tensor SHA256 must equal `ae43f4a8d86b1c491f5719f0fe7a08efada6adf8e446a674cfd50e4eabefc2ee`. Zero optimizer updates. Existing model/runner source and historical results remain untouched; a separate instrumented forward copies the same arithmetic. Mechanical tests require exact zero-change logits/routes/weights/edge masses, no mutation under any policy, zero own-argmax mixing error at each hardened read, and posthoc-oracle output independence.

## Budget and stopping

Existing joint uninstrumented forward is≈15seconds for512events. Four policies plus smaller shapes suggest≈65seconds before instrumentation/export; provisionally estimate90–140seconds and cap180 for main. Profile first on32 largest-shape events×four policies, disjoint231M/232M namespaces, cap60 including export and durable receipt. Profile is timing/contract validation, not policy selection. Main launch requires measured budget approval and separate root release after profile. If unchanged equivalence, frozen tensors, exact hash checks, or instrumentation contracts fail, stop before main. No automatic follow-up experiment or acquired gate promotion.
