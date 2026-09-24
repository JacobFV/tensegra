# A13 frozen shared destination addresses

Prospectively registered after A12. Test whether enforcing a common destination distribution across value heads changes moderate/deep transport. This is a supplied-kernel address-consistency intervention, not retraining, an acquired-soft gate pass, or an already-established explanation of failure. A12's queried mean-path exactness and all-node/head correctness have different denominators; their gap alone does not establish query-path head error.

## Four fixed policies, one original record stage

Keep A11's frozen original soft record stage, destination scores/probabilities, residual update, normalization and readout unchanged. For each node and reverse step, compute every head's original destination probability distribution, then take the arithmetic mean **of probabilities**, matching the currently reported mean-head route. Do not average logits and do not test either alternative after observing outcomes.

1. Unchanged: retain separate original destination distributions per head.
2. Shared soft: apply that same mean probability distribution to every value head.
3. Shared hard: apply the same one-hot argmax of that mean probability distribution to every value head. The full payload vector therefore comes from one common model-selected node.
4. Oracle common: replace destination reads with one-hot at the exact symbolic successor for every head. This uses privileged routing labels and is solely a routing intervention reference, not learned behavior, a Bayes ceiling, or a general task-competence claim. The unchanged residual/readout can still fail.

The first three policies use no gold targets, oracle successor or gold query trajectory to choose an address. Changing those annotations must leave their logits/routes exactly unchanged. The oracle common policy is intentionally exempt and prominently labeled. No record hardening, scale sweep, learned weight change, condition-specific policy selection or follow-up tuning.

## Matched support and observations

Retain A12's all-node event/step/head record and destination target/max masses, used masses, original/used argmax correctness, key/payload local MSE and payload energy. Report original destination-head agreement separately from agreement after sharing. Export maximum address-distribution difference from the first head; it must be zero under all shared policies. Shared-hard/oracle-common payload MSE versus the used own-argmax current payload must be zero. As before, local payload references are current model states, not imagined ground-truth hidden vectors.

Additionally retain correctness for each original record head, original destination head, used destination head and mean route along the **same queried oracle trajectory**, plus corresponding all-head agreement. Select this diagnostic trajectory only after all forward model steps finish; these gathered annotations never feed any read. Store event×reverse-step×head observations, preserving full-path conjunctions and per-step means as different summaries. Compare queried mean-route full-path correctness with queried per-head full-path correctness, and per-step means with per-step means. Do not compare a path conjunction directly with an all-node/head average. Export all-node summaries separately, never conflating their support.

Compare paired task fixes/breaks, all-node and complete-suffix correctness, head agreement and local transport distortion. Shared soft changes which mixture each head sees without making an exact choice; shared hard additionally removes probability mixing. The oracle reference separates model-address selection from transport/update/readout limitations under privileged exact routing. No outcome retroactively promotes A11's failed moderate-depth gate.

## Frozen data and checkpoint

Main512 paired fresh events for each N32/D4/K4, N64/D8/K4, N128/D32/K8, batch16. Previously unused attention namespaces241M+shape×100k+offset for data and242M+shape×100k+offset for record order. All four policies share the same graphs/records/targets. Profile32 largest-shape events uses disjoint251M/252M namespaces. Public arrays stored once, per-policy arrays separately.

Same A11 checkpoint SHA256 `f78aa380ddb2bb7e18234f73c5de751f5ebea2d457211ccc35591c62dd48609f`; initial/final tensor SHA256 `ae43f4a8d86b1c491f5719f0fe7a08efada6adf8e446a674cfd50e4eabefc2ee`; zero updates. Separate A13 source preserves historical A12 bytes. Tests require exact unchanged parity at width1024 and tiny mechanical dimensions, all-policy frozen tensors, oracle-independence for three nonoracle policies, common address equality/zero hard-read mixing, mean-probability semantics and queried-path metric agreement with existing route metrics. Mechanical export round-trip is checked.

## Budget and review

A12's comparable four-policy main cost114.503seconds; A13 adds small query-trajectory observation arrays. Provisional estimate120–150seconds, maincap180; profilecap60. These are preparation bounds, subject to independent source review and measured profile before main release. Root must explicitly release profile and main separately. No GPU job is launched by registration. Stop on equivalence, frozen-hash, label-isolation or shared-address contract failures; preserve failures and make no automatic follow-up experiment.
