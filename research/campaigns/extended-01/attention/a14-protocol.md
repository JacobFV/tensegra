# A14 fresh three-seed shared-read confirmation

Registered before any new confirmation forward. Confirm the A13 frozen shared-hard versus original task gain across three fresh initializations, not a new learning mechanism or a preferred winner. Preserve all prior failures and A13's499/512 (97.46%) shared-hard joint development result, which is below the98% competence threshold.

## Fixed training and populations

Train fresh seeds1401/1402/1403 from scratch for exactly6000 updates each; retain every seed and fixed endpoint even if the original policy is already strong or an intervention fails. Same width1024 all-record architecture, AdamW learning rate.0003/weight decay.0001, batch16, clipping1, generated training stream171M+step, N32/K4 and depth1+step%4, dense all-node suffix loss. Each seed receives96000 presentations; public training examples are paired across seeds, while initialization/record-order RNG varies by seed. No early stopping, retries for bad performance, adjusted scales or selected checkpoint.

**A07/A11 objective denominator:** there is no supervised routing loss, either per-head or mean-head. The source applies `cross_entropy(logits.flatten(0,2), gold.flatten())`, averaging payload-class loss across batch×reverse-step×node entries. Depth cycles1–4; exact successor/path labels never supervise attention scores. Mean-head route correctness is an evaluation-only diagnostic. A14 preserves this objective unchanged. The finding concerns a shallow-supervised payload objective interacting with a supplied multi-head read interface, not proof of generic memory failure and not failure to satisfy an explicit per-head route target.

Development curves only at1000/3000/6000, on173M IID/moderate populations256events each. Evaluation runs in a saved/restored RNG context with explicit monitor-order seed263M+step, so observations cannot change subsequent updates. This is the same architecture/optimizer/example recipe as A11, but not a claim of bitwise replay of its segmented monitoring schedule. A mechanical test verifies inserting a monitor leaves final model/AdamW/RNG exactly unchanged. Final model/AdamW/CPU+CUDA RNG state is archived.

Confirmation uses1024 fresh events per core shape N32/D4/K4, N64/D8/K4, N128/D32/K8. Namespaces271M+shape×100k+offset for data and272M+shape×100k+offset for record order; batch16. All five policies and all three seeds see identical heldout graphs/records/labels. No single-axis expansion. Compare exact decompressed public-array hashes across all seeds and references before pooling. No heldout outcome changes training or policies.

Public pairing hashes include full keys, attributes, instructions and adjacency, as well as gold/successor/start/relation/value arrays. Stream the large public tensors into hashes rather than duplicating them in every archive. Record-order hashes are additionally compared across all three trained seeds; supplied-neighborhood references do not consume record order and therefore omit that separate field.

## Five fixed policies per frozen model

1. Original learned soft record and per-head destination reads.
2. Both hard: per-head model-score argmax record read then per-head destination argmax, exactly the A12 both-hard policy, predeclared as its strongest deep development hardening. It does not force agreement across heads.
3. Shared soft: original soft record stage, arithmetic mean of original destination probabilities reused for every value head.
4. Shared hard: original soft record stage, one-hot argmax of that same mean probability distribution reused for every head.
5. Oracle common: privileged exact symbolic successor read for every head. Reference only, never learned competence or a universal ceiling.

No mean-logit alternative, scale grid or policy selection. Historical A12/A13 forwards are exact arithmetic references for the new five-policy wrapper. Instrumented and uninstrumented versions must return exactly identical logits/routes/weights. Gold annotations cannot affect any of the four nonoracle policies. Query-trajectory observations remain post-forward and separate from all-node means. Under both-hard, fields describing destination scores refer to scores after its hardened record stage; they are not labeled an unchanged-stage counterfactual.

## Primary effect and competence rules

Primary effect: shared-hard minus original exact joint task accuracy. A replicated positive-effect result requires a strictly positive paired gain in **each** fresh seed and a95% paired event-bootstrap interval for the mean three-seed gain with lower endpoint above zero. Use10000 replicates and fixed bootstrap seed291M. Resample the same1024 event indices jointly across all seeds/policies; do not pretend3072 graphs are independent and do not bootstrap three seeds as a population estimate. Report individual seed gains/ranges even when the pooled interval is narrow. Failure of either rule remains failure; no seed removal.

Co-primary mechanistic quantity: per-seed P(task correct | complete queried mean route correct), calculated from actual task×route contingency tables. Report each support count, plus the pooled intersection divided by pooled route-correct support; no product of marginal probabilities. Shared-hard transport conditional passes the restricted mechanistic criterion only if every seed has≥98% conditional task accuracy with nonzero route-correct support. A zero denominator is undefined and cannot pass. Report bootstrap intervals from the same shared event resamples and keep conditional and unconditional performance separate.

Competence is a separate gate: all three seeds must meet≥98%IID,≥95%moderate and≥98%joint exact task for the specified policy. Passing shared hard is an **engineered inference-policy** result, never acquired-soft competence. Acquired-soft competence is evaluated only under original policy. Positive effect or perfect conditional transport with sub98% joint route accuracy cannot satisfy the joint competence gate. Secondary outcomes are complete all-node suffix, queried full mean/head paths, head agreement, paired fixes/breaks, all-node values and local payload distortion; report them without choosing a replacement primary metric. No universal reliability or unique-attention claim follows.

## Strong engineering references

Evaluate all nine frozen A06 soft/gather/keyed-context checkpoints on exactly the same fresh1024-event/core-shape population. Fixed successful policies are content16 for soft/gather and content16+address16 for keyed context. Hash-enforced configs name every checkpoint, with no reference selection. These models have different training histories (1000updates, seeds601–603, supplied neighborhood interfaces); label them engineering references, not paired architecture-training comparisons or an unconstrained graph-token transformer. The no-graph ablation is not an equal-information reference. Report task/suffix and descriptive runtime/parameter/memory observations, with public-array equality checked.

## Cost, profile and stopping

Profile new seed1499 for100updates, development32events, then32largest-shape events×all five policies on disjoint281M/order282M. Also measure one batch16 largest-shape uninstrumented forward per policy: one warmup plus three recorded repeats, fresh benchmark283M, recording preparation/warmup/repeat costs separately. The oracle remains privileged here too. Profile outcomes cannot tune a policy, seed, threshold or model. Initial profile cap60seconds, subject to independent preflight/root release.

Provisional per-seed main estimate330–410seconds (6000 training updates plus five instrumented1024-event evaluations), cap480 pending measured profile. Three separately released seeds; engineering-reference estimate45–70seconds/cap120. Root must approve measured caps and release each job; no automatic chaining. Main configs/five policies are frozen before profile. A contract failure stops that run and preserves artifacts; a performance failure never authorizes a rerun or substitute seed.

Record allocated/participating parameter totals, training+monitor time and separately measured monitor time, presentations, training+monitor peak allocation, post-optimizer-cleanup inference peak, process RSS, checkpoint/state/source/config hashes, final frozen tensor equality and atomic process occupancy. Main instrumented timings are not bare inference latency. Profile's explicitly uninstrumented benchmark excludes tokenization/preparation from its forward interval but reports those costs separately; it is descriptive hardware evidence, not a compute-normalized superiority claim. No broad latency claim from three repeats or unequal histories.
