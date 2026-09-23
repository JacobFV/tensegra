# A05-development-v2: learned routes survive; finite reads still mix payloads

The separately versioned block-permutation generator prevents the identity coalescence discovered before the original A05 draft ran. After the first hop, exactlyN/K identities remain distinct under arbitrary subsequent instructions. That property is programmed into the generator and verified independently of any learned model; it is not a learned memory result. The original A04 results and unrun A05 draft remain intact.

In one fresh exploratory seed501, all three graph models again acquire100% task accuracy through the registered depth8 conditions by25updates. At the harder N128/depth32/K8 condition, the unchanged1000-update models diverge:

| Interface | Task | All-node/depth values | Complete suffix trajectory | Mean-head argmax path |
|---|---:|---:|---:|---:|
|Soft graph bias|67.19%|92.91%|15.23%|100%|
|Exact-address neighbor attention|74.22%|94.92%|25.78%|100%|
|Keyed neighborhood selector|21.88%|63.87%|0%|100%|
|No graph|16.41%|—|—|0%|

These are256 fresh development events, not three-seed confirmation. Exact gather/hard masked neighbor attention is one algebraic comparator. Keyed context includes a supplied hard neighborhood and a known-key address prior; it is not an unconstrained graph-text model.

The prospectively registered frozen **content** scale16 intervention changes no weights. It restores100% task, all-node and complete suffix-value accuracy for soft and exact-gather models. The keyed selector improves to52.34% task and90.42% all-node accuracy, with1.17% complete suffix trajectories. Its mean-head path remains correct. The intervention is distinct from increasing graph biasλ in A03.

Thus the narrow development finding is that correct diagnostic route maxima are insufficient for accurate repeated weighted payload transfer. Content-score sharpening repairs two interfaces on these events. It does not prove every individual attention head was correct, nor identify a unique nonlinear dynamical explanation. There is no soft-attention advantage over exact-gather neighbor attention.

Training losses saturate near1e-5 by the final updates; deep accuracy peaks earlier and does not improve with1000updates. Blindly extending the same short-horizon training is not justified. Before confirmation, a separately registered frozen address-scale diagnostic will test whether the keyed selector's extra retrieval distribution explains its remaining weakness. Both original and stronger baseline results will be retained.

Source53bf204, width1024, four matched1000-update runs,16,000 graph draws each. Full controller occupancy149.54seconds (soft37.27, gather34.18, keyed context44.93, no graph33.12); preceding profile3.76seconds. Nine CPU contract tests pass. Raw predictions/targets, instruction counterfactuals, checkpoints, source/config hashes and curves are preserved. Independent raw audit is in progress; no composition or rare-error claim follows.
