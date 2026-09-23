# A03: frozen confirmation of routing interfaces

Registered after exploratory A01/A02 and before confirmation outcomes. The selected soft intervention initializes the trainable per-relation/head strength at8, with no explicit size correction. Strength learns during training and is frozen during evaluation; it is not constrained to remain exactly8. Architecture and optimizer are unchanged. Training remains N16, depths1–4, excluding adjacent relation pair(2,2), with500updates and batch16. The public reverse schedule and explicit node correspondence remain programmed priors.

## Primary comparisons

Three paired initialization seeds201/202/203. Five arms: soft4 reference, selected soft8, strong structured address context, learned message passing, and hard attention. Hard/message are identical routing operators on clean functional graphs; three seeds do not turn their agreement into independent architectural evidence. The no-graph ablation from development remains context, not a confirmation baseline.

Primary task metric: exact start-node payload at N64/D16. Report all three seeds and paired differences for soft8 minus soft4 and soft8 minus each strong comparator. No significance-dependent selection. Secondary metrics: IID, separate size/depth shifts, new relation composition, all-node and suffix-value correctness, mean-head argmax path, clean-edge mass and log-read margin. The argmax path is a diagnostic, not a causal proof.

Every final cell uses1,024 fresh examples. Evaluation cases are shared across seeds and arms. There are eight distinct generated base conditions; corruption/zero-strength/permutation reuse the corresponding base events, not new independent examples. Condition groups0–7 use data seeds32,000,000+group×100,000. Training seeds42,000,000+init_seed×100,000. Intermediate curves use separate development seeds22,000,000 and two256-example cells; confirmation runs only at the fixed500-update endpoint. No checkpoint selection, threshold fitting or hyperparameter adjustment uses confirmation data.

Primary clean conditions: N16/D4, N32/D4, N16/D8, N64/D16, N16/D4 with the reserved(2,2) composition, N64/D4, N16/D16, N32/D8. Targeted corruption at N16/D4 and N64/D16: degree-preserving wrong permutations, identity-correspondence permutation,25% missing edges with explicit self fallback,25% added edges. Clean answers stay fixed. Missing/wrong edges can remove the information needed to identify clean answers; these conditions measure dependence/robustness, not a promise that soft routing can recover unavailable facts.

Zero λ is a frozen evaluation intervention on soft models, not a separately trained no-graph model. A consistent node permutation at N64/D16 is evaluated on the same events and mapped back to original coordinates before comparison. Exact pointer evaluation on each clean graph provides a programmed reference and should be1.0 by construction; it is not learned performance.

## Compute and capacity

Width1024 throughout. All models allocate4,241,449 parameters, but gradient participation differs. Report allocated parameters, tensors participating in autograd, and tensors with nonzero gradients at the final update separately. These counts are not a proof of functional capacity or acquired learning; hard single-successor attention gives Q/K zero gradients. Context uses a known-key cosine prior and a lossless list of successor addresses; it is more structured than a transformer parsing adjacency text. Message passing uses supplied adjacency and the common learned update, and avoids contentQK computation.

CUDA events bracket model forward calls; report isolated forward seconds, throughput, and forward time per correctly solved example separately from process occupancy and condition wall time. The dense implementations do not establish sparse-compute advantages. Node count and depth are separate compute axes.

The largest-shape random-initialization profile used N64/D16, three128-example cells:2.90 seconds process occupancy,0.89seconds within the runner including export. Steady large-cell model forward cost was approximately0.137seconds per128examples; initial CUDA warmup was slower. Projected full confirmation occupancy is300–450seconds. The root scheduler may release one paired seed (five arms) at a time, with a hard cap of300seconds per seed and900seconds total. This replaces the preliminary600second aggregate estimate before any confirmation outcome, and includes startup/training/inference/export. If the cap interrupts a run, retain it and register completion without changing protocol or results; do not drop incomplete seeds.

## Interpretation and advancement

The confirmation can establish or reject a finite-bias repair and characterize competitive alternatives. It does not presume an attention advantage. A useful explicit-routing result does not establish learned grounding, learned planning or autonomous composition. Any extension to predicted grounding requires a new protocol. No broad historical gate is overridden.

Report seed variability separately from event-sampling uncertainty. For paired bootstrap intervals, resample shared event indices within a condition and preserve the same sampled indices across all seeds/arms. Do not count repeated seeds or corruption views as new independent examples.

Naming: `soft4` and `soft8` denote initial trainable strengths, not permanently fixed learned parameters. The frozen-checkpoint `strength_override=8` intervention is exactly fixed at8. Zero-strength evaluation is meaningful only for soft attention; other arms ignore that flag, so their duplicate zero-strength cells are no-op consistency checks rather than graph-removal ablations.
