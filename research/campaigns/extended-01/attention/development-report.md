# Attention development: explicit routing comparison

This is an exploratory result from seed 101, not confirmation. The strongest equal-information baselines dominate soft λ=4 at the combined size/depth shift. We retain those baselines.

| Interface | N16/D4 | N32/D4 | N16/D8 | N32/D8 | Held-out (2,2) pair |
|---|---:|---:|---:|---:|---:|
| Soft λ=4 | 1.000 | .988 | .895 | .246 | 1.000 |
| Structured address context | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Message passing | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hard attention | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| No graph | .160 | .168 | .156 | .102 | .188 |

These are results after 500 updates and 8,000 generated graph presentations, with 256 fresh validation graphs per cell. Arms share width 1024, initial parameters, data, and update count. All allocate 4,241,449 parameters, but not all parameters participate in every arm. Message passing avoids content QK computation and is cheaper; matching allocated parameters does not equate active capacity or FLOPs. Hard and message routing are identical on functional graphs. Their agreement is not independent replication.

Context, message passing, and hard attention reach 1.0 in every cell at the first checkpoint, after 10 updates. Soft attention retains 1.0 mean-head argmax full-pointer-path accuracy in every cell, despite .246 value accuracy at N32/D8. In that cell, clean-edge mass is .626 for soft attention, .982 for context, and 1.0 for hard attention and message passing. Correct argmax identity does not guarantee useful weighted payload propagation. This motivates investigating dilution without attributing the failure to identity binding.

The reverse ordered schedule, explicit node grounding, and unique outgoing typed edge are supplied. Hard routing, message passing, and context with a known-key prior receive powerful routing priors; the shared payload embedding, update, and classifier are learned. This experiment does not establish learned planning, surface understanding, or an advantage from latent grounding. With a correct single-successor graph, hard routing selects the neighbor exactly; that selection is programmed.

[Learning curves](a01-curves.svg) separate task value, mean-head argmax path, and edge mass. The pointer diagnostic is not causal proof of neural computation. Structural strength and normalization can alter weighted mixtures without changing their argmax.

A02 is registered before its outcomes: stronger initial λ, a public size correction, and score interventions on frozen weights. It will test whether finite structural mass, rather than the learned state update, explains this discrepancy. Confirmation selection has not occurred.

Process occupancy was 36.30 seconds, including five process initializations, training, evaluation, and export. Per-arm process times were: soft 7.07s, context 7.49s, message 5.29s, hard 7.20s, and no graph 9.21s. These whole-process times are not pure inference latency. The original 2.54s profile is charged separately. Eight CPU mechanical tests pass after A02 instrumentation; the primary experimental width remains 1024.

Naming: `soft4` and `soft8` denote initial trainable strengths, not permanently fixed learned parameters. The frozen-checkpoint `strength_override=8` intervention is exactly fixed at8. Zero-strength evaluation is meaningful only for soft attention; other arms ignore that flag, so their duplicate zero-strength cells are no-op consistency checks rather than graph-removal ablations.
