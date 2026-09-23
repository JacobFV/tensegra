# Stage 3: latent grounding into runtime graphs

This stage isolates differentiable grounding, not language interpretation. Existing stage 1/2 code and artifacts remain unchanged.

## Preregistered scope

Runtime entities have randomly sampled identity keys and independently sampled categorical payloads. Directed typed functional relations give unambiguous paths. Token order, graph identities, topology, payloads and distractors are randomized. Entity embeddings carry identity only. Learned forward passes receive immutable entity memory, the start key, relation instructions and runtime adjacency; intermediate nodes and answers are diagnostic/training targets, never routing inputs.

A shared small transformer block updates a query residual once per supplied relation instruction. Grounding is recomputed from that residual at every step. Thus depth extrapolation receives proportionally more computation and an externally supplied relation schedule; it does not demonstrate learned execution scheduling. Per-layer projection ablations use a cyclic period, avoiding untrained parameters at unseen depths. Query and key grounding are separate and include unbound capacity. Structural bias is Pq A Pk^T with typed strengths. Ordinary attention remains available.

Controls: deterministic exact routing oracle, known identity grounding, learned soft, frozen grounding, permuted grounding, no structure, graph-input message passing, hard routing. The oracle verifies routing algebra and is privileged, not a neural performance claim. Graph-input may be competitive or superior: the hypothesis is tested rather than assumed.

## Evaluation

Train on depths 1–4, evaluate independent graphs at depths 1,2,4,8,16,32 and node counts 16,32,64. Hold out the consecutive relation composition (0,1); evaluate it separately. New random identity keys on every batch prevent persistent ID memorization. Compare paired seeds and identical batch schedules. Report task accuracy first, then per-step grounding accuracy, entropy, next-node mass, relation-consistent attention, complete grounding trajectory accuracy, null/distractor behavior, structural strengths, elapsed time and RSS. Independently corrupt supplied edges without changing gold answers.

A finite experimental budget cannot establish a scaling law. Report depth/size curves and failures without extrapolating to interpreters, natural language or pretrained models. If learned grounding fails, separate oracle capability, binding failure, recursive drift, topology use and optimization. Any supervised grounding experiment is separately labeled auxiliary supervision, not the primary answer-only result.

## Execution plan

1. Implement and test runtime graph, grounding, randomized data and recurrent model independently.
2. Verify oracle traversal through depth 32 and zero-strength equivalence, gradients, null and directionality.
3. Run short optimization diagnostics before committing a reproducible controlled experiment configuration.
4. Run paired-seed controls and focused architectural ablations remotely with two CPU threads.
5. Commit raw metrics, source/config provenance, reports and diagnostic figures; independently review leakage and conclusions.
