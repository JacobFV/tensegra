# Stage 4 controlled binding study

The new runner is isolated from Stage 1–3. Its public model-input allowlist reuses Stage 3 and excludes targets, token identities and gold paths. Auxiliary losses receive gold paths only after the forward pass, and variants explicitly label that supervision.

## Preregistered grid

Three paired seeds, 400 optimizer steps, 32 fresh training examples per step, depths cycling 1–4, N=16. AdamW 0.001, no weight decay, gradient norm clip 1. Checkpoints 0/25/50/100/200/400; no model selection on test conditions. The 23 variants are:

- Stage 3 dot/mixed/full controls: soft4, soft8, known cosine matcher (strength8), keyed graph-input (strength16, content identity bias8).
- Identity-initialized cosine mixed/full; cosine attention-write/full; cosine attention-write/identity-only; cosine pointer-write/identity-only; dot attention-write/identity-only. Structural strength starts4.
- Cold random cosine mixed/full, attention/identity-only, pointer/identity-only, all null-logit0 and answer-only.
- Cold random cosine attention/identity-only with null-logit0.65, separating the null prior from initialization.
- For each cold attention and pointer architecture: supervised weights (ground,null,cycle) = (.01,.1,.1), (.1,.1,.1), (1,.1,.1), (1,0,0), (0,.1,.1).

Thus 69 runs. All ordinary computation parameters consume the same seeded initialization. Exact baseline tests compare constructor-created Stage 3 and Stage 4 outputs without loading one model's weights into the other. Cold versus identity initialization changes grounding projections, not the ordinary transformer's initial weights.

Final evaluation is the full Cartesian N={16,32,64,128}, D={4,8,16,32,64} matrix, 128 examples per cell, batch16. Initial evaluation covers D4/N16 and D64/N128, exposing built-in routing. Validation uses a separate seed namespace. Evaluation and training hashes expose paired batches. Final source file hashes, Git revision, initial parameter hashes, schedule hashes and final state hashes are logged. Fixed final budget is evaluated; checkpoint metrics do not select the test checkpoint.

Raw diagnostics retain correctness bitstrings and per-example feature means, with those floating features rounded to six significant digits for artifact size. Aggregate statistics retain full precision. Model state checkpoints are written under the remote output directory for follow-up audits and are not intended for Git. Inference timing excludes diagnostics, losses and oracle computation. Peak RSS is process-wide high-water memory, not an isolated per-variant peak.

## Verification and resource pilot

Nine runner tests pass remotely, including four exact Stage 3 output regressions, full-matrix contract, input leakage boundary, registry supervision distinction, resource validation, end-to-end supervised smoke and resume checks. Six configured smoke runs (2 training steps, 2 evaluation examples) completed.

One unpinned implementation benchmark of cosine attention/identity-only at the full 400-step and 128-example matrix budget took 14.63 seconds (4.77 training seconds), peak process RSS 572.6 MiB, using two remote CPU threads. This benchmark is for resource planning only; it is not a reported confirmatory result. No training was run on the local machine.
