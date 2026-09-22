# Stage 4: stable binding substrate

The user-defined next milestone is stable binding, not an interpreter. Preserve Stage1–3 code and artifacts. Use the same shuffled runtime-entity task, fresh continuous identity keys and answer-only primary training at depths1–4. Distinguish initialized behavior from learned behavior using step-zero validation and held-out matrix evaluation.

## Interventions

1. Compare original raw-dot mixed state with cosine-normalized learned matching while retaining generic residual updates.
2. Protect identity coordinates from value transforms/content MLPs. Write them from attention-weighted immutable token keys; content updates remain neural. Compare full-state versus identity-only grounding inputs to separate write protection from read isolation.
3. Separately label an explicit pointer transition: p_next = p_query A_r (with edge-free null retained); write its expected immutable entity key into identity coordinates. This programs the graph transition and bypasses token-attention identity retrieval, so it is a stronger algorithmic prior, not evidence for superiority of logit bias. Content still retrieves payloads through differentiable attention.
4. Compare aligned and random projection initialization with null initialization controlled independently. Auxiliary ground/null/consistency losses are explicit supervision, never forward-pass inputs or teacher-forced state writes. Sweep beta=0,.01,.1,1 with fixed disclosed gamma/eta for supervised families; isolate null/cycle contributions where practical.
5. Only after a fixed-strength configuration clears the preregistered stability gate, evaluate entropy-controlled strength. The gate is >=.95 complete grounding trajectories and >=.95 task accuracy at D64/N128 in each of three seeds. Gate assessment is diagnostic, not a checkpoint or hyperparameter selector. No interpreter follows automatically.

## Diagnostics and protocol

Evaluate full N={16,32,64,128} x D={4,8,16,32,64} matrix, paired seeds0/1/2 and 128 examples per cell with small batches. Record task/complete-path accuracy separately; all pre/post query correctness per example; first-error survival, next-step error persistence/recovery, final destination correctness, path reconvergence, distinct nodes, entropy, top-two probability margin, query/identity norms, cosine and Euclidean drift, MLP identity interference and strength. Store joint per-example aggregates so correlations with failure are computable rather than correlating condition means. Compare empirical full paths with the product of per-step marginal accuracies as a descriptive reference; it is not an iid null test or proof of attractors. Revisited nodes, selection and persistent graph-specific difficulty may explain correlation.

Use fixed 400-step main budgets, three paired seeds, final checkpoint evaluation and step-zero/shallow curves. Include Stage3-matched baselines and strong keyed graph-input control. Any additional longer-budget or adaptive run is separately named and justified; never retune against matrix test accuracy. Preserve raw metrics, config/source hashes, standalone heatmaps and reviewable reports. Training only on gb10-direct, two CPU threads, evaluation batch <=16 and bounded memory; no local Torch install/training.

## Implementation ownership / sequence

Separate reusable binding model, losses/diagnostic functions, runner, and artifact-only analysis. First verify tensor contracts, isolation, gradients, null semantics and no gold leakage; run smoke; freeze source; execute and inspect initialization/training/matrix. Review the stability gate before any adaptive experiment. Commit/push notes, raw data and reports, then integrate to main after independent review.
