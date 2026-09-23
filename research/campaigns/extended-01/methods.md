# Targeted primary-method notes

These references guide mechanism selection; they do not establish novelty of this campaign. No external implementation code has been copied.

- [Universal Transformers](https://arxiv.org/pdf/1807.03819), sections2–3: attention exchanges information between positions, followed by a transition shared across recurrent depth. Reusing a cell differs from adding uniquely parameterized layers. Adaptive computation is an optional additional mechanism; it is not assumed learned here.
- [Recurrent depth approach](https://arxiv.org/html/2502.05171v1), sections3.1–3.3: separates input processing, recurrent core and output processing; reinjects input features and samples recurrence counts during training. Its randomized-horizon recipe motivates an optional later return-training comparison, not a claim that it will repair the current interface. We have not adopted its truncated backpropagation or large-model training recipe.
- [Deep Biaffine Attention](https://arxiv.org/pdf/1611.01734), section3.1: scores ordered pairs using an interaction term plus affine terms, after role-specific projections. This is relevant to graph-output expressivity. Current semantic experiments first preserve the existing actor and edge-conditional objective; direct text access is a separately attributable possible change.

Workers record any additional graph/message-passing sources with their protocols. Literature is used to strengthen baselines, not to relabel existing mechanisms as novel.
