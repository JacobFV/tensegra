# Targeted primary-method notes

These references guide mechanism selection; they do not establish novelty of this campaign. No external implementation code has been copied.

- [Universal Transformers](https://arxiv.org/pdf/1807.03819), sections2–3: attention exchanges information between positions, followed by a transition shared across recurrent depth. Reusing a cell differs from adding uniquely parameterized layers. Adaptive computation is an optional additional mechanism; it is not assumed learned here.
- [Recurrent depth approach](https://arxiv.org/html/2502.05171v1), sections3.1–3.3: separates input processing, recurrent core and output processing; reinjects input features and samples recurrence counts during training. Its randomized-horizon recipe motivates an optional later return-training comparison, not a claim that it will repair the current interface. We have not adopted its truncated backpropagation or large-model training recipe.
- [Deep Biaffine Attention](https://arxiv.org/pdf/1611.01734), section3.1: scores ordered pairs using an interaction term plus affine terms, after role-specific projections. This is relevant to graph-output expressivity. Current semantic experiments first preserve the existing actor and edge-conditional objective; direct text access is a separately attributable possible change.

Workers record any additional graph/message-passing sources with their protocols. Literature is used to strengthen baselines, not to relabel existing mechanisms as novel.

- [Pointer Networks](https://arxiv.org/html/1506.03134v2), sections2.2–2.3: contrasts attention-weighted blending with a distribution over input positions whose selected input is copied. This is relevant prior art for interpreting A13's learned-score shared-address selection. A13 is a frozen multihead inference intervention, not an implementation of their recurrent pointer decoder or beam search, and establishes no novelty of neural addressing. The experiment separately measures soft blending, hard selection and privileged routing. No external code was copied.
