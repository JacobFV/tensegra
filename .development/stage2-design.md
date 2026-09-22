# Stage 2: how much learning does programmed topology save?

User-authorized implementation and empirical study, 2026-09-22. Extend the pilot without introducing language models or latent entity grounding. Preserve the pilot API and artifacts.

## Questions and design

1. Efficiency: nested training trajectory counts 8,16,32,64,128,256 with curves through 600 steps, evaluated at 0,25,50,100,200,300,600. Compare none, soft 1/4, hard, and matched-strength permuted controls. Common held-out trajectories and training normalization across the nested count sweep. Report normalized validation curve AUC, first observed threshold crossing (censor failures), and minimum tested trajectory count reaching predeclared oracle-gap thresholds. Threshold = oracle validation error + 25% or 50% of the zero-predictor excess; never select thresholds from test results.
2. Incomplete structure: drop 10/25/50% true nonself edges, add spurious edges at 10/25/50% of true edge count, mix drop/add, compare soft/hard on identical corruptions. Corrupt supplied graph only; generating process unchanged. Retain self edges and record realized precision/recall. Also evaluate clean-trained models on corrupted runtime graphs to separate training robustness from runtime robustness where supported.
3. Runtime generalization: one shared model per seed trained across 16 independent sparse graphs with 12 nodes, validate on four unseen graphs and test on eight unseen graphs at 12,32,64,128 nodes. Also train on mixed 9/12/16 sizes. Record graph seeds/hashes and ensure split disjointness. Compare none, true soft, hard, permuted, and graph-as-data. Fixed expected indegree for size transfer to avoid density confounding. Include robot morphology as synthetic proxy, not physical VLA.
4. Graph-as-data: permutation-equivariant neighbor-history aggregation concatenated/projected into node input while all attention remains unbiased. It receives the same full adjacency through input preprocessing; report this as an explicit graph-conditioned input/message-passing baseline, not a serialization-equivalent transformer. Include parameters and compute costs.
5. Heterogeneous mechanisms: signed nonuniform weights with absolute row sum <=0.8 (sufficient contraction bound). Compare adjacency and weighted/signed information; identify the extra edge-information advantage. Do not claim adjacency identifies hidden arbitrary coefficients. Relation-aware model supports positive/negative edge channels.
6. Learned per-layer/head strength: unconstrained alpha initialized exactly zero, record all head/layer values at checkpoints. No assumed positivity or depth trend. Include signed relation channels as a separate experiment.

## Implementation boundaries

Create `study_data.py` for process families, graph corruption, graph-indexed trajectory splits and deterministic rollout targets. Create `study_model.py` for input-graph and learned/relation-bias models built from the existing AttentionBlock; no change to scalar core attention needed (multiply per-head coefficients into bias). Create `study.py` for paired deterministic runner/CLI with bounded CPU resources, incremental JSONL, strict configuration, provenance, validation-only selection and test metrics. Create `study_analysis.py` for reports and figures from committed artifacts.

All raw metrics, curves, seeds, graph/schedule/state hashes, normalization, parameter count, optimizer steps, time and memory are retained. Use 3 seeds for initial complete study. Pretrained work remains conditional on the empirical milestone, not an automatic next job.

## Evaluation integrity

Noise-free recursive reference starts from the same observed test history and iterates the known noise-free transition. This isolates deterministic trajectory error; it is not the exact stochastic multistep conditional expectation. Report both stochastic and deterministic rollout, oracle and zero baselines. Exact graph identity is disjoint across train/validation/test; larger graph evaluation is chunked. Scientific claims are descriptive, paired across seeds, with population SD and individual effects; no causal exclusivity or statistical significance claimed from three seeds.

## Resources

Remote gb10-direct CPU venv, OMP/BLAS threads 2; no local training. Single study process, bounded per-suite deadline and output overwrite protection. Checkpoint curves provide optimizer budgets without redundant restarts. Old artifacts remain immutable. Commit and push code, then source-revision-bound results, then detailed reports.
