# Stage 4 artifact analysis

Task accuracy is primary; complete grounding trajectories are a separate mechanism diagnostic. Values are mean ± sample SD across seeds. This is not a confidence interval.

The stability gate requires task accuracy AND complete trajectory accuracy >=0.95 at D64/N128 in EACH declared seed. It is not a checkpoint selector and does not automatically launch adaptive-strength experiments.

| Variant | Stability gate |
|---|---|
| pointer_fixed4 | PASS |
| pointer_adaptive4 | PASS |

## Initialization versus final checkpoint

| Variant | N | D | Initial task | Final task | Initial complete path | Final complete path |
|---|---:|---:|---:|---:|---:|---:|
| pointer_adaptive4 | 16 | 4 | 0.0964 ± 0.0325 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| pointer_adaptive4 | 128 | 64 | 0.1276 ± 0.0393 | 0.9896 ± 0.0119 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| pointer_fixed4 | 16 | 4 | 0.0964 ± 0.0325 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| pointer_fixed4 | 128 | 64 | 0.1276 ± 0.0393 | 0.9896 ± 0.0119 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## Full task and trajectory matrix

| Variant | N | D | Task | Complete path | Seed task/path pairs |
|---|---:|---:|---:|---:|---|
| pointer_adaptive4 | 16 | 4 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 16 | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 16 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 16 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 16 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 32 | 4 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 32 | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 32 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 32 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 32 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 64 | 4 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 64 | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 64 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 64 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 64 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 128 | 4 | 0.9948 ± 0.0090 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 0.9844/1.0000 |
| pointer_adaptive4 | 128 | 8 | 0.9974 ± 0.0045 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 0.9922/1.0000 |
| pointer_adaptive4 | 128 | 16 | 0.9974 ± 0.0045 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 0.9922/1.0000 |
| pointer_adaptive4 | 128 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_adaptive4 | 128 | 64 | 0.9896 ± 0.0119 | 1.0000 ± 0.0000 | 0: 0.9922/1.0000; 1: 1.0000/1.0000; 2: 0.9766/1.0000 |
| pointer_fixed4 | 16 | 4 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 16 | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 16 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 16 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 16 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 32 | 4 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 32 | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 32 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 32 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 32 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 64 | 4 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 64 | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 64 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 64 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 64 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 128 | 4 | 0.9974 ± 0.0045 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 0.9922/1.0000 |
| pointer_fixed4 | 128 | 8 | 0.9974 ± 0.0045 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 0.9922/1.0000 |
| pointer_fixed4 | 128 | 16 | 0.9974 ± 0.0045 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 0.9922/1.0000 |
| pointer_fixed4 | 128 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0: 1.0000/1.0000; 1: 1.0000/1.0000; 2: 1.0000/1.0000 |
| pointer_fixed4 | 128 | 64 | 0.9896 ± 0.0119 | 1.0000 ± 0.0000 | 0: 0.9922/1.0000; 1: 1.0000/1.0000; 2: 0.9766/1.0000 |

## Paired intervention contrasts

Differences are treatment minus control on matched evaluation examples and training schedules. The summary records all differing initial parameter hashes; architecture/initialization differences are not silently called parameter matched. Pointer writes have a stronger programmed-transition prior than attention writes.

| Treatment | Control | N | D | Task difference |
|---|---|---:|---:|---:|

## Failure persistence and survival

The canonical sequence contains initial grounding once followed by post-hop grounding at every hop. Error persistence/recovery divide pooled transition counts by the number of transitions starting wrong. The marginal-product reference multiplies accuracy at these same nonduplicated states. It is descriptive: trajectory heterogeneity, revisits and reconvergence can explain divergence; it does not prove an attractor or test an iid error model.

| Variant | N | D | Complete | Marginal product | Error→error / eligible | Error→correct / eligible |
|---|---:|---:|---:|---:|---:|---:|
| pointer_adaptive4 | 16 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 16 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 16 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 16 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 16 | 64 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 32 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 32 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 32 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 32 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 32 | 64 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 64 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 64 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 64 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 64 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 64 | 64 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 128 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 128 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 128 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 128 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_adaptive4 | 128 | 64 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 16 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 16 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 16 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 16 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 16 | 64 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 32 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 32 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 32 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 32 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 32 | 64 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 64 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 64 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 64 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 64 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 64 | 64 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 128 | 4 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 128 | 8 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 128 | 16 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 128 | 32 | 1.0000 | 1 | 0/0 | 0/0 |
| pointer_fixed4 | 128 | 64 | 1.0000 | 1 | 0/0 | 0/0 |

## Within-cell diagnostic associations

Pearson correlations below use per-example feature means and complete-path failure, within each variant × depth × size cell (pooled paired seeds). Features include observations after errors, so associations are not causal predictors. Constant features or outcomes have undefined (N/A) correlations. summary.json includes complete/failed feature means and sample SD, survival curves and full diagnostics.

| Variant | N | D | Feature | Correlation with failure | Complete mean ± SD | Failed mean ± SD |
|---|---:|---:|---|---:|---:|---:|
| pointer_adaptive4 | 16 | 4 | distinct_gold_nodes | N/A | 4.2005 ± 0.9304 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_entropy | N/A | 0.0020 ± 0.0034 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_margin | N/A | 0.9995 ± 0.0013 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_post_entropy | N/A | 0.0021 ± 0.0035 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_post_identity_drift | N/A | 0.0003 ± 0.0009 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_post_identity_norm | N/A | 0.9998 ± 0.0007 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_post_margin | N/A | 0.9995 ± 0.0013 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_pre_identity_drift | N/A | 0.0002 ± 0.0008 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0006 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_projected_query_norm | N/A | 4.0007 ± 0.0090 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_strength | N/A | 3.9970 ± 0.0048 | N/A |
| pointer_adaptive4 | 16 | 4 | mean_write_override_norm | N/A | 0.5915 ± 0.0528 | N/A |
| pointer_adaptive4 | 16 | 8 | distinct_gold_nodes | N/A | 6.4870 ± 1.4506 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_entropy | N/A | 0.0021 ± 0.0030 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_margin | N/A | 0.9995 ± 0.0011 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_post_entropy | N/A | 0.0021 ± 0.0029 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_post_identity_drift | N/A | 0.0003 ± 0.0008 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_post_identity_norm | N/A | 0.9997 ± 0.0006 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_post_margin | N/A | 0.9995 ± 0.0010 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0007 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0005 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_projected_query_norm | N/A | 4.0007 ± 0.0075 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_strength | N/A | 3.9968 ± 0.0042 | N/A |
| pointer_adaptive4 | 16 | 8 | mean_write_override_norm | N/A | 0.5899 ± 0.0490 | N/A |
| pointer_adaptive4 | 16 | 16 | distinct_gold_nodes | N/A | 9.2240 ± 1.9083 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_entropy | N/A | 0.0021 ± 0.0031 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_margin | N/A | 0.9995 ± 0.0012 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_post_entropy | N/A | 0.0021 ± 0.0032 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_post_identity_drift | N/A | 0.0003 ± 0.0007 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_post_identity_norm | N/A | 0.9998 ± 0.0005 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_post_margin | N/A | 0.9995 ± 0.0012 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0006 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0005 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_projected_query_norm | N/A | 4.0004 ± 0.0067 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_strength | N/A | 3.9968 ± 0.0044 | N/A |
| pointer_adaptive4 | 16 | 16 | mean_write_override_norm | N/A | 0.5914 ± 0.0381 | N/A |
| pointer_adaptive4 | 16 | 32 | distinct_gold_nodes | N/A | 11.5859 ± 2.0497 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_entropy | N/A | 0.0021 ± 0.0024 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_margin | N/A | 0.9995 ± 0.0008 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_post_entropy | N/A | 0.0021 ± 0.0024 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_post_identity_drift | N/A | 0.0003 ± 0.0006 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_post_identity_norm | N/A | 0.9998 ± 0.0004 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_post_margin | N/A | 0.9995 ± 0.0008 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0005 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0004 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_projected_query_norm | N/A | 4.0002 ± 0.0060 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_strength | N/A | 3.9969 ± 0.0034 | N/A |
| pointer_adaptive4 | 16 | 32 | mean_write_override_norm | N/A | 0.5914 ± 0.0339 | N/A |
| pointer_adaptive4 | 16 | 64 | distinct_gold_nodes | N/A | 13.3568 ± 1.9337 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_entropy | N/A | 0.0021 ± 0.0026 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_margin | N/A | 0.9995 ± 0.0009 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_post_entropy | N/A | 0.0021 ± 0.0026 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_post_identity_drift | N/A | 0.0003 ± 0.0006 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_post_identity_norm | N/A | 0.9998 ± 0.0004 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_post_margin | N/A | 0.9995 ± 0.0009 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0005 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0004 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_projected_query_norm | N/A | 4.0002 ± 0.0059 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_strength | N/A | 3.9968 ± 0.0036 | N/A |
| pointer_adaptive4 | 16 | 64 | mean_write_override_norm | N/A | 0.5915 ± 0.0327 | N/A |
| pointer_adaptive4 | 32 | 4 | distinct_gold_nodes | N/A | 4.6667 ± 0.6843 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_entropy | N/A | 0.0035 ± 0.0050 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_margin | N/A | 0.9992 ± 0.0022 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_post_entropy | N/A | 0.0035 ± 0.0051 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_post_identity_drift | N/A | 0.0006 ± 0.0014 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_post_identity_norm | N/A | 0.9996 ± 0.0009 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_post_margin | N/A | 0.9991 ± 0.0023 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_pre_identity_drift | N/A | 0.0004 ± 0.0009 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_pre_identity_norm | N/A | 0.9997 ± 0.0007 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_projected_query_norm | N/A | 3.9998 ± 0.0079 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_strength | N/A | 3.9959 ± 0.0057 | N/A |
| pointer_adaptive4 | 32 | 4 | mean_write_override_norm | N/A | 0.6597 ± 0.0552 | N/A |
| pointer_adaptive4 | 32 | 8 | distinct_gold_nodes | N/A | 7.6276 ± 1.4305 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_entropy | N/A | 0.0035 ± 0.0033 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_margin | N/A | 0.9992 ± 0.0012 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_post_entropy | N/A | 0.0035 ± 0.0037 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_post_identity_drift | N/A | 0.0006 ± 0.0009 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_post_identity_norm | N/A | 0.9996 ± 0.0006 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_post_margin | N/A | 0.9992 ± 0.0017 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_pre_identity_drift | N/A | 0.0005 ± 0.0008 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_pre_identity_norm | N/A | 0.9996 ± 0.0006 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_projected_query_norm | N/A | 3.9987 ± 0.0067 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_strength | N/A | 3.9958 ± 0.0038 | N/A |
| pointer_adaptive4 | 32 | 8 | mean_write_override_norm | N/A | 0.6666 ± 0.0435 | N/A |
| pointer_adaptive4 | 32 | 16 | distinct_gold_nodes | N/A | 12.2760 ± 2.0431 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_entropy | N/A | 0.0037 ± 0.0034 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_margin | N/A | 0.9991 ± 0.0013 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_post_entropy | N/A | 0.0037 ± 0.0033 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_post_identity_drift | N/A | 0.0006 ± 0.0009 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_post_identity_norm | N/A | 0.9995 ± 0.0006 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_post_margin | N/A | 0.9991 ± 0.0013 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_pre_identity_drift | N/A | 0.0006 ± 0.0008 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_pre_identity_norm | N/A | 0.9996 ± 0.0006 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_projected_query_norm | N/A | 3.9993 ± 0.0055 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_strength | N/A | 3.9955 ± 0.0038 | N/A |
| pointer_adaptive4 | 32 | 16 | mean_write_override_norm | N/A | 0.6670 ± 0.0330 | N/A |
| pointer_adaptive4 | 32 | 32 | distinct_gold_nodes | N/A | 17.6901 ± 2.6334 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_entropy | N/A | 0.0037 ± 0.0032 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_margin | N/A | 0.9991 ± 0.0013 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_post_entropy | N/A | 0.0036 ± 0.0032 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_post_identity_drift | N/A | 0.0006 ± 0.0009 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_post_identity_norm | N/A | 0.9995 ± 0.0007 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_post_margin | N/A | 0.9991 ± 0.0014 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_pre_identity_drift | N/A | 0.0006 ± 0.0009 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_pre_identity_norm | N/A | 0.9995 ± 0.0006 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_projected_query_norm | N/A | 3.9992 ± 0.0053 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_strength | N/A | 3.9956 ± 0.0036 | N/A |
| pointer_adaptive4 | 32 | 32 | mean_write_override_norm | N/A | 0.6671 ± 0.0275 | N/A |
| pointer_adaptive4 | 32 | 64 | distinct_gold_nodes | N/A | 22.7422 ± 2.7524 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_entropy | N/A | 0.0036 ± 0.0027 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_margin | N/A | 0.9991 ± 0.0010 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_post_entropy | N/A | 0.0036 ± 0.0026 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_post_identity_drift | N/A | 0.0006 ± 0.0007 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_post_identity_norm | N/A | 0.9995 ± 0.0005 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_post_margin | N/A | 0.9991 ± 0.0010 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_pre_identity_drift | N/A | 0.0006 ± 0.0007 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_pre_identity_norm | N/A | 0.9996 ± 0.0005 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_projected_query_norm | N/A | 3.9992 ± 0.0046 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_strength | N/A | 3.9957 ± 0.0030 | N/A |
| pointer_adaptive4 | 32 | 64 | mean_write_override_norm | N/A | 0.6668 ± 0.0252 | N/A |
| pointer_adaptive4 | 64 | 4 | distinct_gold_nodes | N/A | 4.7917 ± 0.5440 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_entropy | N/A | 0.0071 ± 0.0066 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_margin | N/A | 0.9983 ± 0.0024 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_post_entropy | N/A | 0.0069 ± 0.0062 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_post_identity_drift | N/A | 0.0012 ± 0.0016 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_post_identity_norm | N/A | 0.9991 ± 0.0012 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_post_margin | N/A | 0.9984 ± 0.0021 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_pre_identity_drift | N/A | 0.0009 ± 0.0014 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_pre_identity_norm | N/A | 0.9993 ± 0.0010 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_projected_query_norm | N/A | 3.9986 ± 0.0098 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_strength | N/A | 3.9930 ± 0.0064 | N/A |
| pointer_adaptive4 | 64 | 4 | mean_write_override_norm | N/A | 0.7529 ± 0.0534 | N/A |
| pointer_adaptive4 | 64 | 8 | distinct_gold_nodes | N/A | 8.2135 ± 1.0797 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_entropy | N/A | 0.0068 ± 0.0055 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_margin | N/A | 0.9984 ± 0.0020 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_post_entropy | N/A | 0.0068 ± 0.0054 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_post_identity_drift | N/A | 0.0011 ± 0.0014 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_post_identity_norm | N/A | 0.9991 ± 0.0010 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_post_margin | N/A | 0.9984 ± 0.0020 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_pre_identity_drift | N/A | 0.0010 ± 0.0012 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_pre_identity_norm | N/A | 0.9992 ± 0.0009 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_projected_query_norm | N/A | 3.9981 ± 0.0073 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_strength | N/A | 3.9933 ± 0.0053 | N/A |
| pointer_adaptive4 | 64 | 8 | mean_write_override_norm | N/A | 0.7555 ± 0.0421 | N/A |
| pointer_adaptive4 | 64 | 16 | distinct_gold_nodes | N/A | 14.2214 ± 1.8706 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_entropy | N/A | 0.0067 ± 0.0037 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_margin | N/A | 0.9984 ± 0.0014 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_post_entropy | N/A | 0.0066 ± 0.0036 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_post_identity_drift | N/A | 0.0011 ± 0.0010 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_post_identity_norm | N/A | 0.9991 ± 0.0007 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_post_margin | N/A | 0.9984 ± 0.0014 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_pre_identity_drift | N/A | 0.0011 ± 0.0010 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_pre_identity_norm | N/A | 0.9992 ± 0.0007 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_projected_query_norm | N/A | 3.9975 ± 0.0053 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_strength | N/A | 3.9934 ± 0.0035 | N/A |
| pointer_adaptive4 | 64 | 16 | mean_write_override_norm | N/A | 0.7539 ± 0.0315 | N/A |
| pointer_adaptive4 | 64 | 32 | distinct_gold_nodes | N/A | 23.6094 ± 3.0877 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_entropy | N/A | 0.0068 ± 0.0031 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_margin | N/A | 0.9984 ± 0.0013 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_post_entropy | N/A | 0.0068 ± 0.0031 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_post_identity_drift | N/A | 0.0012 ± 0.0009 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_post_identity_norm | N/A | 0.9991 ± 0.0006 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_post_margin | N/A | 0.9984 ± 0.0013 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_pre_identity_drift | N/A | 0.0011 ± 0.0009 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_pre_identity_norm | N/A | 0.9991 ± 0.0006 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_projected_query_norm | N/A | 3.9974 ± 0.0049 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_strength | N/A | 3.9933 ± 0.0030 | N/A |
| pointer_adaptive4 | 64 | 32 | mean_write_override_norm | N/A | 0.7572 ± 0.0244 | N/A |
| pointer_adaptive4 | 64 | 64 | distinct_gold_nodes | N/A | 35.0495 ± 3.7486 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_entropy | N/A | 0.0067 ± 0.0027 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_margin | N/A | 0.9984 ± 0.0011 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_post_entropy | N/A | 0.0067 ± 0.0027 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_post_identity_drift | N/A | 0.0011 ± 0.0007 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_post_identity_norm | N/A | 0.9991 ± 0.0005 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_post_margin | N/A | 0.9984 ± 0.0011 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_pre_identity_drift | N/A | 0.0011 ± 0.0007 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_pre_identity_norm | N/A | 0.9992 ± 0.0005 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_projected_query_norm | N/A | 3.9974 ± 0.0042 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_strength | N/A | 3.9934 ± 0.0026 | N/A |
| pointer_adaptive4 | 64 | 64 | mean_write_override_norm | N/A | 0.7562 ± 0.0217 | N/A |
| pointer_adaptive4 | 128 | 4 | distinct_gold_nodes | N/A | 4.9010 ± 0.3966 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_entropy | N/A | 0.0131 ± 0.0096 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_margin | N/A | 0.9970 ± 0.0035 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_post_entropy | N/A | 0.0131 ± 0.0099 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_post_identity_drift | N/A | 0.0021 ± 0.0023 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_post_identity_norm | N/A | 0.9983 ± 0.0016 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_post_margin | N/A | 0.9969 ± 0.0037 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_pre_identity_drift | N/A | 0.0016 ± 0.0019 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_pre_identity_norm | N/A | 0.9988 ± 0.0013 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_projected_query_norm | N/A | 3.9959 ± 0.0092 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_strength | N/A | 3.9890 ± 0.0079 | N/A |
| pointer_adaptive4 | 128 | 4 | mean_write_override_norm | N/A | 0.8423 ± 0.0454 | N/A |
| pointer_adaptive4 | 128 | 8 | distinct_gold_nodes | N/A | 8.5599 ± 0.8499 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_entropy | N/A | 0.0137 ± 0.0087 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_margin | N/A | 0.9966 ± 0.0044 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_post_entropy | N/A | 0.0138 ± 0.0097 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0002 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_post_identity_drift | N/A | 0.0024 ± 0.0030 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_post_identity_norm | N/A | 0.9981 ± 0.0020 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_post_margin | N/A | 0.9965 ± 0.0049 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0002 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_pre_identity_drift | N/A | 0.0021 ± 0.0028 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_pre_identity_norm | N/A | 0.9983 ± 0.0019 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_projected_query_norm | N/A | 3.9942 ± 0.0095 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_strength | N/A | 3.9885 ± 0.0071 | N/A |
| pointer_adaptive4 | 128 | 8 | mean_write_override_norm | N/A | 0.8413 ± 0.0336 | N/A |
| pointer_adaptive4 | 128 | 16 | distinct_gold_nodes | N/A | 15.3021 ± 1.7425 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_entropy | N/A | 0.0132 ± 0.0061 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_margin | N/A | 0.9969 ± 0.0026 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_post_entropy | N/A | 0.0133 ± 0.0064 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_post_identity_drift | N/A | 0.0022 ± 0.0018 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_post_identity_norm | N/A | 0.9982 ± 0.0013 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_post_margin | N/A | 0.9968 ± 0.0029 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_pre_identity_drift | N/A | 0.0021 ± 0.0017 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_pre_identity_norm | N/A | 0.9984 ± 0.0012 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_projected_query_norm | N/A | 3.9941 ± 0.0065 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_strength | N/A | 3.9889 ± 0.0050 | N/A |
| pointer_adaptive4 | 128 | 16 | mean_write_override_norm | N/A | 0.8436 ± 0.0264 | N/A |
| pointer_adaptive4 | 128 | 32 | distinct_gold_nodes | N/A | 27.3281 ± 2.8241 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_entropy | N/A | 0.0130 ± 0.0044 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_margin | N/A | 0.9969 ± 0.0018 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_post_entropy | N/A | 0.0130 ± 0.0044 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_post_identity_drift | N/A | 0.0022 ± 0.0012 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_post_identity_norm | N/A | 0.9983 ± 0.0009 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_post_margin | N/A | 0.9969 ± 0.0018 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_pre_identity_drift | N/A | 0.0021 ± 0.0012 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_pre_identity_norm | N/A | 0.9983 ± 0.0008 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_projected_query_norm | N/A | 3.9941 ± 0.0048 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_strength | N/A | 3.9891 ± 0.0036 | N/A |
| pointer_adaptive4 | 128 | 32 | mean_write_override_norm | N/A | 0.8422 ± 0.0215 | N/A |
| pointer_adaptive4 | 128 | 64 | distinct_gold_nodes | N/A | 45.5443 ± 4.3392 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_entropy | N/A | 0.0131 ± 0.0041 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_margin | N/A | 0.9969 ± 0.0016 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_post_entropy | N/A | 0.0131 ± 0.0041 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_post_identity_drift | N/A | 0.0022 ± 0.0012 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_post_identity_norm | N/A | 0.9982 ± 0.0008 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_post_margin | N/A | 0.9969 ± 0.0016 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_pre_identity_drift | N/A | 0.0022 ± 0.0011 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_pre_identity_norm | N/A | 0.9983 ± 0.0008 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_projected_query_norm | N/A | 3.9938 ± 0.0046 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_strength | N/A | 3.9890 ± 0.0034 | N/A |
| pointer_adaptive4 | 128 | 64 | mean_write_override_norm | N/A | 0.8429 ± 0.0180 | N/A |
| pointer_fixed4 | 16 | 4 | distinct_gold_nodes | N/A | 4.2005 ± 0.9304 | N/A |
| pointer_fixed4 | 16 | 4 | mean_entropy | N/A | 0.0021 ± 0.0033 | N/A |
| pointer_fixed4 | 16 | 4 | mean_margin | N/A | 0.9995 ± 0.0012 | N/A |
| pointer_fixed4 | 16 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 4 | mean_post_entropy | N/A | 0.0022 ± 0.0034 | N/A |
| pointer_fixed4 | 16 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 4 | mean_post_identity_drift | N/A | 0.0003 ± 0.0009 | N/A |
| pointer_fixed4 | 16 | 4 | mean_post_identity_norm | N/A | 0.9998 ± 0.0007 | N/A |
| pointer_fixed4 | 16 | 4 | mean_post_margin | N/A | 0.9995 ± 0.0013 | N/A |
| pointer_fixed4 | 16 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 4 | mean_pre_identity_drift | N/A | 0.0002 ± 0.0008 | N/A |
| pointer_fixed4 | 16 | 4 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0006 | N/A |
| pointer_fixed4 | 16 | 4 | mean_projected_query_norm | N/A | 4.0009 ± 0.0084 | N/A |
| pointer_fixed4 | 16 | 4 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 4 | mean_write_override_norm | N/A | 0.5915 ± 0.0529 | N/A |
| pointer_fixed4 | 16 | 8 | distinct_gold_nodes | N/A | 6.4870 ± 1.4506 | N/A |
| pointer_fixed4 | 16 | 8 | mean_entropy | N/A | 0.0023 ± 0.0030 | N/A |
| pointer_fixed4 | 16 | 8 | mean_margin | N/A | 0.9995 ± 0.0011 | N/A |
| pointer_fixed4 | 16 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 8 | mean_post_entropy | N/A | 0.0023 ± 0.0029 | N/A |
| pointer_fixed4 | 16 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 8 | mean_post_identity_drift | N/A | 0.0004 ± 0.0008 | N/A |
| pointer_fixed4 | 16 | 8 | mean_post_identity_norm | N/A | 0.9997 ± 0.0006 | N/A |
| pointer_fixed4 | 16 | 8 | mean_post_margin | N/A | 0.9995 ± 0.0010 | N/A |
| pointer_fixed4 | 16 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 8 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0007 | N/A |
| pointer_fixed4 | 16 | 8 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0005 | N/A |
| pointer_fixed4 | 16 | 8 | mean_projected_query_norm | N/A | 4.0007 ± 0.0071 | N/A |
| pointer_fixed4 | 16 | 8 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 8 | mean_write_override_norm | N/A | 0.5899 ± 0.0490 | N/A |
| pointer_fixed4 | 16 | 16 | distinct_gold_nodes | N/A | 9.2240 ± 1.9083 | N/A |
| pointer_fixed4 | 16 | 16 | mean_entropy | N/A | 0.0023 ± 0.0032 | N/A |
| pointer_fixed4 | 16 | 16 | mean_margin | N/A | 0.9995 ± 0.0012 | N/A |
| pointer_fixed4 | 16 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 16 | mean_post_entropy | N/A | 0.0023 ± 0.0032 | N/A |
| pointer_fixed4 | 16 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 16 | mean_post_identity_drift | N/A | 0.0003 ± 0.0007 | N/A |
| pointer_fixed4 | 16 | 16 | mean_post_identity_norm | N/A | 0.9997 ± 0.0005 | N/A |
| pointer_fixed4 | 16 | 16 | mean_post_margin | N/A | 0.9995 ± 0.0012 | N/A |
| pointer_fixed4 | 16 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 16 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0006 | N/A |
| pointer_fixed4 | 16 | 16 | mean_pre_identity_norm | N/A | 0.9997 ± 0.0005 | N/A |
| pointer_fixed4 | 16 | 16 | mean_projected_query_norm | N/A | 4.0003 ± 0.0061 | N/A |
| pointer_fixed4 | 16 | 16 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 16 | mean_write_override_norm | N/A | 0.5913 ± 0.0382 | N/A |
| pointer_fixed4 | 16 | 32 | distinct_gold_nodes | N/A | 11.5859 ± 2.0497 | N/A |
| pointer_fixed4 | 16 | 32 | mean_entropy | N/A | 0.0022 ± 0.0024 | N/A |
| pointer_fixed4 | 16 | 32 | mean_margin | N/A | 0.9995 ± 0.0008 | N/A |
| pointer_fixed4 | 16 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 32 | mean_post_entropy | N/A | 0.0022 ± 0.0024 | N/A |
| pointer_fixed4 | 16 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 32 | mean_post_identity_drift | N/A | 0.0003 ± 0.0006 | N/A |
| pointer_fixed4 | 16 | 32 | mean_post_identity_norm | N/A | 0.9997 ± 0.0004 | N/A |
| pointer_fixed4 | 16 | 32 | mean_post_margin | N/A | 0.9995 ± 0.0008 | N/A |
| pointer_fixed4 | 16 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 32 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0005 | N/A |
| pointer_fixed4 | 16 | 32 | mean_pre_identity_norm | N/A | 0.9998 ± 0.0004 | N/A |
| pointer_fixed4 | 16 | 32 | mean_projected_query_norm | N/A | 4.0003 ± 0.0055 | N/A |
| pointer_fixed4 | 16 | 32 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 32 | mean_write_override_norm | N/A | 0.5914 ± 0.0340 | N/A |
| pointer_fixed4 | 16 | 64 | distinct_gold_nodes | N/A | 13.3568 ± 1.9337 | N/A |
| pointer_fixed4 | 16 | 64 | mean_entropy | N/A | 0.0023 ± 0.0026 | N/A |
| pointer_fixed4 | 16 | 64 | mean_margin | N/A | 0.9995 ± 0.0009 | N/A |
| pointer_fixed4 | 16 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 64 | mean_post_entropy | N/A | 0.0023 ± 0.0026 | N/A |
| pointer_fixed4 | 16 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 64 | mean_post_identity_drift | N/A | 0.0003 ± 0.0006 | N/A |
| pointer_fixed4 | 16 | 64 | mean_post_identity_norm | N/A | 0.9997 ± 0.0004 | N/A |
| pointer_fixed4 | 16 | 64 | mean_post_margin | N/A | 0.9995 ± 0.0009 | N/A |
| pointer_fixed4 | 16 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 64 | mean_pre_identity_drift | N/A | 0.0003 ± 0.0005 | N/A |
| pointer_fixed4 | 16 | 64 | mean_pre_identity_norm | N/A | 0.9997 ± 0.0004 | N/A |
| pointer_fixed4 | 16 | 64 | mean_projected_query_norm | N/A | 4.0002 ± 0.0053 | N/A |
| pointer_fixed4 | 16 | 64 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 16 | 64 | mean_write_override_norm | N/A | 0.5915 ± 0.0328 | N/A |
| pointer_fixed4 | 32 | 4 | distinct_gold_nodes | N/A | 4.6667 ± 0.6843 | N/A |
| pointer_fixed4 | 32 | 4 | mean_entropy | N/A | 0.0036 ± 0.0049 | N/A |
| pointer_fixed4 | 32 | 4 | mean_margin | N/A | 0.9992 ± 0.0022 | N/A |
| pointer_fixed4 | 32 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 4 | mean_post_entropy | N/A | 0.0036 ± 0.0051 | N/A |
| pointer_fixed4 | 32 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 4 | mean_post_identity_drift | N/A | 0.0006 ± 0.0014 | N/A |
| pointer_fixed4 | 32 | 4 | mean_post_identity_norm | N/A | 0.9996 ± 0.0009 | N/A |
| pointer_fixed4 | 32 | 4 | mean_post_margin | N/A | 0.9991 ± 0.0023 | N/A |
| pointer_fixed4 | 32 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 4 | mean_pre_identity_drift | N/A | 0.0004 ± 0.0009 | N/A |
| pointer_fixed4 | 32 | 4 | mean_pre_identity_norm | N/A | 0.9997 ± 0.0006 | N/A |
| pointer_fixed4 | 32 | 4 | mean_projected_query_norm | N/A | 3.9998 ± 0.0073 | N/A |
| pointer_fixed4 | 32 | 4 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 4 | mean_write_override_norm | N/A | 0.6594 ± 0.0553 | N/A |
| pointer_fixed4 | 32 | 8 | distinct_gold_nodes | N/A | 7.6276 ± 1.4305 | N/A |
| pointer_fixed4 | 32 | 8 | mean_entropy | N/A | 0.0036 ± 0.0033 | N/A |
| pointer_fixed4 | 32 | 8 | mean_margin | N/A | 0.9992 ± 0.0012 | N/A |
| pointer_fixed4 | 32 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 8 | mean_post_entropy | N/A | 0.0036 ± 0.0037 | N/A |
| pointer_fixed4 | 32 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 8 | mean_post_identity_drift | N/A | 0.0006 ± 0.0008 | N/A |
| pointer_fixed4 | 32 | 8 | mean_post_identity_norm | N/A | 0.9995 ± 0.0006 | N/A |
| pointer_fixed4 | 32 | 8 | mean_post_margin | N/A | 0.9991 ± 0.0017 | N/A |
| pointer_fixed4 | 32 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 8 | mean_pre_identity_drift | N/A | 0.0005 ± 0.0008 | N/A |
| pointer_fixed4 | 32 | 8 | mean_pre_identity_norm | N/A | 0.9996 ± 0.0006 | N/A |
| pointer_fixed4 | 32 | 8 | mean_projected_query_norm | N/A | 3.9990 ± 0.0063 | N/A |
| pointer_fixed4 | 32 | 8 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 8 | mean_write_override_norm | N/A | 0.6663 ± 0.0435 | N/A |
| pointer_fixed4 | 32 | 16 | distinct_gold_nodes | N/A | 12.2760 ± 2.0431 | N/A |
| pointer_fixed4 | 32 | 16 | mean_entropy | N/A | 0.0039 ± 0.0033 | N/A |
| pointer_fixed4 | 32 | 16 | mean_margin | N/A | 0.9991 ± 0.0013 | N/A |
| pointer_fixed4 | 32 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 16 | mean_post_entropy | N/A | 0.0038 ± 0.0033 | N/A |
| pointer_fixed4 | 32 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 16 | mean_post_identity_drift | N/A | 0.0006 ± 0.0009 | N/A |
| pointer_fixed4 | 32 | 16 | mean_post_identity_norm | N/A | 0.9995 ± 0.0006 | N/A |
| pointer_fixed4 | 32 | 16 | mean_post_margin | N/A | 0.9991 ± 0.0013 | N/A |
| pointer_fixed4 | 32 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 16 | mean_pre_identity_drift | N/A | 0.0006 ± 0.0008 | N/A |
| pointer_fixed4 | 32 | 16 | mean_pre_identity_norm | N/A | 0.9995 ± 0.0006 | N/A |
| pointer_fixed4 | 32 | 16 | mean_projected_query_norm | N/A | 3.9995 ± 0.0053 | N/A |
| pointer_fixed4 | 32 | 16 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 16 | mean_write_override_norm | N/A | 0.6667 ± 0.0331 | N/A |
| pointer_fixed4 | 32 | 32 | distinct_gold_nodes | N/A | 17.6901 ± 2.6334 | N/A |
| pointer_fixed4 | 32 | 32 | mean_entropy | N/A | 0.0038 ± 0.0032 | N/A |
| pointer_fixed4 | 32 | 32 | mean_margin | N/A | 0.9991 ± 0.0014 | N/A |
| pointer_fixed4 | 32 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 32 | mean_post_entropy | N/A | 0.0038 ± 0.0033 | N/A |
| pointer_fixed4 | 32 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 32 | mean_post_identity_drift | N/A | 0.0006 ± 0.0010 | N/A |
| pointer_fixed4 | 32 | 32 | mean_post_identity_norm | N/A | 0.9995 ± 0.0007 | N/A |
| pointer_fixed4 | 32 | 32 | mean_post_margin | N/A | 0.9991 ± 0.0014 | N/A |
| pointer_fixed4 | 32 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 32 | mean_pre_identity_drift | N/A | 0.0006 ± 0.0009 | N/A |
| pointer_fixed4 | 32 | 32 | mean_pre_identity_norm | N/A | 0.9995 ± 0.0006 | N/A |
| pointer_fixed4 | 32 | 32 | mean_projected_query_norm | N/A | 3.9993 ± 0.0050 | N/A |
| pointer_fixed4 | 32 | 32 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 32 | mean_write_override_norm | N/A | 0.6669 ± 0.0276 | N/A |
| pointer_fixed4 | 32 | 64 | distinct_gold_nodes | N/A | 22.7422 ± 2.7524 | N/A |
| pointer_fixed4 | 32 | 64 | mean_entropy | N/A | 0.0037 ± 0.0027 | N/A |
| pointer_fixed4 | 32 | 64 | mean_margin | N/A | 0.9991 ± 0.0010 | N/A |
| pointer_fixed4 | 32 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 64 | mean_post_entropy | N/A | 0.0037 ± 0.0027 | N/A |
| pointer_fixed4 | 32 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 64 | mean_post_identity_drift | N/A | 0.0006 ± 0.0007 | N/A |
| pointer_fixed4 | 32 | 64 | mean_post_identity_norm | N/A | 0.9995 ± 0.0005 | N/A |
| pointer_fixed4 | 32 | 64 | mean_post_margin | N/A | 0.9991 ± 0.0010 | N/A |
| pointer_fixed4 | 32 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 64 | mean_pre_identity_drift | N/A | 0.0006 ± 0.0007 | N/A |
| pointer_fixed4 | 32 | 64 | mean_pre_identity_norm | N/A | 0.9995 ± 0.0005 | N/A |
| pointer_fixed4 | 32 | 64 | mean_projected_query_norm | N/A | 3.9994 ± 0.0042 | N/A |
| pointer_fixed4 | 32 | 64 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 32 | 64 | mean_write_override_norm | N/A | 0.6666 ± 0.0252 | N/A |
| pointer_fixed4 | 64 | 4 | distinct_gold_nodes | N/A | 4.7917 ± 0.5440 | N/A |
| pointer_fixed4 | 64 | 4 | mean_entropy | N/A | 0.0072 ± 0.0066 | N/A |
| pointer_fixed4 | 64 | 4 | mean_margin | N/A | 0.9983 ± 0.0023 | N/A |
| pointer_fixed4 | 64 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 4 | mean_post_entropy | N/A | 0.0070 ± 0.0061 | N/A |
| pointer_fixed4 | 64 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 4 | mean_post_identity_drift | N/A | 0.0012 ± 0.0016 | N/A |
| pointer_fixed4 | 64 | 4 | mean_post_identity_norm | N/A | 0.9991 ± 0.0011 | N/A |
| pointer_fixed4 | 64 | 4 | mean_post_margin | N/A | 0.9984 ± 0.0021 | N/A |
| pointer_fixed4 | 64 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 4 | mean_pre_identity_drift | N/A | 0.0009 ± 0.0013 | N/A |
| pointer_fixed4 | 64 | 4 | mean_pre_identity_norm | N/A | 0.9993 ± 0.0010 | N/A |
| pointer_fixed4 | 64 | 4 | mean_projected_query_norm | N/A | 3.9987 ± 0.0092 | N/A |
| pointer_fixed4 | 64 | 4 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 4 | mean_write_override_norm | N/A | 0.7522 ± 0.0534 | N/A |
| pointer_fixed4 | 64 | 8 | distinct_gold_nodes | N/A | 8.2135 ± 1.0797 | N/A |
| pointer_fixed4 | 64 | 8 | mean_entropy | N/A | 0.0069 ± 0.0054 | N/A |
| pointer_fixed4 | 64 | 8 | mean_margin | N/A | 0.9984 ± 0.0019 | N/A |
| pointer_fixed4 | 64 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 8 | mean_post_entropy | N/A | 0.0069 ± 0.0053 | N/A |
| pointer_fixed4 | 64 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 8 | mean_post_identity_drift | N/A | 0.0011 ± 0.0013 | N/A |
| pointer_fixed4 | 64 | 8 | mean_post_identity_norm | N/A | 0.9991 ± 0.0010 | N/A |
| pointer_fixed4 | 64 | 8 | mean_post_margin | N/A | 0.9984 ± 0.0019 | N/A |
| pointer_fixed4 | 64 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 8 | mean_pre_identity_drift | N/A | 0.0010 ± 0.0012 | N/A |
| pointer_fixed4 | 64 | 8 | mean_pre_identity_norm | N/A | 0.9992 ± 0.0009 | N/A |
| pointer_fixed4 | 64 | 8 | mean_projected_query_norm | N/A | 3.9982 ± 0.0068 | N/A |
| pointer_fixed4 | 64 | 8 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 8 | mean_write_override_norm | N/A | 0.7549 ± 0.0422 | N/A |
| pointer_fixed4 | 64 | 16 | distinct_gold_nodes | N/A | 14.2214 ± 1.8706 | N/A |
| pointer_fixed4 | 64 | 16 | mean_entropy | N/A | 0.0068 ± 0.0036 | N/A |
| pointer_fixed4 | 64 | 16 | mean_margin | N/A | 0.9984 ± 0.0014 | N/A |
| pointer_fixed4 | 64 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 16 | mean_post_entropy | N/A | 0.0067 ± 0.0036 | N/A |
| pointer_fixed4 | 64 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 16 | mean_post_identity_drift | N/A | 0.0011 ± 0.0010 | N/A |
| pointer_fixed4 | 64 | 16 | mean_post_identity_norm | N/A | 0.9991 ± 0.0007 | N/A |
| pointer_fixed4 | 64 | 16 | mean_post_margin | N/A | 0.9984 ± 0.0014 | N/A |
| pointer_fixed4 | 64 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 16 | mean_pre_identity_drift | N/A | 0.0011 ± 0.0010 | N/A |
| pointer_fixed4 | 64 | 16 | mean_pre_identity_norm | N/A | 0.9992 ± 0.0007 | N/A |
| pointer_fixed4 | 64 | 16 | mean_projected_query_norm | N/A | 3.9976 ± 0.0051 | N/A |
| pointer_fixed4 | 64 | 16 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 16 | mean_write_override_norm | N/A | 0.7533 ± 0.0316 | N/A |
| pointer_fixed4 | 64 | 32 | distinct_gold_nodes | N/A | 23.6094 ± 3.0877 | N/A |
| pointer_fixed4 | 64 | 32 | mean_entropy | N/A | 0.0069 ± 0.0031 | N/A |
| pointer_fixed4 | 64 | 32 | mean_margin | N/A | 0.9984 ± 0.0013 | N/A |
| pointer_fixed4 | 64 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 32 | mean_post_entropy | N/A | 0.0069 ± 0.0031 | N/A |
| pointer_fixed4 | 64 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 32 | mean_post_identity_drift | N/A | 0.0012 ± 0.0009 | N/A |
| pointer_fixed4 | 64 | 32 | mean_post_identity_norm | N/A | 0.9991 ± 0.0006 | N/A |
| pointer_fixed4 | 64 | 32 | mean_post_margin | N/A | 0.9984 ± 0.0013 | N/A |
| pointer_fixed4 | 64 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 32 | mean_pre_identity_drift | N/A | 0.0011 ± 0.0009 | N/A |
| pointer_fixed4 | 64 | 32 | mean_pre_identity_norm | N/A | 0.9991 ± 0.0006 | N/A |
| pointer_fixed4 | 64 | 32 | mean_projected_query_norm | N/A | 3.9976 ± 0.0045 | N/A |
| pointer_fixed4 | 64 | 32 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 32 | mean_write_override_norm | N/A | 0.7566 ± 0.0244 | N/A |
| pointer_fixed4 | 64 | 64 | distinct_gold_nodes | N/A | 35.0495 ± 3.7486 | N/A |
| pointer_fixed4 | 64 | 64 | mean_entropy | N/A | 0.0068 ± 0.0028 | N/A |
| pointer_fixed4 | 64 | 64 | mean_margin | N/A | 0.9984 ± 0.0011 | N/A |
| pointer_fixed4 | 64 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 64 | mean_post_entropy | N/A | 0.0068 ± 0.0028 | N/A |
| pointer_fixed4 | 64 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 64 | mean_post_identity_drift | N/A | 0.0011 ± 0.0008 | N/A |
| pointer_fixed4 | 64 | 64 | mean_post_identity_norm | N/A | 0.9991 ± 0.0005 | N/A |
| pointer_fixed4 | 64 | 64 | mean_post_margin | N/A | 0.9984 ± 0.0011 | N/A |
| pointer_fixed4 | 64 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 64 | mean_pre_identity_drift | N/A | 0.0011 ± 0.0008 | N/A |
| pointer_fixed4 | 64 | 64 | mean_pre_identity_norm | N/A | 0.9991 ± 0.0005 | N/A |
| pointer_fixed4 | 64 | 64 | mean_projected_query_norm | N/A | 3.9976 ± 0.0040 | N/A |
| pointer_fixed4 | 64 | 64 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 64 | 64 | mean_write_override_norm | N/A | 0.7556 ± 0.0217 | N/A |
| pointer_fixed4 | 128 | 4 | distinct_gold_nodes | N/A | 4.9010 ± 0.3966 | N/A |
| pointer_fixed4 | 128 | 4 | mean_entropy | N/A | 0.0132 ± 0.0097 | N/A |
| pointer_fixed4 | 128 | 4 | mean_margin | N/A | 0.9970 ± 0.0035 | N/A |
| pointer_fixed4 | 128 | 4 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 4 | mean_post_entropy | N/A | 0.0132 ± 0.0099 | N/A |
| pointer_fixed4 | 128 | 4 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 4 | mean_post_identity_drift | N/A | 0.0021 ± 0.0023 | N/A |
| pointer_fixed4 | 128 | 4 | mean_post_identity_norm | N/A | 0.9983 ± 0.0016 | N/A |
| pointer_fixed4 | 128 | 4 | mean_post_margin | N/A | 0.9969 ± 0.0037 | N/A |
| pointer_fixed4 | 128 | 4 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 4 | mean_pre_identity_drift | N/A | 0.0016 ± 0.0019 | N/A |
| pointer_fixed4 | 128 | 4 | mean_pre_identity_norm | N/A | 0.9988 ± 0.0013 | N/A |
| pointer_fixed4 | 128 | 4 | mean_projected_query_norm | N/A | 3.9960 ± 0.0088 | N/A |
| pointer_fixed4 | 128 | 4 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 4 | mean_write_override_norm | N/A | 0.8412 ± 0.0456 | N/A |
| pointer_fixed4 | 128 | 8 | distinct_gold_nodes | N/A | 8.5599 ± 0.8499 | N/A |
| pointer_fixed4 | 128 | 8 | mean_entropy | N/A | 0.0139 ± 0.0087 | N/A |
| pointer_fixed4 | 128 | 8 | mean_margin | N/A | 0.9966 ± 0.0044 | N/A |
| pointer_fixed4 | 128 | 8 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 8 | mean_post_entropy | N/A | 0.0139 ± 0.0097 | N/A |
| pointer_fixed4 | 128 | 8 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0002 | N/A |
| pointer_fixed4 | 128 | 8 | mean_post_identity_drift | N/A | 0.0024 ± 0.0031 | N/A |
| pointer_fixed4 | 128 | 8 | mean_post_identity_norm | N/A | 0.9981 ± 0.0021 | N/A |
| pointer_fixed4 | 128 | 8 | mean_post_margin | N/A | 0.9965 ± 0.0049 | N/A |
| pointer_fixed4 | 128 | 8 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0002 | N/A |
| pointer_fixed4 | 128 | 8 | mean_pre_identity_drift | N/A | 0.0022 ± 0.0028 | N/A |
| pointer_fixed4 | 128 | 8 | mean_pre_identity_norm | N/A | 0.9983 ± 0.0019 | N/A |
| pointer_fixed4 | 128 | 8 | mean_projected_query_norm | N/A | 3.9942 ± 0.0093 | N/A |
| pointer_fixed4 | 128 | 8 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 8 | mean_write_override_norm | N/A | 0.8401 ± 0.0337 | N/A |
| pointer_fixed4 | 128 | 16 | distinct_gold_nodes | N/A | 15.3021 ± 1.7425 | N/A |
| pointer_fixed4 | 128 | 16 | mean_entropy | N/A | 0.0133 ± 0.0060 | N/A |
| pointer_fixed4 | 128 | 16 | mean_margin | N/A | 0.9968 ± 0.0025 | N/A |
| pointer_fixed4 | 128 | 16 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 16 | mean_post_entropy | N/A | 0.0134 ± 0.0064 | N/A |
| pointer_fixed4 | 128 | 16 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 16 | mean_post_identity_drift | N/A | 0.0023 ± 0.0018 | N/A |
| pointer_fixed4 | 128 | 16 | mean_post_identity_norm | N/A | 0.9982 ± 0.0013 | N/A |
| pointer_fixed4 | 128 | 16 | mean_post_margin | N/A | 0.9968 ± 0.0029 | N/A |
| pointer_fixed4 | 128 | 16 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 16 | mean_pre_identity_drift | N/A | 0.0021 ± 0.0017 | N/A |
| pointer_fixed4 | 128 | 16 | mean_pre_identity_norm | N/A | 0.9984 ± 0.0012 | N/A |
| pointer_fixed4 | 128 | 16 | mean_projected_query_norm | N/A | 3.9943 ± 0.0062 | N/A |
| pointer_fixed4 | 128 | 16 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 16 | mean_write_override_norm | N/A | 0.8425 ± 0.0264 | N/A |
| pointer_fixed4 | 128 | 32 | distinct_gold_nodes | N/A | 27.3281 ± 2.8241 | N/A |
| pointer_fixed4 | 128 | 32 | mean_entropy | N/A | 0.0131 ± 0.0044 | N/A |
| pointer_fixed4 | 128 | 32 | mean_margin | N/A | 0.9969 ± 0.0018 | N/A |
| pointer_fixed4 | 128 | 32 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 32 | mean_post_entropy | N/A | 0.0131 ± 0.0043 | N/A |
| pointer_fixed4 | 128 | 32 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 32 | mean_post_identity_drift | N/A | 0.0022 ± 0.0012 | N/A |
| pointer_fixed4 | 128 | 32 | mean_post_identity_norm | N/A | 0.9983 ± 0.0009 | N/A |
| pointer_fixed4 | 128 | 32 | mean_post_margin | N/A | 0.9969 ± 0.0018 | N/A |
| pointer_fixed4 | 128 | 32 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 32 | mean_pre_identity_drift | N/A | 0.0021 ± 0.0012 | N/A |
| pointer_fixed4 | 128 | 32 | mean_pre_identity_norm | N/A | 0.9983 ± 0.0008 | N/A |
| pointer_fixed4 | 128 | 32 | mean_projected_query_norm | N/A | 3.9943 ± 0.0046 | N/A |
| pointer_fixed4 | 128 | 32 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 32 | mean_write_override_norm | N/A | 0.8411 ± 0.0216 | N/A |
| pointer_fixed4 | 128 | 64 | distinct_gold_nodes | N/A | 45.5443 ± 4.3392 | N/A |
| pointer_fixed4 | 128 | 64 | mean_entropy | N/A | 0.0132 ± 0.0041 | N/A |
| pointer_fixed4 | 128 | 64 | mean_margin | N/A | 0.9969 ± 0.0017 | N/A |
| pointer_fixed4 | 128 | 64 | mean_mlp_identity_proposal_norm | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 64 | mean_post_entropy | N/A | 0.0132 ± 0.0041 | N/A |
| pointer_fixed4 | 128 | 64 | mean_post_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 64 | mean_post_identity_drift | N/A | 0.0022 ± 0.0012 | N/A |
| pointer_fixed4 | 128 | 64 | mean_post_identity_norm | N/A | 0.9982 ± 0.0008 | N/A |
| pointer_fixed4 | 128 | 64 | mean_post_margin | N/A | 0.9969 ± 0.0017 | N/A |
| pointer_fixed4 | 128 | 64 | mean_pre_identity_cosine_drift | N/A | 0.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 64 | mean_pre_identity_drift | N/A | 0.0022 ± 0.0011 | N/A |
| pointer_fixed4 | 128 | 64 | mean_pre_identity_norm | N/A | 0.9983 ± 0.0008 | N/A |
| pointer_fixed4 | 128 | 64 | mean_projected_query_norm | N/A | 3.9940 ± 0.0044 | N/A |
| pointer_fixed4 | 128 | 64 | mean_strength | N/A | 4.0000 ± 0.0000 | N/A |
| pointer_fixed4 | 128 | 64 | mean_write_override_norm | N/A | 0.8419 ± 0.0181 | N/A |

## Provenance

```json
{
  "source": {
    "commit": "11b921a18d5f6839b11e49ca6b17db6eb8c83547",
    "dirty": false,
    "files": {
      "src/topoformer/__init__.py": "a597933984bcde7b51e2dcc92f39749bc2c7600623265647e60156eab7265730",
      "src/topoformer/attention.py": "ac1486c8f5398bba9e084e2ddda85ba402ed9633638e68029c54149fe0152854",
      "src/topoformer/binding_analysis.py": "2266e3ece4998dcb4d92bf354ec26b15fa92e284de7a43ea55948b59f62eb8e0",
      "src/topoformer/binding_metrics.py": "aca234bb0fa99d6c58f6ef076e979590b09f5a127a0f443f7537d1ae0f9c1741",
      "src/topoformer/binding_model.py": "c4cf128528f8faf6fcca7b8e9ea66aaaaf3bed21bc589b9750ec741334c6f2c4",
      "src/topoformer/binding_study.py": "f88e71ce05532384abf653fd723f04571a6fae971e49a36b7a6a0f10a721c0ee",
      "src/topoformer/data.py": "c4ce845820c7efb26a701c0dfcd2f6ecbcb7e38b045f3918e61416e17b387b21",
      "src/topoformer/evaluation.py": "c5fe751c5e6cb55d3ee88a2c3f1f8b08e8513438a76f11eeeb39aeb34a86ee11",
      "src/topoformer/experiment.py": "fa34e5fe18132ad1a4d59bfc42fc74949aeabe88977877a8b67b087757f3d763",
      "src/topoformer/graphs.py": "16ce61cc40a48d9e294fa215f6ec512c4fdc2d82273867b59146f75e112d9a07",
      "src/topoformer/grounding.py": "2296c904aa8e997c219e67a7e0d4f9791723a4fc05d168668537731dde449ae7",
      "src/topoformer/grounding_analysis.py": "5f9725b42562edadff60170392d65216767433ed44fd3929b4f95ff855bb700a",
      "src/topoformer/grounding_study.py": "f2f51cfa2801332aab2edad6a28069a8890afa2a3b2c373aa31f8c14e6820927",
      "src/topoformer/model.py": "3eeb52c71fb6a23dcee7a7064fb198b17a8ccbc7eae05ec918075c5a3ca5237b",
      "src/topoformer/runtime_graph.py": "c7c9884febf8dc15009ddb337ae4c885058186d0faa7741c38247697e9e3eefe",
      "src/topoformer/study.py": "27c6667a7c0ea68cbd2b33796b95ef3bc3083b399078946336e80c62352fe71b",
      "src/topoformer/study_analysis.py": "b012d9edb9756fb641a92f6331377e01bb2e9f84d638c44cc764051b8fe461a3",
      "src/topoformer/study_data.py": "59d3293c7114598ff14d533f0cf040184068d67c77f36418286431cf276a1b1d",
      "src/topoformer/study_model.py": "455ed9b4e8eea422cb1da5db36dd525319b1bae81743bc407586b1565170e94f",
      "src/topoformer/training.py": "697cfd27c0db940bcc489f02e25b7504aaff80305a46dc24dc465cbd11176a43",
      "src/topoformer/traversal_data.py": "714cf6d2cecf03bccc3ad7dd17cb092fa233b6599d8e0a702ef62fc579a874a6",
      "src/topoformer/traversal_model.py": "3bce439db58f4482b7760ae174a103eda151958e3a44f281d675aa72634afeb6",
      "src/topoformer/traversal_oracle.py": "d690fe5d4b10ce8accf5dd8d1157166194a5dd6fde80ff4d122c1e52cf3bcb66"
    }
  },
  "config_hash": "5a0c0ab46e88081a860ca781d8c1d3444d489a083e67308d637aba2adf8e36f1",
  "artifact": {
    "metrics_file": "metrics.jsonl.gz",
    "metrics_sha256": "ae17b0e7519fe0be104af5acde2a4ec3cc15506dd17e6832f18510892c2f490d",
    "analyzer_sha256": "2266e3ece4998dcb4d92bf354ec26b15fa92e284de7a43ea55948b59f62eb8e0",
    "shared_analysis_sha256": "5f9725b42562edadff60170392d65216767433ed44fd3929b4f95ff855bb700a",
    "config_file_sha256": "840201c2005874e0f9c85b333a296aa9a35230fd4581f2baddd3d32728e2525e"
  }
}
```
