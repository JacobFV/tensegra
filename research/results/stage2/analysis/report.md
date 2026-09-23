# Stage 2 artifact analysis

Complete runs: 570 / 570.

Incomplete runs and non-finite evaluations are excluded rather than counted as observations. Standard deviations are population SDs across seeds; transfer observations are averaged within seed first.

## Exclusions

| reason | count |
|---|---:|
| incomplete runs | 0 |
| nonfinite evaluations | 0 |
| duplicate evaluations | 0 |
| nonfinite coefficients | 0 |
| invalid evaluation step | 0 |
| duplicate runs identical | 0 |
| duplicate runs conflicting | 0 |
| invalid efficiency curves | 0 |
| efficiency reference mismatch | 0 |
| efficiency normalization mismatch | 0 |

## corruption

| domain | mode | selected | config | count | corruption | fraction | train→eval | split | metric | mean | pop. SD | seeds | graphs | precision | recall |
|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| robot | hard | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.063437 | 0.0388931 | 3 | 3 | 0.916667 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | one_step_normalized_mse | 0.509477 | 0.0313184 | 3 | 3 | 0.916667 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.881397 | 0.0374824 | 3 | 3 | 0.916667 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0630395 | 0.033056 | 3 | 3 | 0.785714 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | one_step_normalized_mse | 0.518889 | 0.0310667 | 3 | 3 | 0.785714 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.880558 | 0.0448451 | 3 | 3 | 0.785714 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0706689 | 0.0320764 | 3 | 3 | 0.666667 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | one_step_normalized_mse | 0.524597 | 0.0279015 | 3 | 3 | 0.666667 | 1 |
| robot | hard | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.893721 | 0.0423352 | 3 | 3 | 0.666667 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0777419 | 0.0446149 | 3 | 3 | 0.916667 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.510345 | 0.031114 | 3 | 3 | 0.916667 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.902458 | 0.0485953 | 3 | 3 | 0.916667 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0966476 | 0.0425816 | 3 | 3 | 0.785714 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.526139 | 0.0313192 | 3 | 3 | 0.785714 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.922242 | 0.0542416 | 3 | 3 | 0.785714 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.108982 | 0.0472426 | 3 | 3 | 0.666667 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.537507 | 0.0279995 | 3 | 3 | 0.666667 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.937879 | 0.0609254 | 3 | 3 | 0.666667 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0724573 | 0.0439823 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.504196 | 0.0317141 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.893345 | 0.0491967 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0758241 | 0.0446186 | 3 | 3 | 1 | 0.909091 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.511945 | 0.032358 | 3 | 3 | 1 | 0.909091 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.90314 | 0.0445121 | 3 | 3 | 1 | 0.909091 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0959239 | 0.0511655 | 3 | 3 | 1 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.526565 | 0.0284602 | 3 | 3 | 1 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.91645 | 0.0567548 | 3 | 3 | 1 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.115428 | 0.052085 | 3 | 3 | 1 | 0.5 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.545347 | 0.0275686 | 3 | 3 | 1 | 0.5 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.941411 | 0.0690598 | 3 | 3 | 1 | 0.5 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.124292 | 0.0526328 | 3 | 3 | 0.727273 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.551057 | 0.0328403 | 3 | 3 | 0.727273 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.957595 | 0.0511886 | 3 | 3 | 0.727273 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0713203 | 0.0360989 | 3 | 3 | 1 | 0.909091 |
| robot | hard | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | one_step_normalized_mse | 0.51261 | 0.0321165 | 3 | 3 | 1 | 0.909091 |
| robot | hard | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.895055 | 0.0433662 | 3 | 3 | 1 | 0.909091 |
| robot | hard | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0565569 | 0.0157081 | 3 | 3 | 1 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | one_step_normalized_mse | 0.524347 | 0.0302139 | 3 | 3 | 1 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.869911 | 0.0345958 | 3 | 3 | 1 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0534063 | 0.00224667 | 3 | 3 | 1 | 0.5 |
| robot | hard | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | one_step_normalized_mse | 0.53447 | 0.029827 | 3 | 3 | 1 | 0.5 |
| robot | hard | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.862518 | 0.030415 | 3 | 3 | 1 | 0.5 |
| robot | hard | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0697112 | 0.022784 | 3 | 3 | 0.727273 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | one_step_normalized_mse | 0.537697 | 0.0324576 | 3 | 3 | 0.727273 | 0.727273 |
| robot | hard | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.891919 | 0.0203449 | 3 | 3 | 0.727273 | 0.727273 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 0.916667 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 0.916667 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 0.916667 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 0.785714 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 0.785714 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 0.785714 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 0.666667 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 0.666667 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 0.666667 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 1 | 0.909091 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 1 | 0.909091 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 1 | 0.909091 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 1 | 0.727273 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 1 | 0.727273 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 1 | 0.727273 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 1 | 0.5 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 1 | 0.5 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 1 | 0.5 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 0.727273 | 0.727273 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 0.727273 | 0.727273 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0826881 | 0.00664794 | 3 | 3 | 0.916667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | one_step_normalized_mse | 0.520964 | 0.0338312 | 3 | 3 | 0.916667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.896962 | 0.0209289 | 3 | 3 | 0.916667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.086471 | 0.0145805 | 3 | 3 | 0.785714 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | one_step_normalized_mse | 0.528263 | 0.033443 | 3 | 3 | 0.785714 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.911374 | 0.0201502 | 3 | 3 | 0.785714 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.089341 | 0.0169784 | 3 | 3 | 0.666667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | one_step_normalized_mse | 0.532169 | 0.0312716 | 3 | 3 | 0.666667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.918179 | 0.0276641 | 3 | 3 | 0.666667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0922682 | 0.017489 | 3 | 3 | 0.916667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.520838 | 0.0345233 | 3 | 3 | 0.916667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.900031 | 0.0367693 | 3 | 3 | 0.916667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0975275 | 0.0183707 | 3 | 3 | 0.785714 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.52949 | 0.0349872 | 3 | 3 | 0.785714 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.906405 | 0.0352744 | 3 | 3 | 0.785714 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.101844 | 0.0196033 | 3 | 3 | 0.666667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.535378 | 0.0330737 | 3 | 3 | 0.666667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.91075 | 0.031431 | 3 | 3 | 0.666667 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0902515 | 0.0185822 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.517743 | 0.0348917 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.89651 | 0.0404985 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0928831 | 0.0184128 | 3 | 3 | 1 | 0.909091 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.522567 | 0.0344459 | 3 | 3 | 1 | 0.909091 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.900023 | 0.0394073 | 3 | 3 | 1 | 0.909091 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0996247 | 0.0205485 | 3 | 3 | 1 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.529702 | 0.0322353 | 3 | 3 | 1 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.911287 | 0.041429 | 3 | 3 | 1 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.105307 | 0.0204473 | 3 | 3 | 1 | 0.5 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.53627 | 0.0304482 | 3 | 3 | 1 | 0.5 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.91986 | 0.0347235 | 3 | 3 | 1 | 0.5 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.108649 | 0.0192505 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.544181 | 0.0362212 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.922273 | 0.0333498 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0909251 | 0.0143097 | 3 | 3 | 1 | 0.909091 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | one_step_normalized_mse | 0.523048 | 0.0344995 | 3 | 3 | 1 | 0.909091 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.900255 | 0.0284071 | 3 | 3 | 1 | 0.909091 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.101241 | 0.0206131 | 3 | 3 | 1 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | one_step_normalized_mse | 0.530513 | 0.032347 | 3 | 3 | 1 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.914793 | 0.0356043 | 3 | 3 | 1 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.120278 | 0.029757 | 3 | 3 | 1 | 0.5 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | one_step_normalized_mse | 0.537484 | 0.0298939 | 3 | 3 | 1 | 0.5 |
| robot | soft1 | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.933321 | 0.0376407 | 3 | 3 | 1 | 0.5 |
| robot | soft1 | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0960639 | 0.0252482 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | one_step_normalized_mse | 0.54042 | 0.0338532 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft1 | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.9281 | 0.0351764 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0674485 | 0.0430026 | 3 | 3 | 0.916667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | one_step_normalized_mse | 0.510284 | 0.0317542 | 3 | 3 | 0.916667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.888814 | 0.049507 | 3 | 3 | 0.916667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0669735 | 0.0356348 | 3 | 3 | 0.785714 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | one_step_normalized_mse | 0.519679 | 0.0312681 | 3 | 3 | 0.785714 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.88764 | 0.0515836 | 3 | 3 | 0.785714 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0732076 | 0.0330691 | 3 | 3 | 0.666667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | one_step_normalized_mse | 0.525177 | 0.0283312 | 3 | 3 | 0.666667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.898684 | 0.0470319 | 3 | 3 | 0.666667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0787967 | 0.0506894 | 3 | 3 | 0.916667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.510571 | 0.0319216 | 3 | 3 | 0.916667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.907931 | 0.0639299 | 3 | 3 | 0.916667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0958666 | 0.0504272 | 3 | 3 | 0.785714 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.524718 | 0.0319515 | 3 | 3 | 0.785714 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.924547 | 0.068979 | 3 | 3 | 0.785714 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.106785 | 0.0546508 | 3 | 3 | 0.666667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.534874 | 0.0290473 | 3 | 3 | 0.666667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.937992 | 0.0757448 | 3 | 3 | 0.666667 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0746878 | 0.0506047 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.505028 | 0.032435 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.900719 | 0.0633136 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0769345 | 0.0499578 | 3 | 3 | 1 | 0.909091 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.512138 | 0.0328944 | 3 | 3 | 1 | 0.909091 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.907802 | 0.0605746 | 3 | 3 | 1 | 0.909091 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0876383 | 0.0516058 | 3 | 3 | 1 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.524077 | 0.0298308 | 3 | 3 | 1 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.915233 | 0.0651407 | 3 | 3 | 1 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.100367 | 0.0578513 | 3 | 3 | 1 | 0.5 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.537259 | 0.0274122 | 3 | 3 | 1 | 0.5 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.932185 | 0.0753864 | 3 | 3 | 1 | 0.5 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.115841 | 0.0531726 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.546878 | 0.0335201 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.950873 | 0.0609932 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0775864 | 0.0487478 | 3 | 3 | 1 | 0.909091 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | one_step_normalized_mse | 0.512912 | 0.0325154 | 3 | 3 | 1 | 0.909091 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.90614 | 0.0634112 | 3 | 3 | 1 | 0.909091 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0705905 | 0.0384013 | 3 | 3 | 1 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | one_step_normalized_mse | 0.523492 | 0.0307436 | 3 | 3 | 1 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.892322 | 0.0572662 | 3 | 3 | 1 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0766518 | 0.0494168 | 3 | 3 | 1 | 0.5 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | one_step_normalized_mse | 0.532578 | 0.0292432 | 3 | 3 | 1 | 0.5 |
| robot | soft4 | True | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.903784 | 0.0733929 | 3 | 3 | 1 | 0.5 |
| robot | soft4 | True | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0763507 | 0.0293537 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | one_step_normalized_mse | 0.537226 | 0.0325868 | 3 | 3 | 0.727273 | 0.727273 |
| robot | soft4 | True | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.902395 | 0.0234513 | 3 | 3 | 0.727273 | 0.727273 |
| sparse | hard | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0387764 | 0.0155723 | 3 | 3 | 0.903139 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | one_step_normalized_mse | 0.599407 | 0.0106913 | 3 | 3 | 0.903139 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.951016 | 0.0602062 | 3 | 3 | 0.903139 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0402327 | 0.0110717 | 3 | 3 | 0.800089 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | one_step_normalized_mse | 0.604497 | 0.0101199 | 3 | 3 | 0.800089 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.948594 | 0.0575896 | 3 | 3 | 0.800089 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0364281 | 0.014138 | 3 | 3 | 0.666804 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | one_step_normalized_mse | 0.610008 | 0.00752138 | 3 | 3 | 0.666804 | 1 |
| sparse | hard | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.943268 | 0.0663431 | 3 | 3 | 0.666804 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0396899 | 0.00522618 | 3 | 3 | 0.903139 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.600416 | 0.0108804 | 3 | 3 | 0.903139 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.953376 | 0.0505369 | 3 | 3 | 0.903139 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.042819 | 0.00484738 | 3 | 3 | 0.800089 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.606935 | 0.0103037 | 3 | 3 | 0.800089 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.952089 | 0.0471966 | 3 | 3 | 0.800089 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0427595 | 0.00394252 | 3 | 3 | 0.666804 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.614444 | 0.00800991 | 3 | 3 | 0.666804 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.954299 | 0.0533724 | 3 | 3 | 0.666804 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0378409 | 0.00479414 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.59399 | 0.0082201 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.946492 | 0.0515629 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0416496 | 0.00662907 | 3 | 3 | 1 | 0.892739 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.595454 | 0.00561625 | 3 | 3 | 1 | 0.892739 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.952088 | 0.0482461 | 3 | 3 | 1 | 0.892739 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0655074 | 0.0270554 | 3 | 3 | 1 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.617053 | 0.00533621 | 3 | 3 | 1 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.993236 | 0.0607579 | 3 | 3 | 1 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0809826 | 0.0297106 | 3 | 3 | 1 | 0.500231 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.636314 | 0.0148572 | 3 | 3 | 1 | 0.500231 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.993786 | 0.0662531 | 3 | 3 | 1 | 0.500231 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0529521 | 0.00973275 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.622078 | 0.00557376 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.97157 | 0.0623424 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0429428 | 0.0115277 | 3 | 3 | 1 | 0.892739 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | one_step_normalized_mse | 0.5948 | 0.0050689 | 3 | 3 | 1 | 0.892739 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.955362 | 0.0396815 | 3 | 3 | 1 | 0.892739 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0526779 | 0.011553 | 3 | 3 | 1 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | one_step_normalized_mse | 0.612293 | 0.00321636 | 3 | 3 | 1 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.953768 | 0.0462869 | 3 | 3 | 1 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0527963 | 0.00716179 | 3 | 3 | 1 | 0.500231 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | one_step_normalized_mse | 0.621574 | 0.0081795 | 3 | 3 | 1 | 0.500231 |
| sparse | hard | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.925314 | 0.0415756 | 3 | 3 | 1 | 0.500231 |
| sparse | hard | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0582924 | 0.03246 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | one_step_normalized_mse | 0.617184 | 0.00536757 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | hard | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.96206 | 0.0759633 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 0.903139 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 0.903139 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 0.903139 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 0.800089 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 0.800089 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 0.800089 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 0.666804 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 0.666804 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 0.666804 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 1 | 0.892739 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 1 | 0.892739 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 1 | 0.892739 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 1 | 0.750116 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 1 | 0.750116 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 1 | 0.750116 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 1 | 0.500231 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 1 | 0.500231 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 1 | 0.500231 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.05437 | 0.023473 | 3 | 3 | 0.903139 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | one_step_normalized_mse | 0.606364 | 0.00960046 | 3 | 3 | 0.903139 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.968145 | 0.0709414 | 3 | 3 | 0.903139 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0531763 | 0.0260728 | 3 | 3 | 0.800089 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | one_step_normalized_mse | 0.609704 | 0.00868114 | 3 | 3 | 0.800089 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.965896 | 0.0714154 | 3 | 3 | 0.800089 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0493956 | 0.0238776 | 3 | 3 | 0.666804 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | one_step_normalized_mse | 0.613781 | 0.00683165 | 3 | 3 | 0.666804 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.962371 | 0.0718317 | 3 | 3 | 0.666804 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0593885 | 0.0250874 | 3 | 3 | 0.903139 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.606552 | 0.0101885 | 3 | 3 | 0.903139 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.973586 | 0.0745408 | 3 | 3 | 0.903139 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.061747 | 0.0252925 | 3 | 3 | 0.800089 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.610797 | 0.00996624 | 3 | 3 | 0.800089 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.974842 | 0.0743578 | 3 | 3 | 0.800089 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.063167 | 0.0249359 | 3 | 3 | 0.666804 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.616901 | 0.00815125 | 3 | 3 | 0.666804 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.976667 | 0.0757964 | 3 | 3 | 0.666804 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.058167 | 0.0253818 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.601981 | 0.00797885 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.969459 | 0.0727734 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.059718 | 0.0260194 | 3 | 3 | 1 | 0.892739 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.6025 | 0.00618329 | 3 | 3 | 1 | 0.892739 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.971324 | 0.0725055 | 3 | 3 | 1 | 0.892739 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0627283 | 0.0270733 | 3 | 3 | 1 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.612055 | 0.00576353 | 3 | 3 | 1 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.975329 | 0.0732756 | 3 | 3 | 1 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0648578 | 0.0275531 | 3 | 3 | 1 | 0.500231 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.619435 | 0.00943978 | 3 | 3 | 1 | 0.500231 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.974811 | 0.0739044 | 3 | 3 | 1 | 0.500231 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0648934 | 0.0274814 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.620012 | 0.00428591 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.976944 | 0.0751089 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0556126 | 0.0288337 | 3 | 3 | 1 | 0.892739 |
| sparse | soft1 | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | one_step_normalized_mse | 0.602496 | 0.00606186 | 3 | 3 | 1 | 0.892739 |
| sparse | soft1 | False | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.967462 | 0.0672487 | 3 | 3 | 1 | 0.892739 |
| sparse | soft1 | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0548452 | 0.0317379 | 3 | 3 | 1 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | one_step_normalized_mse | 0.61184 | 0.00569481 | 3 | 3 | 1 | 0.750116 |
| sparse | soft1 | False | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.967678 | 0.0671521 | 3 | 3 | 1 | 0.750116 |
| sparse | soft1 | True | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0587115 | 0.0419471 | 3 | 3 | 1 | 0.500231 |
| sparse | soft1 | True | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | one_step_normalized_mse | 0.617866 | 0.00847139 | 3 | 3 | 1 | 0.500231 |
| sparse | soft1 | True | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.971193 | 0.0802518 | 3 | 3 | 1 | 0.500231 |
| sparse | soft1 | True | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.052841 | 0.0245506 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft1 | True | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | one_step_normalized_mse | 0.617036 | 0.00530874 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft1 | True | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.964549 | 0.074428 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0416133 | 0.0166818 | 3 | 3 | 0.903139 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | one_step_normalized_mse | 0.599558 | 0.0106979 | 3 | 3 | 0.903139 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.1 | 128 | add | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.956307 | 0.0602158 | 3 | 3 | 0.903139 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0425736 | 0.0128128 | 3 | 3 | 0.800089 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | one_step_normalized_mse | 0.604675 | 0.0100595 | 3 | 3 | 0.800089 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.25 | 128 | add | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.95277 | 0.0576374 | 3 | 3 | 0.800089 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0381832 | 0.0149609 | 3 | 3 | 0.666804 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | one_step_normalized_mse | 0.610093 | 0.00748944 | 3 | 3 | 0.666804 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:add@0.5 | 128 | add | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.945832 | 0.0659034 | 3 | 3 | 0.666804 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0400289 | 0.00855536 | 3 | 3 | 0.903139 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.600237 | 0.0107632 | 3 | 3 | 0.903139 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.955787 | 0.0569047 | 3 | 3 | 0.903139 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0436749 | 0.00780832 | 3 | 3 | 0.800089 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.60684 | 0.0104134 | 3 | 3 | 0.800089 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.955652 | 0.0540659 | 3 | 3 | 0.800089 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0437426 | 0.00803996 | 3 | 3 | 0.666804 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.613983 | 0.0080219 | 3 | 3 | 0.666804 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | add | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.957813 | 0.0595748 | 3 | 3 | 0.666804 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0379028 | 0.00908875 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.594014 | 0.00806895 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.948364 | 0.0578793 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0418994 | 0.0106558 | 3 | 3 | 1 | 0.892739 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | one_step_normalized_mse | 0.595301 | 0.00550046 | 3 | 3 | 1 | 0.892739 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.1 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.954246 | 0.0552903 | 3 | 3 | 1 | 0.892739 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0549113 | 0.0204256 | 3 | 3 | 1 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.612963 | 0.00475146 | 3 | 3 | 1 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.977268 | 0.0582719 | 3 | 3 | 1 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0631646 | 0.0215987 | 3 | 3 | 1 | 0.500231 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | one_step_normalized_mse | 0.628344 | 0.0123859 | 3 | 3 | 1 | 0.500231 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | drop | 0.5 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.97497 | 0.0623389 | 3 | 3 | 1 | 0.500231 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | deterministic_rollout_normalized_mse | 0.0500383 | 0.0132977 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | one_step_normalized_mse | 0.619538 | 0.00430233 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | mixed | 0.25 | —→12 | runtime_corruption | stochastic_rollout_normalized_mse | 0.967271 | 0.0605223 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | deterministic_rollout_normalized_mse | 0.0432386 | 0.0152512 | 3 | 3 | 1 | 0.892739 |
| sparse | soft4 | True | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | one_step_normalized_mse | 0.594767 | 0.00501833 | 3 | 3 | 1 | 0.892739 |
| sparse | soft4 | True | uniform:fixed:n=128:drop@0.1 | 128 | drop | 0.1 | —→12 | test | stochastic_rollout_normalized_mse | 0.957374 | 0.0467847 | 3 | 3 | 1 | 0.892739 |
| sparse | soft4 | True | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0487693 | 0.0220262 | 3 | 3 | 1 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | one_step_normalized_mse | 0.609393 | 0.00318924 | 3 | 3 | 1 | 0.750116 |
| sparse | soft4 | True | uniform:fixed:n=128:drop@0.25 | 128 | drop | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.968855 | 0.0607402 | 3 | 3 | 1 | 0.750116 |
| sparse | soft4 | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | deterministic_rollout_normalized_mse | 0.0560637 | 0.0171298 | 3 | 3 | 1 | 0.500231 |
| sparse | soft4 | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | one_step_normalized_mse | 0.616702 | 0.00790036 | 3 | 3 | 1 | 0.500231 |
| sparse | soft4 | False | uniform:fixed:n=128:drop@0.5 | 128 | drop | 0.5 | —→12 | test | stochastic_rollout_normalized_mse | 0.964282 | 0.0762984 | 3 | 3 | 1 | 0.500231 |
| sparse | soft4 | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | deterministic_rollout_normalized_mse | 0.0625039 | 0.03858 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft4 | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | one_step_normalized_mse | 0.615826 | 0.00504951 | 3 | 3 | 0.750116 | 0.750116 |
| sparse | soft4 | False | uniform:fixed:n=128:mixed@0.25 | 128 | mixed | 0.25 | —→12 | test | stochastic_rollout_normalized_mse | 0.981784 | 0.101614 | 3 | 3 | 0.750116 | 0.750116 |

## efficiency

| domain | mode | selected | config | count | corruption | fraction | train→eval | split | metric | mean | pop. SD | seeds | graphs | precision | recall |
|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0724573 | 0.0439823 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.504196 | 0.0317141 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.893345 | 0.0491967 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0844814 | 0.0149145 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.519354 | 0.0338251 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.83534 | 0.0299726 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.109711 | 0.0904513 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.507969 | 0.0364525 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.880625 | 0.0943297 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0735247 | 0.0576763 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.505926 | 0.0380996 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.835596 | 0.074243 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0354884 | 0.0280839 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.504136 | 0.033197 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.831212 | 0.0364557 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.260508 | 0.160152 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.625324 | 0.0882034 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.05331 | 0.226812 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.130216 | 0.0410511 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.574707 | 0.0436193 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.907813 | 0.0383896 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.148174 | 0.059158 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.565508 | 0.0373519 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.923485 | 0.0715097 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.136782 | 0.0491563 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.56611 | 0.0406754 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.940089 | 0.044691 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0631176 | 0.012359 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.560991 | 0.0342453 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.890063 | 0.0154232 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.563024 | 0.25626 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626381 | 0.0513715 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.3265 | 0.281557 | 3 | 3 | 1 | 1 |
| robot | permuted1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0903616 | 0.0315781 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.559447 | 0.0324914 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.92807 | 0.0616844 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.152317 | 0.0638398 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.573408 | 0.0389107 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.936465 | 0.0476758 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.142943 | 0.0552961 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.564678 | 0.0357873 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.919057 | 0.0683464 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.112995 | 0.0407813 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.56378 | 0.0372934 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.913853 | 0.0485782 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0607469 | 0.00909347 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.560652 | 0.0336035 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.885837 | 0.0173451 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.92785 | 0.157164 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.653967 | 0.0720806 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted1 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.61357 | 0.113145 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0731934 | 0.0144584 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.559634 | 0.0326819 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.889926 | 0.0463266 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.268499 | 0.149135 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.590111 | 0.0377749 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.0009 | 0.121313 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.18193 | 0.0830182 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.567371 | 0.0362748 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.936534 | 0.106766 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.115275 | 0.0584162 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.569387 | 0.035652 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.900296 | 0.0619024 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.06847 | 0.00649313 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.562447 | 0.0353783 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.870933 | 0.0460966 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.631922 | 0.296807 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.728194 | 0.0828989 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.39306 | 0.419609 | 3 | 3 | 0.181818 | 0.181818 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0902515 | 0.0185822 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.517743 | 0.0348917 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.89651 | 0.0404985 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0787876 | 0.0206875 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.527064 | 0.0416545 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.857415 | 0.0267018 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.332046 | 0.406743 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.525785 | 0.0496285 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.08372 | 0.381764 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0632896 | 0.0376525 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.520866 | 0.0420016 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.831227 | 0.0556627 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0469405 | 0.0122352 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.516858 | 0.0365805 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.849207 | 0.0484091 | 3 | 3 | 1 | 1 |
| robot | soft1 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.37901 | 0.099337 | 3 | 3 | 1 | 1 |
| robot | soft1 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.596824 | 0.0675171 | 3 | 3 | 1 | 1 |
| robot | soft1 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.16096 | 0.181033 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0746878 | 0.0506047 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.505028 | 0.032435 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.900719 | 0.0633136 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0680344 | 0.0265524 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.516331 | 0.032966 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.834113 | 0.0256097 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.145986 | 0.134615 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.509566 | 0.0376085 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.918211 | 0.133256 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0672661 | 0.0427308 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.50665 | 0.0373112 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.833467 | 0.0666435 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0335023 | 0.0246716 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.505207 | 0.0340143 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.833366 | 0.0347354 | 3 | 3 | 1 | 1 |
| robot | soft4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.279141 | 0.146609 | 3 | 3 | 1 | 1 |
| robot | soft4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.60797 | 0.0916574 | 3 | 3 | 1 | 1 |
| robot | soft4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.11997 | 0.224009 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0378409 | 0.00479414 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.59399 | 0.0082201 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.946492 | 0.0515629 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.159528 | 0.0661496 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.628204 | 0.0136561 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.980903 | 0.0513482 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.199201 | 0.21513 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.59826 | 0.0155876 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.05445 | 0.259782 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0726283 | 0.054226 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.600074 | 0.0123028 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.91857 | 0.0879282 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0348248 | 0.0173433 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.595989 | 0.0115068 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.899944 | 0.0552594 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.389936 | 0.234284 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.785445 | 0.0125382 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.2472 | 0.196106 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0733962 | 0.0102392 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.640387 | 0.00926907 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.934277 | 0.0734021 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.251828 | 0.162483 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.63317 | 0.0139516 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.07765 | 0.200959 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0854357 | 0.0652516 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.630523 | 0.0128899 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.947613 | 0.0798455 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0401558 | 0.00932239 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626512 | 0.00847388 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.920834 | 0.0441247 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.580116 | 0.22414 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.718145 | 0.0220237 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.36756 | 0.273144 | 3 | 3 | 1 | 1 |
| sparse | permuted1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0494851 | 0.0266617 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.627358 | 0.00941021 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.956416 | 0.0721416 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.174038 | 0.109937 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.64381 | 0.011604 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.0562 | 0.194912 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.238834 | 0.169254 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.633422 | 0.0151341 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.06977 | 0.204719 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0935072 | 0.0643168 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.631701 | 0.0142757 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.968913 | 0.0897543 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0405194 | 0.00171992 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.627957 | 0.00943504 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.926708 | 0.0547722 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.413959 | 0.0738219 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.744041 | 0.0203233 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted1 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.24864 | 0.17789 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0513232 | 0.0232586 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.632955 | 0.00953346 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.945113 | 0.0573547 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.151092 | 0.047576 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.661404 | 0.0109743 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.980509 | 0.0891421 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.220273 | 0.15932 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.637712 | 0.0154341 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.05131 | 0.197624 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.144476 | 0.0918211 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.642053 | 0.0139437 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.04624 | 0.135662 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0539268 | 0.0172072 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.634307 | 0.00982994 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.929919 | 0.0607552 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.341002 | 0.0288013 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.812144 | 0.0342645 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.15716 | 0.02812 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.058167 | 0.0253818 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.601981 | 0.00797885 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.969459 | 0.0727734 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.354897 | 0.241134 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.622675 | 0.0142769 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.18498 | 0.235757 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.241909 | 0.24652 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.606187 | 0.017031 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.06945 | 0.265534 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0716075 | 0.00867399 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.603963 | 0.0121252 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.90334 | 0.0372891 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0444623 | 0.00940244 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.601048 | 0.0105003 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.921462 | 0.0745367 | 3 | 3 | 1 | 1 |
| sparse | soft1 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.272702 | 0.0307581 | 3 | 3 | 1 | 1 |
| sparse | soft1 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.698556 | 0.00339682 | 3 | 3 | 1 | 1 |
| sparse | soft1 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.14456 | 0.127633 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0379028 | 0.00908875 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.594014 | 0.00806895 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.948364 | 0.0578793 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.128536 | 0.094435 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626909 | 0.0121975 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.964983 | 0.0346978 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.225129 | 0.229743 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.598474 | 0.0156115 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.07507 | 0.276 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0644423 | 0.0452139 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.599668 | 0.0124464 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.9011 | 0.0822409 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0336185 | 0.0184304 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.595252 | 0.0108532 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.901693 | 0.0584684 | 3 | 3 | 1 | 1 |
| sparse | soft4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.436471 | 0.132774 | 3 | 3 | 1 | 1 |
| sparse | soft4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.759051 | 0.0132915 | 3 | 3 | 1 | 1 |
| sparse | soft4 | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.2599 | 0.112777 | 3 | 3 | 1 | 1 |

## efficiency_identity

| domain | mode | selected | config | count | corruption | fraction | train→eval | split | metric | mean | pop. SD | seeds | graphs | precision | recall |
|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0671926 | 0.0384878 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.503589 | 0.0328008 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.890157 | 0.0402665 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.103761 | 0.00881749 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.523767 | 0.0371476 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.849758 | 0.0501449 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.109718 | 0.0839557 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.507345 | 0.0373722 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.878463 | 0.0912335 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0811415 | 0.0558376 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.506192 | 0.0391235 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.840215 | 0.081877 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0405258 | 0.0286893 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.504101 | 0.0337368 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.837188 | 0.0394864 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.284317 | 0.0595394 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.65315 | 0.0776898 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.11067 | 0.111783 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0975946 | 0.030019 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.559611 | 0.0336986 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.940592 | 0.0515416 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.154762 | 0.063686 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.58092 | 0.0500922 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.929409 | 0.0461506 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.151755 | 0.0646427 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.564995 | 0.0378547 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.926021 | 0.0775381 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.129246 | 0.0411108 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.566772 | 0.0414327 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.935056 | 0.035722 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0631925 | 0.0121746 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.560983 | 0.0341499 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.888065 | 0.0178599 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.383671 | 0.118875 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.674961 | 0.0821737 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.13725 | 0.147909 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0675776 | 0.042403 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.504439 | 0.0334806 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.89439 | 0.0501067 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0768485 | 0.0122474 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.520206 | 0.0358804 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.83689 | 0.040928 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.141731 | 0.121401 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.509146 | 0.03858 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.910661 | 0.124713 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0706006 | 0.0362289 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.50687 | 0.0381685 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.836972 | 0.0693901 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.037647 | 0.023092 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.50517 | 0.0346104 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.838046 | 0.0383901 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.347894 | 0.0820499 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.639955 | 0.0960618 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.25165 | 0.136997 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0355135 | 0.0107486 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.591174 | 0.00838077 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.946151 | 0.0589652 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.143603 | 0.069312 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.631909 | 0.0130893 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.984422 | 0.0812286 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.212489 | 0.221999 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.595737 | 0.0163935 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.06461 | 0.273482 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0711996 | 0.043061 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.597075 | 0.0125977 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.905524 | 0.0814043 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0336982 | 0.0198263 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.593853 | 0.0109092 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.900858 | 0.0627057 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.616076 | 0.116209 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.841589 | 0.0192424 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.53763 | 0.237067 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0483592 | 0.0289022 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626056 | 0.00846312 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.961216 | 0.0699583 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0875592 | 0.00687029 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.64455 | 0.011933 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.928326 | 0.0699533 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.251577 | 0.163668 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.632579 | 0.0142574 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.07807 | 0.20591 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0825156 | 0.0626101 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.629568 | 0.0130084 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.943603 | 0.0771677 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0373958 | 0.00797739 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626189 | 0.00824003 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.917884 | 0.0473949 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.675968 | 0.105464 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.766002 | 0.0267473 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.45334 | 0.0884233 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0362035 | 0.0118188 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.591695 | 0.0083175 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.947263 | 0.0619563 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.213179 | 0.232514 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.630561 | 0.0128345 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=16:clean@0.0 | 16 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.04888 | 0.15823 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.232605 | 0.223951 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.596716 | 0.0163792 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=256:clean@0.0 | 256 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.08019 | 0.2756 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0686098 | 0.0348506 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.597167 | 0.0126965 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=32:clean@0.0 | 32 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.895449 | 0.0671376 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0348589 | 0.0203453 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.59382 | 0.0107041 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=64:clean@0.0 | 64 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.903239 | 0.0661441 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.502482 | 0.0812176 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.825371 | 0.019441 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=8:clean@0.0 | 8 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.41301 | 0.138413 | 3 | 3 | 1 | 1 |

## heterogeneous

| domain | mode | selected | config | count | corruption | fraction | train→eval | split | metric | mean | pop. SD | seeds | graphs | precision | recall |
|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| robot | graph_input | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0508177 | 0.0115195 | 3 | 3 | 1 | 1 |
| robot | graph_input | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.620669 | 0.0578457 | 3 | 3 | 1 | 1 |
| robot | graph_input | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.905451 | 0.0192715 | 3 | 3 | 1 | 1 |
| robot | hard | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0454745 | 0.0100982 | 3 | 3 | 1 | 1 |
| robot | hard | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.610174 | 0.058664 | 3 | 3 | 1 | 1 |
| robot | hard | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.909367 | 0.00778108 | 3 | 3 | 1 | 1 |
| robot | none | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0532212 | 0.00559014 | 3 | 3 | 1 | 1 |
| robot | none | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.633773 | 0.0574241 | 3 | 3 | 1 | 1 |
| robot | none | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.91225 | 0.00310741 | 3 | 3 | 1 | 1 |
| robot | permuted4 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0520866 | 0.00712666 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.633466 | 0.0578506 | 3 | 3 | 0.181818 | 0.181818 |
| robot | permuted4 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.905825 | 0.016278 | 3 | 3 | 0.181818 | 0.181818 |
| robot | soft1 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0514188 | 0.00480847 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.621647 | 0.0562335 | 3 | 3 | 1 | 1 |
| robot | soft1 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.911786 | 0.00147772 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0456013 | 0.00929154 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.610647 | 0.0585992 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.906992 | 0.00433575 | 3 | 3 | 1 | 1 |
| robot | typed | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0315996 | 0.010892 | 3 | 3 | 1 | 1 |
| robot | typed | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.573811 | 0.0446414 | 3 | 3 | 1 | 1 |
| robot | typed | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.904172 | 0.0254464 | 3 | 3 | 1 | 1 |
| sparse | graph_input | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0446772 | 0.0164618 | 3 | 3 | 1 | 1 |
| sparse | graph_input | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.721663 | 0.0258595 | 3 | 3 | 1 | 1 |
| sparse | graph_input | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.974708 | 0.0141314 | 3 | 3 | 1 | 1 |
| sparse | hard | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0334425 | 0.0117865 | 3 | 3 | 1 | 1 |
| sparse | hard | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.714821 | 0.0261611 | 3 | 3 | 1 | 1 |
| sparse | hard | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.961756 | 0.0140857 | 3 | 3 | 1 | 1 |
| sparse | none | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0413388 | 0.00886126 | 3 | 3 | 1 | 1 |
| sparse | none | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.726977 | 0.0271277 | 3 | 3 | 1 | 1 |
| sparse | none | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.973084 | 0.0220331 | 3 | 3 | 1 | 1 |
| sparse | permuted4 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0457829 | 0.0186461 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.72774 | 0.0270647 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | permuted4 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.974141 | 0.00981796 | 3 | 3 | 0.302341 | 0.302341 |
| sparse | soft1 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0347596 | 0.00446943 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.720838 | 0.0275776 | 3 | 3 | 1 | 1 |
| sparse | soft1 | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.965703 | 0.0262946 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0327827 | 0.0100964 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.714902 | 0.0265445 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.961035 | 0.015767 | 3 | 3 | 1 | 1 |
| sparse | typed | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0280803 | 0.00969519 | 3 | 3 | 1 | 1 |
| sparse | typed | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.702647 | 0.0184862 | 3 | 3 | 1 | 1 |
| sparse | typed | False | signed:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.946919 | 0.0121161 | 3 | 3 | 1 | 1 |

## learned

| domain | mode | selected | config | count | corruption | fraction | train→eval | split | metric | mean | pop. SD | seeds | graphs | precision | recall |
|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0724573 | 0.0439823 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.504196 | 0.0317141 | 3 | 3 | 1 | 1 |
| robot | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.893345 | 0.0491967 | 3 | 3 | 1 | 1 |
| robot | learned | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.319522 | 0.273853 | 3 | 3 | 1 | 1 |
| robot | learned | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.528116 | 0.0389506 | 3 | 3 | 1 | 1 |
| robot | learned | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.11572 | 0.253159 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0998471 | 0.0348794 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.560035 | 0.0332414 | 3 | 3 | 1 | 1 |
| robot | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.942964 | 0.0576349 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0746878 | 0.0506047 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.505028 | 0.032435 | 3 | 3 | 1 | 1 |
| robot | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.900719 | 0.0633136 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0378409 | 0.00479414 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.59399 | 0.0082201 | 3 | 3 | 1 | 1 |
| sparse | hard | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.946492 | 0.0515629 | 3 | 3 | 1 | 1 |
| sparse | learned | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.174351 | 0.174623 | 3 | 3 | 1 | 1 |
| sparse | learned | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.61069 | 0.011127 | 3 | 3 | 1 | 1 |
| sparse | learned | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 1.07598 | 0.251661 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0473734 | 0.0257203 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.626624 | 0.00855919 | 3 | 3 | 1 | 1 |
| sparse | none | False | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.958348 | 0.0642423 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0379028 | 0.00908875 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.594014 | 0.00806895 | 3 | 3 | 1 | 1 |
| sparse | soft4 | True | uniform:fixed:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.948364 | 0.0578793 | 3 | 3 | 1 | 1 |

## transfer

| domain | mode | selected | config | count | corruption | fraction | train→eval | split | metric | mean | pop. SD | seeds | graphs | precision | recall |
|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0530865 | 0.0124889 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.541353 | 0.00668095 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.840605 | 0.0239342 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0677211 | 0.0173937 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.538216 | 0.006841 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.847992 | 0.033023 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.064504 | 0.01543 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.539295 | 0.00492389 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.849043 | 0.0125985 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0663807 | 0.0184182 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.53923 | 0.00490147 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.845911 | 0.0223628 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.105882 | 0.0699156 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.544466 | 0.00363885 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.899854 | 0.0707535 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0748586 | 0.0364971 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.539781 | 0.00471101 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.852374 | 0.0299181 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0820315 | 0.0403224 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.540873 | 0.00737335 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.870114 | 0.0257315 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0771166 | 0.0375023 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.54044 | 0.00641213 | 3 | 24 | 1 | 1 |
| sparse | graph_input | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.855415 | 0.0342224 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0277124 | 0.0204214 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.541158 | 0.00618111 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.837066 | 0.0194895 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0237639 | 0.018421 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.537152 | 0.00704732 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.807361 | 0.02057 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0258812 | 0.0184032 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.538527 | 0.00527804 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.808398 | 0.0175224 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0264516 | 0.0219635 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.538264 | 0.00526217 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.80805 | 0.0185244 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0796275 | 0.0636283 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.544962 | 0.00293701 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.88312 | 0.0582464 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.071691 | 0.0666374 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.540254 | 0.00212521 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.850916 | 0.0606389 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0705048 | 0.0594667 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.541521 | 0.00551762 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.858731 | 0.063009 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0698388 | 0.0633006 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.540831 | 0.00342168 | 3 | 24 | 1 | 1 |
| sparse | hard | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.850079 | 0.045133 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0613657 | 0.049777 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.572002 | 0.00497669 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.858725 | 0.0479382 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0693791 | 0.0560684 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.583417 | 0.00700436 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.85526 | 0.0558148 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0642874 | 0.0535238 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.574654 | 0.00661576 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.844914 | 0.0483799 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0707075 | 0.0600547 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.579792 | 0.0045083 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.851558 | 0.058189 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0806402 | 0.0413474 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.573559 | 0.00692969 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.870409 | 0.0460766 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0819111 | 0.0354684 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.584621 | 0.00702395 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.859662 | 0.0276994 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0760407 | 0.0323399 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.575679 | 0.00877808 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.864046 | 0.019433 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0808777 | 0.0353015 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.581085 | 0.00824178 | 3 | 24 | 1 | 1 |
| sparse | none | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.859127 | 0.0249608 | 3 | 24 | 1 | 1 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0419854 | 0.0090994 | 3 | 24 | 0.266887 | 0.266887 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.579115 | 0.00634003 | 3 | 24 | 0.266887 | 0.266887 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.837373 | 0.0127765 | 3 | 24 | 0.266887 | 0.266887 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0303947 | 0.00500608 | 3 | 24 | 0.0234789 | 0.0234789 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.583325 | 0.00662274 | 3 | 24 | 0.0234789 | 0.0234789 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.812482 | 0.0100572 | 3 | 24 | 0.0234789 | 0.0234789 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.034192 | 0.00895268 | 3 | 24 | 0.0988459 | 0.0988459 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.577967 | 0.00611388 | 3 | 24 | 0.0988459 | 0.0988459 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.819944 | 0.00107307 | 3 | 24 | 0.0988459 | 0.0988459 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0317557 | 0.00737189 | 3 | 24 | 0.0443512 | 0.0443512 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.580964 | 0.0046277 | 3 | 24 | 0.0443512 | 0.0443512 |
| sparse | permuted4 | False | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.813506 | 0.012408 | 3 | 24 | 0.0443512 | 0.0443512 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.107838 | 0.0463782 | 3 | 24 | 0.266887 | 0.266887 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.581533 | 0.00547316 | 3 | 24 | 0.266887 | 0.266887 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.895583 | 0.0623697 | 3 | 24 | 0.266887 | 0.266887 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0932068 | 0.0232587 | 3 | 24 | 0.0234789 | 0.0234789 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.585052 | 0.00615499 | 3 | 24 | 0.0234789 | 0.0234789 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.868672 | 0.0201247 | 3 | 24 | 0.0234789 | 0.0234789 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0944637 | 0.0267231 | 3 | 24 | 0.0988459 | 0.0988459 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.58016 | 0.00752526 | 3 | 24 | 0.0988459 | 0.0988459 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.889177 | 0.0298472 | 3 | 24 | 0.0988459 | 0.0988459 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0919274 | 0.0229577 | 3 | 24 | 0.0443512 | 0.0443512 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.583184 | 0.00722386 | 3 | 24 | 0.0443512 | 0.0443512 |
| sparse | permuted4 | False | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.871205 | 0.00779843 | 3 | 24 | 0.0443512 | 0.0443512 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0334465 | 0.0300995 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.540518 | 0.0059362 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.842636 | 0.0311555 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0321961 | 0.0242996 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.546471 | 0.00627374 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.816536 | 0.0252527 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0267973 | 0.0250184 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.536505 | 0.0051917 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.808213 | 0.0220091 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0296101 | 0.0270865 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.539037 | 0.00483259 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=12:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.811132 | 0.0246967 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | deterministic_rollout_normalized_mse | 0.0800971 | 0.071858 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | one_step_normalized_mse | 0.543881 | 0.0034792 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→12 | test | stochastic_rollout_normalized_mse | 0.881788 | 0.0700943 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | deterministic_rollout_normalized_mse | 0.0761479 | 0.0681511 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | one_step_normalized_mse | 0.548011 | 0.00810795 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→128 | test | stochastic_rollout_normalized_mse | 0.854625 | 0.0618638 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | deterministic_rollout_normalized_mse | 0.0651831 | 0.061478 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | one_step_normalized_mse | 0.538845 | 0.00788439 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→32 | test | stochastic_rollout_normalized_mse | 0.853295 | 0.0633973 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | deterministic_rollout_normalized_mse | 0.0705132 | 0.066319 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | one_step_normalized_mse | 0.540877 | 0.00721017 | 3 | 24 | 1 | 1 |
| sparse | soft4 | True | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | clean | 0 | —→64 | test | stochastic_rollout_normalized_mse | 0.850419 | 0.0476772 | 3 | 24 | 1 | 1 |

## Efficiency thresholds and AUC by seed

| suite | domain | mode | seed | train count | AUC | s10 | s25 | s50 | n10 | n25 | n50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| efficiency | robot | hard | 0 | 8 | 0.508083 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 0 | 16 | 0.491169 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 0 | 32 | 0.484575 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 0 | 64 | 0.489413 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 0 | 128 | 0.486437 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 0 | 256 | 0.484924 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 1 | 8 | 0.535387 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 1 | 16 | 0.502594 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 1 | 32 | 0.497192 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 1 | 64 | 0.495013 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 1 | 128 | 0.495115 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 1 | 256 | 0.493739 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 2 | 8 | 0.608915 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 2 | 16 | 0.556405 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 2 | 32 | 0.556605 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 2 | 64 | 0.557211 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 2 | 128 | 0.555444 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | hard | 2 | 256 | 0.557186 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | none | 0 | 8 | 0.552951 | — | 25 | 25 | 32 | 8 | 8 |
| efficiency | robot | none | 0 | 16 | 0.535692 | — | 25 | 25 | 32 | 8 | 8 |
| efficiency | robot | none | 0 | 32 | 0.530011 | 600 | 25 | 25 | 32 | 8 | 8 |
| efficiency | robot | none | 0 | 64 | 0.531051 | 200 | 25 | 25 | 32 | 8 | 8 |
| efficiency | robot | none | 0 | 128 | 0.528941 | 300 | 25 | 25 | 32 | 8 | 8 |
| efficiency | robot | none | 0 | 256 | 0.527881 | 300 | 25 | 25 | 32 | 8 | 8 |
| efficiency | robot | none | 1 | 8 | 0.554821 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 1 | 16 | 0.54125 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 1 | 32 | 0.541694 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 1 | 64 | 0.540669 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 1 | 128 | 0.539778 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 1 | 256 | 0.538602 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 2 | 8 | 0.648745 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 2 | 16 | 0.624148 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 2 | 32 | 0.620941 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 2 | 64 | 0.619079 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 2 | 128 | 0.615204 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | none | 2 | 256 | 0.616524 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 0 | 8 | 0.554385 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 0 | 16 | 0.538368 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 0 | 32 | 0.532128 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 0 | 64 | 0.533795 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 0 | 128 | 0.531039 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 0 | 256 | 0.530668 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 1 | 8 | 0.561865 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 1 | 16 | 0.541593 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 1 | 32 | 0.542634 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 1 | 64 | 0.541372 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 1 | 128 | 0.54043 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 1 | 256 | 0.53933 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 2 | 8 | 0.664354 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 2 | 16 | 0.62091 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 2 | 32 | 0.615928 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 2 | 64 | 0.616743 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 2 | 128 | 0.612827 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted1 | 2 | 256 | 0.613714 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 0 | 8 | 0.574535 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 0 | 16 | 0.547767 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 0 | 32 | 0.53776 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 0 | 64 | 0.539933 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 0 | 128 | 0.537719 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 0 | 256 | 0.538771 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 1 | 8 | 0.583562 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 1 | 16 | 0.553264 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 1 | 32 | 0.546802 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 1 | 64 | 0.542595 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 1 | 128 | 0.541839 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 1 | 256 | 0.541477 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 2 | 8 | 0.70398 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 2 | 16 | 0.629317 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 2 | 32 | 0.619164 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 2 | 64 | 0.620521 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 2 | 128 | 0.617715 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | permuted4 | 2 | 256 | 0.61836 | — | 25 | 25 | — | 8 | 8 |
| efficiency | robot | soft1 | 0 | 8 | 0.522835 | 200 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 0 | 16 | 0.504241 | 100 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 0 | 32 | 0.498999 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 0 | 64 | 0.498935 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 0 | 128 | 0.498477 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 0 | 256 | 0.494458 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 1 | 8 | 0.52895 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 1 | 16 | 0.511354 | 100 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 1 | 32 | 0.513235 | 100 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 1 | 64 | 0.512341 | 100 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 1 | 128 | 0.509494 | 200 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 1 | 256 | 0.507026 | 100 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 2 | 8 | 0.609268 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 2 | 16 | 0.574725 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 2 | 32 | 0.574704 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 2 | 64 | 0.572302 | 100 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 2 | 128 | 0.568247 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft1 | 2 | 256 | 0.576456 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 0 | 8 | 0.507454 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 0 | 16 | 0.490951 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 0 | 32 | 0.484908 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 0 | 64 | 0.488901 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 0 | 128 | 0.486685 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 0 | 256 | 0.484701 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 1 | 8 | 0.524387 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 1 | 16 | 0.501213 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 1 | 32 | 0.497476 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 1 | 64 | 0.495262 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 1 | 128 | 0.495342 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 1 | 256 | 0.493682 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 2 | 8 | 0.603574 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 2 | 16 | 0.557091 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 2 | 32 | 0.557689 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 2 | 64 | 0.558094 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 2 | 128 | 0.555811 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | robot | soft4 | 2 | 256 | 0.558108 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 0 | 8 | 0.669469 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 0 | 16 | 0.621678 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 0 | 32 | 0.613427 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 0 | 64 | 0.611484 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 0 | 128 | 0.612309 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 0 | 256 | 0.610969 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 1 | 8 | 0.640536 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 1 | 16 | 0.585771 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 1 | 32 | 0.581901 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 1 | 64 | 0.581524 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 1 | 128 | 0.581197 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 1 | 256 | 0.578157 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 2 | 8 | 0.674675 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 2 | 16 | 0.614606 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 2 | 32 | 0.606014 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 2 | 64 | 0.602642 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 2 | 128 | 0.600896 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | hard | 2 | 256 | 0.605698 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | none | 0 | 8 | 0.655777 | — | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 0 | 16 | 0.641793 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 0 | 32 | 0.634169 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 0 | 64 | 0.633694 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 0 | 128 | 0.636285 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 0 | 256 | 0.636211 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 1 | 8 | 0.649013 | — | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 1 | 16 | 0.623802 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 1 | 32 | 0.622068 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 1 | 64 | 0.621502 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 1 | 128 | 0.624037 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 1 | 256 | 0.621768 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 2 | 8 | 0.670448 | — | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 2 | 16 | 0.638257 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 2 | 32 | 0.636494 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 2 | 64 | 0.631296 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 2 | 128 | 0.629409 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | none | 2 | 256 | 0.633633 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 0 | 8 | 0.663343 | — | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 0 | 16 | 0.642745 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 0 | 32 | 0.635276 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 0 | 64 | 0.635056 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 0 | 128 | 0.637073 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 0 | 256 | 0.636245 | 100 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 1 | 8 | 0.650477 | — | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 1 | 16 | 0.623513 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 1 | 32 | 0.620154 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 1 | 64 | 0.620137 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 1 | 128 | 0.622088 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 1 | 256 | 0.619877 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency | sparse | permuted1 | 2 | 8 | 0.677173 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted1 | 2 | 16 | 0.642767 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted1 | 2 | 32 | 0.64024 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted1 | 2 | 64 | 0.634701 | 200 | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted1 | 2 | 128 | 0.632799 | 200 | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted1 | 2 | 256 | 0.636628 | 200 | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted4 | 0 | 8 | 0.699177 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 0 | 16 | 0.656694 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 0 | 32 | 0.645584 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 0 | 64 | 0.645342 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 0 | 128 | 0.644914 | 600 | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 0 | 256 | 0.641602 | 300 | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 1 | 8 | 0.687348 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted4 | 1 | 16 | 0.635694 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted4 | 1 | 32 | 0.626131 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted4 | 1 | 64 | 0.625345 | 600 | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted4 | 1 | 128 | 0.627851 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted4 | 1 | 256 | 0.626383 | — | 25 | 25 | 64 | 8 | 8 |
| efficiency | sparse | permuted4 | 2 | 8 | 0.716972 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 2 | 16 | 0.649645 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 2 | 32 | 0.646852 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 2 | 64 | 0.640619 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 2 | 128 | 0.638961 | 600 | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | permuted4 | 2 | 256 | 0.641476 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency | sparse | soft1 | 0 | 8 | 0.651407 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 0 | 16 | 0.627618 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 0 | 32 | 0.615737 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 0 | 64 | 0.617217 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 0 | 128 | 0.617813 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 0 | 256 | 0.612918 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 1 | 8 | 0.626681 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 1 | 16 | 0.597298 | 100 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 1 | 32 | 0.595597 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 1 | 64 | 0.596095 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 1 | 128 | 0.597033 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 1 | 256 | 0.594163 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 2 | 8 | 0.645747 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 2 | 16 | 0.617764 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 2 | 32 | 0.616468 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 2 | 64 | 0.610944 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 2 | 128 | 0.608377 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft1 | 2 | 256 | 0.613116 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 0 | 8 | 0.660453 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 0 | 16 | 0.620444 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 0 | 32 | 0.612327 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 0 | 64 | 0.610332 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 0 | 128 | 0.61174 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 0 | 256 | 0.611536 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 1 | 8 | 0.637323 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 1 | 16 | 0.586248 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 1 | 32 | 0.582114 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 1 | 64 | 0.581992 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 1 | 128 | 0.582033 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 1 | 256 | 0.578816 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 2 | 8 | 0.667709 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 2 | 16 | 0.612647 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 2 | 32 | 0.606108 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 2 | 64 | 0.602781 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 2 | 128 | 0.601186 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency | sparse | soft4 | 2 | 256 | 0.605876 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 0 | 8 | 0.517304 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 0 | 16 | 0.491697 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 0 | 32 | 0.484577 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 0 | 64 | 0.48941 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 0 | 128 | 0.486462 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 0 | 256 | 0.484531 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 1 | 8 | 0.543635 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 1 | 16 | 0.503667 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 1 | 32 | 0.496859 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 1 | 64 | 0.4944 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 1 | 128 | 0.494557 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 1 | 256 | 0.492984 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 2 | 8 | 0.612391 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 2 | 16 | 0.56 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 2 | 32 | 0.5578 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 2 | 64 | 0.55687 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 2 | 128 | 0.554929 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | hard | 2 | 256 | 0.556216 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | none | 0 | 8 | 0.567226 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency_identity | robot | none | 0 | 16 | 0.537913 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency_identity | robot | none | 0 | 32 | 0.530839 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency_identity | robot | none | 0 | 64 | 0.531689 | — | 25 | 25 | 128 | 8 | 8 |
| efficiency_identity | robot | none | 0 | 128 | 0.529563 | 300 | 25 | 25 | 128 | 8 | 8 |
| efficiency_identity | robot | none | 0 | 256 | 0.527898 | 300 | 25 | 25 | 128 | 8 | 8 |
| efficiency_identity | robot | none | 1 | 8 | 0.563368 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 1 | 16 | 0.542027 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 1 | 32 | 0.541959 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 1 | 64 | 0.540642 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 1 | 128 | 0.539504 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 1 | 256 | 0.538299 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 2 | 8 | 0.674528 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 2 | 16 | 0.627056 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 2 | 32 | 0.620884 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 2 | 64 | 0.618673 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 2 | 128 | 0.615311 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | none | 2 | 256 | 0.616403 | — | 25 | 25 | — | 8 | 8 |
| efficiency_identity | robot | soft4 | 0 | 8 | 0.512036 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 0 | 16 | 0.491864 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 0 | 32 | 0.485182 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 0 | 64 | 0.489129 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 0 | 128 | 0.486997 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 0 | 256 | 0.484556 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 1 | 8 | 0.536837 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 1 | 16 | 0.502502 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 1 | 32 | 0.49731 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 1 | 64 | 0.494843 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 1 | 128 | 0.494904 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 1 | 256 | 0.493156 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 2 | 8 | 0.622832 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 2 | 16 | 0.560053 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 2 | 32 | 0.5588 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 2 | 64 | 0.557792 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 2 | 128 | 0.555271 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | robot | soft4 | 2 | 256 | 0.557294 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 0 | 8 | 0.687776 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 0 | 16 | 0.622537 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 0 | 32 | 0.611115 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 0 | 64 | 0.60898 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 0 | 128 | 0.610025 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 0 | 256 | 0.608826 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 1 | 8 | 0.661452 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 1 | 16 | 0.58811 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 1 | 32 | 0.582051 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 1 | 64 | 0.581513 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 1 | 128 | 0.580793 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 1 | 256 | 0.577601 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 2 | 8 | 0.686526 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 2 | 16 | 0.617958 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 2 | 32 | 0.607177 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 2 | 64 | 0.602889 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 2 | 128 | 0.601123 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | hard | 2 | 256 | 0.605328 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | none | 0 | 8 | 0.667743 | — | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 0 | 16 | 0.644263 | — | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 0 | 32 | 0.634832 | 100 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 0 | 64 | 0.633722 | 200 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 0 | 128 | 0.63633 | 300 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 0 | 256 | 0.636167 | 100 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 1 | 8 | 0.662455 | — | 25 | 25 | 16 | 8 | 8 |
| efficiency_identity | sparse | none | 1 | 16 | 0.624713 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency_identity | sparse | none | 1 | 32 | 0.621793 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency_identity | sparse | none | 1 | 64 | 0.620773 | 300 | 25 | 25 | 16 | 8 | 8 |
| efficiency_identity | sparse | none | 1 | 128 | 0.623359 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency_identity | sparse | none | 1 | 256 | 0.620768 | 200 | 25 | 25 | 16 | 8 | 8 |
| efficiency_identity | sparse | none | 2 | 8 | 0.692866 | — | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 2 | 16 | 0.64202 | — | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 2 | 32 | 0.637472 | 100 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 2 | 64 | 0.631682 | 200 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 2 | 128 | 0.629882 | 50 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | none | 2 | 256 | 0.633518 | 100 | 25 | 25 | 32 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 0 | 8 | 0.684326 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 0 | 16 | 0.619903 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 0 | 32 | 0.610505 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 0 | 64 | 0.608244 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 0 | 128 | 0.60959 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 0 | 256 | 0.609366 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 1 | 8 | 0.653623 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 1 | 16 | 0.588795 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 1 | 32 | 0.582318 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 1 | 64 | 0.582017 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 1 | 128 | 0.581736 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 1 | 256 | 0.578355 | 50 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 2 | 8 | 0.675401 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 2 | 16 | 0.615298 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 2 | 32 | 0.607281 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 2 | 64 | 0.60309 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 2 | 128 | 0.601415 | 25 | 25 | 25 | 8 | 8 | 8 |
| efficiency_identity | sparse | soft4 | 2 | 256 | 0.605522 | 25 | 25 | 25 | 8 | 8 | 8 |

## Learned signed coefficients

| suite | domain | mode | seed | step | coefficients | min | mean | max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| heterogeneous | sparse | typed | 0 | 0 | 16 | 0 | 0 | 0 |
| heterogeneous | sparse | typed | 0 | 25 | 16 | -0.0223723 | 0.00108849 | 0.0225888 |
| heterogeneous | sparse | typed | 0 | 50 | 16 | -0.0403043 | 0.00131924 | 0.0422596 |
| heterogeneous | sparse | typed | 0 | 100 | 16 | -0.0684014 | 0.00368665 | 0.0803013 |
| heterogeneous | sparse | typed | 0 | 200 | 16 | -0.13714 | 0.0073333 | 0.144226 |
| heterogeneous | sparse | typed | 0 | 300 | 16 | -0.229245 | 0.0116588 | 0.23896 |
| heterogeneous | sparse | typed | 0 | 600 | 16 | -0.697372 | 0.0327484 | 0.714279 |
| heterogeneous | sparse | typed | 1 | 0 | 16 | 0 | 0 | 0 |
| heterogeneous | sparse | typed | 1 | 25 | 16 | -0.0247348 | -0.000658693 | 0.0240809 |
| heterogeneous | sparse | typed | 1 | 50 | 16 | -0.0479715 | -0.000413133 | 0.0473146 |
| heterogeneous | sparse | typed | 1 | 100 | 16 | -0.0906273 | 0.00179958 | 0.0924915 |
| heterogeneous | sparse | typed | 1 | 200 | 16 | -0.185338 | 0.00720824 | 0.197703 |
| heterogeneous | sparse | typed | 1 | 300 | 16 | -0.309811 | 0.0153242 | 0.338128 |
| heterogeneous | sparse | typed | 1 | 600 | 16 | -0.867656 | 0.0483753 | 0.951701 |
| heterogeneous | sparse | typed | 2 | 0 | 16 | 0 | 0 | 0 |
| heterogeneous | sparse | typed | 2 | 25 | 16 | -0.0221453 | -0.000142406 | 0.0242927 |
| heterogeneous | sparse | typed | 2 | 50 | 16 | -0.0412752 | 0.00081437 | 0.0485733 |
| heterogeneous | sparse | typed | 2 | 100 | 16 | -0.0839805 | 0.00158174 | 0.0981462 |
| heterogeneous | sparse | typed | 2 | 200 | 16 | -0.187378 | 0.00696736 | 0.217978 |
| heterogeneous | sparse | typed | 2 | 300 | 16 | -0.293996 | 0.0122058 | 0.33878 |
| heterogeneous | sparse | typed | 2 | 600 | 16 | -0.798616 | 0.0378338 | 0.874186 |
| heterogeneous | robot | typed | 0 | 0 | 16 | 0 | 0 | 0 |
| heterogeneous | robot | typed | 0 | 25 | 16 | -0.0239924 | 0.00151122 | 0.0227629 |
| heterogeneous | robot | typed | 0 | 50 | 16 | -0.0430531 | 0.00339789 | 0.0419051 |
| heterogeneous | robot | typed | 0 | 100 | 16 | -0.0762073 | 0.0092221 | 0.0946418 |
| heterogeneous | robot | typed | 0 | 200 | 16 | -0.193771 | 0.0234825 | 0.221852 |
| heterogeneous | robot | typed | 0 | 300 | 16 | -0.329404 | 0.0376642 | 0.366337 |
| heterogeneous | robot | typed | 0 | 600 | 16 | -0.899228 | 0.0731543 | 0.973836 |
| heterogeneous | robot | typed | 1 | 0 | 16 | 0 | 0 | 0 |
| heterogeneous | robot | typed | 1 | 25 | 16 | -0.0228944 | 0.00040457 | 0.0236007 |
| heterogeneous | robot | typed | 1 | 50 | 16 | -0.0481176 | 0.00198409 | 0.0468097 |
| heterogeneous | robot | typed | 1 | 100 | 16 | -0.0909873 | 0.00558056 | 0.095019 |
| heterogeneous | robot | typed | 1 | 200 | 16 | -0.180722 | 0.0125848 | 0.214656 |
| heterogeneous | robot | typed | 1 | 300 | 16 | -0.318865 | 0.0219712 | 0.374718 |
| heterogeneous | robot | typed | 1 | 600 | 16 | -0.827813 | 0.0454156 | 0.936118 |
| heterogeneous | robot | typed | 2 | 0 | 16 | 0 | 0 | 0 |
| heterogeneous | robot | typed | 2 | 25 | 16 | -0.0239861 | 0.00032258 | 0.0244783 |
| heterogeneous | robot | typed | 2 | 50 | 16 | -0.0449131 | 0.00155655 | 0.0503521 |
| heterogeneous | robot | typed | 2 | 100 | 16 | -0.0926806 | 0.00440714 | 0.104421 |
| heterogeneous | robot | typed | 2 | 200 | 16 | -0.216818 | 0.0131935 | 0.25154 |
| heterogeneous | robot | typed | 2 | 300 | 16 | -0.348028 | 0.0222985 | 0.407324 |
| heterogeneous | robot | typed | 2 | 600 | 16 | -0.900353 | 0.046566 | 0.934067 |
| learned | sparse | learned | 0 | 0 | 8 | 0 | 0 | 0 |
| learned | sparse | learned | 0 | 25 | 8 | -0.0157195 | 0.000405897 | 0.0227532 |
| learned | sparse | learned | 0 | 50 | 8 | -0.0189535 | 0.0113141 | 0.0459877 |
| learned | sparse | learned | 0 | 100 | 8 | -0.0190648 | 0.0390156 | 0.090019 |
| learned | sparse | learned | 0 | 200 | 8 | -0.00556685 | 0.105701 | 0.197897 |
| learned | sparse | learned | 0 | 300 | 8 | 0.00363301 | 0.173154 | 0.312926 |
| learned | sparse | learned | 0 | 600 | 8 | -0.0712627 | 0.36979 | 0.750395 |
| learned | sparse | learned | 1 | 0 | 8 | 0 | 0 | 0 |
| learned | sparse | learned | 1 | 25 | 8 | -0.00931716 | 0.010775 | 0.0240741 |
| learned | sparse | learned | 1 | 50 | 8 | -0.00744215 | 0.0261009 | 0.0501163 |
| learned | sparse | learned | 1 | 100 | 8 | -0.023088 | 0.0624258 | 0.111165 |
| learned | sparse | learned | 1 | 200 | 8 | -0.07241 | 0.132556 | 0.222879 |
| learned | sparse | learned | 1 | 300 | 8 | -0.160313 | 0.207362 | 0.358971 |
| learned | sparse | learned | 1 | 600 | 8 | -0.567913 | 0.385589 | 0.800419 |
| learned | sparse | learned | 2 | 0 | 8 | 0 | 0 | 0 |
| learned | sparse | learned | 2 | 25 | 8 | -0.00921325 | 0.0104378 | 0.0242941 |
| learned | sparse | learned | 2 | 50 | 8 | -0.0054329 | 0.0282152 | 0.0482654 |
| learned | sparse | learned | 2 | 100 | 8 | 0.0196282 | 0.070694 | 0.0961576 |
| learned | sparse | learned | 2 | 200 | 8 | 0.0694187 | 0.157283 | 0.191449 |
| learned | sparse | learned | 2 | 300 | 8 | 0.109298 | 0.25402 | 0.302083 |
| learned | sparse | learned | 2 | 600 | 8 | -0.00310744 | 0.520501 | 0.710336 |
| learned | robot | learned | 0 | 0 | 8 | 0 | 0 | 0 |
| learned | robot | learned | 0 | 25 | 8 | -0.0167714 | 0.00141713 | 0.0216631 |
| learned | robot | learned | 0 | 50 | 8 | -0.0215436 | 0.0138497 | 0.0454027 |
| learned | robot | learned | 0 | 100 | 8 | -0.0286709 | 0.0452344 | 0.095874 |
| learned | robot | learned | 0 | 200 | 8 | -0.0314929 | 0.121448 | 0.225866 |
| learned | robot | learned | 0 | 300 | 8 | -0.0504456 | 0.194133 | 0.357126 |
| learned | robot | learned | 0 | 600 | 8 | -0.293079 | 0.346585 | 0.840658 |
| learned | robot | learned | 1 | 0 | 8 | 0 | 0 | 0 |
| learned | robot | learned | 1 | 25 | 8 | -0.0148241 | 0.00816831 | 0.0210922 |
| learned | robot | learned | 1 | 50 | 8 | -0.0308836 | 0.0175066 | 0.0494385 |
| learned | robot | learned | 1 | 100 | 8 | -0.0392728 | 0.0433463 | 0.111518 |
| learned | robot | learned | 1 | 200 | 8 | -0.0956929 | 0.089285 | 0.232792 |
| learned | robot | learned | 1 | 300 | 8 | -0.205177 | 0.140892 | 0.389363 |
| learned | robot | learned | 1 | 600 | 8 | -0.571561 | 0.244925 | 0.837969 |
| learned | robot | learned | 2 | 0 | 8 | 0 | 0 | 0 |
| learned | robot | learned | 2 | 25 | 8 | -0.0158915 | 0.0094449 | 0.0248516 |
| learned | robot | learned | 2 | 50 | 8 | -0.0154117 | 0.0266837 | 0.0499627 |
| learned | robot | learned | 2 | 100 | 8 | 0.000420139 | 0.0687454 | 0.103449 |
| learned | robot | learned | 2 | 200 | 8 | 0.0123431 | 0.154679 | 0.213906 |
| learned | robot | learned | 2 | 300 | 8 | -0.0153824 | 0.235206 | 0.344962 |
| learned | robot | learned | 2 | 600 | 8 | -0.357017 | 0.39202 | 0.799268 |

## Validation-only soft-strength selection

| suite | domain | train config | selected strength |
|---|---|---|---:|
| corruption | robot | uniform:fixed:n=128:add@0.1 | 4 |
| corruption | robot | uniform:fixed:n=128:add@0.25 | 4 |
| corruption | robot | uniform:fixed:n=128:add@0.5 | 4 |
| corruption | robot | uniform:fixed:n=128:clean@0.0 | 4 |
| corruption | robot | uniform:fixed:n=128:drop@0.1 | 4 |
| corruption | robot | uniform:fixed:n=128:drop@0.25 | 4 |
| corruption | robot | uniform:fixed:n=128:drop@0.5 | 4 |
| corruption | robot | uniform:fixed:n=128:mixed@0.25 | 4 |
| corruption | sparse | uniform:fixed:n=128:add@0.1 | 4 |
| corruption | sparse | uniform:fixed:n=128:add@0.25 | 4 |
| corruption | sparse | uniform:fixed:n=128:add@0.5 | 4 |
| corruption | sparse | uniform:fixed:n=128:clean@0.0 | 4 |
| corruption | sparse | uniform:fixed:n=128:drop@0.1 | 4 |
| corruption | sparse | uniform:fixed:n=128:drop@0.25 | 4 |
| corruption | sparse | uniform:fixed:n=128:drop@0.5 | 1 |
| corruption | sparse | uniform:fixed:n=128:mixed@0.25 | 1 |
| efficiency | robot | uniform:fixed:n=128:clean@0.0 | 4 |
| efficiency | robot | uniform:fixed:n=16:clean@0.0 | 4 |
| efficiency | robot | uniform:fixed:n=256:clean@0.0 | 4 |
| efficiency | robot | uniform:fixed:n=32:clean@0.0 | 4 |
| efficiency | robot | uniform:fixed:n=64:clean@0.0 | 4 |
| efficiency | robot | uniform:fixed:n=8:clean@0.0 | 1 |
| efficiency | sparse | uniform:fixed:n=128:clean@0.0 | 4 |
| efficiency | sparse | uniform:fixed:n=16:clean@0.0 | 4 |
| efficiency | sparse | uniform:fixed:n=256:clean@0.0 | 4 |
| efficiency | sparse | uniform:fixed:n=32:clean@0.0 | 4 |
| efficiency | sparse | uniform:fixed:n=64:clean@0.0 | 4 |
| efficiency | sparse | uniform:fixed:n=8:clean@0.0 | 1 |
| efficiency_identity | robot | uniform:fixed:n=128:clean@0.0 | 4 |
| efficiency_identity | robot | uniform:fixed:n=16:clean@0.0 | 4 |
| efficiency_identity | robot | uniform:fixed:n=256:clean@0.0 | 4 |
| efficiency_identity | robot | uniform:fixed:n=32:clean@0.0 | 4 |
| efficiency_identity | robot | uniform:fixed:n=64:clean@0.0 | 4 |
| efficiency_identity | robot | uniform:fixed:n=8:clean@0.0 | 4 |
| efficiency_identity | sparse | uniform:fixed:n=128:clean@0.0 | 4 |
| efficiency_identity | sparse | uniform:fixed:n=16:clean@0.0 | 4 |
| efficiency_identity | sparse | uniform:fixed:n=256:clean@0.0 | 4 |
| efficiency_identity | sparse | uniform:fixed:n=32:clean@0.0 | 4 |
| efficiency_identity | sparse | uniform:fixed:n=64:clean@0.0 | 4 |
| efficiency_identity | sparse | uniform:fixed:n=8:clean@0.0 | 4 |
| heterogeneous | robot | signed:fixed:n=128:clean@0.0 | 4 |
| heterogeneous | sparse | signed:fixed:n=128:clean@0.0 | 4 |
| learned | robot | uniform:fixed:n=128:clean@0.0 | 4 |
| learned | sparse | uniform:fixed:n=128:clean@0.0 | 4 |
| transfer | sparse | uniform:sizes=12:n=128:clean@0.0 | 4 |
| transfer | sparse | uniform:sizes=9,12,16:n=128:clean@0.0 | 4 |

## Resource accounting

| suite | domain | mode | train config | count | mechanism | identity | parameters | optimizer examples | training seconds | peak RSS KiB |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| corruption | robot | hard | uniform:fixed:n=128:add@0.1 | 128 | uniform | False | 17089 | 19200 | 1.48314 | 362296 |
| corruption | robot | hard | uniform:fixed:n=128:add@0.25 | 128 | uniform | False | 17089 | 19200 | 1.78363 | 362296 |
| corruption | robot | hard | uniform:fixed:n=128:add@0.5 | 128 | uniform | False | 17089 | 19200 | 2.24485 | 362296 |
| corruption | robot | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.11473 | 362296 |
| corruption | robot | hard | uniform:fixed:n=128:drop@0.1 | 128 | uniform | False | 17089 | 19200 | 2.09199 | 362296 |
| corruption | robot | hard | uniform:fixed:n=128:drop@0.25 | 128 | uniform | False | 17089 | 19200 | 1.6076 | 362296 |
| corruption | robot | hard | uniform:fixed:n=128:drop@0.5 | 128 | uniform | False | 17089 | 19200 | 1.89414 | 362296 |
| corruption | robot | hard | uniform:fixed:n=128:mixed@0.25 | 128 | uniform | False | 17089 | 19200 | 2.12621 | 362296 |
| corruption | robot | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.1512 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:add@0.1 | 128 | uniform | False | 17089 | 19200 | 1.59297 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:add@0.25 | 128 | uniform | False | 17089 | 19200 | 1.44231 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:add@0.5 | 128 | uniform | False | 17089 | 19200 | 1.6837 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.70148 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:drop@0.1 | 128 | uniform | False | 17089 | 19200 | 1.97922 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:drop@0.25 | 128 | uniform | False | 17089 | 19200 | 2.03417 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:drop@0.5 | 128 | uniform | False | 17089 | 19200 | 1.72826 | 362296 |
| corruption | robot | soft1 | uniform:fixed:n=128:mixed@0.25 | 128 | uniform | False | 17089 | 19200 | 1.98373 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:add@0.1 | 128 | uniform | False | 17089 | 19200 | 1.43941 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:add@0.25 | 128 | uniform | False | 17089 | 19200 | 1.6081 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:add@0.5 | 128 | uniform | False | 17089 | 19200 | 2.27489 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.56491 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:drop@0.1 | 128 | uniform | False | 17089 | 19200 | 2.03159 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:drop@0.25 | 128 | uniform | False | 17089 | 19200 | 1.51554 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:drop@0.5 | 128 | uniform | False | 17089 | 19200 | 1.73887 | 362296 |
| corruption | robot | soft4 | uniform:fixed:n=128:mixed@0.25 | 128 | uniform | False | 17089 | 19200 | 1.97552 | 362296 |
| corruption | sparse | hard | uniform:fixed:n=128:add@0.1 | 128 | uniform | False | 17089 | 19200 | 1.59127 | 359312 |
| corruption | sparse | hard | uniform:fixed:n=128:add@0.25 | 128 | uniform | False | 17089 | 19200 | 1.50763 | 359396 |
| corruption | sparse | hard | uniform:fixed:n=128:add@0.5 | 128 | uniform | False | 17089 | 19200 | 1.84729 | 359628 |
| corruption | sparse | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.26587 | 358892 |
| corruption | sparse | hard | uniform:fixed:n=128:drop@0.1 | 128 | uniform | False | 17089 | 19200 | 1.9187 | 359012 |
| corruption | sparse | hard | uniform:fixed:n=128:drop@0.25 | 128 | uniform | False | 17089 | 19200 | 1.60753 | 359100 |
| corruption | sparse | hard | uniform:fixed:n=128:drop@0.5 | 128 | uniform | False | 17089 | 19200 | 2.22189 | 359220 |
| corruption | sparse | hard | uniform:fixed:n=128:mixed@0.25 | 128 | uniform | False | 17089 | 19200 | 1.79118 | 359728 |
| corruption | sparse | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.80925 | 358656 |
| corruption | sparse | soft1 | uniform:fixed:n=128:add@0.1 | 128 | uniform | False | 17089 | 19200 | 1.97452 | 359244 |
| corruption | sparse | soft1 | uniform:fixed:n=128:add@0.25 | 128 | uniform | False | 17089 | 19200 | 1.49748 | 359340 |
| corruption | sparse | soft1 | uniform:fixed:n=128:add@0.5 | 128 | uniform | False | 17089 | 19200 | 1.62297 | 359436 |
| corruption | sparse | soft1 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.01187 | 358728 |
| corruption | sparse | soft1 | uniform:fixed:n=128:drop@0.1 | 128 | uniform | False | 17089 | 19200 | 1.96509 | 358896 |
| corruption | sparse | soft1 | uniform:fixed:n=128:drop@0.25 | 128 | uniform | False | 17089 | 19200 | 2.08493 | 359052 |
| corruption | sparse | soft1 | uniform:fixed:n=128:drop@0.5 | 128 | uniform | False | 17089 | 19200 | 1.63651 | 359136 |
| corruption | sparse | soft1 | uniform:fixed:n=128:mixed@0.25 | 128 | uniform | False | 17089 | 19200 | 1.89759 | 359652 |
| corruption | sparse | soft4 | uniform:fixed:n=128:add@0.1 | 128 | uniform | False | 17089 | 19200 | 1.93346 | 359276 |
| corruption | sparse | soft4 | uniform:fixed:n=128:add@0.25 | 128 | uniform | False | 17089 | 19200 | 1.4441 | 359368 |
| corruption | sparse | soft4 | uniform:fixed:n=128:add@0.5 | 128 | uniform | False | 17089 | 19200 | 1.8817 | 359624 |
| corruption | sparse | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.32156 | 358812 |
| corruption | sparse | soft4 | uniform:fixed:n=128:drop@0.1 | 128 | uniform | False | 17089 | 19200 | 1.93709 | 358988 |
| corruption | sparse | soft4 | uniform:fixed:n=128:drop@0.25 | 128 | uniform | False | 17089 | 19200 | 2.03918 | 359076 |
| corruption | sparse | soft4 | uniform:fixed:n=128:drop@0.5 | 128 | uniform | False | 17089 | 19200 | 1.44246 | 359160 |
| corruption | sparse | soft4 | uniform:fixed:n=128:mixed@0.25 | 128 | uniform | False | 17089 | 19200 | 1.76566 | 359684 |
| efficiency | robot | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.00878 | 350384 |
| efficiency | robot | hard | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 2.07772 | 349880 |
| efficiency | robot | hard | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.66996 | 350636 |
| efficiency | robot | hard | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 2.04407 | 350056 |
| efficiency | robot | hard | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 2.19947 | 350216 |
| efficiency | robot | hard | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.57786 | 349708 |
| efficiency | robot | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.52106 | 350308 |
| efficiency | robot | none | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.61613 | 349784 |
| efficiency | robot | none | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.6467 | 350496 |
| efficiency | robot | none | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 1.65943 | 349976 |
| efficiency | robot | none | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.61057 | 350172 |
| efficiency | robot | none | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.41924 | 349592 |
| efficiency | robot | permuted1 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.92151 | 350424 |
| efficiency | robot | permuted1 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.76655 | 349908 |
| efficiency | robot | permuted1 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.62221 | 350664 |
| efficiency | robot | permuted1 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 2.13027 | 350116 |
| efficiency | robot | permuted1 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.81778 | 350236 |
| efficiency | robot | permuted1 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.4637 | 349732 |
| efficiency | robot | permuted4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.891 | 350452 |
| efficiency | robot | permuted4 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.4046 | 349952 |
| efficiency | robot | permuted4 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.58392 | 350688 |
| efficiency | robot | permuted4 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 1.65513 | 350140 |
| efficiency | robot | permuted4 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.69866 | 350280 |
| efficiency | robot | permuted4 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.45966 | 349760 |
| efficiency | robot | soft1 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.56497 | 350332 |
| efficiency | robot | soft1 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.55674 | 349812 |
| efficiency | robot | soft1 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.70636 | 350556 |
| efficiency | robot | soft1 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 1.91664 | 350000 |
| efficiency | robot | soft1 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.66513 | 350172 |
| efficiency | robot | soft1 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.43942 | 349616 |
| efficiency | robot | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.68136 | 350360 |
| efficiency | robot | soft4 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.80536 | 349840 |
| efficiency | robot | soft4 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.66153 | 350612 |
| efficiency | robot | soft4 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 1.76145 | 350024 |
| efficiency | robot | soft4 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.92369 | 350196 |
| efficiency | robot | soft4 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.44504 | 349676 |
| efficiency | sparse | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.85468 | 346360 |
| efficiency | sparse | hard | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.84299 | 345832 |
| efficiency | sparse | hard | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.50559 | 346360 |
| efficiency | sparse | hard | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 2.23561 | 346020 |
| efficiency | sparse | hard | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.5789 | 346360 |
| efficiency | sparse | hard | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.71063 | 345584 |
| efficiency | sparse | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.70675 | 346360 |
| efficiency | sparse | none | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.33732 | 345696 |
| efficiency | sparse | none | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.62356 | 346360 |
| efficiency | sparse | none | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 1.56288 | 345924 |
| efficiency | sparse | none | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 2.22981 | 346232 |
| efficiency | sparse | none | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.59289 | 345508 |
| efficiency | sparse | permuted1 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.79546 | 346360 |
| efficiency | sparse | permuted1 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.67067 | 345856 |
| efficiency | sparse | permuted1 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.39669 | 346360 |
| efficiency | sparse | permuted1 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 2.04655 | 346040 |
| efficiency | sparse | permuted1 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.44232 | 346360 |
| efficiency | sparse | permuted1 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.61989 | 345628 |
| efficiency | sparse | permuted4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.00426 | 346360 |
| efficiency | sparse | permuted4 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.76767 | 345896 |
| efficiency | sparse | permuted4 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.44493 | 346360 |
| efficiency | sparse | permuted4 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 2.51057 | 346040 |
| efficiency | sparse | permuted4 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.6208 | 346360 |
| efficiency | sparse | permuted4 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.53418 | 345656 |
| efficiency | sparse | soft1 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.63589 | 346360 |
| efficiency | sparse | soft1 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.40698 | 345780 |
| efficiency | sparse | soft1 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.82077 | 346360 |
| efficiency | sparse | soft1 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 1.52813 | 345948 |
| efficiency | sparse | soft1 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 2.01858 | 346280 |
| efficiency | sparse | soft1 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.67722 | 345536 |
| efficiency | sparse | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.60675 | 346360 |
| efficiency | sparse | soft4 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | False | 17089 | 19200 | 1.43204 | 345808 |
| efficiency | sparse | soft4 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | False | 17089 | 19200 | 1.79386 | 346360 |
| efficiency | sparse | soft4 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | False | 17089 | 19200 | 1.56412 | 345976 |
| efficiency | sparse | soft4 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | False | 17089 | 19200 | 1.61561 | 346332 |
| efficiency | sparse | soft4 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | False | 17089 | 19200 | 1.62028 | 345556 |
| efficiency_identity | robot | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | True | 17473 | 19200 | 1.67057 | 356600 |
| efficiency_identity | robot | hard | uniform:fixed:n=16:clean@0.0 | 16 | uniform | True | 17473 | 19200 | 2.00721 | 356336 |
| efficiency_identity | robot | hard | uniform:fixed:n=256:clean@0.0 | 256 | uniform | True | 17473 | 19200 | 1.87143 | 356772 |
| efficiency_identity | robot | hard | uniform:fixed:n=32:clean@0.0 | 32 | uniform | True | 17473 | 19200 | 2.29157 | 356428 |
| efficiency_identity | robot | hard | uniform:fixed:n=64:clean@0.0 | 64 | uniform | True | 17473 | 19200 | 1.68154 | 356528 |
| efficiency_identity | robot | hard | uniform:fixed:n=8:clean@0.0 | 8 | uniform | True | 17473 | 19200 | 1.81383 | 356236 |
| efficiency_identity | robot | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | True | 17473 | 19200 | 1.42863 | 356552 |
| efficiency_identity | robot | none | uniform:fixed:n=16:clean@0.0 | 16 | uniform | True | 17473 | 19200 | 1.36054 | 356280 |
| efficiency_identity | robot | none | uniform:fixed:n=256:clean@0.0 | 256 | uniform | True | 17473 | 19200 | 1.43327 | 356648 |
| efficiency_identity | robot | none | uniform:fixed:n=32:clean@0.0 | 32 | uniform | True | 17473 | 19200 | 2.03527 | 356376 |
| efficiency_identity | robot | none | uniform:fixed:n=64:clean@0.0 | 64 | uniform | True | 17473 | 19200 | 1.61954 | 356452 |
| efficiency_identity | robot | none | uniform:fixed:n=8:clean@0.0 | 8 | uniform | True | 17473 | 19200 | 1.73826 | 356168 |
| efficiency_identity | robot | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | True | 17473 | 19200 | 1.70445 | 356576 |
| efficiency_identity | robot | soft4 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | True | 17473 | 19200 | 1.61497 | 356304 |
| efficiency_identity | robot | soft4 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | True | 17473 | 19200 | 1.46969 | 356728 |
| efficiency_identity | robot | soft4 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | True | 17473 | 19200 | 2.09473 | 356404 |
| efficiency_identity | robot | soft4 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | True | 17473 | 19200 | 1.46879 | 356496 |
| efficiency_identity | robot | soft4 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | True | 17473 | 19200 | 1.84899 | 356208 |
| efficiency_identity | sparse | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | True | 17473 | 19200 | 1.63462 | 354848 |
| efficiency_identity | sparse | hard | uniform:fixed:n=16:clean@0.0 | 16 | uniform | True | 17473 | 19200 | 1.91817 | 354456 |
| efficiency_identity | sparse | hard | uniform:fixed:n=256:clean@0.0 | 256 | uniform | True | 17473 | 19200 | 1.78407 | 354984 |
| efficiency_identity | sparse | hard | uniform:fixed:n=32:clean@0.0 | 32 | uniform | True | 17473 | 19200 | 1.93068 | 354584 |
| efficiency_identity | sparse | hard | uniform:fixed:n=64:clean@0.0 | 64 | uniform | True | 17473 | 19200 | 1.85486 | 354664 |
| efficiency_identity | sparse | hard | uniform:fixed:n=8:clean@0.0 | 8 | uniform | True | 17473 | 19200 | 1.95905 | 354376 |
| efficiency_identity | sparse | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | True | 17473 | 19200 | 1.52909 | 354800 |
| efficiency_identity | sparse | none | uniform:fixed:n=16:clean@0.0 | 16 | uniform | True | 17473 | 19200 | 1.51375 | 354404 |
| efficiency_identity | sparse | none | uniform:fixed:n=256:clean@0.0 | 256 | uniform | True | 17473 | 19200 | 1.43368 | 354884 |
| efficiency_identity | sparse | none | uniform:fixed:n=32:clean@0.0 | 32 | uniform | True | 17473 | 19200 | 1.46387 | 354484 |
| efficiency_identity | sparse | none | uniform:fixed:n=64:clean@0.0 | 64 | uniform | True | 17473 | 19200 | 1.40926 | 354612 |
| efficiency_identity | sparse | none | uniform:fixed:n=8:clean@0.0 | 8 | uniform | True | 17473 | 19200 | 1.4686 | 354336 |
| efficiency_identity | sparse | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | True | 17473 | 19200 | 1.48694 | 354824 |
| efficiency_identity | sparse | soft4 | uniform:fixed:n=16:clean@0.0 | 16 | uniform | True | 17473 | 19200 | 1.74992 | 354428 |
| efficiency_identity | sparse | soft4 | uniform:fixed:n=256:clean@0.0 | 256 | uniform | True | 17473 | 19200 | 1.61805 | 354912 |
| efficiency_identity | sparse | soft4 | uniform:fixed:n=32:clean@0.0 | 32 | uniform | True | 17473 | 19200 | 1.70621 | 354524 |
| efficiency_identity | sparse | soft4 | uniform:fixed:n=64:clean@0.0 | 64 | uniform | True | 17473 | 19200 | 1.49005 | 354640 |
| efficiency_identity | sparse | soft4 | uniform:fixed:n=8:clean@0.0 | 8 | uniform | True | 17473 | 19200 | 1.62539 | 354352 |
| heterogeneous | robot | graph_input | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17217 | 19200 | 1.44387 | 473988 |
| heterogeneous | robot | hard | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.93128 | 473988 |
| heterogeneous | robot | none | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.51801 | 473988 |
| heterogeneous | robot | permuted4 | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.67566 | 473988 |
| heterogeneous | robot | soft1 | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.73765 | 473988 |
| heterogeneous | robot | soft4 | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.62603 | 473988 |
| heterogeneous | robot | typed | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17105 | 19200 | 1.86507 | 473988 |
| heterogeneous | sparse | graph_input | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17217 | 19200 | 1.48909 | 473988 |
| heterogeneous | sparse | hard | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.90615 | 473988 |
| heterogeneous | sparse | none | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.82643 | 473988 |
| heterogeneous | sparse | permuted4 | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.73136 | 473988 |
| heterogeneous | sparse | soft1 | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.77389 | 473988 |
| heterogeneous | sparse | soft4 | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17089 | 19200 | 1.65374 | 473988 |
| heterogeneous | sparse | typed | signed:fixed:n=128:clean@0.0 | 128 | signed | False | 17105 | 19200 | 2.14354 | 473988 |
| learned | robot | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.04222 | 473988 |
| learned | robot | learned | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17097 | 19200 | 2.18622 | 473988 |
| learned | robot | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.78511 | 473988 |
| learned | robot | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.0579 | 473988 |
| learned | sparse | hard | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.68964 | 473988 |
| learned | sparse | learned | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17097 | 19200 | 2.19652 | 473988 |
| learned | sparse | none | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.67151 | 473988 |
| learned | sparse | soft4 | uniform:fixed:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.77984 | 473988 |
| transfer | sparse | graph_input | uniform:sizes=12:n=128:clean@0.0 | 128 | uniform | False | 17217 | 19200 | 2.2087 | 473988 |
| transfer | sparse | graph_input | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | uniform | False | 17217 | 19200 | 2.12353 | 473988 |
| transfer | sparse | hard | uniform:sizes=12:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.57447 | 473988 |
| transfer | sparse | hard | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.96741 | 473988 |
| transfer | sparse | none | uniform:sizes=12:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.53108 | 470116 |
| transfer | sparse | none | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.67743 | 473988 |
| transfer | sparse | permuted4 | uniform:sizes=12:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.45317 | 473988 |
| transfer | sparse | permuted4 | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.04671 | 473988 |
| transfer | sparse | soft4 | uniform:sizes=12:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 2.0039 | 470116 |
| transfer | sparse | soft4 | uniform:sizes=9,12,16:n=128:clean@0.0 | 128 | uniform | False | 17089 | 19200 | 1.93355 | 473988 |

Thresholds and model choices use validation metrics only. Unreached efficiency thresholds are represented by null values with censoring, never by fabricated counts or speedups.
