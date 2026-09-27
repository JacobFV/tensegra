# A-CF-SMALL confirmation

Confirm population: 2852 instances / 640 episodes; oracle certified 0.999 (4 scored vs UB). Hidden-state oracle 0.9826.

## Primary (per lineage): U(learned) − U(best simple chosen on select), episode-clustered 95% CI

| lineage | best simple (select) | one-shot − best simple | pass | sequential − best simple | pass | one-shot − best single (GA-2 analogue) | seq − best single |
|---|---|---|---|---|---|---|---|
| 0 | tree_d3 | +0.0032 [+0.0006, +0.0067] | True | +0.0034 [+0.0004, +0.0069] | True | +0.0351 [+0.0296, +0.0407] | +0.0353 [+0.0297, +0.0410] |
| 1 | tree_d3 | +0.0025 [+0.0005, +0.0046] | True | +0.0032 [+0.0012, +0.0053] | True | +0.0349 [+0.0294, +0.0406] | +0.0356 [+0.0301, +0.0413] |
| 2 | tree_d3 | +0.0043 [+0.0014, +0.0080] | True | +0.0052 [+0.0022, +0.0087] | True | +0.0352 [+0.0298, +0.0408] | +0.0361 [+0.0307, +0.0418] |

learned_oneshot: lineages passing 3/3; mean over lineages (paired) +0.0033 [+0.0016, +0.0055]

learned_sequential: lineages passing 3/3; mean over lineages (paired) +0.0039 [+0.0021, +0.0062]

**Primary (one-shot, 3/3 lineages): PASS**; sequential all lineages: True.
A pass establishes a small margin below the registered .02 practical-headroom threshold (registry practical_note).

## Utilities on confirm (per lineage)

U select = the select-population score used for choosing; charge = selector compute (features + inference) in work units, already subtracted from U (the tuned single/cascade/strong/hand families carry their own charges inside their utilities and show 0 here).

### lineage 0 (chosen: {'logistic_tuned': 'logistic[lam=0.001]', 'learned_oneshot': 'learned_oneshot[cfg=100x3]', 'best_simple': 'tree_d3', 'best_single': 'A0_single'}; fit.pkl e0ff1ace72b2)

| policy | U confirm | 95% CI | U select | charge (work units) |
|---|---|---|---|---|
| learned_seq[cons:RV+cons:GR] | 0.9604 | 0.9558 – 0.9644 | 0.9596 | 1384 |
| learned_seq[cons:GR] | 0.9596 | 0.9549 – 0.9637 | 0.9600 | 1384 |
| learned_sequential | 0.9592 | 0.9542 – 0.9637 | 0.9596 | 1684 |
| learned_oneshot | 0.9590 | 0.9544 – 0.9631 | 0.9585 | 1384 |
| best_simple | 0.9558 | 0.9502 – 0.9608 | 0.9558 | 487 |
| tree_d3 | 0.9558 | 0.9502 – 0.9608 | 0.9558 | 487 |
| tree_d2 | 0.9529 | 0.9473 – 0.9582 | 0.9520 | 486 |
| learned_oneshot_direct | 0.9516 | 0.9454 – 0.9571 | 0.9539 | 1084 |
| learned_seq[cons:PD1000] | 0.9514 | 0.9467 – 0.9559 | 0.9515 | 1384 |
| logistic_tuned | 0.9505 | 0.9453 – 0.9551 | 0.9479 | 3004 |
| threshold_any | 0.9492 | 0.9431 – 0.9547 | 0.9475 | 485 |
| strong_baseline | 0.9441 | 0.9380 – 0.9495 | 0.9408 | 0 |
| hand | 0.9371 | 0.9279 – 0.9455 | 0.9319 | 0 |
| learned_seq[opt:GR] | 0.9317 | 0.9237 – 0.9394 | 0.9317 | 1384 |
| cascade_tuned | 0.9261 | 0.9193 – 0.9326 | 0.9252 | 0 |
| best_single | 0.9239 | 0.9170 – 0.9308 | 0.9216 | 0 |
| A0_single | 0.9239 | 0.9170 – 0.9308 | 0.9216 | 0 |

Reported, not a gate: one-shot − strongest individual simple family in hindsight on confirm (tree_d3): +0.0032 [+0.0007, +0.0067]

### lineage 1 (chosen: {'logistic_tuned': 'logistic[lam=0.001]', 'learned_oneshot': 'learned_oneshot[cfg=100x3]', 'best_simple': 'tree_d3', 'best_single': 'A0_single'}; fit.pkl 1bff3f53ee32)

| policy | U confirm | 95% CI | U select | charge (work units) |
|---|---|---|---|---|
| learned_seq[cons:RV+cons:GR] | 0.9601 | 0.9556 – 0.9643 | 0.9592 | 1384 |
| learned_seq[cons:GR] | 0.9600 | 0.9554 – 0.9642 | 0.9598 | 1384 |
| learned_sequential | 0.9595 | 0.9549 – 0.9637 | 0.9591 | 1684 |
| learned_oneshot | 0.9588 | 0.9542 – 0.9630 | 0.9586 | 1384 |
| best_simple | 0.9563 | 0.9512 – 0.9606 | 0.9553 | 487 |
| tree_d3 | 0.9563 | 0.9512 – 0.9606 | 0.9553 | 487 |
| tree_d2 | 0.9554 | 0.9505 – 0.9598 | 0.9535 | 486 |
| learned_oneshot_direct | 0.9533 | 0.9476 – 0.9585 | 0.9515 | 1084 |
| learned_seq[cons:PD1000] | 0.9516 | 0.9469 – 0.9560 | 0.9503 | 1384 |
| logistic_tuned | 0.9480 | 0.9421 – 0.9531 | 0.9447 | 3004 |
| threshold_any | 0.9478 | 0.9422 – 0.9528 | 0.9461 | 485 |
| strong_baseline | 0.9440 | 0.9383 – 0.9494 | 0.9427 | 0 |
| hand | 0.9371 | 0.9279 – 0.9455 | 0.9319 | 0 |
| learned_seq[opt:GR] | 0.9331 | 0.9253 – 0.9409 | 0.9316 | 1384 |
| cascade_tuned | 0.9261 | 0.9193 – 0.9326 | 0.9252 | 0 |
| best_single | 0.9239 | 0.9170 – 0.9308 | 0.9216 | 0 |
| A0_single | 0.9239 | 0.9170 – 0.9308 | 0.9216 | 0 |

Reported, not a gate: one-shot − strongest individual simple family in hindsight on confirm (tree_d3): +0.0025 [+0.0004, +0.0046]

### lineage 2 (chosen: {'logistic_tuned': 'logistic[lam=0.001]', 'learned_oneshot': 'learned_oneshot[cfg=100x3]', 'best_simple': 'tree_d3', 'best_single': 'A0_single'}; fit.pkl 924604941e1a)

| policy | U confirm | 95% CI | U select | charge (work units) |
|---|---|---|---|---|
| learned_seq[cons:GR] | 0.9602 | 0.9556 – 0.9643 | 0.9599 | 1384 |
| learned_seq[cons:RV+cons:GR] | 0.9601 | 0.9555 – 0.9642 | 0.9594 | 1384 |
| learned_sequential | 0.9601 | 0.9554 – 0.9641 | 0.9593 | 1684 |
| learned_oneshot | 0.9592 | 0.9545 – 0.9632 | 0.9591 | 1384 |
| tree_d2 | 0.9556 | 0.9507 – 0.9600 | 0.9535 | 486 |
| best_simple | 0.9549 | 0.9494 – 0.9599 | 0.9549 | 487 |
| tree_d3 | 0.9549 | 0.9494 – 0.9599 | 0.9549 | 487 |
| learned_oneshot_direct | 0.9528 | 0.9444 – 0.9593 | 0.9553 | 1084 |
| learned_seq[cons:PD1000] | 0.9521 | 0.9475 – 0.9566 | 0.9514 | 1384 |
| logistic_tuned | 0.9481 | 0.9421 – 0.9535 | 0.9490 | 3004 |
| threshold_any | 0.9477 | 0.9416 – 0.9534 | 0.9475 | 485 |
| strong_baseline | 0.9440 | 0.9383 – 0.9494 | 0.9427 | 0 |
| hand | 0.9371 | 0.9279 – 0.9455 | 0.9319 | 0 |
| learned_seq[opt:GR] | 0.9333 | 0.9256 – 0.9407 | 0.9313 | 1384 |
| cascade_tuned | 0.9261 | 0.9193 – 0.9326 | 0.9252 | 0 |
| best_single | 0.9239 | 0.9170 – 0.9308 | 0.9216 | 0 |
| A0_single | 0.9239 | 0.9170 – 0.9308 | 0.9216 | 0 |

Reported, not a gate: one-shot − strongest individual simple family in hindsight on confirm (tree_d2): +0.0035 [+0.0019, +0.0055]

## Drivers (confirm tertiles; one-shot − best simple per lineage)

| driver | range | n | oracle | L0 best simple | L0 1-shot − simple | L0 seq − simple | L1 best simple | L1 1-shot − simple | L1 seq − simple | L2 best simple | L2 1-shot − simple | L2 seq − simple |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 16–26 | 913 | 0.986 | 0.959 | +0.0011 | +0.0019 | 0.959 | +0.0013 | +0.0016 | 0.958 | +0.0027 | +0.0027 |
| n | 27–35 | 931 | 0.982 | 0.960 | -0.0002 | +0.0009 | 0.955 | +0.0047 | +0.0050 | 0.956 | +0.0039 | +0.0051 |
| n | 36–50 | 1008 | 0.980 | 0.949 | +0.0083 | +0.0070 | 0.955 | +0.0017 | +0.0030 | 0.951 | +0.0060 | +0.0075 |
| corr_class | 0–0 | 544 | 0.981 | 0.955 | +0.0017 | +0.0014 | 0.954 | +0.0021 | +0.0025 | 0.952 | +0.0041 | +0.0050 |
| corr_class | 1–2 | 1191 | 0.984 | 0.960 | +0.0044 | +0.0038 | 0.963 | +0.0017 | +0.0024 | 0.960 | +0.0046 | +0.0051 |
| corr_class | 3–4 | 1117 | 0.982 | 0.952 | +0.0027 | +0.0040 | 0.950 | +0.0037 | +0.0044 | 0.951 | +0.0040 | +0.0054 |
| tight_loc_mean | 0.0854–0.32 | 951 | 0.977 | 0.946 | +0.0024 | +0.0019 | 0.946 | +0.0032 | +0.0040 | 0.945 | +0.0040 | +0.0055 |
| tight_loc_mean | 0.32–0.485 | 950 | 0.981 | 0.951 | +0.0043 | +0.0058 | 0.954 | +0.0023 | +0.0025 | 0.954 | +0.0021 | +0.0027 |
| tight_loc_mean | 0.485–0.908 | 951 | 0.989 | 0.970 | +0.0029 | +0.0025 | 0.969 | +0.0022 | +0.0032 | 0.966 | +0.0066 | +0.0074 |
| tight_glob | 0.143–0.558 | 951 | 0.984 | 0.953 | +0.0049 | +0.0056 | 0.955 | +0.0021 | +0.0029 | 0.955 | +0.0032 | +0.0046 |
| tight_glob | 0.559–0.958 | 950 | 0.980 | 0.951 | +0.0017 | +0.0012 | 0.950 | +0.0028 | +0.0044 | 0.944 | +0.0085 | +0.0098 |
| tight_glob | 0.959–1.65 | 951 | 0.984 | 0.963 | +0.0030 | +0.0035 | 0.964 | +0.0027 | +0.0024 | 0.966 | +0.0010 | +0.0012 |
| conf_density | 0–0.126 | 948 | 0.982 | 0.962 | +0.0014 | -0.0002 | 0.958 | +0.0057 | +0.0059 | 0.962 | +0.0016 | +0.0019 |
| conf_density | 0.126–0.249 | 926 | 0.981 | 0.949 | +0.0058 | +0.0070 | 0.953 | +0.0024 | +0.0031 | 0.952 | +0.0029 | +0.0048 |
| conf_density | 0.25–0.6 | 978 | 0.985 | 0.956 | +0.0025 | +0.0035 | 0.958 | -0.0004 | +0.0006 | 0.951 | +0.0081 | +0.0086 |
| n_comp | 1–9 | 2852 | 0.983 | 0.956 | +0.0032 | +0.0034 | 0.956 | +0.0025 | +0.0032 | 0.955 | +0.0043 | +0.0052 |
| hidden_frac | 0–0.0364 | 951 | 0.994 | 0.981 | +0.0011 | +0.0005 | 0.981 | +0.0022 | +0.0025 | 0.978 | +0.0059 | +0.0066 |
| hidden_frac | 0.037–0.13 | 949 | 0.985 | 0.957 | +0.0040 | +0.0050 | 0.958 | +0.0006 | +0.0021 | 0.959 | +0.0008 | +0.0022 |
| hidden_frac | 0.132–0.381 | 952 | 0.969 | 0.930 | +0.0045 | +0.0047 | 0.930 | +0.0048 | +0.0051 | 0.928 | +0.0061 | +0.0067 |
| m | 0–0.212 | 951 | 0.988 | 0.962 | +0.0025 | +0.0022 | 0.961 | +0.0023 | +0.0018 | 0.958 | +0.0060 | +0.0056 |
| m | 0.213–0.597 | 950 | 0.981 | 0.954 | +0.0042 | +0.0041 | 0.957 | +0.0025 | +0.0035 | 0.955 | +0.0049 | +0.0062 |
| m | 0.597–1 | 951 | 0.978 | 0.951 | +0.0028 | +0.0039 | 0.951 | +0.0028 | +0.0044 | 0.951 | +0.0019 | +0.0037 |
| log_c | -7.29–-6.62 | 949 | 0.987 | 0.963 | +0.0055 | +0.0056 | 0.964 | +0.0055 | +0.0056 | 0.966 | +0.0028 | +0.0031 |
| log_c | -6.61–-5.96 | 950 | 0.983 | 0.955 | +0.0063 | +0.0075 | 0.958 | +0.0020 | +0.0022 | 0.952 | +0.0090 | +0.0094 |
| log_c | -5.96–-5.3 | 953 | 0.978 | 0.949 | -0.0021 | -0.0028 | 0.947 | +0.0002 | +0.0019 | 0.946 | +0.0009 | +0.0030 |
| log_o | -3–-2.3 | 948 | 0.991 | 0.973 | +0.0050 | +0.0051 | 0.975 | +0.0032 | +0.0024 | 0.976 | +0.0018 | +0.0012 |
| log_o | -2.29–-1.65 | 952 | 0.982 | 0.957 | +0.0046 | +0.0033 | 0.958 | +0.0036 | +0.0045 | 0.951 | +0.0105 | +0.0118 |
| log_o | -1.64–-1.01 | 952 | 0.974 | 0.938 | -0.0001 | +0.0019 | 0.936 | +0.0009 | +0.0027 | 0.938 | +0.0004 | +0.0025 |