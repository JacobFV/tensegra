# A-CF-SMALL confirmation (SMOKE, dev_builder seeds; not protocol)

Confirm population: 540 instances / 120 episodes; oracle certified 1.000 (0 scored vs UB). Hidden-state oracle 0.9839.

## Primary (per lineage): U(learned) − U(best simple chosen on select), episode-clustered 95% CI

| lineage | best simple (select) | one-shot − best simple | pass | sequential − best simple | pass | one-shot − best single (GA-2 analogue) | seq − best single |
|---|---|---|---|---|---|---|---|
| 0 | tree_d2 | +0.0064 [-0.0118, +0.0231] | False | +0.0112 [-0.0014, +0.0255] | False | +0.0473 [+0.0218, +0.0755] | +0.0520 [+0.0287, +0.0798] |
| 1 | threshold_any | +0.0085 [-0.0058, +0.0293] | False | +0.0119 [+0.0011, +0.0309] | True | +0.0508 [+0.0277, +0.0786] | +0.0542 [+0.0318, +0.0806] |
| 2 | tree_d2 | +0.0066 [+0.0023, +0.0111] | True | +0.0070 [+0.0029, +0.0115] | True | +0.0400 [+0.0284, +0.0526] | +0.0404 [+0.0287, +0.0532] |

learned_oneshot: lineages passing 1/3; mean over lineages (paired) +0.0072 [-0.0026, +0.0161]

learned_sequential: lineages passing 2/3; mean over lineages (paired) +0.0100 [+0.0030, +0.0178]

**Primary (one-shot, 3/3 lineages): FAIL**; sequential all lineages: False.
A pass establishes a small margin below the registered .02 practical-headroom threshold (registry practical_note).

## Utilities on confirm (per lineage; select-population score in parentheses; selector charge in work units)

### lineage 0 (chosen: {'logistic_tuned': 'logistic[lam=0.001]', 'learned_oneshot': 'learned_oneshot[cfg=100x3]', 'best_simple': 'tree_d2', 'best_single': 'A0_single'}; fit.pkl 86eba90fe644)

| policy | U confirm | 95% CI | U select | charge (work units) |
|---|---|---|---|---|
| learned_sequential | 0.9574 | 0.9471 – 0.9663 | 0.9539 | 1680 |
| learned_seq[cons:GR] | 0.9571 | 0.9466 – 0.9662 | 0.9547 | 1380 |
| learned_oneshot | 0.9526 | 0.9356 – 0.9651 | 0.9525 | 1380 |
| learned_seq[cons:RV+cons:GR] | 0.9516 | 0.9325 – 0.9659 | 0.9560 | 1380 |
| learned_seq[cons:PD1000] | 0.9510 | 0.9407 – 0.9602 | 0.9446 | 1380 |
| logistic_tuned | 0.9474 | 0.9343 – 0.9584 | 0.9353 | 3000 |
| best_simple | 0.9462 | 0.9308 – 0.9588 | 0.9359 | 482 |
| tree_d2 | 0.9462 | 0.9308 – 0.9588 | 0.9359 | 482 |
| threshold_any | 0.9448 | 0.9292 – 0.9574 | 0.9304 | 481 |
| strong_baseline | 0.9433 | 0.9307 – 0.9547 | 0.9288 | 0 |
| hand | 0.9401 | 0.9183 – 0.9576 | 0.9323 | 0 |
| tree_d3 | 0.9314 | 0.9086 – 0.9500 | 0.9319 | 483 |
| learned_oneshot_direct | 0.9249 | 0.9001 – 0.9451 | 0.9330 | 1080 |
| learned_seq[opt:GR] | 0.9221 | 0.8934 – 0.9453 | 0.9236 | 1380 |
| cascade_tuned | 0.9071 | 0.8758 – 0.9328 | 0.9035 | 0 |
| best_single | 0.9054 | 0.8742 – 0.9309 | 0.9073 | 0 |
| A0_single | 0.9054 | 0.8742 – 0.9309 | 0.9073 | 0 |

Reported, not a gate: one-shot − strongest individual simple family in hindsight on confirm (logistic_tuned): +0.0052 [-0.0118, +0.0191]

### lineage 1 (chosen: {'logistic_tuned': 'logistic[lam=0.01]', 'learned_oneshot': 'learned_oneshot[cfg=100x3]', 'best_simple': 'threshold_any', 'best_single': 'A0_single'}; fit.pkl c707c4ff0764)

| policy | U confirm | 95% CI | U select | charge (work units) |
|---|---|---|---|---|
| learned_seq[cons:GR] | 0.9607 | 0.9527 – 0.9682 | 0.9568 | 1380 |
| learned_sequential | 0.9595 | 0.9514 – 0.9671 | 0.9542 | 1680 |
| learned_seq[cons:RV+cons:GR] | 0.9571 | 0.9455 – 0.9662 | 0.9561 | 1380 |
| learned_oneshot | 0.9561 | 0.9457 – 0.9653 | 0.9525 | 1380 |
| tree_d2 | 0.9483 | 0.9289 – 0.9618 | 0.9371 | 482 |
| learned_oneshot_direct | 0.9482 | 0.9345 – 0.9595 | 0.9401 | 1080 |
| logistic_tuned | 0.9480 | 0.9371 – 0.9579 | 0.9401 | 3000 |
| best_simple | 0.9476 | 0.9284 – 0.9617 | 0.9432 | 481 |
| threshold_any | 0.9476 | 0.9284 – 0.9617 | 0.9432 | 481 |
| learned_seq[cons:PD1000] | 0.9463 | 0.9330 – 0.9577 | 0.9465 | 1380 |
| strong_baseline | 0.9453 | 0.9315 – 0.9566 | 0.9318 | 0 |
| tree_d3 | 0.9404 | 0.9179 – 0.9586 | 0.9288 | 483 |
| hand | 0.9401 | 0.9183 – 0.9576 | 0.9323 | 0 |
| learned_seq[opt:GR] | 0.9241 | 0.8941 – 0.9482 | 0.9275 | 1380 |
| best_single | 0.9054 | 0.8742 – 0.9309 | 0.9073 | 0 |
| A0_single | 0.9054 | 0.8742 – 0.9309 | 0.9073 | 0 |
| cascade_tuned | 0.9039 | 0.8724 – 0.9294 | 0.9081 | 0 |

Reported, not a gate: one-shot − strongest individual simple family in hindsight on confirm (tree_d2): +0.0078 [-0.0064, +0.0264]

### lineage 2 (chosen: {'logistic_tuned': 'logistic[lam=0.001]', 'learned_oneshot': 'learned_oneshot[cfg=100x3]', 'best_simple': 'tree_d2', 'best_single': 'A0_single'}; fit.pkl bf42f7a889b5)

| policy | U confirm | 95% CI | U select | charge (work units) |
|---|---|---|---|---|
| learned_seq[cons:GR] | 0.9615 | 0.9535 – 0.9690 | 0.9546 | 1380 |
| learned_seq[cons:RV+cons:GR] | 0.9613 | 0.9535 – 0.9687 | 0.9560 | 1380 |
| learned_sequential | 0.9609 | 0.9528 – 0.9685 | 0.9539 | 1680 |
| learned_oneshot | 0.9604 | 0.9524 – 0.9680 | 0.9534 | 1380 |
| learned_seq[cons:PD1000] | 0.9542 | 0.9456 – 0.9622 | 0.9413 | 1380 |
| best_simple | 0.9539 | 0.9443 – 0.9631 | 0.9500 | 482 |
| tree_d2 | 0.9539 | 0.9443 – 0.9631 | 0.9500 | 482 |
| tree_d3 | 0.9536 | 0.9425 – 0.9638 | 0.9301 | 483 |
| learned_oneshot_direct | 0.9478 | 0.9290 – 0.9614 | 0.9413 | 1080 |
| strong_baseline | 0.9433 | 0.9307 – 0.9547 | 0.9288 | 0 |
| hand | 0.9404 | 0.9188 – 0.9574 | 0.9326 | 0 |
| logistic_tuned | 0.9400 | 0.9235 – 0.9531 | 0.9292 | 3000 |
| threshold_any | 0.9348 | 0.9138 – 0.9500 | 0.9486 | 481 |
| learned_seq[opt:GR] | 0.9243 | 0.8939 – 0.9484 | 0.9252 | 1380 |
| cascade_tuned | 0.9210 | 0.9071 – 0.9347 | 0.9138 | 0 |
| best_single | 0.9204 | 0.9057 – 0.9348 | 0.9071 | 0 |
| A0_single | 0.9204 | 0.9057 – 0.9348 | 0.9071 | 0 |

Reported, not a gate: one-shot − strongest individual simple family in hindsight on confirm (tree_d2): +0.0066 [+0.0023, +0.0112]

## Drivers (confirm tertiles; one-shot − best simple per lineage)

| driver | range | n | oracle | L0 best simple | L0 1-shot − simple | L0 seq − simple | L1 best simple | L1 1-shot − simple | L1 seq − simple | L2 best simple | L2 1-shot − simple | L2 seq − simple |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 16–26 | 176 | 0.992 | 0.961 | +0.0071 | +0.0105 | 0.974 | -0.0033 | -0.0017 | 0.969 | +0.0010 | +0.0015 |
| n | 27–35 | 179 | 0.980 | 0.936 | +0.0005 | +0.0108 | 0.953 | -0.0092 | +0.0031 | 0.952 | +0.0059 | +0.0070 |
| n | 36–50 | 185 | 0.980 | 0.942 | +0.0115 | +0.0121 | 0.917 | +0.0369 | +0.0333 | 0.942 | +0.0125 | +0.0123 |
| corr_class | 0–0 | 99 | 0.987 | 0.955 | -0.0336 | -0.0147 | 0.911 | +0.0237 | +0.0434 | 0.952 | +0.0043 | +0.0032 |
| corr_class | 1–2 | 228 | 0.980 | 0.929 | +0.0289 | +0.0302 | 0.951 | +0.0061 | +0.0063 | 0.950 | +0.0064 | +0.0079 |
| corr_class | 3–4 | 213 | 0.986 | 0.961 | +0.0010 | +0.0028 | 0.961 | +0.0041 | +0.0032 | 0.959 | +0.0078 | +0.0079 |
| tight_loc_mean | 0.108–0.332 | 180 | 0.978 | 0.955 | +0.0028 | +0.0030 | 0.958 | -0.0003 | -0.0011 | 0.952 | +0.0070 | +0.0074 |
| tight_loc_mean | 0.333–0.515 | 180 | 0.984 | 0.948 | +0.0058 | +0.0075 | 0.946 | +0.0063 | +0.0087 | 0.946 | +0.0094 | +0.0101 |
| tight_loc_mean | 0.516–1.09 | 180 | 0.990 | 0.935 | +0.0107 | +0.0230 | 0.939 | +0.0197 | +0.0281 | 0.964 | +0.0033 | +0.0036 |
| tight_glob | 0.195–0.532 | 180 | 0.990 | 0.961 | -0.0107 | -0.0015 | 0.964 | -0.0043 | +0.0032 | 0.965 | +0.0066 | +0.0069 |
| tight_glob | 0.533–0.967 | 179 | 0.981 | 0.948 | +0.0085 | +0.0125 | 0.933 | +0.0245 | +0.0255 | 0.951 | +0.0062 | +0.0070 |
| tight_glob | 0.969–1.58 | 181 | 0.980 | 0.930 | +0.0214 | +0.0225 | 0.945 | +0.0055 | +0.0070 | 0.945 | +0.0068 | +0.0071 |
| conf_density | 0–0.0909 | 177 | 0.989 | 0.959 | +0.0079 | +0.0091 | 0.963 | +0.0050 | +0.0062 | 0.965 | +0.0046 | +0.0053 |
| conf_density | 0.0921–0.211 | 179 | 0.976 | 0.922 | +0.0279 | +0.0315 | 0.946 | +0.0042 | +0.0048 | 0.944 | +0.0054 | +0.0067 |
| conf_density | 0.221–0.565 | 184 | 0.986 | 0.957 | -0.0158 | -0.0066 | 0.935 | +0.0161 | +0.0243 | 0.953 | +0.0096 | +0.0089 |
| n_comp | 1–8 | 540 | 0.984 | 0.946 | +0.0064 | +0.0112 | 0.948 | +0.0085 | +0.0119 | 0.954 | +0.0066 | +0.0070 |
| hidden_frac | 0–0.04 | 176 | 0.995 | 0.978 | +0.0020 | +0.0026 | 0.978 | +0.0030 | +0.0038 | 0.980 | +0.0025 | +0.0027 |
| hidden_frac | 0.0408–0.137 | 181 | 0.985 | 0.943 | -0.0050 | +0.0066 | 0.929 | +0.0178 | +0.0276 | 0.949 | +0.0089 | +0.0084 |
| hidden_frac | 0.138–0.29 | 183 | 0.972 | 0.918 | +0.0219 | +0.0239 | 0.937 | +0.0047 | +0.0041 | 0.933 | +0.0081 | +0.0098 |
| m | 0–0.213 | 180 | 0.987 | 0.935 | +0.0316 | +0.0305 | 0.950 | +0.0169 | +0.0160 | 0.961 | +0.0050 | +0.0049 |
| m | 0.217–0.625 | 180 | 0.984 | 0.953 | -0.0160 | -0.0037 | 0.955 | +0.0030 | +0.0031 | 0.950 | +0.0090 | +0.0097 |
| m | 0.629–0.999 | 180 | 0.981 | 0.950 | +0.0037 | +0.0066 | 0.938 | +0.0058 | +0.0166 | 0.950 | +0.0056 | +0.0065 |
| log_c | -7.29–-6.72 | 180 | 0.990 | 0.964 | -0.0083 | +0.0020 | 0.951 | +0.0108 | +0.0218 | 0.965 | +0.0094 | +0.0096 |
| log_c | -6.69–-5.97 | 178 | 0.981 | 0.940 | +0.0187 | +0.0198 | 0.960 | -0.0000 | -0.0005 | 0.958 | +0.0016 | +0.0018 |
| log_c | -5.95–-5.36 | 182 | 0.980 | 0.934 | +0.0090 | +0.0117 | 0.932 | +0.0147 | +0.0142 | 0.938 | +0.0086 | +0.0095 |
| log_o | -2.99–-2.44 | 178 | 0.992 | 0.966 | -0.0098 | +0.0010 | 0.946 | +0.0199 | +0.0289 | 0.973 | +0.0048 | +0.0050 |
| log_o | -2.43–-1.69 | 182 | 0.985 | 0.951 | +0.0126 | +0.0118 | 0.966 | -0.0018 | -0.0046 | 0.957 | +0.0076 | +0.0085 |
| log_o | -1.69–-1.02 | 180 | 0.975 | 0.921 | +0.0162 | +0.0206 | 0.930 | +0.0077 | +0.0118 | 0.932 | +0.0073 | +0.0076 |