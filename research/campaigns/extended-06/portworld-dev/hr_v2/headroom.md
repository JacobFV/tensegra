# portworld headroom (pw-v2; N=1090 instances, 250 episodes, 0 uncertified episodes)

| policy | U | 95% CI |
|---|---|---|
| oracle_arms | 0.9623 | 0.9538 – 0.9703 |
| oracle_arms+cascades | 0.9623 | 0.9538 – 0.9703 |
| learned_oneshot_structured | 0.9552 | 0.9462 – 0.9634 |
| learned_seq[cons:GR] | 0.9514 | 0.9422 – 0.9599 |
| learned_oneshot_gbt | 0.9498 | 0.9405 – 0.9583 |
| learned_sequential | 0.9496 | 0.9404 – 0.9582 |
| learned_seq[cons:RV] | 0.9488 | 0.9391 – 0.9576 |
| learned_seq[cons:PD0] | 0.9474 | 0.9383 – 0.9561 |
| tree_d3 | 0.9468 | 0.9375 – 0.9556 |
| logistic_tuned | 0.9447 | 0.9338 – 0.9547 |
| tree_d2 | 0.9443 | 0.9343 – 0.9541 |
| hand | 0.9424 | 0.9323 – 0.9521 |
| threshold_price | 0.9394 | 0.9288 – 0.9490 |
| threshold_any | 0.9394 | 0.9288 – 0.9490 |
| cascade_tuned | 0.9219 | 0.9102 – 0.9327 |
| cascade_cheap_first | 0.9219 | 0.9102 – 0.9327 |
| threshold_tightness | 0.9213 | 0.9094 – 0.9322 |
| threshold_size | 0.9210 | 0.9072 – 0.9336 |
| A0_single | 0.9204 | 0.9085 – 0.9312 |
| threshold_decomposability | 0.9204 | 0.9064 – 0.9328 |
| cascade_reuse_first | 0.9181 | 0.9063 – 0.9287 |
| cascade_expensive_first | 0.8353 | 0.7846 – 0.8767 |

best simple: **tree_d3**; best learned: **learned_oneshot_structured**
HEADROOM = 0.0084 (CI 0.0053 – 0.0117); /SD(between) = 0.111
best simple − A0 single = 0.0264 (CI 0.0184 – 0.0355)
headroom by learned variant: learned_sequential 0.0028 [-0.0012, 0.0066], learned_oneshot_gbt 0.0030 [-0.0010, 0.0070], learned_oneshot_structured 0.0084 [0.0053, 0.0117]

best-arm frequency (oracle argmax): cons:RV 0.28, insp:RV 0.15, cons:GR 0.11, cons:BM4 0.08, cons:G 0.07, insp:GR 0.04, cons:RUBB10000 0.04, insp:G 0.03, cons:BM16 0.02, cons:RU 0.02, cons:BB10000 0.02, cons:PD10000 0.02

## drivers (tertiles)

| driver | bin range | n | A0 | best simple | hand | learned | oracle | modal best arm |
|---|---|---|---|---|---|---|---|---|
| n | 16–27 | 353 | 0.914 | 0.944 | 0.952 | 0.955 | 0.960 | cons:RV (0.24) |
| n | 28–39 | 343 | 0.911 | 0.942 | 0.945 | 0.950 | 0.957 | cons:RV (0.26) |
| n | 40–52 | 394 | 0.934 | 0.953 | 0.932 | 0.960 | 0.969 | cons:RV (0.34) |
| tight_loc_mean | 0.116–0.327 | 362 | 0.878 | 0.929 | 0.930 | 0.934 | 0.942 | cons:RV (0.25) |
| tight_loc_mean | 0.327–0.497 | 364 | 0.923 | 0.944 | 0.940 | 0.953 | 0.961 | cons:RV (0.27) |
| tight_loc_mean | 0.498–0.877 | 364 | 0.960 | 0.967 | 0.957 | 0.979 | 0.984 | cons:RV (0.33) |
| tight_glob | 0.184–0.585 | 363 | 0.922 | 0.944 | 0.943 | 0.953 | 0.961 | cons:RV (0.26) |
| tight_glob | 0.586–0.98 | 363 | 0.925 | 0.956 | 0.943 | 0.964 | 0.970 | cons:RV (0.30) |
| tight_glob | 0.98–1.59 | 364 | 0.915 | 0.941 | 0.941 | 0.948 | 0.955 | cons:RV (0.29) |
| conf_density | 0–0.121 | 362 | 0.921 | 0.961 | 0.958 | 0.965 | 0.969 | cons:RV (0.25) |
| conf_density | 0.122–0.239 | 363 | 0.927 | 0.947 | 0.942 | 0.953 | 0.962 | cons:RV (0.28) |
| conf_density | 0.24–0.496 | 365 | 0.913 | 0.933 | 0.928 | 0.947 | 0.956 | cons:RV (0.32) |
| n_comp | 1–11 | 1090 | 0.920 | 0.947 | 0.942 | 0.955 | 0.962 | cons:RV (0.28) |
| hidden_frac | 0–0.0286 | 361 | 0.977 | 0.987 | 0.982 | 0.991 | 0.996 | cons:RV (0.38) |
| hidden_frac | 0.0312–0.128 | 364 | 0.919 | 0.941 | 0.939 | 0.954 | 0.962 | cons:RV (0.24) |
| hidden_frac | 0.128–0.35 | 365 | 0.866 | 0.913 | 0.906 | 0.921 | 0.929 | cons:RV (0.23) |
| m | 0–0.194 | 363 | 0.921 | 0.951 | 0.949 | 0.960 | 0.965 | cons:RV (0.50) |
| m | 0.195–0.605 | 363 | 0.922 | 0.946 | 0.938 | 0.954 | 0.963 | cons:RV (0.19) |
| m | 0.607–1 | 364 | 0.918 | 0.944 | 0.940 | 0.951 | 0.960 | cons:RV (0.16) |
| log_c | -8.96–-7.83 | 360 | 0.923 | 0.946 | 0.954 | 0.958 | 0.961 | cons:RV (0.26) |
| log_c | -7.83–-6.7 | 364 | 0.926 | 0.956 | 0.959 | 0.962 | 0.968 | cons:RV (0.28) |
| log_c | -6.66–-5.51 | 366 | 0.912 | 0.939 | 0.915 | 0.946 | 0.957 | cons:RV (0.31) |
| log_o | -2.99–-2.28 | 362 | 0.925 | 0.982 | 0.976 | 0.985 | 0.991 | insp:RV (0.25) |
| log_o | -2.28–-1.44 | 360 | 0.930 | 0.948 | 0.946 | 0.962 | 0.970 | cons:RV (0.28) |
| log_o | -1.44–-0.7 | 368 | 0.906 | 0.911 | 0.906 | 0.919 | 0.926 | cons:RV (0.37) |

## full-data depth-3 policy tree (descriptive)

```
if width < 0.1038:
  if log_c < -7.383:
    if log_o < -2.551:
      -> insp:RUBB100000
    else:
      -> cons:PD1000000
  else:
    if cache_size < 12:
      -> cons:PD100000
    else:
      -> ('cons', 'GR', 'RUBB', 1000, 0.02)
else:
  if tight_loc_mean < 0.3513:
    if log_obs_cost < -0.6168:
      -> insp:RUBB10000
    else:
      -> ('cons', 'GR', 'RUBB', 10000, 0.08)
  else:
    if log_obs_cost < -1.088:
      -> insp:RUBB10000
    else:
      -> ('cons', 'GR', 'RUBB', 1000, 0.02)

```