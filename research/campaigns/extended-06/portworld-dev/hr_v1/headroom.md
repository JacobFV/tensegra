# portworld headroom (pw-v1; N=1099 instances, 249 episodes, 1 uncertified episodes)

| policy | U | 95% CI |
|---|---|---|
| oracle_arms | 0.9854 | 0.9823 – 0.9881 |
| oracle_arms+cascades | 0.9854 | 0.9823 – 0.9881 |
| learned_oneshot_structured | 0.9790 | 0.9752 – 0.9824 |
| learned_oneshot_gbt | 0.9781 | 0.9743 – 0.9815 |
| learned_seq[cons:GR] | 0.9779 | 0.9741 – 0.9813 |
| learned_sequential | 0.9778 | 0.9740 – 0.9811 |
| tree_d2 | 0.9774 | 0.9732 – 0.9809 |
| learned_seq[cons:RV] | 0.9770 | 0.9732 – 0.9804 |
| tree_d3 | 0.9760 | 0.9710 – 0.9801 |
| logistic_tuned | 0.9755 | 0.9707 – 0.9795 |
| learned_seq[cons:PD0] | 0.9735 | 0.9693 – 0.9772 |
| threshold_size | 0.9713 | 0.9662 – 0.9758 |
| threshold_price | 0.9710 | 0.9653 – 0.9759 |
| cascade_tuned | 0.9709 | 0.9659 – 0.9754 |
| cascade_cheap_first | 0.9709 | 0.9659 – 0.9754 |
| threshold_any | 0.9704 | 0.9649 – 0.9751 |
| threshold_decomposability | 0.9701 | 0.9645 – 0.9751 |
| threshold_tightness | 0.9697 | 0.9643 – 0.9743 |
| A0_single | 0.9691 | 0.9639 – 0.9737 |
| hand | 0.9661 | 0.9599 – 0.9716 |
| cascade_reuse_first | 0.9661 | 0.9610 – 0.9705 |
| cascade_expensive_first | 0.9214 | 0.8979 – 0.9403 |

best simple: **tree_d2**; best learned: **learned_oneshot_structured**
HEADROOM = 0.0016 (CI -0.0002 – 0.0034); /SD(between) = 0.048
best simple − A0 single = 0.0083 (CI 0.0052 – 0.0120)
headroom by learned variant: learned_sequential 0.0004 [-0.0014, 0.0022], learned_oneshot_gbt 0.0007 [-0.0010, 0.0025], learned_oneshot_structured 0.0016 [-0.0002, 0.0034]

best-arm frequency (oracle argmax): cons:RV 0.29, insp:RV 0.27, cons:BM4 0.06, cons:GR 0.04, cons:RU 0.04, insp:GR 0.04, insp:BM4 0.03, cons:G 0.03, insp:RU 0.03, cons:BM16 0.03, cons:RUBB10000 0.02, insp:G 0.01

## drivers (tertiles)

| driver | bin range | n | A0 | best simple | hand | learned | oracle | modal best arm |
|---|---|---|---|---|---|---|---|---|
| n | 16–28 | 313 | 0.978 | 0.981 | 0.978 | 0.983 | 0.986 | insp:RV (0.33) |
| n | 29–39 | 402 | 0.968 | 0.980 | 0.968 | 0.981 | 0.987 | cons:RV (0.34) |
| n | 40–52 | 384 | 0.963 | 0.972 | 0.955 | 0.974 | 0.983 | cons:RV (0.28) |
| tight_loc_mean | 0.117–0.341 | 366 | 0.969 | 0.975 | 0.971 | 0.975 | 0.984 | insp:RV (0.26) |
| tight_loc_mean | 0.342–0.529 | 366 | 0.966 | 0.976 | 0.965 | 0.978 | 0.984 | insp:RV (0.31) |
| tight_loc_mean | 0.532–0.888 | 367 | 0.972 | 0.981 | 0.962 | 0.984 | 0.989 | cons:RV (0.35) |
| tight_glob | 0.176–0.59 | 366 | 0.971 | 0.975 | 0.965 | 0.979 | 0.986 | cons:RV (0.28) |
| tight_glob | 0.593–0.927 | 366 | 0.968 | 0.981 | 0.967 | 0.981 | 0.988 | cons:RV (0.31) |
| tight_glob | 0.927–1.6 | 367 | 0.968 | 0.976 | 0.966 | 0.977 | 0.983 | cons:RV (0.27) |
| conf_density | 0–0.101 | 365 | 0.974 | 0.976 | 0.968 | 0.978 | 0.985 | insp:RV (0.27) |
| conf_density | 0.101–0.2 | 365 | 0.970 | 0.978 | 0.970 | 0.978 | 0.985 | insp:RV (0.30) |
| conf_density | 0.203–0.467 | 369 | 0.964 | 0.978 | 0.961 | 0.981 | 0.986 | cons:RV (0.33) |
| n_comp | 1–8 | 1099 | 0.969 | 0.977 | 0.966 | 0.979 | 0.985 | cons:RV (0.29) |
| hidden_frac | 0–0.0256 | 365 | 0.987 | 0.991 | 0.979 | 0.992 | 0.997 | cons:RV (0.50) |
| hidden_frac | 0.027–0.117 | 363 | 0.975 | 0.980 | 0.970 | 0.981 | 0.987 | insp:RV (0.37) |
| hidden_frac | 0.12–0.36 | 371 | 0.946 | 0.962 | 0.950 | 0.964 | 0.973 | insp:RV (0.37) |
| m | 0–0.079 | 366 | 0.975 | 0.981 | 0.975 | 0.983 | 0.987 | cons:RV (0.46) |
| m | 0.0791–0.298 | 366 | 0.968 | 0.977 | 0.962 | 0.979 | 0.986 | cons:RV (0.22) |
| m | 0.298–0.5 | 367 | 0.964 | 0.974 | 0.961 | 0.976 | 0.984 | cons:RV (0.19) |
| log_c | -8.99–-7.76 | 363 | 0.969 | 0.979 | 0.980 | 0.983 | 0.987 | cons:RV (0.28) |
| log_c | -7.69–-6.67 | 368 | 0.971 | 0.978 | 0.972 | 0.979 | 0.986 | cons:RV (0.27) |
| log_c | -6.67–-5.51 | 368 | 0.967 | 0.975 | 0.946 | 0.975 | 0.983 | cons:RV (0.31) |
| log_o | -3.5–-2.81 | 362 | 0.985 | 0.989 | 0.974 | 0.990 | 0.995 | cons:RV (0.34) |
| log_o | -2.8–-2.19 | 368 | 0.978 | 0.982 | 0.971 | 0.982 | 0.988 | insp:RV (0.32) |
| log_o | -2.18–-1.5 | 369 | 0.944 | 0.962 | 0.954 | 0.966 | 0.974 | cons:RV (0.31) |

## full-data depth-3 policy tree (descriptive)

```
if log_obs_cost < -1.26:
  if log_c < -7.613:
    if hidden_caps < 1:
      -> cons:RUBB1000000
    else:
      -> insp:RUBB1000000
  else:
    if log_c < -6.641:
      -> ('insp', 'GR', 'RUBB', 10000, 0.02)
    else:
      -> ('insp', 'GR', 'RUBB', 1000, 0.04)
else:
  if L < 0.9772:
    -> cons:RU
  else:
    if width < 0.1999:
      -> cons:RUBB100000
    else:
      -> ('insp', 'GR', 'RUBB', 1000, 0.04)

```