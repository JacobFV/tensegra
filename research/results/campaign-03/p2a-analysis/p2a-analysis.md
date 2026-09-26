# P2a screening analysis (p2a-analysis-v1)

Evaluation: `/home/brand/tensegra-campaign03/results/p2a-screening` (config sha256 `d7c7d551651b403644b9750280331d77ae24cabe5e4f2d74606d4af993965b05`)

Non-sealed, non-confirmatory screening (protocol-P2a.md). Readings are not claims.

## Readings

| Reading | Holds | Detail |
|---|---|---|
| S0 replay (C0 collapses) | yes | boot dev 0.953, C0 final dev 0.000 |
| S1 preservation | yes | success 0.949 vs boot 0.949; no-progress 0.032 vs boot 0.037 |
| S2 improvement | no | utility 0.864 vs boot 0.872; work/success 166.8 vs boot 137.0 |
| S3 c0 | - | final 0.000, deployed 0.900: deployment rule, not stable learning: the deployed checkpoint helps where the final one fails |
| S3 c1 | - | final 0.949, deployed 0.949: final preserved; deployment adds nothing |

## Groups (IID: iid_f0, iid_f2)

| policy/mode | success | utility | work/success | no-progress | idempotent | short-cycle | to cap |
|---|---|---|---|---|---|---|---|
| bootstrap/greedy | 0.949 | 0.872 | 137.0 | 0.037 | 0.036 | 0.037 | 0.023 |
| bootstrap/sampled | 0.896 | 0.815 | 163.4 | 0.013 | 0.002 | 0.013 | 0.014 |
| c0_deployed/greedy | 0.900 | 0.820 | 135.7 | 0.177 | 0.089 | 0.177 | 0.100 |
| c0_deployed/sampled | 0.896 | 0.792 | 210.7 | 0.066 | 0.015 | 0.066 | 0.094 |
| c0_final/greedy | 0.000 | -0.145 | n/a | 0.712 | 0.700 | 0.712 | 1.000 |
| c0_final/sampled | 0.908 | 0.799 | 234.4 | 0.080 | 0.033 | 0.080 | 0.090 |
| c1_deployed/greedy | 0.949 | 0.864 | 166.8 | 0.032 | 0.029 | 0.032 | 0.051 |
| c1_deployed/sampled | 0.916 | 0.827 | 181.9 | 0.031 | 0.005 | 0.031 | 0.066 |
| c1_final/greedy | 0.949 | 0.864 | 166.8 | 0.032 | 0.029 | 0.032 | 0.051 |
| c1_final/sampled | 0.916 | 0.827 | 181.9 | 0.031 | 0.005 | 0.031 | 0.066 |
| dep_reuse/greedy | 0.979 | 0.905 | 126.2 | 0.000 | 0.000 | 0.000 | 0.000 |
| p1_rl/greedy | 0.000 | -0.145 | n/a | 0.712 | 0.700 | 0.712 | 1.000 |
| p1_rl/sampled | 0.908 | 0.799 | 234.4 | 0.080 | 0.033 | 0.080 | 0.090 |

## Per condition

| policy/mode/condition | success | utility | work_per_success | correct_reuse_rate | invalid_reuse_rate | idempotent_repeat_rate | short_cycle_rate | no_progress_rate | no_progress_per_episode | steps_to_cap_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| bootstrap/greedy/events_train_kinds_p1 | 0.965 | 0.887 | 134.1 | 1.000 | 0.000 | 0.035 | 0.036 | 0.036 | 1.082 | 0.016 |
| bootstrap/greedy/foreign4 | 0.934 | 0.863 | 106.0 | 0.996 | 0.000 | 0.069 | 0.072 | 0.072 | 2.109 | 0.043 |
| bootstrap/greedy/iid_f0 | 0.941 | 0.862 | 144.7 | 1.000 | 0.000 | 0.043 | 0.044 | 0.044 | 1.402 | 0.027 |
| bootstrap/greedy/iid_f2 | 0.957 | 0.883 | 129.2 | 0.989 | 0.000 | 0.028 | 0.030 | 0.030 | 0.852 | 0.020 |
| bootstrap/sampled/events_train_kinds_p1 | 0.879 | 0.795 | 167.5 | 0.729 | 0.001 | 0.002 | 0.011 | 0.011 | 0.359 | 0.023 |
| bootstrap/sampled/foreign4 | 0.848 | 0.774 | 129.2 | 0.863 | 0.000 | 0.002 | 0.012 | 0.012 | 0.363 | 0.031 |
| bootstrap/sampled/iid_f0 | 0.898 | 0.813 | 172.2 | 0.500 | 0.002 | 0.002 | 0.016 | 0.016 | 0.547 | 0.023 |
| bootstrap/sampled/iid_f2 | 0.895 | 0.816 | 154.7 | 0.858 | 0.001 | 0.002 | 0.009 | 0.009 | 0.289 | 0.004 |
| c0_deployed/greedy/events_train_kinds_p1 | 0.895 | 0.814 | 128.5 | 1.000 | 0.000 | 0.054 | 0.189 | 0.189 | 6.742 | 0.105 |
| c0_deployed/greedy/foreign4 | 0.867 | 0.793 | 98.1 | 1.000 | 0.000 | 0.073 | 0.248 | 0.248 | 8.781 | 0.133 |
| c0_deployed/greedy/iid_f0 | 0.918 | 0.834 | 150.8 | 1.000 | 0.000 | 0.118 | 0.141 | 0.141 | 4.914 | 0.082 |
| c0_deployed/greedy/iid_f2 | 0.883 | 0.806 | 120.6 | 0.989 | 0.000 | 0.059 | 0.214 | 0.214 | 7.484 | 0.117 |
| c0_deployed/sampled/events_train_kinds_p1 | 0.906 | 0.798 | 207.5 | 0.789 | 0.022 | 0.016 | 0.068 | 0.068 | 3.184 | 0.078 |
| c0_deployed/sampled/foreign4 | 0.891 | 0.794 | 165.6 | 0.889 | 0.022 | 0.014 | 0.068 | 0.068 | 2.984 | 0.102 |
| c0_deployed/sampled/iid_f0 | 0.895 | 0.785 | 229.2 | 0.510 | 0.020 | 0.016 | 0.070 | 0.070 | 3.215 | 0.098 |
| c0_deployed/sampled/iid_f2 | 0.898 | 0.799 | 192.3 | 0.819 | 0.024 | 0.014 | 0.062 | 0.062 | 2.664 | 0.090 |
| c0_final/greedy/events_train_kinds_p1 | 0.000 | -0.146 | n/a | 0.687 | 0.000 | 0.691 | 0.710 | 0.710 | 68.148 | 1.000 |
| c0_final/greedy/foreign4 | 0.000 | -0.143 | n/a | 0.714 | 0.000 | 0.700 | 0.723 | 0.723 | 69.434 | 1.000 |
| c0_final/greedy/iid_f0 | 0.000 | -0.146 | n/a | 0.091 | 0.000 | 0.695 | 0.702 | 0.702 | 67.406 | 1.000 |
| c0_final/greedy/iid_f2 | 0.000 | -0.143 | n/a | 0.665 | 0.000 | 0.705 | 0.722 | 0.722 | 69.305 | 1.000 |
| c0_final/sampled/events_train_kinds_p1 | 0.898 | 0.788 | 222.7 | 0.791 | 0.023 | 0.029 | 0.075 | 0.075 | 3.469 | 0.102 |
| c0_final/sampled/foreign4 | 0.887 | 0.783 | 201.2 | 0.797 | 0.026 | 0.033 | 0.083 | 0.083 | 3.742 | 0.109 |
| c0_final/sampled/iid_f0 | 0.910 | 0.797 | 252.0 | 0.696 | 0.014 | 0.033 | 0.081 | 0.081 | 3.645 | 0.086 |
| c0_final/sampled/iid_f2 | 0.906 | 0.800 | 216.8 | 0.791 | 0.027 | 0.033 | 0.080 | 0.080 | 3.543 | 0.094 |
| c1_deployed/greedy/events_train_kinds_p1 | 0.965 | 0.880 | 161.0 | 1.000 | 0.000 | 0.027 | 0.029 | 0.029 | 0.914 | 0.035 |
| c1_deployed/greedy/foreign4 | 0.938 | 0.863 | 119.8 | 0.996 | 0.000 | 0.053 | 0.062 | 0.062 | 1.938 | 0.062 |
| c1_deployed/greedy/iid_f0 | 0.941 | 0.851 | 181.9 | 1.000 | 0.000 | 0.026 | 0.029 | 0.029 | 1.020 | 0.059 |
| c1_deployed/greedy/iid_f2 | 0.957 | 0.877 | 151.6 | 0.989 | 0.000 | 0.032 | 0.035 | 0.035 | 1.070 | 0.043 |
| c1_deployed/sampled/events_train_kinds_p1 | 0.898 | 0.809 | 172.8 | 0.799 | 0.008 | 0.005 | 0.030 | 0.030 | 1.102 | 0.098 |
| c1_deployed/sampled/foreign4 | 0.879 | 0.796 | 146.9 | 0.837 | 0.006 | 0.005 | 0.038 | 0.038 | 1.367 | 0.109 |
| c1_deployed/sampled/iid_f0 | 0.906 | 0.811 | 203.7 | 0.595 | 0.005 | 0.006 | 0.037 | 0.037 | 1.352 | 0.082 |
| c1_deployed/sampled/iid_f2 | 0.926 | 0.843 | 160.1 | 0.862 | 0.003 | 0.003 | 0.025 | 0.025 | 0.840 | 0.051 |
| c1_final/greedy/events_train_kinds_p1 | 0.965 | 0.880 | 161.0 | 1.000 | 0.000 | 0.027 | 0.029 | 0.029 | 0.914 | 0.035 |
| c1_final/greedy/foreign4 | 0.938 | 0.863 | 119.8 | 0.996 | 0.000 | 0.053 | 0.062 | 0.062 | 1.938 | 0.062 |
| c1_final/greedy/iid_f0 | 0.941 | 0.851 | 181.9 | 1.000 | 0.000 | 0.026 | 0.029 | 0.029 | 1.020 | 0.059 |
| c1_final/greedy/iid_f2 | 0.957 | 0.877 | 151.6 | 0.989 | 0.000 | 0.032 | 0.035 | 0.035 | 1.070 | 0.043 |
| c1_final/sampled/events_train_kinds_p1 | 0.898 | 0.809 | 172.8 | 0.799 | 0.008 | 0.005 | 0.030 | 0.030 | 1.102 | 0.098 |
| c1_final/sampled/foreign4 | 0.879 | 0.796 | 146.9 | 0.837 | 0.006 | 0.005 | 0.038 | 0.038 | 1.367 | 0.109 |
| c1_final/sampled/iid_f0 | 0.906 | 0.811 | 203.7 | 0.595 | 0.005 | 0.006 | 0.037 | 0.037 | 1.352 | 0.082 |
| c1_final/sampled/iid_f2 | 0.926 | 0.843 | 160.1 | 0.862 | 0.003 | 0.003 | 0.025 | 0.025 | 0.840 | 0.051 |
| dep_reuse/greedy/events_train_kinds_p1 | 0.992 | 0.915 | 133.3 | 0.969 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dep_reuse/greedy/foreign4 | 0.969 | 0.903 | 91.8 | 0.940 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dep_reuse/greedy/iid_f0 | 0.980 | 0.904 | 129.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dep_reuse/greedy/iid_f2 | 0.977 | 0.905 | 122.6 | 0.974 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| p1_rl/greedy/events_train_kinds_p1 | 0.000 | -0.146 | n/a | 0.687 | 0.000 | 0.691 | 0.710 | 0.710 | 68.148 | 1.000 |
| p1_rl/greedy/foreign4 | 0.000 | -0.143 | n/a | 0.714 | 0.000 | 0.700 | 0.723 | 0.723 | 69.434 | 1.000 |
| p1_rl/greedy/iid_f0 | 0.000 | -0.146 | n/a | 0.091 | 0.000 | 0.695 | 0.702 | 0.702 | 67.406 | 1.000 |
| p1_rl/greedy/iid_f2 | 0.000 | -0.143 | n/a | 0.665 | 0.000 | 0.705 | 0.722 | 0.722 | 69.305 | 1.000 |
| p1_rl/sampled/events_train_kinds_p1 | 0.898 | 0.788 | 222.7 | 0.791 | 0.023 | 0.029 | 0.075 | 0.075 | 3.469 | 0.102 |
| p1_rl/sampled/foreign4 | 0.887 | 0.783 | 201.2 | 0.797 | 0.026 | 0.033 | 0.083 | 0.083 | 3.742 | 0.109 |
| p1_rl/sampled/iid_f0 | 0.910 | 0.797 | 252.0 | 0.696 | 0.014 | 0.033 | 0.081 | 0.081 | 3.645 | 0.086 |
| p1_rl/sampled/iid_f2 | 0.906 | 0.800 | 216.8 | 0.791 | 0.027 | 0.033 | 0.080 | 0.080 | 3.543 | 0.094 |
