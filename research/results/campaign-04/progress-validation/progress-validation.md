# Progress diagnostic v1 vs P2a registered no-progress (progress-validate-v1)

Evaluation: `/home/brand/tensegra-campaign03/results/p2a-screening`; limit None; window 6.

- decisions: 602220; per-step agreement 0.9922; both 164701, P2a-only 50, v1-only 4628
- replay feedback mismatches: 0

- registered P2a rates reproduced (all rows, registered code): 52/52 cells

Columns: registered = p2a-analysis.json; all-rows = registered code on every row; the remaining columns are on the replayed subset (limit).

| cell | replayed | P2a registered | P2a all-rows | P2a subset | v1 subset | P2a eps | v1 eps | agreement |
|---|---|---|---|---|---|---|---|---|
| bootstrap/greedy/events_train_kinds_p1 | 256 | 0.0362 | 0.0362 | 0.0362 | 0.0374 | 0.0312 | 0.0430 | 0.9988 |
| bootstrap/greedy/foreign4 | 256 | 0.0718 | 0.0718 | 0.0718 | 0.0765 | 0.0547 | 0.0742 | 0.9953 |
| bootstrap/greedy/iid_f0 | 256 | 0.0442 | 0.0442 | 0.0442 | 0.0472 | 0.0273 | 0.0664 | 0.9970 |
| bootstrap/greedy/iid_f2 | 256 | 0.0295 | 0.0295 | 0.0295 | 0.0324 | 0.0273 | 0.0508 | 0.9972 |
| bootstrap/sampled/events_train_kinds_p1 | 256 | 0.0109 | 0.0109 | 0.0109 | 0.0130 | 0.1211 | 0.1328 | 0.9977 |
| bootstrap/sampled/foreign4 | 256 | 0.0118 | 0.0118 | 0.0118 | 0.0182 | 0.1094 | 0.2070 | 0.9936 |
| bootstrap/sampled/iid_f0 | 256 | 0.0163 | 0.0163 | 0.0163 | 0.0179 | 0.0977 | 0.1016 | 0.9979 |
| bootstrap/sampled/iid_f2 | 256 | 0.0094 | 0.0094 | 0.0094 | 0.0122 | 0.1211 | 0.1445 | 0.9972 |
| c0_deployed/greedy/events_train_kinds_p1 | 256 | 0.1892 | 0.1892 | 0.1892 | 0.1954 | 0.1055 | 0.2070 | 0.9938 |
| c0_deployed/greedy/foreign4 | 256 | 0.2485 | 0.2485 | 0.2485 | 0.2579 | 0.1328 | 0.2812 | 0.9906 |
| c0_deployed/greedy/iid_f0 | 256 | 0.1410 | 0.1410 | 0.1410 | 0.1497 | 0.0820 | 0.1133 | 0.9913 |
| c0_deployed/greedy/iid_f2 | 256 | 0.2136 | 0.2136 | 0.2136 | 0.2193 | 0.1172 | 0.1836 | 0.9943 |
| c0_deployed/sampled/events_train_kinds_p1 | 256 | 0.0681 | 0.0681 | 0.0681 | 0.0761 | 0.7930 | 0.8203 | 0.9920 |
| c0_deployed/sampled/foreign4 | 256 | 0.0678 | 0.0678 | 0.0678 | 0.0791 | 0.7344 | 0.7891 | 0.9884 |
| c0_deployed/sampled/iid_f0 | 256 | 0.0705 | 0.0705 | 0.0705 | 0.0796 | 0.7617 | 0.7695 | 0.9907 |
| c0_deployed/sampled/iid_f2 | 256 | 0.0620 | 0.0620 | 0.0620 | 0.0688 | 0.7422 | 0.7578 | 0.9932 |
| c0_final/greedy/events_train_kinds_p1 | 256 | 0.7099 | 0.7099 | 0.7099 | 0.7170 | 1.0000 | 1.0000 | 0.9929 |
| c0_final/greedy/foreign4 | 256 | 0.7233 | 0.7233 | 0.7233 | 0.7406 | 0.9844 | 1.0000 | 0.9826 |
| c0_final/greedy/iid_f0 | 256 | 0.7021 | 0.7021 | 0.7021 | 0.7069 | 1.0000 | 1.0000 | 0.9952 |
| c0_final/greedy/iid_f2 | 256 | 0.7219 | 0.7219 | 0.7219 | 0.7304 | 0.9922 | 1.0000 | 0.9915 |
| c0_final/sampled/events_train_kinds_p1 | 256 | 0.0748 | 0.0748 | 0.0748 | 0.0866 | 0.8828 | 0.8984 | 0.9879 |
| c0_final/sampled/foreign4 | 256 | 0.0830 | 0.0830 | 0.0830 | 0.1013 | 0.8633 | 0.9141 | 0.9816 |
| c0_final/sampled/iid_f0 | 256 | 0.0807 | 0.0807 | 0.0807 | 0.0936 | 0.8594 | 0.8633 | 0.9868 |
| c0_final/sampled/iid_f2 | 256 | 0.0798 | 0.0798 | 0.0798 | 0.0906 | 0.8750 | 0.8906 | 0.9890 |
| c1_deployed/greedy/events_train_kinds_p1 | 256 | 0.0288 | 0.0288 | 0.0288 | 0.0310 | 0.0508 | 0.0508 | 0.9978 |
| c1_deployed/greedy/foreign4 | 256 | 0.0624 | 0.0624 | 0.0624 | 0.0668 | 0.0625 | 0.0859 | 0.9956 |
| c1_deployed/greedy/iid_f0 | 256 | 0.0291 | 0.0291 | 0.0291 | 0.0326 | 0.0430 | 0.0625 | 0.9965 |
| c1_deployed/greedy/iid_f2 | 256 | 0.0347 | 0.0347 | 0.0347 | 0.0396 | 0.0430 | 0.0547 | 0.9951 |
| c1_deployed/sampled/events_train_kinds_p1 | 256 | 0.0302 | 0.0302 | 0.0302 | 0.0339 | 0.1719 | 0.1758 | 0.9954 |
| c1_deployed/sampled/foreign4 | 256 | 0.0381 | 0.0381 | 0.0381 | 0.0460 | 0.2383 | 0.2891 | 0.9905 |
| c1_deployed/sampled/iid_f0 | 256 | 0.0366 | 0.0366 | 0.0366 | 0.0411 | 0.1445 | 0.1445 | 0.9953 |
| c1_deployed/sampled/iid_f2 | 256 | 0.0255 | 0.0255 | 0.0255 | 0.0291 | 0.1523 | 0.1641 | 0.9951 |
| c1_final/greedy/events_train_kinds_p1 | 256 | 0.0288 | 0.0288 | 0.0288 | 0.0310 | 0.0508 | 0.0508 | 0.9978 |
| c1_final/greedy/foreign4 | 256 | 0.0624 | 0.0624 | 0.0624 | 0.0668 | 0.0625 | 0.0859 | 0.9956 |
| c1_final/greedy/iid_f0 | 256 | 0.0291 | 0.0291 | 0.0291 | 0.0326 | 0.0430 | 0.0625 | 0.9965 |
| c1_final/greedy/iid_f2 | 256 | 0.0347 | 0.0347 | 0.0347 | 0.0396 | 0.0430 | 0.0547 | 0.9951 |
| c1_final/sampled/events_train_kinds_p1 | 256 | 0.0302 | 0.0302 | 0.0302 | 0.0339 | 0.1719 | 0.1758 | 0.9954 |
| c1_final/sampled/foreign4 | 256 | 0.0381 | 0.0381 | 0.0381 | 0.0460 | 0.2383 | 0.2891 | 0.9905 |
| c1_final/sampled/iid_f0 | 256 | 0.0366 | 0.0366 | 0.0366 | 0.0411 | 0.1445 | 0.1445 | 0.9953 |
| c1_final/sampled/iid_f2 | 256 | 0.0255 | 0.0255 | 0.0255 | 0.0291 | 0.1523 | 0.1641 | 0.9951 |
| dep_reuse/greedy/events_train_kinds_p1 | 256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| dep_reuse/greedy/foreign4 | 256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| dep_reuse/greedy/iid_f0 | 256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| dep_reuse/greedy/iid_f2 | 256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| p1_rl/greedy/events_train_kinds_p1 | 256 | 0.7099 | 0.7099 | 0.7099 | 0.7170 | 1.0000 | 1.0000 | 0.9929 |
| p1_rl/greedy/foreign4 | 256 | 0.7233 | 0.7233 | 0.7233 | 0.7406 | 0.9844 | 1.0000 | 0.9826 |
| p1_rl/greedy/iid_f0 | 256 | 0.7021 | 0.7021 | 0.7021 | 0.7069 | 1.0000 | 1.0000 | 0.9952 |
| p1_rl/greedy/iid_f2 | 256 | 0.7219 | 0.7219 | 0.7219 | 0.7304 | 0.9922 | 1.0000 | 0.9915 |
| p1_rl/sampled/events_train_kinds_p1 | 256 | 0.0748 | 0.0748 | 0.0748 | 0.0866 | 0.8828 | 0.8984 | 0.9879 |
| p1_rl/sampled/foreign4 | 256 | 0.0830 | 0.0830 | 0.0830 | 0.1013 | 0.8633 | 0.9141 | 0.9816 |
| p1_rl/sampled/iid_f0 | 256 | 0.0807 | 0.0807 | 0.0807 | 0.0936 | 0.8594 | 0.8633 | 0.9868 |
| p1_rl/sampled/iid_f2 | 256 | 0.0798 | 0.0798 | 0.0798 | 0.0906 | 0.8750 | 0.8906 | 0.9890 |

## Disagreement categories (steps)

- 2212: v1-only: repeated call with unchanged inputs (records content-keyed; P2a: calls break cycles)
- 1731: v1-only: retrieval of an already retrieved record content
- 641: v1-only: cycle through a repeated call/content-equal record (add_constraint)
- 50: P2a-only: triggered revision exempt from cycles (v1 precedence)
- 44: v1-only: cycle through a repeated call/content-equal record (use_return)

## References (v1 no-progress on the same worlds)

| reference/condition | episodes | decisions | no-progress |
|---|---|---|---|
| dep_recompute/events_train_kinds_p1 | 256 | 8069 | 0 |
| dep_recompute/foreign4 | 256 | 7918 | 0 |
| dep_recompute/iid_f0 | 256 | 7828 | 0 |
| dep_recompute/iid_f2 | 256 | 7877 | 0 |
| dep_reuse/events_train_kinds_p1 | 256 | 7323 | 0 |
| dep_reuse/foreign4 | 256 | 6826 | 0 |
| dep_reuse/iid_f0 | 256 | 7583 | 0 |
| dep_reuse/iid_f2 | 256 | 7051 | 0 |
