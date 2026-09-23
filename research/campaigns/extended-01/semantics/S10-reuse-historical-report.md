# S10 reuse: historical matched comparators

Prospective protocol `8042d2f`, unchanged audited decoder `2085053`. S11/S09 new outputs are still pending. The original S04 **u24576** checkpoint is fixed by matched exposure (196,608 presentations), not chosen using grammar-repair outcomes. All results are secondary supplied-schema diagnostics, not replacements for primary model gates.

| Checkpoint | Split | Policy | Baseline | Schema | Bookkeeping | Combined | Combined typed F1 | Combined ordered F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S04_mid | train | raw | 0/128 | 0 | 1 | 4 | 0.94743 | 0.86752 |
| S04_mid | train | calibrated | 0/128 | 0 | 11 | 40 | 0.98479 | 0.96740 |
| S04_mid | development | raw | 0/512 | 0 | 3 | 25 | 0.94520 | 0.86809 |
| S04_mid | development | calibrated | 0/512 | 1 | 53 | 183 | 0.98014 | 0.96199 |
| S01 | train | raw | 0/128 | 0 | 0 | 0 | 0.84177 | 0.70801 |
| S01 | train | calibrated | 0/128 | 0 | 0 | 5 | 0.91808 | 0.87885 |
| S01 | development | raw | 0/512 | 0 | 0 | 0 | 0.84455 | 0.70675 |
| S01 | development | calibrated | 0/512 | 0 | 2 | 33 | 0.92264 | 0.88152 |

Every one of the 5,120 S01 per-graph records exactly reproduces the earlier S10 diagnostic, including metrics, prerequisites and complete tensor hashes. CPU wall 48.16 seconds, RSS 897,816 KiB, zero GPU. Independent raw reconstruction pending. Learned node attributes and structural associations remain frozen; bookkeeping/type rules are supplied. Cross-checkpoint paired outcomes await the registered new archives. No best-checkpoint substitution.
