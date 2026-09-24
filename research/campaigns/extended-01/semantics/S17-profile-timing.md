# S17 profile archive and main timing proposal

The released paired profile completed with exit0 and GPUFREE: outer25.65s (0.01s precision; charge25.66s), inner25.60548456106335s. Archive: `research/results/campaign-01/semantics/s17-calibration-profile-v1`. Both fixed endpoints ran128actualTRAIN calibration plus64DEV each; no training or selection occurred. Artifact and calibration NPZ hashes match the manifest. Independent numerical review is pending.

Measured additive paired phases (seconds):

| Phase | Control | Mixed | Main treatment |
|---|---:|---:|---|
| Setup | 4.129949 | 4.128791 | Fixed |
| TRAIN128 forward, threshold fit, TRAIN packing | 4.335783 | 3.222655 | Fixed128 |
| DEV forward, metrics, packing | 0.926018 | 0.960572 | 32× for2048/64 |
| Export and summary | 1.787508 | 1.619497 | Conservatively32× |
| Verification | 0.613837 | 0.371321 | Conservatively32× |

The remaining outer/setup-independent residual is 3.554069s. Keeping setup, calibration, and residual fixed, scaling DEV/export/verification32× gives **220.291s**. Export contains fixed calibration plus variable DEV payload, so scaling all export32× is conservative; verification includes fixed model/hash checks, also conservatively scaled. This is a timing projection, not a guaranteed bound; first64 balanced DEV may vary in cost from the full population.

Propose one **300second whole-process cap** for the paired main, with GNUtime outside GNUtimeout and unchanged scientific source65e43fa7/launcher99589a36. Main uses the already-prepared128TRAIN policy per endpoint and all2048DEV, same original endpoints, output `results/s17-train-calibration-main-v1`. No policy, threshold algorithm, endpoint, selection, or failed S15 gate changes. Main freeze and GPU launch require separate root allocation/release after profile review.
