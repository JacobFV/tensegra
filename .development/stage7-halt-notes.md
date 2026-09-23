# Stage7 independent halting diagnostic

`python -m topoformer.halting_study --config configs/stage7-halt-smoke.json --output NEW_DIRECTORY` explicitly starts the runner. Smoke/acquisition budgets are provisional; root must authorize training and freeze a main budget separately. No training was performed while implementing this track. Unit integration uses zero optimizer steps only.

## Task and supplied priors

Each episode asks for XOR of two certified shares belonging to the queried name. Public rows encode seven name bits, share role, certification, and binary value. Every current snapshot has four shuffled, visible rows: certified first share, second share (or an unrelated replacement), an uncertified rumor, and a certified wrong-name share. Before sufficient arrival the rumor uses the queried name; after arrival its name changes to the distractor name, balancing the arrival of the genuine queried share. Role, certification, and name marginals remain constant: sufficiency requires their conjunction. Distractor values are sampled independently of the missing true share. Masks remain constant; neither missingness nor fixed row position reveals sufficiency. Arrived evidence remains publicly available: this diagnostic does **not** establish learned long-term retention. Certification, share role, key coding, and persistent current evidence are explicit strong representational priors.

Solvable training arrival times are 2/4/6; validation and test use 3/5/7. Training, validation, and test name pools are disjoint (0–31/32–63/64–95). A quarter of episodes never provide a complete pair and require reject class 2 at the observable step-8 deadline. Before that deadline, absence alone cannot establish no solution. The generator's answer, sufficiency step, and rejection status are private loss/evaluation labels; the actor receives only current context, evidence rows/masks, and recurrent workspace. No symbolic runtime or proposal model is invoked.

The actor reuses `ThinkingModel.step`, including its four recurrent blocks. Task CE is supervised only after sufficient evidence or at the rejection deadline; halt BCE supervises continue versus emit across the eight updates. No intermediate gold answer or gold sufficiency is inserted into latent state.

## Frozen controls, metrics, and gate

All controls use exactly the same trained parameters and public observations, changing only the stopping rule: learned logit, minimum step 1, fixed step 8, private oracle sufficiency, fixed soft timing bias alone, and learned logit with the existing cell's soft bias. Oracle stopping is explicitly privileged. Evaluations execute only active examples at each update, so reported updates count actual four-block transitions rather than selecting a prediction from a fully executed rollout.

Reports contain task accuracy, exact halt, joint correct stop/reject, no-solution rejection, premature rate, extra steps, raw per-example results, stop histograms by evidence arrival, and task-minus-compute utility at costs 0/.01/.05/.1. The halt gate requires every validation seed to exceed .95 joint stop/task accuracy and .95 no-solution rejection accuracy, stay strictly below .01 premature rate, exceed .95 joint accuracy at at least three sufficient arrival times, and exceed minimum-stop task accuracy by strictly more than .20. Halting competence alone never authorizes composition. Pilot/acquisition runs evaluate validation only; frozen main runs inspect test once at the final budget.

Each run saves configuration, current source hashes (including reused thinking cell), Torch version, dataset/initial/checkpoint files and hashes, seed, parameter count, optimizer exposure, step-0 and periodic evaluation, every task/halt loss, elapsed time, and a validation gate report. Output directories must be new; accidental overwrite fails.

## Verification

Tests were written before implementation and failed remotely on the missing diagnostic. Follow-up leakage and pilot-test-isolation regressions also failed before their fixes. Eight focused tests passed on `gb10-direct`, using the existing Python environment but source copied into a fresh `/tmp/stage7-halt-red.eJxRT5` directory with at most two CPU threads. No previous frozen checkout was modified. Full-suite result is reported separately after completion.

Full remote bare `pytest -q`: **487 passed, 6 subtests passed** in 15.54 seconds. Initial incomplete snapshot collection lacked `scripts/audit_binding_checkpoints.py` and concurrent workers' source modules; after refreshing those files, two old study tests failed solely because the copied directory lacked Git metadata (`test_smoke_resume_and_reject_changed_configuration`, `test_resume_checks_config_and_source`). Initializing a repository and committing the isolated temporary snapshot resolved those provenance-environment failures; no existing checkout was changed. The passing snapshot includes concurrent tracks as they existed when copied; root must revalidate the final integrated tree if those tracks change.

## Authorized acquisition and main freeze

Following explicit root authorization, smoke completed 2 updates in 0.695s. Acquisition used seeds 0/1/2, 200 updates, batch 24, width 16, 96 fixed training and 96 validation examples (24 per arrival/reject stratum), with no test inspection. Seed elapsed times were 15.32/17.19/17.46s. Learned joint stop/task accuracies were 9.38/2.08/3.13%, premature rates 53.13/48.96/63.54%, task accuracies 38.54/29.17/35.42%, and rejection 0/24 for every seed. Minimum task accuracies were 35.42/32.29/30.21%. At private oracle timing, learned training-task accuracy was only 50/50/61.46%; basic task acquisition is incomplete. Every prespecified gate failed. Raw checkpoints/data/per-step losses/curves/output records and source hashes are preserved remotely in `/tmp/stage7-halt-probes.WDQCn9` and locally in ignored `results/stage7-halt-probes/`.

Before main launch, root explicitly authorized and froze `configs/stage7-halt-main.json`: three seeds, width32, 2000 updates, batch32, 2048 fixed training examples, 512 validation and 512 test examples, evaluation every500. Test is inspected only at the final budget. No architecture changes are permitted after launch; failures remain failures and never trigger composition.

The generator now supports an independently seeded missing-share value for causal counterfactual controls. A regression test changes that value while holding every insufficient public prefix (and entire no-solution sequence) identical. Ten focused remote tests pass. A separate exact public-evidence XOR rule is logged as an observability/task upper bound: it performs zero neural updates and is explicitly not a parameter-matched learned baseline. The existing private oracle-timing control still uses learned output heads, separating perception/task failure from learned stopping failure.

## Frozen main outcome: joint gate failed

The prespecified all-seed **joint timing/task competence gate failed**: seeds0 and2 passed, seed1 failed. No thresholds, source, architecture, or budget changed after main launch; no runtime composition was run. Each seed used73,017 parameters and64,000 optimizer examples (2000×32), against2048 fixed training examples. Elapsed times were155.64/166.10/174.50s (496.24s total). Each validation/test split has512 episodes:128 each for arrivals3/5/7 and no solution.

Pure halt exactness and joint exact-stop-plus-correct-task are distinct. The gate remains the original stricter joint criterion; its failure does not erase timing learned by seed1 on validation.

| Split | Seed | Task % | Pure exact halt % | Joint % | Premature count | Correct reject | Mean updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0 | 99.80 | 99.02 | 99.02 | 0/512 | 128/128 | 5.760 |
| validation | 1 | 92.97 | 100.00 | 92.97 | 0/512 | 128/128 | 5.750 |
| validation | 2 | 99.80 | 99.22 | 99.22 | 4/512 | 127/128 | 5.717 |
| test | 0 | 100.00 | 100.00 | 100.00 | 0/512 | 128/128 | 5.750 |
| test | 1 | 85.74 | 93.95 | 83.20 | 11/512 | 112/128 | 5.764 |
| test | 2 | 99.22 | 99.22 | 99.22 | 0/512 | 128/128 | 5.758 |

Seed1 validation halted exactly at every arrival (128/128 for each arrival3/5/7 and rejection), but XOR task mistakes reduced joint success to476/512. Its final test independently showed timing/rejection weaknesses:11 premature episodes (2 late-arrival,9 no-solution),112/128 correct rejections, and481/512 exact stops. Seeds0/2 do not compensate for that all-seed failure.

Matched frozen controls use the same73,017 parameters and public evidence; only stopping changes. Values below are task accuracy / mean neural updates.

| Split / seed | Learned | Minimum | Fixed8 | Oracle timing, learned output | Timing bias only | Learned + bias |
|---|---:|---:|---:|---:|---:|---:|
| validation / 0 | 99.80% / 5.760 | 38.87% / 1.000 | 99.80% / 8.000 | 100.00% / 5.750 | 63.67% / 6.000 | 99.80% / 5.760 |
| validation / 1 | 92.97% / 5.750 | 38.48% / 1.000 | 93.36% / 8.000 | 92.97% / 5.750 | 59.57% / 6.000 | 92.97% / 5.750 |
| validation / 2 | 99.80% / 5.717 | 33.20% / 1.000 | 99.80% / 8.000 | 99.61% / 5.750 | 61.91% / 6.000 | 99.80% / 5.717 |
| test / 0 | 100.00% / 5.750 | 37.50% / 1.000 | 100.00% / 8.000 | 100.00% / 5.750 | 62.89% / 6.000 | 100.00% / 5.750 |
| test / 1 | 85.74% / 5.764 | 36.52% / 1.000 | 90.43% / 8.000 | 85.16% / 5.750 | 57.81% / 6.000 | 85.74% / 5.764 |
| test / 2 | 99.22% / 5.758 | 38.09% / 1.000 | 98.24% / 8.000 | 100.00% / 5.750 | 62.50% / 6.000 | 99.22% / 5.758 |

The exact public-evidence XOR rule achieved512/512 task answers,512/512 exact observation stops, and128/128 rejections on every validation and test split. It is a supplied rule upper bound with zero neural updates, **not** a matched learned baseline. The learned-output/private-oracle-timing control exposes perception/task error: seed1 remained92.97% validation and85.16% test even when given perfect stop timing.

[Actual validation stopping distributions](stage7-halt-main-arrivals.png) show counts and frequencies by sufficient arrival/no-solution. [Main report](stage7-halt-main-report.json) includes pure exact per-arrival counts separately from joint accuracy, all controls, costs, step0/curves, loss endpoints, denominators, and hashes. [Acquisition report](stage7-halt-acquisition-report.json) and its separate plot preserve the earlier failure. Compressed raw metric archives include all per-example records and per-loss curves for each phase.

Durable remote artifacts on `gb10-direct`:

- `/home/brandonin/topoformer-stage7-artifacts/halting/probes-1ee0a6a/` —153 files, separate smoke/acquisition outputs and source.
- `/home/brandonin/topoformer-stage7-artifacts/halting/main-b49d53d/` —148 files, main checkpoint/data/source/raw curves/plot.
- Each directory contains `SHA256SUMS.json` covering every preserved file. Main source hash: `c9ad90611739611149f7f5bb4d2bc7f7ae74197c19319738ca9b451323b270b0`; source/config freeze commit `b49d53d`.
- Committed raw archive SHA256: main `9d35a03e1a08867c4e105e0e6f055d398c0eed5bd44ade34b7a32b2751804084`; acquisition `d88197adbefb9ebb7e4e74cf2d32bc415f284be9f5dc0a7b709c3c0e602ee165`.
