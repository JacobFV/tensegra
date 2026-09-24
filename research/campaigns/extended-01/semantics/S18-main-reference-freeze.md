# Prospective S18 development release freeze

Root allocated main4000 and reference240 hard whole-process seconds; this is a budget freeze, not a GPU release. Commit00a79951 introduces both immutable configurations and profile-bound evidence (cost proposal, executed manifest and receipt). Exact source hash maps, original reference bindings, model/input/calibration policies match the executed profile-v2 without scientific changes.

- Main config SHA256: `4ffecfea1a6aa3b968a9568b3545c8fba0fe43044e248e821fd6209eead44fdf`. Both fixed new arms,4096updates, all four curves, both policies, actual TRAIN128 and full four-cell DEV remain.
- Reference config SHA256: `944b342c8a4ae58c3ebce6ba9a4b143f02d9d033eac17875ce0d4557c768a6d6`. Frozen original checkpoints at added0/1024/2048 only; no optimizer/training and no final checkpoint inference. Final matched TRAIN/DEV remain reused from S17.

Both are staged under `/home/brandonin/topoformer-campaign01-semantics/s18-source-00a79951`, with every directory/file read-only (no write-mode entries). Source/config/archive/utility hashes and exact argv appear in `S18-main-outer-command.json` and `S18-reference-outer-command.json`, also staged as `s18-main-outer-command-v1.json` and `s18-reference-outer-command-v1.json` beside the source snapshots. GNUtime is outside GNUtimeout's default process-group KILL deadline at4000 or240, itself outside the audited launcher at the same cap. Child PYTHONPATH resolves to the launcher source. No grace interval, automatic retry or cap increase is included.

Separate root releases and reviewer approval remain required for each job. Report GPUFREE/full outer receipt before outcome analysis; preserve incomplete artifacts on any failure. Combined4240seconds stays within the allocated development envelope. Analysis must use these frozen hashes and retain the existing primary endpoint gates, all comparisons, calibration-fit overlap and depth-shift caveats.
