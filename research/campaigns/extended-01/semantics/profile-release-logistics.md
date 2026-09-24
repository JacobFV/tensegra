# Post-S12 mechanical profile releases: preparation only

No diagnostic/model profile may start before all six fixed S12 fork outcomes complete. No main budget is allocated. New immutable `*-profile-frozen-v2.json` files grant **no launch permission**: root must separately allocate a cap and explicitly release each job. Frozen files preserve every prepared recipe field except `budget_status`; they add the prepared-file SHA, exact profile dependency/wrapper SHA bindings, and an explicit separate-release requirement. Any source/config revision creates a new output filename; the freezer rejects overwrite. The old prepared files remain unchanged.

The CPU freezer is `python3 src/topoformer/campaign_semantics_profile_freeze.py LANE PREPARED OUTPUT --source-root src/topoformer`, where LANE is `s13`, `s14` or `rename`. Freeze hashes were built from this branch's exact sources plus the already-audited coordinator-only multilingual data/contract dependencies. After integration, source verification must match every bound hash. Stage only the bound source set; do not replace or mutate the ongoing frozen S12 training sources. All new wrappers check the complete S12 matrix's success/endpoint/checkpoint bytes before spawning a child. They retain empty-GPU/prefix checks, external cap, atomically fsynced start/completion receipts and whole-child-process occupancy.

| Profile | Frozen config | Wrapper | Fixed work |
|---|---|---|---|
| S13 English | `campaign-s13-english-profile-frozen-v2.json` | `campaign_semantics_multisurface_launch.py` | Fixed S11 parent,20updates×8; original step0/20 bilingual evaluations and one English TRAIN128 calibration policy |
| S13 mixed | `campaign-s13-mixed-profile-frozen-v2.json` | Same | Same model/AdamW/construction schedule; prespecified renderer phases;20updates×8 |
| S14 motif | `campaign-s14-motif-profile-frozen-v2.json` | `campaign_semantics_motif_launch.py` | First8 fixed arity3 rows for each of all6forks; inherited thresholds/control |
| S12 rename | `campaign-s12-renamed-profile-frozen-v2.json` | `campaign_semantics_confirmation_rename_profile_launch.py` | First8 fixed renamed rows for each of all6forks; inherited thresholds; timing/digests only |

S13 profiles are a pair: retain both regardless of results, then use elapsed time/resource usage alone to inform later budget allocation. No main recipe, checkpoint, language subset or policy is selected from profile outcomes. The original S13 trainer is unchanged. The S14 evaluator is unchanged; its wrapper adds source/matrix profile guards. Rename uses a separately versioned mechanical profiler; the already-frozen all1,024 main evaluator at4a05aba remains unchanged. The rename profiler preflights all six sources/checkpoints/primary population order before any model and never refits thresholds. It archives timing plus output digests, not model-selection statistics.

Remote byte-only staging check: S13audit/train/dev, fixed S11 checkpoint and renamed cache all exist with exact registered hashes. S14cache is **not yet staged**: copy the immutable `s14-arity3-cache/diagnostic.jsonl.gz` to remote `data/s14-arity3-v1/diagnostic.jsonl.gz` and verifySHA8b75c555e33ea69e3cc15a1ed7a922acd5b631294a0b1c57ccd9429711eea4d5. New profile source modules/wrappers/configs must also be staged after review, before release. No production remote source was mutated in this preparation.

After root's specific release, use the same detached command form from remote `~/topoformer-campaign01-semantics`:

```bash
nohup ~/topoformer-stage8-cuda/bin/python -u src/topoformer/WRAPPER.py configs/FROZEN_CONFIG.json --cap "$ROOT_ALLOCATED_PROFILE_CAP" --python /home/brandonin/topoformer-stage8-cuda/bin/python --prefix UNIQUE_PROFILE_PREFIX > UNIQUE_PROFILE_PREFIX.launch.out 2>&1 < /dev/null &
echo $! > UNIQUE_PROFILE_PREFIX.launch.pid
```

The four jobs receive distinct prefixes, and each paired S13 arm needs its own release. Poll existing jobs only. Report GPUFREE and exact full occupancy before interpretation. Record failed/partial profiles and do not silently rerun or promote. Main freezes/budgets follow a separate root decision.

Validation:2stdlib CPU tests pass for exact immutable conversion, source mutation rejection and all-six completion/corruption/missing-file rejection; existing success/timeout/duplicate receipt test also passes. New rename profile module imports successfully with CUDA hidden, without model construction or forwarding. Syntax passes. Independent review is requested for the additive freezer, S13launcher, rename mechanical profiler and S14wrapper delta before GPUrelease.

Accounting correction before any release: frozen-v1 files remain immutable historical preparation but are superseded by frozen-v2 bindings. Current wrappers persist the start receipt before CPU source/matrix preflight, include that time in reported occupancy, subtract it from the child timeout, and retain completion/error receipts even when preflight rejects execution. A dedicated CPU test covers all three affected wrappers with synthetic4s preflight/6s remaining timeout and failed-preflight/no-child cases; it passes in0.075s. No evaluator/trainer or active S12 wrapper changed.

## Staged post-matrix readiness

All four frozen-v2 profiles are now staged on `gb10-direct` in the semantic checkout, including the audited S14cache. Exact source/config/cache/parent bytes pass; all six fixed S12 checkpoints/endpoint evaluations now pass the completion guard. No target/model imports or inference occurred in this staging verification. Remote output directories and launch prefixes are unused. See `postmatrix-profile-staging.json` for complete source/config/wrapper/checkpoint hashes and the checked timestamp.

Proposed cap is60whole-process seconds **per profile**, subject to root's explicit release. The all-six checkpoint-hash preflight measured5.2264CPU seconds and is included in the wrapper clock. S13 adds20updates plus two1152-example bilingual evaluation/calibration blocks and checkpoint writes; prior S12same-size evaluation blocks took about15–17s, suggesting roughly40–55s including this preflight and I/O. This is an estimate, not a measured S13forward result; preserve timeout outcomes. S14/rename use48forwards but repeatedly validate/load six checkpoints;60s leaves room for those I/O costs. No main cap is proposed or allocated from unmeasured profiles.

Use distinct unused prefixes `s13-english-profile-v2`, `s13-mixed-profile-v2`, `s14-motif-profile-v2`, and `s12-renamed-profile-v2`. Their wrapper/config hashes are recorded in the staging receipt. Each needs a separate root release; current GPU work owned by another queue entry is unaffected.
