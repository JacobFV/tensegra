# Track C acquisition and main results

## Acquisition (fixed set; not a heldout competence claim)

Three matched initializations of the same 84,708-parameter model (one seed, three availability arms) each trained for1,000 optimizer steps on the same32-example fixed set at retention delays1/2/4. Each arm reached32/32 for every field and for joint reconstruction at all three delays. C1 optimizes only reconstruction: late-query task and validity heads remain untrained diagnostics.

| Arm | Joint at delays1/2/4 | Total seconds including evaluation |
|---|---|---:|
| once | 32/32,32/32,32/32 | 35.02 |
| protected | 32/32,32/32,32/32 | 40.38 |
| gated | 32/32,32/32,32/32 | 41.74 |

All three initial-state hashes match: `5a414d53af8436b195cfd1c4b06b0ad4dbef4d848eb18eb31f4def55d08cc43c`. Fixed acquisition success authorizes independently measured main retention, not C2 or composition. Two-step timing smoke took0.212s including its tiny evaluation.

The compare operand witness initially reversed the existing less-than primitive. Commit07885d3 corrected it before any authorized experiment; a regression executes256 generated witnesses through the existing exact runtime and verifies returned value and type for every primitive.

Acquisition artifact (raw counts/classes/targets, hashes, checkpoints, source snapshot and plots): `results/stage7-c/stage7-c-acquisition-artifacts.tar.gz`; SHA256 `de0602a03ab2b2000bb581e6873a2adf50ee746e57b5ea5577e2b5eff9aa0f17`.

## Frozen generated main

Main config `configs/stage7-c-main.json` is frozen at commitc7f305d (file SHA256 `dc83fc162d699cb7a861bedcc9a7d7aa25089c2cc29a83fcb347c13018534230`). Three initialization seeds × three distinct arms receive matched fresh batches for1,000 steps each. Every metric cohort has512 examples. Three validation and three disjoint untouched test seeds are evaluated with2/8 distractors, clean retention at0/1/2/4/8/16/32 updates, and nine prespecified interventions at16 updates.

The timing-only resource amendment and preserved truncated original run are documented in [stage7-c-resource-amendment.md](stage7-c-resource-amendment.md). All nine main runs finished. Every arm failed GateC; C2 and composition remain blocked.


## Main result: short-delay acquisition failure plus recurrent degradation

Fixed-set reconstruction was learned, but it did not generalize to unseen nonce identities and values at the required level. Even at one recurrent update, test value accuracy is only57.5–58.2% and identity accuracies are about62–69%; type/operation are near ceiling. Additional recurrence lowers performance. These results diagnose substantial shared neural-interface acquisition failure before the retention stress, then further recurrent degradation. They do not establish that exact protected storage is ineffective: all arms use the same existing encoder, which sums value/type/operation and linear projections of two feature_dim32 argument vectors plus provenance into width32, followed by LayerNorm/MLP. Repeated access to that mixed representation does not itself guarantee learnable reconstruction of all identities.

The table pools three initialization seeds × three untouched test data seeds × two distractor conditions: **9,216 predictions per cell**, with paired reuse of examples across models. These are descriptive pooled counts; gates are evaluated separately for every validation cohort, never on this pooled table.

| Arm | Delay | Value | Type | Operation | Arg0 | Arg1 including null | Provenance | Joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| once | 1 | 57.90% | 99.91% | 99.07% | 63.94% | 67.49% | 62.00% | 13.32% |
| once | 16 | 57.00% | 95.71% | 92.06% | 61.51% | 61.01% | 58.24% | 9.30% |
| once | 32 | 51.74% | 90.15% | 83.09% | 58.51% | 56.55% | 55.73% | 5.85% |
| protected | 1 | 58.19% | 99.95% | 99.44% | 66.45% | 68.75% | 63.18% | 15.13% |
| protected | 16 | 54.07% | 96.59% | 92.93% | 62.80% | 64.37% | 60.42% | 10.72% |
| protected | 32 | 49.15% | 92.18% | 85.66% | 59.79% | 60.74% | 57.62% | 7.11% |
| gated | 1 | 57.50% | 99.96% | 99.18% | 65.18% | 67.71% | 62.33% | 13.95% |
| gated | 16 | 54.67% | 96.21% | 92.60% | 62.15% | 61.97% | 58.78% | 9.38% |
| gated | 32 | 48.70% | 90.83% | 83.70% | 58.84% | 57.73% | 56.35% | 5.82% |

Required second-operand identity excludes unary neg rows and has separate exact denominators in `identity-denominators.json`. The recomputed gate includes this stricter check and all six field thresholds (>99% type/operation, >98% value and each identity). Every initialization of every arm fails. Immediate delay0 readout is untrained (no cell decoding pass) and must not be interpreted as a matched learned retention baseline. All retention metrics are before the late query. After-query task and validity heads are untrained C1 diagnostics; their performance is not evidence for or against acquired return-use. Persistent storage is evaluated only as a frozen gated-model lifecycle control, not an independent trained architecture.

Release/overwrite validity is architectural at storage level; no free workspace reset occurs. Stale old-fact reconstruction is recorded separately, with no claim of learned reliable invalidation. Wrong type/provenance/value and drop controls are frozen16-step probes. They cannot establish successful downstream use given the failed C1 gate.

Standalone figures are supplied for every field/mode/validation-test/distractor condition and for frozen interventions (only the measured intervention delays). Failure examples contain raw target/prediction classes, sample index, seed and reproducible batch hash. Example graphics: [test eight-distractor retention](results/stage7-c/retention-test-distractors8.png), [validation eight-distractor retention](results/stage7-c/retention-validation-distractors8.png), [step0 and training losses](results/stage7-c/training-loss.png).

## Audit and artifacts

The main ran9,000 optimizer updates and288,000 presented examples (32,000 per run), with500.70s summed run wall time under two CPU threads. All source/config/initial/checkpoint hashes are in the manifest; paired initialization hashes match within each seed. `stage7-c-audit.py` independently reconstructs every logged six-field and joint count from prediction classes and targets, verifies512 examples per evaluation row and the expected validation matrix, checks required-argument denominators, and recomputes per-arm gates. Twelve focused remote tests passed before main; the root handles final integrated regression.

Durable full raw artifacts, all nine checkpoints, plots and SHA256 inventory: `gb10-direct:~/topoformer-stage7-artifacts/retention/c-main`; exact main sources live in sibling `source-main`. Acquisition artifacts are in sibling `acquisition`, and the resource-stopped original run is isolated in `interrupted-original-main`. Main compact archive contains all raw JSONL, plots, audits and source, omitting checkpoint binaries only. No later architecture/training changes were made from observed results.

Main compact archive SHA256: `4736f2b5958be7d2d5d8b0b50248baae0810986f97a6027f70dd9f7f15d2056a`.
