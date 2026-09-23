# Stage 11 return horizon study

## Status

The bounded acquisition diagnostic passes; the paired fresh-example main comparison has not run. No inherited interface gate has passed, and composition remains blocked.

## Independent acquisition diagnostic

The unchanged 1024-wide factorized/persistent model, initialized from the archived Stage 8 seed-0 checkpoint, can fit 32 complete return events at every covered horizon. Both arms used a fresh AdamW optimizer, 300 updates, the same fixed events, and the registered delay schedules. The short arm first achieved 32/32 six-field joint correctness across delays 0/1/2/4 at the update-60 check. The wide arm first achieved 32/32 across 0/1/2/4/8/16 at the update-120 check. Both retained that complete fit at update 300. This passes the prespecified fixed-set acquisition gate, not a fresh-context or interface gate.

Fresh-context joint correctness remained much lower:

| Arm | Delay 0 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| Short | 357/512 | 320/512 | 308/512 | 292/512 | Not evaluated | Not evaluated |
| Wide | 339/512 | 316/512 | 311/512 | 304/512 | 290/512 | 265/512 |

These are separate fresh events generated from seed 11002001. They demonstrate why fixed-set fitting is a necessary acquisition diagnostic rather than a generalization claim. They do not select or alter the prespecified fresh-example main recipe.

## Compute and provenance

Profile source fc93130 took 1.84 seconds, with measured batch-32 forward/backward/Adam times of 0.007/0.193/0.321 seconds at delays 0/4/16 and 0.121 seconds for delay-32 inference. Peak allocated CUDA memory was 2.16 GB, versus approximately 2.25 GB process RSS. These are different quantities.

The acquisition run froze source 1b2860b, verified the initial checkpoint against its archived SHA-256, and completed in 45.90 seconds. Peak allocated CUDA memory was 2.16 GB and process RSS 2.27 GB. Raw predictions, targets, event identities, per-field training losses, parameter counts and checkpoint hashes are in `research/results/stage11/return-acquisition/`. Checkpoints remain immutable on GB10. Independent audit (7ec2d70) reconstructed all 70 raw rows, both final acquisition gates, fresh counts, source hashes and both output checkpoint hashes without discrepancies.

Based on the observed acquisition runtime, the six 1,000-update fresh-example continuations should take approximately 8–15 minutes including all-field evaluation and checkpoint writes. A conservative 35-minute timeout stays below the 75-minute return-track ceiling. This estimate is published before main launch. Delay coverage changes recurrent compute; optimizer presentations are matched, FLOPs are not.
