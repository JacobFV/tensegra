# Frozen Stage 6 cost decomposition

This bounded diagnostic profiles the final local checkpoints from all three seeds at depths 4 and 32, using two paired examples per seed (24 total rollouts). It ran with two CPU threads against frozen source `c258785ee6b76f2d492f722e93a1b1539ac6de76`; no training or frozen-source edits occurred. Raw source, checkpoint, input and script hashes are in `profile.json`.

Times below are mean milliseconds per rollout across six examples. Categories are disjoint. “Other” includes input construction, neural initialization, output/audit adapters, auxiliary oracle diagnostics and instrumentation; it is not symbolic execution.

| Mode | Depth | Steps | Neural cell wall | Exact session wall | Event adapter wall | Other wall | Total process CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| free | 4 | 2.0 | 31.333 | 0.024 | 0.009 | 9.198 | 47.430 |
| free | 32 | 2.0 | 10.469 | 0.028 | 0.010 | 58.717 | 79.187 |
| oracle_minimal | 4 | 8.0 | 26.864 | 0.157 | 2.941 | 31.377 | 88.220 |
| oracle_minimal | 32 | 36.0 | 181.253 | 1.470 | 32.638 | 823.459 | 1244.393 |

Free policy halted at two microsteps in every sampled rollout and obtained neither the exact numeric result nor the complete task answer. Oracle-minimal execution obtained the exact numeric result in all 12 rollouts, but complete learned task output remained wrong in all 12. Consequently the low free-policy cost reflects premature stopping rather than successful computation.

The exact runtime itself used approximately 0.16 ms at depth 4 and 1.47 ms at depth 32 under the privileged minimal trace. Learned event encoding/injection and neural recurrence cost more; oracle auxiliary audits and Python/tensor adapters dominate the remaining depth-32 time. These figures measure this implementation and diagnostic workload, not a general neural-versus-symbolic efficiency claim.

Concurrent main training was running on the same host. Wall time includes scheduling delays; process CPU time includes Torch worker-thread CPU. No warmup was excluded, so cold-start cost remains (especially the first shallow free rollout). Repeated timer calls measured approximately 0.52 microseconds per pair of start/end wall/CPU readings; this calibration does not remove wrapper, dictionary or tensor-audit overhead. Oracle rows compute privileged auxiliary losses/audits and should not be compared to free rows as equal-work throughput.

Verification: all 24 category sums equal total wall and process CPU time; every category is nonnegative; neural call counts equal recorded microsteps; every rollout satisfies the post-event recurrent-pass invariant; downloaded script hash matches the executed script; checkpoint hashes matched run manifests.
