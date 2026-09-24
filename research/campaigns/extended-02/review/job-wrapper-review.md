# Job wrapper independent review

Reviewed original `ae106c9d` and repair `bed4795a`. No GPU use. Original/repair sources are SHA-bound in compact receipts; harness is `audit_job_wrapper.py`.

**Repair passes the exercised subprocess lifecycle/accounting contract.** Linux `/proc/PID/stat` offsets are correct after splitting after the command's closing parenthesis: process group is index2; user/system CPU indices11/12; reaped child CPU13/14. Live children have their own rows; reaped child time propagates into the parent. The scan is not atomic and its peak is an approximate cap monitor, not the authoritative accounting receipt.

Original reproduced two failures: a parent exiting before a normal child leaves that child alive, and timeout SIGTERM kills the leader while a TERM-ignoring child survives. Both lose remaining descendant CPU from the final receipt. The reviewer killed these explicitly bounded sleeper processes; no work remains running.

The root repair sets Linux child-subreaper status, unconditionally kills the remaining process group and reaps adopted descendants. Repeating both fixtures leaves no surviving child. Normal parent0.15CPU plus reaped child0.25CPU produces0.487736 inclusive seconds including startup/wrapper scanning. A0.3wall cap produces a wall-cap receipt at0.485736s. A two-worker busy loop at0.35aggregate CPU cap terminates for CPU cap, sampled0.47 and final0.587703core-seconds. These exceed nominal thresholds because monitoring is sampled; final actual consumption, not cap values, must be charged.

Remaining qualifications:

- The five-second TERM grace can exceed a tight resource ceiling if the leader ignores TERM. Reserve overshoot in campaign scheduling or use immediate SIGKILL for budget exhaustion. Per-process RLIMIT_CPU is not an aggregate guarantee.
- No child may change session/process group. This is a declared controlled-harness contract, not an adversarial process sandbox. General untrusted child programs need cgroup containment or equivalent. `waitpid(-1)` also assumes one dedicated wrapper without unrelated owned children.
- RUSAGE_CHILDREN is cumulative, but before/after CPU differencing avoids recharging previous jobs. `ru_maxrss` is a cumulative maximum, not a per-job delta or total process-tree memory. Label it accordingly; use process RSS sampling for memory-specific claims.
- A process-group scan can race reaping and transiently double count a child while its CPU also becomes parent child-time. This may stop conservatively early; it does not alter final resource receipts.
- Nested receipts are diagnostic only. Charge one inclusive outer receipt per experiment root, never both outer and child CPU. GPU occupancy similarly charges the reserved outer interval only once. Auditor/test parent CPU already includes its nested wrapper jobs.

Measured reviewer experiment CPU: original harness0.74464012 + patched harness0.784934992 + aggregate-cap fixture0.597266784 = **2.126841896 CPU-core seconds**. These three inclusive receipts are the review test charge; do not add nested job figures again. CLI/import/filesystem administration not included as experimental work. The disposable source export is not an installed dependency change.

Post-test source amendment `389d8805` replaces TERM/grace with immediate group SIGKILL on budget exhaustion. Static inspection confirms the specific five-second grace concern is removed; the above numeric tests remain receipts for `bed4795a`, not rerun claims for the amendment. Sampling/startup overhead remains and must be reserved and charged from actual receipts. The subreaper and unconditional descendant cleanup remain.
