# Stage 8 belief interfaces

## Primary: 1024-wide, observation-ID matched

Both learned arms receive identical candidate, role, action, value and 16-bit
observation-ID features. Each has 16,867,395 parameters; four distinct 1024-wide
residual MLP phases (FF 2048) recur over evidence. This is a candidate-local MLP
comparison, not a replication of Stage7's attention transformer. The protected
arm is supplied an idempotent observation ledger; neural learning supplies
compatibility increments. The comparator must maintain evidence in unrestricted
candidate states. Both receive the same direct posterior/compatibility supervision.

Three paired seeds, 1,000 updates × 32 fresh episodes, 512 validation and 512 test per
N8/N16 condition, eight evidence conditions. Training episodes are paired across
arms. Every model is evaluated at step 0. No settings were tuned after the main
freeze. Source/config freeze: `bac2950`; source hashes are identical across runs.

| Test condition, across seeds and N8/N16 | Protected | Recurrent |
|---|---:|---:|
| Clean final support accuracy |100%|99.80–100%|
| Independent evidence reorder |100%|98.83–99.41%|
| Eight-fold repeated IDs, final support accuracy |100%|98.05–99.80%|
| Explicit retraction final support accuracy |100%|99.80–100%|
| Eight-fold repeated IDs, mean posterior L1 |.0011–.0071|.0685–.1274|

For clean complete evidence, final support is a singleton, so support accuracy is
complete joint proposal accuracy. In partial/empty conditions, membership alone
is not semantic certainty; the full posterior and calibration are evaluated too.

**Both prespecified gates fail.** Protected 14/48 and recurrent 24/48 validation
cells fail at least one threshold. The protected failures are sharply localized:
its initial empty-evidence prior assigns 3.40–5.68% null mass at N8 and
1.73–2.92% at N16, above the 1% impossible-mass limit. This initial error also
raises frame-averaged errors in shorter conditions. Learned prior calibration has
not reached the required level within the fixed budget.

For posthoc attribution only, excluding that initial frame gives a worst condition-mean
posterior L1 of **.00244** and impossible mass **.00122** for protected updates
across nonempty validation conditions. The recurrent comparator reaches
**.13172 / .06586**, respectively, on long repetition. This diagnostic does **not**
relax the registered gate, and does not authorize readiness or composition.

The positive result is learned compatibility plus stable supplied bookkeeping,
not learned duplicate/retraction algorithms. The recurrent comparator also
acquires ordinary progressive beliefs under this wider, fresh-data recipe.
Generic recurrence is therefore not intrinsically incapable of this task.
Long repetition is 33 evidence frames, mostly redundant observations; it is not
33 novel reasoning operations or algorithmic-depth extrapolation.

An independent audit reconstructs 192 cells / 98,304 episode evaluations from full
raw posteriors and targets. Repeated conditions and model arms share episodes;
that evaluation count is not a count of independent unique semantic graphs.

Primary artifacts: `results/stage8/belief-idmatched/` includes every posterior,
target, calibration curve, source/config/checkpoint hash, gate reason and plot.
Checkpoints remain at `gb10-direct:~/topoformer-stage8-artifacts/beliefs/idmatched/`.
Protected training plus probes took about 35 seconds per seed; recurrent about 50.
Those exclude final raw export/oracle evaluation; they are not complete job times.
The recurrent arm executes additional phase calls for its auxiliary loss, so
parameter/exposure matching does not mean matched FLOPs.

## Historical width control

The prespecified historical control uses width 32 / FF 64, the same ID-matched task,
objective, optimizer, 1000 updates, paired seeds and evaluation cells. It has
19,267 parameters. Width 1024 remains the default and primary configuration.

| Test condition, across seeds and N8/N16 | Protected32 | Recurrent32 | Recurrent1024 |
|---|---:|---:|---:|
| Clean final accuracy |100%|100%|99.80–100%|
| Explicit retraction |100%|71.29–88.67%|99.80–100%|
| Eight-fold duplicate IDs |100%|0–80.27%|98.05–99.80%|

Clean proposal acquisition alone therefore misses a substantial robustness gap.
The smaller recurrent model collapses on long repetition in seed 2: it predicts
null in 512/512 N8 and 511/512 N16 test cases, although valid candidates still
exist. Protected bookkeeping works at both widths; the learned initial prior
still prevents either width from passing the full gate. Historical protected
fails 25/48 validation cells and recurrent 46/48.

The width intervention was frozen before primary outcomes and was not used to
tune them. Historical runs used two CPU threads while the primary used CUDA;
this execution-device difference is disclosed and no throughput comparison is
claimed. Training mathematics, data, supervision and exposure match, but this is
not an identical-hardware numerical experiment. The width contrast also cannot
identify which changes from Stage7 caused its improvement.

Historical training plus probes took 22.6–35.5 seconds per seed while sharing the
host with semantic GPU work. All raw results, prior/trajectory calibration, gates,
and a width comparison figure are in `results/stage8/belief-historical32/`.
The source remains `bac2950`; checkpoints are archived remotely under
`~/topoformer-stage8-artifacts/beliefs/historical32/`.

## Completed diagnostic

Three paired seeds, width1024/FF 2048, 1,000 updates × 32 fresh episodes per arm.
Every cell has 512 validation and 512 test episodes. Eight conditions at N8/N16.
Both arms contain16,851,011 parameters; the recurrent arm does more phase calls
because its evidence-compatibility auxiliary and recurrent state updates are
separate. Neither arm executes runtime operations. Supervision is explicit.

| Held-out condition | Protected final support accuracy | Recurrent final support accuracy |
|---|---:|---:|
| Clean, N8/N16 |100% allseeds|99.61–100%|
| Explicit retraction, N8/N16 |100% allseeds|99.61–100%|
| Eight-fold duplicate IDs, N8/N16 |100% allseeds|91.99–100%|

This changes the interpretation of Stage7's failure: the wider, fresh-episode,
directly supervised recurrent comparator **does acquire ordinary belief updates**.
The experiment does not isolate which change from Stage7 was decisive. It is not
evidence that generic recurrence intrinsically cannot maintain structured state.

Long-duplicate mean framewise posterior L1 is approximately .002–.007 protected
versus .063–.138 recurrent. Protected duplicate/retraction behavior is supplied by
its ledger; neural learning supplies compatibility increments, not ledger logic.
ID-information asymmetry prevents treating this as the primary matched comparison.

**Both diagnostic gates fail.** Empty-evidence null calibration remains wrong:
protected N8 prior L1 is .054–.120, N16 .028–.062. Support membership is100% there
because every ordinary candidate is plausible; that does not make the posterior
correct. A nonzero null mass violates the prespecified impossible-mass threshold.
Neither readiness nor composition is authorized by these results.

## Provenance and audit

Canonical diagnostic source: `9745438`; acquisition source/configs and complete
raw metrics are separately retained. Local artifacts:
`results/stage8/belief-main-diagnostic/` contains all predicted/gold posteriors
(gzip), calibration bins, sample public tensors, step0/training curves, oracle
results, source/config hashes, checkpoint hashes, gate failures, and plots.
Durable checkpoints and frozen source remain at
`gb10-direct:~/topoformer-stage8-artifacts/beliefs/`.

The matched follow-up source/config freeze is `bac2950`. It adds the same16-bit
observation-ID feature to both neural encoders (+16,384parameters each) without
changing data, objectives, optimizer, or exposure. This correction was registered
before that study began. The diagnostic is not discarded or relabeled as matched.
