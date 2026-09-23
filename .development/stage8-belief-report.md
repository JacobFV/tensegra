# Stage 8 belief interfaces

## Status

The strict observation-ID-matched primary study is frozen and pending execution.
The completed initial main is preserved as a **metadata-asymmetric diagnostic**:
its recurrent encoder did not consume observation IDs while the protected ledger
did. Both consumed identical candidate/evidence/role/action payloads. This limits
causal interpretation of its ID-dependent stress comparisons.

## Completed diagnostic

Three paired seeds, width1024/FF2048, 1000updates×32 fresh episodes per arm.
Every cell has512validation and512test episodes. Eight conditions at N8/N16.
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
