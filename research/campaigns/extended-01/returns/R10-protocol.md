# R10: balanced context diversity for the remaining scalar tail

Status: prospective development; no GPU released. Historical R04–R09 outcomes and composition consumers remain immutable. Primary workspace width1024; the only trained component is the existing33,825-parameter linear33-class scalar consumer.

## Evidence and hypothesis

R06 changed only fitting-population selection:16,384 generator events balanced over52 legal type/value strata versus an equally large original mixture. Its fresh worst grid improved41→55/64; late fitting accuracy was16,358/16,384. Both fresh controls independently fixed R04's float-zero ingestion failure, so that failure does not justify another ingestion-specific repair. R07 nonlinear capacity failed acquisition; R08 reduced learning rate gave only55→57/64; R09 public binary-phase heads failed the registered calibration improvement rule. These trials are retained, not repeated.

R06 sampled88,781 of98,304 available delay/event rows, all16,384events, with315/316distinct contexts per stratum. Its balanced mean loss over successive100-step blocks was.08260,.02636,.01729,.01180,.00873,.00826,.00881,.00506,.00408. Optimization was still changing, but training accuracy was far stronger than the worst fresh stratum. This motivates testing additional balanced nuisance-context diversity before changing the representation again.

Hypothesis: a larger balanced set of distinct nuisance contexts improves late exact-value generalization with the same linear consumer and optimizer presentation budget.

## Attributable comparison

Use frozen Stage11 wide backbone11 (selected development backbone because it had the largest R04 late float deficit; R06–R09 used12). Compare nested balanced pools16,384 versus65,536; both receive exactly3,600AdamW updates, batch256, LR.003,921,600presentations. This longer common fitting budget is fixed in advance for both arms; it is not a retrospective extension ofR06. The controlled difference between the new arms is distinct balanced fitting contexts. Both start from the unchanged wide scalar head. Population-specific train normalization follows the unchanged fitting rule and is disclosed as a consequence of changed population.

One fresh131,072-event pool comes from the unchanged generator. Select untouched original events round-robin across a config-seeded permutation of52strata. The16kselection is an exact prefix of64k. Per-stratum counts differ by at mostone:315/316versus1260/1261. Abort on inadequate support. Do not change scalar values, types, operands, identifiers, or operation witnesses. This is an explicit balanced training-population prior, not numerical-range generalization.

Capture the64kfeature pool once; fit its16kprefix without duplicating features on disk. Train delays0/1/2/4/8/16, two distractors. Both arms reset the same sampling RNG, update count, batch size, and optimizer. The draw distribution is uniform over each arm's own delay/event rows; different cardinality means the actual selected indices differ. Do not call them identical row sequences. Preserve visited-row bitsets, actual unique events and rows, and per-stratum counts. Expected repeat exposure differs (~9.375versus2.344draws per available row).

The forward input is a perfect public return event, public keys and distractors; scalar targets supervise only the consumer. No gold facet selection, phase bit, value bypass, encoder change, recurrent training, memory change, or delay-specific head is introduced. All non-value outputs remain fixed. Delay32 is excluded from capture, fitting, selection, and development analysis; it remains available for later fresh confirmation.

## Development observations and decision

Fresh calibration mixture1,024 and validation mixture4,096, both distractor counts2/8 and all six delays. Separate fresh balanced grids have128examples per legal type/value stratum, delays0/1/16 and eight distractors. These are distinct populations and must never be pooled into one accuracy. Namespaces611–615million do not reuse inspectedR04–R09events. The supplied config fixes all seeds. Capture calibration curves every300updates; only fixed3,600endpoint selects advancement, with no best-checkpoint search.

Report exact scalar accuracy, original population cells, all52strata, signed/error-distance counts, per-type/primitive values, train-prefix accuracy, non-value joint/full joint, and paired outcomes. Report the16kconsumer on its unseen64kprefix tail separately from its own fitting set. Larger-pool training membership must not be called validation.

Advance this *diversity recipe* only if, at the fixed endpoint: (1) every64kmixture calibration cell≥98%; (2) worst balanced calibration stratum≥95% (≥122/128); and (3) that minimum improves by at least4/128 over16k. Report fresh validation regardless. This is a development screen, not a rare-error or uniform98% claim. If criterion fails, diagnose training fit versus fresh errors; do not automatically extend or launch confirmation.

If it advances, a separately frozen confirmation would use all three historical backbones10/11/12, fresh fitting/calibration/test events, all prescribed seeds, at least512examples per original cell (prefer4096) and512per balanced stratum when practical. Its original-population and balanced contracts remain separate. No automatic replacement ofR04/R05/C04consumers or autonomous composition follows.

## Budget, freeze and audits

First mechanical profile: same1024backbone,512/2048fitting prefixes,100updates,128mixture rows and8per grid stratum. Separate profile namespace611.9million. Measure capture, fit, evaluation, compression/export and external whole-process occupancy; source/config/checkpoint hashes required. Suggested profile cap60s, subject to sole root scheduler release. Main estimate250–500s pending measured scaling; no main allocation is fixed before profile. Two linear heads only; no broad sweep.

CPU tests verify nested balanced sampling without edits, insufficient-support rejection, and feature/target/hash prefix alignment. Independent preflight checks population pairing, targets/inference separation, untouched historical imports and protocol/config agreement. Postrun audit independently regenerates selected events, replays cached normalized linear predictions, verifies non-value invariance, source/checkpoint/cache hashes, support counts, paired outcomes and fixed-endpoint gate decisions. Lossless feature/logit caches are remote immutable; compact predictions/manifests/receipts and reports are committed. Full process occupancy includes compression/export. No claim here concerns programmable attention.
