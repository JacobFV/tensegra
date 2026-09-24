# S21: broader motif acquisition without omitted-combination transfer

**The fixed joint development gate fails.** Broad training acquires all three newly exposed motifs and preserves old-known competence within the registered tolerance, but neither arm recovers a single complete heldout3×4 graph at any curve. There is no automatic extension, and the prospectively conditional three-seed confirmation is **ineligible**. Its sealed cache remains unopened by model inference or this analysis.

| Final4,096 criterion | Result | Decision |
|---|---|---|
| Broad3×4≥52/512 and gain≥26 |0/512 versus original0/512; gain0 |Fail|
| Old-known complete loss≤76 |Original1,469/1,536; broad1,443/1,536; loss26 |Pass|
| Broad≥256/512 in each new motif |2×4:504;5×3:504;5×4:437 |Pass|
| Joint advancement |One required component fails |**Fail**|

Both arms complete128/128 of their declared actual-TRAIN diagnostic panels at the endpoint, with100% teacher-forced accuracy on all eight fields. Panel mixtures differ by design, so these are separate in-sample acquisition diagnostics. The outcome is not general training failure: the broad learner acquires its six trained motifs while still failing the omitted combination.

## Complete curves

Every cell denominator is512; TRAIN denominator128. Fixed endpoint4,096 alone determines the gates.

| Arm | Updates | TRAIN |2×4|3×3|3×4|4×3|4×4|5×3|5×4|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Original|0|0|0|0|0|0|0|0|0|
|Original|1,024|30|0|231|0|69|11|0|0|
|Original|2,048|98|0|455|0|396|170|0|0|
|Original|4,096|128|0|507|0|507|455|0|0|
|Broad|0|0|0|0|0|0|0|0|0|
|Broad|1,024|31|175|180|0|95|10|88|8|
|Broad|2,048|90|407|456|0|349|165|374|175|
|Broad|4,096|128|504|505|0|502|436|504|437|

## Paired effects and residual boundary

| Cell | Broad-minus-original complete | Gained / lost paired graphs | Conditional95% CI, percentage points |
|---|---:|---:|---|
|2×4|+504|504 /0|[97.266,99.414]|
|3×3|−2|2 /4|[−1.367,0.586]|
|3×4|0|0 /0|[0,0]|
|4×3|−5|2 /7|[−2.148,0.195]|
|4×4|−19|28 /47|[−7.031,−0.391]|
|5×3|+504|504 /0|[97.266,99.414]|
|5×4|+437|437 /0|[82.227,88.281]|

Intervals use the prospectively declared10,000 paired event resamples, RNG21021, sorted semantic IDs within each cell. They are descriptive conditional on this single paired initialization and inspected DEV population, not initialization uncertainty or fresh confirmation. The count gates do not use these intervals. A[0,0] interval when both models fail every sampled event is a property of this empirical difference, not proof of impossibility on every future example.

Heldout validity is395/512 original and403/512 broad, but no graph in either arm has exact kind, value, copy, edge or slot components. Original has308 exact-presence graphs; broad217. Original's117 invalid cases comprise115 generated-reference errors and two unsupported multiple pair-slot labels. Broad's109 comprise102 generated-reference errors, five multiple pair-slot labels and two duplicates. Well-formed but semantically wrong graphs remain the majority, so improved validity cannot account for missing transfer.

Aggregate seven-cell DEV teacher-forced field accuracies improve from original to broad: type99.600→99.996%, kind88.015→98.507%, value97.985→99.928%, copy70.288→97.753%, source92.506→98.847%, target83.609→98.031%, role88.201→97.881%, slot89.855→97.910%. These aggregate diagnostics include newly trained motifs and gold prefixes; they do not establish heldout success. Complete per-cell components, generated failure IDs, validity reasons, all fields/curves and paired transitions are retained in `s21-analysis.json`. Detailed first-prefix/layout localization is separately audited and cannot alter the fixed gate.

## Work, provenance and limits

Both arms use identical scratch initialization2101, unchanged62,677,315-parameter S19 sequential actor/loss/LR,4,096updates/batch8,32,768 presentations and schedule15115. Original contains three motifs; broad replaces examples with three additional motifs, for six training motifs. Broad shares2,048 original constructions, and3×4 remains omitted. No generator semantics, vocabulary, output capacity, renderer or architecture changed.

Presentation counts match; computational work does not. Original exposes1,720,320 tokens,988,008 nodes,2,208,616 edges and3,229,392 records; broad1,835,008/1,074,344/2,444,456/3,551,568. Optimizer time is184.579 seconds original and191.216 broad. Whole paired main charged**818.90 seconds** (outer818.89), exit0/GPU-free, with four complete evaluation curves and all seven DEV cells. Profile29.54 yields total**848.44 seconds**, within the1,800 development envelope. Unused cap is not consumed compute or permission for another run.

The frozen analysisb2613fb6/root49838bbf executed unchanged against the complete closed remote archive, with CUDA hidden/two CPU threads and no model imports. Source/config/receipt/data hashes, both initial/final/checkpoint links, per-corpus exposures, all public cache identities, strict validity flags, packed components and artifact summaries passed. Full archivec5f7636f; mainconfig5a13908ec22e9ad47350d7d85433b46291dada4552a474300fcef57871f30db3. Result inventory paths refer to the remote campaign working directory; local archives preserve the same bytes. Independent strict generated-record/checkpoint and localization audits remain separate from this aggregate loader.

S19/S20 known-motif results and heldout failures remain unchanged. S21 shows that broadening exposure from three to six available ordered motifs improves acquisition on those added motifs but does not meet the registered omitted-combination criterion. Preserve this negative transfer result, close the conditional confirmation path, and use archived localization only for any separately registered next question. No additional exposure or threshold tuning is authorized.
