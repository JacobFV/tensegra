# C01 profiles and measured development allocation

All three reviewed mechanical phases completed unchanged width1024/key_dim32; neural profiles used16updates,batch256,one LR. These64-example validation populations do not establish development acquisition or sufficient causal support. No architecture, optimizer, exposure or selection rule is changed from profile outcomes.

| Phase | Full process seconds | Internal seconds | Peak CUDA bytes | Parameters |
|---|---:|---:|---:|---:|
|hybrid|6.259259745|5.206025692|447417856|frozen component models|
|N1|2.877469811|1.736682333|125318656|1246248|
|N2|4.049843965|2.820607345|774238208|17028136|

Caps60/60/120s were not approached. Sum experiment occupancy13.186573521s; batch wall13.321254755s. GPU released before analysis. Hybrid parameters/buffers/normalization hashes match before/after. All12clean hybrid joint lowering-and-answer cells are64/64; wrong/swap changed-label agreement is34/34 and21/21, explicitly below256 required support. The causal pass fields correctly remain false. N1/N2 fresh profile answers are37/64 and32/64 after calibration-selected16/0 checkpoints. These early learning points do not support any comparison of achievable competence.

Original remote calibration export mtimes isolate step0→16 spans0.315001s(N1) and1.299004s(N2); step8→16 spans0.080000s and0.446001s. These include checkpoint evaluation/export and selected-state I/O; they are not pure optimizer timings. They imply roughly80–158s(N1) and446–650s(N2) for8000total updates, with first-interval warmup retained in the upper estimate. Add larger population generation,2048-example calibration,4096-example controls and export. Conservative full-process ranges: hybrid200–400s,N1110–300s,N2500–850s. Short-run timing uncertainty is explicit; no additional GPU profile is needed for safe caps.

Freeze development allocation at hybrid600s,N1900s,N21800s, each separately released by root. The optional N1-frozen diagnostic remains unscheduled. Each scheduled neural arm has TWO4000-update LR runs at batch256:2048000 presentations per arm,4096000 across N1+N2. Both use identical public examples and sampling streams. The hybrid does no new fitting. Main body/selection/data/checkpoints remain unchanged; only the provisional N2 cap drops from3600 to1800. Total scheduled cap3300s is a ceiling, not anticipated occupancy. Record actual per-phase receipts and charge failures/exports.

[Raw compact outputs, summaries and receipts](../../../results/campaign-01/composition/c01-profile/batch-occupancy.json) are committed under the adjacent result tree. Full lossless PT logits/public inputs remain at `~/topoformer-campaign-01/composition/c01-profile-{hybrid,n1,n2}/`. [Timing evidence](../../../results/campaign-01/composition/c01-profile/timing-estimate.json) records the original artifact timestamps. All compact files retain original PT hashes; hybrid original/supplied/requested targets and refusal masks remain distinct. Main comparisons must use answer-only versus answer-only; the hybrid joint lowering-and-answer count is its separate semantic gate.
