# S12 fixed paired confirmation result

**The LR reduction has a replicated directional advantage on these three fixed seeds, but the preregistered all-seed competence gate fails.** Calibrated complete recovery is66/1,024,131/1,024 and139/1,024 for decay versus0/1,024 for every constant-LR fork. Seed701 misses the required103/1,024; retain that failure. All nine parent/fork jobs completed at their fixed endpoints without changing recipe or selection.

| Seed | Constant complete | Decay complete | Paired gain | 95% paired example interval | Raw constant→decay |
|---|---:|---:|---:|---:|---:|
| 701 | 0/1024 | 66/1024 | 6.445pp | [4.980, 7.910]pp | 0→2 |
| 702 | 0/1024 | 131/1024 | 12.793pp | [10.840, 14.844]pp | 0→6 |
| 703 | 0/1024 | 139/1024 | 13.574pp | [11.523, 15.723]pp | 0→11 |

The mean calibrated paired gain is10.9375percentage points, range6.4453–13.5742points across the three seeds. The95% shared-event bootstrap interval for the mean is[9.7005,12.2396]points. Each seed has a positive difference and the shared-event lower bound ispositive, meeting the frozen directional criterion. These10,000resamples use the same example indices jointly across seeds, RNG12012: uncertainty is conditional on these three fixed seeds, **not** seed-population certainty. Seed variability remains visible in the table.

Raw recovery gains are2,6,11graphs; mean0.6185points with shared-event95% interval[0.3581,0.9115]points. Raw seed701interval is[0,0.4883]points; the other raw intervals are[0.1953,1.0742] and[0.4883,1.7578]points. Raw reporting is parallel, not selected after results.

For calibrated decoding, both-correct and constant-only counts arezero for every seed; decay-only counts are66/131/139 and neither counts958/893/885. Raw both/constant-only alsozero; decay-only2/6/11 and neither1022/1018/1013. Frozen full-TRAIN frequency reference remains0/1,024. Engineered S10 schema/reference outcomes are reported separately and cannot replace learned recovery.

| Seed/arm | Copy accuracy | Calibrated typed F1 | Calibrated ordered F1 |
|---|---:|---:|---:|
| 701/constant | 0.97604 | 0.89400 | 0.91588 |
| 701/decay | 0.98976 | 0.96053 | 0.97928 |
| 702/constant | 0.98723 | 0.90674 | 0.92853 |
| 702/decay | 0.99573 | 0.97078 | 0.98554 |
| 703/constant | 0.98170 | 0.90753 | 0.92796 |
| 703/decay | 0.99451 | 0.97043 | 0.98358 |

All three parents plus six forks used7551.076717full-process GPU-slot seconds (2.0975hours), within the9,000second summed hard caps. Parent/fork receipts and all raw/calibrated endpoints are archived; GPUFREE was reported before analysis. Each pair shares the same parent model, AdamW steps, first construction/negative-pair batch and inherited source bytes; initial TRAIN/DEV predictions replay exactly. Independent final endpoint/analysis review is requested.

The original analysis source at6efe785/c6cbed2 (SHA9dc6ea686490b55413e21bb22d2ef544acf99f75c966c0a73612da37551d0dc6) and original config were used with no statistical changes. A stale remote copy was replaced with that already-frozen source before analysis; inherited training source was untouched. Exact input/output/config provenance is in `s12-confirmation-summary.json`.

Scope remains English renderer, known spelling and two known ordered tree motifs at depth3, with fresh equality/reference constructions. This confirms a narrow optimization effect; it does not establish broad graph competence, new-motif/depth/language transfer, or programmable-attention generality. S14/rename and paired S13 acquisition retain their separately frozen protocols; no outcome grants an automatic promotion or GPU release.
