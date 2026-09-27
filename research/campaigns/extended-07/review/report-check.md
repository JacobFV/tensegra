# Extended-07 report number check (independent; written by root from the checker's final message)

The checker could not write this file itself because of an environment restriction, so root transcribed its findings.

## Scope
Every number in report.md at d4f0f052/deb63606 was checked against the staged, hash-checked results and the audits.

## P3-REPLICATE, the only previously unaudited result: numbers verified
- **From frontier.json:** every bil − mlp contrast was recomputed from the per-seed arm values, for both contracts, pooled, without q2_after_notH and per decision type. Every mean and sign count matches.
- **From the raw b6d logs** on pro6000, with the checker's own script and bootstrap: every point estimate and sign count reproduces exactly, and the CIs agree within ±.005.
  - Metered receipt: `results/dev/e07-audit-report-p3r-20260927T222549-1295599-process`, 4.2 core-s.
- **Numbers:**
  - Mix contract: pooled +.018 [−.018, +.056] 3/5; without q2_after_notH −.011; q2_after_notH +.061; after failed probes −.146 [−.255, −.049] 0/5.
  - Exact contract: −.010 [−.061, +.038] 2/5.

## Checked and OK
- the P2 frontier table (9 rows × 3 columns);
- the P2/P3 screen figures and the frontier statistics;
- the P0/P1 claims, including bit-identical historical reproduction;
- GPU 0, legacy occupancy 14.2k, the staged hashes, and no e07 jobs running at check time.

## FIX items (15), all applied by root
1. Close timing.
2. Q1 per-group nMAE, taking b6c values instead of dev.
3. The full − single CI and the all-unit value.
4. The slope range and R².
5. The comparable-scorer SEP numbers.
6. In-support substitution rates (.013 vs .061).
7. Shaping seed counts, and the shaping-alone contrast (+.060, unresolved).
8. Q4 ranges: consumers vs SUP.
9. The acquisition cost range and where it sits.
10. Q6 replication wording with CIs, per contract.
11. Localization bullet: hits vs false-change suppression.
12. The CPU line (local core-s were counted twice).
13. The unspent-budget figure.
14. Incident 3: two stalls, not three.
15. Next-decision item 1.

**Compute:** about 31 core-s (4.2 metered, plus about 25 local and about 2 unmetered read-only).
