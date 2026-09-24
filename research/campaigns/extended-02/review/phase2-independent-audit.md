# Phase-2 independent audit (E08/E09v2/E11/E12)

Auditor: independent reconstruction. Branch `campaign/e02-phase2-audit` (from `campaign/extended-02` @ 88f781c5).
Code: `audit_phase2_remote.py` extracts from raw results on `gb10-direct` using read-only stdlib access. `audit_phase2_v1_remote.py` compares the v1 and v2 runs. `audit_phase2.py` runs the analysis locally.
Numbers: `phase2-independent-audit.json`. None of `campaign02_e09_analysis.py`, `mechanism.py` or `dynamics.py` was used or read.

## Method
- I read every sealed file directly: 6 roots × 26 conditions × 23 arms = 598 arm files, plus `worlds.jsonl.gz`. For each world I recomputed `spec_hash` as sha256 of the canonical JSON of `spec`. For each episode I recomputed the means from `outcome.utility` and `verified_success`, and the mechanism flags from `outcome.history[].action.kind`.
- Population runs used: E07 legacy/recurrent-v2, E08 bank r0–2, E08 main ×9, E11 ×2, E12 ×3 and E13 ×3 (E13 is still running). Their `state.json` and `protocol.json` were read on the host. The repo copies of E07/E08 state, protocol and lineage are byte-identical to the host copies. E11, E12 and E13 have no repo copies on this branch.
- I hashed with sha256 every checkpoint bound in the six `configs/campaign02/e09v2-sealed-*.json` files and every registered finalist, directly from the host files.
- Bootstrap: 4,000 resamples, stratified by replicate × condition (24 strata × 256 paired episodes), seed 20260924.

## Verdicts

**1. Sealed IID means: CONFIRMED.** All 19 claimed values reproduce exactly to 4 d.p.
single 0.94296 / 0.94295 / 0.93015; pbt 0.92887 / 0.89595 / 0.92843; multistart 0.92821 / 0.92918 / 0.92510; bank0 0.80210 / 0.74140 / 0.92131; reference-cheap_first 0.93971; reference-cheap 0.80467; arch-lightweight-rl 0.94257; arch-recurrent-rl 0.77064; curriculum-pbt 0.92772 / 0.92720 / 0.92829. Each value is the mean over 8 conditions × 256 worlds.
Arms not in the claim: reference-cheap_first_fallback_v2 0.93971, which is identical to cheap_first on IID and on transfer; reference-always_tool 0.74852; arch-lightweight-boot 0.81039; arch-recurrent-boot 0.76262.

**2. Primary rule fails: CONFIRMED.** PBT − multistart per replicate is +0.00066 / −0.03323 / +0.00332. The signs are mixed. Pooled paired mean is −0.00975, 95% CI [−0.01240, −0.00733]. The pooled difference goes against PBT, not merely short of zero.
PBT < single holds in all three replicates: −0.01409 / −0.04701 / −0.00172. Pooled −0.02094, CI [−0.02337, −0.01874]. The r2 gap is small (−0.0017), so "single best in all 3" holds only narrowly in r2.

**3. Identical worlds: CONFIRMED.** The same 26 conditions are present in all six roots, with all arms bound in each config. Nothing is missing and there are no extra arm files. Every arm file has the same `(seed, spec_hash, semantic_spec_hash)` sequence, in order, as its root's `worlds.jsonl.gz`. Every `worlds.jsonl.gz` is identical across the six roots, reference root included. Seeds are exactly `range(seed_start, seed_start+256)` with no duplicates. All recomputed `spec_hash` values match. Each host `summary.json` config equals its repo config file.

**4. Seed disjointness: CONFIRMED.** I checked 558 used intervals: every allocation `training_seed_interval`, every retained recipient seed cursor, and the development range `[development_seed_start, +development_examples)` of all 22 runs above. None overlaps any sealed interval (70.0M–71.7M+256 and 75.0M/75.1M+256). The lowest seed used anywhere is 310,000,000 (E07); the highest is 780,014,400 (E11). Development panels are 391M (E07), 590–592M (banks), 595–597M (E08/E12/E13) and 598M (E11).
Limit: this covers the recorded intervals only. I did not regenerate the training streams.

**5. Finalist integrity: CONFIRMED.** For all 19 bindings (plus the 2 arch-boot bindings), the file on disk has the bound sha256. Each finalist also equals the one recomputed from the allocations: last round, max development utility, lowest slot on ties; single and E11 take the latest allocation (round 4 slot 5). Each also equals `state.finalist`, and the bound `development_utility` equals the allocation utility.
Close calls:
- pbt-r0: slot 3 at 0.935620 vs slot 0 at 0.935550.
- curriculum-pbt-r1: exact tie between slots 1 and 4 at 0.919268984375; the slot-index tie-break gives slot 1, correctly.

bank0 is bank member 0 by protocol, not the bank run's own finalist (slot 2). This is disclosed. Every E08, E12 and E13 initial checkpoint matches a bank allocation hash, and every E11 initial checkpoint matches an E07 allocation hash.

**6. PBT lineage legality: CONFIRMED.** The E08 pbt ×3 and E12 ×3 runs have 8 replacements each, 2 per round in rounds 0–3 and none in the final round 4. In every replacement:
- the donor's utility is strictly greater than the recipient's (smallest margin +0.00176, pbt-r2);
- the donor is in the top two and the recipient in the bottom two;
- the lr, entropy and kl factors are each in {0.8, 1.2};
- the parent and recipient sha256 equal the round's allocation checkpoints.

Multistart, single and the banks have zero replacements. E13 (in progress) has had no violations so far.

**7. Mechanism: CONFIRMED, with one qualification.** Greedy-first rate on int_4x4_control:
- single-r0, single-r1, multistart-r1 and arch-lightweight-rl: 1.000 each.
- Tool-first finalists: pbt-r0 0.0039, pbt-r1 0.0078; pbt-r2, multistart-r0, multistart-r2, single-r2 and curriculum-pbt r0/r1/r2 all 0.000.

Success on int_4x4_no_tools:
- single-r0 0.836 and single-r1 0.844, both ≥ 0.8.
- Tool-first finalists: pbt-r1 0.0234, pbt-r0 0.0039, all others 0.000.

Under no tools, the tool-first finalists never abstain (abstain rate 0). They take the full 48 actions in 255/256 (pbt-r0), 251/256 (pbt-r1) and 256/256 (all others) episodes. These are world step-limit endings, not policy truncation (`truncated` = 0). Utility is −0.069.
Qualification: greedy-first does not imply robustness to having no tools. multistart-r1 is greedy-first at 1.00 but succeeds on only 0.441 of no-tools episodes, and 58% of its episodes reach 48 actions. arch-lightweight-rl scores 0.816.

**8. Ledger arithmetic: CONFIRMED.** There are 37 phase-2 jobs. Their CPU sum is 104,099.389633 s, equal to `phase2_charged_cpu_core_seconds`, and each job equals its host `occupancy.json`. The union of GPU-job `[launch.started_unix, occupancy.ended_unix]` intervals is 10,520.087027 s, equal to `phase2_charged_gpu_seconds`. Each job's interval matches the ledger.
The failed jobs are charged: e07-memory v1/v2 (exit −9) and the e09 v1 ×4 runs (exit 1).
Caveats:
- The ledger does not yet include the three E13 jobs. They are running and have no occupancy receipt, so the phase-2 totals are provisional.
- The `receipt` paths for the e09, e09v2, e11 and e12 jobs point to `research/results/campaign-02/<job>-process/occupancy.json`, which is not committed on this branch. I verified those jobs against the host copies only.

## Other checks and concerns
- **Interface and feature version:** there is no mismatch. Every evaluated learned arm has the policy config recorded in its summary, and it agrees with its training protocol: family, width 1024, feature_version v2, legacy interface. Evaluation sources match training sources for policy, world, protocol and references in every run. `population.py` differs, but evaluation does not use it. E07 legacy-v2, used only for the arch-lightweight-boot arm, was trained with an older `training.py` hash.
- **Step-cap asymmetry between learned and reference arms (real, disclosed nowhere I saw):**
  - The evaluator runs learned arms with `max_steps` = 48 from the training config. Worlds with `step_limit` 64 are therefore cut at 48 for learned policies: iid_4x5, xfer_5x4_larger and xfer_5x5_subset_tool_invalid. References get all 64 steps.
  - Affected IID episodes (iid_4x5): bank0-r0 35, bank0-r1 40, arch-lightweight-boot 33, arch-recurrent-boot 37, arch-recurrent-rl 39, multistart-r1 2. The pbt, single and curriculum finalists have 0.
  - On xfer_5x5, every tool-first finalist is truncated in 254–256/256 episodes.
  - The cap matches training, and it is symmetric within the primary comparison (all 48), so the PBT result is unaffected. It does handicap learned-vs-reference comparisons on 64-step conditions, and it depresses bank0 and recurrent IID means slightly.
- **v1 to v2 re-run:** the 300 (condition, arm/worlds) files present in both the failed v1 partial roots and v2 are identical per seed in spec_hash, utility, success and full action sequence. The re-run could not have changed any outcome, so no selective re-draw was possible.
- **Determinism cross-check:** pbt-r0 slot 0 and multistart-r0 slot 0 have identical development utility, and so do pbt-r1 slot 0 and multistart-r1 slot 0. Both members are never-replaced root-0 lineages, which is consistent with the paired design.
- **Development selection is weakly informative:** pbt-r1's final round contains collapsed members (development utility 0.013 and −0.066). Its finalist is also the replicate's weakest sealed arm among the non-bank arms.
- **References:** cheap_first and cheap_first_fallback_v2 have identical IID and transfer means, so they are not two independent baselines. Only single-r0 and single-r1 (+0.0033 IID each) and arch-lightweight-rl (+0.0029) exceed cheap_first. All three are greedy-first.

## Audit cost (measured)
- Remote extraction: 122.86 user + 0.69 sys = 123.55 core-s, 124 s wall, 167 MB RSS (`/usr/bin/time -v`).
- Remote v1/v2 identity check: 69.51 core-s.
- Local analysis: 8.17 core-s.
- Total: about 201 core-s CPU and 0 GPU. A few sub-second read-only inspection commands were not timed.
