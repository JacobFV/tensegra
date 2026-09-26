# Protocol F: confirmation on fresh lineages and fresh worlds

Registered 2026-09-26T21:50Z, before any Phase F evaluation or training. Promotion inputs:
- **A2:** no arm promoted.
- **Track C:** C-H1 failed; nothing promoted for learned control.

The confirmations target the two findings that met their screening criteria and matter for the brief's deliverables:
- **F1:** a public recovery rule removes computational stagnation at no utility cost for competent policies. It is a supplied rule; this is not a learned-control claim.
- **F2:** factorized (action-set and above) supervision learns rational probing and cost-sensitive structure use that transfers to held-out generator regions.

## F1: deployment rule on fresh lineages (depworld)
- **Policies:** fresh X1 bootstraps r3, r4 and r5 (configs/campaign04/f-boot-x1-r{3,4,5}.json; P1 recipe; fresh disjoint streams). Each bootstrap's latest checkpoint; no selection.
- **Modes:** greedy vs r_mask (diagnostic v1, registered release semantics).
- **Worlds:** **sealed fresh seeds 170,000,000 + 100,000·i, 512 per condition** over iid_f0, iid_f2, events_train_kinds_p1 and foreign4.
- **F1-H (supported only if all hold in 3/3 lineages, IID group):**
  - r_mask utility ≥ greedy − .005;
  - r_mask success ≥ greedy − .01;
  - the no-progress **episode** rate under r_mask ≤ .5 × greedy's.
- **Reporting:** per-condition readings (events and foreign4) are reported with an absolute floor: r_mask success ≥ .8 × dep_reuse success. dep_reuse runs on the same worlds.

## F2: probeworld supervision on fresh seeds and fresh world draws
- **Runs:** rungs L0, L1 and L4 × **fresh training seeds 3, 4, 5** (9 runs). The same labels and configuration pools as B1.
- **Evaluation:** the held-out splits (heldout_price, heldout_k, heldout_comp) with **fresh world draws** (`--world-offset 700`, replacing B1's 500), 512 episodes per split. References are re-run on the same draws.
- **F2-H1 (B-H1 as corrected by the audit):** L1 and L4 held-out mean regret ≤ .8 × L0 in 3/3 seed pairs, with the transfer floor (success ≥ .8 × π*).
- **F2-H2 (B-H3):** L4's effective-ρ build-rate deviation from π* is ≤ .15 on heldout_k and heldout_price in ≥ 2/3 seeds.
- **F2-H3 (B-H4, pooled held-out as registered in B1):** L4's first-probe rate is ≤ .10 when probing is not optimal and ≥ .80 when it is uniquely optimal, in ≥ 2/3 seeds. The **per-split** readings, including heldout_comp where B1 failed, are reported as the primary caveat.
- **F2-H4 (B-H5, pooled):** L4's unjustified switch fraction is ≤ .10 and its justified-after-failed-probe ratio is ≥ .8 × π*, in ≥ 2/3 seeds. Per-split readings are reported.

## Budget
- F1 ≈ 4k core-s (3 × 2 modes × 2,048 episodes, plus references).
- F2 ≈ 5.5k (9 × ~530 training, 9 evaluations, references).
- The independent audit of A2, C and F is ≈ 3k.
- Projected total ≈ 125k of 172.8k (reserve intact).
