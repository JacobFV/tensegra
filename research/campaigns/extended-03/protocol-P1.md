# P1: acquisition, emergent reuse, and input ablations (pre-registration draft)

Status: design is registered before any training. Numeric condition parameters are filled from the Stage A leverage profile *before* any learned model exists. They are frozen at the "P1-freeze" commit, and no later change is allowed without a new protocol version.

## Arms (3 fresh lineages each; training streams non-overlapping across all lineages and arms)

| Arm | Bootstrap teacher (600 supervised updates) | RL (actor-critic v2, 1,800 updates, single lineage) | Features |
|---|---|---|---|
| X1 `reuse-boot` | `dep_reuse` | yes | d1 |
| X2 `recompute-boot` | `dep_recompute` (never reuses) | yes | d1 |
| X3 `noapp` | `dep_reuse` | yes | d1 with request/dependency-match relations zeroed |
| X4 `noattempt` | `dep_reuse` | yes | d1 with attempted-action features zeroed |

- **Controller:** lightweight, width 1024, the extended-02 recipe (lr 3e-5, entropy .003, KL .3, advantage normalization).
- **Lineage independence:** initialization seeds and training/development streams are disjoint for every lineage and arm. A test checks this before launch; the extended-02 overlap is not repeated.
- **Checkpoints:** latest only. There is no checkpoint selection.

## Worlds

- **Training mixture:** the Stage A default size, event kinds {edge_closed, capacity_reduced, slot_closed} with p_event 0.5, foreign_records 0–2, and work_price at the profiled value.
- **Sealed evaluation:** fresh seeds ≥ 110,000,000, generated after freeze. Every condition gets 256 worlds.
  - **IID:** the training mixture.
  - **no-event**, as the easy anchor.
  - **Held-out event kind:** deadline_moved.
  - **Heavier foreign records:** foreign_records 4.
  - **Larger size:** +1 category.
  - **Cost shift:** work_price ×4.
- **References** on the same worlds: greedy, recompute, reuse, naive_reuse, reuse_norevise.

## Endpoints

**Primary:**
1. **X1:** IID success, and IID utility relative to the `dep_reuse` teacher.
2. **X2:** **reuse emergence.** The rate of correct reuse when an applicable record exists, compared at the X2 bootstrap vs the X2 RL endpoint, together with work units per success.

**Secondary:**
- **Invalid-reuse rate:** use_return of a non-applicable record.
- **Stale-use rate.**
- **Revision quality:** after an event, recomputing only the invalidated pieces.
- **Identical-retry rate:** repeating an action_key whose last outcome was a rejection, with its dependencies unchanged.
- **Transfer:** held-out event kind and larger size.
- **X3/X4 vs X1**, on the conditions where the ablated information is needed.

## Decision rules (explicit aggregation)

All means are equal-weighted over the conditions of a group. The per-lineage values are reported alongside.

| Rule | Supported if |
|---|---|
| **R1 (acquisition)** | X1 IID success ≥ 0.95 in ≥ 2/3 lineages **and** the X1 lineage-mean IID utility ≥ `dep_reuse` − 0.02 |
| **R2 (emergent reuse)** | In ≥ 2/3 X2 lineages, correct-reuse rate at the RL endpoint − bootstrap ≥ 0.25 (absolute) **and** work per success decreases, with IID success not below the bootstrap by more than 0.02 |
| **R3 (applicability input is used)** | X3 invalid-reuse or stale-use rate exceeds X1's by ≥ 0.05 in ≥ 2/3 lineage pairs (paired by lineage index), on the heavier-foreign-records and event conditions |
| **R4 (attempt memory is used)** | X4 identical-retry rate exceeds X1's by ≥ 0.05 in ≥ 2/3 lineage pairs, or X4 success < X1 by ≥ 0.02 |
| **R5 (transfer)** | X1 held-out event-kind success ≥ 0.9 × X1 IID success in ≥ 2/3 lineages; reported as a secondary |

Each rule is reported with **both** the aggregate and the per-condition reading where they differ.

## Integrity

- **Preflight gate.** An independent Stage B audit verifies, on encoded tensors, that d1 preserves every distinction in the applicability table. It also confirms that the X3 and X4 ablations remove exactly the intended distinctions.
- **Teacher labelling.** The teachers are privileged training supervision that uses only public observations. The X2 teacher never reuses, so any reuse in X2 is learned from reward.
- **Sealed-world use.** Worlds are evaluated once per frozen endpoint. Any later researcher-adaptive intervention on the same worlds is labelled as paired and not confirmatory.
- **Independent audit.** A raw-row reconstruction is done by a separate subagent before the claims are finalized.

## Freeze amendment (recorded before any training; commit tagged P1-freeze in decisions.md)

- **Condition values.**
  - Training mixture: size s3 (3×3 items, 7 locations, 6 slots), p_event 0.5, foreign_records ∈ {0, 2}, event kinds {edge_closed, capacity_reduced, slot_closed}, `event_trigger="progress"`, work_price 2e-4 (DepSpec default), compute_price 1e-4.
  - Sealed conditions (seeds 110,000,000 + 100,000·i, 256 worlds each):
    - iid_f0, iid_f2, noevent_f2;
    - events_train_kinds_p1;
    - heldout_deadline_moved (p = 1, the kind never used in training);
    - foreign4;
    - larger_s4 (4×3, 8 locations, 7 slots);
    - work_price_x4 (8e-4).
  - The IID group = {iid_f0, iid_f2}.
- **Step cap.** The learned-arm decision cap is 96, equal to the world step limit, so there is no cap asymmetry.
- **Pairing, clarifying the "disjoint" rule.**
  - Across lineage indices r, all initialization seeds and training/development streams are disjoint (verified by script).
  - Within a lineage index, arms X1–X4 share initialization seeds and stream starts. That makes X3/X4 paired input ablations of X1, and X2 a paired teacher change.
  - The protocol text said "disjoint for every lineage and arm". It is amended here, before training, because paired ablations are the stronger design for R3/R4.
- **Configs.** `research/tools/campaign03_p1_configs.py` → `configs/campaign03/p1-*.json`.
