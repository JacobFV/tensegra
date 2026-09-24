# Extended-02 phase-2 claim map

Categories follow the campaign prompt. Everything is development/sealed evidence from one environment family.

| Category | Claim | Evidence | Limit |
|---|---|---|---|
| a. Semantic translation | Public problem builder + agent-chosen constraints give correct reductions | 100% correct reductions where built (E09) | Builder supplied; not free-form formalization. RL02 semantic branch failed (0/512) |
| b. Algorithmic execution | Supplied exact solvers with typed statuses and certificates | Phase-1 protocol tests; 116 CPU tests | Supplied, not learned |
| c. Return addressing | Learned multi-return uses satisfy the address contract; corruption of the selected return changes outcomes | 100% contract-valid multi-return uses; corrupted-return success drops to 0.38–0.57 for tool-first and 0.92 for direct-first | Contract validity ≠ semantic choice among valid returns; stale-route fault rarely engaged |
| d. Metareasoning | Single-lineage RL can learn a direct-first → solver → escalate policy that **matches** the public teacher and is robust to tool loss and corrupted returns | 5/7 single lineages; E14 fresh lineages +0.001 vs teacher (CI includes 0); 0.25–0.35 vs 1.57 solver calls/episode | E14 pre-registered rule only partially confirmed (1/3 met all criteria); not reached reliably; "exceeds teacher" not supported; 48-step cap caveat |
| d. Input contract | Budget-escalation failure repaired by supplied public relational features | E07: 119/128 vs 57/128 (exact reproduction of E05 under v1) | Features supplied, not learned; one seed per arm |
| e. Composition | Unscheduled subset → route → deliver → verify → replan sequences transfer to unseen condition combinations | E09 transfer (obstacle+tight, obstacle+expensive) | No new primitive semantics, motifs or longer compositions tested |
| f. Evolutionary advantage | **Not supported**: PBT ≤ multistart < single | Pooled PBT − multistart −0.0097 [−0.012, −0.007]; PBT < single 3/3; E12 and E13 variants no better | 3 replicates; banks shared across E08/E12/E13 |
| f. Mechanism | Short-horizon exploit/replace selection removes the direct-first mode (E08/E12) or, with robustness-visible fitness, selects tool-avoidance (E13) | Greedy-first dynamics; paired multistart identical before replacement; audited | One environment family; not a general law |
| f. Adaptive curriculum | No advantage | E12 | 3 replicates, shared banks |
| g. Structural attention | Not tested | — | Memory interface (ordinary attention) wall-capped |
| Architecture | Lightweight beats recurrent workspace under RL | E11 0.943 vs 0.771 (recurrent never calls tools) | One seed per family |
