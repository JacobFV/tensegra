# E14: prospective confirmation of the single-lineage finding on fresh lineages and fresh sealed worlds (pre-registration)

**Claim under test** (exploratory support: 3 of 4 lineages in E08/E11). Concentrated actor-critic v2 training of one lightweight lineage (E08 single recipe: row-0 hyperparameters lr 3e-5 / entropy .003 / KL .3, 5 rounds × 6 slots × 60 updates = 1,800 updates on the E08 leverage mixture) reaches a policy with three properties:
- (a) it attempts the direct path first (greedy-first ≥0.9 on control worlds);
- (b) it matches or exceeds the public teacher's sealed IID utility;
- (c) it retains no-tools competence (success ≥0.7).

**Fresh lineages.** There are three new lineages (r0–r2). Each has new initialization seeds (86000+), a new supervised bootstrap (600 updates, E05 mixture, v2 features, stream 900,000,000+), new RL streams (940,000,000+), and a new development panel (985,000,000+). No E08/E11 checkpoint, bank or seed range is reused.

**Fresh sealed worlds.** The same 26 condition definitions, on a new seed base of 80,000,000 with a new address namespace. References are re-run on these worlds.

**Decision.** The claim is **confirmed** if ≥2/3 lineages satisfy (a) and (c), and the lineage-mean IID utility minus teacher ≥ −0.002. It is **partially confirmed** if 1/3 lineages satisfy them, and **not confirmed** otherwise. Every lineage is reported. There is no checkpoint selection: the single learner's latest checkpoint is final. The step-cap caveat (learned arms capped at 48 decisions) applies as in E09.
