# Protocol A1: deployment matrix over frozen policies (exploratory)

Registered 2026-09-26T19:45Z, before any A1 output. The builder's 16-world smoke is indicative only and is not used below.

- **Policies (frozen, hashes from extended-03):**
  - X1 bootstraps r0–r2;
  - P1 RL finals X1 r0–r2;
  - P2a C1 final;
  - references dep_reuse, dep_recompute and dep_greedy.
- **Modes:** greedy; sampled (T = 1, sampling seed 20260927, one trajectory per world); **r_mask**; **r_sample** (design v2, review F5 release semantics).
- **Worlds:** fresh seeds 130,000,000 + 100,000·i, 256 per condition over iid_f0, iid_f2, events_train_kinds_p1 and foreign4. The same worlds are used for every policy and mode. The solver cache is on (result-identical).
- **Metrics:**
  - success, utility, total cost, work per success;
  - no-progress rate and episodes (diagnostic v1), cycle lengths;
  - intervention counts;
  - per-world variance, used for Track C power calculations.
- **Aggregation:** per lineage, never pooled across lineages for a claim; the IID group is {iid_f0, iid_f2}.
- **Diagnostic cost (F16):** the diagnostic/recovery-rule compute is **not charged in utility**. Its CPU is recorded per row and reported alongside, as it is small compared with a width-1024 actor forward. A learned metacontroller's forward (Track C) **is** charged, as neural work units in proportion to its parameter count relative to the actor. This asymmetry is disclosed, and Track C comparisons also report utility with the rule's diagnostic CPU charged at the same rate.
- **Readings (descriptive; not claims):**
  - (i) Does r_mask or r_sample recover the collapsed P1-RL policies' greedy success, and how close to sampled?
  - (ii) Does any recovery rule cost the competent bootstraps? Does a best fixed mode exist per policy?
  - (iii) What headroom is left for learned control (Track C)?
