# E17: wrong-type return exposure as a repair for the E16 composition failure (pre-registration)

**E16 finding.** RL lineages solve every training composition, but fail when select must follow assign (rl-r2 0/256; rl-r0 0.77). They repeatedly apply the existing CSP return as the select result ("return type mismatch" until the step limit). In training, select was always the first stage, so the policy never faced a current select stage while a computation record of another type existed.

**Intervention.** Identical to E16 (initialization seeds, bootstrap/RL recipes, streams, train mixture, m1 features, held-out splits). The one change: every training world may start with 0–2 public *distractor* computation records. These are certificate-valid results of primitives **not in that goal**, so they are the wrong type for every stage of the episode. They carry no information about the solution and are never charged. The teacher ignores them (it uses only its own drafts). Address streams shift because the records consume handle draws. Held-out orders stay held out. Distractor types are drawn from primitives absent from the goal, which does not reveal the held-out orders.

**Evaluation.** The same frozen E16 sealed conditions (90M+), plus distractor variants of the IID and held-out pairs, for both E16 and E17 endpoints.

**Endpoints.**
- **Primary:** assign→select held-out success (pair and triples with S after A), E17 vs E16, per lineage.
- **Secondary:** overall held-out pair/triple success; IID retention; return-type-mismatch rate.

**Decision.** The repair is supported if the E17 mean over lineages of assign→select success exceeds E16's by ≥0.2 **and** no lineage falls below 0.8 on it, with IID pairs ≥0.95. Otherwise the result is partial or failed.
