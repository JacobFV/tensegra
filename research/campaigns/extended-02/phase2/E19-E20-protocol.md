# E19/E20: provenance binding under same-type distractor returns (pre-registration)

**E19 (diagnostic).** E17 RL endpoints (m1 features) and references are evaluated on sealed worlds (seeds 96,000,000+) that start with 0–2 public prior results of primitives **in** the goal. These are certificate-valid solutions to unrelated instances: right type, wrong provenance. The conditions are the IID pairs S→R, S→A, A→R, the held-out pairs R→S, A→S, R→A, and the singles S and A.

m1 candidate features contain no provenance, so E19 measures whether a controller can avoid same-type distractors *without* that input. Failure is expected and would indicate a missing input, not a learning limit.

**E20 (repair, run only if E19 success < 0.9 averaged over conditions and lineages).** The E17 recipe (same seeds and streams, 3 lineages), with two changes:
- **(a) m2 features:** m1 + public provenance for record candidates (the record's problem is one of this episode's drafts; that draft is unchanged since the call). The records' `prior` field is never read.
- **(b) training worlds:** 0–2 wrong-type **and** 0–2 same-type distractors.

This is a combined input-and-data repair; the two changes are not separated.

**Endpoint.** Mean success over the same-type conditions, E20 vs E17, per lineage. The E16 sealed set (90M+) plus wrong-type distractor conditions are evaluated for retention.

**Decision.** The repair is supported if the E20 mean exceeds E17 by ≥0.2 in ≥2/3 lineages, with no retention loss >0.05 on the E16 IID pairs.
