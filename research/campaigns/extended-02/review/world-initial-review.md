# Workshop prospective independent review

Source `a2be534a`, before root-requested movement/travel repairs. This review is not clearance for a later uninspected version. No policy training or GPU work was run.

**Observation isolation:** initial inspection found no direct hidden-solution input in observe, action catalog or feature encoders. Hidden item weight/price appears only following inspection. Incompatibilities, categories, capacity/funds and destination are public by contract. Constraint builders read `_spec` fields, but these fields are already observable; this is supplied exact translation of a named public constraint, not learned formulation. Build-route similarly copies an inspected public map. Goal ambiguity is not yet represented; do not claim ambiguous-goal interpretation.

**Confirmed feature/provenance defect:** the public-action fixture in `world_feature_fixture.py` inspects four items, builds a subset draft, solves before capacity is added, adds capacity, and solves again. The old result selects weight10; the new result selects weight2 under capacity2. Both are retrieved. Their `use_return` feature vectors are identical because return payload is not encoded and both records resolve the same *current mutable draft*. Raw JSON distinguishes them, but learned feature-only policies do not receive that information at the read boundary. A recurrent controller might exploit sequence history, yet that is not evidence of robust semantic multi-return selection and is unavailable to a stateless comparator.

Required before multi-return claims: immutable call-time draft snapshot/revision and source-dependency provenance in each result; public retrieved payload summaries or a typed memory path conveying relevant assignments/costs/routes; separate current-draft features from result-draft features. Test order/handle permutations and repeated calls with changed drafts. Feature mapping need not be lossless, but demonstrated aliases that reverse correct action must be addressed or explicitly delimit the task.

The fixture uses a tiny independently enumerated executor and only public actions. It confirms differing payloads and identical feature vectors; it does not measure neural failure. Measured reviewer CPU0.012933680seconds, no children. Root should charge this inclusive fixture once.

**Formal reduction:** environmental `validate_subset` correctly rejects missing/duplicated categories, excess weight/cost and forbidden pairs; solver validity alone is not credited as world validity. However successful world action does not prove the submitted formal instance included all required constraints. A lucky omitted constraint can succeed. Record a separate evaluator-only reduction check against public goal/observed facts, and preserve omitted-item domain restrictions as such rather than silently correcting them. Restricting to inspected candidates can find a valid subset without proving global optimality.

**Costs and fairness:** root already identified that edges had weights but no travel penalty, making shortest-path optimality irrelevant, and that the catalog did not permit non-tool multihop motion. Those are under repair by the world owner; this review does not duplicate them. Neural computation is not yet a separate cost term in this world, so a `think` action's flat action price is only a proxy until controller execution cost is measured. Parent `process_time` around a callback measures solver CPU only in trusted in-process mode; isolated-child CPU must come from inclusive external accounting.

**Further checks before learned rollout:**

- Version invalidation should follow actual dependencies, not a global version for immutable subset facts; old route start/goal and world changes need explicit public compatibility features.
- A rejected environmental commit may reveal a permitted reason, but training masks cannot anticipate that reason from hidden facts.
- `use_return` must consume the immutable returned assignment and its original handle mapping. Later draft edits cannot reinterpret stored indices.
- Actor candidates currently receive hand-engineered summary features; all neural comparisons must receive the same features, and raw-JSON heuristics have a richer interface. Report this difference instead of claiming blanket information equality.
- Main world generation plants feasibility, not computational leverage or optimality. Empirical same-observation references must establish a cost crossover before claiming useful learned solver allocation.
