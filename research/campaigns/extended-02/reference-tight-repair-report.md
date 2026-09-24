# E02 reference repair: independent archive audit

Frozen run source: `a11bb1e6`. This is development evidence, not learned-policy
confirmation. The initial launch missing `--config` remains a failed launch;
the successful retry is the evaluated population. This audit does not replace
that process-accounting record.

## Paired result

All four arms solve all512 standard-budget instances. On the same512 hidden
worlds under the128-work public budget:

| Public call catalogue | Original reference | Route-fallback v2 |
|---|---:|---:|
|16/128/1024 only|138/512 (26.95%)|455/512 (88.87%)|
|Above plus exact remaining budget|501/512 (97.85%)|501/512 (97.85%)|

Fallback rescues317 original failures and breaks no original success. Remaining
budget rescues363 original failures and breaks no original success. Relative to
fallback alone, remaining budget adds46 successes without a reversal. The two
remaining-budget arms have identical success indicators, including the same11
failures. Repeated policies and public conditions do not create additional
independent support:4096 archived episodes cover512 underlying world instances.

This narrows the original low score substantially. A legal allocation action
and a basic route fallback repair most failure without learning or architectural
change. It is not evidence of emergent metareasoning or evolution. The original
frozen outcomes remain valid under their more restrictive catalogue and policy.

## Remaining11 failures

Every remaining failure has the same observable mechanism:

1. local greedy selection does not produce a feasible full subset;
2. explicit subset search with16 work times out without a validated incumbent;
3. fresh restarted search with112 work also times out without an incumbent;
4. all128 work is spent; the policy abstains before selecting a subset.

No routing call, returned-value mixup, rejected feasible selection, or false
infeasibility declaration explains these11 trajectories. Both calls have
recorded correct formal reductions; certificate validity is false because no
feasible result was returned. The independent audit did not rerun those solvers.

This is a search-budget/allocation boundary of this reference, **not proof that
these tasks require more than128 work**. The second call restarts, repeating its
initial search; different ordering, a single larger initial allocation, a
resumable solver, or a better cheap feasible-selection policy could differ.
Those would be separately registered interventions. The traces alone cannot
credit their counterfactual success.

## Comparability and verification

The independent audit regenerated each frozen world from its archived source
and seed, checked all2048 condition/world hashes, and compared all fields after
removing only public `work_limit` and `include_remaining_budget`. All remaining
hidden/world parameters match across the four conditions for each seed.
Address seeds also match within each paired world, independently of mode.

It reconstructed every cell count and all28 paired2×2 tables from raw histories;
recomputed action/observation/work/travel/compute costs and utility; checked all
3643 successful final selections against category, uniqueness, capacity, funds,
and incompatibility constraints; and checked destination/delivery/verify events.
Every recorded formal-reduction result is correct. Complete path replay and
solver certificate reexecution were not performed and are not implied.

The reproducible script is
`research/tools/campaign02_tight_repair_audit.py`; its receipt records archive
and summary hashes and about0.403 CPU seconds. No model or solver inference ran.

## Teacher recommendation

The remaining-budget reference is a substantially more useful structured
bootstrap teacher than the original quantized-action policy. It uses the same
public observation and catalogue as the learner; its schedule, greedy rule,
constraint choices, and return-addressing rule are supplied privileged action
labels. The learner must still acquire closed-loop control, not inherit the
heuristic code at deployment.

Keep the11 failures and their abstentions in any teacher-quality accounting.
Do not call the teacher uniformly reliable, or train only successful examples
without disclosing that selection. Retain route-fallback v2 as a recovery
reference under tool unavailability; it gives no additional success on this
particular remaining-budget population. Further autonomous-policy evaluation
must include cost shifts and held-out episodes instead of reusing these
inspected development anchors as confirmation.
