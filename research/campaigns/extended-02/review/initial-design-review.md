# Extended-02 independent initial design review

Reviewed baseline `65d44ae6`. This is a prospective design review, not a competence or experiment approval receipt. The root owns all scheduling. No experiment or GPU work was launched by this review.

The previous claim map, C04, R04, A14, S20–S22 and final audit/addendum support restricted learned interfaces, not an autonomous controller. C04 supplied a schedule and one return; R04 has balanced numerical blind spots; A14 supplies addressing and relation order; S20 fits fresh known motifs but S21/S22 leave a relation-generation transfer boundary. Preserve these distinctions in new baseline labels. Historical audit receipts were inspected rather than regenerating their data.

## Recommended first environment

Use a small workshop with public goals, typed partial observations, inventory and a directed route graph. A goal requires choosing compatible components, assembling, then delivering. Some compatibility or route facts require paid inspection. Start with two to four components and short routes; expand only after a policy can complete easy episodes. Keep environmental actions distinct from computation and let failed commits reveal only the simulator's permitted feedback.

The actor constructs a finite-domain compatibility instance from observed records, chooses search budget, stores the result, constructs a route problem, and finally binds environmental actions to selected returns. It can also inspect, use cheap filtering/greedy selection, perform one latent step, verify, or abstain. CSP and shortest path give complementary exact primitives; constrained subset selection can reuse the explicit constraint builder as a separately typed optimization operation. Small bounded exhaustive solvers are acceptable initially if independently checked and charged; a broad solver library is unnecessary.

Do not silently turn candidate enumeration into an exact planner. Public action schemas may enumerate entities and type-valid choices, but must not precompute the successful assignment, complete constraint set, cheapest route, next optimal operation, readiness time, or budget. Explicitly identify any supplied compiler from observations to formal constraints. If it exists, that branch tests scheduling, not semantic reduction.

At least one episode family must require two concurrently available result records, with random opaque handles, provenance and world versions. Add a distractor valid return for a different goal. Otherwise copying remains the old single-return upper bound. Track both the actually selected result's validity and whether it satisfies the original goal.

## Early discriminating checks

1. A same-observation capable reference must solve medium episodes within the public cap. An omniscient plan only checks generator validity. Record feasible-but-budget-limited cases separately from infeasible worlds.
2. Establish neighboring easy/medium/hard cells and a cost crossover: cheap direct action wins on easy cases, a useful solver wins on medium cases, and neither always-largest nor always-abstain maximizes utility. Measure this before population training.
3. Validate formal reductions separately from certificates. A valid assignment to omitted constraints is not world success; timeout or unknown never means infeasible. Verify returned paths, assignments and actual environmental transitions with independent routines.
4. Counterfactual pairs must preserve observations until the changed fact is inspected. Hidden changes must not alter action masks, padding, candidate counts, filenames, RNG streams or return order before revelation. A simulator's private optimal action is never a public feature.
5. Reorder opaque entity/result handles consistently. Replacing a result with a stale/wrong result must score original-goal correctness and supplied-record consistency separately. Version invalidation is public program semantics, not learned calibration.
6. Candidate policies receive serialized observation/API data only, not Python environment objects. Keep evaluation seeds, validators, costs and hidden-state files inaccessible to their interface. Bound solver instances before dispatch; use child resource limits and terminate/reap timeouts.

## Population comparison to freeze after profiling

Use three independently initialized **population processes**, each containing six members. Pair each process with a six-member independent multistart control using the same initial checkpoint/hyperparameter bank. Descendants within a process are not independent seeds. One conventional learner with the same aggregate work is an additional reference if affordable; architecture screening should precede, rather than multiply, the confirmatory population comparison.

A simple synchronous PBT adaptation is easier to audit on one time-multiplexed GPU than asynchronous selection. After a fixed training tranche, evaluate every member on the same fresh development tranche. Replace the bottom two from the top two only under the registered selection rule; inherit model and optimizer state, then mutate bounded learning rate/exploration/curriculum parameters. If optimizer reset is preferred, declare it for every replacement and isolate its effect later. Record source checkpoint hash, parent, replacement, mutations, data exposure and compute. Do not inherit replay buffers unless explicitly declared and accounted.

Multistart receives the same six initial configurations, aggregate gradient/rollout allocation, development tranche schedule and final best-development selection. It keeps training its own members without inheritance or mutation. Retain a fixed-curriculum comparison before claiming adaptive-environment benefit. Equal updates alone are not equal resource use when episode length or solver demand differs; publish both work-normalized and measured total-cost results. The PBT selection/evaluation overhead must count.

Freeze finalists before sealed confirmation. Suggested required cells: fresh familiar composition, two independently omitted ordered primitive combinations, longer dependency chain, high versus low public computation price, reduced solver budget, and obstacle/replanning. Use at least 512 fresh semantic episodes per cell and paired interventions on those same episodes. Primary outcome is verified utility with raw success/cost frontier retained. Report all three process differences; event bootstrap conditions on those processes and cannot turn three lineages into a large seed sample. Development selection is not confirmation evidence.

## Resource accounting requirements

GPU occupancy includes training, rollout inference, evaluation, failed jobs and exports while holding the slot. CPU-core time must use process user+system accounting, including reaped solver descendants; parent `process_time` alone misses subprocesses. Record inclusive job roots and do not add their nested child totals again. Keep wall time and deterministic solver expansion counts as separate quantities. Charge teacher generation, environment rejection sampling, selection, verification and cache preparation. Pin thread counts and solver limits; projected worst-case population/evaluation CPU can consume the 48-core-hour ceiling before GPU does.

Reserve confirmation capacity before increasing population size. A six-member population is an implementation choice, not proof that evolution is useful. If PBT does not beat matched multistart, retain the result and continue another question rather than tuning on the sealed test.

## Method provenance

[PBT, method §3](https://arxiv.org/html/1711.09846) alternates within-member optimization with exploit/explore, copying promising trained states and modifying hyperparameters. The proposed synchronous single-device schedule borrows that mechanism, not the original asynchronous wall-time claim. Weight inheritance plus actual replacement is necessary to distinguish this experiment from independent seeds.

[POET, method §3](https://arxiv.org/html/1901.01753) maintains environment–agent pairs, generates environments under a minimal criterion, optimizes agents and tests transfers. A bounded learning-progress archive here is only inspired by those ideas; without paired environment reproduction and transfer it should not be called a POET replication or open-ended evolution.

[PAIRED, method §4](https://arxiv.org/html/2012.02096) uses a protagonist/antagonist return gap to train environment generation. An approximate best-reference-minus-policy score here is a regret-inspired heuristic, not true optimal-policy regret or the paper's equilibrium guarantee. Preserve an external anchor distribution; maximizing learner failure alone rewards impossible tasks.

Reading/audit JSON parsing used 0.001486 measured parent CPU seconds, no children. Git/filesystem and document authoring are administrative overhead, not experiment workloads. No training/inference/solver workload was run.
