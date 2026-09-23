# Stage 6 development journal

## Initial design and provenance

Stage 5 release `1082074` is the unchanged baseline. Work proceeds in the isolated `feat/thinking` worktree. The requested stage spans independently measured recurrent computation, semantic graph compilation, candidate-local readiness, protected execution, reintegration and supervision annealing. Four-phase recurrent weights are shared across microsteps; workspace feature rows are not an ontology of thoughts.

TCN source is pinned to `018c9ce9286fa4961292fe1e024b49fcd7e2dd7f`, engine0.2.0, generator manifest1, MIT. Read-only exploration verified the engine can run without site packages. We will vendor a bounded dependency closure and preserve licensing/source hashes. The same seed is not sufficient to establish equivalent semantics across every language-dependent lesson: build one semantic construction and render it repeatedly. Identical synonym surfaces must not count as independent renderers.

The Stage5 batch-lowering/direct-lifter path cannot establish reintegration. Stage6 therefore uses a new incremental protected session whose inputs exclude privileged task metadata. A typed return is injected into the current workspace, followed by a recurrent update before another action or emission. No extra exact-result argument reaches the output head.

For interpretability, actual TCN semantic-supervision experiments and controlled progressively disclosed execution experiments are separately named. They share canonical graph and recurrent model interfaces, but a controlled supplied numeric-memory interface is not evidence of free-form language parsing. Hidden graph targets never become actor runtime graph memory. Structural attention can use public or predicted structure only, with that difference explicitly recorded.

Owners: semantic compiler/vendor, recurrent model, protected runtime, controlled tasks, experiment runner. Independent reviews will inspect leakage, causal return use, emission and losses before any main experiment is frozen.

## Wiring pilot and controls before main freeze

The integrated source82524 snapshot passed 413 tests. The actual TCN wiring pilot completed all12 seed/arm runs with eight updates each. It is retained as a pipeline check, not a learned-generalization result. Before main runs, metric support counts/null denominators and initialization/checkpoint hashes were added without modifying its frozen checkout.

Review hardened the model's topology predictor to depend on current public context, preserved typed relation strengths, and added query/key null grounding. A graph predicted only from random identifier features could not infer the input's relationships. Grounding supervision is candidate-routed rather than forcing every workspace component onto every operand.

Halting now requires a correctness gradient through survival/hazard-weighted per-microstep task losses, not only end-state classification plus pressure to stop. Pure fixed/recurrent neural controls receive identical complete evidence from the start. Progressive local/global/runtime controls share an exogenous evidence/context-release schedule; access to the context cue must never depend on successful symbolic execution. Same-checkpoint event interventions and post-recurrent semantic round trips isolate reintegration from its encoder alone.

The language main budget is fixed at 240 training constructions,48 held-out constructions,256 updates,batch4,three seeds/four arms, and checkpoints0/64/128/256. This choice follows pilot resource measurement rather than an observed main/OOD result. Exact TCN rewrites remain outside this language-only track; the controlled rewrite workload is separately reported.

## Controlled study budget fixed before outcome inspection

The two-arm/eight-update timing run completed in about14 seconds; per-update training was .58 seconds at batch4. It is a wiring/timing pilot, not evidence of acquisition. Main controlled configuration is fixed to15 variants×3 paired seeds,256 updates,batch2,depth1–4 training,12 held-out examples per condition, final depth4/8/16/32 and explicit size/surface/motif/composition/ambiguity probes. Mid-training checkpoints use shallow validation; autonomous evaluation has a common80-step hard budget. The readiness operating threshold is .8, with separately labeled frozen threshold probes. Soft halt biases stay fixed when the hard budget changes.

Three seed shards may run concurrently on the linked119GiB host, each limited to two CPU threads; at most six controlled-training cores and a few GiB are expected. This resource choice reduces wall time without changing any batch, update or seed. Local training remains disabled. Each shard retains its resolved configuration/raw metrics/source and checkpoint hashes; merging must validate that configs differ only by seed list and that all expected seed/variant runs are disjoint and complete. The actual TCN main remains on its separate frozen4c365 snapshot; controlled-source changes do not alter it.

## Controlled source freeze and language outcome

Frozen controlled source is `c258785`, including runner `02be34e`. The full remote regression passed452 tests and6 subtests in5.60 seconds. Three seed shards started from the same immutable checkout; each uses the previously fixed budget. The independent artifact analyzer may be updated separately and records its own source hash.

Final review found and corrected a halting boundary mismatch: free inference could emit before a new action, while the training hazard had been masked after that action. Both now use pre-execution eligibility. Only newly injected returns require another recurrent pass; idempotent duplicate events do not keep reinjecting old values. Learned protected comparison writes threshold a regressed Boolean at.5 and never fall back to exact arithmetic. These were correctness fixes before main outcome inspection.

The language main completed all12 runs at256 updates and was archived in `beb1cf4`. Choice performance remains around chance and exact canonical graphs are never recovered. Privileged supervision improves node presence but typed-edge precision stays below1%; this is not successful semantic grounding. The report separates renderer exposure and partial graph metrics from strict exact-graph accuracy. No larger-model or interpreter integration is justified by this result alone.

## Completed controlled study

All45 controlled runs completed at256 updates, with the slowest seed shard taking1633.46 seconds. The merged grid contains780 evaluation cells,11520 training rows and225 checkpoint hashes. The independent generator audit reproduced all9360 evaluation example hashes and all training-data hashes. Final-source regression including the analyzer passed454 tests and6 subtests in4.89 seconds.

The local free policy settles on two microsteps and produces no correct complete execution trajectories. Its task accuracy at depths4/8/16/32 is1/36,0/36,1/36,2/36. These occasional answers do not demonstrate execution. Oracle-minimal scheduling preserves every exact trajectory at depths4 and32, while learned task readout is only1/36 at each depth. The full80-step oracle preserves the same symbolic correctness but has0/36 correct task outputs in each condition. This is a failed acquisition result at the declared budget, not a successful latent-to-symbolic-to-latent loop.

The artifact-only lowering audit distinguishes exact arithmetic on actual operands from correct task lowering. Across final primary conditions, executed arithmetic is correct on its actual input values in1215/1215 cases, while recognized execution bindings match the task in0/1111 cases. These populations include repeated episodes across controls; they are not independent statistical trials. Free-policy return interventions with no injected events are uninformative about causal reintegration. Auxiliary removal cannot establish loss of an acquired skill when the fully supervised interface never worked reliably.

Next scientific work should establish acquisition on a smaller binding-and-return problem, inspect emission/readout objective balance, and demonstrate value preservation after a single return before increasing semantic breadth or using a pretrained decoder. Those are follow-up hypotheses, not changes to the completed frozen study.

## Integration

The complete code, raw artifacts and reports were fast-forwarded into main at `15bde6b` and pushed. A fresh remote test of that merged revision passed454 tests and6 subtests in4.83 seconds. Independent final review `02b1743` verified all45 controlled archive files and2080 aggregate/conditional statistics;70 local documentation links resolved. The clean agent-owned local worktree was removed after integration. Frozen remote experiment checkouts and checkpoint binaries remain available for reproduction; their committed hashes identify them. Completing this implementation plan does not mark the scientific acquisition criterion as achieved.
