# E05 matched acquisition: independent raw audit

The fixed600-update endpoints reconstruct to **57/128 lightweight** and **46/128 recurrent** verified successes. Mean utility is.35963140625 and.246318125 respectively. All12 archived development panels reproduce allocation success/cost/utility; raw hashes, training interval continuity and fixed-finalist selection pass. No inference was rerun.

| Update | Lightweight successes/128 | Recurrent successes/128 |
|---:|---:|---:|
|100|70|1|
|200|64|80|
|300|28|50|
|400|57|46|
|500|65|52|
|600|57|46|

These are the **same128 development worlds** reused six times, shared between arms. Each arm has one initialization and4800 training episodes. The runner mode is `single`: six sequential100-update allocations train one learner, not six evolved agents. The fixed endpoint must not be replaced with the earlier80/128 recurrent or70/128 lightweight result. Development curves are nonmonotonic; no new confirmation or superiority claim follows.

Final paired outcomes, recurrent→lightweight: both wrong66, both correct41, recurrent-only5, lightweight-only16. Episode support is128, not256 and not1536.

## Dominant failure boundary

Lightweight has71 failed episodes:57 end in solver timeout,4 repeatedly build routes,4 abstain, and6 end in rejected pending commitments (capacity1, compatibility1, funds4). Recurrent has82 failures:76 end in solver timeout,5 build routes,1 retrieves.

Every terminal-timeout episode in both arms uses **budget16 on every call**. No budget escalation occurs in those57/76 trajectories. Submitted reductions are correct wherever a call occurs; there are25 lightweight and18 recurrent episodes without calls. The primary observed limitation is closed-loop budget escalation after timeout, alongside smaller commitment/fallback failures—not demonstrated incorrect executor semantics.

The constructive feature-collision proof remains independently valid, but it does not measure what fraction of these failures it causes. Do not claim these timeout loops are proven consequences of that collision. The broader teacher and new input distribution differ from E03, so the decrease is not a controlled forgetting result either.

E05 uses supervised public-teacher bootstrap; training-success metrics refer to teacher rollouts. The raw development results above are learned closed-loop actions. Both use1024 width and matched training episodes/updates; actual model work and parameter counts differ. Exact teacher target collisions and omitted public relationships motivate E06, but a lossless codec still must earn acquisition and cost after profiling.

Pause large population training on this known insufficient teacher-feature interface while its versioned repair is evaluated. Preserve E05 unchanged. Test whether E06 can consume complete observed relations and escalate budgets on fresh development episodes before evolutionary selection is interpreted as resource intelligence.

Audit cost: full archived panels4.115171216 + budget-localization0.441025392 = **4.556196608 CPU-core seconds**. No GPU/solver/model inference. Compact JSON retains input bindings, all curves, failure examples and paired support; no large repeated vector exports.
