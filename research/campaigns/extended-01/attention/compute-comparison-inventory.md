# Existing direct-attention comparison evidence

A06 is the strongest equal-public-information comparison: three paired seeds, identical graphs/instructions/identity coordinates, width1024,1000updates and16000 training presentations per arm. Soft graph bias, exact-gather neighbor attention and keyed neighborhood context all reach100% joint task/suffix with their prospectively fixed concentration policies. Keyed context receives supplied neighborhood addresses and an identity-initialized trainable address matcher; it is not an unconstrained graph-token transformer. The no-graph arm is an information ablation, not an equal-information baseline. A08 preserves these nine models and equal paired corrupt graphs; all tie at100% supplied-corrupt execution fidelity.

A11 adds dense unmasked attention over all public edge records followed by dense destination-node attention. It retains explicit identity/source/relation coordinates, static target-attribute joins, ordered instructions and programmed reverse execution. It accesses the same kinds of graph facts, but uses a different supplied representation and two-stage address decomposition. It is not generic graph-text understanding. Its one seed and6000 logical training updates are unmatched to A06's three seeds/1000updates; historical replay also adds charged computation. It reaches99.22%IID/91.21%moderate/19.73%joint, so incomplete acquisition prevents an accuracy-matched efficiency conclusion.

| Existing accounting | Soft | Gather | Keyed context | All-record A11 |
|---|---:|---:|---:|---:|
| Allocated parameters | 4,306,986 | 4,306,986 | 4,306,986 | 4,575,275 |
| Parameters in tensors receiving nonzero gradients at last update | 4,298,793 | 4,298,769 | 4,306,962 | 2,412,562 |
| A06 full-run CUDA peak bytes | 863,058,944 | 832,386,560 | 869,449,216 | — |
| A06 successful-policy joint forward seconds/1024events, seed mean | 3.639 | 3.680 | 4.074 | — |
| A08 clean joint forward seconds/1024events, seed mean | 3.484 | 3.536 | 3.816 | — |

A11 joint forward time is15.159seconds/512events (29.61ms/event;150.09ms/correct event). Its full continuation-run CUDA peak is2,105,541,632bytes. A08's pooled nine-model inference-run peak is437,634,560bytes. These peak-allocation scopes differ: training+evaluation versus frozen inference, whole-run maxima rather than per-cell reset measurements. They cannot establish an inference-memory ranking. Allocated/participating parameter counts do not measure effective capacity or work. Forward timing excludes much generation/scoring/export; full-process receipts separately preserve those costs. A12 instrumented timing will include observations and must not be treated as bare-kernel latency.

**Missing for compute-normalized superiority claims:** an acquired, matched-seed record baseline; common event populations and policy/accuracy targets; equal training/search compute budgets or full accuracy-versus-compute curves; repeated controlled warmup/timing with variability and throughput/latency distinctions; per-cell inference-only peak memory at matched batch sizes; measured FLOPs/MACs and tokenization/address-preparation costs. Current soft/gather timing differences are too small to claim superiority. Dense products remain even with supplied graph structure; no sparse-kernel benefit is demonstrated. Existing evidence supports a restricted engineering tie and descriptive implementation costs, not a compute-normalized attention advantage.

Sources: A06 report and per-run manifests; A08 per-checkpoint JSON/manifest; A11 final evaluation JSON/manifest. Inventory only: no new run, artifact rewrite or model change.
