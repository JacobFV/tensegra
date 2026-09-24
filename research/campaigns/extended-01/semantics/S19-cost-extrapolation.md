# S19 prospective phase-based main cost formula

Planning before profile outcomes; no GPU launch or main cap is authorized here. Use the released profile's complete outer receipt and phase timings after independent mechanical review. The fixed main remains4096updates, four checkpoints, actualTRAIN128 and all2048DEV at each, teacher-forced diagnostics plus naturally stopping greedy decoding. Do not shorten a condition to fit a forecast. The branch envelope includes charged profiling and any separately registered extension.

There are272full evaluation batches in the main:4×(128+2048)/32. Profile has12batches across two checkpoints, plus the separate160-step batch32stress. Training/public caches confirm maximum64public tokens and123gold records; the stress uses32longest public inputs and160own-prediction steps, so source-length/record-cap coverage is adequate. It is a computational stress, not a quality result. Natural-stop profile outputs can be extremely short or invalid and must not determine the expected acquired-output length.

Let `P` denote measured20-update optimizer seconds; `W` the complete batch32stress decode seconds; `T` and `D` the sums of profile TRAIN and DEV teacher-forced seconds respectively, over both checkpoints. Let `ET,ED` be corresponding export sums, `RT,RD` evaluation residuals (`total - teacher_forced - free_running - export`), and `C0,C20` checkpoint export/write/hash seconds. A transparent planning decomposition is:

| Main term | Prospective calculation | Limitation |
|---|---|---|
| Optimizer work |204.8×P|Includes the profile's first-step startup; inspect fixed corpus padded-length coverage, not losses, before extrapolation. Excludes preparation/hash/logging outside the timer.|
| Teacher-forced TRAIN |2×T|Same128examples, four versus two checkpoints.|
| Teacher-forced DEV |64×D|32times population, twice checkpoints. Profile combines16/cell in batch32; main groups full cells. Report batching/padding differences.|
| Free-running decode |272×W|Assumes every batch runs160steps; replaces, rather than adds, scaled natural-stop profile decoding. Cached encoder and per-step host argmax transfer are already included. Not a strict latency bound for every generated record/status path.|
| Exports |2×ET+64×ED, then explicit payload allowance|Early invalid outputs lack dense predicted tensors. Main valid graphs can add a second packed graph plus canonical records; use a stated2×payload allowance initially, or measured CPU-only synthetic full-payload export evidence. Do not call it measured main export.|
| Evaluation residual CPU work |2×RT+64×RD, plus explicit valid-graph allowance|Includes record conversion/metrics/packing/state work inside each evaluation timer. Invalid early profiles underexercise valid-graph parsing and metrics.|
| Checkpoint writes/hashes |C0+3×C20|Scratch checkpoint0 has no AdamW moments; multiplying both profile saves by2 would undercount three full optimizer checkpoints.|
| Setup/preflight/import/other overhead |Fixed measured setup plus separately identified residual and reserve|Main builds2048dense DEV targets versus64in profile. The pre-evaluation state hash is outside its evaluation timer. Batch assembly/logging and other tails must be included.|

The stress record-only export is useful evidence for160records×32payload, but it does not replace ordinary graph/target export; adding it to every batch without accounting for existing raw-record exports would double-count. Conversely, it cannot measure full valid tensor packing. The initial2×export/residual allowance is an engineering contingency, not a mathematical upper bound. Keep each allowance visible rather than absorbing it into an unexplained global multiplier.

Reconcile the profile outer clock against all measured phases before projecting: optimizer + setup + preflight + all evaluation totals + both checkpoint phases + stress decode/export. Their remaining difference captures wrapper/import/setup and uninstrumented work; do not drop it. For main, add an explicit training-preparation/logging reserve (or a CPU-only measured4096batch preparation pass) instead of blindly multiplying this mixed residual by204.8. Report best-supported central estimate and conservative cap proposal separately. Whole-main memory, checkpoint volume and natural-stop output lengths remain profile/audit considerations.

A4000second S18 cap was not an estimate of its1940.08second actual charge; likewise no S19 cap should be described as expected runtime. If the profile-derived full condition costs exceed the remaining S19 development envelope, seek a separately justified redesign or defer; do not reduce widths, update counts, curves, populations or decoding cap silently. Root alone allocates/freezes/releases the next job after review.
