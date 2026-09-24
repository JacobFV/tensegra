# S21 measured full-main cost proposal

Propose **1,770 seconds** for the unchanged paired main. The conservative planning estimate is **1,723.998 seconds**, including200 seconds of preparation/process reserve. Additional margin is46.002 seconds. With the completed profile charge29.54, the proposed total allocation is1,799.54 seconds, within the1,800-second branch envelope by0.46 seconds. The margin beyond reserves is narrow; a timeout remains incomplete evidence and does not authorize automatic extension or reduced evaluation support.

Profile archiveb4ad6074; manifest SHA527a3f914a6c24757454545e88d854a439c2222f27381449c565c7d887e0b03b. Only measured timing fields and public-input lengths were inspected for this forecast. Model quality does not select recipes or budgets. Exact arithmetic and profile timing sums are in `S21-measured-main-cost.json`.

| Main term | Original seconds | Broad seconds |
|---|---:|---:|
| Optimizer,20-update timer ×204.8 |233.612|209.737|
| Greedy,464 batches × worst measured160-step time |386.289|386.289|
| Teacher forcing,TRAIN sums ×2 +DEV sums ×64 |11.932|8.480|
| Export, population-scaled then doubled |31.601|30.882|
| Evaluation packing/scoring residual, scaled then doubled |100.451|96.755|
| Initial checkpoint + three full-state checkpoints |7.009|6.354|
| Fixed model setup |0.950|0.661|
| **Arm subtotal** |**771.844**|**739.157**|

Paired subtotal1,511.001, doubled measured preflight7.967, unclassified outer residual5.031, and additional reserve200 yield1,723.998.

Each arm evaluates four TRAIN128 plus four DEV3,584 populations: `4 × (128+3584) / 32 = 464` full batches; pair928. Both actual profile stress batches reached74 visible tokens, verified from their frozen public-pool indices and cache metadata. Original stress measured0.832519 seconds and broad0.406493 seconds for160 cached own-prediction steps/batch32. Use the **larger0.832519 for both arms**, avoiding optimistic reliance on that runtime disparity or on naturally short/invalid outputs. Ordinary main decoding still stops at predicted EOS. No gold lengths are supplied.

TRAIN teacher-forcing/export/residual profile sums scale2× (four rather than two curves). DEV sums scale64× (four3,584 versus two112 populations). Export and residual terms then double to allow valid-graph payload, packing and codec/metric work absent from short outputs. These are explicit engineering allowances, not measured upper bounds. The preflight allowance includes larger DEV target preparation; the extra200-second reserve covers that uncertainty, untimed update preparation, allocation, and process variance. Checkpoint scaling distinguishes initial empty AdamW state from the three full-state checkpoints. Main repeats no separate stress benchmark.

All scientific settings remain frozen: paired scratch2101,4,096updates/batch8, four fixed curves, unchanged S19 learner/loss/schedule, both exact TRAIN128 panels and all seven DEV512 cells. Broad data increases tokens and serialized work; no equal-FLOP claim is made. Root may choose a different budget explicitly before main if the remaining margin is judged inadequate, but no condition is silently shortened. Main allocation/freeze, independent timing review and explicit GPU release are still separate steps. S21 sealed confirmation is not opened.
