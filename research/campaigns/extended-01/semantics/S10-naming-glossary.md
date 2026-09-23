# Naming guide for campaign reports

**Historical Stage 11** means the completed pre-campaign stage at baseline commit `123299a`: wide/short return continuation and eight-public-graph acquisition. Its source/results/gates are immutable.

**Campaign experiment S11** means the later semantic learning-rate continuation experiment: the original S03 parent is continued from131,072 to196,608 presentations with LR1e-5, compared with original constant-LR S04 midpoint. It is not Historical Stage11 and does not overwrite that stage's conclusions.

| Prefix/name | Meaning | Scope |
|---|---|---|
| Historical Stage1–11 | Pre-campaign research stages | Frozen prior results; refer to `research/stages/` |
| Campaign R01, R02, … | Return-access/use experiments | `extended-01/returns` |
| Campaign S01, S02, … | Public-text semantic experiments | `extended-01/semantics` |
| Campaign A03, A04, … | Direct attention experiments | `extended-01/attention` |
| Campaign P01 | Typed proposal acquisition/confirmation | Composition prerequisite |
| Campaign C01 | Bounded two-operation/returned-fact composition | Explicit consumed-interface contract |
| Campaign S10 | Frozen public-schema decoder diagnostic | Supplied output contract over learned predictions; not a training stage |
| S10-on-S11 | Same frozen S10 decoder applied to campaign S11 | Secondary engineered decoding, primary S11 unchanged |
| Campaign S12 | Three-paired-seed late-LR confirmation | Six endpoints; constant versus decay |
| S10-on-S12 | Same frozen decoder applied to S12 endpoints | Secondary supplied-grammar results; not primary learned recovery |
| S04 midpoint/u24576 | Original constant-LR196,608-presentation comparator | Different from S04 final/u32768 at262,144 presentations |
| `seed201` | Original semantic development initialization | Not three-seed confirmation |
| `seed701/702/703` | S12 independent parent initializations | Each parent forks into both LR arms |

Use **“campaign S11”**, **“Historical Stage11”**, and **“S10 secondary decoding”** explicitly in final prose. Report source hash, update count/presentations, split, and raw/calibrated policy beside numerical claims when ambiguity is possible. Supplied schema recovery is not fully learned graph induction; stage and experiment numbers do not imply promotion.
