# Existing-generator motif feasibility (preparation only)

**The requested nine-motif grid is unavailable without a versioned generator change.** Current hardened unification creates one matching fact and `randint(2,3)` near misses: facts3 or4 only. The existing unhardened branch creates exactly one fact. No existing setting creates two facts. Difficulty changes arity, not this fact-count support. Thus arity2..4 × facts2..4 cannot furnish eight TRAIN motifs while holding out3×4 under unchanged semantics/distribution.

## Bounded fresh probes

Each feasible cell received a separate fresh namespace and exactly256 generator attempts, selecting only that cell's fact count. Total1,536 generated constructions. No old or new confirmation file was read, no dataset was reserved, and no model was run.

| Cell | Selected draws | Alpha-distinct | Lexically distinct | Alpha repeats | Public tokens | Node range | Record range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2×3 | 122 | 104 | 122 | 18 | 30 | 20–24 | 64–72 |
| 2×4 | 124 | 123 | 124 | 1 | 34 | 23–27 | 75–83 |
| 3×3 | 124 | 121 | 124 | 3 | 46 | 25–28 | 82–88 |
| 3×4 | 110 | 110 | 110 | 0 | 54 | 29–32 | 97–103 |
| 4×3 | 137 | 136 | 137 | 1 | 54 | 30–32 | 100–104 |
| 4×4 | 129 | 129 | 129 | 0 | 64 | 35–37 | 119–123 |

All observed examples fit the unchanged128-node/160-record codec. The existing generator's worst bound over arity≤4/facts≤4 is at most41nodes and132records:30 term occurrences plus at most10 identities plus scope, with29 tree edges,30 contains,21 refers_to and10 declares. The measured maximum is37nodes/123records. This mechanical capacity is not an acquisition result.

The only equal-length overlap in these six cells is **3×4 versus4×3 (54 tokens)**. Newly added arity-two cells are lengths30/34, not counterexamples at54. Broader training may help compositionality, but this specific diversity extension does not independently break the length-to-shape association for the held-out54-token cell.

## Semantics and diversity

The pinned vendor manifest was verified. For every1,536 draw, an independent graph traversal extracted the query variable, ordered pattern arguments and facts, checked that exactly one fact matches every bound argument, and recovered the generator's answer and option index. For746 selected draws, public copying and the existing strict record decoder were checked. No same-public-text incompatible-target collision occurred in the probe. Inputs, source hashes, seeds, counts and runtime are recorded in the JSON.

Each cell has one occurrence-tree motif; all have the same shallow nesting depth. Alpha-distinct variation reflects **identity equality/sharing structure and ordered bindings**, not merely different names, but does not mean hundreds of different tree shapes. Lexically distinct counts alone overstate diversity, particularly2×3, where18 of122 selected draws repeat an alpha construction.

A256-attempt probe cannot establish512 distinct examples per cell. The higher2×3 collision rate makes saturation/exclusion checks especially important. The other cells show low collision at this support, but512 after historical TRAIN/DEV/confirmation exclusions remains unverified. No extrapolated available-corpus count is asserted. Existing six motifs would leave only **five TRAIN motifs** when3×4 is held out.

## Decision implications

Do not launch the requested nine-motif dataset under the current generator. A five-motif unchanged-generator experiment would be a different declared design and needs a bounded support reservation/exclusion audit before promising512/cell. Alternatively, adding two-fact draws requires an explicitly versioned generator extension with independent correctness checks; this feasibility task did not implement or authorize that change.

Reproduce using `research/campaigns/extended-01/semantics/S21-generator-feasibility.py`. Results: `research/results/campaign-01/semantics/s21-generator-feasibility.json`. Actual generator/audit wall time: **1.022915s CPU**. No GPU use, source semantics changes, S20 outcome inspection, or main allocation.
