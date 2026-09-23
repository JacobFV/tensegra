# A05-development-v2: content selection without progressive identity collapse

This is a new exploratory generator contract, not a silent replacement of A04 or an unreported confirmation adjustment. The original A05 confirmation draft is retained with status `superseded_before_run`; no model outcomes were obtained. A CPU-only audit showed the independently sampled neighbor maps rapidly merge identities at long depth. We preserve the valid A04 fresh-code matching result but do not claim its long paths maintained distinct identities.

## Generator and exact guarantee

Represent nodes as `(attribute_group c, identity i)` internally during generation, then randomize observable node order. Neither internal group index nor identity index is an extra model input. For each relation r and ordered source/target group pair(c,b), independently sample a permutation `π[r,c,b]` over `i=0..N/K−1`. The supplied graph contains

`(c,i) --r--> (b,π[r,c,b](i))`

for every target group b. A public instruction `(r,code_b)` selects the sole neighbor with that fresh attribute code. Every relation has indegree and outdegreeK. Every source has one neighbor per code; each code labelsN/K nodes globally, preventing a terminal-code answer shortcut.

After the first instruction, all N possible starts span exactlyN/K terminal identities. Subsequent instructions compose bijections between the active groups, so that cardinality cannot shrink. Any two distinct starts in the same attribute group remain distinct at every depth. This is a **programmed generator guarantee**, not learned identity maintenance.

Independent CPU checks use three separate512-graph batches atN128/K8/D32. The v1 generator leaves only1.42/1.38/1.34 mean terminal identities at32; v2 leaves exactly16 in every graph at every tested depth. Distinct same-code start pairs survive512/512 in every v2 batch, versus78/64/64 in v1. First-instruction swaps change final entities481/478/479 of512 under v2, versus75/83/65 under v1. Target differences are separately counted because payload collisions remain possible. The reproducible generator-only tool and full counts are committed; no neural outcome was used to select this contract.

## Unchanged learned interface and screen

Keep the A04 architecture, width1024, four arms and1000-update training recipe. The sole task change is `generator=block_permutation_v2`. No route-label auxiliary, new primitive or new latent architecture is introduced. Query/key content projections start random; known-key address initialization remains disclosed in the keyed-neighborhood arm. Exact gather/hard masked neighbor attention is one algebraic baseline.

Development seed501, training91,000,000+step, evaluation92,000,000+group×100,000+offset. Batch16, N32/K4, depth cycling1–4, same AdamW learning rate.0003/weight decay1e-4. Eight monitoring checkpoints0/10/25/50/100/250/500/1000,256 examples per cell. Reuse the A04 clean/size/depth/composition/K-shift, instruction-swap, wrong-instruction, content-null and consistent permutation conditions. Add N128/D32/K8 and its paired frozen content-scale16 intervention on every arm. This is a content-score sharpness diagnostic, not a graph-strengthλ change. No weights update during intervention.

Primary development question is IID acquisition; secondary is retained task/path/mass under the noncoalescing long path. If IID is below95%, investigate acquisition. If a deep failure has correct argmax paths and frozen sharpening repairs it, localize soft value mixing rather than claiming missing identity. If a strong alternative dominates soft bias, retain that result. Do not extend a saturated training curve; any additional recipe requires a newly registered branch.

## Profile and budget

Before the screen, profile fresh mechanical seed599 for12updates and64 largest-conditionN128/D32/K8 examples. No profile outcome selects architecture or data. Root must release the GPU. Proposed profile cap60seconds. Use its measured cost to declare the four-arm development envelope before launch; preliminary cap300seconds. The preceding A04 four-arm1000-update study cost63.01seconds, but the new long-path evaluation adds work and must be measured.

After this exploratory screen, register a fresh three-seed confirmation only for informative acquired behavior. The superseded A05 confirmation seeds/data remain uninspected; they are not reported as failed model runs. Keep trained horizon1–4, tested32, public inputs, privileged suffix labels, exact guarantees, learned matching and statistical limits separate.
