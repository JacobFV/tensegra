# A08: frozen joint edge replacement diagnostic

Prospective, inference-only robustness diagnostic after A06. No historical gate change, retraining, threshold fitting or promotion. Preserve all three A06 seeds and graph interfaces under their declared successful policies: soft/gather content16, keyed context address16+content16. Exact checkpoint hashes are frozen in config.

## Public intervention

For each typed relation and pair of public attribute groups, the v2 graph supplies a bijection between equal-size groups. Select a fraction of source rows in each block and cyclically rotate their existing destinations. Every selected edge is removed and one different edge inserted. Both in- and out-degrees remain unchanged; each source still has exactly one neighbor with every public attribute. This is a **joint missing-plus-spurious edge replacement**, not pure deletion, and is not necessarily representative of arbitrary graph noise. No hidden target/path determines replacements.

Fractions0/.10/.25/.50 are requested. At both tested shapes N/K=16, the smallest nontrivial permutation changes2/16edges, so requested10% is realized12.5%; the others are exact25/50%. Export actual removed and added fractions for every event. Minimum2 rule is explicit. All arms and seeds receive the same supplied corrupt graph. Clean labels remain unchanged; additionally compute exact symbolic traversal under the corrupt graph and agreement with that supplied computation. Clean exact pointer execution is the programmed100% reference, not learned behavior.

Missing edges can make the clean answer unidentifiable. A loss in clean-answer accuracy is therefore sensitivity to altered information, not proof that a method should infer missing true edges. Compare learned output with the corrupt-graph exact result to separate routing fidelity from factual graph correctness. Pure deletion and free-degree spurious addition are **not covered**: fixed-K interfaces would otherwise pad nonexistent edges. No such padding is introduced.

## Population and metrics

Use1024fresh graph events per shape, paired across fractions, all9checkpoints and policies. ShapesN64/D8/K4 andN128/D32/K8. Base seed181M+shape×100k+offset, replacement seed182M+shape×100k+offset. They are new diagnostic populations and not new confirmation used for model selection. No scale or model can be changed after inspection.

Measure exact clean-target task, corrupt-pointer task, learned agreement with corrupt-pointer output, complete suffix, diagnostic clean path, supplied/clean edge mass, actual replacement fraction, paired changed-answer support, and hardware cost. Preserve per-seed outcomes. Store public targets/topology once in a shared lossless archive, with per-checkpoint predictions and summaries separately. Check input checkpoint SHA and initial/final tensor equality; optimizer steps are zero. Replay fraction0 as an unchanged-policy control. It is fresh data, so it need not reproduce A06's exact historical counts.

## Budget and stopping

CPU tests establish degree/code support preservation, requested-versus-realized fraction, deterministic replacements and exact fraction0 replay. Small width32 is a labeled mechanical fixture only. Main width1024 is unchanged.

Profile one frozen checkpoint on64largest-shape events and all4fractions, cap60seconds with coordinator release. Main preliminary estimate150–250seconds from A06 forward timing, cap300seconds subject to measured export cost. Root must release GPU explicitly. If the profile shows contract violations or excessive cost, stop this branch before main evaluation. No experiments launch automatically. No confirmation or composition gate follows from this robustness diagnostic.
