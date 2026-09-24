# Bounded existing-generator support audit

Protocol committed as `e692a283` before execution. Exactly8,192 generator attempts at each arity2/3/4/5,32,768 total; no extension, campaign split reservation, model evaluation, or historical/confirmation lookup.

**Seven cells exceeded512 observed alpha-distinct constructions. The2×3 cell did not:496 after4,136 cell draws.** The latter is a bounded shortfall, not proof that the full family contains fewer than512. Do not promise512 from this cell without a separately authorized support investigation or change the requested sample count silently.

| Cell | Cell draws | Alpha unique | Lexical unique | Alpha unique at arity attempts256 /1024 /4096 /8192 |
|---|---:|---:|---:|---|
|2×3|4136|496|4121|113 /293 /446 /496|
|2×4|4056|3384|4056|119 /489 /1852 /3384|
|3×3|4152|3109|4150|141 /520 /1800 /3109|
|3×4|4040|4023|4040|113 /480 /1987 /4023|
|4×3|4094|3933|4094|134 /528 /2007 /3933|
|4×4|4098|4095|4098|122 /492 /2050 /4095|
|5×3|4033|3999|4033|129 /501 /1991 /3999|
|5×4|4159|4158|4159|127 /523 /2097 /4158|

The four curve checkpoints count total attempts at that arity; facts3/4 are drawn naturally, not forced. The2×3 curve slows markedly: only50 additional alpha structures appeared in the last4,096 arity attempts. Its4,121 lexically unique examples are not4,121 distinct equality/ordered-binding constructions.

All32,768 independent answer checks and strict codec/public-copy checks passed. No same-public-text conflicting target was observed. Public lengths remain30/34/46/54/54/64/62/74 in table order. All outputs use the existing finite nonlexical vocabulary(parent,unify,null), existing identities, relation types and ordered slots. Largest observed graph42nodes and143records fits the existing128/160 capacities. There is one occurrence-tree motif per cell; alpha diversity concerns equality/sharing and ordered identity assignments, not thousands of tree shapes.

Availability before exclusions is not availability after disjoint TRAIN/DEV/confirmation reservation. The probe never reads existing reserved datasets and therefore cannot certify those exclusions. Holding out3×4 leaves seven candidate TRAIN motifs, of which six demonstrate more than512 alpha structures in this probe and2×3 falls short. Facts2 remain unavailable unchanged.

Reproduce with `research/campaigns/extended-01/semantics/S21-support-8192.py`. Counts/source/protocol hashes and curves: `research/results/campaign-01/semantics/s21-support-8192.json`. Actual CPU wall time **29.064549s**; zero GPU occupancy. No further sampling was performed after the prespecified bound.
