# C01 N1 development: strong clean answers, downstream operand-order weakness

The registered calibration rule selects learning rate.0003 at4000updates, with2031/2048 calibration answers. Fresh answer accuracy is4056/4096 (99.02%), exceeding the numerical98% clean-answer threshold. Its direct answer head is the primary comparator; its auxiliary scalar head reaches3597/4096. The full two-LR search receives2048000 optimizer presentations and remains reported, including unselected checkpoints.

On the same examples, the frozen hybrid answers4081/4096 at delays0–8 and4076/4096 at16. Paired disagreements at0 are38 hybrid-only successes and13 N1-only successes (two shared errors); at16,38 versus18 (two shared errors). This is one development population with different inherited compute, not a confirmed intrinsic advantage. The comparison is answer-only against answer-only; the hybrid joint semantic gate is separate.

## Reversed-role localization from existing exports

N1's full ordered semantic proposal is correct on4096/4096 clean examples and4096/4096 reversed-role examples. Its fine-tuned lowerer's argmax binding does not drift in this control. Reversed answer/value accuracy conditional on correct lowering is therefore the unconditional3775/4096 and2245/4096.

| Primitive | Clean answer | Reversed answer | Reversed auxiliary value | Reversed changed-answer subset |
|---|---:|---:|---:|---:|
|add|755/764|746/764|503/764|0 examples|
|sub|828/838|823/838|573/838|422/431|
|mul|828/831|621/831|30/831|0 examples|
|neg|817/829|817/829|721/829|0 examples|
|compare|828/834|768/834|418/834|24/56|

The changed-answer aggregate is446/487. Of321 total reversed-answer errors,280 occur where the requested answer did not change;210 are commutative multiplication cases. The original generator always witnesses multiplication as(value,1); reversal gives(1,value), an unseen operand arrangement for neural numerical fitting even though the instruction schema is clear. This supports a downstream witness-position generalization failure, not a failure to select the actual ordered pointers. It does not uniquely identify an internal shortcut mechanism.

Soft retrieval is also nearly unchanged: reversed multiplication operand-value absolute residual mean0.000271,99th percentile0.004251,max0.108566, matching the clean distribution; minimum true-operation softmax feature is0.999919. These are descriptive internal feature scores, not calibrated probabilities. Rare soft-mixture errors remain possible despite correct argmax binding (largest observed subtraction residual1.54149). Auxiliary reversed multiplication predictions range0.5–4, despite the true scalar range−8–8; avoid reducing this result to an unverified exact copy-left mechanism.

Record/inventory order and fresh-name-no-op controls retain4056/4096; unrelated-instruction replacement yields4057/4096. No learning recipe was changed using these outcomes. Last1000 calibration-answer gains are−1/2048 at.001 and+3/2048 at.0003, below the preregistered0.5-point continuation trigger; no N1 extension is indicated by that rule. Higher auxiliary-value acquisition in another unselected checkpoint is not substituted for the direct-answer selection criterion.

Full process occupancy75.141586956s, exit0 under900s cap; internal73.882206661s, peak125318656CUDAbytes,1246248 trainable parameters. GPU released before analysis. [Raw predictions and provenance](../../../results/campaign-01/composition/c01-development/n1/raw/manifest.json), [primitive localization](../../../results/campaign-01/composition/c01-development/n1/localization.json) and [soft-retrieval audit](../../../results/campaign-01/composition/c01-development/n1/soft-retrieval-audit.json) retain evidence. Full tensor logits remain at `~/topoformer-campaign-01/composition/c01-development-n1/`. N2 remains pending; subsequent interventions must be newly registered rather than retroactively repairing this baseline.
