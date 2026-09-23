# R05 development: use a frozen returned scalar in a subsequent learned decision

This protocol is conditional on R04 independent audit. It is an oracle-first-operation interface study: the exact return is supplied, the unchanged Stage 11 wide workspace and R04 16k/900 scalar consumer access it, and only a downstream neural comparator learns. It does not learn the first operation, schedule symbolic execution, or establish two-operation autonomy.

## Task and observable inputs

Keep the actual registered generator mixture and bounded 33 half-unit values. Reuse the public query and exact target already in `retention_data`: `(returned_value > threshold) XOR polarity`. The scalar accessor's 33 scores are transformed by softmax and concatenated with the two public query fields (threshold divided by eight, polarity unchanged). Softmax is a feature transformation, not a claim of calibrated uncertainty. The actor receives no exact scalar or answer except in the explicitly privileged ceiling.

Compare three paired comparators: learned-access scores plus query; query only (zeroed 33-value channel); and exact-value one-hot plus query. Each is a 35→1024→2 GELU MLP with identical initialization and optimizer exposure. Freeze all backbone, return memory and scalar-accessor weights. No new primitive, recurrent cell, encoder or exact-copy path is introduced.

Start development on historical backbone 11, already used in failure-directed development; do not call it a held-out initialization. Fresh populations: 8,192 fitting events, 1,024 calibration events, 2,048 development-validation events. Capture covered delays 0/1/2/4/8/16; train with two distractors, evaluate with two/eight. Labels and query distribution remain unchanged. Delay 32 is excluded from development and any endpoint selection. Development data namespaces start at 31 million, disjoint from all readout fitting and confirmation data.

## Frozen development exposure and selection

AdamW .001, batch 256, 2,000 updates per comparator; all arms use the same sampled event/delay indices. Save/evaluate steps 0/250/500/1000/1500/2000. Select an endpoint per arm by minimum correct count over covered calibration cells, then total count, then earlier endpoint. Save all candidate outcomes and report the selection search. Do not select on development validation, interventions or balanced-grid outcomes. Main confirmation requires a subsequent recipe freeze and three fresh population seeds.

Development acquisition target: learned-access comparator exceeds 95% on every clean validation cell; the exact-value comparator must do so too, otherwise localize comparator acquisition before interpreting return errors. Query-only behavior measures the no-return baseline rather than being expected to acquire the task. No claim follows from a single development backbone passing.

## Causal interventions, after comparator selection

Keep each original public query and distractor realization fixed. Evaluate matched examples at delays 0/1/16, eight distractors:

- Correct return.
- Dropped return: omit initial retrieval and all persistent memory reads. No supplied scalar exists; report only original-answer accuracy and query-only comparison.
- Wrong value: reflect the value within the same legal type (Boolean complement, integer/float sign reflection; zero maps to eight), and rebuild the primitive's operand witnesses. Identity, type, operation and provenance remain unchanged. Verify exact primitive execution gives the altered value.
- Swapped return: use a separately generated valid event and its argument/provenance key inventory, retaining the original query and unrelated context. No hidden target is passed to the actor.

For wrong/swap, retain both original and supplied-fact labels. Report all-example and answer-changing-subset accuracies, with actual subset support. Intended-use evidence requires at least a 20-point drop in original-answer accuracy when the event is absent, and at least 90% agreement with the supplied fact on answer-changing wrong/swap subsets (minimum 256 such examples per cell). These are prospective development checks, not implied by a clean accuracy pass. Failures select a diagnostic branch instead of changing thresholds.

The exact-value comparator receives the actually supplied value under wrong/swap, as a privileged ceiling. Under drop its value channel is zeroed and explicitly out of its training distribution. The query-only comparator always receives the same query and zero value channel. Learned-access features always come from the manipulated workspace, never a direct scalar bypass.

## Stratified-use diagnostic and claims

Evaluate a separate fresh full typed-value grid with 64 events per legal type/value at delays 0/1/16. Its query draws remain independent of value. Report each type/value and the float-zero ingestion stratum prominently. Do not replace the registered mixture, retrain on the grid, or infer uniform-value competence from aggregate use accuracy. R04's whole-class float-zero error remains part of the evidence even if the binary decision is sometimes insensitive to it.

Persist raw scalar scores, consumer logits, original/supplied targets, event hashes, all selected endpoints and causal subset masks. Audit actor inputs, exact witness validity, frozen-weight hashes and paired outcomes independently. Profile before launch; initial estimate 2–4 minutes capture plus fitting, requested development cap 360 seconds, subject to coordinator approval. No GPU work is authorized by this document alone.
