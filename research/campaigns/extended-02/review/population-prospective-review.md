# Population runner prospective static audit

Reviewed population `5d40f03d`, training CLI `174c77f5`, policy/reference sources and population mechanical tests. No GPU/solver/training workload launched by this review. This checks design/code paths; it does not establish execution or population advantage.

## Allocation and ancestry

Each round has six training slots. PBT and multistart train one member per slot; single routes all six slots to member0. Final selection is among current final-round members, not an unregistered best historical checkpoint; single uses its latest state only. All modes evaluate after every slot, so development invocation counts match. Training update and episode counts match at configuration level, but decision counts, solver work, wall time, initialization/checkpoint overhead and neural work can differ. Do not call this equal actual compute without measured ledger evidence or a separate compute-normalized analysis.

PBT copies donor model and optionally optimizer state; reset clears moments while retaining the recipient parameter-group structure and explicitly replacing LR. Recipient RNG, data cursor and resource counters survive inheritance, preventing sibling reuse of the donor's next training stream. The replacement event preserves donor/recipient checkpoint hashes and explicit mutations. Cumulative slot counters are **not full ancestral acquisition exposure**: a descendant inherits a donor's learning. Audit ancestry through checkpoint parent edges and do not add shared ancestors repeatedly when reporting process-wide work.

Two donors replace two strictly worse recipients; ties do not replace. Parents and recipients are disjoint under this rule. Architecture compatibility is checked. All initial hyperparameters currently come from the common TrainConfig; mutations introduce hyperparameter adaptation only in PBT. The comparison therefore tests the joint inheritance/mutation procedure versus initialization multistart, not an isolated inheritance effect. At least three independent initialization banks and mutation seeds are required; three descendants of one bank are not replications.

## Resume and data boundaries

Source/protocol hashes are checked on resume. Interrupted slot attempts become explicitly interrupted and receive new immutable output filenames; outer process accounting must retain their cost. Completed cursors prevent replay of already committed slots. Replacement files written before metadata commit are checked against the deterministic replayed event. CPU parking avoids persistent GPU residency of all six members.

Development seed intervals are separated from training intervals, and no confirmation-data reader exists in this runner. The **same development episodes are reused across slots and rounds**. That is legitimate selection data but not fresh-outcome evidence; adaptive selection can overfit it. Freeze a separately constructed confirmation population, including semantic/composition separation, outside this runner. Seed inequality alone does not prove semantic nonoverlap.

Changing source or protocol should fork a new registered run. Model/RNG restoration is explicit; data hash chaining deliberately rehashes the prior digest rather than pretending to reconstruct the original hash object. Exact equivalence to uninterrupted optimization still requires a mechanical interrupted/resumed-run test in the actual environment; the existing tests exercise selection and tensor inheritance, not the whole crash lifecycle. No such equivalence is claimed by this static audit.

## Learning objective and public information

The lightweight controller scores each candidate against a shared observation summary; it does not pool other candidate facts. Recurrent controller additionally reads candidate memory. Both consume public hand-engineered features, while reference teacher receives full public JSON. This is a legitimate supplied bootstrap but is not an equal-information architecture comparison with that teacher. If the teacher's decision relies on facts omitted from features, more optimization cannot recover them.

Exact teacher action indices contain arbitrary distinctions. Multiple uninspected same-category items have identical feature vectors; teacher selects first inventory entry. Equal-weight/price greedy items are tied by opaque handle spelling, excluded from neural features. Such labels may leave nonzero cross-entropy even when any tied choice is task-equivalent. Use verified episode success as acquisition criterion. If equivalent-action supervision is introduced, version it and construct the equivalence set from public semantics without silently supplying hidden successful actions.

Supervised training success currently records the **teacher's** rollout outcomes, not learner closed-loop success. Curves must label this explicitly; only evaluate/live episodes measure acquired behavior. Actor-critic rewards include private verified outcome as allowed training feedback, not inference features. Entropy mutation has no effect during pure cross-entropy bootstrap, so a supervised-only PBT result cannot credit exploration adaptation.

## Promotion requirements

Before main PBT: demonstrate nontrivial learned closed-loop acquisition, profile actual costs, freeze neural tariff and public world mix, register independent banks and sealed cells. Show fixed same-observation reference performance and computational leverage. Record every failed offspring/evaluation and all lineage costs. Preserve conventional, multistart, and simple heuristic alternatives even if they dominate the recurrent architecture.
