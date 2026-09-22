# Stage 4 independent core review

Scope: binding model, supervised losses, trajectory diagnostics and runner controls. Existing Stage 1–3 model/grounding/data/runner source remains unchanged. This is an implementation review; empirical conclusions require the completed experiment artifacts.

## Verified contracts

- Public forward inputs contain runtime identities, immutable memory values, current start and supplied relation instructions. Gold trajectories, answers and token-node correspondences enter only training losses and diagnostics. No teacher-forced intermediate identity writes were found.
- Cosine grounding uses independent query/key latent and entity projections. Identity-only reads remove content coordinates at the grounding interface. Protected attention writes use raw immutable key coordinates, bypassing direct learned value/MLP identity overwrites. Ordinary attention still reads content through its query and normalization: this is not total causal noninterference.
- Pointer writes use predicted real grounding pushed through the supplied relation and then the immutable entity keys. Null/lost mass is not renormalized. This is a stronger, explicit graph-transition prior and continues to use the graph at zero logit strength; it is not a no-graph control.
- Mixed raw-dot/full-state mode retains the original recurrent forward algebra. Tests cover bit-exact legacy outputs, permutation equivariance, separate role gradients, protected writes and zero-strength equivalence for non-pointer variants.
- Grounding CE uses labeled pre/post query states and real-memory identities. Null BCE balances real/distractor groups. Cycle KL uses a detached **predicted forward pushforward** target; it does not presume inverse relations or a correct gold transition. The null/cycle-only beta-zero variant is distinct from answer-only training.
- Persistence/recovery use initial grounding followed by each post-update state, avoiding duplicate overlapping pre/post transitions. Integer conditional counts pool before division, and absent denominators remain null. Complete pre/post paths and canonical complete trajectories are separately identified. First error is zero-based, with D+1 indicating no observed error; survival is cumulative correctness.
- Main runner evaluates initialization at shallow validation and the joint largest/deepest corner, plus the complete final size/depth matrix. Initialization/final corner examples use the same seeds. Matched models reset the same initialization seed and receive the same deterministic training schedule, recorded with hashes. Final evaluation is not a best-checkpoint selector.

## Findings resolved or communicated

1. The draft runner inherited cosine/protected-write defaults for Stage 3 labels. Explicit dot/mixed/full-state defaults now preserve the legacy control, with a runner-level bit-exact regression test. A temporary diagnostics import mismatch was also corrected before experiments.
2. Proposal-minus-write was labeled identity interference although it is mechanically zero for mixed updates. Requested an override-magnitude name and separate MLP identity-delta instrumentation. These are descriptive interventions/drift diagnostics, not proof of content interference.
3. Relation-attention argmax is valid for the functional-relation generator but does not represent general multi-edge or missing-edge relations. Requested support-based instrumentation or explicit scope.

## Interpretation boundaries

Correlated failures, persistence and a discrepancy from products of marginal correctness are descriptive evidence. They do not establish dynamical attractors: revisited nodes, shared graph difficulty and trajectory selection also correlate errors. Learned cosine/null behavior must be compared with initialization; strong initialized pointer routing is an architectural prior rather than discovered computation.

The preregistered gate requires both task and complete-path accuracy >=95% at D64/N128 in **each** of three seeds. Adaptive strength remains gated; no interpreter is authorized by passing this implementation review alone.
