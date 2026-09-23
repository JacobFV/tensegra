# Stage 8 isolated belief updates

Default workspace width is **1024**, with four distinct full-width residual MLP
phases and 2048-wide feed-forward interiors. Phases are reused over evidence
frames; there is no KV history and no attention in this deliberately minimal
comparator. This is not represented as a replication of the Stage 7 transformer.

Each episode supplies shuffled joint proposal records: primitive, destination,
ordered operand zero, ordered operand one. Entity identity is an episode-specific
random nonce vector. Multiple proposals share role values, so evidence arrives
while alternatives remain ambiguous. Public observations provide a typed role,
value feature, action, and stable observation ID. Roles are supplied, not inferred
from text. Current tiny acquisition family contains add/sub operation alternatives;
this tests ordered joint belief dynamics, not primitive-library expansion.

Both learned arms receive the same candidate/observation feature pairs and full
per-frame posterior supervision plus per-observation compatibility supervision.
The candidate role selector is architectural. Compatibility is *not* calculated
or supplied to the learned forward pass. Four shared-parameter candidate phases
learn nonce matching and operation-cue compatibility. Models have identical
parameter inventories; the protected arm's recurrent readout is unused, and the
recurrent arm incurs extra phase calls for its auxiliary evidence head. Report
actual parameter count and compute rather than implying equal FLOPs.

- Recurrent: unrestricted candidate states carry evidence across frames.
- Protected: learned log-compatibility increments are stored in an explicit
  observation ledger and summed into candidate logits. A learned null head reads
  candidate score statistics. Duplicate IDs use first-write-wins semantics;
  retraction removes a contribution. These guarantees are supplied by architecture.
- Oracle: exact public feature equality forms candidate support; a uniform
  posterior covers consistent candidates, or the explicit null candidate when
  contradictory evidence eliminates them all. No learned proposal is executed.

Controls are clean, independently reordered evidence, redundant duplicate IDs,
contradiction, explicit retraction, partial information, and empty evidence.
Distinct episode-local RNG streams make control variants matched. Validation and
test seed streams are separate; fresh training episodes replace the Stage 7 finite
trajectory bank. Candidate-count OOD changes real actor tensors, not metadata.

Metrics include each frame's posterior, support accuracy, impossible mass,
entropy, L1 posterior error, and reliability bins. Reliability compares confidence
with *expected correctness under the gold joint posterior*, not mere membership
in the plausible set. Membership can be perfect under unresolved ambiguity and
must not be confused with certain identification. Raw sample trajectories remain
available separately.

Provisional validation competence requires every declared seed/condition/count
cell, >=512 examples/cell, final support accuracy >98%, impossible mass <2%, and
posterior L1 <0.1 at every frame. No empty/missing matrix passes. Test results do
not select thresholds. Readiness, composition, and supervision withdrawal remain
blocked until independently audited gates authorize them.

Only timing configuration is currently supplied; it is **not a scientific main
run**. Primary training budget awaits CUDA measurement and root approval. Tests
use explicitly reduced dimensions solely for contract checks. No local Torch
training is used. Initial four remote unit tests passed; full review pending.
