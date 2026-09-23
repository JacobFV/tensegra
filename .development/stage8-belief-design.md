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
cell, >=512 examples/cell, final support accuracy >98% IID / >95% moderate OOD, mean framewise impossible
mass <1%, and mean framewise posterior L1 <0.05, matching the root gate registry. No empty/missing matrix passes. Test results do
not select thresholds. Readiness, composition, and supervision withdrawal remain
blocked until independently audited gates authorize them.

Only timing configuration is currently supplied; it is **not a scientific main
run**. Primary training budget awaits CUDA measurement and root approval. Tests
use explicitly reduced dimensions solely for contract checks. No local Torch
training is used. Initial four remote unit tests passed; full review pending.

## Hardware timing (not acquisition)

Six contract tests pass remotely. CUDA 2.14/cu130 on GB10 completed two optimizer
updates for each arm sequentially at width1024 / inner2048 / batch4. Both arms
have 16,851,011 total parameters. Peak allocated CUDA memory was 404,143,616 bytes
(protected) and 404,162,048 bytes (recurrent). The second warm update plus eight
probe examples took about 0.028 and 0.040 seconds, respectively; two updates plus
initial/intermediate probes took about 0.73 seconds. These tiny timings are only
hardware feasibility checks, not performance estimates for main batches/evaluation.

Timing ran in isolated `/tmp/stage8-belief-tests`; its manifest git hash denotes
that temporary snapshot, while per-file source SHA256 identifies actual code.
Raw timing/config/metric JSON is committed under results/stage8/belief-timing.
No acquisition conclusion or gate decision follows from two updates.

## Acquisition probe and main freeze

Seed0, 256 updates ×32 fresh episodes (8,192 presentations), validation128 only:
protected clean/contradiction/retraction final accuracy was100% at N8/N16. Clean
mean posterior L1 was .029/.0154. However empty-evidence posterior L1 was
.1357/.0703: support membership is perfect but null calibration remains wrong.
The recurrent comparator clean accuracy was89.1%/88.3%, retraction14.1%/19.5%.
No gate passes from this acquisition probe. All exact oracle controls match.

Training+probe elapsed9.73s protected /13.29s recurrent, peaks564MB/802MB.
These are measured GPU runs, not estimates. Acquisition raw JSON and a trajectory
plot are committed. No test split was inspected.

Main configuration frozen after acquisition: same source/architecture and lr,
1000 updates ×32 fresh episodes, paired seeds0/1/2, 512 validation and512 test per
N8/N16 condition, all7 evidence controls. No further acquisition-driven tuning.
All expected validation cells (42 per arm) must pass the root registry; every-frame
calibration retained. Test cells cannot select settings. Runtime composition and
readiness remain prohibited while any required belief gate fails.
