# Stage 6 protected incremental runtime

`thinking_runtime.py` is independent of tensor code and keeps Stage 5 unchanged. It deliberately accepts only a sequence of observable `ValueRegister`s, never a task object, labels, future transitions, hidden graph, or gold actions. Python objects are an interface boundary rather than a security sandbox against hostile Python reflection.

## Frozen interface

- `ValueRegister(id, value, type, provenance=())` is frozen, with integer/float/boolean types and immutable scalar payloads.
- `Candidate(id, primitive, arguments, readiness=1.0)` has ordered register-ID arguments. Routing remains with the model.
- `ProtectedSession(initial_values, max_abs_value=1e12)` owns memory. `registers` is a read-only mapping and `events` a tuple of committed returns.
- `execute(proposals, threshold=.5)` returns one `ReturnEvent` per proposal. Event fields are `candidate_id, primitive, arguments, value, type, register_id, provenance, status, reason, affected`.
- Status is executed, duplicate, rejected, deferred, or conflict. Only executed appends memory/history; duplicate returns the original typed payload but an empty affected set.
- `PRIMITIVE_NAMES = ('add','sub','mul','neg','compare')`; compare means ordered less-than, returning boolean. Integer arithmetic is exact within the configured magnitude bound. Float arithmetic follows Python finite floating-point semantics; it is not exact real arithmetic. Booleans cannot be arithmetic operands.

PrimitiveSchema exposes P/name, Sigma/input-output signatures, named V constraints and T transition semantics, plus external L/U adapters. L/U callbacks never receive session state. These hooks do not authorize direct learned writes. The fixed transition implementation appends results only after schema, availability, type and numeric-bound checks.

## Execution and audit semantics

All proposals in one execute call read the same current-state snapshot. Independent proposals and overlapping read-only arguments can execute together; a result produced in this batch becomes an operand only on a later call. There is no target-based correctness filter or winner chosen by hidden labels. Threshold decisions are candidate-local. Incompatible actions claiming the same candidate identity conflict symmetrically, including on retries; identity conflicts are checked before readiness so an ambiguous identity cannot be reused by adjusting its score. Identical retries are idempotent. Deterministic result IDs are `result:<candidate.id>` and an existing literal can never be overwritten.

Each invalid primitive refuses atomically, while unrelated valid primitives may still commit. Malformed Python API containers raise before any batch writes. Rejected/deferred/conflicting outcomes are returned for caller-side audit but do not alter protected memory or committed event history. Constructor initialization is the sole literal-loading boundary.

The operation is an append-only register transition, not a general graph rewrite engine. `affected` exactly lists newly created result register IDs; inputs are unchanged. Ordered argument identities and the transitive ordered unique source provenance are retained, with the candidate identity appended. Scalar arithmetic can be replayed exactly for integers, and append events can reconstruct state from original literals. There is no public undo/inverse/retraction API, no scope/binding mutation, and no claim of algebraic reversibility of arithmetic or arbitrary graph rewrites.

## Verification

Twelve focused stdlib tests pass locally using an import-only package stub to bypass the package's unrelated Torch import (Torch is unavailable locally). Tests cover order, types, persistent returns, candidate-local readiness, snapshot availability, multiple independent/shared-read operations, duplicate/conflict stability, rejected-write atomicity, malformed-batch atomicity, immutable interfaces, bounded/nonfinite values, schema adapters, and output namespace protection. Full Torch-enabled regression remains the coordinator's remote verification responsibility.
