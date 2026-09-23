# Isolated semantic return retention implementation plan

**Goal:** independently measure retention and learned use of perfect return events.
**Architecture:** existing four-phase recurrent cell; one-shot versus persistent event access; late query and workspace-only output.
**Spec:** [stage7a-design.md](stage7a-design.md).
**Execution:** user-authorized subagents, isolated worktree, remote bounded CPU tests/training, independent review. Scope stops before candidate construction and halting.

- [ ] Data: test deterministic generation, typed exact events, ordered operands, nonce/key permutation, public/target separation and late-query counterfactuals; implement `retention_data.py`.
- [ ] Model: test shared parameter count, immutable event state, no target access, fixed delay plus query step, gradient flow and memory interventions; implement `retention.py` using existing blocks.
- [ ] Study: paired initialization/batches, fixed-set acquisition gate, step0/curves, fixed-source main protocol and heldout delay/distractor evaluations; implement `retention_study.py` and configs.
- [ ] Review and run: verify information equivalence, protect/free distinction, target leakage and late query; full remote regression, bounded acquisition and main experiments as justified.
- [ ] Analyze: raw metrics/provenance, field-specific failure analysis, conditional and counterfactual results, concise report; preserve earlier artifacts, commit/push and integrate.

Review focus: final query revealed early; output bypasses workspace; persistent memory silently counted as learned retention; type or provenance position shortcut; changing facts without changing expected answers; overfit scores mistaken for generalization; comparisons change parameter count or training data.
