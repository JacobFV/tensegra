# Extended-02 closeout observations

Extended-02 closed on 2026-09-25 after the Mother Agent's review of the final report at `d7058255`. The remaining CPU allowance (~3.5 core-h) was intentionally left unspent. This note records the closing observations that shape how the campaign's results should be read and what comes next. The wording corrections below were applied to the final report, synthesis and claim maps.

## 1. Observability is the main lesson

The encoded-input counterfactual audit ([diagnostics/encoded-counterfactuals.md](diagnostics/encoded-counterfactuals.md)) showed three things that were missing from the policy's effective input:
- The original candidate encoding is identical for a current and a foreign same-type return.
- It is also identical for a current and a stale return.
- A rejected commit followed by any other action loses its rejection history.

When φ(x) = φ(x′), a decision rule over φ cannot use the distinction between x and x′. More optimization, more selection or a larger downstream network cannot recover it from those features alone. An interactive agent could still gather new observations, so this concerns the tested decision interface, not every conceivable strategy.

**Standard phrasing:** "the original representation does not support discriminating these return candidates by provenance; the observed collapse is therefore not evidence that provenance binding is unlearnable."

**Standing rule for future campaigns (preflight):** before training a policy to make a distinction, construct paired cases and verify that the actual encoded observation and full candidate set preserve it.

## 2. Provenance: repaired in-distribution, not solved

- **The repair.** The m2 features plus same-type training exposure solved all tested same-type distractor conditions in 3/3 lineages. The features and the data changed together; no matched m1 + same-type-training arm isolates the two.
- **The shortcut is relational, not lexical.** m2 tests whether the record's problem is in the current problem registry, and whether the saved snapshot equals the registered problem. It does not read the name `prior_i`. **Renaming foreign records therefore will not remove the shortcut:** an unregistered `problem_938` is still outside the registry. The next test must break the correlation at the level of relationships.
- **The target is applicability, not "mine / new / not prior".**

| Return | What must be distinguished |
|---|---|
| Current result for the requested computation | Applicable |
| Result for another registered problem of the same type | Not necessarily applicable |
| Result from an earlier version of the same problem | Potentially stale |
| Older result still valid under current requirements | Potentially reusable |

- **What the repaired controllers do is defensive, not reuse.** 214/2,094 same-type distractors happened to be valid answers. The repaired controllers never retrieved or used a foreign record. That is effective defence in this benchmark, not intelligent cache reuse. "Always recompute" may be much worse than validating and reusing an applicable result, and the current success metric does not measure that tradeoff.

## 3. Rejection memory: repaired, not general recovery

- **What the counters did.** Supplied per-stage rejection counters removed the audited rejected-commit loops (126 → 0; total failures 126 → 6 over 22,272 episodes per arm, mostly from the previously weak lineage).
- **What remains.** Five of the six remaining failures are a different loop (inspect, then repeated `choose_item`, with no completion attempt).
- **Correct claim:** supplied rejection history let the learned policy avoid the observed rejected-commit loop. It does **not** show that the agent detects and escapes arbitrary loops.
- **Reason visibility.** The dedicated feedback encoding does not separate capacity from funds rejections. Some reason-relevant information appears in other candidate features, so "reason-blind" should not be read as "no route to infer the reason".
- **Proposed next abstraction, not yet demonstrated:** an explicit **attempted-action record**. Each entry holds the action + arguments, the relevant input/state version, the outcome, the rejection reason, and whether those dependencies have since changed. That supports the question "is this a retry of the same failed computation, or have conditions changed enough to justify retrying?" It avoids ad-hoc counters per loop and does not forbid legitimate retries after new observations, revised drafts or a larger budget.

## 4. Selection: three claims, kept separate

| Claim | Evidence here |
|---|---|
| Finding a useful configuration | Some positive evidence (halving found the tuned row in 5/6 runs) |
| Allocating training efficiently | Some positive evidence (halving > PBT and multistart, point estimates) |
| Discovering a more robust computational strategy | Unresolved (no population design reliably produced it) |

- **PBT** underperformed matched multistart on average.
- **Successive halving** has no mutation or recombination; its advantage comes from which candidates receive more sequential training.
- **Selection optimizes the signal and horizon it receives.** Populations discarded a temporarily weaker lineage before it acquired a better fallback. Adding tool-removal worlds rewarded indiscriminate tool avoidance. No complete causal explanation or universal repair was established.
- **Difficulty is not the target.** Extreme difficulty does not guarantee good computational habits: it can reward a brittle shortcut, erase stepping stones or flatten fitness differences. The target remains **conditional resource use**, not maximal tool use, maximal difficulty or diversity for its own sake.

## 5. Orchestration within a supplied stage order, not functional composition

The controllers learn action sequences, inspection choices, solver use and budget escalation, and reuse them under unseen stage orders. The high-level stage order is public and enforced by the environment.

**Correct wording:** "learned action sequencing within a supplied high-level stage order."

Functional composition, y = f(x), z = g(y), a = h(z), where one result changes the next computation's inputs, constraints or meaning, **was not tested.**

## 6. Architecture

- The lightweight controller is the engineering reference. The recurrent workspace was larger and did worse under RL, but this was **one seed per family**, not a universal ranking.
- **Rule:** the workspace must earn its cost on an identified requirement: structured-history integration, ambiguous argument binding, or genuinely data-dependent composition. It must not win by crippling the baseline.
- Several successes depend on hand-engineered relational, provenance and failure-history features. That is legitimate supplied structure, **not learned latent structure discovery.**
- There is no positive result for programmable attention, spontaneous crystallization or broad semantic graph induction.

## 7. Decision

Extended-02 is closed. Spending the remaining allowance is not needed. The next campaign is defined in [../extended-03-proposal.md](../extended-03-proposal.md).
