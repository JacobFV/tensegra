# Pending returned-fact-use experiment (not released or frozen)

This is a proposal only. R01's declared contract failed, and R02 is the next authorized experiment. No consumer training, numerical-domain narrowing or composition promotion follows automatically from this draft.

## Smallest useful task

Reuse the existing retention generator's public query and task semantics:

`answer = (returned_value > public_threshold) XOR public_polarity`.

The exact runtime return is teacher supplied. The frozen return workspace and confirmed scalar accessor produce 33 learned class scores. A new learned comparator receives a softmax transformation of those scores plus the two public query fields, and predicts the binary answer. The softmax outputs are neural features, not assumed calibrated probabilities. A single 1024-wide hidden layer suffices for the first consumer; no new runtime primitive, recurrent memory, exact arithmetic bypass or proposal policy is introduced.

This would establish use of a learned scalar interface in a subsequent learned decision. It would not establish general free-workspace reintegration, recurrent multistep reasoning or autonomous symbolic scheduling.

## Matched controls

- Learned scalar accessor plus learned comparator.
- Query-only comparator with the scalar feature channel removed, matched hidden width.
- Privileged exact-value one-hot plus learned comparator as a ceiling for the downstream task. This cannot pass the learned-access contract.

Freeze the return backbone and accessor. Fit only the comparator, using new threshold/polarity queries sampled independently of returned values. Labels are privileged exact tensor comparisons. Every original return field and its provenance remains stored. Training includes the declared consumed horizons; longer lengths are separate tests rather than implied scope.

## Causal interventions

Keep query and unrelated context fixed. Evaluate:

1. Correct teacher return.
2. Drop the event before initial retrieval and prevent persistent rereads. There is no replacement semantic value; report performance against the original label as an ablation, not agreement with a fictitious zero fact.
3. Supply a different valid value of the same type. Preserve integer/half-unit/bool validity and regenerate operand witnesses so the teacher event remains auditable. Score agreement with both the original and the actually supplied value's answer.
4. Swap the whole return event and its corresponding identity-key inventory with a separate valid teacher event. Again score both original and actually supplied answers.

Report the answer-changing subset separately with its actual support; raw average accuracy can hide intervention insensitivity. Include public-query-only accuracy and label balance. Interventions are part of evaluation, not hidden target inputs. Incorrect access and incorrect downstream consumption must be counted separately.

## Prerequisites and prospective choices

A specific consumed scalar domain/horizon contract must first pass >=98% fresh validation in all three prescribed seeds. If narrower than the failed R01 contract, declare it prospectively and use fresh validation; preserve the failure. Numeric-range and per-value claims require their own support, since aggregate accuracy alone is insufficient.

Before launching this task, freeze its training/query distributions, fresh data namespaces, comparator exposure, selection rule, exact consumption threshold, required intervention effect and minimum answer-changing support. Use three paired confirmation seeds for a main claim. Do not reuse previous confirmation tests for model selection. No GPU has been requested for this draft.
