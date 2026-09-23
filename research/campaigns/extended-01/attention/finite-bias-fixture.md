# Finite-bias mathematical fixture

This calculation explains a possible mechanism; it is not a fitted account of the trained model.

Assume one outgoing edge per row, all content logits equal, and structural strength λ. The correct-edge mass is

`p = exp(λ) / (exp(λ) + N - 1)`.

For a permutation adjacency P, the entire attention operator is

`W = α P + (1 - α) J`,

where `α = (exp(λ) - 1) / (exp(λ) + N - 1)` and J is the uniform averaging matrix. Since every permutation commutes with J and leaves it unchanged, a sequence of D such operators gives

`W_1 ... W_D = α^D (P_1 ... P_D) + (1 - α^D) J`.

The distinct destination signal shrinks even though the attention argmax selects the correct edge at every step. This is an exact matrix identity in the stated fixture, not multiplication of marginal empirical correctness probabilities. It assumes no learned nonlinear update between routing steps.

At N32 and λ=4, α=0.6262 and α^8=0.0236. This α is not the measured edge mass in the trained model. The mechanical test checks the single-row probability formula directly. The learned model has nonuniform content scores and a nonlinear shared update, so its measured edge mass and task accuracy must remain separate empirical quantities.

A02 intervenes on λ while retaining a frozen learned model, then trains a separately declared strength comparison. If stronger λ restores values without changing argmax routes, it provides causal evidence about the score interface in this model. It would not establish that generic recurrent networks necessarily lose information, or that soft attention is universally inferior.
