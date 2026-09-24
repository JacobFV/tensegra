# Batched on-policy prospective static review

Snapshot `58693400`, source-only review; root owns mechanical execution and profiling.

No hidden-state actor input found. Current public observations/candidate features feed scores; evaluator utility is used afterward as permitted privileged reward. No teacher actions enter this rollout. Reward increments telescope to the final verified utility, avoiding repeated-total reward counting. Actor advantages are detached, while the value term learns undiscounted return-to-go; entropy is applied explicitly.

Active episode indices select their own hidden rows, and functional index-copy restores them without an in-place autograd overwrite. Hidden state is detached only at declared BPTT intervals. Shared model parameters naturally couple gradient updates, but runtime states do not mix between episodes. The policy uses no cross-example batch normalization. Retired rows remain unselected. Actual gradient/terminal-row behavior still needs the root's mechanical tests; this review is not a runtime proof.

Sampling consumes RNG by time and active batch rather than serial episode. This is explicitly versioned and cannot be called bitwise-equivalent serial sampling. The population protocol must freeze rollout_mode, schedule and checkpoint RNG. Switching source/method requires a separately registered run; inherited optimizer/RNG handling remains the population runner's responsibility.

Two objective qualifications:

- Current implementation averages losses over all realized decisions. With variable episode lengths, this is a decision-normalized estimator, not literally the mean of per-episode summed policy gradients for expected episode utility. If exact episode-weighting is desired, sum actor terms within each episode then average episodes; normalize value/entropy losses under an explicit separate rule. Any change is an objective intervention and must precede the main protocol. Otherwise disclose the existing normalization.
- Reaching max_steps without environment termination gives zero future value. That is a legitimate truncated finite-horizon task if the cap is part of the contract; it is not a bootstrapped estimate of continuing value. Keep truncation counts and ensure compared policies share the horizon.

The batched timing allocation is throughput cost, not per-episode serial latency. Parent CPU excludes live solver children; charge the inclusive outer ledger once. Architecture tariffs and fixed public prices remain separate from measured experiment expenditure.

No new experiment, GPU workload or proposed architecture change in this review.
