# P2a Part D: collapse-audit tool

Implements [protocol-P2a.md](protocol-P2a.md) Part D. Tool: `research/tools/campaign03_p2a_collapse_audit.py`; tests: `tests/test_campaign03_p2a_collapse_audit.py`. It is descriptive only and does no training.

## Subjects and probe

- **Subjects.** X1-rl-r0/r1/r2 tranche endpoints at attempts 18–29, plus each lineage's bootstrap bank.
  - Attempt 17 is loaded only as attempt 18's previous-tranche reference.
  - Checkpoints are hash-verified. Bootstraps are checked against `configs/campaign03/p1-banks.json`, attempts against each run's `state.json` allocations. Files are opened read-only.
- **Probe worlds.** 64 worlds using the P1 `SEALED` definitions: `iid_f0` on seeds 1,990,000,000–031 and `iid_f2` on seeds 1,990,100,000–031. The address namespace is `e03p2a-partd-probe`.
  - The two conditions are interleaved, so each 8-world batch holds 4 of each.
- **Sampled rollouts.** Sampling seeds are `torch.manual_seed(1,990,500,000 + 1000·sample + batch)`.
  - Rollouts use training's own `batched_on_policy` in batches of 8, the P1 `batch_size`.
- **Hyperparameters.** They come from `configs/campaign03/p1-rl-x1-r{k}.json`: lr 3e-5, entropy .003, KL .3, value .5, advantage normalization on, decision_mean.
  - Every audited RL checkpoint's saved config must match them.

## Measurements

| # | What | How |
|---|---|---|
| 1 | Greedy and sampled rollouts | Metrics: success, utility, action mix, idempotent-repeat, short-cycle and no-progress rates, steps-to-cap. Greedy is argmax, as in P1 evaluation. |
| 2 | KL(boot‖ck), KL(prev‖ck) | Training's masked KL (tested equal to `batched_on_policy`'s). Computed on bootstrap greedy states and on the checkpoint's own greedy states, plus the argmax-disagreement rate. |
| 3 | Critic | V vs undiscounted return-to-go on own sampled rollouts: MSE and bias (V−G). Mean advantage by action class (loop / productive / other), both raw and batch-standardized. Advantages are computed exactly as `actor_critic_objective` computes them, which is tested. |
| 4 | Gradients | `autograd.grad` of each weighted term (actor, 0.5·critic, −.003·entropy, .3·KL) on one fixed 8-episode sampled batch. Norms are split into shared parameters (observation + context encoder), the actor head (candidate + scorer) and the critic head (value). Pairwise cosines are taken on the shared parameters. No optimizer step is taken. The KL reference is the tranche start (attempt k−1); for the bootstrap it is the bootstrap itself. |

**Definitions.** Accepted action, public state, decision state, idempotent repeat, short cycle and productive action are defined in the module docstring. They are this tool's reading of protocol-P2a.md "Metrics", and the analysis-layer implementation should be cross-checked against them.
- **Short cycle.** An accepted action whose resulting decision state equals one of the 6 preceding states, including the state it was taken from, with no solver call and no new inspection in between.
- **Idempotent repeat.** The same action_key as the last accepted action, with the public state unchanged.

## Cost and launch

**Smoke (pro6000, CPU, 1 thread per lineage).**

| Phase | Measured cost |
|---|---|
| Greedy, with KL companions | ≈ 8.5–9.5 ms per step |
| Sampled | ≈ 6.8 ms per step |
| Gradient batch | ≈ 2.3–5.8 s |
| Bootstrap-state pass (14 models) | ≈ 60 s per lineage |

**Full run at the protocol's 4 samples.** ≈ 4.8k core-s, over the ≤ 3k Part D budget. Following the protocol, the number of sampled repeats is reduced first:
- `--samples 2 --grad-batches 1`: ≈ 2.9k core-s (estimate ± 10%).
- Fallback `--samples 1`: ≈ 2.1k core-s.

Each lineage writes `lineage-r{k}.partial.json` after every subject. If the run stops at the CPU cap, `--report DIR` rebuilds `partd.json` and `partd.md` from whatever was written.

Launch (root, from a source snapshot):

```bash
ssh -n pro6000 'wsl -d Ubuntu -- /home/brand/tensegra-campaign03/bin/detach.sh p2a-partd /home/brand/tensegra-campaign03/source-<sha> /home/brand/tensegra-campaign03/env/bin/python research/tools/campaign02_job.py --output /home/brand/tensegra-campaign03/results/p2a-partd-process --wall-cap 3600 --cpu-cap 3300 -- /home/brand/tensegra-campaign03/env/bin/python research/tools/campaign03_p2a_collapse_audit.py --output /home/brand/tensegra-campaign03/results/p2a-partd --samples 2 --grad-batches 1 --processes 3'
```

**Outputs** (in `results/p2a-partd/`):
- `lineage-r{0,1,2}.json`, the per-checkpoint records;
- `partd.json`, which holds the records plus the `locations` reading aids;
- `partd.md`, the tables plus a short summary;
- `manifest.json`, the source hashes and options;
- `lineage-r*.log`.
