"""Deployment procedures D for frozen depworld actors (extended-04 design §1, §2 A1, v2 rev. 4-6).

A deployment procedure maps (logits, public history) to an action. It never changes
the policy, the recorded facts or the costs. Modes:

- ``greedy``          argmax of the logits (historical default).
- ``sampled``         one fixed-seed sample per decision from the float64 softmax
                      (inverse CDF, one uniform per decision from the per-world stream
                      ``sampling_rng_seed(world seed, sampling_seed)``); T = 1.
- ``r_mask``          greedy, but candidates whose canonical action key the progress
                      diagnostic flagged at the *current* relevant signature are masked
                      and the argmax is taken over the rest. A mask applies only while the
                      signature equals the one it was flagged at, so any relevant change
                      releases it. ``verify``/``abstain`` are never masked; if every
                      candidate is masked the unmasked argmax is used (``fallback``).
- ``r_sample``        greedy, but at a flagged state (the diagnostic holds a mask at the
                      current signature) one action is sampled from the full policy π
                      (one uniform from the per-world stream per such decision).
- ``masked_sampled``  sample from π renormalized over the unmasked candidates,
                      π_M(a) ∝ π(a)·[a not masked]; the log-probability under π_M is
                      recorded (A2-dep on-policy training uses ``masked_log_probs``).

Temperature: argmax(ℓ/T) = argmax(ℓ) for every T > 0, so no temperature is involved
in greedy, r_mask or the greedy steps of r_sample.

Greedy and sampled choices, probabilities, compute charges and trace rows are the
same as ``campaign02_training.batched_episodes`` (tested bit-identical); the
diagnostic only observes. ``diagnostic_compute_units`` (default 0) is charged per
decision in the modes that consult the diagnostic (r_mask, r_sample,
masked_sampled); the diagnostic's own CPU time is always measured and recorded.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time

from .campaign04_progress import ProgressTracker

MODES = ("greedy", "sampled", "r_mask", "r_sample", "masked_sampled")
SAMPLER_MODES = ("sampled", "r_sample", "masked_sampled")
DIAGNOSTIC_MODES = ("r_mask", "r_sample", "masked_sampled")
DEPLOY_VERSION = "deploy-v1"


def masked_log_probs(logits, masked):
    """log π_M for training (torch): logits [B, A], masked bool [B, A] (True = masked).
    Rows whose real candidates are all masked must be unmasked by the caller."""
    return logits.masked_fill(masked, float("-inf")).log_softmax(-1)


def _renormalized(probabilities, masked):
    kept = [0.0 if m else p for p, m in zip(probabilities, masked)]
    total = sum(kept)
    if total <= 0:
        return None
    return [p / total for p in kept]


def deploy_episodes(model, environments, *, mode="greedy", device="cpu", max_steps=64, neural_work_per_forward=1.0,
                    samplers=None, trackers=None, diagnostic_compute_units=0.0, tracker_options=None):
    """Batched deployment of a frozen actor. Returns batched_episodes-style rows, each with
    ``progress`` (diagnostic summary incl. interventions) and ``deployment`` fields."""
    import torch
    from .campaign02_training import inverse_cdf_choice, policy_batch, prepare_active, score_batch
    if mode not in MODES:
        raise ValueError(f"unknown deployment mode {mode!r}")
    if mode in SAMPLER_MODES and (samplers is None or len(samplers) != len(environments)):
        raise ValueError(f"{mode} needs one sampler per environment")
    observations = [env.observe() for env in environments]
    diag_cpu = [0.0] * len(environments)
    if trackers is None:
        trackers = []
        for i, o in enumerate(observations):
            start = time.process_time()
            trackers.append(ProgressTracker(o, **(tracker_options or {})))
            diag_cpu[i] += time.process_time() - start
    traces = [[] for _ in environments]
    unsupported = {}
    hidden = None
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for step in range(max_steps):
        active, public = prepare_active(model, observations, unsupported)
        if not active:
            break
        neural_start = time.perf_counter()
        batch = policy_batch(model, [frame for _, frame in public], device)
        current_hidden = None if hidden is None else hidden[active]
        logits, _, next_hidden = score_batch(model, batch, current_hidden)
        extra = [dict() for _ in active]
        if mode == "greedy":
            selected = logits.argmax(-1)
            chosen = selected.cpu().tolist()
            probabilities = logits.softmax(-1).gather(1, selected[:, None]).squeeze(1).cpu().tolist()
        elif mode == "sampled":
            distribution = logits.double().softmax(-1).cpu().tolist()
            chosen = [inverse_cdf_choice(distribution[row], samplers[index].random()) for row, index in enumerate(active)]
            probabilities = [distribution[row][c] for row, c in enumerate(chosen)]
        else:
            greedy = logits.argmax(-1).cpu().tolist()
            softmax = logits.softmax(-1)
            distribution = logits.double().softmax(-1).cpu().tolist() if mode != "r_mask" else None
            chosen = []
            for row, index in enumerate(active):
                actions = public[row][0]
                start = time.process_time()
                masked = trackers[index].mask(actions)
                flagged_state = trackers[index].is_flagged_state()
                diag_cpu[index] += time.process_time() - start
                n_masked = sum(masked)
                choice = greedy[row]
                if mode == "r_mask":
                    if n_masked and n_masked < len(actions):
                        row_logits = logits[row, :len(actions)].clone()
                        row_logits[[i for i, m in enumerate(masked) if m]] = float("-inf")
                        choice = int(row_logits.argmax().item())
                        trackers[index].note_intervention(step, "mask", n_masked, choice != greedy[row])
                    elif n_masked:
                        trackers[index].note_intervention(step, "fallback", n_masked, False)
                elif mode == "r_sample":
                    if flagged_state:
                        choice = inverse_cdf_choice(distribution[row], samplers[index].random())
                        trackers[index].note_intervention(step, "sample", n_masked, choice != greedy[row])
                else:  # masked_sampled
                    probs = distribution[row][:len(actions)]
                    renorm = _renormalized(probs, masked) if n_masked else probs
                    if renorm is None:  # everything masked: unmasked distribution
                        renorm = probs
                        trackers[index].note_intervention(step, "fallback", n_masked, False)
                    choice = inverse_cdf_choice(renorm, samplers[index].random())
                    extra[row]["masked_log_probability"] = math.log(renorm[choice])
                    if n_masked and renorm is not probs:
                        trackers[index].note_intervention(step, "msample", n_masked, False)
                chosen.append(choice)
            index_tensor = torch.tensor(chosen, dtype=torch.long, device=logits.device)
            probabilities = softmax.gather(1, index_tensor[:, None]).squeeze(1).cpu().tolist()
        neural_wall = time.perf_counter() - neural_start
        if next_hidden is not None:
            if hidden is None:
                hidden = next_hidden.new_zeros((len(environments), *next_hidden.shape[1:]))
            hidden[active] = next_hidden
        for row, index in enumerate(active):
            actions = public[row][0]
            before = observations[index]
            environments[index].charge_compute(neural_work_per_forward)
            if mode in DIAGNOSTIC_MODES and diagnostic_compute_units:
                environments[index].charge_compute(diagnostic_compute_units)
            action = actions[chosen[row]]
            after = environments[index].step(action)
            observations[index] = after
            start = time.process_time()
            trackers[index].update(after, action)
            diag_cpu[index] += time.process_time() - start
            traces[index].append({"step": step, "observation": before.to_dict(),
                "action_index": chosen[row], "action": asdict(action),
                "candidate_count": len(actions), "probability": probabilities[row],
                "neural_forward_wall_seconds_allocated": neural_wall / len(active),
                "active_batch_size": len(active), "neural_work_units": neural_work_per_forward,
                "remaining_steps": after.remaining_steps, "remaining_work": after.remaining_work,
                "feedback": after.feedback, **extra[row]})
    timing = {"batch_process_cpu_seconds": time.process_time()-cpu_start,
              "batch_wall_seconds": time.perf_counter()-wall_start,
              "batch_size": len(environments),
              "neural_timing_scope": "collate/device transfer + model scoring + synchronized CPU choice/probabilities; allocated equally among active batch, not serial latency"}
    return [{"trace": trace, "outcome": env.evaluate(), "truncated": not obs.done,
             "timing": timing if i == 0 else None, "unsupported_interface": unsupported.get(i),
             "progress": trackers[i].summary(),
             "deployment": {"version": DEPLOY_VERSION, "mode": mode, "diagnostic_cpu_seconds": diag_cpu[i],
                            "diagnostic_compute_units_per_decision": diagnostic_compute_units
                            if mode in DIAGNOSTIC_MODES else 0.0}}
            for i, (trace, env, obs) in enumerate(zip(traces, environments, observations))]


class TrackedEnvironment:
    """Proxy that runs the progress diagnostic alongside any actor (e.g. a supplied
    reference via ``run_episode``). Pure observation: every call is delegated."""

    def __init__(self, env, **tracker_options):
        self.env = env
        self.tracker = ProgressTracker(env.observe(), **tracker_options)

    def observe(self):
        return self.env.observe()

    def step(self, action):
        observation = self.env.step(action)
        self.tracker.update(observation, action)
        return observation

    def __getattr__(self, name):
        return getattr(self.env, name)
