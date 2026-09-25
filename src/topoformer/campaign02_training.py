"""Closed-loop learning harness. Teachers see only the declared public API.

Evaluation never receives a teacher or hidden optimal actions. Environment reward
is evaluator-only terminal feedback. Candidate padding means API availability,
not semantic correctness. The coordinator owns all experiment launches.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time
from typing import Callable

import torch
from torch import nn


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    width: int = 1024
    family: str = "lightweight"
    learning_rate: float = 3e-4
    batch_size: int = 8
    method: str = "supervised"
    entropy_weight: float = .01
    value_weight: float = .5
    gradient_clip: float = 1.0
    device: str = "cpu"
    max_steps: int = 64
    bptt_steps: int = 8
    evaluation_batch: int = 32
    training_seed_start: int | None = None
    neural_work_per_forward: float = 1.0
    rollout_mode: str = "serial"
    policy_loss_reduction: str = "decision_mean"
    # Actor-critic v2 stabilizers (defaults reproduce historical v1 exactly):
    # KL(round-start policy || current) trust region on on-policy states, and
    # per-batch standardization of policy-gradient advantages.
    kl_weight: float = 0.0
    advantage_normalization: bool = False

    def __post_init__(self):
        if not math.isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError("Invalid KL anchor weight")
        if not math.isfinite(self.neural_work_per_forward) or self.neural_work_per_forward < 0:
            raise ValueError("Invalid frozen neural tariff")
        if self.policy_loss_reduction not in {"decision_mean", "episode_mean"}:
            raise ValueError("Unknown policy objective reduction")
        if self.rollout_mode not in {"serial", "batched"}:
            raise ValueError("Unknown on-policy rollout mode")
        if self.method not in {"supervised", "actor_critic"}:
            raise ValueError("Unknown learning method")
        if min(self.width, self.batch_size, self.max_steps, self.bptt_steps, self.evaluation_batch) < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid training dimensions/rate")


@dataclass
class Frame:
    observation: list[float]
    candidates: list[list[float]]
    target: int


def public_frame(observation, feature_version="v1"):
    if feature_version in ("m1", "m2", "m3") or getattr(observation, "version", "") == "workshop-modular-v1":
        from .campaign02_modular import action_catalog as modular_catalog, encode_public as modular_encode
        if feature_version not in ("m1", "m2", "m3"):
            raise ValueError("Modular workshop observations require public feature version m1 or m2")
        actions = modular_catalog(observation)
        obs, features = modular_encode(observation, actions, feature_version)
        return actions, obs, features
    from .campaign02_world import action_catalog, encode_public
    actions = action_catalog(observation)
    obs, features = encode_public(observation, actions, feature_version)
    return actions, obs, features


def normalized_policy_config(raw: dict) -> dict:
    """Checkpoints predating feature versioning used public features v1."""
    return {"feature_version": "v1", **raw}


def collate(frames: list[Frame], device="cpu"):
    if not frames or any(not f.candidates or not 0 <= f.target < len(f.candidates) for f in frames):
        raise ValueError("Every frame needs a valid candidate target")
    count, dim = max(len(f.candidates) for f in frames), len(frames[0].candidates[0])
    candidates = torch.zeros(len(frames), count, dim, device=device)
    mask = torch.zeros(len(frames), count, dtype=torch.bool, device=device)
    for i, frame in enumerate(frames):
        candidates[i, :len(frame.candidates)] = torch.tensor(frame.candidates, device=device)
        mask[i, :len(frame.candidates)] = True
    return (torch.tensor([f.observation for f in frames], device=device), candidates, mask,
            torch.tensor([f.target for f in frames], device=device))


class PublicInterfaceCapacityError(ValueError):
    """Declared actor representation capacity exceeded; never silently truncate."""


def prepare_active(model, observations, unsupported):
    active, public = [], []
    for index, observation in enumerate(observations):
        if observation.done or index in unsupported:
            continue
        try:
            frame = prepare_frame(observation, model)
        except PublicInterfaceCapacityError as exc:
            unsupported[index] = str(exc)
            continue
        active.append(index)
        public.append(frame)
    return active, public


def prepare_frame(observation, model=None):
    version = getattr(getattr(model, "config", None), "feature_version", "v1")
    actions, obs, features = public_frame(observation, version)
    frame = Frame(obs, features, 0)
    if model is not None and hasattr(model, "prepare_public_frame"):
        frame = model.prepare_public_frame(observation, actions, frame)
    return actions, frame


def policy_batch(model, frames, device):
    return model.collate_public_frames(frames, device) if hasattr(model, "collate_public_frames") else collate(frames, device)


def score_batch(model, batch, hidden=None):
    extra = {"memory": batch[4]} if len(batch) == 5 else {}
    return model.score(batch[0], batch[1], hidden, batch[2], **extra)


def collect_teacher(env, teacher, max_steps=64, model=None):
    """Environment and teacher are separate objects; no evaluator feeds teacher."""
    frames, trace = [], []
    observation = env.observe()
    for _ in range(max_steps):
        if observation.done:
            break
        actions, frame = prepare_frame(observation, model)
        action = teacher.choose(observation, actions)
        index = actions.index(action)  # Never silently repair a teacher proposal.
        frame.target = index
        frames.append(frame)
        trace.append({"observation_hash": digest(observation.to_dict()), "action": asdict(action), "index": index})
        observation = env.step(action)
    return frames, {"steps": trace, "outcome": env.evaluate(), "truncated": not observation.done}


def supervised_loss(model, trajectories: list[list[Frame]], device="cpu", bptt_steps=8):
    if not trajectories or any(not t for t in trajectories):
        raise ValueError("Empty teacher trajectory")
    if model.config.family == "lightweight":
        frames = [f for t in trajectories for f in t]
        batch = policy_batch(model, frames, device)
        logits, _, _ = score_batch(model, batch)
        return nn.functional.cross_entropy(logits, batch[3]), len(frames)
    # Preserve the full supplied observation sequence; no randomly reset hidden
    # state in the middle of a teacher episode. Padding rows do not enter loss.
    hidden = None
    losses = []
    for step in range(max(map(len, trajectories))):
        if hidden is not None and step % bptt_steps == 0:
            hidden = hidden.detach()
        live = [step < len(t) for t in trajectories]
        frames = [t[step] if active else t[-1] for t, active in zip(trajectories, live)]
        batch = policy_batch(model, frames, device)
        logits, _, hidden = score_batch(model, batch, hidden)
        losses.extend(nn.functional.cross_entropy(logits, batch[3], reduction="none")[torch.tensor(live, device=device)].unbind())
    return torch.stack(losses).mean(), len(losses)


def live_episode(model, env, *, device="cpu", max_steps=64, sample=False, gradients=False, neural_work_per_forward=1.0, bptt_steps=8):
    """All choices are learned. Exact actor-visible observations retained for replay."""
    observation, hidden, trace, terms = env.observe(), None, [], []
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    previous_utility = 0.0
    unsupported = None
    for step in range(max_steps):
        if observation.done:
            break
        if gradients and hidden is not None and step % bptt_steps == 0:
            hidden = hidden.detach()
        try:
            actions, frame = prepare_frame(observation, model)
        except PublicInterfaceCapacityError as exc:
            unsupported = str(exc)
            break
        neural_start = time.perf_counter()
        batch = policy_batch(model, [frame], device)
        with torch.set_grad_enabled(gradients):
            logits, value, hidden = score_batch(model, batch, hidden)
            distribution = torch.distributions.Categorical(logits=logits[0])
            selected = distribution.sample() if sample else logits[0].argmax()
            index = int(selected.detach().cpu())
            if gradients:
                terms.append((distribution.log_prob(selected), value[0], distribution.entropy()))
        probability = float(distribution.probs[index].detach().cpu())
        neural_wall = time.perf_counter() - neural_start
        before = observation
        env.charge_compute(neural_work_per_forward)
        observation = env.step(actions[index])
        if gradients:
            current_utility = float(env.evaluate()["utility"])
            terms[-1] = (*terms[-1], current_utility - previous_utility)
            previous_utility = current_utility
        trace.append({"step": step, "observation": before.to_dict(), "action_index": index,
                      "action": asdict(actions[index]), "candidate_count": len(actions),
                      "probability": probability, "neural_forward_wall_seconds": neural_wall,
                      "neural_work_units": neural_work_per_forward,
                      "remaining_steps": observation.remaining_steps, "remaining_work": observation.remaining_work,
                      "feedback": observation.feedback})
    outcome = env.evaluate()
    return {"trace": trace, "outcome": outcome, "truncated": not observation.done,
            "unsupported_interface": unsupported,
            "episode_process_cpu_seconds": time.process_time()-cpu_start,
            "episode_wall_seconds": time.perf_counter()-wall_start}, terms


def batched_episodes(model, environments, *, device="cpu", max_steps=64, neural_work_per_forward=1.0):
    """Batch current public states; hidden rows retain their own episode identity."""
    observations = [env.observe() for env in environments]
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
        # One device synchronization for all choices and probabilities in a step.
        selected = logits.argmax(-1)
        chosen = selected.cpu().tolist()
        probabilities = logits.softmax(-1).gather(1, selected[:, None]).squeeze(1).cpu().tolist()
        neural_wall = time.perf_counter() - neural_start
        if next_hidden is not None:
            if hidden is None:
                hidden = next_hidden.new_zeros((len(environments), *next_hidden.shape[1:]))
            hidden[active] = next_hidden
        for row, index in enumerate(active):
            actions = public[row][0]
            before = observations[index]
            environments[index].charge_compute(neural_work_per_forward)
            after = environments[index].step(actions[chosen[row]])
            observations[index] = after
            traces[index].append({"step": step, "observation": before.to_dict(),
                "action_index": chosen[row], "action": asdict(actions[chosen[row]]),
                "candidate_count": len(actions), "probability": probabilities[row],
                "neural_forward_wall_seconds_allocated": neural_wall / len(active),
                "active_batch_size": len(active), "neural_work_units": neural_work_per_forward,
                "remaining_steps": after.remaining_steps, "remaining_work": after.remaining_work,
                "feedback": after.feedback})
    timing = {"batch_process_cpu_seconds": time.process_time()-cpu_start,
              "batch_wall_seconds": time.perf_counter()-wall_start,
              "batch_size": len(environments),
              "neural_timing_scope": "collate/device transfer + model scoring + synchronized CPU choice/probabilities; allocated equally among active batch, not serial latency"}
    # Batch timing stored once, not repeated as if independent episode work.
    return [{"trace": trace, "outcome": env.evaluate(), "truncated": not obs.done,
             "timing": timing if i == 0 else None, "unsupported_interface": unsupported.get(i)}
            for i, (trace, env, obs) in enumerate(zip(traces, environments, observations))]


def batched_on_policy(model, environments, *, device="cpu", max_steps=64,
                      neural_work_per_forward=1.0, bptt_steps=8, sample=True, reference=None):
    """Differentiable current-state-only rollouts with one RNG draw per batch.

    This is an explicit alternative to serial episode sampling: categorical RNG
    draws are consumed by time then active episode, so identical seeds do not
    imply identical sampled trajectories across the two algorithms. Episodes
    retain independent hidden rows; gradients never cross episode identities.
    """
    observations = [env.observe() for env in environments]
    traces, terms = [[] for _ in environments], [[] for _ in environments]
    kls = [[] for _ in environments]
    previous_utilities = [0.0 for _ in environments]
    unsupported = {}
    hidden = reference_hidden = None
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for step in range(max_steps):
        active, public = prepare_active(model, observations, unsupported)
        if not active:
            break
        if hidden is not None and step % bptt_steps == 0:
            hidden = hidden.detach()
        neural_start = time.perf_counter()
        batch = policy_batch(model, [frame for _, frame in public], device)
        indices = torch.tensor(active, device=device)
        current_hidden = None if hidden is None else hidden.index_select(0, indices)
        logits, values, next_hidden = score_batch(model, batch, current_hidden)
        distribution = torch.distributions.Categorical(logits=logits)
        selected = distribution.sample() if sample else logits.argmax(-1)
        logp, entropy = distribution.log_prob(selected), distribution.entropy()
        if reference is not None:
            # Frozen round-start policy on the same public states; no gradient.
            with torch.no_grad():
                reference_current = None if reference_hidden is None else reference_hidden.index_select(0, indices)
                reference_logits, _, reference_next = score_batch(reference, batch, reference_current)
                if reference_next is not None:
                    if reference_hidden is None:
                        reference_hidden = reference_next.new_zeros((len(environments), *reference_next.shape[1:]))
                    reference_hidden = reference_hidden.index_copy(0, indices, reference_next)
            valid = batch[2]
            p_ref = reference_logits.log_softmax(-1).masked_fill(~valid, 0.0)
            p_cur = logits.log_softmax(-1).masked_fill(~valid, 0.0)
            kl = (p_ref.exp()*(p_ref-p_cur)).sum(-1)
        chosen = selected.detach().cpu().tolist()
        probabilities = distribution.probs.gather(1, selected[:, None]).squeeze(1).detach().cpu().tolist()
        neural_wall = time.perf_counter()-neural_start
        if next_hidden is not None:
            if hidden is None:
                hidden = next_hidden.new_zeros((len(environments), *next_hidden.shape[1:]))
            # Functional update: do not mutate a tensor saved by autograd.
            hidden = hidden.index_copy(0, indices, next_hidden)
        for row, index in enumerate(active):
            before = observations[index]
            action = public[row][0][chosen[row]]
            environments[index].charge_compute(neural_work_per_forward)
            after = environments[index].step(action)
            observations[index] = after
            utility = float(environments[index].evaluate()["utility"])
            terms[index].append((logp[row], values[row], entropy[row], utility-previous_utilities[index]))
            if reference is not None:
                kls[index].append(kl[row])
            previous_utilities[index] = utility
            traces[index].append({"step": step, "observation": before.to_dict(), "action": asdict(action),
                "action_index": chosen[row], "candidate_count": len(public[row][0]), "probability": probabilities[row],
                "neural_work_units": neural_work_per_forward, "neural_forward_wall_seconds_allocated": neural_wall/len(active),
                "active_batch_size": len(active), "remaining_steps": after.remaining_steps,
                "remaining_work": after.remaining_work, "feedback": after.feedback})
    timing = {"batch_process_cpu_seconds": time.process_time()-cpu_start,
              "batch_wall_seconds": time.perf_counter()-wall_start, "batch_size": len(environments),
              "neural_timing_scope": "collate/device+model+CPU sampling sync; equally allocated, not serial latency"}
    results = [{"trace": trace, "outcome": env.evaluate(), "truncated": not observation.done,
                "timing": timing if i == 0 else None, "unsupported_interface": unsupported.get(i)}
               for i, (trace, env, observation) in enumerate(zip(traces, environments, observations))]
    return (results, terms, kls) if reference is not None else (results, terms)


def actor_critic_terms(episode_terms, config):
    """Undiscounted return-to-go of incremental utility, not repeated totals."""
    future_reward, losses = 0.0, []
    for logp, value, entropy, reward_delta in reversed(episode_terms):
        future_reward += reward_delta
        advantage = value.new_tensor(future_reward)-value
        losses.append(-logp*advantage.detach()+config.value_weight*advantage.square()-config.entropy_weight*entropy)
    return losses


def actor_critic_objective(episodes, config, kls=None):
    """Policy weighting is declared separately from critic regression weighting.

    decision_mean retains the historical per-decision objective. episode_mean
    sums policy-gradient and entropy terms within each episode, then averages
    episodes, including zero-decision episodes. In either mode, critic squared
    error is averaged over actual decisions; it is a separately weighted fit.
    """
    policy_by_episode, entropy_by_episode, values = [], [], []
    if getattr(config, "advantage_normalization", False) and any(episodes):
        # Standardize detached advantages over all decisions in this batch.
        raw = []
        for episode in episodes:
            future_reward, part = 0.0, []
            for logp, value, entropy, reward_delta in reversed(episode):
                future_reward += reward_delta
                part.append(future_reward-float(value.detach()))
            raw.append(list(reversed(part)))
        flat = [x for part in raw for x in part]
        mean = sum(flat)/len(flat)
        std = math.sqrt(sum((x-mean)**2 for x in flat)/len(flat)) + 1e-6
        scaled = [[(x-mean)/std for x in part] for part in raw]
    else:
        scaled = None
    for e, episode in enumerate(episodes):
        future_reward, policy_terms, entropy_terms = 0.0, [], []
        for t in reversed(range(len(episode))):
            logp, value, entropy, reward_delta = episode[t]
            future_reward += reward_delta
            advantage = value.new_tensor(future_reward)-value
            weight = advantage.detach() if scaled is None else value.new_tensor(scaled[e][t])
            policy_terms.append(-logp*weight)
            entropy_terms.append(entropy)
            values.append(advantage.square())
        policy_by_episode.append(policy_terms)
        entropy_by_episode.append(entropy_terms)
    if not values:
        raise ValueError("No live policy decisions")
    zero = values[0].new_zeros(())
    if config.policy_loss_reduction == "episode_mean":
        policy_loss = torch.stack([torch.stack(parts).sum() if parts else zero for parts in policy_by_episode]).mean()
        entropy = torch.stack([torch.stack(parts).sum() if parts else zero for parts in entropy_by_episode]).mean()
    else:
        policy_loss = torch.stack([part for parts in policy_by_episode for part in parts]).mean()
        entropy = torch.stack([part for parts in entropy_by_episode for part in parts]).mean()
    value_loss = torch.stack(values).mean()
    loss = policy_loss+config.value_weight*value_loss-config.entropy_weight*entropy
    parts = {"policy_loss": policy_loss, "critic_decision_mean_loss": value_loss, "entropy": entropy}
    if kls is not None:
        flat = [k for episode in kls for k in episode]
        kl = torch.stack(flat).mean() if flat else zero
        loss = loss+config.kl_weight*kl
        parts["kl_to_round_start"] = kl
    return loss, parts


class Learner:
    def __init__(self, model, config: TrainConfig):
        self.model, self.config = model.to(config.device), config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.updates = self.presentations = self.episodes = 0
        self.seed_cursor = config.seed * 1_000_000 if config.training_seed_start is None else config.training_seed_start
        self.resume_history: list[dict] = []
        self.curves: list[dict] = []
        self.data_hash = hashlib.sha256()

    def train_tranche(self, updates: int, world_factory: Callable, teacher_factory: Callable | None = None):
        cfg = self.config
        self.model.train()
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        reference = None
        if cfg.method == "actor_critic" and cfg.kl_weight > 0:
            if cfg.rollout_mode != "batched":
                raise ValueError("KL trust region is implemented for batched rollouts only")
            from copy import deepcopy
            reference = deepcopy(self.model).eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
        batch_kls = None
        for _ in range(updates):
            trajectories, outcomes, rollout_episodes = [], [], []
            if cfg.method == "actor_critic" and cfg.rollout_mode == "batched":
                seeds = list(range(self.seed_cursor, self.seed_cursor+cfg.batch_size))
                self.seed_cursor += cfg.batch_size
                rollout = batched_on_policy(self.model, [world_factory(seed) for seed in seeds],
                    device=cfg.device, max_steps=cfg.max_steps, neural_work_per_forward=cfg.neural_work_per_forward,
                    bptt_steps=cfg.bptt_steps, reference=reference)
                results, all_terms = rollout[:2]
                batch_kls = rollout[2] if reference is not None else None
                for result, episode_terms in zip(results, all_terms):
                    rollout_episodes.append(episode_terms)
                    self.data_hash.update(json.dumps(result["trace"], sort_keys=True).encode())
                    outcomes.append(result["outcome"])
            else:
                for _ in range(cfg.batch_size):
                    seed = self.seed_cursor
                    self.seed_cursor += 1
                    env = world_factory(seed)
                    if cfg.method == "supervised":
                        if teacher_factory is None:
                            raise ValueError("Supervised bootstrap needs an explicit public teacher")
                        frames, result = collect_teacher(env, teacher_factory(), cfg.max_steps, self.model)
                        trajectories.append(frames)
                        self.data_hash.update(json.dumps(result["steps"], sort_keys=True).encode())
                    else:
                        result, episode_terms = live_episode(self.model, env, device=cfg.device,
                            max_steps=cfg.max_steps, sample=True, gradients=True,
                            neural_work_per_forward=cfg.neural_work_per_forward, bptt_steps=cfg.bptt_steps)
                        rollout_episodes.append(episode_terms)
                        self.data_hash.update(json.dumps(result["trace"], sort_keys=True).encode())
                    outcomes.append(result["outcome"])
            self.optimizer.zero_grad(set_to_none=True)
            gradients_ready = False
            if cfg.method == "supervised":
                if hasattr(self.model, "backward_supervised"):
                    loss, count = self.model.backward_supervised(trajectories, cfg.device, cfg.bptt_steps)
                    gradients_ready = True
                else:
                    loss, count = supervised_loss(self.model, trajectories, cfg.device, cfg.bptt_steps)
            else:
                loss, objective_parts = actor_critic_objective(rollout_episodes, cfg, batch_kls)
                count = sum(map(len, rollout_episodes))
            if not gradients_ready:
                loss.backward()
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
            self.optimizer.step()
            self.updates += 1
            self.presentations += count
            self.episodes += cfg.batch_size
            self.curves.append({"update": self.updates, "decision_presentations": self.presentations,
                "unique_training_episodes": self.episodes, "loss": float(loss.detach()), "gradient_norm": float(norm),
                "mean_training_utility": sum(o["utility"] for o in outcomes)/len(outcomes),
                "training_success": sum(o["verified_success"] for o in outcomes)/len(outcomes),
                "rollout_mode": cfg.rollout_mode if cfg.method == "actor_critic" else None,
                "policy_loss_reduction": cfg.policy_loss_reduction if cfg.method == "actor_critic" else None,
                "objective_parts": {k: float(v.detach()) for k,v in objective_parts.items()} if cfg.method == "actor_critic" else None,
                "behavior_source": "supplied_public_teacher" if cfg.method == "supervised" else "sampled_learned_policy",
                "training_utility_scope": "teacher world outcome, not learned closed-loop performance" if cfg.method == "supervised" else "learned on-policy outcome including fixed neural tariff",
                "supervised_target_decisions": count if cfg.method == "supervised" else 0,
                "on_policy_decisions": count if cfg.method == "actor_critic" else 0,
                "solver_cpu_seconds": sum(o["solver_cpu_seconds"] for o in outcomes)})
        return {"updates": updates, "process_cpu_seconds": time.process_time()-start_cpu,
                "wall_seconds": time.perf_counter()-start_wall, "last": self.curves[-1] if updates else None}

    def evaluate(self, seeds, world_factory, output: Path | None = None):
        self.model.eval()
        rows = []
        seeds = list(seeds)
        with torch.no_grad():
            for offset in range(0, len(seeds), self.config.evaluation_batch):
                chunk = seeds[offset:offset+self.config.evaluation_batch]
                results = batched_episodes(self.model, [world_factory(s) for s in chunk],
                    device=self.config.device, max_steps=self.config.max_steps, neural_work_per_forward=self.config.neural_work_per_forward)
                rows.extend({"seed": seed, **result} for seed, result in zip(chunk, results))
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(output, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True)+"\n")
        n = len(rows)
        if not n:
            raise ValueError("Evaluation requires explicit nonempty seeds")
        return {"examples": n, "success": sum(r["outcome"]["verified_success"] for r in rows)/n,
                "utility": sum(r["outcome"]["utility"] for r in rows)/n,
                "cost": sum(r["outcome"]["cost"] for r in rows)/n,
                "unsupported_interface_count": sum(r.get("unsupported_interface") is not None for r in rows),
                "seeds_hash": digest([r["seed"] for r in rows])}

    def save(self, path: Path, metadata=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": 1, "config": asdict(self.config), "policy_config": asdict(self.model.config),
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "updates": self.updates, "presentations": self.presentations, "episodes": self.episodes,
            "seed_cursor": self.seed_cursor, "curves": self.curves, "data_hash": self.data_hash.hexdigest(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(), "metadata": metadata or {},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "resume_history": self.resume_history}, path)

    def load(self, path: Path, *, allow_config_changes=False, reset_learning_rate=False):
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        if normalized_policy_config(checkpoint["policy_config"]) != asdict(self.model.config):
            raise ValueError("Checkpoint architecture mismatch")
        current = asdict(self.config)
        differences = {k: {"checkpoint": checkpoint["config"].get(k), "requested": v}
                       for k, v in current.items() if checkpoint["config"].get(k) != v}
        semantic_changes = set(differences) - {"device", "evaluation_batch"}
        if semantic_changes and not allow_config_changes:
            raise ValueError(f"Explicit resume configuration change authorization required: {sorted(semantic_changes)}")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if reset_learning_rate:
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate
        self.resume_history = list(checkpoint.get("resume_history", []))
        self.resume_history.append({"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config_changes": differences, "learning_rate_policy": "requested" if reset_learning_rate else "inherited",
            "actual_learning_rates": [group["lr"] for group in self.optimizer.param_groups]})
        for key in ("updates", "presentations", "episodes", "seed_cursor", "curves"):
            setattr(self, key, checkpoint[key])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        random.setstate(checkpoint["python_rng"])
        if checkpoint["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        # Hash chain restart explicitly binds the previous digest, not fake replay.
        self.data_hash = hashlib.sha256(checkpoint["data_hash"].encode())
        return checkpoint["metadata"]


def independent_address_seed(world_seed: int, namespace: str) -> int:
    """Separate deterministic RNG stream; spelling is never a policy feature."""
    return int(digest({"namespace": namespace, "world_seed": world_seed, "role": "record-addresses"})[:16], 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-seed-start", type=int)
    parser.add_argument("--address-namespace", default="extended-02-training-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--family", choices=("lightweight", "recurrent"), default="lightweight")
    parser.add_argument("--method", choices=("supervised", "actor_critic"), default="supervised")
    parser.add_argument("--policy-loss-reduction", choices=("decision_mean", "episode_mean"), default="decision_mean")
    parser.add_argument("--rollout-mode", choices=("serial", "batched"), default="serial")
    parser.add_argument("--teacher", choices=("always_tool", "cheap_first", "cheap", "cheap_first_fallback_v2"), default="always_tool")
    parser.add_argument("--executor", choices=("isolated", "persistent"), default="persistent")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--bptt-steps", type=int, default=8)
    parser.add_argument("--evaluation-batch", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--neural-work-per-forward", type=float, default=1.0)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--evaluate-every", type=int, default=0,
                        help="Development only; fixed seeds reused and recorded, never confirmation")
    parser.add_argument("--world-json", default="{}")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-resume-config-changes", action="store_true")
    parser.add_argument("--reset-learning-rate", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.evaluate_every < 0 or args.validation_examples < 1:
        parser.error("Nonnegative update/interval and positive evaluation support required")
    from .campaign02_policy import CandidatePolicy, PolicyConfig
    from .campaign02_world import Workshop, generate_world, protocol_executor
    from .campaign02_references import make_reference
    from .campaign02_protocol import BoundedSolver
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = TrainConfig(seed=args.seed, width=args.width, family=args.family, batch_size=args.batch,
        method=args.method, device=args.device, learning_rate=args.learning_rate, max_steps=args.max_steps,
        bptt_steps=args.bptt_steps, evaluation_batch=args.evaluation_batch, training_seed_start=args.training_seed_start, neural_work_per_forward=args.neural_work_per_forward, rollout_mode=args.rollout_mode, policy_loss_reduction=args.policy_loss_reduction)
    world_kwargs = json.loads(args.world_json)
    args.output.mkdir(parents=True, exist_ok=True)
    manager = BoundedSolver() if args.executor == "persistent" else nullcontext(None)
    with manager as solver:
        executor = partial(protocol_executor, execute_call=solver.execute) if solver else protocol_executor
        def factory(seed):
            return Workshop(generate_world(seed, **world_kwargs), executor=executor,
                address_seed=independent_address_seed(seed, args.address_namespace))
        _, obs, candidates = public_frame(factory(cfg.seed).observe())
        policy = CandidatePolicy(PolicyConfig(len(obs), len(candidates[0]), width=cfg.width, family=cfg.family))
        learner = Learner(policy, cfg)
        if args.resume:
            learner.load(args.resume, allow_config_changes=args.allow_resume_config_changes,
                         reset_learning_rate=args.reset_learning_rate)
        train_start = learner.seed_cursor
        train_end = train_start + args.steps * cfg.batch_size
        validation = range(args.validation_seed, args.validation_seed+args.validation_examples)
        if train_start < validation.stop and validation.start < train_end:
            raise ValueError("Training and development episode seed intervals overlap")
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(cfg.device))
        development, timings = [], []
        remaining = args.steps
        while remaining:
            count = min(remaining, args.evaluate_every or remaining)
            timings.append(learner.train_tranche(count, factory, lambda: make_reference(args.teacher)))
            remaining -= count
            if args.evaluate_every:
                result = learner.evaluate(validation, factory, args.output/f"development-{learner.updates}.jsonl.gz")
                development.append({"update": learner.updates, **result})
                learner.save(args.output/"checkpoint.pt", {"selection_rule": "last update", "development": development})
        metrics = development[-1] if development else learner.evaluate(validation, factory, args.output/"validation.jsonl.gz")
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ("campaign02_training.py", "campaign02_policy.py", "campaign02_world.py", "campaign02_protocol.py", "campaign02_references.py")}
        metadata = {"selection_rule": "last update; fixed development seeds descriptive, no confirmation",
            "world": world_kwargs, "world_hash": digest(world_kwargs), "timings": timings, "validation": metrics,
            "development": development, "parameter_count": sum(p.numel() for p in policy.parameters()),
            "workspace_width": cfg.width, "observation_dim": len(obs), "candidate_dim": len(candidates[0]),
            "memory": {"process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "process_peak_rss_scope": "whole process lifetime; Linux KiB, excludes child solvers",
                "cuda_scope": "peak since before fit/development evaluation; allocated/reserved are not device capacity",
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(cfg.device)) if torch.device(cfg.device).type == "cuda" else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(cfg.device)) if torch.device(cfg.device).type == "cuda" else None,
                "cuda_device_capacity_bytes": torch.cuda.get_device_properties(torch.device(cfg.device)).total_memory if torch.device(cfg.device).type == "cuda" else None},
            "workspace_rows": policy.config.workspace_rows if cfg.family == "recurrent" else 0,
            "recurrent_phases": 4 if cfg.family == "recurrent" else 0,
            "decision_presentations": learner.presentations, "unique_training_episodes": learner.episodes,
            "training_seed_interval_this_invocation": [train_start, train_end],
            "address_seed_rule": "SHA256 of world seed + role + independent namespace", "address_namespace": args.address_namespace,
            "teacher": args.teacher if cfg.method == "supervised" else None, "executor": args.executor,
            "threads": args.threads, "source_hashes": sources,
            "neural_work_per_forward": cfg.neural_work_per_forward,
            "teacher_cost_scope": "public reference environmental/solver costs only; supervised prediction cost is training work in outer ledger",
            "solver_accounting": ({"startup_wall_seconds": solver.startup_wall_seconds,
                "startup_child_cpu_seconds": solver.startup_child_cpu_seconds, "call_wall_seconds": solver.call_wall_seconds,
                "restarts": solver.restarts} if solver else {"version": "isolated-per-call"})}
        learner.save(args.output/"checkpoint.pt", metadata)
        (args.output/"curves.json").write_text(json.dumps(learner.curves, indent=2))
        (args.output/"summary.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
