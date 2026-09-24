"""Closed-loop learning harness. Teachers see only the declared public API.

Evaluation never receives a teacher or hidden optimal actions. Environment reward
is evaluator-only terminal feedback. Candidate padding means API availability,
not semantic correctness. The coordinator owns all experiment launches.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
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

    def __post_init__(self):
        if self.method not in {"supervised", "actor_critic"}:
            raise ValueError("Unknown learning method")
        if min(self.width, self.batch_size, self.max_steps) < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid training dimensions/rate")


@dataclass
class Frame:
    observation: list[float]
    candidates: list[list[float]]
    target: int


def public_frame(observation):
    from .campaign02_world import action_catalog, encode_action, encode_observation
    actions = action_catalog(observation)
    return actions, encode_observation(observation), [encode_action(observation, a) for a in actions]


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


def collect_teacher(env, teacher, max_steps=64):
    """Environment and teacher are separate objects; no evaluator feeds teacher."""
    frames, trace = [], []
    observation = env.observe()
    for _ in range(max_steps):
        if observation.done:
            break
        actions, obs, features = public_frame(observation)
        action = teacher.choose(observation, actions)
        index = actions.index(action)  # Never silently repair a teacher proposal.
        frames.append(Frame(obs, features, index))
        trace.append({"observation_hash": digest(observation.to_dict()), "action": asdict(action), "index": index})
        observation = env.step(action)
    return frames, {"steps": trace, "outcome": env.evaluate(), "truncated": not observation.done}


def supervised_loss(model, trajectories: list[list[Frame]], device="cpu", bptt_steps=8):
    if not trajectories or any(not t for t in trajectories):
        raise ValueError("Empty teacher trajectory")
    if model.config.family == "lightweight":
        frames = [f for t in trajectories for f in t]
        obs, candidates, mask, targets = collate(frames, device)
        logits, _, _ = model.score(obs, candidates, mask=mask)
        return nn.functional.cross_entropy(logits, targets), len(frames)
    # Preserve the full supplied observation sequence; no randomly reset hidden
    # state in the middle of a teacher episode. Padding rows do not enter loss.
    hidden = None
    losses = []
    for step in range(max(map(len, trajectories))):
        if hidden is not None and step % bptt_steps == 0:
            hidden = hidden.detach()
        live = [step < len(t) for t in trajectories]
        frames = [t[step] if active else t[-1] for t, active in zip(trajectories, live)]
        obs, candidates, mask, targets = collate(frames, device)
        logits, _, hidden = model.score(obs, candidates, hidden, mask)
        losses.extend(nn.functional.cross_entropy(logits, targets, reduction="none")[torch.tensor(live, device=device)].unbind())
    return torch.stack(losses).mean(), len(losses)


def live_episode(model, env, *, device="cpu", max_steps=64, sample=False, gradients=False):
    """All choices are learned. Exact actor-visible observations retained for replay."""
    observation, hidden, trace, terms = env.observe(), None, [], []
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    previous_utility = 0.0
    for step in range(max_steps):
        if observation.done:
            break
        actions, obs, features = public_frame(observation)
        batch = collate([Frame(obs, features, 0)], device)
        with torch.set_grad_enabled(gradients):
            logits, value, hidden = model.score(batch[0], batch[1], hidden, batch[2])
            distribution = torch.distributions.Categorical(logits=logits[0])
            selected = distribution.sample() if sample else logits[0].argmax()
            index = int(selected.detach().cpu())
            if gradients:
                terms.append((distribution.log_prob(selected), value[0], distribution.entropy()))
        before = observation
        observation = env.step(actions[index])
        if gradients:
            current_utility = float(env.evaluate()["utility"])
            terms[-1] = (*terms[-1], current_utility - previous_utility)
            previous_utility = current_utility
        trace.append({"step": step, "observation": before.to_dict(), "action_index": index,
                      "action": asdict(actions[index]), "candidate_count": len(actions),
                      "probability": float(distribution.probs[index].detach().cpu()),
                      "remaining_steps": observation.remaining_steps, "remaining_work": observation.remaining_work,
                      "feedback": observation.feedback})
    outcome = env.evaluate()
    return {"trace": trace, "outcome": outcome, "truncated": not observation.done,
            "episode_process_cpu_seconds": time.process_time()-cpu_start,
            "episode_wall_seconds": time.perf_counter()-wall_start}, terms


def batched_episodes(model, environments, *, device="cpu", max_steps=64):
    """Batch current public states; hidden rows retain their own episode identity."""
    observations = [env.observe() for env in environments]
    traces = [[] for _ in environments]
    hidden = None
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for step in range(max_steps):
        active = [i for i, o in enumerate(observations) if not o.done]
        if not active:
            break
        public = [public_frame(observations[i]) for i in active]
        batch = collate([Frame(obs, features, 0) for _, obs, features in public], device)
        current_hidden = None if hidden is None else hidden[active]
        logits, _, next_hidden = model.score(batch[0], batch[1], current_hidden, batch[2])
        # One device synchronization for all choices and probabilities in a step.
        selected = logits.argmax(-1)
        chosen = selected.cpu().tolist()
        probabilities = logits.softmax(-1).gather(1, selected[:, None]).squeeze(1).cpu().tolist()
        if next_hidden is not None:
            if hidden is None:
                hidden = next_hidden.new_zeros((len(environments), *next_hidden.shape[1:]))
            hidden[active] = next_hidden
        for row, index in enumerate(active):
            actions = public[row][0]
            before = observations[index]
            after = environments[index].step(actions[chosen[row]])
            observations[index] = after
            traces[index].append({"step": step, "observation": before.to_dict(),
                "action_index": chosen[row], "action": asdict(actions[chosen[row]]),
                "candidate_count": len(actions), "probability": probabilities[row],
                "remaining_steps": after.remaining_steps, "remaining_work": after.remaining_work,
                "feedback": after.feedback})
    timing = {"batch_process_cpu_seconds": time.process_time()-cpu_start,
              "batch_wall_seconds": time.perf_counter()-wall_start,
              "batch_size": len(environments)}
    # Batch timing stored once, not repeated as if independent episode work.
    return [{"trace": trace, "outcome": env.evaluate(), "truncated": not obs.done,
             "timing": timing if i == 0 else None}
            for i, (trace, env, obs) in enumerate(zip(traces, environments, observations))]


class Learner:
    def __init__(self, model, config: TrainConfig):
        self.model, self.config = model.to(config.device), config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.updates = self.presentations = self.episodes = 0
        self.seed_cursor = config.seed * 1_000_000
        self.curves: list[dict] = []
        self.data_hash = hashlib.sha256()

    def train_tranche(self, updates: int, world_factory: Callable, teacher_factory: Callable | None = None):
        cfg = self.config
        self.model.train()
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        for _ in range(updates):
            trajectories, outcomes, terms = [], [], []
            for _ in range(cfg.batch_size):
                seed = self.seed_cursor
                self.seed_cursor += 1
                env = world_factory(seed)
                if cfg.method == "supervised":
                    if teacher_factory is None:
                        raise ValueError("Supervised bootstrap needs an explicit public teacher")
                    frames, result = collect_teacher(env, teacher_factory(), cfg.max_steps)
                    trajectories.append(frames)
                    self.data_hash.update(json.dumps(result["steps"], sort_keys=True).encode())
                else:
                    result, episode_terms = live_episode(self.model, env, device=cfg.device,
                        max_steps=cfg.max_steps, sample=True, gradients=True)
                    future_reward = 0.0
                    for logp, value, entropy, reward_delta in reversed(episode_terms):
                        future_reward += reward_delta
                        advantage = value.new_tensor(future_reward) - value
                        terms.append(-logp * advantage.detach() + cfg.value_weight * advantage.square()
                                     - cfg.entropy_weight * entropy)
                    self.data_hash.update(json.dumps(result["trace"], sort_keys=True).encode())
                outcomes.append(result["outcome"])
            if cfg.method == "supervised":
                loss, count = supervised_loss(self.model, trajectories, cfg.device, cfg.bptt_steps)
            else:
                if not terms:
                    raise ValueError("No live policy decisions")
                loss, count = torch.stack(terms).mean(), len(terms)
            self.optimizer.zero_grad(set_to_none=True)
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
                    device=self.config.device, max_steps=self.config.max_steps)
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
                "seeds_hash": digest([r["seed"] for r in rows])}

    def save(self, path: Path, metadata=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": 1, "config": asdict(self.config), "policy_config": asdict(self.model.config),
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "updates": self.updates, "presentations": self.presentations, "episodes": self.episodes,
            "seed_cursor": self.seed_cursor, "curves": self.curves, "data_hash": self.data_hash.hexdigest(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(), "metadata": metadata or {},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, path)

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        if checkpoint["policy_config"] != asdict(self.model.config):
            raise ValueError("Checkpoint architecture mismatch")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for key in ("updates", "presentations", "episodes", "seed_cursor", "curves"):
            setattr(self, key, checkpoint[key])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        random.setstate(checkpoint["python_rng"])
        if checkpoint["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        # Hash chain restart explicitly binds the previous digest, not fake replay.
        self.data_hash = hashlib.sha256(checkpoint["data_hash"].encode())
        return checkpoint["metadata"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--family", choices=("lightweight", "recurrent"), default="lightweight")
    parser.add_argument("--method", choices=("supervised", "actor_critic"), default="supervised")
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--world-json", default="{}")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    from .campaign02_policy import CandidatePolicy, PolicyConfig
    from .campaign02_world import Workshop, generate_world
    from .campaign02_references import ReferencePolicy
    from .campaign02_world import protocol_executor
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = TrainConfig(seed=args.seed, width=args.width, family=args.family, batch_size=args.batch,
                      method=args.method, device=args.device)
    world_kwargs = json.loads(args.world_json)
    factory = lambda seed: Workshop(generate_world(seed, **world_kwargs), executor=protocol_executor)
    env = factory(cfg.seed)
    _, obs, candidates = public_frame(env.observe())
    policy = CandidatePolicy(PolicyConfig(len(obs), len(candidates[0]), width=cfg.width, family=cfg.family))
    learner = Learner(policy, cfg)
    if args.resume:
        learner.load(args.resume)
    timing = learner.train_tranche(args.steps, factory, lambda: ReferencePolicy(mode="cheap_first"))
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = learner.evaluate(range(args.validation_seed, args.validation_seed+args.validation_examples),
                               factory, args.output/"validation.jsonl.gz")
    metadata = {"selection_rule": "last update; validation descriptive only", "world": world_kwargs,
                "world_hash": digest(world_kwargs), "timing": timing, "validation": metrics,
                "parameter_count": sum(p.numel() for p in policy.parameters())}
    learner.save(args.output/"checkpoint.pt", metadata)
    (args.output/"curves.json").write_text(json.dumps(learner.curves, indent=2))
    (args.output/"summary.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
