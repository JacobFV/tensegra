"""Versioned public-table consumer; no supplied plan or structural attention bias.

Both policy families receive identical typed public memory and candidate links.
The keyed mean is a supplied addressing prior. Neural queries/readout are learned.
Old CandidatePolicy and historical checkpoint entry points remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import torch
from torch import nn

from .campaign02_policy import CandidatePolicy, PolicyConfig
from .campaign02_training import (Frame, Learner, TrainConfig, PublicInterfaceCapacityError,
                                 collate, independent_address_seed, public_frame)


@dataclass(frozen=True)
class MemoryPolicyConfig(PolicyConfig):
    memory_dim: int = 1
    max_memory_nodes: int = 8192
    max_handles: int = 512
    zero_memory: bool = False
    supervised_frame_batch: int = 8
    interface_version: str = "public-tree-memory-v1"

    def __post_init__(self):
        super().__post_init__()
        if min(self.memory_dim, self.max_memory_nodes, self.max_handles, self.supervised_frame_batch) < 1:
            raise ValueError("Positive memory dimensions and explicit caps required")
        if self.interface_version != "public-tree-memory-v1":
            raise ValueError("Unknown public memory interface")


@dataclass
class MemoryFrame(Frame):
    rows: torch.Tensor
    links: list[list[int]]
    stats: dict


class MemoryCandidatePolicy(CandidatePolicy):
    def __init__(self, config: MemoryPolicyConfig):
        super().__init__(config)
        self.memory_encoder = nn.Sequential(nn.Linear(config.memory_dim, config.width), nn.LayerNorm(config.width))
        self.memory_read = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.memory_query_norm = nn.LayerNorm(config.width)
        self.encoding_profile = {"frames": 0, "rows": 0, "max_rows": 0, "cpu_seconds": 0., "wall_seconds": 0.}
        self.collate_profile = {"calls": 0, "max_frames": 0, "max_padded_rows": 0, "cpu_seconds": 0., "wall_seconds": 0.}

    def prepare_public_frame(self, observation, actions, legacy):
        from .campaign02_memory import encode_memory, MemoryLimits, MemoryCapacityError
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            memory = encode_memory(observation, actions,
                limits=MemoryLimits(max_nodes=self.config.max_memory_nodes, max_handles=self.config.max_handles))
        except MemoryCapacityError as exc:
            raise PublicInterfaceCapacityError(str(exc)) from exc
        rows = torch.tensor(memory.rows, dtype=torch.float32)
        if rows.ndim != 2 or rows.shape[1] != self.config.memory_dim:
            raise ValueError("Memory schema dimension mismatch")
        profile = self.encoding_profile
        profile["frames"] += 1
        profile["rows"] += len(rows)
        profile["max_rows"] = max(profile["max_rows"], len(rows))
        profile["cpu_seconds"] += time.process_time()-cpu
        profile["wall_seconds"] += time.perf_counter()-wall
        return MemoryFrame(legacy.observation, legacy.candidates, legacy.target, rows, memory.action_links, memory.stats)

    def collate_public_frames(self, frames, device):
        wall, cpu = time.perf_counter(), time.process_time()
        legacy = collate(frames, device)
        count, actions, rows = len(frames), legacy[1].shape[1], max(len(f.rows) for f in frames)
        memory = torch.zeros(count, rows, self.config.memory_dim, device=device)
        available = torch.zeros(count, rows, dtype=torch.bool, device=device)
        links = torch.zeros(count, actions, rows, device=device)
        for i, frame in enumerate(frames):
            if len(frame.links) != len(frame.candidates):
                raise ValueError("Every candidate requires its own public binding row")
            memory[i, :len(frame.rows)] = frame.rows.to(device)
            available[i, :len(frame.rows)] = True
            for j, indices in enumerate(frame.links):
                if any(k < 0 or k >= len(frame.rows) for k in indices):
                    raise ValueError("Candidate binding outside public memory")
                if indices:
                    links[i, j, indices] = 1/len(set(indices))
        if self.config.zero_memory:
            # Preserve shape/parameters/nominal compute, remove factual values,
            # relation links AND varying attention availability as information.
            memory.zero_()
            links.zero_()
            available.zero_()
            available[:, 0] = True
        profile = self.collate_profile
        profile["calls"] += 1
        profile["max_frames"] = max(profile["max_frames"], count)
        profile["max_padded_rows"] = max(profile["max_padded_rows"], rows)
        profile["cpu_seconds"] += time.process_time()-cpu
        profile["wall_seconds"] += time.perf_counter()-wall
        return (*legacy, {"rows": memory, "available": available, "links": links})

    def score(self, obs, candidates, hidden=None, mask=None, *, memory):
        if mask is None:
            mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        if not mask.any(-1).all() or not memory["available"].any(-1).all():
            raise ValueError("Every actor needs public candidates and memory")
        encoded = self.candidate(candidates)
        facts = self.memory_encoder(memory["rows"])
        linked = torch.bmm(memory["links"], facts)
        read = self.memory_read(self.memory_query_norm(encoded+linked), facts, facts,
                                key_padding_mask=~memory["available"], need_weights=False)[0]
        encoded = encoded+linked+read
        observed = self.observation(obs)
        next_hidden = None
        if self.config.family == "recurrent":
            if hidden is None:
                hidden = self.initial.unsqueeze(0).expand(obs.shape[0], -1, -1)
            contents = torch.cat((observed[:, None], encoded, facts), dim=1)
            padding = torch.cat((torch.zeros(obs.shape[0], 1, dtype=torch.bool, device=obs.device),
                                 ~mask, ~memory["available"]), dim=1)
            next_hidden = hidden
            for phase in self.phases:
                next_hidden = phase(next_hidden, contents, padding)
            context = next_hidden.mean(1)
        else:
            if hidden is not None:
                raise ValueError("Lightweight memory policy has no recurrent hidden state")
            context = self.context(observed)
        logits = self.scorer(encoded+context[:, None]).squeeze(-1).masked_fill(~mask, -torch.inf)
        return logits, self.value(context).squeeze(-1), next_hidden

    forward = score

    def backward_supervised(self, trajectories, device, bptt_steps):
        """Exact decision-mean gradient accumulation bounds lightweight memory.

        Recurrent episodes retain the original sequence/BPTT objective. Their
        episode batch must be profiled explicitly before large runs.
        """
        if self.config.family == "recurrent":
            from .campaign02_training import supervised_loss
            loss, count = supervised_loss(self, trajectories, device, bptt_steps)
            loss.backward()
            return loss.detach(), count
        frames = [frame for trajectory in trajectories for frame in trajectory]
        count = len(frames)
        total = torch.zeros((), device=device)
        for offset in range(0, count, self.config.supervised_frame_batch):
            batch = self.collate_public_frames(frames[offset:offset+self.config.supervised_frame_batch], device)
            logits, _, _ = self.score(batch[0], batch[1], mask=batch[2], memory=batch[4])
            loss = nn.functional.cross_entropy(logits, batch[3], reduction="sum")/count
            loss.backward()
            total += loss.detach()
        return total, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an acquisition directory")
    raw = json.loads(args.config.read_text())
    from .campaign02_memory import ROW_DIM
    from .campaign02_protocol import BoundedSolver
    from .campaign02_references import make_reference
    from .campaign02_world import Workshop, generate_world, protocol_executor
    torch.set_num_threads(raw.get("threads", 1))
    train = TrainConfig(**raw["train"])
    if train.method != "supervised":
        raise ValueError("Initial public-memory acquisition CLI only supports declared supervised bootstrap")
    random.seed(train.seed)
    torch.manual_seed(train.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    with BoundedSolver() as solver:
        executor = partial(protocol_executor, execute_call=solver.execute)
        world_mix = raw.get("world_mix", [{}])
        namespace = raw.get("address_namespace", "extended-02-public-memory-v1")
        def factory(seed):
            from .campaign02_population import mix_index
            return Workshop(generate_world(seed, **world_mix[mix_index(seed, len(world_mix))]), executor=executor,
                             address_seed=independent_address_seed(seed, namespace))
        _, obs, candidates = public_frame(factory(train.seed).observe())
        policy_cfg = MemoryPolicyConfig(observation_dim=len(obs), candidate_dim=len(candidates[0]),
            width=train.width, family=train.family, memory_dim=ROW_DIM, **raw.get("memory", {}))
        model = MemoryCandidatePolicy(policy_cfg)
        learner = Learner(model, train)
        if torch.device(train.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(train.device))
        teacher = raw.get("teacher", "cheap_first_fallback_v2")
        updates = raw["updates"]
        eval_every = raw.get("evaluate_every", updates)
        if updates < 1 or eval_every < 1:
            raise ValueError("Positive acquisition/evaluation intervals required")
        dev_start, dev_n = raw["development_seed_start"], raw.get("development_examples", 128)
        train_end = learner.seed_cursor+updates*train.batch_size
        if learner.seed_cursor < dev_start+dev_n and dev_start < train_end:
            raise ValueError("Training/development overlap")
        development, timings = [], []
        while learner.updates < updates:
            count = min(eval_every, updates-learner.updates)
            timings.append(learner.train_tranche(count, factory, lambda: make_reference(teacher)))
            result = learner.evaluate(range(dev_start, dev_start+dev_n), factory,
                                      args.output/f"development-{learner.updates}.jsonl.gz")
            development.append({"update": learner.updates, **result})
            learner.save(args.output/"checkpoint.pt", {"interface": policy_cfg.interface_version,
                "selection_rule": "last update; fixed development only", "development": development})
        metadata = {"config": raw, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("campaign02_memory.py", "campaign02_training.py", "campaign02_policy.py", "campaign02_world.py", "campaign02_references.py", "campaign02_protocol.py")},
            "policy_config": asdict(policy_cfg), "parameter_count": sum(p.numel() for p in model.parameters()),
            "selection_rule": "last update; no confirmation", "development": development, "timings": timings,
            "encoding": model.encoding_profile, "collate": model.collate_profile,
            "wall_seconds": time.perf_counter()-start_wall, "parent_cpu_seconds": time.process_time()-start_cpu,
            "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(train.device)) if torch.device(train.device).type == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(train.device)) if torch.device(train.device).type == "cuda" else None,
            "device_capacity_bytes": torch.cuda.get_device_properties(torch.device(train.device)).total_memory if torch.device(train.device).type == "cuda" else None,
            "solver_startup_wall_seconds": solver.startup_wall_seconds, "solver_call_wall_seconds": solver.call_wall_seconds}
        learner.save(args.output/"checkpoint.pt", metadata)
        (args.output/"curves.json").write_text(json.dumps(learner.curves, indent=2))
        (args.output/"summary.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
