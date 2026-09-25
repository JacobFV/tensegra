"""Public-feature policies and auditable population bookkeeping for extended-02.

No environment, executor, hidden task state, or action schedule enters this module.
Candidate masks must encode publicly decidable API validity, never optimality.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyConfig:
    observation_dim: int
    candidate_dim: int
    width: int = 1024
    family: Literal["lightweight", "recurrent"] = "lightweight"
    workspace_rows: int = 4
    heads: int = 8
    feature_version: str = "v1"

    def __post_init__(self):
        if self.feature_version not in {"v1", "v2", "m1", "m2"}:
            raise ValueError("Unknown public feature version")
        if min(self.observation_dim, self.candidate_dim, self.width, self.workspace_rows, self.heads) < 1:
            raise ValueError("Dimensions must be positive")
        if self.family not in {"lightweight", "recurrent"}:
            raise ValueError("Unknown policy family")
        if self.width % self.heads:
            raise ValueError("Width must be divisible by heads")


class _Phase(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, state: Tensor, memory: Tensor, padding: Tensor) -> Tensor:
        query = self.norm(state)
        # Recomputed from this state and current public inputs; no KV history cache.
        keys = torch.cat((query, memory), dim=1)
        mask = torch.cat((torch.zeros(state.shape[:2], device=state.device, dtype=torch.bool), padding), dim=1)
        state = state + self.attention(query, keys, keys, key_padding_mask=mask, need_weights=False)[0]
        return state + self.ff(state)


class CandidatePolicy(nn.Module):
    """Same-width candidate policies; recurrent state contains no exact registers.

    One score call advances one microstep. A runner may select a public latent-step
    action and call again without advancing the world. Detach hidden explicitly at
    rollout boundaries; this class does not silently truncate gradients.
    """
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        w = config.width
        self.observation = nn.Linear(config.observation_dim, w)
        self.candidate = nn.Linear(config.candidate_dim, w)
        if config.family == "recurrent":
            self.initial = nn.Parameter(torch.randn(config.workspace_rows, w) / math.sqrt(w))
            self.phases = nn.ModuleList([_Phase(w, config.heads) for _ in range(4)])
        else:
            self.context = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU())
        self.scorer = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1))
        self.value = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 1))

    def score(self, obs: Tensor, candidates: Tensor, hidden: Tensor | None = None,
              mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
        c = self.config
        if obs.ndim != 2 or obs.shape[-1] != c.observation_dim:
            raise ValueError("Expected observation [batch, observation_dim]")
        if candidates.ndim != 3 or candidates.shape[0] != obs.shape[0] or candidates.shape[-1] != c.candidate_dim:
            raise ValueError("Expected candidates [batch, actions, candidate_dim]")
        if mask is None:
            mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        if mask.dtype != torch.bool or mask.shape != candidates.shape[:2]:
            raise ValueError("Mask must be boolean [batch, actions], true means available")
        if not mask.any(dim=1).all():
            raise ValueError("Every row requires a public action, e.g. abstain")
        observed = self.observation(obs)
        encoded = self.candidate(candidates)
        next_hidden = None
        if c.family == "recurrent":
            if hidden is None:
                hidden = self.initial.unsqueeze(0).expand(obs.shape[0], -1, -1)
            if hidden.shape != (obs.shape[0], c.workspace_rows, c.width):
                raise ValueError("Incorrect recurrent state shape")
            memory = torch.cat((observed.unsqueeze(1), encoded), dim=1)
            padding = torch.cat((torch.zeros((obs.shape[0], 1), device=mask.device, dtype=torch.bool), ~mask), dim=1)
            next_hidden = hidden
            for phase in self.phases:
                next_hidden = phase(next_hidden, memory, padding)
            context = next_hidden.mean(dim=1)
        else:
            if hidden is not None:
                raise ValueError("Lightweight policy has no recurrent state")
            context = self.context(observed)
        logits = self.scorer(encoded + context.unsqueeze(1)).squeeze(-1)
        return logits.masked_fill(~mask, -torch.inf), self.value(context).squeeze(-1), next_hidden

    forward = score


@dataclass(frozen=True)
class Member:
    member_id: str
    lineage_id: str
    generation: int = 0
    parent_id: str | None = None
    learning_rate: float = 3e-4
    entropy_weight: float = 0.01
    optimizer_policy: str = "reset"
    allocation_index: int = 0
    mutation: tuple[tuple[str, float], ...] = ()

    def record(self) -> dict:
        return asdict(self)


def initial_population(size: int = 6, *, prefix: str = "population") -> list[Member]:
    if size < 1:
        raise ValueError("Population must be nonempty")
    return [Member(f"{prefix}/g0/m{i}", f"{prefix}/root{i}") for i in range(size)]


def equal_allocations(members: list[Member], round_index: int, updates: int) -> list[dict]:
    """Declarative equal work slots; runtime and evaluation costs remain metered."""
    if round_index < 0 or updates < 1 or len({m.member_id for m in members}) != len(members):
        raise ValueError("Invalid allocation")
    return [{"member_id": m.member_id, "round": round_index, "updates": updates,
             "allocation_index": round_index * len(members) + i} for i, m in enumerate(members)]


def offspring(parent: Member, member_id: str, *, rng: random.Random,
              optimizer_policy: Literal["reset", "inherit"], allocation_index: int) -> Member:
    if optimizer_policy not in {"reset", "inherit"} or member_id == parent.member_id:
        raise ValueError("Explicit optimizer policy and new identity required")
    lr_factor, entropy_factor = rng.choice((0.8, 1.2)), rng.choice((0.8, 1.2))
    return Member(member_id, parent.lineage_id, parent.generation + 1, parent.member_id,
                  min(1e-2, max(1e-6, parent.learning_rate * lr_factor)),
                  min(1.0, max(1e-6, parent.entropy_weight * entropy_factor)),
                  optimizer_policy, allocation_index,
                  (("learning_rate_factor", lr_factor), ("entropy_factor", entropy_factor)))


def clone_checkpoint(model: CandidatePolicy, optimizer: torch.optim.Optimizer,
                     *, optimizer_policy: Literal["reset", "inherit"]) -> dict:
    """Independent CPU snapshot; checkpoints must be coordinator-controlled files."""
    if optimizer_policy not in {"reset", "inherit"}:
        raise ValueError("Explicit reset/inherit required")
    state = deepcopy(optimizer.state_dict()) if optimizer_policy == "inherit" else None
    def cpu(value):
        if isinstance(value, Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        return value
    return {"format_version": 1, "config": asdict(model.config),
            "model": cpu(model.state_dict()), "optimizer": cpu(state),
            "optimizer_policy": optimizer_policy,
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"}


def restore_checkpoint(snapshot: dict, model: CandidatePolicy, optimizer: torch.optim.Optimizer,
                       *, learning_rate: float) -> None:
    if snapshot["format_version"] != 1 or {"feature_version": "v1", **snapshot["config"]} != asdict(model.config):
        raise ValueError("Incompatible checkpoint architecture")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if snapshot["optimizer_class"] != f"{type(optimizer).__module__}.{type(optimizer).__qualname__}":
        raise ValueError("Incompatible optimizer")
    model.load_state_dict(snapshot["model"], strict=True)
    if snapshot["optimizer_policy"] == "inherit":
        optimizer.load_state_dict(deepcopy(snapshot["optimizer"]))
    elif snapshot["optimizer_policy"] == "reset":
        optimizer.state.clear()
    else:
        raise ValueError("Unknown optimizer inheritance policy")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
