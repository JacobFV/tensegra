"""Relation-instructed recurrent attention over immutable shuffled memory.

The identity-coordinate initialization is an explicit architectural prior. The
relation schedule controls computation length; neither intermediate entities nor
answer labels enter this module.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import structural_attention
from .grounding import SoftGrounding, induce_bias
from .runtime_graph import RuntimeGraph


class TraversalTransformer(nn.Module):
    MODES = {"none", "soft", "known", "frozen", "permuted", "hard", "graph_input"}

    def __init__(self, key_dim=16, width=32, heads=2, classes=8, relations=3,
                 temperature=0.05, learned_temperature=False, projection_period=1,
                 shared_strength=False, typed=True, strength=8.0, identity_init=True,
                 content_identity_bias=0.0):
        super().__init__()
        if min(key_dim, width, heads, classes, relations, projection_period) < 1 or width < key_dim + classes or width % heads:
            raise ValueError("width must fit keys/classes and divide into positive heads")
        if not math.isfinite(content_identity_bias):
            raise ValueError("content_identity_bias must be finite")
        self.content_identity_bias = float(content_identity_bias)
        self.key_dim, self.width, self.heads = key_dim, width, heads
        self.classes, self.relations = classes, relations
        self.typed, self.projection_period = typed, projection_period
        self.grounders = nn.ModuleList(
            SoftGrounding(width, key_dim, key_dim, temperature=temperature,
                          learned_temperature=learned_temperature)
            for _ in range(projection_period)
        )
        if identity_init:
            for grounder in self.grounders:
                with torch.no_grad():
                    for projection in (grounder.query_projection, grounder.key_projection):
                        projection.weight.zero_()
                        projection.weight[:, :key_dim].copy_(torch.eye(key_dim) * key_dim ** 0.5)
                        if projection.bias is not None:
                            projection.bias.zero_()
                    for projection in (grounder.query_entity_projection, grounder.key_entity_projection):
                        projection.weight.copy_(torch.eye(key_dim))
                        if projection.bias is not None:
                            projection.bias.zero_()
                    grounder.query_null_logit.fill_(0.65)
                    grounder.key_null_logit.fill_(0.65)
        self.register_buffer("known_temperature", torch.tensor(float(temperature)))
        self.strengths = nn.Parameter(torch.full((projection_period, 1 if shared_strength else heads,
                                                relations if typed else 1), float(strength)))
        self.norm = nn.LayerNorm(width)
        self.instruction = nn.Embedding(relations, width)
        nn.init.zeros_(self.instruction.weight)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))
        self.update_gate = nn.Parameter(torch.tensor(2.0))
        self.readout = nn.Linear(width, classes)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _known(self, state, entities):
        scores = F.normalize(state[..., :self.key_dim], dim=-1) @ F.normalize(entities, dim=-1).transpose(-1, -2)
        # A fixed cosine threshold gives unknown identities a genuine null option.
        null = torch.full_like(scores[..., :1], 0.65)
        return torch.softmax(torch.cat((scores, null), -1) / self.known_temperature, -1)

    def _heads(self, tensor):
        return tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads,
                              self.width // self.heads).transpose(1, 2)

    def forward(self, batch, *, mode="soft", return_diagnostics=False):
        if mode not in self.MODES:
            raise ValueError(f"unknown traversal mode: {mode}")
        entities, adjacency = batch["entity_keys"], batch["adjacency"]
        tokens, values = batch["token_keys"], batch["token_values"]
        instructions = batch["relations"]
        if instructions.ndim != 2 or instructions.shape[1] < 1:
            raise ValueError("relations must have positive path length [B,depth]")
        if adjacency.shape[1] != self.relations:
            raise ValueError("relation count differs from model")
        b, n, _ = entities.shape
        graph = RuntimeGraph(batch["node_ids"], entities, adjacency)
        memory = F.pad(torch.cat((tokens, F.one_hot(values, self.classes).to(tokens)), -1),
                       (0, self.width - self.key_dim - self.classes))
        state = F.pad(batch["start_keys"], (0, self.width - self.key_dim)).unsqueeze(1)
        memory_keys = self._heads(self.key(self.norm(memory)))
        frozen = None
        diagnostics = []
        for step in range(instructions.shape[1]):
            slot = step % self.projection_period
            relation = instructions[:, step]
            selected = adjacency[torch.arange(b, device=adjacency.device), relation]
            if not self.typed:
                selected = adjacency.amax(dim=1)
            fast_none = mode == "none" and not return_diagnostics
            if fast_none:
                pq = pk = None
            elif mode in {"known", "graph_input"}:
                pq, pk = self._known(state, entities), self._known(memory, entities)
            elif mode == "frozen" and frozen is not None:
                pq, pk = frozen
            else:
                pq, pk = self.grounders[slot](state, memory, graph)
                if mode == "frozen":
                    frozen = (pq, pk)
            if mode == "permuted":
                pq = torch.cat((pq[..., :-1].roll(1, -1), pq[..., -1:]), -1)
            route = None if fast_none else induce_bias(pq, selected[:, None], pk)[:, 0]
            strengths = self.strengths[slot].expand(self.heads, -1)
            strengths = strengths[:, relation if self.typed else torch.zeros_like(relation)].T
            bias = None if fast_none else route[:, None] * strengths[:, :, None, None]
            allowed = None
            current_memory = memory
            if mode == "hard":
                oq = F.one_hot(pq.argmax(-1), n + 1).to(pq)
                ok = F.one_hot(pk.argmax(-1), n + 1).to(pk)
                legal = induce_bias(oq, selected[:, None], ok)[:, 0] > 0
                # No mapped legal destination => ordinary attention, never NaN rows.
                legal = legal | ~legal.any(-1, keepdim=True)
                allowed = legal[:, None]
                bias = None
            elif mode == "graph_input":
                token_edges = induce_bias(pk, selected[:, None], pk)[:, 0]
                current_memory = token_edges @ memory / token_edges.sum(-1, keepdim=True).clamp_min(1e-8)
                bias = None
            elif mode == "none":
                bias = None
            content_bias = None
            if self.content_identity_bias:
                similarity = F.normalize(state[..., :self.key_dim], dim=-1) @ F.normalize(tokens, dim=-1).transpose(-1, -2)
                content_bias = self.content_identity_bias * similarity[:, None]
                bias = content_bias if bias is None else bias + content_bias
            attended, attention = structural_attention(
                self._heads(self.query(self.norm(state + self.instruction(relation)[:, None]))), memory_keys,
                self._heads(current_memory + self.value(current_memory)),
                bias=bias, strength=1.0, allowed=allowed)
            retrieved = attended.transpose(1, 2).reshape(b, 1, self.width)
            state = state + self.update_gate.sigmoid() * (retrieved - state)
            state = state + self.mlp(state)
            if return_diagnostics:
                pq_after = (self._known(state, entities) if mode in {"known", "graph_input"}
                            else self.grounders[slot].query(state, graph))
                if mode == "permuted":
                    pq_after = torch.cat((pq_after[..., :-1].roll(1, -1), pq_after[..., -1:]), -1)
                diagnostics.append({"pq": pq, "pq_after": pq_after, "pk": pk, "attention": attention,
                                    "bias": route, "content_bias": content_bias,
                                    "strengths": strengths, "state": state})
        logits = self.readout(state[:, 0])
        return (logits, diagnostics) if return_diagnostics else logits

    def structure_coefficients(self):
        return self.strengths.detach().cpu().tolist()
