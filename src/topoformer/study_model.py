"""Shared predictor architecture with runtime-programmable graph conditioning."""

import math

import torch

from .attention import graph_structure
from .model import GraphPredictor


class StudyPredictor(GraphPredictor):
    """Node-count agnostic predictor with matched common parameter names.

    ``graph_input`` provides normalized neighbor histories through an extra input
    projection; attention itself is unbiased. This is a graph-conditioned input /
    message-passing control, not a serialized graph baseline. ``typed`` exposes
    positive and negative edge-weight magnitudes, and therefore receives more
    information than adjacency-only variants.
    """

    def __init__(self, history, width, heads, layers, variant="none", strength=4.0):
        if variant not in {"none", "soft", "hard", "graph_input", "learned", "typed"}:
            raise ValueError("unknown study predictor variant")
        if not math.isfinite(strength):
            raise ValueError("strength must be finite")
        super().__init__(history, width, heads, layers)
        self.variant = variant
        self.strength = float(strength)
        if variant == "graph_input":
            self.graph_input = torch.nn.Linear(history, width, bias=False)
        elif variant == "learned":
            self.alpha = torch.nn.Parameter(torch.zeros(layers, heads))
        elif variant == "typed":
            self.alpha = torch.nn.Parameter(torch.zeros(layers, heads, 2))

    @staticmethod
    def _batch_relation(relation, x, name):
        if not isinstance(relation, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        batch, nodes, _ = x.shape
        if relation.ndim == 2:
            relation = relation.unsqueeze(0)
        if relation.shape == (1, nodes, nodes):
            relation = relation.expand(batch, -1, -1)
        if relation.shape != (batch, nodes, nodes):
            raise ValueError(f"{name} must have shape [N,N], [1,N,N], or [B,N,N]")
        return relation.to(device=x.device)

    def forward(self, x, graph, weights=None):
        if x.ndim != 3 or x.shape[-1] != self.history:
            raise ValueError("x must have shape [B, N, history]")
        graph = self._batch_relation(graph, x, "graph")
        if graph.dtype != torch.bool:
            raise TypeError("graph must be boolean")
        hidden = self.input(x)
        bias, allowed = None, None
        if self.variant in {"soft", "hard"}:
            bias, allowed = graph_structure(graph, self.variant)
        elif self.variant == "graph_input":
            adjacency = graph.to(dtype=x.dtype)
            neighbors = adjacency @ x / adjacency.sum(-1, keepdim=True).clamp_min(1)
            hidden = hidden + self.graph_input(neighbors)
        elif self.variant == "typed":
            if weights is None:
                raise ValueError("typed variant requires weights")
            weights = self._batch_relation(weights, x, "weights")
            if not weights.is_floating_point() or not torch.isfinite(weights).all():
                raise ValueError("weights must be finite floating point values")
            # Channels encode magnitudes; alpha independently learns each sign's
            # attraction or repulsion. Edges outside the supplied graph stay absent.
            weights = weights.to(dtype=hidden.dtype) * graph
            relations = torch.stack((weights.clamp_min(0), (-weights).clamp_min(0)), -1)
        for layer, block in enumerate(self.blocks):
            strength = self.strength
            if self.variant == "learned":
                bias = graph[:, None].to(hidden.dtype) * self.alpha[layer][None, :, None, None]
                strength = 1.0
            elif self.variant == "typed":
                bias = torch.einsum("bijr,hr->bhij", relations, self.alpha[layer])
                strength = 1.0
            hidden = block(hidden, bias, allowed, strength)
        return self.readout(self.final_norm(hidden)).squeeze(-1)

    def structure_coefficients(self):
        """Detached layer/head(/relation) coefficients for checkpoint logging."""
        if hasattr(self, "alpha"):
            return self.alpha.detach().cpu().tolist()
        return []
