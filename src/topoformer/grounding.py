"""Differentiable, role-specific latent bindings and induced graph geometry."""
import math

import torch
from torch import Tensor, nn

from .runtime_graph import RuntimeGraph


class SoftGrounding(nn.Module):
    """Return distributions over graph slots followed by one null/unbound slot.

    Each role has independent latent and entity projections. A learned scalar
    null logit lets unmatched content remain unbound; null contributes no edges.
    Temperature is positive, optionally learned in log space. Call anew on each
    layer's current residual states; this module intentionally caches nothing.
    """

    def __init__(self, latent_dim: int, entity_dim: int, grounding_dim: int | None = None,
                 temperature: float = 1., learned_temperature: bool = False):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        dim = entity_dim if grounding_dim is None else grounding_dim
        if min(latent_dim, entity_dim, dim) < 1:
            raise ValueError('projection dimensions must be positive')
        self.query_projection = nn.Linear(latent_dim, dim, bias=False)
        self.key_projection = nn.Linear(latent_dim, dim, bias=False)
        self.query_entity_projection = nn.Linear(entity_dim, dim, bias=False)
        self.key_entity_projection = nn.Linear(entity_dim, dim, bias=False)
        self.query_null_logit = nn.Parameter(torch.zeros(()))
        self.key_null_logit = nn.Parameter(torch.zeros(()))
        log_temperature = torch.tensor(math.log(temperature))
        if learned_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer('log_temperature', log_temperature)

    @property
    def temperature(self) -> Tensor:
        # Numerical limits avoid zero/infinite temperatures during optimization.
        return self.log_temperature.clamp(-12., 12.).exp()

    def _ground(self, latents: Tensor, graph: RuntimeGraph, role: str) -> Tensor:
        projection = getattr(self, role + '_projection')
        entity_projection = getattr(self, role + '_entity_projection')
        if latents.ndim != 3 or latents.shape[0] != graph.node_ids.shape[0] or latents.shape[-1] != projection.in_features:
            raise ValueError('latent shape must be [B,T,latent_dim] matching graph batch')
        if graph.embeddings.shape[-1] != entity_projection.in_features:
            raise ValueError('graph embedding shape does not match entity_dim')
        if not torch.isfinite(latents).all():
            raise ValueError('grounding latents must be finite')
        q, e = projection(latents), entity_projection(graph.embeddings)
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        scores = q.to(dtype) @ e.to(dtype).transpose(-2, -1) / math.sqrt(q.shape[-1])
        null = getattr(self, role + '_null_logit').to(dtype).expand(*scores.shape[:-1], 1)
        scores = torch.cat((scores, null), dim=-1) / self.temperature.to(dtype)
        allowed = torch.cat((graph.node_mask, torch.ones_like(graph.node_ids[:, :1], dtype=torch.bool)), dim=-1)
        # Empty graphs still need the single null slot.
        if graph.node_ids.shape[1] == 0:
            allowed = torch.ones(graph.node_ids.shape[0], 1, device=scores.device, dtype=torch.bool)
        return scores.masked_fill(~allowed[:, None], -torch.inf).softmax(-1)

    def query(self, latents: Tensor, graph: RuntimeGraph) -> Tensor:
        return self._ground(latents, graph, 'query')

    def key(self, latents: Tensor, graph: RuntimeGraph) -> Tensor:
        return self._ground(latents, graph, 'key')

    def forward(self, query_latents: Tensor, key_latents: Tensor,
                graph: RuntimeGraph) -> tuple[Tensor, Tensor]:
        return self.query(query_latents, graph), self.key(key_latents, graph)


def induce_bias(pq: Tensor, adjacency: Tensor, pk: Tensor) -> Tensor:
    """Compute Pq A_r Pk^T; last probability column is an edge-free null.

    Accepts either N real columns or N+1 including null, independently for
    query/key. Preserves gradients through probabilities and real edge weights.
    """
    if pq.ndim != 3 or pk.ndim != 3 or adjacency.ndim != 4:
        raise ValueError('bias tensor shape must be P[B,T,N(+1)], A[B,R,N,N]')
    b, _, n, m = adjacency.shape
    if m != n or pq.shape[0] != b or pk.shape[0] != b or pq.shape[-1] not in (n, n + 1) or pk.shape[-1] not in (n, n + 1):
        raise ValueError('grounding and adjacency shape mismatch')
    if not pq.is_floating_point() or not pk.is_floating_point():
        raise TypeError('grounding probabilities must have floating dtype')
    if any(t.device != pq.device for t in (pk, adjacency)):
        raise ValueError('grounding and adjacency must share a device')
    if not all(torch.isfinite(t).all() for t in (pq, pk, adjacency)):
        raise ValueError('grounding and adjacency must be finite')
    dtype = torch.promote_types(pq.dtype, pk.dtype)
    if adjacency.is_floating_point():
        dtype = torch.promote_types(dtype, adjacency.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    q, k, a = pq[..., :n].to(dtype), pk[..., :n].to(dtype), adjacency.to(dtype)
    return (q[:, None] @ a) @ k[:, None].transpose(-2, -1)
