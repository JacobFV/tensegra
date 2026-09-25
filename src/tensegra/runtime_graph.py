"""Batched runtime entities and directed, typed relationships.

Graph slots are storage positions; ``node_ids`` provide stable external identity.
Null is deliberately absent here: grounding appends a relation-free null slot.
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RuntimeGraph:
    node_ids: Tensor  # [batch, nodes], integer identity, unique among valid slots
    embeddings: Tensor  # [batch, nodes, entity_dim], no target information
    adjacency: Tensor  # [batch, relations, source, destination]
    valid: Tensor | None = None  # [batch, nodes]; padding has no incident edges

    def __post_init__(self):
        if self.node_ids.ndim != 2 or self.embeddings.ndim != 3 or self.adjacency.ndim != 4:
            raise ValueError('runtime graph tensor shape must be [B,N], [B,N,D], [B,R,N,N]')
        b, n = self.node_ids.shape
        if self.embeddings.shape[:2] != (b, n) or self.adjacency.shape[0] != b or self.adjacency.shape[2:] != (n, n):
            raise ValueError('runtime graph tensor shape mismatch')
        if b == 0 or self.embeddings.shape[-1] == 0 or self.adjacency.shape[1] == 0:
            raise ValueError('runtime graph needs nonempty batch, embedding and relation dimensions')
        if self.node_ids.dtype != torch.long:
            raise TypeError('node_ids must have long dtype')
        if not self.embeddings.is_floating_point():
            raise TypeError('embeddings must have floating dtype')
        if not (self.adjacency.is_floating_point() or self.adjacency.dtype == torch.bool):
            raise TypeError('adjacency must have floating or boolean dtype')
        tensors = [self.node_ids, self.embeddings, self.adjacency]
        if self.valid is not None:
            if self.valid.shape != (b, n):
                raise ValueError('valid mask shape must be [B,N]')
            if self.valid.dtype != torch.bool:
                raise TypeError('valid mask must be boolean')
            tensors.append(self.valid)
        if any(t.device != self.embeddings.device for t in tensors):
            raise ValueError('runtime graph tensors must share a device')
        if not torch.isfinite(self.embeddings).all() or not torch.isfinite(self.adjacency).all():
            raise ValueError('runtime graph embeddings and adjacency must be finite')
        mask = self.node_mask
        for ids, keep in zip(self.node_ids, mask):
            if ids[keep].unique().numel() != keep.sum().item():
                raise ValueError('valid node ids must be unique within each graph')
        incident = mask[:, None, :, None] & mask[:, None, None, :]
        if self.adjacency.masked_select(~incident).count_nonzero():
            raise ValueError('padding nodes must have zero incident adjacency')

    @property
    def node_mask(self) -> Tensor:
        return torch.ones_like(self.node_ids, dtype=torch.bool) if self.valid is None else self.valid

    def to(self, device: torch.device | str) -> 'RuntimeGraph':
        return RuntimeGraph(self.node_ids.to(device), self.embeddings.to(device),
                            self.adjacency.to(device),
                            None if self.valid is None else self.valid.to(device))
