"""Privileged, nonlearned exact-identity control using structural attention.

Unlike direct adjacency indexing, every transition passes through shuffled token
memory. This verifies the grounding/bias/attention/rebinding pipeline itself.
"""
from collections.abc import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .attention import structural_attention
from .grounding import induce_bias


def _bind(keys: Tensor, entities: Tensor) -> Tensor:
    matches = (keys[:, :, None, :] == entities[:, None, :, :]).all(-1)
    if (matches.sum(-1) > 1).any():
        raise ValueError('exact identity control requires unique entity keys')
    return torch.cat((matches, ~matches.any(-1, keepdim=True)), dim=-1).to(entities.dtype)


@torch.no_grad()
def exact_attention_traverse(batch: Mapping[str, Tensor], classes: int | None = None) -> dict[str, Tensor]:
    """Execute instructed relations through exact one-hot grounded hard attention.

    Reads only entity_keys, adjacency, token_keys, token_values, start_keys and
    relations. Gold identities/paths/targets are neither read nor required. This
    oracle knows the exact key equality rule and uses raw identity/payload values;
    it is a mechanism check, not evidence of learning or latent generalization.
    """
    entities, adjacency = batch['entity_keys'], batch['adjacency']
    keys, values = batch['token_keys'], batch['token_values']
    current, relations = batch['start_keys'][:, None, :], batch['relations']
    b, n, dim = entities.shape
    if adjacency.ndim != 4 or adjacency.shape[0] != b or adjacency.shape[-2:] != (n, n):
        raise ValueError('adjacency shape must match entity graph')
    if not (((adjacency == 0) | (adjacency == 1)).all() and (adjacency.sum(-1) == 1).all()):
        raise ValueError('exact routing requires binary functional adjacency')
    if keys.ndim != 3 or keys.shape[0] != b or keys.shape[-1] != dim or values.shape != keys.shape[:2]:
        raise ValueError('token key/value shapes must match')
    if relations.ndim != 2 or relations.shape[0] != b or ((relations < 0) | (relations >= adjacency.shape[1])).any():
        raise ValueError('invalid relation instructions')
    pk = _bind(keys, entities)
    if not (pk[..., :n].sum(1) == 1).all():
        raise ValueError('every entity must have exactly one token in memory')
    pq = _bind(current, entities)
    if pq[..., -1].any():
        raise ValueError('start identity must be bound to a graph entity')
    count = int(values.max().item()) + 1 if classes is None else classes
    payload = F.one_hot(values, count).to(keys.dtype)
    memory = torch.cat((keys, payload), dim=-1)[:, None]
    # Content scores are zero: exact topology alone chooses a unique token.
    q = torch.zeros(b, 1, 1, 1, device=keys.device, dtype=keys.dtype)
    k = torch.zeros(b, 1, keys.shape[1], 1, device=keys.device, dtype=keys.dtype)
    rows = torch.arange(b, device=entities.device)
    path = [pq[:, 0, :n].argmax(-1)]
    grounding, attention = [], []
    # Depth zero retrieves the start payload through exact identity attention.
    output, _ = structural_attention(q, k, memory, allowed=(pq[:, None] @ pk[:, None].transpose(-2, -1)).bool())
    for instruction in relations.unbind(1):
        pq = _bind(current, entities)
        grounding.append(pq[:, 0])
        bias = induce_bias(pq, adjacency, pk)[rows, instruction][:, None]
        output, weights = structural_attention(q, k, memory, allowed=bias.bool())
        current = output[:, 0, :, :dim]
        rebound = _bind(current, entities)
        if rebound[..., -1].any():
            raise RuntimeError('hard attention failed exact identity rebinding')
        path.append(rebound[:, 0, :n].argmax(-1))
        attention.append(weights[:, 0, 0])
    return {
        'predictions': output[:, 0, 0, dim:].argmax(-1),
        'path_nodes': torch.stack(path, dim=1),
        'pq': torch.stack(grounding, dim=1) if grounding else entities.new_empty(b, 0, n + 1),
        'pk': pk,
        'attention': torch.stack(attention, dim=1) if attention else entities.new_empty(b, 0, keys.shape[1]),
    }
