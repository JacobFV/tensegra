"""Random identity-keyed traversal tasks; gold paths are diagnostics, never inputs.

Adjacency is query/source -> key/destination. Each relation is a total function,
so every supplied relation sequence has exactly one answer, including in cycles.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def corrupt_adjacency(adjacency: Tensor, fraction: float, seed: int = 0) -> Tensor:
    """Rewire a Bernoulli fraction of functional edges, preserving one per row.

Rewired edges always change destination when there is more than one node.
Randomness is generated on CPU, independent of the model/device RNG stream.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    if adjacency.ndim != 4 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must have shape [batch, relations, nodes, nodes]")
    nodes = adjacency.shape[-1]
    if nodes < 1:
        raise ValueError("at least one node is required")
    if fraction == 0 or nodes == 1:
        return adjacency.clone()
    generator = torch.Generator().manual_seed(seed)
    destinations = adjacency.detach().cpu().argmax(-1)
    selected = torch.rand(destinations.shape, generator=generator) < fraction
    offsets = torch.randint(1, nodes, destinations.shape, generator=generator)
    destinations = torch.where(selected, (destinations + offsets) % nodes, destinations)
    return F.one_hot(destinations, nodes).to(device=adjacency.device, dtype=adjacency.dtype)


def oracle_traverse(adjacency: Tensor, start_nodes: Tensor, relations: Tensor) -> Tensor:
    """Execute functional traversal from starts and instructions; return full path.

No target, payload, or intermediate gold node is accepted by this oracle.
    """
    if adjacency.ndim != 4 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must have shape [batch, relations, nodes, nodes]")
    batch, relation_count, nodes, _ = adjacency.shape
    if start_nodes.shape != (batch,) or relations.ndim != 2 or relations.shape[0] != batch:
        raise ValueError("starts and relations must have matching batch dimensions")
    if torch.any((start_nodes < 0) | (start_nodes >= nodes)):
        raise ValueError("invalid start node")
    if torch.any((relations < 0) | (relations >= relation_count)):
        raise ValueError("invalid relation instruction")
    current = start_nodes.to(adjacency.device)
    rows = torch.arange(batch, device=adjacency.device)
    paths = [current]
    for instruction in relations.to(adjacency.device).unbind(1):
        current = adjacency[rows, instruction, current].argmax(-1)
        paths.append(current)
    return torch.stack(paths, dim=1)


def _instructions(batch: int, depth: int, count: int, composition: str,
                  generator: torch.Generator) -> Tensor:
    if composition not in {"any", "train", "heldout"}:
        raise ValueError("composition must be any, train, or heldout")
    if composition == "heldout" and (depth < 2 or count < 2):
        raise ValueError("heldout composition requires depth >= 2 and relations >= 2")
    instructions = torch.randint(count, (batch, depth), generator=generator)
    if composition == "train" and count > 1:
        # Reject the forbidden symbol at each step, producing a conditional
        # uniform next-symbol distribution, without silently replacing by zero.
        for step in range(1, depth):
            forbidden = (instructions[:, step - 1] == 0) & (instructions[:, step] == 1)
            while forbidden.any():
                instructions[forbidden, step] = torch.randint(count, (int(forbidden.sum()),), generator=generator)
                forbidden = (instructions[:, step - 1] == 0) & (instructions[:, step] == 1)
    elif composition == "heldout":
        positions = torch.randint(depth - 1, (batch,), generator=generator)
        rows = torch.arange(batch)
        instructions[rows, positions] = 0
        instructions[rows, positions + 1] = 1
    return instructions


def make_batch(batch_size: int = 32, nodes: int = 16, depth: int = 3,
               relations: int = 3, key_dim: int = 16, classes: int = 8,
               distractors: int = 4, seed: int = 0, device: str = "cpu",
               composition: str = "any", corruption: float = 0.) -> dict[str, Tensor]:
    """Create an independently shuffled batch with opaque, fresh identity keys.

Model inputs: entity_keys [B,N,D], adjacency [B,R,N,N], node_ids [B,N],
 token_keys [B,N+K,D], token_values [B,N+K], start_keys [B,D], relations [B,L].
Labels/diagnostics ONLY: targets [B], token_nodes [B,N+K] (-1 for null),
 start_nodes [B], path_nodes [B,L+1], clean_adjacency [B,R,N,N].
Node IDs are opaque integers; normalized random keys carry no payload or edges.
Targets always use the clean graph even when supplied adjacency is corrupted.
    """
    if min(batch_size, nodes, relations, key_dim, classes) < 1 or depth < 0 or distractors < 0:
        raise ValueError("positive dimensions, nonnegative depth and distractors required")
    if not 0 <= corruption <= 1:
        raise ValueError("corruption must be in [0, 1]")
    generator = torch.Generator().manual_seed(seed)
    entity_keys = F.normalize(torch.randn(batch_size, nodes, key_dim, generator=generator), dim=-1)
    # A shared random offset plus a random permutation ensures unique opaque IDs
    # within each graph, without a costly permutation of the entire ID universe.
    offsets = torch.randint(0, 2**40, (batch_size, 1), generator=generator)
    node_ids = offsets + torch.stack([torch.randperm(nodes, generator=generator) for _ in range(batch_size)])
    destinations = torch.randint(nodes, (batch_size, relations, nodes), generator=generator)
    clean = F.one_hot(destinations, nodes).float()
    values = torch.randint(classes, (batch_size, nodes + distractors), generator=generator)
    extra_keys = F.normalize(torch.randn(batch_size, distractors, key_dim, generator=generator), dim=-1)
    all_keys = torch.cat((entity_keys, extra_keys), dim=1)
    all_nodes = torch.cat((torch.arange(nodes).expand(batch_size, -1),
                           torch.full((batch_size, distractors), -1)), dim=1)
    order = torch.stack([torch.randperm(nodes + distractors, generator=generator) for _ in range(batch_size)])
    starts = torch.randint(nodes, (batch_size,), generator=generator)
    instructions = _instructions(batch_size, depth, relations, composition, generator)
    path = oracle_traverse(clean, starts, instructions)
    result = {
        "entity_keys": entity_keys,
        "adjacency": corrupt_adjacency(clean, corruption, seed + 1000003),
        "node_ids": node_ids,
        "token_keys": all_keys.gather(1, order.unsqueeze(-1).expand(-1, -1, key_dim)),
        "token_values": values.gather(1, order),
        "token_nodes": all_nodes.gather(1, order),
        "start_keys": entity_keys[torch.arange(batch_size), starts],
        "relations": instructions,
        "targets": values[torch.arange(batch_size), path[:, -1]],
        "start_nodes": starts,
        "path_nodes": path,
        "clean_adjacency": clean,
    }
    return {key: value.to(device) for key, value in result.items()}
