"""Explicit-grounding routing comparison; no learned program planning.

All nodes propagate suffix values under a public reverse relation schedule.
Only the loss/evaluation uses gold intermediate labels.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RoutingBatch:
    keys: torch.Tensor
    values: torch.Tensor
    adjacency: torch.Tensor
    relations: torch.Tensor
    starts: torch.Tensor


def generate(batch, nodes, depth, *, seed, device='cpu', key_dim=64, classes=16,
             heldout_composition=False, train=False):
    g = torch.Generator().manual_seed(seed)
    successors = torch.rand(batch, 3, nodes, generator=g).argsort(-1)
    adjacency = F.one_hot(successors, nodes).float()
    keys = F.normalize(torch.randn(batch, nodes, key_dim, generator=g), dim=-1)
    values = torch.randint(classes, (batch, nodes), generator=g)
    relations = torch.randint(3, (batch, depth), generator=g)
    if train:
        for t in range(1, depth):
            mask = (relations[:, t-1] == 2) & (relations[:, t] == 2)
            relations[mask, t] = 0
    if heldout_composition and depth >= 2:
        relations[:, :2] = 2
    starts = torch.randint(nodes, (batch,), generator=g)
    return RoutingBatch(*(x.to(device) for x in (keys, values, adjacency, relations, starts)))


def targets(batch):
    """Privileged suffix-value labels, separate from model forward."""
    current = batch.values
    out = []
    bi = torch.arange(len(current), device=current.device)
    for r in batch.relations.flip(1).unbind(1):
        current = current.gather(1, batch.adjacency[bi, r].argmax(-1))
        out.append(current)
    return torch.stack(out, 1)


def permute_nodes(batch, order):
    b, n = order.shape
    bi = torch.arange(b, device=order.device)[:, None]
    inv = order.argsort(-1)
    adj = batch.adjacency.gather(2, order[:, None, :, None].expand(-1, 3, -1, n))
    adj = adj.gather(3, order[:, None, None, :].expand(-1, 3, n, -1))
    return RoutingBatch(batch.keys[bi, order], batch.values[bi, order], adj,
                        batch.relations, inv.gather(1, batch.starts[:, None]).squeeze(1))


def corrupt(batch, kind, seed):
    """Alter public topology only; clean labels must be retained by caller."""
    g = torch.Generator(device=batch.adjacency.device).manual_seed(seed)
    a = batch.adjacency.clone()
    b, r, n, _ = a.shape
    if kind == 'wrong':  # independent random permutations preserve both degrees
        a = F.one_hot(torch.rand(b, r, n, generator=g, device=a.device).argsort(-1), n).to(a)
    elif kind == 'identity':  # conjugation without moving node memory
        order = torch.rand(b, n, generator=g, device=a.device).argsort(-1)
        a = permute_nodes(batch, order).adjacency
    elif kind == 'missing':
        drop = torch.rand(b, r, n, generator=g, device=a.device) < .25
        a[drop] = 0
    elif kind == 'spurious':
        add = torch.rand(b, r, n, generator=g, device=a.device) < .25
        j = torch.randint(n, (b, r, n), generator=g, device=a.device)
        a = torch.maximum(a, F.one_hot(j, n).to(a) * add[..., None])
    elif kind != 'clean':
        raise ValueError(kind)
    # Public fallback, shared across graph arms; corrupted semantics stay corrupted.
    empty = a.sum(-1) == 0
    a = a + empty[..., None] * torch.eye(n, device=a.device)[None, None]
    return RoutingBatch(batch.keys, batch.values, a, batch.relations, batch.starts)


class RoutingModel(nn.Module):
    def __init__(self, width=1024, key_dim=64, classes=16, heads=8, strength=4.):
        super().__init__()
        if width % heads:
            raise ValueError('width must divide heads')
        self.width, self.heads = width, heads
        self.embed = nn.Embedding(classes, width)
        self.norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.context_q = nn.Linear(key_dim, key_dim, bias=False)
        self.context_k = nn.Linear(key_dim, key_dim, bias=False)
        # Shared identity coordinate prior; transforms remain learned.
        nn.init.eye_(self.context_q.weight)
        nn.init.eye_(self.context_k.weight)
        self.context_log_scale = nn.Parameter(torch.tensor(math.log(8.)))
        self.strength = nn.Parameter(torch.full((3, heads), float(strength)))
        self.update = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.readout = nn.Linear(width, classes)

    def forward(self, batch, mode='soft', *, zero_strength=False, strength_override=None, size_adjust=False):
        if mode not in {'soft', 'hard', 'message', 'context', 'none'}:
            raise ValueError(mode)
        h = self.embed(batch.values)
        b, n, w = h.shape
        bi = torch.arange(b, device=h.device)
        logits, routes, masses, weights_by_step = [], [], [], []
        for relation in batch.relations.flip(1).unbind(1):
            a = batch.adjacency[bi, relation]
            if mode == 'message':
                weights = a / a.sum(-1, keepdim=True)
                retrieved = weights @ h
            elif mode == 'context':
                # Query every supplied address separately, then average address reads.
                # This retains distinct neighbors even under spurious-edge corruption.
                q = F.normalize(self.context_q(batch.keys), dim=-1)
                k = F.normalize(self.context_k(batch.keys), dim=-1)
                address_reads = torch.softmax(q @ k.transpose(-1, -2) * self.context_log_scale.exp().clamp(max=64), -1)
                weights = (a / a.sum(-1, keepdim=True)) @ address_reads
                retrieved = weights @ h
            else:
                z = self.norm(h)
                q = self.q(z).reshape(b, n, self.heads, w//self.heads).transpose(1, 2)
                k = self.k(z).reshape(b, n, self.heads, w//self.heads).transpose(1, 2)
                score = q @ k.transpose(-1, -2) / math.sqrt(w//self.heads)
                if mode == 'soft' and not zero_strength:
                    strength = self.strength[relation, :, None, None]
                    if strength_override is not None:
                        strength = torch.full_like(strength, float(strength_override))
                    if size_adjust:
                        strength = strength + math.log(n / 16)
                    score = score + strength * a[:, None]
                if mode == 'hard':
                    score = score.masked_fill(~a[:, None].bool(), -torch.inf)
                attention = score.softmax(-1)
                # Head-specific value partitions; no hidden exact value copy.
                v = h.reshape(b, n, self.heads, w//self.heads).transpose(1, 2)
                retrieved = (attention @ v).transpose(1, 2).reshape(b, n, w)
                weights = attention.mean(1)
            h = retrieved + .1 * self.update(retrieved)
            logits.append(self.readout(self.norm(h)))
            routes.append(weights.argmax(-1))
            masses.append((weights * a).sum(-1))
            weights_by_step.append(weights)
        return {'logits': torch.stack(logits, 1), 'routes': torch.stack(routes, 1),
                'edge_mass': torch.stack(masses, 1), 'weights': torch.stack(weights_by_step, 1)}


def metrics(output, gold, batch):
    pred = output['logits'].argmax(-1)
    correct = pred == gold
    b, d, n = correct.shape
    bi = torch.arange(b, device=pred.device)
    task = correct[:, -1].gather(1, batch.starts[:, None]).squeeze(1)
    # Layer t is relation D-1-t. Traverse the induced routing tree from final
    # start back through layers; compare exact node identities, not payloads.
    proposed, actual = batch.starts.clone(), batch.starts.clone()
    path = torch.ones(b, dtype=torch.bool, device=pred.device)
    for t in reversed(range(d)):
        proposed = output['routes'][bi, t, proposed]
        a = batch.adjacency[bi, batch.relations[:, d-1-t]]
        actual = a.argmax(-1)[bi, actual]
        path &= proposed == actual
    return {'task': task, 'all_node': correct.float().mean((1,2)),
            'suffix_value_trajectory': correct.all((1,2)), 'exact_pointer_path': path,
            'edge_mass': (output['weights'] * batch.adjacency[bi[:, None], batch.relations.flip(1)]).sum(-1).mean((1,2))}
