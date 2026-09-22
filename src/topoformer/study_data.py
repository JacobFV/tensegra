"""Process families and supplied-topology controls for the second study."""

import math

import torch

from .data import Dynamics, _positive_integer, _weights_from_graph, dynamics_step, make_dynamics


def make_system(kind, n, seed, mechanism='uniform', degree=3):
    """Create a process; sparse expected nonself indegree is size independent.

    Signed mechanisms share the same adjacency as uniform mechanisms. Their
    absolute row mass is bounded by 0.8, giving a transition Lipschitz bound
    of 0.9 in the infinity norm. Edge weights remain in ``Dynamics.weights``.
    """
    _positive_integer(n, 'n')
    if isinstance(degree, bool) or not isinstance(degree, (int, float)) or not math.isfinite(degree) or degree < 0:
        raise ValueError('degree must be finite and nonnegative')
    if mechanism not in ('uniform', 'signed'):
        raise ValueError('mechanism must be uniform or signed')
    generator = torch.Generator().manual_seed(seed)
    if kind == 'sparse':
        probability = min(degree / (n - 1), 1) if n > 1 else 0
        neighbors = torch.rand(n, n, generator=generator) < probability
        neighbors.fill_diagonal_(False)
        labels = tuple(f'node_{i}' for i in range(n))
    elif kind == 'robot':
        robot = make_dynamics(kind, n, seed)
        neighbors = robot.read_graph.clone()
        neighbors.fill_diagonal_(False)
        labels = robot.labels
    else:
        raise ValueError('kind must be sparse or robot')
    weights = _weights_from_graph(neighbors)
    if mechanism == 'signed':
        # Positive magnitudes keep support exact. Double precision normalization
        # and a small float32 margin prevent row-sum roundoff exceeding the bound.
        magnitudes = (0.1 + torch.rand(n, n, generator=generator, dtype=torch.float64)) * neighbors
        signs = 2 * torch.randint(2, (n, n), generator=generator) - 1
        weights = (0.7999999 * magnitudes / magnitudes.sum(-1, keepdim=True).clamp_min(1e-12) * signs).float()
    return Dynamics(weights, neighbors | torch.eye(n, dtype=torch.bool), labels)


def _check_graph(graph):
    if not isinstance(graph, torch.Tensor) or graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError('graph must have shape [N,N]')
    if graph.dtype != torch.bool:
        raise ValueError('graph must be boolean')


def supply_graph(graph, corruption='clean', fraction=0.0, seed=0):
    """Return seeded corruption without mutating truth; diagonal stays unchanged.

    Drop/add counts use round(fraction * original nonself edge count).
    Additions are capped by original available nonedges. Mixed corruption
    independently drops true edges and adds original nonedges, so it cannot
    accidentally restore a removed edge. Counts refer to directed read edges.
    """
    _check_graph(graph)
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be finite and between 0 and 1')
    if corruption not in ('clean', 'permuted', 'drop', 'add', 'mixed'):
        raise ValueError('unknown corruption')
    original = graph.detach().cpu()
    result = original.clone()
    generator = torch.Generator().manual_seed(seed)
    n = graph.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool)
    if corruption == 'permuted':
        order = torch.randperm(n, generator=generator)
        result = original[order][:, order].clone()
        result.diagonal().copy_(original.diagonal())
    else:
        edges = (original & off_diagonal).nonzero()
        count = round(fraction * len(edges))
        if corruption in ('drop', 'mixed'):
            selected = edges[torch.randperm(len(edges), generator=generator)[:count]]
            result[selected[:, 0], selected[:, 1]] = False
        if corruption in ('add', 'mixed'):
            absent = (~original & off_diagonal).nonzero()
            selected = absent[torch.randperm(len(absent), generator=generator)[:count]]
            result[selected[:, 0], selected[:, 1]] = True
    return result.to(graph.device)


def graph_quality(truth, supplied):
    """Directed nonself precision/recall, with vacuous empty denominators = 1."""
    _check_graph(truth)
    _check_graph(supplied)
    if truth.shape != supplied.shape:
        raise ValueError('graphs must have equal shapes')
    off_diagonal = ~torch.eye(truth.shape[0], dtype=torch.bool)
    true_edges = truth.detach().cpu() & off_diagonal
    supplied_edges = supplied.detach().cpu() & off_diagonal
    correct = int((true_edges & supplied_edges).sum())
    total_true, total_supplied = int(true_edges.sum()), int(supplied_edges.sum())
    return {
        'true_edges': total_true,
        'supplied_edges': total_supplied,
        'correct_edges': correct,
        'dropped_edges': total_true - correct,
        'added_edges': total_supplied - correct,
        'precision': correct / total_supplied if total_supplied else 1.0,
        'recall': correct / total_true if total_true else 1.0,
    }


def deterministic_future(system, history, horizon):
    """Noise-free recursive reference from observed history [B,N,H].

    Returns [B,horizon,N]. This deterministic transition rollout is not the
    multistep conditional expectation of the nonlinear stochastic process.
    Weights adapt locally to the floating history's dtype/device.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError('horizon must be a nonnegative integer')
    if not isinstance(history, torch.Tensor) or not history.is_floating_point():
        raise ValueError('history must be a floating tensor')
    if history.ndim != 3 or history.shape[1] != system.weights.shape[0] or history.shape[2] < 1:
        raise ValueError('history must have shape [B,N,H] with H >= 1')
    if horizon == 0:
        return history.new_empty(history.shape[0], 0, history.shape[1])
    local_system = Dynamics(
        system.weights.to(device=history.device, dtype=history.dtype),
        system.read_graph, system.labels,
    )
    state = history[:, :, -1]
    future = []
    for _ in range(horizon):
        state = dynamics_step(local_system, state)
        future.append(state)
    return torch.stack(future, dim=1)
