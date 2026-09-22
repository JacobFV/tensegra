from dataclasses import dataclass
import math

import torch


@dataclass
class Dynamics:
    """Synthetic dynamics with [N, N] target-source weights and support."""

    weights: torch.Tensor
    read_graph: torch.Tensor
    labels: tuple[str, ...] = ()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _robot_labels(n):
    branches = [
        ["left_shoulder", "left_hand"],
        ["right_shoulder", "right_hand"],
        ["left_hip", "left_foot"],
        ["right_hip", "right_foot"],
    ]
    joint_names = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
    for offset in range(n - 9):
        branch = offset % 4
        layer = offset // 4
        name = joint_names[branch] if layer == 0 else f"{joint_names[branch]}_{layer + 1}"
        branches[branch].insert(-1, name)
    return ("torso",), branches


def _weights_from_graph(neighbors):
    weights = torch.zeros(neighbors.shape, dtype=torch.float32)
    degree = neighbors.sum(dim=1)
    active = degree > 0
    row_mass = torch.nextafter(torch.tensor(0.8), torch.tensor(0.0))
    weights[active] = neighbors[active].to(torch.float32) * (
        row_mass / degree[active, None]
    )
    return weights


def make_dynamics(kind: str, n: int, seed: int) -> Dynamics:
    """Create sparse or labeled robot coupling with [N, N] CPU tensors."""
    _positive_integer(n, "n")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if kind == "sparse":
        neighbors = torch.rand((n, n), generator=generator) < 0.25
        neighbors.fill_diagonal_(False)
        if n > 1 and (not neighbors.any() or torch.equal(neighbors, neighbors.T)):
            order = torch.randperm(n, generator=generator)
            neighbors[order[0], order[1]] = True
            neighbors[order[1], order[0]] = False
        labels = tuple(f"node_{index}" for index in range(n))
    elif kind == "robot":
        if n < 9:
            raise ValueError("robot morphology requires at least 9 nodes")
        root, branches = _robot_labels(n)
        labels = root + tuple(label for branch in branches for label in branch)
        neighbors = torch.zeros((n, n), dtype=torch.bool)
        label_index = {label: index for index, label in enumerate(labels)}
        for branch in branches:
            chain = ["torso", *branch]
            for left, right in zip(chain, chain[1:]):
                i, j = label_index[left], label_index[right]
                neighbors[i, j] = True
                neighbors[j, i] = True
    else:
        raise ValueError("kind must be one of: sparse, robot")

    weights = _weights_from_graph(neighbors)
    read_graph = neighbors | torch.eye(n, dtype=torch.bool)
    return Dynamics(weights=weights, read_graph=read_graph, labels=labels)


def dynamics_step(system: Dynamics, state: torch.Tensor, *, noise=None) -> torch.Tensor:
    """Advance states shaped [..., N] by one synthetic nonlinear step."""
    update = 0.5 * state + 0.5 * torch.tanh(state @ system.weights.T)
    return update if noise is None else update + noise


def trajectories(
    system: Dynamics,
    *,
    count: int,
    steps: int,
    seed: int,
    noise: float = 0.01,
    initial_scale: float = 1.0,
    burn_in: int = 32,
) -> torch.Tensor:
    """Return CPU float32 observations shaped [count, steps, N]."""
    _positive_integer(count, "count")
    _positive_integer(steps, "steps")
    if not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if not isinstance(initial_scale, (int, float)) or not math.isfinite(initial_scale):
        raise ValueError("initial_scale must be finite and nonnegative")
    if initial_scale < 0:
        raise ValueError("initial_scale must be finite and nonnegative")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int):
        raise TypeError("burn_in must be an integer")
    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative")
    if system.weights.ndim != 2 or system.weights.shape[0] != system.weights.shape[1]:
        raise ValueError("system weights must have shape [N, N]")

    weights = system.weights.detach().to(device="cpu", dtype=torch.float32)
    cpu_system = Dynamics(weights, system.read_graph.detach().to(device="cpu"), system.labels)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = initial_scale * torch.randn(count, weights.shape[0], generator=generator)

    def advance(current):
        innovation = noise * torch.randn(current.shape, generator=generator)
        return dynamics_step(cpu_system, current, noise=innovation)

    for _ in range(burn_in):
        state = advance(state)

    result = torch.empty(count, steps, weights.shape[0], dtype=torch.float32)
    for index in range(steps):
        result[:, index] = state
        if index + 1 < steps:
            state = advance(state)
    return result


def windows(series: torch.Tensor, history: int):
    """Return trajectory-local x [examples, N, history] and y [examples, N]."""
    if not isinstance(series, torch.Tensor) or series.ndim != 3:
        raise ValueError("series must have shape [trajectories, steps, N]")
    if isinstance(history, bool) or not isinstance(history, int):
        raise TypeError("history must be an integer")
    if history <= 0 or history >= series.shape[1]:
        raise ValueError("history must be positive and smaller than the series length")

    inputs = []
    targets = []
    for trajectory in series:
        for target_index in range(history, series.shape[1]):
            inputs.append(trajectory[target_index - history : target_index].T)
            targets.append(trajectory[target_index])
    return torch.stack(inputs), torch.stack(targets)
