import pytest
import torch

from tensegra.data import Dynamics, dynamics_step, make_dynamics, trajectories, windows
from tensegra.graphs import corrupt_graph


def test_sparse_dynamics_are_reproducible_and_seeded():
    first = make_dynamics("sparse", 12, seed=7)
    again = make_dynamics("sparse", 12, seed=7)
    other = make_dynamics("sparse", 12, seed=8)

    assert torch.equal(first.weights, again.weights)
    assert torch.equal(first.read_graph, again.read_graph)
    assert not torch.equal(first.read_graph, other.read_graph)


def test_generators_do_not_mutate_global_rng_state():
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()
    system = make_dynamics("sparse", 8, seed=3)
    trajectories(system, count=2, steps=4, seed=4)
    corrupt_graph(system.read_graph, mode="permuted", seed=5)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_sparse_support_is_exact_asymmetric_and_uniform_by_row():
    system = make_dynamics("sparse", 12, seed=2)
    identity = torch.eye(12, dtype=torch.bool)

    assert torch.equal(system.read_graph, system.weights.ne(0) | identity)
    assert not torch.equal(system.read_graph, system.read_graph.T)
    assert torch.all(system.weights >= 0)
    neighbor_rows = system.weights.ne(0).any(dim=1)
    torch.testing.assert_close(
        system.weights.sum(dim=1)[neighbor_rows],
        torch.full((int(neighbor_rows.sum()),), 0.8),
    )
    assert system.weights.abs().sum(dim=1).max() <= 0.8


def test_dynamics_step_uses_only_declared_parents_and_self():
    weights = torch.tensor(
        [[0.0, 0.8, 0.0], [0.0, 0.0, 0.0], [0.4, 0.4, 0.0]]
    )
    graph = weights.ne(0) | torch.eye(3, dtype=torch.bool)
    system = Dynamics(weights, graph)
    state = torch.tensor([[0.2, -0.3, 0.7]])
    changed_non_parent = state.clone()
    changed_non_parent[0, 2] += 10

    actual = dynamics_step(system, state)
    unchanged = dynamics_step(system, changed_non_parent)
    expected = 0.5 * state + 0.5 * torch.tanh(state @ weights.T)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0, 0], unchanged[0, 0])


def test_isolated_node_has_only_self_support_and_self_recurrence():
    system = Dynamics(
        torch.zeros(2, 2),
        torch.eye(2, dtype=torch.bool),
    )
    state = torch.tensor([[2.0, -4.0]])

    torch.testing.assert_close(dynamics_step(system, state), 0.5 * state)


def test_robot_is_a_connected_labeled_undirected_tree():
    system = make_dynamics("robot", 13, seed=0)
    graph = system.read_graph & ~torch.eye(13, dtype=torch.bool)
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in graph[node].nonzero().flatten().tolist():
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)

    assert graph.equal(graph.T)
    assert int(graph.sum()) == 2 * (13 - 1)
    assert reached == set(range(13))
    assert len(system.labels) == 13
    assert len(set(system.labels)) == 13
    for label in ("torso", "left_hand", "right_hand", "left_foot", "right_foot"):
        assert label in system.labels


def test_robot_requires_nine_nodes_and_unknown_kinds_are_rejected():
    with pytest.raises(ValueError, match="at least 9"):
        make_dynamics("robot", 8, seed=0)
    with pytest.raises(ValueError, match="kind"):
        make_dynamics("mesh", 12, seed=0)


def test_trajectories_are_reproducible_finite_bounded_and_disjoint():
    system = make_dynamics("sparse", 10, seed=11)
    first = trajectories(system, count=3, steps=20, seed=13)
    again = trajectories(system, count=3, steps=20, seed=13)
    other_seed = trajectories(system, count=3, steps=20, seed=14)

    assert first.shape == (3, 20, 10)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, again)
    assert torch.isfinite(first).all()
    assert first.abs().max() < 10
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first, other_seed)


def test_initial_scale_is_visible_when_burn_in_is_disabled():
    system = make_dynamics("sparse", 6, seed=1)
    base = trajectories(
        system, count=2, steps=3, seed=9, noise=0, initial_scale=1, burn_in=0
    )
    shifted = trajectories(
        system, count=2, steps=3, seed=9, noise=0, initial_scale=2, burn_in=0
    )

    torch.testing.assert_close(shifted[:, 0], 2 * base[:, 0])
    assert not torch.equal(shifted, base)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0, "steps": 2}, "count"),
        ({"count": 1, "steps": 0}, "steps"),
        ({"count": 1, "steps": 2, "noise": -0.1}, "noise"),
        ({"count": 1, "steps": 2, "initial_scale": -1}, "initial_scale"),
        ({"count": 1, "steps": 2, "initial_scale": float("nan")}, "initial_scale"),
        ({"count": 1, "steps": 2, "burn_in": -1}, "burn_in"),
        ({"count": 1, "steps": 2, "burn_in": 1.5}, "burn_in"),
    ],
)
def test_trajectories_reject_invalid_controls(kwargs, message):
    system = make_dynamics("sparse", 4, seed=0)
    with pytest.raises((TypeError, ValueError), match=message):
        trajectories(system, seed=0, **kwargs)


def test_windows_preserve_trajectory_boundaries_and_align_targets():
    series = torch.arange(20.0).reshape(2, 5, 2)
    x, y = windows(series, history=2)

    assert x.shape == (6, 2, 2)
    assert y.shape == (6, 2)
    torch.testing.assert_close(x[0], series[0, :2].T)
    torch.testing.assert_close(y[0], series[0, 2])
    torch.testing.assert_close(x[3], series[1, :2].T)
    torch.testing.assert_close(y[3], series[1, 2])


@pytest.mark.parametrize("history", [0, -1, 5, 6, 1.5])
def test_windows_reject_invalid_history(history):
    series = torch.zeros(2, 5, 3)
    with pytest.raises((TypeError, ValueError), match="history"):
        windows(series, history)


@pytest.mark.parametrize("mode", ["reversed", "permuted", "dropped"])
def test_graph_corruptions_are_seeded_boolean_and_shape_preserving(mode):
    graph = make_dynamics("sparse", 12, seed=4).read_graph
    first = corrupt_graph(graph, mode=mode, seed=17)
    again = corrupt_graph(graph, mode=mode, seed=17)

    assert first.dtype == torch.bool
    assert first.shape == graph.shape
    assert torch.equal(first, again)
    assert torch.equal(first.diagonal(), graph.diagonal())


def test_graph_corruption_semantics_are_explicit():
    graph = torch.tensor(
        [[True, True, False], [False, True, True], [False, False, True]]
    )
    reversed_graph = corrupt_graph(graph, mode="reversed", seed=0)
    permuted = corrupt_graph(graph, mode="permuted", seed=2)
    dropped = corrupt_graph(graph, mode="dropped", seed=2)

    assert torch.equal(reversed_graph, graph.T)
    assert torch.equal(permuted.sum(dim=0).sort().values, graph.sum(dim=0).sort().values)
    assert torch.equal(permuted.sum(dim=1).sort().values, graph.sum(dim=1).sort().values)
    assert not (dropped & ~graph).any()


def test_robot_reversal_can_be_a_redundant_control():
    graph = make_dynamics("robot", 9, seed=0).read_graph
    with pytest.warns(UserWarning, match="redundant"):
        reversed_graph = corrupt_graph(graph, mode="reversed", seed=8)
    assert torch.equal(reversed_graph, graph)
