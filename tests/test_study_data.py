import pytest
import torch

from tensegra.data import dynamics_step, make_dynamics
from tensegra.study_data import deterministic_future, graph_quality, make_system, supply_graph


def test_sparse_degree_stays_fixed_across_sizes_and_seeds_are_local():
    before = torch.random.get_rng_state().clone()
    for n in (32, 128):
        degree = sum(int(make_system('sparse', n, s).read_graph.sum()) - n for s in range(20)) / (20 * n)
        assert abs(degree - 3) < 0.25
    assert torch.equal(before, torch.random.get_rng_state())
    graphs = [make_system('sparse', 12, s).read_graph for s in range(12)]
    assert all(not torch.equal(a, b) for i, a in enumerate(graphs) for b in graphs[i + 1:])
    assert torch.equal(graphs[0], make_system('sparse', 12, 0).read_graph)


@pytest.mark.parametrize('mechanism', ['uniform', 'signed'])
@pytest.mark.parametrize('kind,n', [('sparse', 32), ('robot', 12)])
def test_mechanism_support_and_contraction(kind, n, mechanism):
    system = make_system(kind, n, 5, mechanism)
    assert (system.weights.abs().sum(-1) <= 0.8).all()
    assert torch.equal(system.weights.ne(0) | torch.eye(n, dtype=torch.bool), system.read_graph)
    assert torch.equal(system.weights, make_system(kind, n, 5, mechanism).weights)
    if mechanism == 'signed':
        assert (system.weights < 0).any() and (system.weights > 0).any()
    if kind == 'robot':
        assert torch.equal(system.read_graph, make_dynamics(kind, n, 0).read_graph)


@pytest.mark.parametrize('mode', ['clean', 'permuted', 'drop', 'add', 'mixed'])
def test_corruption_seeded_immutable_and_exact(mode):
    graph = make_system('sparse', 12, 4).read_graph
    original = graph.clone()
    result = supply_graph(graph, mode, 0.25, 9)
    assert torch.equal(graph, original)
    assert torch.equal(result, supply_graph(graph, mode, 0.25, 9))
    assert result.diagonal().all()
    count = round((int(graph.sum()) - 12) * 0.25)
    if mode in ('drop', 'mixed'):
        assert int((graph & ~result).sum()) == count
    if mode in ('add', 'mixed'):
        assert int((result & ~graph).sum()) == count
    if mode == 'clean':
        assert torch.equal(result, graph) and result.data_ptr() != graph.data_ptr()
    if mode == 'permuted':
        assert result.sum() == graph.sum()
        assert not torch.equal(result, graph)


def test_corruption_empty_and_full_graphs_and_quality():
    empty = torch.eye(3, dtype=torch.bool)
    full = torch.ones(3, 3, dtype=torch.bool)
    assert torch.equal(supply_graph(empty, 'mixed', 1), empty)
    assert torch.equal(supply_graph(full, 'add', 1), full)
    quality = graph_quality(full, empty)
    assert quality['recall'] == 0
    assert quality['true_edges'] == 6 and quality['supplied_edges'] == 0
    assert graph_quality(empty, empty)['recall'] == 1
    assert graph_quality(empty, empty)['precision'] == 1


def test_deterministic_future_uses_last_history_and_recurses_without_noise():
    system = make_system('sparse', 5, 2)
    history = torch.arange(30.).reshape(2, 5, 3) / 30
    before = history.clone()
    expected = []
    state = history[:, :, -1]
    for _ in range(4):
        state = dynamics_step(system, state)
        expected.append(state)
    assert torch.equal(deterministic_future(system, history, 4), torch.stack(expected, 1))
    assert torch.equal(history, before)
    assert deterministic_future(system, history, 0).shape == (2, 0, 5)


@pytest.mark.parametrize('fraction', [-1, 1.1, float('nan')])
def test_bad_corruption_fraction(fraction):
    with pytest.raises(ValueError, match='fraction'):
        supply_graph(torch.eye(2, dtype=torch.bool), 'drop', fraction)


def test_boolean_degree_is_rejected():
    with pytest.raises(ValueError, match='degree'):
        make_system('sparse', 4, 0, degree=True)


@pytest.mark.parametrize('history', [None, [[[1.0]]], torch.ones(1, 4, 2, dtype=torch.int64)])
def test_deterministic_future_rejects_nonfloating_history(history):
    with pytest.raises(ValueError, match='floating tensor'):
        deterministic_future(make_system('sparse', 4, 0), history, 2)


def test_deterministic_future_adapts_weights_without_mutating_system():
    system = make_system('sparse', 4, 0)
    original = system.weights.clone()
    history = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2) / 16
    state = history[:, :, -1]
    expected = []
    for _ in range(3):
        state = 0.5 * state + 0.5 * torch.tanh(state @ original.double().T)
        expected.append(state)
    result = deterministic_future(system, history, 3)
    assert result.dtype == history.dtype and result.device == history.device
    torch.testing.assert_close(result, torch.stack(expected, 1), rtol=0, atol=0)
    assert torch.equal(system.weights, original)
    assert system.weights.dtype == torch.float32
