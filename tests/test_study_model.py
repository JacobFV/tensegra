import pytest
import torch

from topoformer.study_model import StudyPredictor


def paired(variant):
    torch.manual_seed(19)
    base = StudyPredictor(3, 8, 2, 2)
    other = StudyPredictor(3, 8, 2, 2, variant=variant)
    other.load_state_dict(base.state_dict(), strict=False)
    return base, other


def inputs():
    torch.manual_seed(2)
    x = torch.randn(2, 4, 3)
    graph = torch.eye(4, dtype=torch.bool)
    graph[0, 1] = True
    graph[1, 2] = True
    return x, graph


@pytest.mark.parametrize('variant', ['learned', 'typed'])
def test_zero_coefficients_exactly_match_unbiased_and_receive_gradients(variant):
    base, model = paired(variant)
    x, graph = inputs()
    weights = graph.float()
    weights[0, 1] = -0.3
    assert torch.count_nonzero(model.alpha) == 0
    assert torch.equal(base(x, graph), model(x, graph, weights))
    model(x, graph, weights).square().sum().backward()
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad).all()
    assert (model.alpha.grad.abs().sum(dim=(0, 1)) > 0).all()
    assert model.structure_coefficients() == model.alpha.detach().tolist()


@pytest.mark.parametrize('variant', ['none', 'soft', 'hard', 'graph_input', 'learned', 'typed'])
def test_batched_graphs_and_permutation_equivariance(variant):
    _, model = paired(variant)
    x, graph = inputs()
    graphs = torch.stack([graph, graph.T])
    weights = graphs.float()
    if hasattr(model, 'alpha'):
        with torch.no_grad():
            model.alpha.fill_(1.5)
    result = model(x, graphs, weights)
    separate = torch.cat([model(x[i:i+1], graphs[i], weights[i]) for i in range(2)])
    torch.testing.assert_close(result, separate)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(x[:, permutation], graphs[:, permutation][:, :, permutation],
                     weights[:, permutation][:, :, permutation])
    torch.testing.assert_close(permuted, result[:, permutation])


def test_graph_input_has_graph_influence_but_unbiased_attention():
    base, model = paired('graph_input')
    x, graph = inputs()
    seen = []
    handles = [block.register_forward_pre_hook(lambda _, args: seen.append(args[1:]))
               for block in model.blocks]
    first = model(x, graph)
    second = model(x, torch.eye(4, dtype=torch.bool))
    assert not torch.allclose(first, second)
    assert all(bias is None and mask is None for bias, mask, strength in seen)
    for handle in handles:
        handle.remove()
    assert set(base.state_dict()).issubset(model.state_dict())


def test_typed_channels_use_signed_weight_magnitudes_and_supplied_edges():
    _, model = paired('typed')
    x, graph = inputs()
    weights = graph.float() * 0.2
    weights[0, 1] = -0.7
    with torch.no_grad():
        model.alpha[..., 0].fill_(2)
        model.alpha[..., 1].fill_(-1)
    seen = []
    handle = model.blocks[0].register_forward_pre_hook(lambda _, args: seen.append(args[1]))
    model(x, graph, weights)
    assert model.alpha.shape == (2, 2, 2)
    torch.testing.assert_close(seen[0][0, :, 0, 1], torch.full((2,), -0.7))
    torch.testing.assert_close(seen[0][0, :, 0, 0], torch.full((2,), 0.4))
    changed = weights.clone()
    changed[3, 0] = 999
    torch.testing.assert_close(model(x, graph, changed), model(x, graph, weights))
    handle.remove()


def test_invalid_inputs_and_empty_graph_input_are_handled():
    x, graph = inputs()
    with pytest.raises(ValueError):
        StudyPredictor(3, 8, 2, 2, variant='typo')
    with pytest.raises(ValueError):
        StudyPredictor(3, 8, 2, 2, strength=float('nan'))
    model = StudyPredictor(3, 8, 2, 2, variant='typed')
    with pytest.raises(ValueError, match='weights'):
        model(x, graph)
    with pytest.raises(ValueError, match='weights'):
        model(x, graph, torch.ones(3, 3))
    with pytest.raises(TypeError, match='boolean'):
        model(x, graph.float(), graph.float())
    graph_input = StudyPredictor(3, 8, 2, 2, variant='graph_input')
    assert torch.isfinite(graph_input(x, torch.zeros_like(graph))).all()


@pytest.mark.parametrize('variant', ['none', 'soft', 'hard'])
def test_fixed_variants_match_pilot_and_singleton_batch_graphs(variant):
    from topoformer.model import GraphPredictor

    torch.manual_seed(7)
    pilot = GraphPredictor(3, 8, 2, 2)
    model = StudyPredictor(3, 8, 2, 2, variant=variant, strength=4)
    model.load_state_dict(pilot.state_dict())
    x, graph = inputs()
    torch.testing.assert_close(model(x, graph[None]),
                               pilot(x, graph, mode=variant, strength=4), rtol=0, atol=0)


def test_node_identity_pairs_zero_alpha_and_receives_gradients():
    torch.manual_seed(21)
    base = StudyPredictor(3, 8, 2, 2, node_count=4)
    model = StudyPredictor(3, 8, 2, 2, variant='learned', node_count=4)
    model.load_state_dict(base.state_dict(), strict=False)
    x, graph = inputs()
    assert base.node_identity.shape == (4, 8)
    assert torch.equal(base(x, graph), model(x, graph))
    model(x, graph).square().sum().backward()
    assert model.node_identity.grad is not None
    assert torch.isfinite(model.node_identity.grad).all()
    assert model.node_identity.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='node_count'):
        model(x[:, :3], graph[:3, :3])


def test_node_identity_is_opt_in_and_validated():
    from topoformer.model import GraphPredictor

    default = StudyPredictor(3, 8, 2, 2)
    assert set(default.state_dict()) == set(GraphPredictor(3, 8, 2, 2).state_dict())
    assert not hasattr(default, 'node_identity')
    for invalid in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match='node_count'):
            StudyPredictor(3, 8, 2, 2, node_count=invalid)


def test_node_identity_equivariance_requires_joint_table_permutation():
    import copy

    model = StudyPredictor(3, 8, 2, 2, variant='soft', node_count=4)
    x, graph = inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = copy.deepcopy(model)
    with torch.no_grad():
        permuted.node_identity.copy_(model.node_identity[permutation])
    torch.testing.assert_close(
        permuted(x[:, permutation], graph[permutation][:, permutation]),
        model(x, graph)[:, permutation],
    )
