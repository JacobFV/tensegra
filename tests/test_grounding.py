import pytest
import torch

from tensegra.runtime_graph import RuntimeGraph
from tensegra.grounding import SoftGrounding, induce_bias


def graph():
    return RuntimeGraph(
        torch.tensor([[91, 4, 55], [8, 2, 0]]),
        torch.randn(2, 3, 4, dtype=torch.float64),
        torch.tensor([[[[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]],
                      [[[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]]], dtype=torch.float64),
        torch.tensor([[True, True, True], [True, True, False]]),
    )


def test_directional_one_hot_grounding_and_null_have_exact_semantics():
    g = graph()
    p = torch.eye(4, dtype=torch.float64)[None].expand(2, -1, -1)
    result = induce_bias(p, g.adjacency, p)
    torch.testing.assert_close(result[:, :, :3, :3], g.adjacency)
    assert result[0, 0, 0, 1] == 1 and result[0, 0, 1, 0] == 0
    assert result[..., -1, :].count_nonzero() == 0
    assert result[..., -1].count_nonzero() == 0


def test_soft_grounding_masks_padding_and_can_be_unbound():
    g = graph()
    module = SoftGrounding(5, 4).double()
    with torch.no_grad():
        module.query_null_logit.fill_(100)
        module.key_null_logit.fill_(100)
    pq, pk = module(torch.randn(2, 2, 5).double(), torch.randn(2, 7, 5).double(), g)
    assert pq.shape == (2, 2, 4) and pk.shape == (2, 7, 4)
    assert torch.equal(pq[1, :, 2], torch.zeros(2, dtype=pq.dtype))
    assert torch.equal(pk[1, :, 2], torch.zeros(7, dtype=pk.dtype))
    assert (pq[..., -1] > .999).all() and (pk[..., -1] > .999).all()
    torch.testing.assert_close(pq.sum(-1), torch.ones(2, 2, dtype=pq.dtype))


def test_empty_valid_graph_grounds_entirely_to_null():
    g = RuntimeGraph(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, 3),
                     torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, dtype=torch.bool))
    p, _ = SoftGrounding(3, 3)(torch.ones(1, 1, 3), torch.ones(1, 1, 3), g)
    torch.testing.assert_close(p, torch.tensor([[[0., 0., 1.]]]))


def test_query_key_roles_are_independent_and_differentiable():
    torch.manual_seed(4)
    g = graph()
    g.embeddings.requires_grad_()
    z = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
    module = SoftGrounding(5, 4, grounding_dim=3, learned_temperature=True).double()
    pq, pk = module(z, z, g)
    assert not torch.allclose(pq, pk)
    loss = (induce_bias(pq, g.adjacency, pk) * torch.randn(2, 1, 5, 5)).sum()
    loss.backward()
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert z.grad.abs().sum() > 0 and g.embeddings.grad.abs().sum() > 0


def test_runtime_node_permutation_preserves_induced_bias():
    torch.manual_seed(8)
    g = graph()
    order = torch.tensor([2, 0, 1])
    permuted = RuntimeGraph(g.node_ids[:, order], g.embeddings[:, order],
                           g.adjacency[:, :, order][:, :, :, order], g.valid[:, order])
    module = SoftGrounding(5, 4).double()
    q, k = torch.randn(2, 2, 5).double(), torch.randn(2, 6, 5).double()
    pq, pk = module(q, k, g)
    rq, rk = module(q, k, permuted)
    torch.testing.assert_close(rq[..., :3], pq[..., order])
    torch.testing.assert_close(induce_bias(pq, g.adjacency, pk),
                               induce_bias(rq, permuted.adjacency, rk))


def test_bias_gradcheck_and_rectangular_multiple_relations():
    p = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True).softmax(-1)
    k = torch.randn(2, 2, 5, dtype=torch.float64, requires_grad=True).softmax(-1)
    a = torch.randn(2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    assert induce_bias(p, a, k).shape == (2, 2, 3, 2)
    assert torch.autograd.gradcheck(induce_bias, (p, a, k))


@pytest.mark.parametrize('mutation, message', [
    ('ids', 'unique'), ('mask', 'boolean'), ('edge', 'padding'),
    ('nan', 'finite'), ('shape', 'shape'), ('id_type', 'long'),
])
def test_runtime_graph_rejects_invalid_contracts(mutation, message):
    g = graph()
    ids, emb, adj, valid = g.node_ids.clone(), g.embeddings.clone(), g.adjacency.clone(), g.valid.clone()
    if mutation == 'ids': ids[0, 1] = ids[0, 0]
    if mutation == 'mask': valid = valid.float()
    if mutation == 'edge': adj[1, 0, 0, 2] = 1
    if mutation == 'nan': emb[0, 0, 0] = float('nan')
    if mutation == 'shape': adj = adj[..., :2]
    if mutation == 'id_type': ids = ids.float()
    with pytest.raises((ValueError, TypeError), match=message):
        RuntimeGraph(ids, emb, adj, valid)


def test_fixed_temperature_is_not_learnable_and_to_preserves_ids():
    module = SoftGrounding(5, 4, temperature=.7)
    assert float(module.temperature) == pytest.approx(.7)
    assert 'log_temperature' not in dict(module.named_parameters())
    g = graph().to('cpu')
    assert g.node_ids.dtype == torch.long and g.embeddings.dtype == torch.float64


@pytest.mark.parametrize('temperature', [0, -1, float('inf'), float('nan')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='temperature'):
        SoftGrounding(5, 4, temperature=temperature)


def test_grounding_and_bias_reject_mismatched_shapes():
    g = graph()
    module = SoftGrounding(5, 4).double()
    with pytest.raises(ValueError, match='shape'):
        module(torch.randn(1, 2, 5).double(), torch.randn(2, 2, 5).double(), g)
    with pytest.raises(ValueError, match='shape'):
        induce_bias(torch.zeros(2, 2, 2), g.adjacency, torch.zeros(2, 2, 4))


def test_zero_entity_graph_has_null_only_distribution_and_zero_bias():
    g = RuntimeGraph(torch.empty(2, 0, dtype=torch.long), torch.empty(2, 0, 3),
                     torch.empty(2, 2, 0, 0))
    pq, pk = SoftGrounding(3, 3)(torch.ones(2, 1, 3), torch.ones(2, 4, 3), g)
    torch.testing.assert_close(pq, torch.ones(2, 1, 1))
    torch.testing.assert_close(induce_bias(pq, g.adjacency, pk), torch.zeros(2, 2, 1, 4))


def test_low_precision_grounding_and_contraction_accumulate_float32():
    g = RuntimeGraph(torch.arange(3)[None], torch.randn(1, 3, 4).half(),
                     torch.eye(3)[None, None].half())
    pq, pk = SoftGrounding(4, 4).half()(torch.randn(1, 2, 4).half(), torch.randn(1, 2, 4).half(), g)
    assert pq.dtype == pk.dtype == torch.float32
    assert induce_bias(pq.half(), g.adjacency, pk.half()).dtype == torch.float32
