import pytest
import torch
from torch.nn import functional as F

from topoformer.binding_model import BindingTransformer
from topoformer.runtime_graph import RuntimeGraph
from topoformer.traversal_data import make_batch
from topoformer.traversal_model import TraversalTransformer


def test_mixed_dot_exact_stage3_compatibility():
    torch.manual_seed(4)
    old = TraversalTransformer(heads=4)
    new = BindingTransformer(matcher='dot', identity_update='mixed', identity_only=False)
    new.load_state_dict(old.state_dict())
    batch = make_batch(batch_size=2, depth=3)
    for mode in old.MODES:
        assert torch.equal(old(batch, mode=mode), new(batch, mode=mode))


@pytest.mark.parametrize('write', ['mixed', 'attention'])
def test_zero_strength_is_ordinary_attention(write):
    model = BindingTransformer(identity_update=write, strength=0)
    batch = make_batch(batch_size=2, depth=3)
    assert torch.equal(model(batch, mode='none'), model(batch, mode='soft'))


def test_public_inputs_only_and_permutation_equivariance():
    model = BindingTransformer()
    batch = make_batch(batch_size=2, depth=3)
    public = {k: batch[k] for k in ('entity_keys', 'adjacency', 'token_keys', 'token_values', 'start_keys', 'relations', 'node_ids')}
    assert torch.equal(model(batch), model(public))
    perm = torch.randperm(batch['token_keys'].shape[1])
    shuffled = dict(public, token_keys=public['token_keys'][:, perm], token_values=public['token_values'][:, perm])
    assert torch.allclose(model(public), model(shuffled), atol=1e-6)
    perm = torch.randperm(batch['entity_keys'].shape[1])
    shuffled = dict(public, entity_keys=public['entity_keys'][:, perm], node_ids=public['node_ids'][:, perm], adjacency=public['adjacency'][:, :, perm][:, :, :, perm])
    assert torch.allclose(model(public), model(shuffled), atol=1e-6)


def test_cosine_grounding_ignores_positive_scale_and_isolates_content():
    model = BindingTransformer(identity_init=False)
    batch = make_batch(batch_size=2)
    graph = RuntimeGraph(batch['node_ids'], batch['entity_keys'], batch['adjacency'])
    state = torch.randn(2, 3, 32)
    changed = state.clone()
    changed[..., 16:] += 20
    for role in ('query', 'key'):
        expected = model.ground(state, graph, role=role)
        assert torch.equal(expected, model.ground(changed, graph, role=role))
        assert torch.allclose(expected, model.ground(state * 7, graph, role=role), atol=1e-6)
    zero = model.ground(torch.zeros_like(state), graph)
    assert (zero[..., -1] > .999).all()


@pytest.mark.parametrize('write', ['attention', 'pointer'])
def test_identity_write_protected_from_mlp_and_value_identity_changes(write):
    model = BindingTransformer(identity_update=write)
    batch = make_batch(batch_size=2, depth=3)
    _, first = model(batch, return_diagnostics=True)
    with torch.no_grad():
        model.value.weight[:16].fill_(12.)
        model.mlp[-1].bias[:16].fill_(20.)
    _, changed = model(batch, return_diagnostics=True)
    # At the first step content is identical; raw key write ignores learned V/MLP.
    assert torch.equal(first[0]['identity_write'], changed[0]['identity_write'])
    assert torch.equal(changed[0]['state_after'][..., :16], changed[0]['identity_write'])
    assert not torch.equal(first[0]['identity_proposal'], changed[0]['identity_proposal'])


def test_pointer_uses_graph_even_at_zero_strength_and_preserves_null_mass():
    model = BindingTransformer(identity_update='pointer', strength=0)
    batch = make_batch(batch_size=2, depth=1)
    _, diag = model(batch, mode='none', return_diagnostics=True)
    selected = batch['adjacency'][torch.arange(2), batch['relations'][:, 0]]
    expected = diag[0]['pq'][..., :-1] @ selected
    assert torch.equal(diag[0]['pnext'], expected)
    assert torch.equal(diag[0]['identity_write'], expected @ batch['entity_keys'])
    empty = dict(batch, adjacency=torch.zeros_like(batch['adjacency']))
    _, null = model(empty, mode='none', return_diagnostics=True)
    assert torch.equal(null[0]['identity_write'], torch.zeros_like(null[0]['identity_write']))
    assert not torch.equal(diag[0]['identity_write'], null[0]['identity_write'])


@pytest.mark.parametrize('write', ['mixed', 'attention', 'pointer'])
def test_gradients_and_diagnostics(write):
    torch.manual_seed(23)
    model = BindingTransformer(identity_update=write, learned_temperature=True)
    batch = make_batch(batch_size=4, depth=3)
    logits, diag = model(batch, return_diagnostics=True)
    F.cross_entropy(logits, batch['targets']).backward()
    for parameter in (model.grounders[0].query_projection.weight,
                      model.grounders[0].key_projection.weight,
                      model.grounders[0].query_entity_projection.weight,
                      model.grounders[0].key_entity_projection.weight,
                      model.grounders[0].log_temperature, model.strengths):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert diag[0]['pq'].requires_grad
    assert diag[0]['pnext'].requires_grad
    assert not torch.equal(diag[0]['pq'], diag[1]['pq'])


def test_null_initialization_is_independent_and_modes_are_finite():
    for aligned in (True, False):
        model = BindingTransformer(identity_init=aligned, null_init=.23)
        assert model.grounders[0].query_null_logit.item() == pytest.approx(.23)
        assert model.grounders[0].key_null_logit.item() == pytest.approx(.23)
    batch = make_batch(batch_size=2)
    for mode in model.MODES:
        assert torch.isfinite(model(batch, mode=mode)).all()


@pytest.mark.parametrize('typed', [True, False])
def test_pointer_rejects_unnormalized_branching_and_preserves_zero_rows(typed):
    model = BindingTransformer(identity_update='pointer', typed=typed)
    batch = make_batch(batch_size=2, depth=1)
    branching = dict(batch, adjacency=torch.ones_like(batch['adjacency']))
    with pytest.raises(ValueError, match='substochastic'):
        model(branching)
    zero = dict(batch, adjacency=torch.zeros_like(batch['adjacency']))
    _, diagnostics = model(zero, return_diagnostics=True)
    assert torch.count_nonzero(diagnostics[0]['pnext']) == 0
    assert torch.count_nonzero(diagnostics[0]['identity_write']) == 0
    # Fractional branching is legal when it does not manufacture mass.
    fractional = dict(batch, adjacency=torch.ones_like(batch['adjacency']) / (2 * batch['entity_keys'].shape[1]))
    _, diagnostics = model(fractional, return_diagnostics=True)
    assert (diagnostics[0]['pnext'].sum(-1) <= .5 + 1e-6).all()


def test_pointer_rejects_untyped_union_with_branching():
    model = BindingTransformer(identity_update='pointer', typed=False)
    batch = make_batch(batch_size=1, nodes=3, depth=1)
    adjacency = torch.eye(3)[None, None].repeat(1, 3, 1, 1)
    adjacency[:, 1] = adjacency[:, 1].roll(1, -1)
    with pytest.raises(ValueError, match='substochastic'):
        model(dict(batch, adjacency=adjacency))


def test_adaptive_disabled_preserves_parameters_rng_and_outputs():
    torch.manual_seed(42)
    default = BindingTransformer()
    state = torch.random.get_rng_state()
    torch.manual_seed(42)
    disabled = BindingTransformer(adaptive_strength=False)
    assert torch.equal(state, torch.random.get_rng_state())
    for name, parameter in default.state_dict().items():
        assert torch.equal(parameter, disabled.state_dict()[name])
    batch = make_batch(batch_size=2, depth=3)
    for mode in default.MODES:
        assert torch.equal(default(batch, mode=mode), disabled(batch, mode=mode))


def test_confidence_real_uniform_null_and_gradients():
    from topoformer.binding_model import binding_confidence
    probabilities = torch.tensor([[1., 0., 0.], [1/3, 1/3, 1/3], [0., 0., 1.]], requires_grad=True)
    confidence = binding_confidence(probabilities)
    assert torch.allclose(confidence, torch.tensor([1., 0., 0.]), atol=1e-7)
    confidence.sum().backward()
    assert torch.isfinite(probabilities.grad).all()
    scores = torch.tensor([[2., 1., -1.]], requires_grad=True)
    value = binding_confidence(scores.softmax(-1))
    value.sum().backward()
    assert 0 < value.item() < 1
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_adaptive_strength_modulates_attention_not_pointer_identity_write():
    fixed = BindingTransformer(identity_update='pointer')
    adaptive = BindingTransformer(identity_update='pointer', adaptive_strength=True)
    adaptive.load_state_dict(fixed.state_dict())
    batch = make_batch(batch_size=2, depth=4)
    _, first = fixed(batch, return_diagnostics=True)
    logits, changed = adaptive(batch, return_diagnostics=True)
    for a, b in zip(first, changed):
        assert torch.equal(a['identity_write'], b['identity_write'])
        assert torch.equal(a['pnext'], b['pnext'])
        assert torch.equal(b['strengths'], a['strengths'] * b['binding_confidence'])
        assert ((b['binding_confidence'] >= 0) & (b['binding_confidence'] <= 1)).all()
    assert not torch.equal(first[0]['attention'], changed[0]['attention'])
    F.cross_entropy(logits, batch['targets']).backward()
    assert torch.isfinite(adaptive.grounders[0].query_projection.weight.grad).all()
    assert adaptive.grounders[0].query_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('write', ['mixed', 'attention'])
def test_adaptive_zero_strength_equivalence(write):
    model = BindingTransformer(identity_update=write, strength=0, adaptive_strength=True)
    batch = make_batch(batch_size=2, depth=3)
    assert torch.equal(model(batch, mode='none'), model(batch, mode='soft'))
