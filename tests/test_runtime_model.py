import pytest
import torch
from topoformer.runtime_model import RuntimeBindingModel, binding_confidence


def sample():
    generator = torch.Generator().manual_seed(7)
    return dict(surface=torch.randint(1, 20, (2, 3, 5), generator=generator),
                reference=torch.randn(2, 3, 16, generator=generator),
                candidate_keys=torch.randn(2, 6, 16, generator=generator),
                candidate_values=torch.randn(2, 6, generator=generator),
                candidate_types=torch.randint(0, 8, (2, 6), generator=generator),
                candidate_mask=torch.ones(2, 6, dtype=torch.bool),
                adjacency=torch.rand(2, 6, 6, 6, generator=generator),
                candidate_payload=torch.randn(2, 6, 3, generator=generator),
                step_mask=torch.ones(2, 3, dtype=torch.bool), style=torch.tensor([0, 1]),
                comparison=torch.tensor([.1, -.2]))


def model():
    torch.manual_seed(3)
    return RuntimeBindingModel(64, 7, width=16)


def test_lowering_and_lifting_train_independently():
    m, public = model(), sample()
    out = m(public)
    assert out['op_logits'].shape == (2, 3, 7)
    assert out['binding_logits'].shape == (2, 3, 7)
    loss = out['op_logits'].square().mean() + out['binding_logits'].square().mean()
    loss += m.lift(torch.tensor([3., -2.]), public['style'], public['comparison']).square().mean()
    loss.backward()
    for parameter in (m.lower_query.weight, m.lower_key.weight, m.operation.weight, m.lifter[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert 'output_logits' not in out  # exact execution is outside neural forward


@pytest.mark.parametrize('mode', ['runtime', 'none', 'graph_data', 'soft', 'protected_learned'])
def test_candidate_permutation_equivariance(mode):
    m, public = model(), sample()
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    changed = dict(public)
    for name in ('candidate_keys', 'candidate_values', 'candidate_types', 'candidate_mask', 'candidate_payload'):
        changed[name] = public[name][:, perm]
    changed['adjacency'] = public['adjacency'][:, :, perm][:, :, :, perm]
    a, b = m(public, mode=mode), m(changed, mode=mode)
    torch.testing.assert_close(a['binding_logits'][..., :-1][..., perm], b['binding_logits'][..., :-1])
    if mode != 'runtime':
        torch.testing.assert_close(a['output_logits'], b['output_logits'], atol=2e-6, rtol=2e-6)


def test_zero_soft_strength_is_unstructured_attention():
    m, public = model(), sample()
    m.strength = 0.
    a, b = m(public, mode='none'), m(public, mode='soft')
    torch.testing.assert_close(a['output_logits'], b['output_logits'], rtol=0, atol=0)


def test_confidence_null_and_uniform_are_low():
    certain_ops = torch.tensor([[20., -20., -20.]])
    certain_binding = torch.tensor([[20., -20., -20.]])
    null_binding = torch.tensor([[-20., -20., 20.]])
    assert binding_confidence(certain_ops, certain_binding).item() > .999
    assert binding_confidence(certain_ops, null_binding).item() < 1e-6
    assert binding_confidence(torch.zeros(1, 3), torch.zeros(1, 3)).item() < 1e-6


def test_no_gold_access_and_masked_padding_finite():
    m, public = model(), sample()
    original = m(public)
    public.update(targets=torch.randn(3), gold={'anything': object()})
    torch.testing.assert_close(original['binding_logits'], m(public)['binding_logits'])
    public['surface'][:, -1] = 0
    public['step_mask'][:, -1] = False
    public['candidate_mask'][0] = False
    out = m(public, mode='protected_learned')
    assert torch.isfinite(out['output_logits']).all()
    assert out['binding_logits'][0, :, -1].isfinite().all()
    assert out['binding_logits'][0, :, :-1].isneginf().all()


def test_neural_controls_receive_output_comparison():
    m, public = model(), sample()
    a = m(public, mode='none')['output_logits']
    public['comparison'] = public['comparison'] + 1
    b = m(public, mode='none')['output_logits']
    assert not torch.allclose(a, b)


@pytest.mark.parametrize('mode', ['runtime', 'none', 'graph_data', 'soft', 'protected_learned'])
def test_scope_and_index_payload_are_observable(mode):
    m, public = model(), sample()
    a = m(public, mode=mode)
    public['candidate_payload'] = public['candidate_payload'] + 1
    b = m(public, mode=mode)
    assert not torch.allclose(a['binding_logits'], b['binding_logits'])
    if mode != 'runtime':
        assert not torch.allclose(a['output_logits'], b['output_logits'])
