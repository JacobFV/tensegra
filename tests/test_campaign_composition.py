"""CPU mechanical tests; small actor widths are explicitly test-only."""
import torch
from topoformer.campaign_composition import make_lowering_batch, model_inputs, execute_proposal
from topoformer.interface_proposals import ProposalModel


def test_exact_original_mixture_fidelity_all_event_fields():
    data = make_lowering_batch(47001, 512)
    for i, (public, (op, pointers)) in enumerate(zip(data['public'], data['labels'])):
        actual = execute_proposal(public, op, pointers)
        assert actual['status'] == 'executed'
        for field, tensor in data['reference']['public']['event'].items():
            assert torch.equal(actual['event'][field], tensor[i:i+1]), (i, field)


def test_public_forward_no_gold_and_random_instruction_order():
    data = make_lowering_batch(47002, 8)
    batch = model_inputs(data['public'])
    assert not {'event', 'targets', 'primitive', 'reference', 'labels'} & batch.keys()
    model = ProposalModel(key_dim=32, hidden=16)  # mechanical test only
    first = model(batch)
    changed = dict(batch)
    for name in ('instruction_destinations', 'instruction_arguments', 'instruction_cues', 'instruction_valid'):
        changed[name] = batch[name][:, [2, 0, 3, 1]]
    second = model(changed)
    assert torch.allclose(first['primitive'], second['primitive'], atol=1e-6)
    assert torch.allclose(first['pointers'], second['pointers'], atol=1e-6)


def test_wrong_order_executes_actual_operands_or_refuses_never_resets():
    data = make_lowering_batch(47003, 128)
    changed = 0
    for public, (op, (d, a, b)) in zip(data['public'], data['labels']):
        if op not in (1, 4): continue
        left, right = public['values'][a], public['values'][b]
        out = execute_proposal(public, op, (d, b, a))
        expected = right - left if op == 1 else right < left
        if abs(float(expected)) <= 8:
            assert out['event']['values'].item() == float(expected)
            changed += float(expected) != float(left-right if op == 1 else left < right)
        else:
            assert out['status'] == 'refused'
    assert changed > 0


def test_explicit_uncertainty_and_schema_refusal():
    data = make_lowering_batch(47004, 1)
    public = data['public'][0]; op, pointers = data['labels'][0]
    assert execute_proposal(public, op, pointers, accepted=False)['status'] == 'refused'
    assert execute_proposal(public, op, (pointers[1], *pointers[1:]))['status'] == 'refused'
    assert execute_proposal(public, op, (100, 0, 1))['status'] == 'refused'
