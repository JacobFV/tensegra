"""CPU mechanical tests; small actor widths are explicitly test-only."""
import torch
from tensegra.campaign_composition import make_lowering_batch, model_inputs, execute_proposal
from tensegra.interface_proposals import ProposalModel


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


def test_malformed_public_records_refuse_without_gold():
    data = make_lowering_batch(47005, 1)
    public = data['public'][0]; op, pointers = data['labels'][0]
    bad = dict(public, instruction_destinations=public['instruction_destinations'].clone())
    bad['instruction_destinations'][0] = bad['instruction_destinations'][1]
    assert execute_proposal(bad, op, pointers)['status'] == 'refused'
    bad = dict(public, keys=public['keys'].clone())
    bad['keys'][0] = bad['keys'][1]
    assert execute_proposal(bad, op, pointers)['status'] == 'refused'


def test_predicted_unary_arity_canonicalizes_null_without_gold():
    data = make_lowering_batch(47006, 64)
    for public, (_, pointers) in zip(data['public'], data['labels']):
        d, a, _ = pointers
        actual = execute_proposal(public, 3, (d, a, a))
        expected = -public['values'][a]
        if abs(expected) <= 8:
            assert actual['status'] == 'executed'
            assert actual['event']['values'].item() == expected
            assert actual['event']['arguments'][0, 0, 1].count_nonzero() == 0
        else:
            assert actual['status'] == 'refused'


def test_wrong_destination_is_actual_provenance_not_gold_reset():
    data = make_lowering_batch(47007, 1)
    public = data['public'][0]; op, (d, a, b) = data['labels'][0]
    wrong = next(i for i, role in enumerate(public['roles']) if role == 'destination' and i != d)
    out = execute_proposal(public, op, (wrong, a, b))
    assert torch.equal(out['event']['provenance'][0, 0], public['keys'][wrong])
    assert not torch.equal(out['event']['provenance'], data['reference']['public']['event']['provenance'])
