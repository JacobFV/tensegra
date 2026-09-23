import torch
from topoformer.interface_proposals import make_proposals, collate_proposals, ProposalModel, proposal_metrics, arithmetic_delta


def test_proposals_keys_order_controls_and_ood():
    data = make_proposals(100, seed=3)
    assert len({x.primitive for x in data}) == 5
    for x in data:
        assert tuple(x.keys[i] for i in x.targets) == x.role_keys
        assert len(set(x.keys)) == len(x.keys)
    normal = collate_proposals(data[:8])
    missing = collate_proposals(data[:8], control='missing')
    reverse = collate_proposals(data[:8], control='reverse')
    assert missing['instruction_arguments'].count_nonzero() == 0
    assert torch.equal(reverse['instruction_arguments'][:, :, 0], normal['instruction_arguments'][:, :, 1])
    assert make_proposals(10, seed=3, split='ood')[0].names != data[0].names


def test_model_pointer_equivariance_and_gradient():
    b = collate_proposals(make_proposals(4, seed=4, candidates=6))
    m = ProposalModel(key_dim=16, hidden=24)
    out = m(b)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    changed = dict(b, keys=b['keys'][:, perm], valid=b['valid'][:, perm])
    assert torch.allclose(m(changed)['pointers'], out['pointers'][:, :, perm], atol=1e-6)
    loss = torch.nn.functional.cross_entropy(out['primitive'], b['primitive'])
    loss += torch.nn.functional.cross_entropy(out['pointers'].flatten(0, 1), b['targets'].flatten())
    loss.backward()
    assert m.query.weight.grad.abs().sum() > 0


def test_ordered_scoring_catches_reversed_subtraction():
    b = collate_proposals(make_proposals(6, seed=1, primitive='sub'))
    p = torch.nn.functional.one_hot(b['targets'], b['keys'].shape[1]).float()
    out = {'primitive': torch.nn.functional.one_hot(b['primitive'], 5).float(), 'pointers': p[:, [0, 2, 1]]}
    metrics = proposal_metrics(out, b)
    assert metrics['primitive'] == 1 and metrics['destination'] == 1
    assert metrics['full'] == 0 and metrics['noncommutative_order'] == 0


def test_delta_order_and_existing_semantics():
    assert arithmetic_delta('sub', 9, 4) == {'remove_operands': [9, 4], 'insert_value': 5, 'value_type':'integer'}
    assert arithmetic_delta('compare', 4, 9)['insert_value'] is True
    assert arithmetic_delta('neg', 9)['insert_value'] == -9


def test_empty_validation_cannot_pass_gate():
    from topoformer.interface_proposals import gate_a
    assert not gate_a([])


def test_strict_gate_rejects_missing_splits_and_boundary():
    from topoformer.interface_proposals import gate_a, gate_a_matrix
    good = dict(count=512, primitive=1., destination=1., operand1=1., operand2=1.,
                full=1., noncommutative_count=100, noncommutative_order=1.)
    assert gate_a([good,good])
    assert not gate_a([good])
    assert not gate_a([good,dict(good,primitive=.99)])
    assert not gate_a([good,dict(good,full=.98)])
    assert not gate_a_matrix({0:{'iid_validation':good}}, [0])
    assert not gate_a_matrix({0:{'iid_validation':good,'ood_validation':good}}, [0,1])


def test_public_instruction_batch_has_no_selected_gold_record():
    b = collate_proposals(make_proposals(4,seed=5,candidates=6))
    m = ProposalModel()
    public = {k:v for k,v in b.items() if k not in ('targets','primitive','binary')}
    assert torch.equal(m(public)['pointers'],m(b)['pointers'])
    order = torch.tensor([2,0,3,1])
    shuffled = dict(public)
    for k in ('instruction_destinations','instruction_arguments','instruction_cues','instruction_valid'):
        shuffled[k] = shuffled[k][:,order]
    assert torch.allclose(m(public)['pointers'],m(shuffled)['pointers'],atol=1e-6)


def test_compare_delta_preserves_existing_boolean_lt_semantics():
    delta = arithmetic_delta('compare', 4, 9)
    assert delta['insert_value'] is True
    assert delta['value_type'] == 'boolean'
    assert arithmetic_delta('compare', 9, 4)['insert_value'] is False


def test_boolean_comparison_results_never_feed_numeric_operands():
    for row in make_proposals(100,seed=19,split='ood'):
        boolean_keys = {r[0] for r in row.instructions if r[1] == 4}
        assert all(r[2] not in boolean_keys and r[3] not in boolean_keys for r in row.instructions)


def test_progressive_joint_support_has_no_crossed_impossible_pairs_or_future_input():
    from topoformer.interface_proposals import joint_evidence, JointPosteriorModel
    episode = joint_evidence(seed=1, steps=4)
    assert episode['hypotheses'] == [('sub', 'd', 'a', 'b'), ('sub', 'd', 'b', 'a')]
    model = JointPosteriorModel(hidden=8)
    full = model(episode['frames'][None])
    prefix = model(episode['frames'][None,:2])
    assert torch.allclose(full[:,:2],prefix)
    assert torch.allclose(full.sum(-1),torch.ones(1,4))
    assert full.shape[-1] == 2
