import pytest
import torch
import torch.nn.functional as F

from topoformer.binding_metrics import binding_diagnostics, binding_losses, summarize_binding
from topoformer.traversal_data import make_batch


def fixture(paths_correct, distractors=1):
    """Explicit canonical correctness, including initial state and destination."""
    correct = torch.tensor(paths_correct, dtype=torch.bool)
    b, length = correct.shape
    batch = make_batch(b, nodes=4, depth=length - 1, key_dim=4, distractors=distractors, seed=72)
    path = batch['path_nodes']
    prediction = torch.where(correct, path, (path + 1) % 4)
    probs = F.one_hot(prediction, 5).float()
    tokens = batch['token_nodes']
    pk = F.one_hot(torch.where(tokens < 0, 4, tokens), 5).float()
    records = []
    for step in range(length - 1):
        records.append({'pq': probs[:, step:step + 1], 'pq_after': probs[:, step + 1:step + 2],
                        'pk': pk, 'pnext': probs[:, step + 1:step + 2, :4]})
    return records, batch


def test_canonical_transitions_do_not_duplicate_pre_post():
    records, batch = fixture([[1, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 0]])
    result = binding_diagnostics(records, batch)
    assert result['canonical_correct'].tolist() == [[True, True, False, False], [True, False, True, True], [True, True, True, False]]
    assert result['per_example']['first_error_hop'].tolist() == [2, 1, 3]
    assert result['counts']['error_to_error'] == 1
    assert result['counts']['error_to_correct'] == 1
    assert result['counts']['error_transition_total'] == 2
    assert result['per_example']['ever_recovered'].tolist() == [False, True, False]
    assert result['per_example']['final_ground_correct'].tolist() == [False, True, False]
    assert not result['per_example']['complete_path'].any()


def test_final_hop_counts_and_empty_denominators():
    records, batch = fixture([[1, 1, 1], [1, 1, 0]], distractors=0)
    result = summarize_binding([binding_diagnostics(records, batch)])
    assert result['metrics']['complete_path'] == .5
    assert result['metrics']['canonical_complete_path'] == .5
    assert result['metrics']['error_persistence'] is None
    assert result['metrics']['error_recovery'] is None
    assert result['metrics']['distractor_null_accuracy'] is None
    assert result['per_example']['first_error_hop'] == [3, 2]
    assert result['metrics']['product_canonical_marginal_accuracy'] == .5


def test_pooled_conditional_counts_not_batch_rate_mean():
    r1, b1 = fixture([[0, 0, 0, 0]])
    r2, b2 = fixture([[0, 1, 1, 1], [1, 1, 1, 1]])
    pooled = summarize_binding([binding_diagnostics(r1, b1), binding_diagnostics(r2, b2)])
    assert pooled['metrics']['error_persistence'] == .75
    assert pooled['metrics']['error_recovery'] == .25


def test_losses_gradient_roles_and_detached_cycle_target():
    records, batch = fixture([[1, 1, 1], [1, 1, 1]])
    leaves = []
    for record in records:
        for field in ('pq', 'pq_after', 'pk'):
            leaf = torch.randn_like(record[field], requires_grad=True)
            record[field] = leaf.softmax(-1)
            leaves.append(leaf)
        target = torch.rand_like(record['pnext'], requires_grad=True)
        record['pnext'] = target / (1 + target.sum(-1, keepdim=True))
        record['target_leaf'] = target
    losses = binding_losses(records, batch)
    assert set(losses) == {'query', 'post', 'key', 'ground', 'null', 'cycle'}
    (losses['ground'] + losses['null'] + losses['cycle']).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in leaves)
    assert all(r['target_leaf'].grad is None for r in records)


def test_perfect_cycle_zero_and_null_balanced_across_group_sizes():
    records, batch = fixture([[1, 1, 1]])
    losses = binding_losses(records, batch)
    assert losses['ground'] == 0
    assert losses['cycle'] == 0
    records[0]['pk'] = torch.full_like(records[0]['pk'], .1)
    records[0]['pk'][..., -1] = .6
    records[1]['pk'] = records[0]['pk']
    result = binding_losses(records, batch)
    expected = -(torch.tensor(.4).log() + torch.tensor(.6).log()) / 2
    assert result['null'] == pytest.approx(expected.item())


def test_drift_and_interference_geometry():
    records, batch = fixture([[1, 1], [1, 1]])
    record = records[0]
    rows = torch.arange(2)
    before = batch['entity_keys'][rows, batch['path_nodes'][:, 0]]
    after = batch['entity_keys'][rows, batch['path_nodes'][:, 1]]
    record.update(state_before=before[:, None], state_after=after[:, None],
                  identity_write=after[:, None], identity_proposal=(after + 1)[:, None],
                  projected_query_norm=torch.ones(2, 1))
    result = binding_diagnostics(records, batch)['per_step']
    assert torch.allclose(result['post_identity_drift'], torch.zeros(2, 1))
    assert torch.allclose(result['write_override_norm'], torch.full((2, 1), 2.))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -0.1, 1.1])
def test_cycle_rejects_nonprobability_pushforward(invalid):
    records, batch = fixture([[1, 1]])
    records[0]['pnext'] = torch.zeros(1, 1, 4)
    records[0]['pnext'][0, 0, 0] = invalid
    with pytest.raises(ValueError, match='cycle target'):
        binding_losses(records, batch)


def test_cycle_rejects_multiedge_mass_and_invalid_full_targets():
    records, batch = fixture([[1, 1]])
    records[0]['pnext'] = torch.full((1, 1, 4), .3)
    with pytest.raises(ValueError, match='cycle target'):
        binding_losses(records, batch)
    records[0]['next_distribution'] = torch.full((1, 1, 5), .1)
    with pytest.raises(ValueError, match='cycle target'):
        binding_losses(records, batch)


def test_cycle_retains_null_and_tolerates_only_roundoff():
    records, batch = fixture([[1, 1]])
    records[0]['pnext'] = torch.zeros(1, 1, 4)
    records[0]['pq_after'] = torch.tensor([[[0., 0., 0., 0., 1.]]])
    assert binding_losses(records, batch)['cycle'] == 0
    records[0]['pnext'][0, 0, 0] = -1e-7
    assert binding_losses(records, batch)['cycle'] == 0


def test_relation_attention_support_handles_multiple_and_missing_edges():
    records, batch = fixture([[1, 1]])
    gold = int(batch['path_nodes'][0, 0])
    relation = int(batch['relations'][0, 0])
    batch['adjacency'][0, relation, gold].zero_()
    batch['adjacency'][0, relation, gold, :2] = 1
    records[0]['attention'] = torch.full((1, 1, 1, 5), .2)
    result = binding_diagnostics(records, batch)
    assert result['per_step']['relation_attention_mass'].item() == pytest.approx(.4)
    batch['adjacency'][0, relation, gold].zero_()
    result = binding_diagnostics(records, batch)
    assert result['per_step']['relation_attention_mass'].item() == 0
