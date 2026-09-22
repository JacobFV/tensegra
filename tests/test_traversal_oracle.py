import pytest
import torch

from topoformer.traversal_data import make_batch
from topoformer.traversal_oracle import exact_attention_traverse

INPUTS = {'entity_keys', 'adjacency', 'node_ids', 'token_keys', 'token_values', 'start_keys', 'relations'}


@pytest.mark.parametrize('depth', [0, 1, 4, 8, 16, 32])
def test_exact_attention_oracle_rebinds_without_gold_on_shuffled_deep_tasks(depth):
    batch = make_batch(batch_size=7, nodes=64, depth=depth, distractors=19, seed=100 + depth)
    inputs = {k: v for k, v in batch.items() if k in INPUTS}
    result = exact_attention_traverse(inputs, classes=8)
    torch.testing.assert_close(result['predictions'], batch['targets'])
    torch.testing.assert_close(result['path_nodes'], batch['path_nodes'])
    assert result['attention'].shape == (7, depth, 83)
    assert result['pq'].shape == (7, depth, 65)
    assert torch.equal(result['pk'][..., -1].bool(), batch['token_nodes'] == -1)
    if depth:
        torch.testing.assert_close(result['attention'].sum(-1), torch.ones(7, depth))
        torch.testing.assert_close(result['pq'].argmax(-1), batch['path_nodes'][:, :-1])


def test_oracle_uses_supplied_graph_and_ignores_poisoned_gold():
    batch = make_batch(batch_size=8, nodes=32, depth=8, corruption=1., seed=14)
    clean = dict(batch, adjacency=batch['clean_adjacency'])
    clean_result = exact_attention_traverse(clean)
    torch.testing.assert_close(clean_result['path_nodes'], batch['path_nodes'])
    for name in ['token_nodes', 'start_nodes', 'path_nodes', 'targets', 'clean_adjacency']:
        batch[name] = torch.full_like(batch[name], -999)
    result = exact_attention_traverse(batch)
    assert not torch.equal(result['path_nodes'], clean_result['path_nodes'])


def test_missing_entity_token_is_rejected_instead_of_hidden_direct_lookup():
    batch = make_batch(batch_size=1, nodes=4, depth=4, distractors=0, seed=9)
    batch['token_keys'] = batch['token_keys'][:, :0]
    batch['token_values'] = batch['token_values'][:, :0]
    with pytest.raises(ValueError, match='exactly one token'):
        exact_attention_traverse(batch)


def test_nonfunctional_graph_is_rejected():
    batch = make_batch(batch_size=2, nodes=5, depth=2)
    batch['adjacency'] = torch.ones_like(batch['adjacency'])
    with pytest.raises(ValueError, match='functional'):
        exact_attention_traverse(batch)
