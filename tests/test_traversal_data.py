import pytest
import torch

from topoformer.traversal_data import corrupt_adjacency, make_batch, oracle_traverse


def test_reproducible_without_global_rng_changes():
    torch.manual_seed(8)
    state = torch.random.get_rng_state()
    first = make_batch(seed=7)
    second = make_batch(seed=7)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not torch.equal(first['entity_keys'], make_batch(seed=8)['entity_keys'])


@pytest.mark.parametrize('depth', [0, 3, 8, 16, 32])
def test_oracle_and_shuffled_token_answer(depth):
    batch = make_batch(batch_size=9, nodes=31, depth=depth, distractors=7)
    path = oracle_traverse(batch['adjacency'], batch['start_nodes'], batch['relations'])
    assert torch.equal(path, batch['path_nodes'])
    match = batch['token_nodes'] == path[:, -1, None]
    assert torch.all(match.sum(1) == 1)
    assert torch.equal(batch['token_values'][match], batch['targets'])
    # Oracle signature has no target or intermediate path input.
    batch['targets'].fill_(-99)
    batch['path_nodes'].fill_(-99)
    assert torch.equal(path, oracle_traverse(batch['adjacency'], batch['start_nodes'], batch['relations']))


def test_identity_token_permutation_and_null_distractors():
    batch = make_batch(batch_size=8, nodes=13, distractors=5)
    for row in range(8):
        mapping = batch['token_nodes'][row]
        valid = mapping >= 0
        assert torch.equal(mapping[valid].sort().values, torch.arange(13))
        assert torch.equal(batch['token_keys'][row, valid], batch['entity_keys'][row, mapping[valid]])
        assert (mapping == -1).sum() == 5
        assert batch['node_ids'][row].unique().numel() == 13
    assert not torch.equal(batch['token_nodes'][0], batch['token_nodes'][1])
    assert torch.allclose(batch['entity_keys'].norm(dim=-1), torch.ones(8, 13))


def test_node_relabeling_equivariance():
    batch = make_batch(batch_size=4, nodes=17, depth=32)
    permutation = torch.randperm(17)
    inverse = permutation.argsort()
    renamed = batch['adjacency'][:, :, permutation][:, :, :, permutation]
    result = oracle_traverse(renamed, inverse[batch['start_nodes']], batch['relations'])
    assert torch.equal(permutation[result], batch['path_nodes'])


def test_graph_identity_keys_do_not_contain_payload():
    first = make_batch(classes=2, seed=15)
    second = make_batch(classes=23, seed=15)
    assert torch.equal(first['entity_keys'], second['entity_keys'])
    assert torch.equal(first['adjacency'], second['adjacency'])
    assert not torch.equal(first['token_values'], second['token_values'])


def test_composition_split():
    train = make_batch(batch_size=128, depth=8, composition='train')['relations']
    heldout = make_batch(batch_size=128, depth=8, composition='heldout')['relations']
    assert not ((train[:, :-1] == 0) & (train[:, 1:] == 1)).any()
    assert ((heldout[:, :-1] == 0) & (heldout[:, 1:] == 1)).any(1).all()
    with pytest.raises(ValueError):
        make_batch(depth=1, composition='heldout')


def test_corruption_preserves_functionality_and_labels():
    clean = make_batch(seed=21)
    corrupted = make_batch(seed=21, corruption=1.)
    assert torch.all(corrupted['adjacency'].sum(-1) == 1)
    assert torch.all(corrupted['adjacency'].argmax(-1) != clean['adjacency'].argmax(-1))
    for key in clean.keys() - {'adjacency'}:
        assert torch.equal(clean[key], corrupted[key])
    assert torch.equal(corrupt_adjacency(clean['adjacency'], 0), clean['adjacency'])
    assert torch.equal(corrupt_adjacency(torch.ones(1, 1, 1, 1), 1), torch.ones(1, 1, 1, 1))
