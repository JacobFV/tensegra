import torch
from topoformer.traversal_data import make_batch
from topoformer.traversal_model import TraversalTransformer


def test_zero_strength_matches_ordinary_attention():
    torch.manual_seed(0)
    model = TraversalTransformer(strength=0)
    batch = make_batch(batch_size=3, depth=3)
    assert torch.equal(model(batch, mode="none"), model(batch, mode="soft"))


def test_model_does_not_read_targets_or_gold_paths():
    model = TraversalTransformer()
    batch = make_batch(batch_size=2)
    public = {key: batch[key] for key in ("entity_keys", "adjacency", "token_keys", "token_values", "start_keys", "relations", "node_ids")}
    assert torch.equal(model(batch), model(public))


def test_gradients_reach_query_key_grounders_and_strengths():
    torch.manual_seed(2)
    model = TraversalTransformer(learned_temperature=True)
    batch = make_batch(batch_size=4)
    torch.nn.functional.cross_entropy(model(batch), batch["targets"]).backward()
    for param in (model.grounders[0].query_projection.weight,
                  model.grounders[0].key_projection.weight,
                  model.grounders[0].log_temperature, model.strengths):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


def test_recompute_and_freeze_are_distinct():
    model = TraversalTransformer()
    batch = make_batch(batch_size=2, depth=3)
    _, fresh = model(batch, return_diagnostics=True)
    _, frozen = model(batch, mode="frozen", return_diagnostics=True)
    assert not torch.equal(fresh[0]["pq"], fresh[1]["pq"])
    assert torch.equal(frozen[0]["pq"], frozen[-1]["pq"])
    assert fresh[0]["pq"].shape == (2, 1, 17)


def test_token_permutation_equivariance_and_all_modes_finite():
    torch.manual_seed(3)
    model = TraversalTransformer()
    batch = make_batch(batch_size=2)
    perm = torch.randperm(batch["token_keys"].shape[1])
    shuffled = dict(batch, token_keys=batch["token_keys"][:, perm], token_values=batch["token_values"][:, perm])
    for mode in model.MODES:
        first, second = model(batch, mode=mode), model(shuffled, mode=mode)
        assert torch.isfinite(first).all()
        assert torch.allclose(first, second, atol=1e-6)


def test_shared_and_cyclic_variants_accept_deeper_computation():
    for period in (1, 4):
        model = TraversalTransformer(projection_period=period, shared_strength=True, typed=False)
        batch = make_batch(batch_size=1, nodes=5, depth=8)
        assert model(batch).shape == (1, 8)


def test_graph_slot_permutation_preserves_learned_output():
    model = TraversalTransformer()
    batch = make_batch(batch_size=2)
    perm = torch.randperm(batch["entity_keys"].shape[1])
    shuffled = dict(batch, entity_keys=batch["entity_keys"][:, perm], node_ids=batch["node_ids"][:, perm],
                    adjacency=batch["adjacency"][:, :, perm][:, :, :, perm])
    assert torch.allclose(model(batch), model(shuffled), atol=1e-6)
