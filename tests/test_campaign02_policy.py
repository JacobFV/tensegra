"""Small widths here are mechanical fixtures, never experimental policies."""
import json
import random

import pytest
import torch

from topoformer.campaign02_policy import (
    CandidatePolicy, PolicyConfig, clone_checkpoint, equal_allocations,
    initial_population, offspring, restore_checkpoint,
)


@pytest.mark.parametrize("family", ["lightweight", "recurrent"])
def test_gradients_masks_and_candidate_permutation(family):
    torch.manual_seed(4)
    model = CandidatePolicy(PolicyConfig(5, 7, width=16, heads=2, family=family))
    obs, actions = torch.randn(2, 5), torch.randn(2, 3, 7)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    logits, value, state = model.score(obs, actions, mask=mask)
    assert logits.shape == (2, 3) and value.shape == (2,)
    assert torch.isneginf(logits[~mask]).all()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])) + value.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    permutation = torch.tensor([2, 0, 1])
    permuted = model.score(obs, actions[:, permutation], mask=mask[:, permutation])[0]
    assert torch.allclose(logits[:, permutation], permuted, atol=1e-6)
    # Masked candidate contents cannot alter available scores/value/workspace.
    altered = actions.clone()
    altered[~mask] = 1000
    changed = model.score(obs, altered, mask=mask)
    assert torch.allclose(logits[mask], changed[0][mask], atol=1e-6)
    assert torch.allclose(value, changed[1], atol=1e-6)
    if family == "recurrent":
        assert state.shape == (2, 4, 16)
        later = model.score(obs, actions, state, mask)[2]
        assert not torch.equal(state, later)
        assert len(model.phases) == 4
        assert len({id(p) for p in model.phases}) == 4


def test_public_inputs_only_and_validation():
    model = CandidatePolicy(PolicyConfig(5, 7, width=16, heads=2))
    obs, actions = torch.randn(2, 5), torch.randn(2, 3, 7)
    assert model.config.width == 16
    assert PolicyConfig(5, 7).width == 1024
    with pytest.raises(TypeError):
        model.score(obs, actions, hidden_world={"answer": 1})
    with pytest.raises(ValueError):
        model.score(obs, actions, mask=torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        model.score(obs, actions, hidden=torch.zeros(2, 16))
    with pytest.raises(ValueError):
        model.score(obs, actions, mask=torch.ones(2, 3))


@pytest.mark.parametrize("mode", ["reset", "inherit"])
def test_checkpoint_independence_and_optimizer_contract(mode):
    model = CandidatePolicy(PolicyConfig(5, 7, width=16, heads=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    obs, actions = torch.randn(2, 5), torch.randn(2, 3, 7)
    model(obs, actions)[0].sum().backward()
    optimizer.step()
    snapshot = clone_checkpoint(model, optimizer, optimizer_policy=mode)
    child = CandidatePolicy(model.config)
    child_optimizer = torch.optim.AdamW(child.parameters(), lr=.02)
    restore_checkpoint(snapshot, child, child_optimizer, learning_rate=.0002)
    assert child_optimizer.param_groups[0]["lr"] == .0002
    assert bool(child_optimizer.state) == (mode == "inherit")
    assert torch.equal(model(obs, actions)[0], child(obs, actions)[0])
    original = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(child.parameters()).add_(1)
    assert torch.equal(next(model.parameters()), original)
    assert torch.equal(next(iter(snapshot["model"].values())), original)


def test_population_is_serializable_and_equal_allocated():
    members = initial_population()
    assert len(members) == 6
    child = offspring(members[0], "child", rng=random.Random(3), optimizer_policy="reset", allocation_index=6)
    assert child.parent_id == members[0].member_id
    assert child.lineage_id == members[0].lineage_id
    assert child.generation == 1
    json.dumps(child.record())
    allocations = equal_allocations(members, 2, 10)
    assert {x["updates"] for x in allocations} == {10}
    assert [x["allocation_index"] for x in allocations] == list(range(12, 18))
