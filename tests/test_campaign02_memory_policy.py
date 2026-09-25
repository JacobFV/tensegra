"""Mechanical small-width fixtures; scientific workspace remains 1024."""
from copy import deepcopy

import pytest
import torch

from tensegra.campaign02_memory_policy import MemoryCandidatePolicy, MemoryPolicyConfig, MemoryFrame
from tensegra.campaign02_training import supervised_loss


def fixture(zero=False, family="lightweight"):
    torch.manual_seed(14)
    model = MemoryCandidatePolicy(MemoryPolicyConfig(2, 3, width=8, heads=2, family=family,
        memory_dim=5, zero_memory=zero, supervised_frame_batch=1))
    frames = [MemoryFrame([.1, .2], [[.1, .3, .4], [.4, .2, .7]], i%2,
              torch.arange((i+2)*5, dtype=torch.float32).reshape(i+2, 5)/10,
              [[0], [1]], {}) for i in range(3)]
    return model, frames


@pytest.mark.parametrize("family", ["lightweight", "recurrent"])
def test_memory_contract_gradients_and_padding(family):
    model, frames = fixture(family=family)
    batch = model.collate_public_frames(frames, "cpu")
    assert batch[4]["available"].tolist() == [[True, True, False, False], [True, True, True, False], [True]*4]
    logits, value, hidden = model.score(batch[0], batch[1], mask=batch[2], memory=batch[4])
    (logits.square().mean()+value.square().mean()).backward()
    assert logits.shape == (3, 2)
    assert model.memory_encoder[0].weight.grad.abs().sum() > 0
    assert (hidden is None) == (family == "lightweight")
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zero_memory_removes_fact_and_binding_information():
    model, frames = fixture(zero=True)
    model.eval()
    batch = model.collate_public_frames(frames, "cpu")
    first = model.score(batch[0], batch[1], mask=batch[2], memory=batch[4])[0]
    for frame in frames:
        frame.rows = frame.rows*100+123
        frame.links = [[1], [0]]
    changed = model.collate_public_frames(frames, "cpu")
    second = model.score(changed[0], changed[1], mask=changed[2], memory=changed[4])[0]
    assert torch.equal(first, second)
    assert changed[4]["available"].sum().item() == len(frames)


def test_frame_accumulation_matches_full_decision_mean_gradient():
    model, frames = fixture()
    whole = deepcopy(model)
    chunk_loss, count = model.backward_supervised([frames], "cpu", 8)
    full_loss, other_count = supervised_loss(whole, [frames])
    full_loss.backward()
    assert count == other_count == 3
    assert torch.allclose(chunk_loss, full_loss, atol=1e-6)
    for (_, a), (_, b) in zip(model.named_parameters(), whole.named_parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else:
            assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=1e-4)


def test_public_capacity_failure_stays_in_episode_denominator():
    from tensegra.campaign02_training import PublicInterfaceCapacityError, batched_episodes
    from tensegra.campaign02_world import Workshop, generate_world
    model, _ = fixture()
    def reject(*args):
        raise PublicInterfaceCapacityError("declared test cap")
    model.prepare_public_frame = reject
    rows = batched_episodes(model, [Workshop(generate_world(1)), Workshop(generate_world(2))])
    assert len(rows) == 2
    assert all(row["unsupported_interface"] == "declared test cap" for row in rows)
    assert all(not row["outcome"]["verified_success"] for row in rows)
    assert all(row["truncated"] for row in rows)


def test_memory_codec_version_is_explicit_checkpoint_contract():
    from dataclasses import asdict
    model, _ = fixture()
    assert asdict(model.config)["codec_version"] == "public-tree-v2"
    with pytest.raises(ValueError, match="codec semantics"):
        MemoryPolicyConfig(2, 3, width=8, heads=2, memory_dim=5, codec_version="public-tree-v1")
