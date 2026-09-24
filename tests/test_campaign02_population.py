"""Small, non-training mechanical fixtures for population search contracts."""
from copy import deepcopy

import pytest
import torch

from topoformer.campaign02_population import (
    PopulationConfig, inherit_state, mix_index, selection_pairs, slot_member,
)


def checkpoint(offset):
    return {"policy_config": {"width": 16}, "model": {"weight": torch.tensor([float(offset)])},
            "optimizer": {"state": {0: {"step": torch.tensor(2.)}}, "param_groups": [{"lr": .1}]},
            "config": {"learning_rate": .1, "entropy_weight": .1},
            "seed_cursor": 100 + offset, "updates": 4 + offset, "presentations": 20 + offset,
            "episodes": 10 + offset, "data_hash": str(offset), "curves": [offset],
            "torch_rng": torch.tensor([offset]), "resume_history": []}


@pytest.mark.parametrize("mode", ["inherit", "reset"])
def test_inheritance_preserves_recipient_data_and_resources(mode):
    parent, recipient = checkpoint(1), checkpoint(2)
    child = inherit_state(parent, recipient, optimizer_policy=mode,
        learning_rate=.0001, entropy_weight=.02, event={"parent": "one"})
    assert child["model"]["weight"].item() == 1
    for key in ("seed_cursor", "updates", "presentations", "episodes", "data_hash", "curves"):
        assert child[key] == recipient[key]
    assert torch.equal(child["torch_rng"], recipient["torch_rng"])
    assert bool(child["optimizer"]["state"]) == (mode == "inherit")
    assert child["optimizer"]["param_groups"][0]["lr"] == .0001
    child["model"]["weight"].add_(10)
    assert parent["model"]["weight"].item() == 1
    assert recipient["model"]["weight"].item() == 2


def test_allocation_and_selection_are_prespecified():
    assert [slot_member("single", s) for s in range(6)] == [0]*6
    for mode in ("pbt", "multistart"):
        assert [slot_member(mode, s) for s in range(6)] == list(range(6))
    scores = [{"slot": s, "utility": 6-s} for s in range(6)]
    assert selection_pairs(scores) == [(0, 5), (1, 4)]
    assert selection_pairs([{**s, "utility": 1} for s in scores]) == []
    with pytest.raises(ValueError):
        selection_pairs(scores[:-1])


def test_config_overlap_guards_and_deterministic_mix():
    config = PopulationConfig.from_json({"initialization_seeds": [1, 2, 3, 4, 5, 6]})
    assert config.train["width"] == 1024
    assert [mix_index(i, 3) for i in range(10)] == [mix_index(i, 3) for i in range(10)]
    with pytest.raises(ValueError):
        PopulationConfig(development_seed_start=100_000_005)
    with pytest.raises(ValueError):
        PopulationConfig(training_seed_stride=1)
    with pytest.raises(ValueError):
        PopulationConfig(methods=("supervised",))
    with pytest.raises(ValueError):
        PopulationConfig(world_mix=({"seed": 1},))
