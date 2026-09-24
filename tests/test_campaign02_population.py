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


def _tiny_runtime_config(mode="pbt", optimizer_policy="inherit"):
    # Width eight is explicitly a CPU-only runtime-mechanics fixture.
    return PopulationConfig(mode=mode, rounds=2, updates_per_slot=1,
        development_examples=2, optimizer_policy=optimizer_policy,
        teacher="cheap", train={"width": 8, "device": "cpu", "batch_size": 1,
            "max_steps": 4, "evaluation_batch": 2, "learning_rate": .001},
        world_mix=({"categories": 1, "choices": 2, "locations": 3},))


def _runtime_factory(seed):
    from topoformer.campaign02_world import Workshop, generate_world
    # The mechanical teacher never needs a solver; no subprocess/GPU is started.
    return Workshop(generate_world(seed, categories=1, choices=2, locations=3), address_seed=seed + 17)


def _runtime_teacher():
    from topoformer.campaign02_references import ReferencePolicy
    return ReferencePolicy("cheap")


def _assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_tree_equal(x, y)
    else:
        assert a == b


def test_actual_runtime_interrupted_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from topoformer.campaign02_population import PopulationRun
    from topoformer.campaign02_training import Learner
    torch.set_num_threads(1)
    cfg = _tiny_runtime_config()
    straight = PopulationRun(cfg, tmp_path / "straight", _runtime_factory, _runtime_teacher)
    straight.run()
    paused = PopulationRun(cfg, tmp_path / "paused", _runtime_factory, _runtime_teacher)
    for _ in range(3):
        paused._run_slot()
    original = Learner.train_tranche
    def interrupt(*args, **kwargs):
        raise RuntimeError("mechanical interruption before allocated update")
    monkeypatch.setattr(Learner, "train_tranche", interrupt)
    with pytest.raises(RuntimeError, match="mechanical interruption"):
        paused._run_slot()
    monkeypatch.setattr(Learner, "train_tranche", original)
    resumed = PopulationRun(cfg, tmp_path / "paused", _runtime_factory, _runtime_teacher)
    resumed.run()
    assert resumed.state["status"] == "completed" and resumed.state["queue"] == []
    failures = [x for x in resumed.state["attempts"] if x["status"] == "failed"]
    assert len(failures) == 1 and failures[0]["cost"]["process_cpu_seconds"] >= 0
    assert len(straight.state["allocations"]) == len(resumed.state["allocations"]) == 12
    for x, y in zip(straight.state["members"], resumed.state["members"]):
        a, b = straight._checkpoint(x), resumed._checkpoint(y)
        for key in ("model", "optimizer", "updates", "episodes", "presentations", "seed_cursor", "torch_rng", "python_rng"):
            _assert_tree_equal(a[key], b[key])
    for x, y in zip(straight.state["allocations"], resumed.state["allocations"]):
        for key in ("utility", "success", "cost", "training_seed_interval", "updates"):
            assert x[key] == y[key]


@pytest.mark.parametrize("optimizer_policy", ["inherit", "reset"])
def test_actual_replacement_retains_cursor_and_optimizer_contract(tmp_path, optimizer_policy):
    from topoformer.campaign02_population import PopulationRun
    torch.set_num_threads(1)
    run = PopulationRun(_tiny_runtime_config(optimizer_policy=optimizer_policy),
        tmp_path / optimizer_policy, _runtime_factory, _runtime_teacher)
    for _ in range(6):
        run._run_slot()
    # Controlled ranking fixture exercises replacement even if the tiny models tie.
    # These are NOT reported as measured policy utilities.
    for score in run.state["round_scores"]:
        score["utility"] = 6 - score["slot"]
    donor = run._checkpoint(run.state["members"][0])
    recipient = run._checkpoint(run.state["members"][5])
    run._selection()
    child = run._checkpoint(run.state["members"][5])
    _assert_tree_equal(child["model"], donor["model"])
    for key in ("seed_cursor", "updates", "presentations", "episodes", "torch_rng", "python_rng"):
        _assert_tree_equal(child[key], recipient[key])
    if optimizer_policy == "inherit":
        _assert_tree_equal(child["optimizer"]["state"], donor["optimizer"]["state"])
    else:
        assert child["optimizer"]["state"] == {}
    replacements = [x for x in run.state["lineage"] if x["kind"] == "replacement"]
    assert len(replacements) == 2
    assert all(x["parent_sha256"] and x["recipient_sha256"] for x in replacements)
    assert sum(x["updates"] for x in run.state["allocations"]) == 6
    assert child["seed_cursor"] != donor["seed_cursor"]


@pytest.mark.parametrize("mode", ["multistart", "single"])
def test_actual_control_allocation_and_finalist(tmp_path, mode):
    from topoformer.campaign02_population import PopulationRun
    torch.set_num_threads(1)
    run = PopulationRun(_tiny_runtime_config(mode), tmp_path / mode, _runtime_factory, _runtime_teacher)
    run.run()
    allocations = run.state["allocations"]
    assert len(allocations) == 12
    assert sum(x["updates"] for x in allocations) == 12
    assert sum(x["examples"] for x in allocations) == 24
    assert len({x["seeds_hash"] for x in allocations}) == 1
    assert not [x for x in run.state["lineage"] if x["kind"] == "replacement"]
    counts = [run._checkpoint(m)["updates"] for m in run.state["members"]]
    assert counts == ([12, 0, 0, 0, 0, 0] if mode == "single" else [2]*6)
    if mode == "single":
        assert run.state["finalist"]["slot"] == 5
    assert run.state["queue"] == []
