"""Public feature v2 and generalized population-interface fixtures (CPU only)."""
from dataclasses import replace

import pytest
import torch

from topoformer.campaign02_world import (Action, Item, WorldSpec, Workshop, action_catalog,
    encode_action, encode_action_v2, encode_observation, encode_observation_v2, encode_public,
    generate_world)


def _timeout_executor(primitive, problem, max_work):
    return {"status": "timeout", "payload": None, "work_units": max_work, "certificate_valid": False}


def _call_state():
    env = Workshop(generate_world(3, categories=2, choices=2), executor=_timeout_executor, address_seed=5)
    for row in env.observe().item_inventory:
        env.step(Action("inspect", {"target": row["handle"]}))
    env.step(Action("start_subset", {"handle": "problem_0"}))
    env.step(Action("call", {"problem": "problem_0", "budget": 16}))
    return env.observe()


def test_v2_extends_v1_prefix_and_is_versioned():
    o = _call_state()
    actions = action_catalog(o)
    obs, features = encode_public(o, actions, "v2")
    assert obs[:len(encode_observation(o))] == encode_observation(o)
    for action, row in zip(actions, features):
        assert row[:len(encode_action(o, action))] == encode_action(o, action)
    assert len({len(r) for r in features}) == 1
    with pytest.raises(ValueError):
        encode_public(o, actions, "v3")


def test_v2_budget_relations_mark_dominated_and_escalating_calls():
    o = _call_state()
    calls = {a.arguments["budget"]: encode_action_v2(o, a) for a in action_catalog(o) if a.kind == "call"}
    base = len(encode_action(o, Action("call", {"problem": "problem_0", "budget": 16})))
    # [log budget, exceeds prior work, dominated by timeout, equals remaining, <= half remaining, ...]
    assert calls[16][base+1] == 0 and calls[16][base+2] == 1
    assert calls[128][base+1] == 1 and calls[128][base+2] == 0
    assert calls[128][base] > calls[16][base]


def test_v2_item_conflicts_visible_without_pending_items():
    items = (Item("a", 0, 1, 1), Item("b", 0, 2, 2), Item("c", 1, 1, 1), Item("d", 1, 7, 7))
    spec = WorldSpec(items, (0, 1), 4, 4, (), ((0, 1, 1),), 0, 1)
    left, right = Workshop(spec), Workshop(replace(spec, incompatible=(("a", "c"),)))
    for env in (left, right):
        for item in items:
            env.step(Action("inspect", {"target": item.handle}))
    choose = Action("choose_item", {"item": "a"})
    assert encode_action(left.observe(), choose) == encode_action(right.observe(), choose)
    assert encode_action_v2(left.observe(), choose) != encode_action_v2(right.observe(), choose)


def test_v2_move_lookahead_distinguishes_downstream_weight():
    items = (Item("a", 0, 1, 1),)
    spec = WorldSpec(items, (0,), 2, 2, (), ((0, 1, 1), (1, 2, 1), (0, 2, 4)), 0, 2)
    worlds = [Workshop(spec), Workshop(replace(spec, edges=((0, 1, 1), (1, 2, 4), (0, 2, 4))))]
    move = Action("move", {"destination": 1})
    rows = []
    for env in worlds:
        env.step(Action("inspect", {"target": "map"}))
        o = env.observe()
        rows.append((encode_action(o, move), encode_action_v2(o, move)))
    assert rows[0][0] == rows[1][0] and rows[0][1] != rows[1][1]


def test_v2_invariant_to_consistent_handle_renaming():
    spec = generate_world(11, categories=2, choices=2)
    names = {x.handle: f"i{900000+i}" for i, x in enumerate(spec.items)}
    renamed = replace(spec, items=tuple(replace(x, handle=names[x.handle]) for x in spec.items),
                      incompatible=tuple((names[a], names[b]) for a, b in spec.incompatible))
    rows = []
    for s in (spec, renamed):
        env = Workshop(s, address_seed=1)
        for row in env.observe().item_inventory:
            env.step(Action("inspect", {"target": row["handle"]}))
        o = env.observe()
        rows.append((encode_observation_v2(o), [encode_action_v2(o, a) for a in action_catalog(o)]))
    assert rows[0] == rows[1]


def _config(interface, version, mode="pbt"):
    from topoformer.campaign02_population import PopulationConfig
    return PopulationConfig(mode=mode, rounds=2, updates_per_slot=1, development_examples=2,
        teacher="cheap", policy={"interface": interface, "feature_version": version},
        member_hyperparameters=tuple({"learning_rate": 1e-3*(i+1), "entropy_weight": .01} for i in range(6)),
        train={"width": 8, "device": "cpu", "batch_size": 1, "max_steps": 4, "evaluation_batch": 2},
        world_mix=({"categories": 1, "choices": 2, "locations": 3},))


def _factory(seed):
    return Workshop(generate_world(seed, categories=1, choices=2, locations=3), address_seed=seed + 17)


def _teacher():
    from topoformer.campaign02_references import ReferencePolicy
    return ReferencePolicy("cheap")


@pytest.mark.parametrize("interface,version", [("legacy", "v2"), ("memory", "v1"), ("memory", "v2")])
def test_population_runs_registered_interfaces_with_member_hyperparameters(tmp_path, interface, version):
    from topoformer.campaign02_population import PopulationRun
    torch.set_num_threads(1)
    run = PopulationRun(_config(interface, version), tmp_path / "run", _factory, _teacher)
    run.run()
    assert run.state["status"] == "completed"
    rates = [m["learning_rate"] for m in run.state["lineage"] if m["kind"] == "initialization"]
    assert rates == pytest.approx([1e-3*(i+1) for i in range(6)])
    saved = run._checkpoint(run.state["members"][0])
    assert saved["policy_config"]["feature_version"] == version
    assert ("memory_dim" in saved["policy_config"]) == (interface == "memory")


def test_population_rejects_unknown_interface_options():
    from topoformer.campaign02_population import PopulationConfig
    with pytest.raises(ValueError):
        PopulationConfig(policy={"interface": "graph_bias"})
    with pytest.raises(ValueError):
        PopulationConfig(member_hyperparameters=({"learning_rate": 1e-3},))
