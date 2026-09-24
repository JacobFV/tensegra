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


def test_actor_critic_v2_kl_is_zero_against_identical_reference_and_normalizes():
    from copy import deepcopy
    from topoformer.campaign02_policy import CandidatePolicy, PolicyConfig
    from topoformer.campaign02_training import TrainConfig, actor_critic_objective, batched_on_policy, public_frame
    torch.manual_seed(0)
    _, obs, cand = public_frame(_factory(1).observe(), "v2")
    model = CandidatePolicy(PolicyConfig(len(obs), len(cand[0]), width=8, feature_version="v2"))
    reference = deepcopy(model).eval()
    results, terms, kls = batched_on_policy(model, [_factory(s) for s in range(3)], max_steps=4, reference=reference)
    assert all(len(k) == len(t) for k, t in zip(kls, terms))
    assert max(abs(float(k)) for row in kls for k in row) < 1e-5
    cfg = TrainConfig(width=8, method="actor_critic", rollout_mode="batched", kl_weight=.5, advantage_normalization=True)
    loss, parts = actor_critic_objective(terms, cfg, kls)
    assert "kl_to_round_start" in parts and torch.isfinite(loss)
    loss.backward()
    with pytest.raises(ValueError):
        TrainConfig(kl_weight=-1.0)


def test_actor_critic_v1_defaults_unchanged():
    from topoformer.campaign02_training import TrainConfig, actor_critic_objective
    value = torch.tensor(0.2, requires_grad=True)
    logp = torch.tensor(-1.0, requires_grad=True)
    episode = [(logp, value, torch.tensor(1.0), 0.5)]
    loss, parts = actor_critic_objective([episode], TrainConfig(method="actor_critic"))
    expected = -(-1.0)*(0.5-0.2) + .5*(0.3**2) - .01*1.0
    assert float(loss) == pytest.approx(expected)
    assert "kl_to_round_start" not in parts


def test_population_imports_bank_with_reset_optimizer_and_fresh_stream(tmp_path):
    from topoformer.campaign02_population import PopulationConfig, PopulationRun, sha256
    torch.set_num_threads(1)
    source = PopulationRun(_config("legacy", "v2", mode="multistart"), tmp_path / "bank", _factory, _teacher)
    source.run()
    bank = [{"path": str(tmp_path / "bank" / m["checkpoint"]), "sha256": m["checkpoint_sha256"]} for m in source.state["members"]]
    cfg = PopulationConfig(mode="pbt", rounds=2, updates_per_slot=1, development_examples=2, teacher="cheap",
        policy={"interface": "legacy", "feature_version": "v2"}, initial_checkpoints=tuple(bank),
        methods=("actor_critic", "actor_critic"), training_seed_start=200_000_000,
        member_hyperparameters=tuple({"learning_rate": 1e-4, "entropy_weight": .01, "kl_weight": .1} for _ in range(6)),
        train={"width": 8, "device": "cpu", "batch_size": 1, "max_steps": 4, "evaluation_batch": 2,
               "rollout_mode": "batched", "advantage_normalization": True},
        world_mix=({"categories": 1, "choices": 2, "locations": 3},))
    run = PopulationRun(cfg, tmp_path / "rl", _factory, _teacher)
    first = run._checkpoint(run.state["members"][0])
    original = torch.load(bank[0]["path"], map_location="cpu", weights_only=False)
    for key, value in original["model"].items():
        assert torch.equal(first["model"][key], value)
    assert first["optimizer"]["state"] == {} and first["seed_cursor"] == 200_000_000
    assert first["config"]["kl_weight"] == .1 and first["resume_history"][-1]["kind"] == "bank_import"
    for _ in range(6):
        run._run_slot()
    for score in run.state["round_scores"]:
        score["utility"] = 6 - score["slot"]
    run._selection()
    events = [x for x in run.state["lineage"] if x["kind"] == "replacement"]
    assert len(events) == 2 and all(x["kl_factor"] in (.8, 1.2) for x in events)
    child = run._checkpoint(run.state["members"][5])
    assert child["config"]["kl_weight"] == pytest.approx(.1*events[0]["kl_factor"])
    bad = dict(bank[0], sha256="0"*64)
    with pytest.raises(ValueError):
        PopulationRun(PopulationConfig(**{**cfg.__dict__, "initial_checkpoints": (bad,)+tuple(bank[1:])}),
                      tmp_path / "bad", _factory, _teacher)


def test_curriculum_mutation_changes_training_mixture_only(tmp_path):
    from topoformer.campaign02_population import (PopulationConfig, PopulationRun, mutate_curriculum,
                                                  weighted_index)
    import random as _random
    assert [weighted_index(s, [1, 0, 0]) for s in range(20)] == [0]*20
    assert {weighted_index(s, [.5, .5]) for s in range(200)} == {0, 1}
    w, info = mutate_curriculum([.25]*4, _random.Random(0))
    assert sum(w) == pytest.approx(1) and min(w) >= .02 and info["factor"] in (.5, 2.)
    torch.set_num_threads(1)
    mix = ({"categories": 1, "choices": 2, "locations": 3}, {"categories": 1, "choices": 3, "locations": 3})
    seen = []
    def component_factory(seed, index):
        seen.append(index)
        return Workshop(generate_world(seed, **mix[index]), address_seed=seed + 17)
    def factory(seed):
        from topoformer.campaign02_population import mix_index
        return component_factory(seed, mix_index(seed, 2))
    cfg = PopulationConfig(mode="pbt", rounds=2, updates_per_slot=1, development_examples=2, teacher="cheap",
        curriculum_mutation=True, train={"width": 8, "device": "cpu", "batch_size": 1, "max_steps": 4, "evaluation_batch": 2},
        world_mix=mix)
    run = PopulationRun(cfg, tmp_path / "cur", factory, _teacher, component_factory=component_factory)
    assert all(m["curriculum"] == [.5, .5] for m in run.state["members"])
    for _ in range(6):
        run._run_slot()
    for score in run.state["round_scores"]:
        score["utility"] = 6 - score["slot"]
    run._selection()
    events = [x for x in run.state["lineage"] if x["kind"] == "replacement"]
    assert len(events) == 2 and all(x["curriculum_child"] != x["curriculum_parent"] for x in events)
    assert run.state["members"][5]["curriculum"] == events[0]["curriculum_child"]
    with pytest.raises(ValueError):
        PopulationConfig(mode="multistart", curriculum_mutation=True, world_mix=mix)
    with pytest.raises(ValueError):
        PopulationRun(cfg, tmp_path / "missing", factory, _teacher)


def test_separate_development_mixture_only_changes_selection_panel(tmp_path):
    from topoformer.campaign02_population import PopulationConfig, PopulationRun
    torch.set_num_threads(1)
    dev_seen, train_seen = [], []
    def train_factory(seed):
        train_seen.append(seed)
        return _factory(seed)
    def dev_factory(seed):
        dev_seen.append(seed)
        return Workshop(generate_world(seed, categories=1, choices=3, locations=3, work_limit=0), address_seed=seed + 17)
    cfg = PopulationConfig(mode="pbt", rounds=1, updates_per_slot=1, development_examples=2, teacher="cheap",
        development_world_mix=({"categories": 1, "choices": 3, "locations": 3, "work_limit": 0},),
        train={"width": 8, "device": "cpu", "batch_size": 1, "max_steps": 4, "evaluation_batch": 2},
        world_mix=({"categories": 1, "choices": 2, "locations": 3},))
    with pytest.raises(ValueError):
        PopulationRun(cfg, tmp_path / "missing", train_factory, _teacher)
    run = PopulationRun(cfg, tmp_path / "dev", train_factory, _teacher, development_factory=dev_factory)
    run._run_slot()
    assert set(dev_seen) == {cfg.development_seed_start, cfg.development_seed_start + 1}
    assert not set(train_seen) & set(dev_seen)
