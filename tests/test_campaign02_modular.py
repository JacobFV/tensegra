"""Modular staged workshop: semantics, public-information boundary, references, integration (CPU)."""
from dataclasses import replace
from functools import partial
import itertools

import pytest
import torch

from tensegra.campaign02_modular import (ModularReference, ModularWorkshop, STAGES, action_catalog,
    encode_action, encode_observation, generate_modular, modular_executor, validate_assignment)
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_world import Action

EXEC = partial(modular_executor, execute_call=execute)
ORDERS = [p for n in (1, 2, 3) for p in itertools.permutations(STAGES, n)]


def run(spec, mode, limit=80):
    env, ref = ModularWorkshop(spec, executor=EXEC, address_seed=3), ModularReference(mode)
    o = env.observe()
    while not o.done and limit:
        o = env.step(ref.choose(o)); limit -= 1
    return env.evaluate()


@pytest.mark.parametrize("stages", ORDERS)
def test_teacher_solves_every_order_and_audits_reductions(stages):
    outcomes = [run(generate_modular(s, stages=stages, categories=3, choices=3, tasks=4, slots=3, step_limit=80), "cheap_first")
                for s in range(12)]
    assert sum(o["verified_success"] for o in outcomes) >= 11
    assert all(r["correct"] for o in outcomes for r in o["reductions"])


def test_leverage_exists_cheap_fails_sometimes_and_tool_budget_matters():
    specs = [generate_modular(s, stages=("select", "assign"), categories=4, choices=4, tasks=6, slots=4, forbid=.35, step_limit=80)
             for s in range(40)]
    cheap = sum(run(x, "cheap")["verified_success"] for x in specs)
    first = sum(run(x, "cheap_first")["verified_success"] for x in specs)
    # Hard assign region: greedy 9/40 vs cheap-first 32/40 measured; failures are budget-bounded CSP timeouts.
    assert cheap < first and first >= 28
    assert all(validate_assignment(x, [0]*len(x.domains))[0] in (True, False) for x in specs)


def test_stage_gate_roster_privacy_and_verify():
    spec = generate_modular(1, stages=("assign", "select"))
    env = ModularWorkshop(spec, executor=EXEC)
    o = env.observe()
    assert o.roster is None and not any(a.kind == "choose_slot" for a in action_catalog(o))
    for r in o.item_inventory:
        env.step(Action("inspect", {"target": r["handle"]}))
    o = env.step(Action("commit_pending"))
    assert o.feedback["reason"] == "stage_order"
    o = env.step(Action("verify"))
    assert not o.done and o.feedback["status"] == "incomplete"
    o = env.step(Action("inspect", {"target": "roster"}))
    assert o.roster is not None and any(a.kind == "choose_slot" for a in action_catalog(o))


def test_encoders_rename_invariant_and_stage_marked():
    spec = generate_modular(5, stages=("route", "select"))
    names = {x.handle: f"i{700000+i}" for i, x in enumerate(spec.items)}
    renamed = replace(spec, items=tuple(replace(x, handle=names[x.handle]) for x in spec.items),
                      incompatible=tuple((names[a], names[b]) for a, b in spec.incompatible))
    rows = []
    for s in (spec, renamed):
        env = ModularWorkshop(s, executor=EXEC, address_seed=9)
        for r in env.observe().item_inventory:
            env.step(Action("inspect", {"target": r["handle"]}))
        env.step(Action("inspect", {"target": "map"}))
        o = env.observe()
        rows.append((encode_observation(o), [encode_action(o, a) for a in action_catalog(o)]))
    assert rows[0] == rows[1]
    assert len({len(r) for r in rows[0][1]}) == 1


def test_training_and_population_integration(tmp_path):
    from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
    from tensegra.campaign02_training import collect_teacher, public_frame
    from tensegra.campaign02_population import PopulationConfig, PopulationRun
    torch.set_num_threads(1)
    spec = generate_modular(2, stages=("select", "route"), categories=2, choices=2, locations=5, step_limit=40)
    _, obs, cand = public_frame(ModularWorkshop(spec).observe(), "m1")
    model = CandidatePolicy(PolicyConfig(len(obs), len(cand[0]), width=8, feature_version="m1"))
    frames, result = collect_teacher(ModularWorkshop(spec, executor=EXEC), ModularReference(), 40, model)
    assert frames and result["outcome"]["verified_success"]
    with pytest.raises(ValueError):
        public_frame(ModularWorkshop(spec).observe(), "v2")
    mix = ({"stages": ["select", "route"], "categories": 2, "choices": 2, "locations": 5, "step_limit": 40},)
    def factory(seed):
        return ModularWorkshop(generate_modular(seed, **mix[0]), executor=EXEC, address_seed=seed)
    cfg = PopulationConfig(mode="single", rounds=1, updates_per_slot=1, development_examples=2, teacher="modular_cheap_first",
        world_family="modular", policy={"feature_version": "m1"},
        train={"width": 8, "device": "cpu", "batch_size": 1, "max_steps": 40, "evaluation_batch": 2}, world_mix=mix)
    from tensegra.campaign02_references import make_reference
    run_ = PopulationRun(cfg, tmp_path / "m", factory, lambda: make_reference("modular_cheap_first"))
    run_.run()
    assert run_.state["status"] == "completed"


def test_distractor_returns_are_public_wrong_typed_and_harmless_to_teacher():
    specs = [generate_modular(s, stages=("select", "assign"), distractors=2) for s in range(30)]
    assert any(x.distractors for x in specs)
    assert all(p == "shortest_path" for x in specs for p, _, _ in x.distractors)
    spec = next(x for x in specs if x.distractors)
    env = ModularWorkshop(spec, executor=EXEC, address_seed=4)
    o = env.observe()
    assert len(o.records) == len(spec.distractors) and not o.problems
    rec = o.records[0]["handle"]
    env.step(Action("retrieve", {"handle": rec}))
    o = env.step(Action("use_return", {"handle": rec, "as": "select"}))
    assert o.feedback["status"] == "invalid_input" and "mismatch" in o.feedback["reason"]
    assert sum(run(x, "cheap_first")["verified_success"] for x in specs) >= 29
    solo = [generate_modular(s, stages=("select",), distractors=2) for s in range(30)]
    assert {p for x in solo for p, _, _ in x.distractors} <= {"csp", "shortest_path"}
    assert generate_modular(3, stages=("select",)).distractors == ()


def test_same_type_distractors_and_m2_provenance():
    from tensegra.campaign02_modular import encode_action_m2
    specs = [generate_modular(s, stages=("select", "assign"), same_type_distractors=2) for s in range(30)]
    kinds = {p for x in specs for p, _, _ in x.distractors}
    assert kinds and kinds <= {"constrained_subset", "csp"}
    assert generate_modular(4, stages=("select",)).distractors == generate_modular(4, stages=("select",), same_type_distractors=0).distractors
    spec = next(x for x in specs if any(p == "csp" for p, _, _ in x.distractors))
    env = ModularWorkshop(spec, executor=EXEC, address_seed=2)
    prior = env.observe().records[0]["handle"]
    ref = ModularReference("always_tool")
    o = env.observe()
    while not o.done and not any(r.get("problem") in o.problems for r in o.records):
        o = env.step(ref.choose(o))
    own = next(r["handle"] for r in o.records if r.get("problem") in o.problems)
    f_prior = encode_action_m2(o, Action("retrieve", {"handle": prior}))
    f_own = encode_action_m2(o, Action("retrieve", {"handle": own}))
    assert f_prior[-2:] == [0.0, 0.0] and f_own[-2:] == [1.0, 1.0]
    assert encode_action_m2(o, Action("verify"))[-3:] == [0.0, 0.0, 0.0]
    assert sum(run(x, "cheap_first")["verified_success"] for x in specs) >= 29


def test_m3_stage_failure_counters_are_public_and_reset():
    from tensegra.campaign02_modular import encode_action_m3, encode_observation_m3
    spec = generate_modular(7, stages=("select", "route"))
    env = ModularWorkshop(spec, executor=EXEC)
    for r in env.observe().item_inventory:
        env.step(Action("inspect", {"target": r["handle"]}))
    o = env.step(Action("commit_pending"))          # nothing pending -> rejected
    o = env.step(Action("commit_pending"))
    o = env.step(Action("think"))                   # rejection no longer the last feedback
    assert o.stage_rejections == 2 and o.feedback["status"] == "success"
    assert encode_observation_m3(o)[-4:-2] == [2/8, 1.0]
    assert encode_action_m3(o, Action("commit_pending"))[-3] == 2/8
    ref = ModularReference()
    while o.current_stage == "select" and not o.done:
        o = env.step(ref.choose(o))
    assert o.stage_rejections == 0 and o.stage_solver_calls == 0
    assert "stage_rejections" in o.to_dict()
