"""depworld-v1 (extended-03 Stage A): semantics, public boundary, applicability, encoders, references (CPU)."""
from dataclasses import replace
from functools import partial
import itertools
import json
import math
import random

import pytest
import torch

from topoformer.campaign02_protocol import execute
from topoformer.campaign02_world import Action
from topoformer.campaign03_depworld import (ATTEMPT_BLOCK, EVENT_KINDS, FOREIGN_KINDS, REASONS, REFERENCE_MODES,
    DepItem, DepReference, DepSpec, DepWorkshop, action_catalog, action_key, applicable, audit_record,
    current_request, depworld_executor, encode_action_d1, encode_observation_d1, encode_public, generate_depworld,
    overlaps, relevant_dependencies, selection_id, PROGRESS_KINDS)

EXEC = partial(depworld_executor, execute_call=execute)
DEV = 2_000_000_000  # development seeds only


def run(spec, mode, address_seed=0, limit=200):
    env, ref = DepWorkshop(spec, executor=EXEC, address_seed=address_seed), DepReference(mode)
    o = env.observe()
    while not o.done and limit:
        o = env.step(ref.choose(o))
        limit -= 1
    return env.evaluate()


def inspect_all(env):
    o = env.observe()
    env.step(Action("inspect", {"target": "requirements"}))
    for r in o.item_inventory:
        env.step(Action("inspect", {"target": r["handle"]}))
    return env.step(Action("inspect", {"target": "map"}))


# --- a tiny hand-built world for exact semantics ------------------------------------------

def tiny(**over):
    items = (DepItem("A1", 0, 4, 2, 2, (0, 1, 2)), DepItem("A2", 0, 1, 6, 1, (0, 3)),
             DepItem("B1", 1, 3, 2, 2, (0, 2)), DepItem("B2", 1, 1, 1, 3, (1, 4)))
    params = dict(items=items, categories=(0, 1), slots=5, capacity=6, funds=7, incompatible=(("A2", "B2"),),
                  deadline=10, edges=((0, 1, 2), (0, 2, 5), (1, 2, 2)), locations=3, destination=2)
    params.update(over)
    return DepSpec(**params)


def commit(env, handles):
    for h in handles:
        env.step(Action("choose_item", {"item": h}))
    return env.step(Action("commit_pending"))


# --- generator: planted feasibility before and after the event ------------------------------

def _shortest(edges, n, start, goal):
    best = {start: 0}
    for u in range(n):  # generated maps are forward DAGs
        if u in best:
            for a, b, w in edges:
                if a == u and (b not in best or best[u] + w < best[b]):
                    best[b] = best[u] + w
    return best.get(goal)


def _feasible(spec, capacity, deadline, closed, edges):
    dist = _shortest(edges, spec.locations, spec.start, spec.destination)
    if dist is None:
        return False
    groups = [[x for x in spec.items if x.category == c] for c in spec.categories]
    bad = {frozenset(p) for p in spec.incompatible}
    for sel in itertools.product(*groups):
        if sum(x.weight for x in sel) > capacity or sum(x.price for x in sel) > spec.funds:
            continue
        if any(frozenset((a.handle, b.handle)) in bad for a, b in itertools.combinations(sel, 2)):
            continue
        domains = [[s for s in x.slots if s not in closed] for x in sel]
        for starts in itertools.product(*domains):
            if any(overlaps(s, a.duration, t, b.duration) for (s, a), (t, b) in
                   itertools.combinations(list(zip(starts, sel)), 2)):
                continue
            if max(s + x.duration for s, x in zip(starts, sel)) + dist <= deadline:
                return True
    return False


def _post_event(spec):
    capacity, deadline, closed, edges = spec.capacity, spec.deadline, set(spec.closed_slots), list(spec.edges)
    if spec.event:
        kind, _, arg = spec.event
        if kind == "capacity_reduced":
            capacity = arg
        elif kind == "deadline_moved":
            deadline = arg
        elif kind == "slot_closed":
            closed |= {arg}
        else:
            edges = [e for e in edges if (e[0], e[1]) != tuple(arg)]
    return capacity, deadline, closed, edges


@pytest.mark.parametrize("trigger", ["step", "progress"])
@pytest.mark.parametrize("size", [dict(categories=3, choices=2, locations=6, slots=5),
                                  dict(categories=3, choices=3, locations=7, slots=6)])
def test_planted_feasible_before_and_after_every_event_kind(size, trigger):
    kinds = set()
    for seed in range(DEV, DEV + 60):
        spec = generate_depworld(seed, p_event=1.0, foreign_records=2, event_trigger=trigger, **size)
        assert spec.event_trigger == trigger
        if trigger == "progress":
            assert spec.event[1] in (1, 2) and spec.event[0] in PROGRESS_KINDS[spec.event[1]]
        pre = (spec.capacity, spec.deadline, set(spec.closed_slots), list(spec.edges))
        assert _feasible(spec, *pre), seed
        assert spec.event is not None
        kinds.add(spec.event[0])
        post = _post_event(spec)
        assert _feasible(spec, *post), (seed, spec.event)
        # the planted plan itself is valid after the event (evaluator-only field)
        chosen, starts, route = spec.planted
        by = {x.handle: x for x in spec.items}
        cap, dl, closed, edges = post
        assert sum(by[h].weight for h in chosen) <= cap and all(s not in closed for _, s in starts)
        w = {(a, b): c for a, b, c in edges}
        finish = max(s + by[h].duration for h, s in starts)
        assert finish + sum(w[(a, b)] for a, b in zip(route, route[1:])) <= dl
        # events only tighten requirements
        assert cap <= spec.capacity and dl <= spec.deadline
    assert kinds == set(EVENT_KINDS)
    assert generate_depworld(DEV, p_event=0.0).event is None


def test_generation_is_deterministic_and_json_safe():
    a = generate_depworld(DEV + 5, p_event=1, foreign_records=4)
    b = generate_depworld(DEV + 5, p_event=1, foreign_records=4)
    assert a == b
    from dataclasses import asdict
    json.dumps(asdict(a))


# --- public information boundary ---------------------------------------------------------

def test_no_hidden_information_before_inspection():
    spec = generate_depworld(DEV + 1, p_event=1.0)
    o = DepWorkshop(spec, executor=EXEC).observe()
    assert o.requirements is None and o.known_items == {} and o.known_edges is None
    assert o.records == () and o.problems == {} and o.events == () and o.attempts == ()
    assert set(o.item_inventory[0]) == {"handle", "category"}
    text = json.dumps(o.to_dict())
    assert "planted" not in text and "label" not in text and "payload" not in text
    # Worlds differing only in hidden facts encode identically before inspection.
    other = replace(spec, capacity=spec.capacity + 7, funds=spec.funds + 3, deadline=spec.deadline + 5,
                    items=tuple(replace(x, weight=x.weight % 7 + 1, duration=x.duration % 4 + 1) for x in spec.items),
                    edges=tuple((a, b, w + 1) for a, b, w in spec.edges), event=None)
    enc = [encode_public(DepWorkshop(s, executor=EXEC).observe(), action_catalog(DepWorkshop(s).observe()))
           for s in (spec, other)]
    assert enc[0] == enc[1]


def test_foreign_records_hide_labels_and_payloads_until_retrieve():
    spec = generate_depworld(DEV + 2, foreign_records=4)
    env = DepWorkshop(spec, executor=EXEC, address_seed=3)
    o = env.observe()
    assert len(o.records) == 4 and len(o.problems) == 4
    for r in o.records:
        assert "payload" not in r and "_label" not in r and "label" not in r
        assert r["problem"] in o.problems and o.problems[r["problem"]]["created_step"] == 0
    h = o.records[0]["handle"]
    o = env.step(Action("retrieve", {"handle": h}))
    assert "payload" in o.retrieved[h] and "_label" not in o.retrieved[h]
    assert "_label" not in json.dumps(o.to_dict())


# --- stage/dependency gating, revocation, rejection reasons ---------------------------------

def test_dependency_gating_uncommit_and_revocation():
    env = DepWorkshop(tiny(), executor=EXEC)
    inspect_all(env)
    o = env.step(Action("start_assign", {"handle": "problem_0"}))
    assert o.feedback == {"status": "rejected", "reason": "missing_dependency"}
    o = env.step(Action("move", {"destination": 1}))
    assert o.feedback["reason"] == "missing_dependency"
    assert not any(a.kind in ("start_assign", "build_route", "choose_slot") for a in action_catalog(o))
    o = commit(env, ["A1", "B2"])
    assert o.selection_id == selection_id(["A1", "B2"]) and o.selected == ("A1", "B2")
    assert any(a.kind == "start_assign" for a in action_catalog(o))
    env.step(Action("choose_slot", {"item": "A1", "slot": 0}))
    env.step(Action("choose_slot", {"item": "B2", "slot": 4}))
    o = env.step(Action("commit_assignment"))
    assert o.assignment_id and o.finish_time == 7
    assert any(a.kind == "build_route" for a in action_catalog(o))
    o = env.step(Action("uncommit", {"target": "select"}))
    assert o.selection_id is None and o.assignment is None and o.assignment_id is None
    o = env.step(Action("uncommit", {"target": "assign"}))
    assert o.feedback["reason"] == "missing_dependency"


@pytest.mark.parametrize("kind,arg,revoked,affected", [
    ("slot_closed", 4, ["assignment"], "slots"),
    ("capacity_reduced", 4, ["selection", "assignment"], "capacity"),
    ("deadline_moved", 9, [], "deadline"),
    ("edge_closed", (0, 1), [], "map")])
def test_events_are_public_versioned_and_revoke(kind, arg, revoked, affected):
    # inspections (1 + 4 + 1) + choose x2 + commit + choose_slot x2 + commit_assignment = 12
    env = DepWorkshop(tiny(event=(kind, 12, arg)), executor=EXEC)
    inspect_all(env)
    commit(env, ["A1", "B2"])
    env.step(Action("choose_slot", {"item": "A1", "slot": 0}))
    env.step(Action("choose_slot", {"item": "B2", "slot": 4}))
    o = env.step(Action("commit_assignment"))
    assert o.feedback["status"] == "success" and o.feedback["event"]["kind"] == kind
    assert o.requirements_version == 2 and o.requirement_versions[affected] == 2
    assert sum(v == 2 for v in o.requirement_versions.values()) == 1
    ev = o.events[-1]
    assert ev["revoked"] == revoked and ev["affected"] == affected and ev["step"] == 12
    assert (o.assignment is None) == ("assignment" in revoked)
    assert (o.selection_id is None) == ("selection" in revoked)
    if kind == "slot_closed":
        assert 4 in o.requirements["closed_slots"]
    if kind == "edge_closed":
        assert (0, 1) not in {(a, b) for a, b, _ in o.known_edges}
    if kind == "deadline_moved":
        assert o.requirements["deadline"] == 9
    if kind == "capacity_reduced":
        assert o.requirements["capacity"] == 4
    assert env.evaluate()["revocations"] == revoked


def test_rejection_reasons_are_declared_and_specific():
    env = DepWorkshop(tiny(), executor=EXEC)
    o = commit(env, ["A1", "B2"])
    assert o.feedback["reason"] == "missing_dependency"  # requirements not inspected
    inspect_all(env)
    seen = {}
    for handles, reason in ((["A1", "B1"], "capacity"), (["A2", "B1"], "funds"), (["A2", "B2"], "incompatibility")):
        env.step(Action("choose_item", {"item": handles[0]}))
        env.step(Action("choose_item", {"item": handles[1]}))
        seen[reason] = env.step(Action("commit_pending")).feedback["reason"]
    assert all(k == v for k, v in seen.items())
    assert commit(env, ["A1", "B2"]).feedback["status"] == "success"
    assert env.step(Action("commit_pending")).feedback["reason"] == "already_committed"
    env.step(Action("choose_slot", {"item": "A1", "slot": 0}))
    assert env.step(Action("commit_assignment")).feedback["reason"] == "incomplete"
    env.step(Action("choose_slot", {"item": "B2", "slot": 1}))
    assert env.step(Action("commit_assignment")).feedback["reason"] == "slot_conflict"
    env.step(Action("choose_slot", {"item": "A1", "slot": 3}))
    assert env.step(Action("commit_assignment")).feedback["reason"] == "slot_unavailable"
    env.step(Action("choose_slot", {"item": "A1", "slot": 0}))
    env.step(Action("choose_slot", {"item": "B2", "slot": 4}))
    assert env.step(Action("commit_assignment")).feedback["status"] == "success"
    assert env.step(Action("move", {"destination": 0})).feedback["reason"] == "missing_edge"
    assert env.step(Action("move", {"destination": 2})).feedback["reason"] == "deadline"  # 7 + 5 > 10
    # solver records: type mismatch, not applicable, retrieve required, stale dependency
    env.step(Action("start_assign", {"handle": "problem_0"}))
    env.step(Action("add_constraint", {"problem": "problem_0", "constraint": "conflicts"}))
    env.step(Action("add_constraint", {"problem": "problem_0", "constraint": "finish_by", "bound": 1}))
    o = env.step(Action("call", {"problem": "problem_0", "budget": 128}))
    infeasible = o.feedback["return"]
    assert o.feedback["status"] == "infeasible"
    env.step(Action("build_route", {"handle": "problem_1"}))
    o = env.step(Action("call", {"problem": "problem_1", "budget": 128}))
    route = o.feedback["return"]
    assert env.step(Action("use_return", {"handle": route, "as": "route"})).feedback["reason"] == "retrieve_required"
    env.step(Action("retrieve", {"handle": route}))
    assert env.step(Action("use_return", {"handle": route, "as": "assign"})).feedback["reason"] == "type_mismatch"
    assert env.step(Action("use_return", {"handle": infeasible, "as": "assign"})).feedback["reason"] == "not_applicable"
    assert env.step(Action("move", {"destination": 1})).feedback["status"] == "success"  # 7 + 2 <= 10
    assert env.step(Action("use_return", {"handle": route, "as": "route"})).feedback["reason"] == "stale_dependency"
    o = env.observe()
    assert all(a["reason"] in REASONS or a["reason"] is None for a in o.attempts)
    reasons = {a["reason"] for a in o.attempts}
    assert {"capacity", "funds", "incompatibility", "slot_conflict", "slot_unavailable", "missing_edge", "deadline",
            "type_mismatch", "not_applicable", "retrieve_required", "stale_dependency", "already_committed",
            "incomplete", "missing_dependency"} <= reasons
    ok = env.step(Action("verify"))
    assert ok.feedback["status"] == "incomplete" and ok.feedback["reason"] == "incomplete"


# --- attempted-action record --------------------------------------------------------------

def _attempt_block(vec):
    return vec[-ATTEMPT_BLOCK:]


def test_attempt_record_and_encoded_retry_features():
    env = DepWorkshop(tiny(), executor=EXEC)
    inspect_all(env)
    env.step(Action("choose_item", {"item": "A1"}))
    env.step(Action("choose_item", {"item": "B1"}))
    before = env.observe()
    o = env.step(Action("commit_pending"))
    at = o.attempts[-1]
    assert at["action_kind"] == "commit_pending" and at["action_key"] == action_key(Action("commit_pending"))
    assert at["outcome_status"] == "rejected" and at["reason"] == "capacity"
    assert at["dependency_versions_at_attempt"] == relevant_dependencies(before, Action("commit_pending"))
    blk = _attempt_block(encode_action_d1(o, Action("commit_pending")))
    assert blk[0] == 1.0 and blk[-1] == 0.0                           # attempted, deps unchanged
    assert blk[2 + 7 + 1 + REASONS.index("capacity")] == 1.0            # last reason one-hot
    assert blk[2 + 1] == 1.0                                           # last outcome = rejected
    o = env.step(Action("think"))                                      # history survives other actions
    assert _attempt_block(encode_action_d1(o, Action("commit_pending"))) == blk
    o = env.step(Action("choose_item", {"item": "B2"}))                # pending changed -> legit retry
    blk2 = _attempt_block(encode_action_d1(o, Action("commit_pending")))
    assert blk2[0] == 1.0 and blk2[-1] == 1.0
    # never-attempted candidates and non-attempt kinds carry a zero block
    assert set(_attempt_block(encode_action_d1(o, Action("uncommit", {"target": "select"})))) == {0.0}
    assert set(_attempt_block(encode_action_d1(o, Action("think")))) == {0.0}
    # the record logs calls/moves/uses as well, not inspections/drafts
    kinds = {a["action_kind"] for a in o.attempts}
    assert kinds == {"commit_pending"}


# --- applicability rule vs evaluator audit ---------------------------------------------------

def test_public_applicability_agrees_with_hidden_audit_on_random_states():
    rng = random.Random(0)
    agree = checked = sound = applicable_seen = inapplicable_seen = 0
    for seed in range(DEV + 100, DEV + 140):
        spec = generate_depworld(seed, p_event=1.0, foreign_records=4)
        env = DepWorkshop(spec, executor=EXEC, address_seed=seed)
        ref = DepReference(rng.choice(("reuse", "naive_reuse", "recompute")))
        o = env.observe()
        for _ in range(70):
            if o.done:
                break
            complete = o.requirements is not None and len(o.known_items) == len(o.item_inventory) and o.known_edges
            for r in o.records:
                for p in ("constrained_subset", "csp", "shortest_path"):
                    pub = applicable(o, r, p)
                    hid = audit_record(env, r["handle"], p)
                    sound += 1
                    assert not pub or hid["applicable_hidden"], (seed, r["handle"], p)
                    if pub and r["status"] == "success":
                        assert hid["payload_valid_now"], (seed, r["handle"], p)
                    if complete:
                        checked += 1
                        agree += pub == hid["applicable_hidden"]
                        applicable_seen += pub
                        inapplicable_seen += not pub and r["primitive"] == p
            acts = [a for a in action_catalog(o) if a.kind != "abstain"]
            action = rng.choice(acts) if rng.random() < .3 else ref.choose(o)
            o = env.step(action)
    assert checked > 2000 and agree == checked
    assert applicable_seen > 100 and inapplicable_seen > 100 and sound >= checked


def test_foreign_registered_records_mix_applicable_and_not():
    by_label = {k: [0, 0] for k in FOREIGN_KINDS}
    for seed in range(DEV + 200, DEV + 260):
        spec = generate_depworld(seed, foreign_records=4)
        env = DepWorkshop(spec, executor=EXEC, address_seed=seed)
        o = inspect_all(env)
        labels = [f["label"] for f in spec.foreign]
        for r, label in zip(o.records, labels):
            assert r["problem"] in o.problems  # registered: registry membership never separates
            if label.startswith("csp"):
                continue
            by_label[label][applicable(o, r)] += 1
        # csp records become applicable once their selection is committed
        for r, label, f in zip(o.records, labels, spec.foreign):
            if label == "csp_current" and f["status"] == "success":
                env2 = DepWorkshop(spec, executor=EXEC, address_seed=seed)
                inspect_all(env2)
                o2 = commit(env2, f["snapshot"]["problem"]["items"])
                if o2.selection_id:
                    rec = next(x for x in o2.records if x["handle"] == r["handle"])
                    by_label[label][applicable(o2, rec)] += 1
                    other = [x for x, lb in zip(o2.records, labels) if lb == "csp_other_selection"]
                    for x in other:
                        by_label["csp_other_selection"][applicable(o2, x)] += 1
    assert by_label["subset_current"][1] > 0 and by_label["subset_current"][0] == 0
    assert by_label["route_current"][1] > 0 and by_label["route_current"][0] == 0
    assert by_label["csp_current"][1] > 0
    for label in ("subset_earlier", "subset_version_mismatch", "route_other_goal", "route_earlier"):
        assert by_label[label][1] == 0 and by_label[label][0] > 0, label


def test_version_mismatch_record_differs_only_by_dependency_relation():
    spec = next(s for s in (generate_depworld(DEV + i, foreign_records=4) for i in range(300, 400))
                if any(f["label"] == "subset_version_mismatch" for f in s.foreign))
    env = DepWorkshop(spec, executor=EXEC)
    o = inspect_all(env)
    i = next(k for k, f in enumerate(spec.foreign) if f["label"] == "subset_version_mismatch")
    rec = o.records[i]
    assert rec["problem_snapshot"] == current_request(o, "constrained_subset")
    assert not applicable(o, rec)
    vec = encode_action_d1(o, Action("retrieve", {"handle": rec["handle"]}))
    base = 17 + 3 + 3 + 3
    assert vec[base:base + 8] == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]  # request match, no dependency match


# --- encoders ------------------------------------------------------------------------------

def _rename(spec):
    names = {x.handle: f"q{900000 + i}" for i, x in enumerate(spec.items)}

    def swap(v):
        if isinstance(v, str):
            return names.get(v, v)
        if isinstance(v, list):
            return [swap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(swap(x) for x in v)
        if isinstance(v, dict):
            return {k: swap(x) for k, x in v.items()}
        return v
    foreign = []
    for f in spec.foreign:
        g = swap(dict(f))
        if f["primitive"] == "csp":
            g["depends_on"]["selection_id"] = selection_id(g["snapshot"]["problem"]["items"])
        foreign.append(g)
    return replace(spec, items=tuple(replace(x, handle=names[x.handle]) for x in spec.items),
                   incompatible=swap(spec.incompatible), planted=swap(spec.planted), foreign=tuple(foreign))


@pytest.mark.parametrize("seed", [DEV + 11, DEV + 12, DEV + 13])
def test_d1_encoders_are_rename_invariant_along_trajectories(seed):
    spec = generate_depworld(seed, p_event=1.0, foreign_records=4)
    renamed = _rename(spec)
    runs = []
    for s in (spec, renamed):
        env, ref, rows = DepWorkshop(s, executor=EXEC, address_seed=5), DepReference("reuse"), []
        o = env.observe()
        while not o.done:
            acts = action_catalog(o)
            rows.append(encode_public(o, acts))
            o = env.step(ref.choose(o))
        runs.append((rows, env.evaluate()["verified_success"]))
    assert runs[0] == runs[1]


def test_d1_dimensions_are_stable_and_finite():
    rng = random.Random(1)
    obs_dims, cand_dims = set(), set()
    for seed in range(DEV + 500, DEV + 512):
        spec = generate_depworld(seed, categories=rng.choice((2, 3, 4)), choices=rng.choice((2, 3)),
                                 p_event=1.0, foreign_records=rng.choice((0, 4)))
        env = DepWorkshop(spec, executor=EXEC)
        o = env.observe()
        for _ in range(60):
            if o.done:
                break
            acts = action_catalog(o)
            ob, cands = encode_public(o, acts)
            obs_dims.add(len(ob))
            cand_dims |= {len(c) for c in cands}
            assert all(math.isfinite(x) for x in ob) and all(math.isfinite(x) for c in cands for x in c)
            o = env.step(rng.choice([a for a in acts if a.kind != "abstain"]))
    assert len(obs_dims) == 1 and len(cand_dims) == 1
    with pytest.raises(ValueError):
        encode_public(o, [], "m1")


def test_encoder_exposes_relations_not_an_applicability_bit():
    spec = generate_depworld(DEV + 20, foreign_records=4)
    env = DepWorkshop(spec, executor=EXEC)
    o = inspect_all(env)
    vecs = {r["handle"]: encode_action_d1(o, Action("retrieve", {"handle": r["handle"]})) for r in o.records}
    base = 17 + 3 + 3 + 3
    for r in o.records:
        v = vecs[r["handle"]]
        rel = v[base + 1:base + 8]  # type, request, canonical, dependency, requirements, selection, usable
        assert (rel[0] and rel[1] and rel[3] and rel[6]) == applicable(o, r)
    # the catalogue is syntactic: no applicability mask, all records offered for all uses
    uses = [a for a in action_catalog(o) if a.kind == "use_return"]
    assert len(uses) == 3 * len(o.records)


# --- references ------------------------------------------------------------------------------

def test_references_are_public_and_in_catalogue():
    spec = generate_depworld(DEV + 30, p_event=1.0, foreign_records=4)
    for mode in REFERENCE_MODES:
        env, ref = DepWorkshop(spec, executor=EXEC), DepReference(mode)
        o = env.observe()
        while not o.done:
            a = ref.choose(o)
            assert a in action_catalog(o), (mode, a)
            o = env.step(a)
    with pytest.raises(ValueError):
        DepReference("oracle")
    from topoformer.campaign02_references import make_reference
    assert make_reference("dep_reuse").reference_name == "dep_reuse"
    assert make_reference("modular_cheap_first").reference_name == "modular_cheap_first"


def test_reference_leverage_profile_small():
    specs = [generate_depworld(DEV + 1000 + i, p_event=0.5, foreign_records=4) for i in range(40)]
    out = {m: [run(s, m, i) for i, s in enumerate(specs)] for m in REFERENCE_MODES}
    succ = {m: sum(r["verified_success"] for r in rs) / len(rs) for m, rs in out.items()}
    cost = {m: sum(r["cost"] for r in rs) / len(rs) for m, rs in out.items()}
    assert succ["reuse"] >= .9 and succ["reuse"] >= succ["recompute"] - .01
    assert cost["reuse"] < .9 * cost["recompute"]
    assert succ["greedy"] < succ["reuse"] - .3
    assert succ["naive_reuse"] < succ["reuse"] - .05
    assert succ["reuse_norevise"] < succ["reuse"]
    assert sum(r["reuse"]["foreign_applicable_uses"] for r in out["reuse"]) > 0
    assert sum(r["reuse"]["invalid_uses"] for r in out["reuse"]) == 0
    assert sum(r["reuse"]["invalid_uses"] for r in out["naive_reuse"]) > 0
    assert all(r["correct"] for r in out["recompute"][0]["reductions"])


# --- executor contract -------------------------------------------------------------------

def test_executor_contract_all_primitives():
    spec = generate_depworld(DEV + 40)
    env = DepWorkshop(spec, executor=EXEC)
    o = inspect_all(env)
    sub = current_request(o, "constrained_subset")
    route = current_request(o, "shortest_path")
    res = depworld_executor("constrained_subset", sub["problem"], 1024, execute_call=execute)
    assert res["status"] == "success" and res["certificate_valid"] and res["work_units"] <= 1024
    tiny_budget = depworld_executor("constrained_subset", sub["problem"], 2, execute_call=execute)
    assert tiny_budget["status"] == "timeout" and tiny_budget["work_units"] <= 2
    res = depworld_executor("shortest_path", route["problem"], 1024, execute_call=execute)
    assert res["status"] == "success" and res["payload"][0][0] == 0 and res["certificate_valid"]
    chosen = [sub["problem"]["handles"][i] for i in depworld_executor(
        "constrained_subset", sub["problem"], 1024, execute_call=execute)["payload"][0]]
    o = commit(env, chosen)
    csp = current_request(o, "csp")
    res = depworld_executor("csp", csp["problem"], 4096, execute_call=execute)
    assert res["status"] in ("success", "infeasible")
    if res["status"] == "success":
        assert res["certificate_valid"] and len(res["payload"]) == len(csp["problem"]["items"])
    bad = depworld_executor("csp", {"domains": [[0]], "forbidden": [[0, 5, 0, 0]]}, 16, execute_call=execute)
    assert bad["status"] == "invalid" and not bad["certificate_valid"]
    isolated = depworld_executor("csp", {"domains": [[0, 1], [1]], "forbidden": [[0, 1, 1, 1]]}, 64)
    assert isolated["status"] == "success" and isolated["payload"] == (0, 1)
    # the environment enforces the work contract and charges work
    env2 = DepWorkshop(spec, executor=lambda *a: {"status": "success", "payload": None, "work_units": 10 ** 9})
    inspect_all(env2)
    env2.step(Action("start_subset", {"handle": "problem_0"}))
    with pytest.raises(RuntimeError):
        env2.step(Action("call", {"problem": "problem_0", "budget": 16}))


def test_utility_accounting_matches_costs():
    spec = generate_depworld(DEV + 50, foreign_records=2)
    r = run(spec, "reuse")
    expected = (r["steps"] * spec.action_price + r["observations"] * spec.observation_price
                + r["travel_distance"] * spec.travel_price + r["work_units"] * spec.work_price)
    assert abs(r["cost"] - expected) < 1e-12 and abs(r["utility"] - (float(r["verified_success"]) - expected)) < 1e-12


# --- integration ---------------------------------------------------------------------------

def test_training_integration_collect_teacher_d1():
    from topoformer.campaign02_policy import CandidatePolicy, PolicyConfig
    from topoformer.campaign02_training import collect_teacher, public_frame, supervised_loss
    torch.set_num_threads(1)
    spec = generate_depworld(DEV + 60, p_event=1.0, foreign_records=2)
    actions, obs, cand = public_frame(DepWorkshop(spec).observe(), "d1")
    assert len(cand) == len(actions)
    with pytest.raises(ValueError):
        public_frame(DepWorkshop(spec).observe(), "m1")
    model = CandidatePolicy(PolicyConfig(len(obs), len(cand[0]), width=8, feature_version="d1"))
    frames, result = collect_teacher(DepWorkshop(spec, executor=EXEC), DepReference("reuse"), 96, model)
    assert frames and result["outcome"]["verified_success"] and not result["truncated"]
    loss, count = supervised_loss(model, [frames])
    assert count == len(frames) and torch.isfinite(loss)


def test_population_integration_depworld(tmp_path):
    from topoformer.campaign02_population import PopulationConfig, PopulationRun
    from topoformer.campaign02_references import make_reference
    torch.set_num_threads(1)
    mix = ({"categories": 2, "choices": 2, "locations": 5, "slots": 5, "p_event": 0.5, "foreign_records": 2},)

    def factory(seed):
        return DepWorkshop(generate_depworld(seed, **mix[0]), executor=EXEC, address_seed=seed)
    cfg = PopulationConfig(mode="single", rounds=1, updates_per_slot=1, development_examples=2, teacher="dep_reuse",
                           world_family="depworld", policy={"feature_version": "d1"},
                           training_seed_start=DEV + 10_000_000, development_seed_start=DEV + 90_000_000,
                           train={"width": 8, "device": "cpu", "batch_size": 1, "max_steps": 64, "evaluation_batch": 2},
                           world_mix=mix)
    run_ = PopulationRun(cfg, tmp_path / "d", factory, lambda: make_reference("dep_reuse"))
    run_.run()
    assert run_.state["status"] == "completed"
    assert "campaign03_depworld.py" in run_.state["source_hashes"]
    with pytest.raises(ValueError):
        PopulationConfig(world_family="unknown")


def test_evaluator_episode_counts_accept_depworld_history():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "research/tools/campaign02_evaluate.py"
    mspec = importlib.util.spec_from_file_location("campaign02_eval_dep", path)
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    outcome = run(generate_depworld(DEV + 70, p_event=1.0, foreign_records=4), "naive_reuse")
    counts = module.episode_counts(outcome)
    assert counts["dep_uses"] == len(outcome["reuse_audit"]) and counts["reductions"] == len(outcome["reductions"])
    assert counts["return_use_attempts"] >= counts["dep_uses"]
    json.dumps(counts)


# --- progress-triggered events: exposure independent of agent speed -------------------------

def _completion_steps(outcome):
    return [h["step"] for h in outcome["history"] if h["feedback"].get("status") == "success"
            and ("selection_id" in h["feedback"] or "assignment_id" in h["feedback"])]


def test_progress_trigger_exposure_is_identical_across_references():
    reached_by_mode = {m: 0 for m in REFERENCE_MODES}
    kinds = set()
    for i in range(80):
        spec = generate_depworld(DEV + 3000 + i, p_event=1.0, foreign_records=(0, 2, 4)[i % 3],
                                 event_trigger="progress")
        k = spec.event[1]
        kinds.add((k, spec.event[0]))
        for mode in REFERENCE_MODES:
            out = run(spec, mode, i)
            steps = _completion_steps(out)
            reached = len(steps) >= k
            # Every reference that makes the k-th completion commit experiences the event,
            # at exactly that step; none experiences it otherwise.
            assert bool(out["events"]) == reached, (i, mode)
            if reached:
                reached_by_mode[mode] += 1
                assert out["events"][0]["step"] == steps[k - 1], (i, mode)
                assert out["events"][0]["kind"] == spec.event[0]
    assert all(v > 40 for v in reached_by_mode.values()), reached_by_mode
    # solver references all reach k whenever they commit anything: identical exposure
    assert len({reached_by_mode[m] for m in ("recompute", "reuse", "reuse_norevise")}) == 1
    assert {k for k, _ in kinds} == {1, 2}
    assert {kind for _, kind in kinds} == set(EVENT_KINDS)


def test_progress_trigger_semantics_on_tiny_world():
    # k=1 fires right after the selection commit, even when the assignment is not reached
    env = DepWorkshop(tiny(event=("capacity_reduced", 1, 4), event_trigger="progress"), executor=EXEC)
    inspect_all(env)
    o = env.step(Action("think"))
    assert not o.events
    o = commit(env, ["A1", "B2"])  # weight 5 > 4: revoked immediately
    assert o.feedback["event"]["kind"] == "capacity_reduced" and o.events[0]["revoked"] == ["selection"]
    assert o.selection_id is None
    # a rejected commit is not a completion
    env = DepWorkshop(tiny(event=("slot_closed", 2, 4), event_trigger="progress"), executor=EXEC)
    inspect_all(env)
    commit(env, ["A1", "B2"])
    env.step(Action("choose_slot", {"item": "A1", "slot": 0}))
    env.step(Action("choose_slot", {"item": "B2", "slot": 1}))
    o = env.step(Action("commit_assignment"))
    assert o.feedback["reason"] == "slot_conflict" and not o.events
    env.step(Action("choose_slot", {"item": "B2", "slot": 4}))
    o = env.step(Action("commit_assignment"))
    assert o.events and o.events[0]["revoked"] == ["assignment"] and o.assignment is None
    with pytest.raises(ValueError):
        tiny(event_trigger="sometimes")
    with pytest.raises(ValueError):
        generate_depworld(DEV, event_trigger="sometimes")


def test_step_trigger_is_backward_compatible():
    spec = generate_depworld(DEV + 7, p_event=1.0, event_trigger="step")
    n = 9
    assert spec.event_trigger == "step" and n + 9 <= spec.event[1] <= n + 19
    assert tiny().event_trigger == "step"
