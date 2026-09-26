"""extended-04 progress diagnostic v1 (campaign04_progress) and branching support
(DepWorkshop.clone, SolverCache, clone_branch)."""
from functools import partial
import json
import random

import pytest

from tensegra.campaign02_protocol import BoundedSolver, execute
from tensegra.campaign02_world import Action
from tensegra.campaign03_depworld import (DepItem, DepReference, DepSpec, DepWorkshop, action_catalog,
                                          depworld_executor, generate_depworld, relations)
from tensegra.campaign04_branch import SolverCache, cache_key, clone_branch, wall_limited
from tensegra.campaign04_progress import (CLASSES, NO_PROGRESS, ProgressTracker, StepClass, plan_context,
                                          record_key, signature)

EXEC = partial(depworld_executor, execute_call=execute)
DEV = 2_150_000_000  # extended-04 development seeds (disjoint from every extended-03 and A1 range)
TRAIN_KINDS = ("edge_closed", "capacity_reduced", "slot_closed")


def tiny(**over):
    items = (DepItem("A1", 0, 4, 2, 2, (0, 1, 2)), DepItem("A2", 0, 1, 6, 1, (0, 3)),
             DepItem("B1", 1, 3, 2, 2, (0, 2)), DepItem("B2", 1, 1, 1, 3, (1, 4)))
    params = dict(items=items, categories=(0, 1), slots=5, capacity=6, funds=7, incompatible=(("A2", "B2"),),
                  deadline=10, edges=((0, 1, 2), (0, 2, 5), (1, 2, 2)), locations=3, destination=2)
    params.update(over)
    return DepSpec(**params)


class Run:
    """Environment + tracker driven step by step."""

    def __init__(self, spec, executor=EXEC, address_seed=0, **kw):
        self.env = DepWorkshop(spec, executor=executor, address_seed=address_seed)
        self.o = self.env.observe()
        self.tr = ProgressTracker(self.o, verify=True, **kw)

    def __call__(self, kind, **args) -> StepClass:
        a = Action(kind, args)
        self.o = self.env.step(a)
        return self.tr.update(self.o, a)


def inspect_all(r):
    r("inspect", target="requirements")
    for x in r.o.item_inventory:
        r("inspect", target=x["handle"])
    return r("inspect", target="map")


def _worlds(n, offset=0):
    for k in range(n):
        seed = DEV + offset + k
        kw = dict(p_event=(0.5, 1.0, 0.0)[k % 3], foreign_records=(0, 2, 4)[k % 3])
        if k % 5 == 4:
            kw.update(categories=4, choices=3, locations=8, slots=7)
        kinds = TRAIN_KINDS if k % 4 else ("deadline_moved",) + TRAIN_KINDS
        trigger = "step" if k % 7 == 6 else "progress"
        yield seed, generate_depworld(seed, event_kinds=kinds, event_trigger=trigger, compute_price=1e-4, **kw)


# ---------------------------------------------------------------------------
# References score zero; classes are well formed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["reuse", "recompute", "greedy", "reuse_norevise"])
def test_references_score_zero_no_progress(mode):
    totals = {c: 0 for c in CLASSES}
    for seed, spec in _worlds(90, offset=1000):
        r = Run(spec, address_seed=seed)
        ref = DepReference(mode)
        while not r.o.done:
            a = ref.choose(r.o)
            r.o = r.env.step(a)
            c = r.tr.update(r.o, a)
            assert not c.no_progress, (mode, seed, c, a)
            totals[c.cls] += 1
        assert r.tr.summary()["no_progress"] == 0 and r.tr.flags == 0
    assert totals["useful_inspection"] > 0 and totals["other_progress"] > 0
    if mode != "greedy":
        assert totals["solver_work"] > 0


def test_every_step_gets_one_class_and_summary_is_compact():
    rng = random.Random(3)
    for seed, spec in _worlds(12, offset=2000):
        r = Run(spec, address_seed=seed)
        while not r.o.done:
            a = rng.choice(action_catalog(r.o))
            r.o = r.env.step(a)
            c = r.tr.update(r.o, a)
            assert c.cls in CLASSES and c.no_progress == (c.cls in NO_PROGRESS)
            assert (c.cycle_length is not None) == c.no_progress
        s = r.tr.summary()
        assert len(s["classes"]) == len(r.env.evaluate()["history"]) == sum(s["counts"].values())
        assert s["no_progress"] == s["counts"]["idempotent"] + s["counts"]["short_cycle"] == len(s["cycle_lengths"])
        json.dumps(s)


# ---------------------------------------------------------------------------
# Signature: equivalence relation
# ---------------------------------------------------------------------------

def test_signature_excludes_counters_and_think_is_idempotent_no_progress():
    r = Run(tiny())
    inspect_all(r)
    before = r.tr.current_signature
    c = r("think")
    assert c.cls == "idempotent" and c.no_progress and c.cycle_length == 1 and not c.latent_stall
    assert r.tr.current_signature == before  # step counter, remaining budget, feedback, log excluded
    assert [r("think").latent_stall for _ in range(3)] == [False, True, True]
    assert r.tr.latent_stalls == 2
    # a think's flag masks think at this signature only
    assert r.tr.mask([Action("think"), Action("verify"), Action("abstain")]) == [True, False, False]


def test_repeated_inspection_is_idempotent_and_new_inspection_is_useful():
    r = Run(tiny())
    assert r("inspect", target="A1").cls == "useful_inspection"
    assert r("inspect", target="A1").cls == "idempotent"
    assert r("inspect", target="requirements").cls == "useful_inspection"
    assert r("inspect", target="map").cls == "useful_inspection"
    assert r("inspect", target="map").cls == "idempotent"


def test_alternating_finish_by_is_a_short_cycle_and_masked_per_signature():
    r = Run(tiny())
    inspect_all(r)
    r("choose_item", item="A1"), r("choose_item", item="B2")
    assert r("commit_pending").cls == "other_progress"
    r("start_assign", handle="p")
    classes = [r("add_constraint", problem="p", constraint="finish_by", bound=6 + i % 2) for i in range(6)]
    assert [c.cls for c in classes] == ["other_progress", "other_progress"] + ["short_cycle"] * 4
    assert all(c.cycle_length == 2 for c in classes[2:])
    # the flagged key is masked exactly at the signature it was taken from
    b8 = Action("add_constraint", {"problem": "p", "constraint": "finish_by", "bound": 6})
    b9 = Action("add_constraint", {"problem": "p", "constraint": "finish_by", "bound": 7})
    assert r.tr.mask([b8, b9, Action("verify")]) == [True, False, False]  # current: bound 7 set -> 6 flagged
    r("add_constraint", problem="p", constraint="conflicts")  # a relevant change releases the mask
    assert r.tr.mask([b8, b9]) == [False, False] and r.tr.flagged_keys() == set()


def test_records_keyed_by_content_repeated_call_and_retrieve_are_no_progress():
    r = Run(tiny())
    inspect_all(r)
    r("start_subset", handle="s")
    for c in ("capacity", "funds", "incompatibility"):
        r("add_constraint", problem="s", constraint=c)
    first = r("call", problem="s", budget=128)
    assert first.cls == "solver_work"
    handle = r.o.feedback["return"]
    again = r("call", problem="s", budget=128)  # same inputs, plan and budget: a new handle, same content
    assert again.cls == "idempotent" and again.no_progress
    other = r.o.feedback["return"]
    assert other != handle
    assert r("call", problem="s", budget=1024).cls == "solver_work"  # larger budget
    assert r("call", problem="s", budget=16).cls == "other_progress"  # smaller budget: new record, not work
    assert r("retrieve", handle=handle).cls == "other_progress"
    assert r("retrieve", handle=other).cls == "idempotent"  # same content already retrieved
    assert r("retrieve", handle=handle).cls == "idempotent"
    # canonical keys: the two content-equal records share one key
    assert r.tr.canonical_key(Action("retrieve", {"handle": handle})) == \
        r.tr.canonical_key(Action("retrieve", {"handle": other}))
    assert r.tr.canonical_key(Action("start_subset", {"handle": "x"})) == \
        r.tr.canonical_key(Action("start_subset", {"handle": "y"}))


def test_call_after_dependency_or_plan_change_is_solver_work():
    r = Run(tiny())
    inspect_all(r)
    assert r("build_route", handle="rt").cls == "rejected"  # no assignment yet
    r("choose_item", item="A1"), r("choose_item", item="B2")
    r("commit_pending")
    r("start_assign", handle="a")
    r("add_constraint", problem="a", constraint="conflicts")
    assert r("call", problem="a", budget=128).cls == "solver_work"
    handle = r.o.feedback["return"]
    r("retrieve", handle=handle)
    assert r("use_return", handle=handle, **{"as": "assign"}).cls == "other_progress"
    first = dict(r.o.assignment)
    r("build_route", handle="rt2")
    assert r("call", problem="rt2", budget=128).cls == "solver_work"
    assert r("call", problem="rt2", budget=128).cls == "idempotent"
    # a new assignment (plan change): the identical route call is new work
    # untriggered (no event/rejection/failure/new retrieval since the commit), but the state
    # now holds the route records, so it is not a cycle
    assert r("uncommit", target="assign").cls == "other_progress"
    other = next({"A1": s, "B2": 4} for s in (0, 1, 2) if {"A1": s, "B2": 4} != first)
    for item, slot in other.items():
        r("choose_slot", item=item, slot=slot)
    assert r("commit_assignment").cls == "other_progress"
    assert r("call", problem="rt2", budget=128).cls == "solver_work"
    # a requirement change (event) makes an identical draft call new work too
    r2 = Run(tiny(event=("capacity_reduced", 1, 6), event_trigger="progress"))
    inspect_all(r2)
    r2("start_subset", handle="s")
    r2("add_constraint", problem="s", constraint="funds")
    assert r2("call", problem="s", budget=128).cls == "solver_work"
    assert r2("call", problem="s", budget=128).cls == "idempotent"
    r2("choose_item", item="A1"), r2("choose_item", item="B2")
    assert r2("commit_pending").event  # capacity version bump (value unchanged: nothing revoked)
    assert r2("call", problem="s", budget=128).cls == "solver_work"


def test_rejections_are_their_own_class_with_repeat_flag():
    r = Run(tiny())
    c = r("commit_pending")
    assert c.cls == "rejected" and c.reason == "missing_dependency" and not c.repeat_rejection
    c = r("commit_pending")
    assert c.cls == "rejected" and c.repeat_rejection and not c.no_progress and not c.flagged
    assert r.tr.flagged_keys() == set()
    r2 = Run(tiny(), mask_repeat_rejections=True)
    r2("commit_pending")
    assert r2("commit_pending").flagged
    assert r2.tr.mask([Action("commit_pending")]) == [True]


def test_uncommit_cycles_triggered_vs_untriggered():
    r = Run(tiny())
    inspect_all(r)
    r("choose_item", item="A1"), r("choose_item", item="B2")
    r("commit_pending")
    c = r("uncommit", target="select")  # nothing happened since the commit: untriggered
    assert c.cls == "short_cycle" and c.cycle_length == 2
    r("commit_pending")
    r("commit_pending")  # rejected (already committed): a trigger
    c = r("uncommit", target="select")
    assert c.cls == "triggered_revision" and not c.no_progress


def test_verify_and_abstain_are_never_masked_and_abstain_is_terminal():
    r = Run(tiny())
    assert r("verify").cls == "rejected"
    r = Run(tiny())
    assert r("abstain").cls == "terminal"


def test_equal_signatures_give_equal_applicability_relations():
    """Relation values are functions of signature components (so they are kept out of the hash)."""
    rng = random.Random(11)
    seen = {}
    for seed, spec in _worlds(15, offset=3000):
        r = Run(spec, address_seed=seed)
        while not r.o.done:
            ref = DepReference("reuse") if rng.random() < .5 else None
            a = ref.choose(r.o) if ref else rng.choice(action_catalog(r.o))
            r.o = r.env.step(a)
            r.tr.update(r.o, a)
            rel = sorted((r.tr._records[x["handle"]], p, json.dumps(relations(r.o, x, p), sort_keys=True))
                         for x in r.o.records for p in ("constrained_subset", "csp", "shortest_path"))
            key = (seed, r.tr.current_signature)
            if key in seen:
                assert seen[key] == rel
            seen[key] = rel


def test_plain_signature_is_record_content_keyed():
    env = DepWorkshop(generate_depworld(DEV + 5, foreign_records=2), executor=EXEC, address_seed=1)
    o = env.observe()
    assert signature(o) == signature(DepWorkshop(generate_depworld(DEV + 5, foreign_records=2), executor=EXEC,
                                                 address_seed=999).observe())  # handles differ, content equal
    rec = o.records[0]
    assert record_key(rec, plan_context(o)) != record_key(rec, None)


def test_tracker_copy_is_independent():
    r = Run(tiny())
    inspect_all(r)
    r("think")
    copy = r.tr.copy()
    snapshot = copy.summary(), copy.current_signature, copy.flagged_keys()
    r("choose_item", item="A1")
    r("think")
    assert (copy.summary(), copy.current_signature, copy.flagged_keys()) == snapshot
    assert r.tr.summary() != copy.summary()


# ---------------------------------------------------------------------------
# Clone and exact solver cache
# ---------------------------------------------------------------------------

def _random_actions(env, rng, k):
    """k random non-terminal actions (never abstain/verify, so the episode stays open)."""
    o, acts = env.observe(), []
    for _ in range(k):
        if o.done:
            break
        a = rng.choice([x for x in action_catalog(o) if x.kind not in ("abstain", "verify")])
        acts.append(a)
        o = env.step(a)
    return acts


def _strip(outcome):
    return {k: v for k, v in outcome.items() if k != "solver_cpu_seconds"}


@pytest.mark.parametrize("audited", [False, True])
def test_clone_reproduces_future_trajectories_exactly(audited):
    from tensegra.campaign03_p1_audit import P1AuditedDepWorkshop
    cls = P1AuditedDepWorkshop if audited else DepWorkshop
    with BoundedSolver() as solver:
        executor = partial(depworld_executor, execute_call=solver.execute)
        for seed, spec in _worlds(8, offset=4000):
            rng = random.Random(seed)
            env = cls(spec, executor=executor, address_seed=seed)
            prefix = _random_actions(env, rng, rng.randint(3, 25))
            branch = env.clone()
            assert branch._executor is env._executor and type(branch) is cls
            assert branch.observe().to_dict() == env.observe().to_dict()
            ref = DepReference(("reuse", "recompute")[seed % 2])
            o = env.observe()
            while not o.done:  # continue both with the same actions
                a = ref.choose(o) if rng.random() < .7 else rng.choice(action_catalog(o))
                o = env.step(a)
                ob = branch.step(a)
                assert ob.to_dict() == o.to_dict()
            assert _strip(branch.evaluate()) == _strip(env.evaluate())
            assert len(branch.evaluate()["history"]) > len(prefix)
        assert solver.restarts == 0


def test_clone_is_independent_of_the_original():
    env = DepWorkshop(generate_depworld(DEV + 77, p_event=1.0, foreign_records=2), executor=EXEC, address_seed=3)
    _random_actions(env, random.Random(1), 10)
    before = env.observe().to_dict()
    branch = env.clone()
    ref = DepReference("reuse")
    o = branch.observe()
    while not o.done:
        o = branch.step(ref.choose(o))
    assert env.observe().to_dict() == before and not env.observe().done


def _collect_calls(n_worlds, offset):
    calls = []

    def recording(primitive, problem, max_work):
        calls.append((primitive, json.loads(json.dumps(problem)), max_work))
        return EXEC(primitive, problem, max_work)
    rng = random.Random(offset)
    for seed, spec in _worlds(n_worlds, offset):
        env = DepWorkshop(spec, executor=recording, address_seed=seed)
        o = env.observe()
        ref = DepReference(("reuse", "recompute", "naive_reuse")[seed % 3])
        while not o.done:
            a = ref.choose(o) if rng.random() < .8 else rng.choice(action_catalog(o))
            o = env.step(a)
    return calls


FIELDS = ("status", "payload", "work_units", "certificate", "certificate_valid")


def test_solver_cache_is_result_identical_on_many_calls():
    calls = _collect_calls(60, 5000)
    assert len(calls) > 200
    cache = SolverCache(EXEC)
    statuses = set()
    for primitive, problem, budget in calls + calls:  # second pass: hits
        cached = cache(primitive, json.loads(json.dumps(problem)), budget)
        raw = EXEC(primitive, json.loads(json.dumps(problem)), budget)
        assert {k: cached[k] for k in FIELDS} == {k: raw[k] for k in FIELDS}
        statuses.add(raw["status"])
    stats = cache.stats()
    assert stats["hits"] >= len(calls) and stats["uncacheable_wall_limited"] == 0
    assert {"success", "timeout", "infeasible"} <= statuses


def test_cached_episodes_equal_uncached_episodes_and_cache_is_shared_by_clones():
    rng = random.Random(9)
    for seed, spec in _worlds(20, 6000):
        cache = SolverCache(EXEC)
        a = DepWorkshop(spec, executor=EXEC, address_seed=seed)
        b = DepWorkshop(spec, executor=cache, address_seed=seed)
        o = a.observe()
        ref = DepReference("recompute")
        while not o.done:
            act = ref.choose(o) if rng.random() < .8 else rng.choice(action_catalog(o))
            o = a.step(act)
            assert b.step(act).to_dict() == o.to_dict()
        assert _strip(a.evaluate()) == _strip(b.evaluate())
        assert b.clone()._executor is cache


def test_wall_limited_results_are_never_cached():
    budget = 128
    outcomes = iter([{"status": "timeout", "payload": None, "work_units": 5, "certificate": (("optimal", False),),
                      "certificate_valid": False, "cpu_seconds": 2.0, "child_cpu_seconds": 2.0},
                     {"status": "success", "payload": ((0,), 1, 1, 1), "work_units": 9,
                      "certificate": (("exact", True),), "certificate_valid": True, "cpu_seconds": .1,
                      "child_cpu_seconds": .1}])
    cache = SolverCache(lambda p, q, w: next(outcomes))
    assert cache("csp", {"x": 1}, budget)["status"] == "timeout"
    assert cache("csp", {"x": 1}, budget)["status"] == "success"
    hit = cache("csp", {"x": 1}, budget)
    assert hit["status"] == "success" and hit["cpu_seconds"] == 0.0
    assert cache.stats()["uncacheable_wall_limited"] == 1 and cache.stats()["hits"] == 1
    assert wall_limited({"status": "timeout", "work_units": budget,
                         "certificate": (("process_killed_or_no_result", True),)}, budget)
    assert wall_limited({"status": "unknown", "work_units": budget}, budget)
    assert not wall_limited({"status": "timeout", "work_units": budget, "certificate": (("optimal", False),)}, budget)
    assert cache_key("csp", {"a": 1, "b": 2}, 16) == cache_key("csp", {"b": 2, "a": 1}, 16)
    assert cache_key("csp", {"a": 1}, 16) != cache_key("csp", {"a": 1}, 128)


def test_clone_branch_copies_env_hidden_and_tracker():
    import torch
    env = DepWorkshop(generate_depworld(DEV + 91, foreign_records=2), executor=EXEC, address_seed=4)
    tracker = ProgressTracker(env.observe())
    for a in _random_actions(env.clone(), random.Random(2), 6):
        tracker.update(env.step(a), a)
    hidden = torch.randn(1, 2, 3)
    e2, h2, t2 = clone_branch(env, hidden, tracker)
    assert torch.equal(h2, hidden) and h2.data_ptr() != hidden.data_ptr()
    assert t2.summary() == tracker.summary() and t2 is not tracker
    assert e2.observe().to_dict() == env.observe().to_dict() and e2._executor is env._executor
    assert clone_branch(env)[1:] == (None, None)


def test_rollout_mask_fn_matches_tracker_mask():
    import random
    from tensegra.campaign03_depworld import DepWorkshop, generate_depworld, action_catalog
    from tensegra.campaign04_progress import ProgressTracker, rollout_mask_fn
    rng = random.Random(3)
    env = DepWorkshop(generate_depworld(2_000_123))
    obs = env.observe()
    fn = rollout_mask_fn(1)
    ref = ProgressTracker(obs)
    history = []
    for _ in range(40):
        cands = action_catalog(obs)
        assert fn(0, obs, cands, history) == ref.mask(cands)
        a = cands[0] if rng.random() < 0.5 else rng.choice(cands)  # bias toward repeats
        after = env.step(a)
        history.append((obs, a, after))
        ref.update(after, a)
        obs = after
        if obs.done:
            break
