"""extended-03 Stage B preflight: what do the ENCODED depworld inputs preserve? (CPU, in-process solver.)

Independent audit of the d1 encoder and the P1 ablation masks (d1-noapp, d1-noattempt).
Every pair is built by stepping real ``DepWorkshop`` instances (hand-built specs, in-process
``campaign02_protocol.execute`` through ``depworld_executor``) and compared on the exact tensors the
policy consumes: ``campaign02_training.public_frame(observation, version)`` -> (catalogue, observation
vector, candidate matrix).

A pair declares its *carrier*: the part of the input that should carry the distinction
("row": the named candidate row(s); "observation"; "state": observation + full candidate matrix;
or an explicit list of candidate coordinate names). A pair is *distinguishable* under a version if
the carrier differs. Each pair states the expected verdict per version; ``None`` = report only.

Usage: PYTHONPATH=src python research/tools/campaign03_preflight.py [--out research/campaigns/extended-03/preflight-audit.json]
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from topoformer.campaign02_protocol import execute  # noqa: E402
from topoformer.campaign02_training import public_frame  # noqa: E402
from topoformer.campaign02_world import Action  # noqa: E402
from topoformer.campaign03_depworld import (CANDIDATE_NAMES_D1, CSP_CONSTRAINTS, FEATURE_VERSIONS,  # noqa: E402
    OBSERVATION_NAMES_D1, SUBSET_CONSTRAINTS, USES, DepItem, DepReference, DepSpec, DepWorkshop, action_key,
    call_budgets, csp_problem, depworld_executor, route_problem, selection_id, subset_problem, _item_row)

EXEC = partial(depworld_executor, execute_call=execute)
VERSIONS = FEATURE_VERSIONS  # ("d1", "d1-noapp", "d1-noattempt")
ALL = {v: True for v in VERSIONS}
NONE = {v: False for v in VERSIONS}
APP = {"d1": True, "d1-noapp": False, "d1-noattempt": True}      # applicability distinction (1-4)
ATT = {"d1": True, "d1-noapp": True, "d1-noattempt": False}      # attempt-memory distinction (6-7)

# ---------------------------------------------------------------------------------------------
# A small hand-built world with exactly controllable rejections, records and events
# ---------------------------------------------------------------------------------------------
# cap 6, funds 7, incompatible (A2,B1).  {A1,B1}: valid (w4 p4).  {A1,B3}: capacity.  {A1,B2}: funds.
# {A2,B1}: incompatibility.  {A3,B1}: capacity.  Slot 3 closed.  Map 0->1->2->3 (4), 0->1->3 (5), 0->3 (6).
ITEMS = (DepItem("A1", 0, 2, 2, 2, (0, 1, 2)), DepItem("A2", 0, 1, 1, 1, (0, 3)), DepItem("A3", 0, 5, 1, 1, (1,)),
         DepItem("B1", 1, 2, 2, 2, (0, 2, 3)), DepItem("B2", 1, 1, 7, 1, (1, 4)), DepItem("B3", 1, 5, 1, 3, (0, 2)))
EDGES = ((0, 1, 2), (0, 3, 6), (1, 2, 1), (1, 3, 3), (2, 3, 1))
BASE = dict(items=ITEMS, categories=(0, 1), slots=5, capacity=6, funds=7, incompatible=(("A2", "B1"),), deadline=9,
            edges=EDGES, locations=4, destination=3, closed_slots=(3,),
            planted=(("A1", "B1"), (("A1", 0), ("B1", 2)), (0, 1, 2, 3)))
FOREIGN_KINDS = ("subset_current", "subset_earlier", "subset_version_mismatch", "subset_other_order",
                 "route_current", "route_other_goal", "route_earlier", "route_missing_edge", "route_slow",
                 "csp_current", "csp_other_selection")


def _solve(snapshot):
    return depworld_executor(snapshot["primitive"], deepcopy(snapshot["problem"]), 100000, execute_call=execute)


def foreign_records(spec: DepSpec, kinds=FOREIGN_KINDS) -> tuple:
    """Registered foreign problems + solved records built from the spec's TRUE facts (like the generator)."""
    rows = [_item_row(x) for x in spec.items]
    by = {r["handle"]: r for r in rows}
    req = {"capacity": spec.capacity, "funds": spec.funds, "incompatible": [list(p) for p in spec.incompatible]}
    subset_v = {"requirements_version": {"capacity": 1, "funds": 1, "incompatible": 1}}
    subset_v0 = {"requirements_version": {"capacity": 0, "funds": 0, "incompatible": 0}}
    cat = lambda hs: sorted((by[h] for h in hs), key=lambda r: r["category"])  # noqa: E731
    first = {c: next(x.handle for x in spec.items if x.category == c) for c in spec.categories}
    cur_sel = [first[c] for c in spec.categories]                     # A1, B1
    other_sel = [x.handle for x in spec.items if x.category == 0][1:2] + \
        [x.handle for x in spec.items if x.category == 1][2:3]      # A2, B3
    out = []
    for kind in kinds:
        if kind == "subset_current":
            snap, deps = subset_problem(rows, req, SUBSET_CONSTRAINTS), subset_v
        elif kind == "subset_earlier":
            snap, deps = subset_problem(rows, {**req, "capacity": req["capacity"] + 2}, SUBSET_CONSTRAINTS), subset_v0
        elif kind == "subset_version_mismatch":
            snap, deps = subset_problem(rows, req, SUBSET_CONSTRAINTS), subset_v0
        elif kind == "subset_other_order":
            snap, deps = subset_problem(rows, {**req, "capacity": req["capacity"] + 2}, SUBSET_CONSTRAINTS), subset_v
        elif kind == "route_current":
            snap, deps = route_problem(spec.edges, spec.locations, spec.start, spec.destination), {"requirements_version": {"map": 1}}
        elif kind == "route_other_goal":
            snap, deps = route_problem(spec.edges, spec.locations, spec.start, 2), {"requirements_version": {"map": 1}}
        elif kind in ("route_earlier", "route_missing_edge"):   # an extra edge 0->2 that does not exist now
            snap = route_problem(list(spec.edges) + [(0, 2, 1)], spec.locations, spec.start, spec.destination)
            deps = {"requirements_version": {"map": 0 if kind == "route_earlier" else 1}}
        elif kind == "route_slow":                                # 0->3 believed to cost 1 (really 6)
            edges = [(u, v, 1 if (u, v) == (0, 3) else w) for u, v, w in spec.edges]
            snap, deps = route_problem(edges, spec.locations, spec.start, spec.destination), {"requirements_version": {"map": 1}}
        elif kind == "csp_current":
            snap = csp_problem(cat(cur_sel), spec.closed_slots, CSP_CONSTRAINTS)
            deps = {"requirements_version": {"slots": 1}, "selection_id": selection_id(cur_sel)}
        elif kind == "csp_other_selection":
            snap = csp_problem(cat(other_sel), spec.closed_slots, CSP_CONSTRAINTS)
            deps = {"requirements_version": {"slots": 1}, "selection_id": selection_id(other_sel)}
        else:
            raise ValueError(kind)
        res = _solve(snap)
        out.append({"label": kind, "primitive": snap["primitive"], "snapshot": snap, "depends_on": deps,
                    "status": res["status"], "payload": res["payload"],
                    "certificate_valid": bool(res["certificate_valid"]), "work_units": int(res["work_units"])})
    return tuple(out)


def make_spec(event=None, foreign_kinds=(), **over) -> DepSpec:
    spec = DepSpec(**{**BASE, **over, "event": event})
    return replace(spec, foreign=foreign_records(spec, foreign_kinds)) if foreign_kinds else spec


class Run:
    """A stepped world plus convenience handles."""

    def __init__(self, spec, address_seed=0):
        self.env = DepWorkshop(spec, executor=EXEC, address_seed=address_seed)
        self.o = self.env.observe()
        self.foreign = {f["label"]: r["handle"] for f, r in zip(spec.foreign, self.o.records)}

    def step(self, kind, **args):
        self.o = self.env.step(Action(kind, args))
        return self.o

    def name(self):
        return f"problem_{len(self.o.problems)}"

    def inspect_all(self, skip=()):
        self.step("inspect", target="requirements")
        for r in self.o.item_inventory:
            if r["handle"] not in skip:
                self.step("inspect", target=r["handle"])
        return self.step("inspect", target="map")

    def commit(self, *handles):
        for h in handles:
            self.step("choose_item", item=h)
        return self.step("commit_pending")

    def assign(self, **slots):
        for h, s in slots.items():
            self.step("choose_slot", item=h, slot=s)
        return self.step("commit_assignment")

    def compute(self, primitive, budget=1024):
        name = self.name()
        if primitive == "constrained_subset":
            self.step("start_subset", handle=name)
            for c in SUBSET_CONSTRAINTS:
                self.step("add_constraint", problem=name, constraint=c)
        elif primitive == "csp":
            self.step("start_assign", handle=name)
            self.step("add_constraint", problem=name, constraint="conflicts")
        else:
            self.step("build_route", handle=name)
        self.step("call", problem=name, budget=budget)
        return self.o.feedback["return"]

    def pad_to(self, step):
        while self.o.step < step:
            self.step("think")
        return self.o


def chain(run: Run):
    """Full public chain: inspect, commit {A1,B1}, assign A1:0 B1:2, and own records of all three primitives."""
    run.inspect_all()
    run.commit("A1", "B1")
    run.assign(A1=0, B1=2)
    return {"subset": run.compute("constrained_subset"), "csp": run.compute("csp"),
            "route": run.compute("shortest_path")}


# ---------------------------------------------------------------------------------------------
# Encoded comparison
# ---------------------------------------------------------------------------------------------

def frame(o, version):
    actions, obs, cands = public_frame(o, version)
    return actions, obs, cands


def _row(o, action, version):
    actions, obs, cands = frame(o, version)
    return cands[actions.index(action)]


def _diff(x, y, names):
    return [names[i] for i, (p, q) in enumerate(zip(x, y)) if p != q]


def compare(o_a, acts_a, o_b, acts_b, carrier="row", map_action=None):
    """Per version: observation diff, row diffs (acts_a[i] vs acts_b[i]), full-matrix diff, verdict."""
    out = {}
    for v in VERSIONS:
        ca, xa, ma = frame(o_a, v)
        cb, xb, mb = frame(o_b, v)
        assert len(xa) == len(OBSERVATION_NAMES_D1) and all(len(r) == len(CANDIDATE_NAMES_D1) for r in ma + mb)
        obs_diff = _diff(xa, xb, OBSERVATION_NAMES_D1)
        rows = []
        for a, b in zip(acts_a, acts_b):
            rows.append(_diff(ma[ca.index(a)], mb[cb.index(b)], CANDIDATE_NAMES_D1))
        row_diff = sorted({n for r in rows for n in r})
        mapped = [map_action(a) for a in ca] if map_action else ca
        same_catalog = mapped == cb
        matrix_rows = (sum(r != s for r, s in zip(ma, mb)) if same_catalog else None)
        if carrier == "row":
            dist = bool(row_diff)
        elif carrier == "observation":
            dist = bool(obs_diff)
        elif carrier == "state":
            dist = bool(obs_diff) or not same_catalog or bool(matrix_rows)
        else:  # explicit coordinate names on the row
            dist = any(n in carrier for n in row_diff)
        out[v] = {"distinguishable": dist, "observation_diff": obs_diff, "row_diff": row_diff,
                  "same_catalog": same_catalog, "matrix_rows_differing": matrix_rows}
    return out


def pair(results, pid, distinction, description, o_a, acts_a, o_b, acts_b, expected, carrier="row",
         map_action=None, notes="", checks=None):
    comp = compare(o_a, acts_a, o_b, acts_b, carrier, map_action)
    ok = all(expected.get(v) is None or comp[v]["distinguishable"] == expected[v] for v in VERSIONS)
    if checks:
        ok = ok and all(checks.values())
    results.append({"id": pid, "distinction": distinction, "description": description,
                    "carrier": carrier if isinstance(carrier, str) else list(carrier),
                    "expected": expected, "versions": comp, "preconditions": checks or {}, "pass": ok,
                    "notes": notes})
    return results[-1]


def same_flags(o, *handles):
    recs = {r["handle"]: r for r in o.records}
    keys = [(recs[h]["status"], recs[h]["certificate_valid"], h in o.retrieved) for h in handles]
    return all(k == keys[0] for k in keys)


def use(h, as_):
    return Action("use_return", {"handle": h, "as": as_})


def catalog_check(o) -> dict:
    """The candidate SET must contain every action the distinctions need."""
    acts = public_frame(o, "d1")[0]
    s = {json.dumps([a.kind, dict(a.arguments)], sort_keys=True) for a in acts}
    has = lambda k, **a: json.dumps([k, a], sort_keys=True) in s  # noqa: E731
    missing = []
    for r in o.records:
        if not has("retrieve", handle=r["handle"]):
            missing.append(("retrieve", r["handle"]))
        for u in USES:
            if not has("use_return", handle=r["handle"], **{"as": u}):
                missing.append(("use_return", r["handle"], u))
    for name in o.problems:
        for b in call_budgets(o):
            if not has("call", problem=name, budget=b):
                missing.append(("call", name, b))
    for k in ("commit_pending", "commit_assignment", "verify"):
        if not has(k):
            missing.append((k,))
    for t in ("select", "assign"):
        if not has("uncommit", target=t):
            missing.append(("uncommit", t))
    if o.known_edges is not None:
        for u, v, _ in o.known_edges:
            if u == o.position and not has("move", destination=v):
                missing.append(("move", v))
    for h in o.selected:
        for slot in o.known_items.get(h, {}).get("slots", []):
            if not has("choose_slot", item=h, slot=slot):
                missing.append(("choose_slot", h, slot))
    return {"candidates": len(acts), "duplicates": len(acts) - len(s), "missing": missing}


# ---------------------------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------------------------

def audit() -> dict:
    R: list[dict] = []
    catalog: dict = {}

    # ---- 1, 4, 5, ownership invariance: one state with own records of every primitive + foreign records
    run = Run(make_spec(foreign_kinds=FOREIGN_KINDS), address_seed=11)
    own = chain(run)
    F = run.foreign
    o = run.o
    catalog["full_chain_with_foreign"] = catalog_check(o)
    for label, retrieve in (("unretrieved", False), ("retrieved", True)):
        if retrieve:
            for h in list(own.values()) + list(F.values()):
                run.step("retrieve", handle=h)
            o = run.o
        exp_app = APP if not retrieve else {"d1": True, "d1-noapp": None, "d1-noattempt": True}
        note_ret = "" if not retrieve else ("Retrieved variant: payload feasibility facts (distance, meets-deadline, "
                                            "finish/duration, selection-valid) are kept by d1-noapp by design and may "
                                            "still separate the rows; see residuals.")
        for pid, own_h, f_label, as_ in (("1a", own["subset"], "subset_earlier", "select"),
                                         ("1b", own["subset"], "subset_other_order", "select"),
                                         ("1c", own["csp"], "csp_other_selection", "assign"),
                                         ("1d", own["route"], "route_other_goal", "route"),
                                         ("1e", own["route"], "route_earlier", "route")):
            pair(R, f"{pid}-{label}", 1, f"own current {as_} result vs foreign registered same-type ({f_label}), {label}",
                 o, [use(own_h, as_)], o, [use(F[f_label], as_)], exp_app, notes=note_ret,
                 checks={"same_status_cert_retrieved": same_flags(o, own_h, F[f_label]),
                         "both_registered": all(r["problem"] in o.problems for r in o.records)})
        for pid, a_label, b_label, as_, why in (
                ("4a", "subset_current", "subset_version_mismatch", "select", "dependency-only mismatch"),
                ("4b", "subset_current", "subset_earlier", "select", "request + dependency mismatch"),
                ("4c", "subset_current", "subset_other_order", "select", "request-only mismatch"),
                ("4d", "route_current", "route_other_goal", "route", "request-only mismatch (other goal)"),
                ("4e", "route_current", "route_earlier", "route", "earlier map, version 0"),
                ("4f", "csp_current", "csp_other_selection", "assign", "other selection")):
            pair(R, f"{pid}-{label}", 4, f"foreign applicable ({a_label}) vs foreign not applicable ({b_label}): {why}, {label}",
                 o, [use(F[a_label], as_)], o, [use(F[b_label], as_)], exp_app, notes=note_ret,
                 checks={"same_status_cert_retrieved": same_flags(o, F[a_label], F[b_label])})
        for pid, own_h, f_label, as_ in (("9o-a", own["subset"], "subset_current", "select"),
                                         ("9o-b", own["csp"], "csp_current", "assign"),
                                         ("9o-c", own["route"], "route_current", "route")):
            pair(R, f"{pid}-{label}", 9, f"ownership invariance: own current {as_} result vs foreign APPLICABLE {f_label}, {label}",
                 o, [use(own_h, as_)], o, [use(F[f_label], as_)], NONE,
                 notes="Age and ownership must be irrelevant: rows must be identical.",
                 checks={"same_status_cert_retrieved": same_flags(o, own_h, F[f_label])})
        # 5: wrong type vs right type for a given use
        pair(R, f"5a-{label}", 5, f"use as select: csp record (wrong type) vs subset record (right type), {label}",
             o, [use(own["csp"], "select")], o, [use(own["subset"], "select")], ALL)
        pair(R, f"5b-{label}", 5, f"same subset record used as assign (wrong) vs as select (right), {label}",
             o, [use(own["subset"], "assign")], o, [use(own["subset"], "select")], ALL)
        pair(R, f"5c-{label}", 5, f"use as route: foreign applicable csp vs applicable route, {label}",
             o, [use(F["csp_current"], "route")], o, [use(F["route_current"], "route")], ALL)

    # ---- 2 and 3: events after the chain.  Paired worlds share the prefix exactly (event fires at step 40).
    def after_event(event, extra=None, retrieve=()):
        r = Run(make_spec(event=event), address_seed=21)
        h = chain(r)
        r.pad_to(40)
        assert r.o.events, "event must have fired"
        if extra:
            h.update(extra(r))
        for x in retrieve:
            r.step("retrieve", handle=h[x])
        return r, h

    def recompute_subset(r):
        name = next(n for n, e in r.o.problems.items() if e["primitive"] == "constrained_subset")
        r.step("add_constraint", problem=name, constraint="capacity")   # rebuild with current requirements
        r.step("call", problem=name, budget=1024)
        return {"subset_new": r.o.feedback["return"]}

    def recompute_route(r):
        return {"route_new": r.compute("shortest_path")}

    def recompute_csp(r):
        name = next(n for n, e in r.o.problems.items() if e["primitive"] == "csp")
        r.step("add_constraint", problem=name, constraint="conflicts")
        r.step("call", problem=name, budget=1024)
        return {"csp_new": r.o.feedback["return"]}

    for label, ret in (("unretrieved", ()), ("retrieved", ("subset", "subset_new"))):
        r, h = after_event(("capacity_reduced", 40, 5), recompute_subset, ret)
        exp = APP if not ret else {"d1": True, "d1-noapp": None, "d1-noattempt": True}
        pair(R, f"2a-{label}", 2, f"own subset result before capacity_reduced(6->5) (stale: request+dependency) vs recomputed, {label}",
             r.o, [use(h["subset"], "select")], r.o, [use(h["subset_new"], "select")], exp,
             checks={"same_status_cert_retrieved": same_flags(r.o, h["subset"], h["subset_new"])})
    r, h = after_event(("capacity_reduced", 40, 6), recompute_subset)
    pair(R, "2b", 2, "own subset result before a same-value capacity event (dependency-only stale) vs recomputed",
         r.o, [use(h["subset"], "select")], r.o, [use(h["subset_new"], "select")], APP,
         checks={"same_status_cert_retrieved": same_flags(r.o, h["subset"], h["subset_new"]),
                 "same_snapshot": {x["handle"]: x for x in r.o.records}[h["subset"]]["problem_snapshot"]
                 == {x["handle"]: x for x in r.o.records}[h["subset_new"]]["problem_snapshot"]})
    r, h = after_event(("edge_closed", 40, (1, 2)), recompute_route)
    pair(R, "2c", 2, "own route before edge_closed(1,2) (stale) vs recomputed route",
         r.o, [use(h["route"], "route")], r.o, [use(h["route_new"], "route")], APP,
         checks={"same_status_cert_retrieved": same_flags(r.o, h["route"], h["route_new"])})
    r, h = after_event(("slot_closed", 40, 4), recompute_csp)
    pair(R, "2d", 2, "own csp before slot_closed(4) (unused slot: dependency-only stale) vs recomputed csp",
         r.o, [use(h["csp"], "assign")], r.o, [use(h["csp_new"], "assign")], APP,
         checks={"same_status_cert_retrieved": same_flags(r.o, h["csp"], h["csp_new"])})

    # 3: old-but-applicable vs stale (paired worlds: edge_closed vs capacity_reduced, identical prefixes)
    for label, ret in (("unretrieved", ()), ("retrieved", ("subset", "route"))):
        re_, he = after_event(("edge_closed", 40, (1, 2)), retrieve=ret)
        rc, hc = after_event(("capacity_reduced", 40, 5), retrieve=ret)
        exp = APP if not ret else {"d1": True, "d1-noapp": None, "d1-noattempt": True}
        pair(R, f"3a-{label}", 3, f"selection result from before edge_closed (still applicable) vs same result before capacity_reduced (stale), {label}",
             re_.o, [use(he["subset"], "select")], rc.o, [use(hc["subset"], "select")], exp,
             notes="Paired worlds; the observation also differs (event kind, capacity) by design; carrier = row.")
        pair(R, f"3b-{label}", 3, f"route result from before capacity_reduced (still applicable) vs same route before edge_closed (stale), {label}",
             rc.o, [use(hc["route"], "route")], re_.o, [use(he["route"], "route")], exp)
    # "the old applicable one must NOT look stale": identical to a fresh recompute of the same request
    r, h = after_event(("edge_closed", 40, (1, 2)), recompute_subset, ("subset", "subset_new"))
    pair(R, "3c", 3, "old-but-applicable selection result (pre-edge_closed) vs fresh recompute after the event (same request)",
         r.o, [use(h["subset"], "select")], r.o, [use(h["subset_new"], "select")], NONE,
         notes="Must be IDENTICAL: age is irrelevant to applicability.")
    r, h = after_event(("capacity_reduced", 40, 5), recompute_route, ("route", "route_new"))
    pair(R, "3d", 3, "old-but-applicable route (pre-capacity_reduced) vs fresh recompute after the event",
         r.o, [use(h["route"], "route")], r.o, [use(h["route_new"], "route")], NONE,
         notes="Must be IDENTICAL.")

    # ---- 6: rejection reasons, state-matched after the intervening action
    def history(steps, spec=None):
        r = Run(spec or make_spec(), address_seed=31)
        r.inspect_all()
        for k, a in steps:
            r.step(k, **a)
        return r

    ci = lambda h: ("choose_item", {"item": h})  # noqa: E731
    cs = lambda h, s: ("choose_slot", {"item": h, "slot": s})  # noqa: E731
    CP, CA, TH = ("commit_pending", {}), ("commit_assignment", {}), ("think", {})
    rej = {"capacity": [ci("A1"), ci("B3"), CP], "funds": [ci("A1"), ci("B2"), CP],
           "incompatibility": [ci("A2"), ci("B1"), CP]}
    fix = {"capacity": [ci("B1")], "funds": [ci("B1")], "incompatibility": [ci("A1")]}
    commit = Action("commit_pending")
    for a, b in (("capacity", "funds"), ("capacity", "incompatibility"), ("funds", "incompatibility")):
        ha, hb = history(rej[a]), history(rej[b])
        assert ha.o.feedback.get("reason") == a and hb.o.feedback.get("reason") == b
        pair(R, f"6-{a}-{b}-immediate", 6, f"commit_pending rejected for {a} vs {b}: the next decision",
             ha.o, [commit], hb.o, [commit], {"d1": True, "d1-noapp": True, "d1-noattempt": True}, carrier="state",
             notes="Immediate: the last feedback reason (observation) and the pending state also differ; not attempt memory.")
        ha, hb = history(rej[a] + fix[a]), history(rej[b] + fix[b])
        pair(R, f"6-{a}-{b}-after1", 6, f"commit_pending rejected for {a} vs {b}, then one repair action to the SAME pending set",
             ha.o, [commit], hb.o, [commit], ATT, carrier="state",
             checks={"same_pending": ha.o.pending == hb.o.pending, "same_step": ha.o.step == hb.o.step})
        ha, hb = history(rej[a] + fix[a] + [TH]), history(rej[b] + fix[b] + [TH])
        pair(R, f"6-{a}-{b}-after2", 6, f"commit_pending rejected for {a} vs {b}, repair + think (two intervening)",
             ha.o, [commit], hb.o, [commit], ATT, carrier="state")
    sel = [ci("A1"), ci("B1"), CP]
    rej_a = {"slot_conflict": sel + [cs("A1", 0), cs("B1", 0), CA], "slot_unavailable": sel + [cs("A1", 0), cs("B1", 3), CA]}
    commit_a = Action("commit_assignment")
    ha, hb = history(rej_a["slot_conflict"]), history(rej_a["slot_unavailable"])
    assert ha.o.feedback.get("reason") == "slot_conflict" and hb.o.feedback.get("reason") == "slot_unavailable"
    pair(R, "6-slot_conflict-slot_unavailable-immediate", 6, "commit_assignment rejected for slot_conflict vs slot_unavailable: next decision",
         ha.o, [commit_a], hb.o, [commit_a], ALL, carrier="state")
    ha, hb = history(rej_a["slot_conflict"] + [cs("B1", 2)]), history(rej_a["slot_unavailable"] + [cs("B1", 2)])
    pair(R, "6-slot_conflict-slot_unavailable-after1", 6,
         "commit_assignment rejected for slot_conflict vs slot_unavailable, then choose_slot to the SAME pending assignment",
         ha.o, [commit_a], hb.o, [commit_a], ATT, carrier="state",
         checks={"same_pending_assignment": ha.o.pending_assignment == hb.o.pending_assignment})
    # move / deliver: deadline vs missing_edge (use_return of a route record; distinct records, carrier = reason coords)
    spec = make_spec(foreign_kinds=("route_slow", "route_missing_edge"))
    reason_coords = [n for n in CANDIDATE_NAMES_D1 if n.startswith("attempt.last_reason.")]
    runs = {}
    for lab in ("route_slow", "route_missing_edge"):
        r = Run(spec, address_seed=41)
        r.inspect_all()
        for k, a in sel + [cs("A1", 0), cs("B1", 2), CA]:
            r.step(k, **a)
        for h in r.foreign.values():
            r.step("retrieve", handle=h)
        r.step("use_return", handle=r.foreign[lab], **{"as": "route"})
        runs[lab] = (r, r.o.feedback.get("reason"))
        r.step("think")
    (rs, why_s), (rm, why_m) = runs["route_slow"], runs["route_missing_edge"]
    assert (why_s, why_m) == ("deadline", "missing_edge"), (why_s, why_m)
    pair(R, "6-deliver-deadline-missing_edge-after1", 6,
         "deliver (use_return route) rejected for deadline vs missing_edge, after one intervening think: the attempted row",
         rs.o, [use(rs.foreign["route_slow"], "route")], rm.o, [use(rm.foreign["route_missing_edge"], "route")],
         ATT, carrier=reason_coords,
         notes="Different records by construction (a single record cannot fail both ways in one world); carrier = "
               "attempt.last_reason.* only. The rows also differ in the records' own payload facts (state).")
    move3 = Action("move", {"destination": 3})
    r = Run(make_spec(), address_seed=42)
    r.inspect_all()
    for k, a in sel + [cs("A1", 0), cs("B1", 2), CA]:
        r.step(k, **a)
    r.step("move", destination=3)
    assert r.o.feedback.get("reason") == "deadline"
    r.step("think")
    r2 = Run(make_spec(), address_seed=42)
    r2.inspect_all()
    for k, a in sel + [cs("A1", 0), cs("B1", 2), CA, TH, TH]:
        r2.step(k, **a)
    pair(R, "6-move-deadline-vs-untried-after1", 6, "move(3) rejected for deadline, then think vs never tried (step-matched)",
         r.o, [move3], r2.o, [move3], ATT, carrier="state",
         notes="A catalogue move can only be rejected for deadline/travel: moves are offered only along known, "
               "existing edges, so missing_edge arises only from use_return of a route (pair above).")

    # ---- 7: identical retry vs never tried; identical retry vs retry after dependencies changed
    ha = history([ci("A1"), ci("B3"), CP, TH])
    hb = history([ci("A1"), ci("B3"), TH, TH])
    pair(R, "7a", 7, "commit_pending already rejected on this exact state (identical retry) vs never attempted (same state)",
         ha.o, [commit], hb.o, [commit], ATT, carrier="state")
    ha = history([ci("A1"), ci("B3"), CP, ci("A3"), ci("B1"), TH])      # attempted at {A1,B3}; now {A3,B1}
    hb = history([ci("A1"), ci("B3"), ci("A3"), ci("B1"), CP, TH])      # attempted at {A3,B1}; now {A3,B1}
    assert [x["reason"] for x in ha.o.attempts] == [x["reason"] for x in hb.o.attempts] == ["capacity"]
    pair(R, "7b", 7, "retry after dependencies changed (rejected at another pending set, same reason) vs identical retry",
         ha.o, [commit], hb.o, [commit], ATT, carrier="state",
         checks={"same_pending": ha.o.pending == hb.o.pending})
    ev = ("slot_closed", 0, 4)
    steps = sel + [cs("A1", 0), cs("B1", 0), CA]
    hx = history(steps, make_spec(event=(ev[0], 16, ev[2])))
    hy = history(steps, make_spec())
    hx.pad_to(17)
    hy.pad_to(17)
    assert hx.o.events and not hy.o.events
    pair(R, "7c", 7, "commit_assignment rejected (slot_conflict), then a slot requirement revision (event) vs unchanged",
         hx.o, [commit_a], hy.o, [commit_a], ATT, carrier=["attempt.dependencies_changed"],
         notes="The observation differs (event) by design; carrier = the dependencies-changed coordinate.")

    # ---- 8: committed vs revoked; which requirement changed
    evs = {"capacity_reduced(3) revokes selection": ("capacity_reduced", 40, 3),
           "capacity_reduced(5) no revocation": ("capacity_reduced", 40, 5),
           "slot_closed(2) revokes assignment": ("slot_closed", 40, 2),
           "slot_closed(4) no revocation": ("slot_closed", 40, 4),
           "edge_closed(1,2)": ("edge_closed", 40, (1, 2)),
           "deadline_moved(8)": ("deadline_moved", 40, 8)}
    states = {k: after_event(e)[0].o for k, e in evs.items()}
    unc = [Action("uncommit", {"target": "select"}), Action("uncommit", {"target": "assign"})]
    for pid, a, b in (("8a", "capacity_reduced(3) revokes selection", "capacity_reduced(5) no revocation"),
                      ("8b", "slot_closed(2) revokes assignment", "slot_closed(4) no revocation"),
                      ("8c", "capacity_reduced(5) no revocation", "slot_closed(4) no revocation"),
                      ("8d", "edge_closed(1,2)", "deadline_moved(8)"),
                      ("8e", "capacity_reduced(5) no revocation", "edge_closed(1,2)"),
                      ("8f", "slot_closed(4) no revocation", "deadline_moved(8)")):
        pair(R, pid, 8, f"after event: {a} vs {b}", states[a], unc, states[b], unc, ALL, carrier="observation")

    # ---- 9: renaming invariance (world-level: items + record handles; observation-level: problem handles)
    names = {"A1": "i884120", "A2": "i019553", "A3": "i500001", "B1": "i777310", "B2": "i302948", "B3": "i640072"}
    def rename_spec(spec):
        m = names
        items = tuple(replace(x, handle=m[x.handle]) for x in spec.items)
        foreign = []
        for f in spec.foreign:
            f = deepcopy(f)
            p = f["snapshot"]["problem"]
            for key in (("handles", "excluded") if f["primitive"] == "constrained_subset" else ("items",)):
                if key in p:
                    p[key] = [m[h] for h in p[key]]
            if "selection_id" in f["depends_on"]:
                f["depends_on"]["selection_id"] = selection_id([m[h] for h in p["items"]])
            foreign.append(f)
        return replace(spec, items=items, incompatible=tuple((m[a], m[b]) for a, b in spec.incompatible),
                       foreign=tuple(foreign),
                       planted=(tuple(m[h] for h in spec.planted[0]), tuple((m[h], s) for h, s in spec.planted[1]),
                                spec.planted[2]))
    spec_a = make_spec(event=("capacity_reduced", 30, 5), foreign_kinds=FOREIGN_KINDS)
    spec_b = rename_spec(spec_a)
    ra, rb = Run(spec_a, address_seed=51), Run(spec_b, address_seed=9_999)
    ref = DepReference("reuse")
    rename_ok, steps_checked, first_mismatch = True, 0, None
    while not ra.o.done and steps_checked < 80:
        rec_map = {x["handle"]: y["handle"] for x, y in zip(ra.o.records, rb.o.records)}
        def mapa(a, rec_map=rec_map):
            args = dict(a.arguments)
            for k in ("item", "target"):
                if args.get(k) in names:
                    args[k] = names[args[k]]
            if args.get("handle") in rec_map:
                args["handle"] = rec_map[args["handle"]]
            return Action(a.kind, args)
        for v in VERSIONS:
            ca, xa, ma = frame(ra.o, v)
            cb, xb, mb = frame(rb.o, v)
            if xa != xb or [mapa(a) for a in ca] != cb or ma != mb:
                rename_ok = False
                first_mismatch = first_mismatch or {"step": ra.o.step, "version": v,
                                                    "obs": _diff(xa, xb, OBSERVATION_NAMES_D1)}
        act = ref.choose(ra.o)
        ra.step(act.kind, **act.arguments)
        b = mapa(act)
        rb.step(b.kind, **b.arguments)
        steps_checked += 1
    R.append({"id": "9a", "distinction": 9, "description": "consistent renaming of item handles, record handles "
              "(address seed) and foreign snapshots/selection ids: full observation + catalogue + matrix identical "
              "at every step of a dep_reuse trajectory (with a capacity event and 11 foreign records)",
              "carrier": "state", "expected": NONE, "versions": {v: {"distinguishable": not rename_ok} for v in VERSIONS},
              "preconditions": {"steps_checked": steps_checked, "first_mismatch": first_mismatch,
                                "success": ra.env.evaluate()["verified_success"]}, "pass": rename_ok and steps_checked > 20,
              "notes": ""})
    # problem-handle spelling: permute own/foreign problem names in the observation
    o = run.o
    keys = list(o.problems)
    perm = dict(zip(keys, reversed(keys)))       # own drafts get problem_0.., foreign get the high numbers
    o2 = rename_problems(o, perm)
    mp = lambda a: Action(a.kind, {**a.arguments, "problem": perm[a.arguments["problem"]]}) \
        if "problem" in a.arguments else a  # noqa: E731
    comp = compare(o, [], o2, [], "state", map_action=mp)
    ok = all(not comp[v]["distinguishable"] for v in VERSIONS)
    R.append({"id": "9b", "distinction": 9, "description": "problem-handle spelling: reverse the problem_n names so "
              "own drafts and foreign problems swap spellings (records, attempts and action keys renamed consistently)",
              "carrier": "state", "expected": NONE, "versions": comp, "preconditions": {"problems": len(keys)},
              "pass": ok, "notes": "Foreign vs own cannot leak through 'problem_' spelling or numbering."})

    # ---- 10: hidden-state leakage
    def trajectory_identical(spec_a, spec_b, script, address_seed=61):
        ra, rb = Run(spec_a, address_seed), Run(spec_b, address_seed)
        diffs = []
        for k, a in [("think", {})] + script:
            if (k, a) != ("think", {}) or ra.o.step:
                ra.step(k, **a)
                rb.step(k, **a)
            for v in VERSIONS:
                ca, xa, ma = frame(ra.o, v)
                cb, xb, mb = frame(rb.o, v)
                if xa != xb or ca != cb or ma != mb:
                    rows = sorted({n for x, y in zip(ma, mb) for n in _diff(x, y, CANDIDATE_NAMES_D1)}) if ca == cb else None
                    diffs.append({"step": ra.o.step, "version": v, "obs": _diff(xa, xb, OBSERVATION_NAMES_D1),
                                  "rows": rows, "same_catalog": ca == cb,
                                  "differing_actions": sorted({json.dumps([c.kind, dict(c.arguments)]) for c, x, y in zip(ca, ma, mb) if x != y})[:6] if ca == cb else None})
        return diffs

    def leak(pid, desc, spec_a, spec_b, script, notes=""):
        diffs = trajectory_identical(spec_a, spec_b, script)
        by_v = {v: {"distinguishable": any(d["version"] == v for d in diffs),
                    "first_difference": next((d for d in diffs if d["version"] == v), None)} for v in VERSIONS}
        R.append({"id": pid, "distinction": 10, "description": desc, "carrier": "state", "expected": NONE,
                  "versions": by_v, "preconditions": {}, "pass": not diffs, "notes": notes})

    a1_hidden = tuple(replace(x, weight=6, price=5, duration=4, slots=(0, 4)) if x.handle == "A1" else x for x in ITEMS)
    script_a = [("inspect", {"target": "requirements"}), ("inspect", {"target": "map"})] + \
        [("inspect", {"target": h}) for h in ("A2", "A3", "B1", "B2", "B3")] + \
        [("choose_item", {"item": "A1"}), ("choose_item", {"item": "B1"}), ("start_subset", {"handle": "problem_0"}),
         ("add_constraint", {"problem": "problem_0", "constraint": "capacity"}), ("call", {"problem": "problem_0", "budget": 128})]
    leak("10a", "uninspected item A1 (weight/price/duration/slots) differs; everything else inspected; A1 pending; own subset draft+call",
         make_spec(), make_spec(items=a1_hidden), script_a)
    chain_script = [("inspect", {"target": "requirements"})] + [("inspect", {"target": x.handle}) for x in ITEMS] + \
        [("inspect", {"target": "map"}), ("choose_item", {"item": "A1"}), ("choose_item", {"item": "B1"}),
         ("commit_pending", {}), ("choose_slot", {"item": "A1", "slot": 0}), ("choose_slot", {"item": "B1", "slot": 2}),
         ("commit_assignment", {}), ("build_route", {"handle": "problem_0"}), ("call", {"problem": "problem_0", "budget": 128}),
         ("think", {})]
    leak("10b", "future event (slot_closed@60 vs deadline_moved@60) and planted solution differ; full chain before the event",
         make_spec(event=("slot_closed", 60, 2)),
         make_spec(event=("deadline_moved", 60, 8), planted=(("A2", "B3"), (("A2", 0), ("B3", 2)), (0, 3))), chain_script)
    b1_short = tuple(replace(x, duration=1) if x.handle == "B1" else x for x in ITEMS)
    sa, sb = make_spec(foreign_kinds=("csp_current",)), make_spec(items=b1_short, foreign_kinds=("csp_current",))
    h = Run(sa).foreign["csp_current"]
    hb_ = Run(sb).foreign["csp_current"]
    assert h == hb_
    leak("10c", "foreign csp record for {A1,B1}; B1's hidden duration differs (2 vs 1); nothing inspected; retrieve the record",
         sa, sb, [("retrieve", {"handle": h}), ("inspect", {"target": "requirements"})],
         notes="Foreign snapshots carry item facts (durations, domains). Leak if the encoder surfaces them before inspection.")
    edges_b = tuple((u, v, 2) if (u, v) == (2, 3) else (u, v, w) for u, v, w in EDGES)
    sa, sb = make_spec(foreign_kinds=("route_current",)), make_spec(edges=edges_b, foreign_kinds=("route_current",))
    h = Run(sa).foreign["route_current"]
    leak("10d", "foreign current-route record; hidden map edge (2,3) weight differs (1 vs 2); map NOT inspected; retrieve",
         sa, sb, [("retrieve", {"handle": h}), ("inspect", {"target": "requirements"})],
         notes="Route snapshots/payloads carry map facts.")
    a1_heavier = tuple(replace(x, weight=3) if x.handle == "A1" else x for x in ITEMS)
    sa, sb = make_spec(foreign_kinds=("subset_current",)), make_spec(items=a1_heavier, foreign_kinds=("subset_current",))
    h = Run(sa).foreign["subset_current"]
    leak("10e", "foreign current-subset record; hidden A1 weight differs (2 vs 3); A1 NOT inspected (others are); retrieve",
         sa, sb, [("inspect", {"target": "requirements"})] + [("inspect", {"target": x}) for x in ("A2", "A3", "B1", "B2", "B3")]
         + [("retrieve", {"handle": h})])
    leak("10f", "as 10c but the record is NOT retrieved (metadata only: status, certificate, snapshot options, work units)",
         make_spec(foreign_kinds=("csp_current",)), make_spec(items=b1_short, foreign_kinds=("csp_current",)),
         [("inspect", {"target": "requirements"})])

    # ---- 11: mask definitions and dimensions
    o = run.o
    dims = {v: (len(frame(o, v)[1]), len(frame(o, v)[2][0])) for v in VERSIONS}
    return {"pairs": R, "catalog_checks": catalog, "dimensions": dims}


def rename_problems(o, perm):
    """Observation-level rename of problem handles (keys, record/retrieved 'problem', call attempts + keys)."""
    problems = {perm[k]: deepcopy(v) for k, v in o.problems.items()}
    recs = tuple({**deepcopy(r), "problem": perm.get(r["problem"], r["problem"])} for r in o.records)
    retrieved = {h: {**deepcopy(r), "problem": perm.get(r["problem"], r["problem"])} for h, r in o.retrieved.items()}
    attempts = []
    for at in o.attempts:
        at = deepcopy(at)
        if "problem" in at["arguments"]:
            at["arguments"]["problem"] = perm[at["arguments"]["problem"]]
            at["action_key"] = action_key(Action(at["action_kind"], at["arguments"]))
        attempts.append(at)
    return replace(o, problems=problems, records=recs, retrieved=retrieved, attempts=tuple(attempts))


def summarize(result) -> list[dict]:
    rows = {}
    for p in result["pairs"]:
        d = rows.setdefault(p["distinction"], {"distinction": p["distinction"], "pairs": 0, "passed": 0,
                                                "d1_fail": [], "ablation_fail": []})
        d["pairs"] += 1
        d["passed"] += bool(p["pass"])
        v = p["versions"]
        if p["expected"].get("d1") is not None and v["d1"]["distinguishable"] != p["expected"]["d1"]:
            d["d1_fail"].append(p["id"])
        for a in ("d1-noapp", "d1-noattempt"):
            if p["expected"].get(a) is not None and v[a]["distinguishable"] != p["expected"][a]:
                d["ablation_fail"].append(f"{p['id']}@{a}")
    return [rows[k] for k in sorted(rows)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/campaigns/extended-03/preflight-audit.json")
    args = ap.parse_args(argv)
    result = audit()
    result["summary"] = summarize(result)
    result["masks"] = {v: {"observation": [OBSERVATION_NAMES_D1[i] for i in m[0]],
                           "candidate": [CANDIDATE_NAMES_D1[i] for i in m[1]],
                           "observation_indices": list(m[0]), "candidate_indices": list(m[1])}
                       for v, m in __import__("topoformer.campaign03_depworld", fromlist=["x"]).FEATURE_MASKS.items()}
    Path(args.out).write_text(json.dumps(result, indent=1, default=str) + "\n")
    for p in result["pairs"]:
        flag = "PASS" if p["pass"] else "FAIL"
        cells = " ".join(f"{v}={'D' if p['versions'][v]['distinguishable'] else '='}" for v in VERSIONS)
        coords = p["versions"]["d1"].get("row_diff") or p["versions"]["d1"].get("observation_diff") or \
            (p["versions"]["d1"].get("first_difference") or {})
        print(f"{flag} {p['id']:<44} {cells}  {coords}")
    for s in result["summary"]:
        print(s)
    return result


if __name__ == "__main__":
    main()
