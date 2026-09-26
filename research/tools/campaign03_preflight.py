"""extended-03 Stage B preflight: what do the ENCODED depworld inputs preserve? (CPU, in-process solver.)

Independent audit of the d1 encoder and the P1 input ablations d1-noapp (X3) and d1-noattempt (X4).

Method (extended-02 encoded-counterfactuals, extended). Every pair is built by stepping real
``DepWorkshop`` instances (hand-built specs, in-process ``campaign02_protocol.execute`` through
``depworld_executor``) and compared on the exact tensors the policy consumes:
``campaign02_training.public_frame(observation, version)`` -> (candidate set, observation vector, candidate
matrix). Nothing is compared on raw observations or on the encoder's helper functions.

A pair declares its *carrier*, the part of the input that must carry the distinction:
  "row"          the named candidate row(s) (acts_a[i] in state a vs acts_b[i] in state b);
  "observation"  the observation vector;
  "state"        observation + candidate set + full candidate matrix;
  [names]        only these candidate coordinates of the named rows.
Every pair records the observation diff, the row diff, whether the candidate sets are equal and how
many matrix rows differ, for every version. Expected verdicts per version:
  True  must differ;  False  must be identical;
  "residual"  (ablations only) the ablated relations must be gone: the pair may still differ, but only
              on the declared payload feasibility facts that the ablation keeps by design.

Distinctions (Stage B brief, decisions/handoff; numbering as in the brief):
  1 own current applicable result vs a same-type result of another registered (foreign) problem
  2 current vs stale result of the SAME problem after a requirement change
  3 old-but-still-applicable vs stale (and old-applicable must look exactly like a fresh result)
  4 foreign applicable vs foreign not applicable
  5 wrong type vs right type for a use
  6 rejection reasons, at the next decision and after intervening actions
  7 identical retry vs never tried / vs retry after dependencies changed
  8 committed vs revoked after an event; which requirement changed
  9 invariance: renaming of opaque handles; ownership (own vs foreign applicable); problem-name spelling
  10 no hidden-state leakage (+ a disclosure probe of the public foreign-snapshot channel)
  11 ablations: dimensions, the per-pair expectations above, and functional invariance tests
     (d1-noapp is invariant to the relation values; d1-noattempt is invariant to the attempt log;
     both agree with d1 on every other coordinate).

Usage: PYTHONPATH=src python research/tools/campaign03_preflight.py \
           [--out research/campaigns/extended-03/review/stage-b-preflight.json]
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensegra import campaign03_depworld as dep  # noqa: E402
from tensegra.campaign02_protocol import execute  # noqa: E402
from tensegra.campaign02_training import public_frame  # noqa: E402
from tensegra.campaign02_world import Action  # noqa: E402
from tensegra.campaign03_depworld import (CANDIDATE_NAMES_D1, CSP_CONSTRAINTS, FEATURE_MASKS,  # noqa: E402
    FEATURE_VERSIONS, OBSERVATION_NAMES_D1, SUBSET_CONSTRAINTS, USES, DepItem, DepReference, DepSpec, DepWorkshop,
    action_catalog, action_key, call_budgets, csp_problem, depworld_executor, generate_depworld, route_problem,
    selection_id, subset_problem, _item_row)

EXEC = partial(depworld_executor, execute_call=execute)
VERSIONS = FEATURE_VERSIONS  # ("d1", "d1-noapp", "d1-noattempt")
ALL = {v: True for v in VERSIONS}
NONE = {v: False for v in VERSIONS}
APP = {"d1": True, "d1-noapp": False, "d1-noattempt": True}          # applicability (1-4), metadata only
APP_R = {"d1": True, "d1-noapp": "residual", "d1-noattempt": True}   # applicability, payload retrieved
ATT = {"d1": True, "d1-noapp": True, "d1-noattempt": False}          # attempt memory (6-7)
# Payload feasibility facts that d1-noapp keeps by design (not request/dependency relations).
NOAPP_KEPT_FACTS = ("payload.route_distance", "payload.route_meets_deadline", "payload.finish_or_total_duration",
                    "payload.meets_bound_or_selection_valid")
# Candidate coordinates that read the context's need_bound (dropped by d1-noapp with the route bound).
NEED_BOUND_DERIVED = ("constraint.bound_within_need", "slot.meets_bound", "commit.assign_meets_bound",
                      "payload.meets_bound_or_selection_valid")
RELATION_KEYS = ("request_match", "canonical_match", "dependency_match", "requirements_match", "selection_match")
LEAK_SEED = 61  # address seed of the leakage pairs (record handles depend on it)
DEV_SEED = 2_000_000_000  # development seeds only (sealed P1 seeds are >= 110,000,000 and < 2e9)

# ---------------------------------------------------------------------------------------------
# A small hand-built world with exactly controllable rejections, records and events
# ---------------------------------------------------------------------------------------------
# cap 6, funds 7, incompatible (A2,B1). {A1,B1}: valid (w4 p4). {A1,B3}: capacity (w7). {A1,B2}: funds (p9).
# {A2,B1}: incompatibility. {A3,B1}: capacity (w7). Slot 3 closed. Assignment A1:0, B1:2 -> finish 4.
# Map 0->1->2->3 (4), 0->1->3 (5), 0->3 (6); deadline 9, so move(3) from 0 misses the deadline.
ITEMS = (DepItem("A1", 0, 2, 2, 2, (0, 1, 2)), DepItem("A2", 0, 1, 1, 1, (0, 3)), DepItem("A3", 0, 5, 1, 1, (1,)),
         DepItem("B1", 1, 2, 2, 2, (0, 2, 3)), DepItem("B2", 1, 1, 7, 1, (1, 4)), DepItem("B3", 1, 5, 1, 3, (0, 2)))
EDGES = ((0, 1, 2), (0, 3, 6), (1, 2, 1), (1, 3, 3), (2, 3, 1))
BASE = dict(items=ITEMS, categories=(0, 1), slots=5, capacity=6, funds=7, incompatible=(("A2", "B1"),), deadline=9,
            edges=EDGES, locations=4, destination=3, closed_slots=(3,),
            planted=(("A1", "B1"), (("A1", 0), ("B1", 2)), (0, 1, 2, 3)))
# Hand-built foreign kinds: the generator's eight plus request-only / physically-failing variants.
FOREIGN_KINDS = ("subset_current", "subset_earlier", "subset_version_mismatch", "subset_other_order",
                 "route_current", "route_other_goal", "route_earlier", "route_missing_edge", "route_slow",
                 "csp_current", "csp_other_selection")


def _solve(snapshot):
    return depworld_executor(snapshot["primitive"], deepcopy(snapshot["problem"]), 100000, execute_call=execute)


def foreign_records(spec: DepSpec, kinds=FOREIGN_KINDS) -> tuple:
    """Registered foreign problems + solved records built from the spec's TRUE facts (as the generator does)."""
    rows = [_item_row(x) for x in spec.items]
    by = {r["handle"]: r for r in rows}
    req = {"capacity": spec.capacity, "funds": spec.funds, "incompatible": [list(p) for p in spec.incompatible]}
    subset_v1 = {"requirements_version": {"capacity": 1, "funds": 1, "incompatible": 1}}
    subset_v0 = {"requirements_version": {"capacity": 0, "funds": 0, "incompatible": 0}}
    cat = lambda hs: sorted((by[h] for h in hs), key=lambda r: r["category"])  # noqa: E731
    first = {c: next(x.handle for x in spec.items if x.category == c) for c in spec.categories}
    cur_sel = [first[c] for c in spec.categories]                       # A1, B1 (the chain's selection)
    other_sel = [[x.handle for x in spec.items if x.category == 0][1], [x.handle for x in spec.items if x.category == 1][2]]
    out = []
    for kind in kinds:
        if kind == "subset_current":            # applicable
            snap, deps = subset_problem(rows, req, SUBSET_CONSTRAINTS), subset_v1
        elif kind == "subset_earlier":          # request + dependency mismatch (earlier shift)
            snap, deps = subset_problem(rows, {**req, "capacity": req["capacity"] + 2}, SUBSET_CONSTRAINTS), subset_v0
        elif kind == "subset_version_mismatch":  # dependency-only mismatch
            snap, deps = subset_problem(rows, req, SUBSET_CONSTRAINTS), subset_v0
        elif kind == "subset_other_order":      # request-only mismatch
            snap, deps = subset_problem(rows, {**req, "capacity": req["capacity"] + 2}, SUBSET_CONSTRAINTS), subset_v1
        elif kind == "route_current":           # applicable (at the start position)
            snap, deps = route_problem(spec.edges, spec.locations, spec.start, spec.destination), {"requirements_version": {"map": 1}}
        elif kind == "route_other_goal":        # request-only mismatch
            snap, deps = route_problem(spec.edges, spec.locations, spec.start, 2), {"requirements_version": {"map": 1}}
        elif kind in ("route_earlier", "route_missing_edge"):   # an extra edge 0->2 that does not exist
            snap = route_problem(list(spec.edges) + [(0, 2, 1)], spec.locations, spec.start, spec.destination)
            deps = {"requirements_version": {"map": 0 if kind == "route_earlier" else 1}}
        elif kind == "route_slow":              # 0->3 believed to cost 1 (really 6): request-only mismatch
            edges = [(u, v, 1 if (u, v) == (0, 3) else w) for u, v, w in spec.edges]
            snap, deps = route_problem(edges, spec.locations, spec.start, spec.destination), {"requirements_version": {"map": 1}}
        elif kind == "csp_current":             # applicable once {A1,B1} is committed
            snap = csp_problem(cat(cur_sel), spec.closed_slots, CSP_CONSTRAINTS)
            deps = {"requirements_version": {"slots": 1}, "selection_id": selection_id(cur_sel)}
        elif kind == "csp_other_selection":     # request + selection-dependency mismatch
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

    def steps(self, script):
        for kind, args in script:
            self.step(kind, **args)
        return self.o

    def name(self):
        return f"problem_{len(self.o.problems)}"

    def inspect_all(self):
        self.step("inspect", target="requirements")
        for r in self.o.item_inventory:
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
        assert self.o.feedback["status"] == "success", self.o.feedback
        return self.o.feedback["return"]

    def pad_to(self, step):
        while self.o.step < step:
            self.step("think")
        return self.o


def chain(run: Run):
    """Full public chain: inspect, commit {A1,B1}, assign A1:0 B1:2, and own records of all three primitives."""
    run.inspect_all()
    assert run.commit("A1", "B1").feedback["status"] == "success"
    assert run.assign(A1=0, B1=2).feedback["status"] == "success"
    return {"subset": run.compute("constrained_subset"), "csp": run.compute("csp"),
            "route": run.compute("shortest_path")}


# ---------------------------------------------------------------------------------------------
# Encoded comparison (the exact policy input path)
# ---------------------------------------------------------------------------------------------

def frame(o, version):
    return public_frame(o, version)


def _diff(x, y, names):
    return [names[i] for i, (p, q) in enumerate(zip(x, y)) if p != q]


def _key(a):
    return json.dumps([a.kind, dict(a.arguments)], sort_keys=True)


def compare(o_a, acts_a, o_b, acts_b, carrier="row", map_action=None):
    """Per version: observation diff, row diffs (acts_a[i] vs acts_b[i]), candidate-set and matrix diff."""
    out = {}
    for v in VERSIONS:
        ca, xa, ma = frame(o_a, v)
        cb, xb, mb = frame(o_b, v)
        assert len(xa) == len(xb) == len(OBSERVATION_NAMES_D1)
        assert all(len(r) == len(CANDIDATE_NAMES_D1) for r in ma + mb)
        obs_diff = _diff(xa, xb, OBSERVATION_NAMES_D1)
        row_diff = sorted({n for a, b in zip(acts_a, acts_b)
                           for n in _diff(ma[ca.index(a)], mb[cb.index(b)], CANDIDATE_NAMES_D1)})
        mapped = [map_action(a) for a in ca] if map_action else ca
        same_catalog = [_key(a) for a in mapped] == [_key(b) for b in cb]
        matrix_rows = sum(r != s for r, s in zip(ma, mb)) if same_catalog else None
        if carrier == "row":
            dist = bool(row_diff)
        elif carrier == "observation":
            dist = bool(obs_diff)
        elif carrier == "state":
            dist = bool(obs_diff) or not same_catalog or bool(matrix_rows)
        else:
            dist = any(n in carrier for n in row_diff)
        out[v] = {"distinguishable": dist, "observation_diff": obs_diff, "row_diff": row_diff,
                  "same_catalog": same_catalog, "matrix_rows_differing": matrix_rows}
    return out


def _verdict(expected, got):
    if expected is None:
        return True
    if expected == "residual":
        return not got["distinguishable"] or (set(got["row_diff"]) <= set(NOAPP_KEPT_FACTS) and not got["observation_diff"])
    return got["distinguishable"] == expected


def pair(results, pid, distinction, description, o_a, acts_a, o_b, acts_b, expected, carrier="row",
         map_action=None, notes="", checks=None, construction=""):
    comp = compare(o_a, acts_a, o_b, acts_b, carrier, map_action)
    per_version = {v: _verdict(expected.get(v), comp[v]) for v in VERSIONS}
    pre_ok = all(checks.values()) if checks else True
    results.append({"id": pid, "distinction": distinction, "description": description,
                    "construction": construction,
                    "actions": [[_key(a), _key(b)] for a, b in zip(acts_a, acts_b)],
                    "carrier": carrier if isinstance(carrier, str) else list(carrier),
                    "expected": expected, "versions": comp, "version_pass": per_version,
                    "preconditions": checks or {}, "pass": pre_ok and all(per_version.values()), "notes": notes})
    return results[-1]


def same_flags(o, *handles):
    recs = {r["handle"]: r for r in o.records}
    keys = [(recs[h]["status"], recs[h]["certificate_valid"], h in o.retrieved) for h in handles]
    return all(k == keys[0] for k in keys)


def use(h, as_):
    return Action("use_return", {"handle": h, "as": as_})


def catalog_check(o) -> dict:
    """The candidate SET must contain every action the distinctions need (no hidden masks)."""
    acts = public_frame(o, "d1")[0]
    s = {_key(a) for a in acts}
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
    return {"candidates": len(acts), "duplicates": len(acts) - len(s), "missing": missing,
            "records": len(o.records), "problems": len(o.problems),
            "pass": not missing and len(acts) == len(s)}


# ---------------------------------------------------------------------------------------------
# The paired-state audit
# ---------------------------------------------------------------------------------------------

def audit_pairs() -> tuple[list[dict], dict, DepWorkshop]:
    R: list[dict] = []
    catalog: dict = {}

    # ---- 1, 4, 5, 9 (ownership): one state with own records of every primitive + 11 foreign records
    run = Run(make_spec(foreign_kinds=FOREIGN_KINDS), address_seed=11)
    own = chain(run)
    F = run.foreign
    catalog["full_chain_with_foreign"] = catalog_check(run.o)
    for label, retrieve in (("unretrieved", False), ("retrieved", True)):
        if retrieve:
            for h in list(own.values()) + list(F.values()):
                run.step("retrieve", handle=h)
            catalog["full_chain_with_foreign_retrieved"] = catalog_check(run.o)
        o = run.o
        exp_app = APP_R if retrieve else APP
        con = ("hand world, 11 registered foreign records; inspect all, commit {A1,B1}, assign A1:0 B1:2, own "
               "subset/csp/route drafts+calls" + ("; then retrieve every record" if retrieve else "; nothing retrieved"))
        for pid, own_h, f_label, as_ in (("1a", own["subset"], "subset_earlier", "select"),
                                         ("1b", own["subset"], "subset_other_order", "select"),
                                         ("1c", own["subset"], "subset_version_mismatch", "select"),
                                         ("1d", own["csp"], "csp_other_selection", "assign"),
                                         ("1e", own["route"], "route_other_goal", "route"),
                                         ("1f", own["route"], "route_earlier", "route")):
            pair(R, f"{pid}-{label}", 1,
                 f"use_return as {as_}: own current result vs foreign registered same-type ({f_label}), {label}",
                 o, [use(own_h, as_)], o, [use(F[f_label], as_)], exp_app, construction=con,
                 checks={"same_status_cert_retrieved": same_flags(o, own_h, F[f_label]),
                         "all_registered": all(r["problem"] in o.problems for r in o.records)})
        for pid, a_label, b_label, as_, why in (
                ("4a", "subset_current", "subset_version_mismatch", "select", "dependency-only mismatch"),
                ("4b", "subset_current", "subset_earlier", "select", "request + dependency mismatch"),
                ("4c", "subset_current", "subset_other_order", "select", "request-only mismatch"),
                ("4d", "route_current", "route_other_goal", "route", "request-only mismatch (other goal)"),
                ("4e", "route_current", "route_earlier", "route", "earlier map, version 0"),
                ("4f", "route_current", "route_slow", "route", "request-only mismatch (wrong edge weight)"),
                ("4g", "csp_current", "csp_other_selection", "assign", "other selection")):
            pair(R, f"{pid}-{label}", 4,
                 f"use_return as {as_}: foreign applicable ({a_label}) vs foreign not applicable ({b_label}): {why}, {label}",
                 o, [use(F[a_label], as_)], o, [use(F[b_label], as_)], exp_app, construction=con,
                 checks={"same_status_cert_retrieved": same_flags(o, F[a_label], F[b_label])})
        for pid, own_h, f_label, as_ in (("9o-a", own["subset"], "subset_current", "select"),
                                         ("9o-b", own["csp"], "csp_current", "assign"),
                                         ("9o-c", own["route"], "route_current", "route")):
            pair(R, f"{pid}-{label}", 9,
                 f"ownership/age invariance: own current {as_} result vs foreign APPLICABLE {f_label}, {label}",
                 o, [use(own_h, as_)], o, [use(F[f_label], as_)], NONE, construction=con,
                 notes="Age and ownership are irrelevant to applicability: the rows must be identical.",
                 checks={"same_status_cert_retrieved": same_flags(o, own_h, F[f_label])})
        pair(R, f"5a-{label}", 5, f"use_return as select: csp record (wrong type) vs subset record (right type), {label}",
             o, [use(own["csp"], "select")], o, [use(own["subset"], "select")], ALL, construction=con)
        pair(R, f"5b-{label}", 5, f"use_return as route: foreign applicable csp (wrong type) vs applicable route, {label}",
             o, [use(F["csp_current"], "route")], o, [use(F["route_current"], "route")], ALL, construction=con)
        pair(R, f"5c-{label}", 5, f"same subset record: use_return as assign (wrong) vs as select (right), {label}",
             o, [use(own["subset"], "assign")], o, [use(own["subset"], "select")], ALL, construction=con,
             notes="Trivially separated by the use one-hot; 5a/5b are the informative type pairs.")
        pair(R, f"5d-{label}", 5,
             f"type relation only: use_return as select of a csp record vs of a route record (both wrong type), {label}",
             o, [use(own["csp"], "select")], o, [use(own["route"], "select")], ALL, construction=con,
             notes="Both wrong type; they differ in primitive one-hot. Reported for completeness.")

    # ---- 2 and 3: events after the chain. Paired worlds share the prefix exactly (event fires at step 40).
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

    def recs(o):
        return {x["handle"]: x for x in o.records}

    ev_con = "hand world; full chain (24 steps); think until the step-40 event; then recompute on the same draft"
    for label, ret in (("unretrieved", ()), ("retrieved", ("subset", "subset_new"))):
        r, h = after_event(("capacity_reduced", 40, 5), recompute_subset, ret)
        rr = recs(r.o)
        pair(R, f"2a-{label}", 2, f"own subset result from before capacity_reduced(6->5) (stale: request+dependency) "
             f"vs the same draft re-called after the event, {label}",
             r.o, [use(h["subset"], "select")], r.o, [use(h["subset_new"], "select")], APP_R if ret else APP,
             construction=ev_con,
             checks={"same_status_cert_retrieved": same_flags(r.o, h["subset"], h["subset_new"]),
                     "same_problem": rr[h["subset"]]["problem"] == rr[h["subset_new"]]["problem"]})
    r, h = after_event(("capacity_reduced", 40, 6), recompute_subset)
    rr = recs(r.o)
    pair(R, "2b", 2, "own subset result from before a same-value capacity event (dependency-only stale) vs re-called",
         r.o, [use(h["subset"], "select")], r.o, [use(h["subset_new"], "select")], APP, construction=ev_con,
         checks={"same_status_cert_retrieved": same_flags(r.o, h["subset"], h["subset_new"]),
                 "same_problem": rr[h["subset"]]["problem"] == rr[h["subset_new"]]["problem"],
                 "same_snapshot": rr[h["subset"]]["problem_snapshot"] == rr[h["subset_new"]]["problem_snapshot"]})
    for label, ret in (("unretrieved", ()), ("retrieved", ("route", "route_new"))):
        r, h = after_event(("edge_closed", 40, (1, 2)), recompute_route, ret)
        pair(R, f"2c-{label}", 2, f"own route from before edge_closed(1,2) (stale) vs route recomputed after, {label}",
             r.o, [use(h["route"], "route")], r.o, [use(h["route_new"], "route")], APP_R if ret else APP,
             construction=ev_con + " (a new route draft: route drafts take no constraints)",
             checks={"same_status_cert_retrieved": same_flags(r.o, h["route"], h["route_new"])})
    r, h = after_event(("slot_closed", 40, 4), recompute_csp)
    rr = recs(r.o)
    pair(R, "2d", 2, "own csp from before slot_closed(4) (slot unused by the selection: dependency-only stale) vs re-called",
         r.o, [use(h["csp"], "assign")], r.o, [use(h["csp_new"], "assign")], APP, construction=ev_con,
         checks={"same_status_cert_retrieved": same_flags(r.o, h["csp"], h["csp_new"]),
                 "same_problem": rr[h["csp"]]["problem"] == rr[h["csp_new"]]["problem"],
                 "same_snapshot": rr[h["csp"]]["problem_snapshot"] == rr[h["csp_new"]]["problem_snapshot"]})

    # 3: old-but-applicable vs stale (paired worlds: edge_closed vs capacity_reduced, identical prefixes)
    for label, ret in (("unretrieved", ()), ("retrieved", ("subset", "route"))):
        re_, he = after_event(("edge_closed", 40, (1, 2)), retrieve=ret)
        rc, hc = after_event(("capacity_reduced", 40, 5), retrieve=ret)
        exp = APP_R if ret else APP
        pair(R, f"3a-{label}", 3, f"selection result from before edge_closed (still applicable) vs the same result "
             f"from before capacity_reduced (stale), {label}",
             re_.o, [use(he["subset"], "select")], rc.o, [use(hc["subset"], "select")], exp, construction=ev_con,
             notes="Paired worlds; the observation also differs (event kind, capacity) by design, so carrier = row.")
        pair(R, f"3b-{label}", 3, f"route from before capacity_reduced (still applicable) vs the same route from "
             f"before edge_closed (stale), {label}",
             rc.o, [use(hc["route"], "route")], re_.o, [use(he["route"], "route")], exp, construction=ev_con)
    r, h = after_event(("edge_closed", 40, (1, 2)), recompute_subset, ("subset", "subset_new"))
    pair(R, "3c", 3, "old-but-applicable selection result (pre-edge_closed) vs a fresh re-call after the event (same request)",
         r.o, [use(h["subset"], "select")], r.o, [use(h["subset_new"], "select")], NONE, construction=ev_con,
         notes="Must be IDENTICAL: an old applicable result must not look stale.")
    r, h = after_event(("capacity_reduced", 40, 5), recompute_route, ("route", "route_new"))
    pair(R, "3d", 3, "old-but-applicable route (pre-capacity_reduced) vs a fresh recompute after the event",
         r.o, [use(h["route"], "route")], r.o, [use(h["route_new"], "route")], NONE, construction=ev_con,
         notes="Must be IDENTICAL.")
    r, h = after_event(("slot_closed", 40, 4), recompute_subset)
    pair(R, "3e", 3, "old-but-applicable selection result (pre-slot_closed) vs a fresh re-call after the event",
         r.o, [use(h["subset"], "select")], r.o, [use(h["subset_new"], "select")], NONE, construction=ev_con,
         notes="Must be IDENTICAL (slots do not feed the selection instance).")

    # ---- 6: rejection reasons, at the next decision and after intervening actions
    def history(steps, spec=None, seed=31):
        r = Run(spec or make_spec(), address_seed=seed)
        r.inspect_all()
        r.steps(steps)
        return r

    ci = lambda h: ("choose_item", {"item": h})  # noqa: E731
    cs = lambda h, s: ("choose_slot", {"item": h, "slot": s})  # noqa: E731
    CP, CA, TH = ("commit_pending", {}), ("commit_assignment", {}), ("think", {})
    rej = {"capacity": [ci("A1"), ci("B3"), CP], "funds": [ci("A1"), ci("B2"), CP],
           "incompatibility": [ci("A2"), ci("B1"), CP]}
    fix = {"capacity": [ci("B1")], "funds": [ci("B1")], "incompatibility": [ci("A1")]}
    commit = Action("commit_pending")
    hcon = "hand world; inspect everything (8 steps); then the listed actions"
    for a, b in (("capacity", "funds"), ("capacity", "incompatibility"), ("funds", "incompatibility")):
        ha, hb = history(rej[a]), history(rej[b])
        assert ha.o.feedback.get("reason") == a and hb.o.feedback.get("reason") == b
        pair(R, f"6-{a}-{b}-immediate", 6, f"commit_pending rejected for {a} vs {b}: the next decision",
             ha.o, [commit], hb.o, [commit], ALL, carrier="state", construction=hcon,
             notes="Immediate: the last feedback reason (observation) differs; that is not attempt memory, so "
                   "both ablations keep it. The pending sets also differ.")
        ha, hb = history(rej[a] + fix[a]), history(rej[b] + fix[b])
        pair(R, f"6-{a}-{b}-after1", 6,
             f"commit_pending rejected for {a} vs {b}, then one repair choose_item to the SAME pending set",
             ha.o, [commit], hb.o, [commit], ATT, carrier="state", construction=hcon,
             checks={"same_pending": ha.o.pending == hb.o.pending, "same_step": ha.o.step == hb.o.step})
        ha, hb = history(rej[a] + fix[a] + [TH]), history(rej[b] + fix[b] + [TH])
        pair(R, f"6-{a}-{b}-after2", 6, f"commit_pending rejected for {a} vs {b}, repair + think (two intervening)",
             ha.o, [commit], hb.o, [commit], ATT, carrier="state", construction=hcon,
             checks={"same_pending": ha.o.pending == hb.o.pending})
    sel = [ci("A1"), ci("B1"), CP]
    rej_a = {"slot_conflict": sel + [cs("A1", 0), cs("B1", 0), CA], "slot_unavailable": sel + [cs("A1", 0), cs("B1", 3), CA]}
    commit_a = Action("commit_assignment")
    ha, hb = history(rej_a["slot_conflict"]), history(rej_a["slot_unavailable"])
    assert ha.o.feedback.get("reason") == "slot_conflict" and hb.o.feedback.get("reason") == "slot_unavailable"
    pair(R, "6-slot_conflict-slot_unavailable-immediate", 6,
         "commit_assignment rejected for slot_conflict vs slot_unavailable: the next decision",
         ha.o, [commit_a], hb.o, [commit_a], ALL, carrier="state", construction=hcon)
    ha, hb = history(rej_a["slot_conflict"] + [cs("B1", 2)]), history(rej_a["slot_unavailable"] + [cs("B1", 2)])
    pair(R, "6-slot_conflict-slot_unavailable-after1", 6,
         "commit_assignment rejected for slot_conflict vs slot_unavailable, then choose_slot to the SAME pending assignment",
         ha.o, [commit_a], hb.o, [commit_a], ATT, carrier="state", construction=hcon,
         checks={"same_pending_assignment": ha.o.pending_assignment == hb.o.pending_assignment})
    ha, hb = (history(rej_a["slot_conflict"] + [cs("B1", 2), TH]),
              history(rej_a["slot_unavailable"] + [cs("B1", 2), TH]))
    pair(R, "6-slot_conflict-slot_unavailable-after2", 6,
         "as above, plus one think (two intervening actions)",
         ha.o, [commit_a], hb.o, [commit_a], ATT, carrier="state", construction=hcon)
    # deliver (use_return of a route): deadline vs missing_edge. One record cannot fail both ways in one world,
    # so the rows are two different records and the carrier is the attempt reason coordinates.
    spec = make_spec(foreign_kinds=("route_slow", "route_missing_edge"))
    reason_coords = [n for n in CANDIDATE_NAMES_D1 if n.startswith("attempt.last_reason.")]
    runs = {}
    for lab in ("route_slow", "route_missing_edge"):
        r = Run(spec, address_seed=41)
        r.inspect_all()
        r.steps(sel + [cs("A1", 0), cs("B1", 2), CA])
        for h in r.foreign.values():
            r.step("retrieve", handle=h)
        r.step("use_return", handle=r.foreign[lab], **{"as": "route"})
        runs[lab] = (r, r.o.feedback.get("reason"))
        r.step("think")
    (rs, why_s), (rm, why_m) = runs["route_slow"], runs["route_missing_edge"]
    assert (why_s, why_m) == ("deadline", "missing_edge"), (why_s, why_m)
    pair(R, "6-deliver-deadline-missing_edge-after1", 6,
         "deliver (use_return as route) rejected for deadline vs missing_edge, after one think: the attempted rows",
         rs.o, [use(rs.foreign["route_slow"], "route")], rm.o, [use(rm.foreign["route_missing_edge"], "route")],
         ATT, carrier=reason_coords,
         construction="hand world + foreign route_slow/route_missing_edge; chain to a committed assignment; retrieve "
                      "both; deliver along one of them; think",
         notes="Carrier = attempt.last_reason.* only (the two records also differ in their own payload facts).")
    move3 = Action("move", {"destination": 3})
    r = Run(make_spec(), address_seed=42)
    r.inspect_all()
    r.steps(sel + [cs("A1", 0), cs("B1", 2), CA])
    r.step("move", destination=3)
    assert r.o.feedback.get("reason") == "deadline"
    r.step("think")
    r2 = Run(make_spec(), address_seed=42)
    r2.inspect_all()
    r2.steps(sel + [cs("A1", 0), cs("B1", 2), CA, TH, TH])
    pair(R, "6-move-deadline-vs-untried-after1", 6, "move(3) rejected for deadline, then think vs never tried (step-matched)",
         r.o, [move3], r2.o, [move3], ATT, carrier="state", construction=hcon,
         notes="A catalogue move can only be rejected for deadline/travel: moves are offered only along known, "
               "existing edges, so missing_edge arises only from use_return of a route (pair above).")

    # ---- 7: identical retry vs never tried; identical retry vs retry after dependencies changed
    ha = history([ci("A1"), ci("B3"), CP, TH])
    hb = history([ci("A1"), ci("B3"), TH, TH])
    pair(R, "7a", 7, "commit_pending already rejected on this exact state (identical retry) vs never attempted (same state)",
         ha.o, [commit], hb.o, [commit], ATT, carrier="state", construction=hcon)
    ha = history([ci("A1"), ci("B3"), CP, ci("A3"), ci("B1"), TH])      # attempted at {A1,B3}; now {A3,B1}
    hb = history([ci("A1"), ci("B3"), ci("A3"), ci("B1"), CP, TH])      # attempted at {A3,B1}; now {A3,B1}
    assert [x["reason"] for x in ha.o.attempts] == [x["reason"] for x in hb.o.attempts] == ["capacity"]
    pair(R, "7b", 7, "retry after the pending set changed (rejected at another pending set, same reason) vs identical retry",
         ha.o, [commit], hb.o, [commit], ATT, carrier="state", construction=hcon,
         checks={"same_pending": ha.o.pending == hb.o.pending, "same_step": ha.o.step == hb.o.step})
    pair(R, "7b-bit", 7, "as 7b, carrier = attempt.dependencies_changed only",
         ha.o, [commit], hb.o, [commit], ATT, carrier=["attempt.dependencies_changed"], construction=hcon)
    steps = sel + [cs("A1", 0), cs("B1", 0), CA]
    hx = history(steps, make_spec(event=("slot_closed", 16, 4)))
    hy = history(steps, make_spec())
    hx.pad_to(17)
    hy.pad_to(17)
    assert hx.o.events and not hy.o.events
    pair(R, "7c", 7, "commit_assignment rejected (slot_conflict), then a slot requirement revision (event) vs unchanged",
         hx.o, [commit_a], hy.o, [commit_a], ATT, carrier=["attempt.dependencies_changed"], construction=hcon,
         notes="The observation differs (event) by design; carrier = the dependencies-changed coordinate.")
    # revision: rejected assignment, then uncommit + re-commit of the same selection (a revision) vs think x3
    hx = history(steps + [("uncommit", {"target": "select"}), ci("A1"), CP])
    hy = history(steps + [TH, TH, TH])
    pair(R, "7d", 7, "commit_assignment rejected, then a selection revision (uncommit + recommit) vs unchanged",
         hx.o, [commit_a], hy.o, [commit_a], ATT, carrier=["attempt.dependencies_changed"], construction=hcon,
         checks={"same_step": hx.o.step == hy.o.step},
         notes="Recommitting clears pending_assignment, so the relevant dependencies changed.")
    # new inspection: commit_pending before inspecting requirements (missing_dependency), then inspect them
    r1 = Run(make_spec(), address_seed=31)
    r1.steps([ci("A1"), ci("B1"), CP, ("inspect", {"target": "requirements"})])
    r2 = Run(make_spec(), address_seed=31)
    r2.steps([ci("A1"), ci("B1"), CP, TH])
    assert r1.o.attempts[0]["reason"] == "missing_dependency"
    pair(R, "7e", 7, "commit_pending rejected (missing_dependency: requirements unknown), then inspect requirements vs think",
         r1.o, [commit], r2.o, [commit], ALL, carrier="state",
         construction="hand world, nothing inspected; choose A1, B1; commit_pending; then inspect requirements vs think",
         notes="The state differs through req.* (kept by both ablations).")
    pair(R, "7e-bit", 7, "as 7e, carrier = attempt.dependencies_changed only (report)",
         r1.o, [commit], r2.o, [commit], {"d1": None, "d1-noapp": None, "d1-noattempt": False},
         carrier=["attempt.dependencies_changed"],
         notes="REPORT: relevant_dependencies(commit_pending) = pending, requirement versions and selection; it does "
               "not include whether requirements are known, so this bit stays 0 after the resolving inspection.")

    # ---- 8: committed vs revoked; which requirement changed
    evs = {"capacity_reduced(3) revokes selection": ("capacity_reduced", 40, 3),
           "capacity_reduced(5) no revocation": ("capacity_reduced", 40, 5),
           "slot_closed(2) revokes assignment": ("slot_closed", 40, 2),
           "slot_closed(4) no revocation": ("slot_closed", 40, 4),
           "edge_closed(1,2)": ("edge_closed", 40, (1, 2)),
           "deadline_moved(8)": ("deadline_moved", 40, 8)}
    states = {k: after_event(e)[0].o for k, e in evs.items()}
    assert states["capacity_reduced(3) revokes selection"].selection_id is None
    assert states["slot_closed(2) revokes assignment"].assignment_id is None
    catalog["after_selection_revocation"] = catalog_check(states["capacity_reduced(3) revokes selection"])
    catalog["after_edge_closed"] = catalog_check(states["edge_closed(1,2)"])
    unc = [Action("uncommit", {"target": "select"}), Action("uncommit", {"target": "assign"})]
    for pid, a, b in (("8a", "capacity_reduced(3) revokes selection", "capacity_reduced(5) no revocation"),
                      ("8b", "slot_closed(2) revokes assignment", "slot_closed(4) no revocation"),
                      ("8c", "capacity_reduced(5) no revocation", "slot_closed(4) no revocation"),
                      ("8d", "edge_closed(1,2)", "deadline_moved(8)"),
                      ("8e", "capacity_reduced(5) no revocation", "edge_closed(1,2)"),
                      ("8f", "slot_closed(4) no revocation", "deadline_moved(8)"),
                      ("8g", "capacity_reduced(3) revokes selection", "slot_closed(2) revokes assignment")):
        pair(R, pid, 8, f"after the event: {a} vs {b}", states[a], unc, states[b], unc, ALL, carrier="observation",
             construction=ev_con.split("; then")[0])
    for pid, a, b in (("8a-row", "capacity_reduced(3) revokes selection", "capacity_reduced(5) no revocation"),
                      ("8b-row", "slot_closed(2) revokes assignment", "slot_closed(4) no revocation")):
        pair(R, pid, 8, f"uncommit rows (target committed?) after: {a} vs {b}", states[a], unc, states[b], unc, ALL,
             carrier="row", construction=ev_con.split("; then")[0])
    # revoked by an event vs never committed (step-matched): the event memory must remain visible
    r = Run(make_spec(), address_seed=21)
    r.inspect_all()
    r.pad_to(40)
    pair(R, "8h", 8, "selection revoked by capacity_reduced(3) vs never committed (after the same inspection)",
         states["capacity_reduced(3) revokes selection"], [], r.o, [], ALL, carrier="observation")

    # ---- 9: renaming invariance (world-level: item handles, record handles, foreign snapshots/selection ids)
    names = {"A1": "i884120", "A2": "i019553", "A3": "i500001", "B1": "i777310", "B2": "i302948", "B3": "i640072"}

    def rename_spec(spec):
        m = names
        items = tuple(replace(x, handle=m[x.handle]) for x in spec.items)
        foreign = []
        for f in spec.foreign:
            f = deepcopy(f)
            p = f["snapshot"]["problem"]
            for key in (("handles", "excluded") if f["primitive"] == "constrained_subset" else
                        ("items",) if f["primitive"] == "csp" else ()):
                p[key] = [m[h] for h in p[key]]
            if "selection_id" in f["depends_on"]:
                f["depends_on"]["selection_id"] = selection_id(p["items"])
            foreign.append(f)
        return replace(spec, items=items, incompatible=tuple((m[a], m[b]) for a, b in spec.incompatible),
                       foreign=tuple(foreign),
                       planted=(tuple(m[h] for h in spec.planted[0]), tuple((m[h], s) for h, s in spec.planted[1]),
                                spec.planted[2]))

    for mode in ("reuse", "naive_reuse", "recompute"):
        spec_a = make_spec(event=("capacity_reduced", 1, 5), event_trigger="progress", foreign_kinds=FOREIGN_KINDS)
        spec_b = rename_spec(spec_a)
        ra, rb = Run(spec_a, address_seed=51), Run(spec_b, address_seed=9_999)
        ref = DepReference(mode)
        rename_ok, steps_checked, first_mismatch = True, 0, None
        while not ra.o.done and steps_checked < 96:
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
                if xa != xb or [_key(mapa(a)) for a in ca] != [_key(b) for b in cb] or ma != mb:
                    rename_ok = False
                    first_mismatch = first_mismatch or {"step": ra.o.step, "version": v,
                                                        "obs": _diff(xa, xb, OBSERVATION_NAMES_D1)}
            act = ref.choose(ra.o)
            ra.step(act.kind, **act.arguments)
            b = mapa(act)
            rb.step(b.kind, **b.arguments)
            steps_checked += 1
        R.append({"id": f"9a-{mode}", "distinction": 9,
                  "description": f"consistent renaming of item handles, record handles (address seed) and foreign "
                  f"snapshots/selection ids: observation, candidate set (mapped) and matrix identical at every step "
                  f"of a dep_{mode} trajectory (capacity event after the first completion, 11 foreign records)",
                  "construction": "hand world vs renamed copy (address seeds 51 vs 9999); the reference acts in world "
                                  "a and the renamed action is replayed in world b",
                  "carrier": "state", "expected": NONE,
                  "versions": {v: {"distinguishable": not rename_ok} for v in VERSIONS},
                  "version_pass": {v: rename_ok for v in VERSIONS},
                  "preconditions": {"event_fired": bool(ra.o.events), "at_least_10_steps": steps_checked >= 10},
                  "info": {"steps_checked": steps_checked, "first_mismatch": first_mismatch,
                           "verified_success": ra.env.evaluate()["verified_success"]},
                  "pass": rename_ok and bool(ra.o.events) and steps_checked >= 10, "notes": ""})
    # problem-handle spelling: swap own and foreign problem names at the observation level
    o = run.o
    keys = list(o.problems)
    perm = dict(zip(keys, reversed(keys)))
    o2 = rename_problems(o, perm)
    mp = lambda a: Action(a.kind, {**a.arguments, "problem": perm[a.arguments["problem"]]}) \
        if "problem" in a.arguments else a  # noqa: E731
    comp = compare(o, [], o2, [], "state", map_action=mp)
    ok = all(not comp[v]["distinguishable"] for v in VERSIONS)
    R.append({"id": "9b", "distinction": 9, "description": "problem-handle spelling: reverse the problem_n names so "
              "own drafts and foreign problems swap spellings (records, retrieved, attempts, action keys renamed)",
              "construction": "the retrieved full-chain state of pairs 1/4", "carrier": "state", "expected": NONE,
              "versions": comp, "version_pass": {v: not comp[v]["distinguishable"] for v in VERSIONS},
              "preconditions": {"own_and_foreign_problems": len(keys) > len(FOREIGN_KINDS)}, "pass": ok,
              "notes": "Foreign vs own cannot leak through 'problem_' spelling or numbering."})

    # ---- 10: hidden-state leakage (identical PUBLIC facts, different hidden facts)
    def trajectory_diffs(spec_a, spec_b, script):
        ra, rb = Run(spec_a, LEAK_SEED), Run(spec_b, LEAK_SEED)
        diffs, n = [], 0
        for i in range(len(script) + 1):
            if i:
                ra.step(script[i - 1][0], **script[i - 1][1])
                rb.step(script[i - 1][0], **script[i - 1][1])
            n += 1
            for v in VERSIONS:
                ca, xa, ma = frame(ra.o, v)
                cb, xb, mb = frame(rb.o, v)
                same_cat = [_key(a) for a in ca] == [_key(b) for b in cb]
                if xa != xb or not same_cat or ma != mb:
                    diffs.append({"step": ra.o.step, "version": v, "obs": _diff(xa, xb, OBSERVATION_NAMES_D1),
                                  "rows": sorted({c for x, y in zip(ma, mb) for c in _diff(x, y, CANDIDATE_NAMES_D1)})
                                  if same_cat else None, "same_catalog": same_cat})
        return diffs, n

    def fh(spec, label):
        return Run(spec, LEAK_SEED).foreign[label]

    def leak(pid, desc, spec_a, spec_b, script, notes="", expected=NONE):
        diffs, n = trajectory_diffs(spec_a, spec_b, script)
        by_v = {v: {"distinguishable": any(d["version"] == v for d in diffs),
                    "first_difference": next((d for d in diffs if d["version"] == v), None)} for v in VERSIONS}
        vp = {v: expected[v] is None or by_v[v]["distinguishable"] == expected[v] for v in VERSIONS}
        R.append({"id": pid, "distinction": 10, "description": desc, "construction": f"{len(script)}-action script, "
                  f"{n} states compared per version", "carrier": "state", "expected": expected, "versions": by_v,
                  "version_pass": vp, "preconditions": {}, "pass": all(vp.values()), "notes": notes})

    a1_hidden = tuple(replace(x, weight=6, price=5, duration=4, slots=(0, 4)) if x.handle == "A1" else x for x in ITEMS)
    script_a = [("inspect", {"target": "requirements"}), ("inspect", {"target": "map"})] + \
        [("inspect", {"target": h}) for h in ("A2", "A3", "B1", "B2", "B3")] + \
        [("choose_item", {"item": "A1"}), ("choose_item", {"item": "B1"}), ("start_subset", {"handle": "problem_0"}),
         ("add_constraint", {"problem": "problem_0", "constraint": "capacity"}),
         ("call", {"problem": "problem_0", "budget": 128}), ("think", {})]
    leak("10a", "uninspected item A1 (weight/price/duration/slots) differs; everything else inspected; A1 pending; "
         "own subset draft + call (the draft covers inspected items only)", make_spec(), make_spec(items=a1_hidden), script_a)
    chain_script = [("inspect", {"target": "requirements"})] + [("inspect", {"target": x.handle}) for x in ITEMS] + \
        [("inspect", {"target": "map"}), ("choose_item", {"item": "A1"}), ("choose_item", {"item": "B1"}),
         ("commit_pending", {}), ("choose_slot", {"item": "A1", "slot": 0}), ("choose_slot", {"item": "B1", "slot": 2}),
         ("commit_assignment", {}), ("build_route", {"handle": "problem_0"}),
         ("call", {"problem": "problem_0", "budget": 128}), ("think", {})]
    leak("10b", "future event (slot_closed@60 vs deadline_moved@60, step trigger) and planted solution differ; "
         "full chain before the event", make_spec(event=("slot_closed", 60, 2)),
         make_spec(event=("deadline_moved", 60, 8), planted=(("A2", "B3"), (("A2", 0), ("B3", 2)), (0, 3))), chain_script)
    pre_commit = chain_script[:9] + [("start_subset", {"handle": "problem_0"}), ("think", {})]
    leak("10c", "future event under the progress trigger (capacity_reduced after completion 1 vs edge_closed after "
         "completion 2) vs no event; everything before the first completion commit",
         make_spec(event=("capacity_reduced", 1, 5), event_trigger="progress"),
         make_spec(event=("edge_closed", 2, (1, 2)), event_trigger="progress"), pre_commit)
    leak("10d", "no event scheduled vs capacity_reduced after completion 1 (progress); everything before that commit",
         make_spec(), make_spec(event=("capacity_reduced", 1, 5), event_trigger="progress"), pre_commit)
    # payload stays hidden until retrieve: same foreign record metadata, different payload
    base_f = foreign_records(make_spec(), ("route_current",))[0]
    alt_f = {**deepcopy(base_f), "payload": [[0, 3], 6]}
    spec_p, spec_q = replace(make_spec(), foreign=(base_f,)), replace(make_spec(), foreign=(alt_f,))
    h = fh(spec_p, "route_current")
    leak("10e", "foreign route record with identical metadata but a different (hidden) payload; not retrieved; "
         "full inspection", spec_p, spec_q, [("inspect", {"target": "requirements"}), ("inspect", {"target": "map"})]
         + [("inspect", {"target": x.handle}) for x in ITEMS])
    leak("10e-control", "as 10e, then retrieve the record: the payload becomes public and must now be visible",
         spec_p, spec_q, [("inspect", {"target": "requirements"}), ("inspect", {"target": "map"})]
         + [("inspect", {"target": x.handle}) for x in ITEMS] + [("retrieve", {"handle": h})], expected=ALL,
         notes="Sensitivity control for 10e.")
    lab_f = {**deepcopy(base_f), "label": "route_earlier"}
    leak("10f", "evaluator-only foreign label differs (route_current vs route_earlier) on an identical record",
         spec_p, replace(make_spec(), foreign=(lab_f,)),
         [("inspect", {"target": "map"}), ("retrieve", {"handle": h}), ("think", {})])
    same_f = foreign_records(make_spec(), ("subset_current",))   # built from the base facts in BOTH worlds
    spec_p, spec_q = replace(make_spec(), foreign=same_f), replace(make_spec(items=a1_hidden), foreign=same_f)
    leak("10g", "uninspected A1's hidden facts differ while a foreign subset record (and its public snapshot) is "
         "identical in both worlds; everything but A1 inspected; retrieve the record",
         spec_p, spec_q,
         [("inspect", {"target": "requirements"})] + [("inspect", {"target": x}) for x in ("A2", "A3", "B1", "B2", "B3")]
         + [("retrieve", {"handle": fh(spec_p, "subset_current")}), ("think", {})])

    # ---- 10 (disclosure probe): the PUBLIC foreign-snapshot channel. The two worlds differ in a hidden fact AND,
    # necessarily, in the public snapshot of a foreign record built from that fact. Report which coordinates
    # carry it before the fact is inspected (not hidden-state leakage: the snapshot and, after retrieve, the
    # payload are public inputs). Expected: None (report).
    b1_short = tuple(replace(x, duration=1) if x.handle == "B1" else x for x in ITEMS)
    report = {v: None for v in VERSIONS}
    sa, sb = make_spec(foreign_kinds=("csp_current",)), make_spec(items=b1_short, foreign_kinds=("csp_current",))
    h = fh(sa, "csp_current")
    leak("10x-csp", "DISCLOSURE: foreign csp record for {A1,B1}; B1's hidden duration differs (2 vs 1) and so does "
         "the record's public snapshot; nothing inspected; retrieve the record; inspect requirements",
         sa, sb, [("retrieve", {"handle": h}), ("inspect", {"target": "requirements"})], expected=report)
    leak("10x-csp-meta", "DISCLOSURE: as 10x-csp, record NOT retrieved (metadata only)",
         sa, sb, [("inspect", {"target": "requirements"})], expected=NONE,
         notes="Metadata only: relations need a completed inspection, so nothing may differ.")
    edges_b = tuple((u, v, 2) if (u, v) == (2, 3) else (u, v, w) for u, v, w in EDGES)
    sa, sb = make_spec(foreign_kinds=("route_current",)), make_spec(edges=edges_b, foreign_kinds=("route_current",))
    h = fh(sa, "route_current")
    leak("10x-route", "DISCLOSURE: foreign current-route record; hidden edge (2,3) weight differs (1 vs 2) and so do "
         "the record's snapshot and payload; map NOT inspected; retrieve", sa, sb,
         [("retrieve", {"handle": h}), ("inspect", {"target": "requirements"})], expected=report)
    a1_heavier = tuple(replace(x, weight=3) if x.handle == "A1" else x for x in ITEMS)
    sa, sb = make_spec(foreign_kinds=("subset_current",)), make_spec(items=a1_heavier, foreign_kinds=("subset_current",))
    h = fh(sa, "subset_current")
    leak("10x-subset", "DISCLOSURE: foreign current-subset record; hidden A1 weight differs (2 vs 3); A1 NOT "
         "inspected (others are); retrieve", sa, sb,
         [("inspect", {"target": "requirements"})] + [("inspect", {"target": x}) for x in ("A2", "A3", "B1", "B2", "B3")]
         + [("retrieve", {"handle": h})], expected=report)
    return R, catalog, run.env


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


# ---------------------------------------------------------------------------------------------
# 11: functional ablation checks on generated worlds (dev seeds)
# ---------------------------------------------------------------------------------------------

@contextmanager
def perturbed_relations(rng: random.Random):
    """Replace every request/dependency relation value (records and drafts) with random booleans."""
    orig_rel, orig_draft = dep.relations, dep._draft_match

    def rel(o, record, primitive=None):
        r = dict(orig_rel(o, record, primitive))
        for k in RELATION_KEYS:
            r[k] = rng.random() < .5
        return r

    def draft(o, entry):
        return float(rng.random() < .5), float(rng.random() < .5)
    dep.relations, dep._draft_match = rel, draft
    try:
        yield
    finally:
        dep.relations, dep._draft_match = orig_rel, orig_draft


def perturbed_attempts(o, rng: random.Random):
    """A different attempt log: shuffled outcomes/reasons, flipped dependency snapshots, dropped entries."""
    out = []
    for at in o.attempts:
        if rng.random() < .2:
            continue
        at = deepcopy(at)
        at["outcome_status"] = rng.choice(dep.OUTCOMES)
        at["reason"] = rng.choice((None,) + dep.REASONS)
        if rng.random() < .5:
            at["dependency_versions_at_attempt"] = {"perturbed": rng.random()}
        out.append(at)
    return replace(o, attempts=tuple(out))


def functional_checks(seeds=6, modes=("reuse", "naive_reuse", "recompute")) -> dict:
    rng = random.Random(0)
    i_noapp = set(FEATURE_MASKS["d1-noapp"][1])
    i_noatt = set(FEATURE_MASKS["d1-noattempt"][1])
    allowed_noapp = i_noapp | {CANDIDATE_NAMES_D1.index(n) for n in NEED_BOUND_DERIVED}
    st = {"states": 0, "candidates": 0,
          "noapp_invariant_to_relations": True, "d1_sensitive_to_relations_states": 0,
          "noattempt_invariant_to_attempts": True, "d1_sensitive_to_attempts_states": 0,
          "noattempt_equals_d1_off_mask": True, "noapp_equals_d1_off_mask": True,
          "noapp_offmask_differences": {}, "masked_coordinates_zero": True, "dimensions_equal": True,
          "first_failure": None}

    def fail(key, info):
        st[key] = False
        st["first_failure"] = st["first_failure"] or {"check": key, **info}
    for seed in range(seeds):
        for mode in modes:
            spec = generate_depworld(DEV_SEED + 7_000 + seed, p_event=1.0, foreign_records=4)
            env, ref = DepWorkshop(spec, executor=EXEC, address_seed=seed), DepReference(mode)
            o = env.observe()
            while not o.done:
                enc = {v: frame(o, v) for v in VERSIONS}
                acts, x1, m1 = enc["d1"]
                st["states"] += 1
                st["candidates"] += len(acts)
                info = {"seed": DEV_SEED + 7_000 + seed, "mode": mode, "step": o.step}
                for v in VERSIONS:
                    if len(enc[v][1]) != len(x1) or any(len(r) != len(CANDIDATE_NAMES_D1) for r in enc[v][2]):
                        fail("dimensions_equal", {**info, "version": v})
                    om, cm = FEATURE_MASKS[v]
                    if any(enc[v][1][i] != 0 for i in om) or any(r[i] != 0 for r in enc[v][2] for i in cm):
                        fail("masked_coordinates_zero", {**info, "version": v})
                # d1-noattempt == d1 off its mask (observation + candidates)
                _, xa, ma = enc["d1-noattempt"]
                om = set(FEATURE_MASKS["d1-noattempt"][0])
                if any(xa[i] != x1[i] for i in range(len(x1)) if i not in om) or any(
                        r[i] != s[i] for r, s in zip(ma, m1) for i in range(len(s)) if i not in i_noatt):
                    fail("noattempt_equals_d1_off_mask", info)
                # d1-noapp == d1 off its mask, except the need_bound-derived coordinates (recorded)
                _, xp, mp_ = enc["d1-noapp"]
                om = set(FEATURE_MASKS["d1-noapp"][0])
                if any(xp[i] != x1[i] for i in range(len(x1)) if i not in om):
                    fail("noapp_equals_d1_off_mask", {**info, "where": "observation"})
                for r, s in zip(mp_, m1):
                    for i in range(len(s)):
                        if r[i] != s[i] and i not in i_noapp:
                            n = CANDIDATE_NAMES_D1[i]
                            st["noapp_offmask_differences"][n] = st["noapp_offmask_differences"].get(n, 0) + 1
                            if i not in allowed_noapp:
                                fail("noapp_equals_d1_off_mask", {**info, "coordinate": n})
                # relation perturbation: d1-noapp must not move; d1 should
                with perturbed_relations(random.Random(rng.random())):
                    pa = frame(o, "d1-noapp")
                    pd = frame(o, "d1")
                if pa[1] != xp or pa[2] != mp_:
                    fail("noapp_invariant_to_relations", info)
                st["d1_sensitive_to_relations_states"] += int(pd[1] != x1 or pd[2] != m1)
                # attempt-log perturbation: d1-noattempt must not move; d1 should (when there are attempts)
                o2 = perturbed_attempts(o, random.Random(rng.random()))
                qa, qd = frame(o2, "d1-noattempt"), frame(o2, "d1")
                if qa[1] != xa or qa[2] != ma:
                    fail("noattempt_invariant_to_attempts", info)
                st["d1_sensitive_to_attempts_states"] += int(qd[1] != x1 or qd[2] != m1)
                o = env.step(ref.choose(o))
    st["pass"] = (st["noapp_invariant_to_relations"] and st["noattempt_invariant_to_attempts"]
                  and st["noattempt_equals_d1_off_mask"] and st["noapp_equals_d1_off_mask"]
                  and st["masked_coordinates_zero"] and st["dimensions_equal"]
                  and st["d1_sensitive_to_relations_states"] > 0 and st["d1_sensitive_to_attempts_states"] > 0)
    return st


# ---------------------------------------------------------------------------------------------

def summarize(pairs) -> list[dict]:
    rows = {}
    for p in pairs:
        d = rows.setdefault(p["distinction"], {"distinction": p["distinction"], "pairs": 0, "passed": 0,
                                                "d1_fail": [], "ablation_fail": [], "precondition_fail": []})
        d["pairs"] += 1
        d["passed"] += bool(p["pass"])
        if not p["version_pass"]["d1"]:
            d["d1_fail"].append(p["id"])
        for a in ("d1-noapp", "d1-noattempt"):
            if not p["version_pass"][a]:
                d["ablation_fail"].append(f"{p['id']}@{a}")
        if not all(p["preconditions"].values()):
            d["precondition_fail"].append(p["id"])
    return [rows[k] for k in sorted(rows)]


def audit(functional_seeds=6) -> dict:
    pairs, catalog, _ = audit_pairs()
    functional = functional_checks(functional_seeds)
    dims = {}
    o = Run(make_spec(foreign_kinds=FOREIGN_KINDS)).o
    for v in VERSIONS:
        _, x, m = frame(o, v)
        dims[v] = [len(x), len(m[0])]
    masks = {v: {"observation": [OBSERVATION_NAMES_D1[i] for i in om], "candidate": [CANDIDATE_NAMES_D1[i] for i in cm],
                 "observation_indices": list(om), "candidate_indices": list(cm)}
             for v, (om, cm) in FEATURE_MASKS.items()}
    summary = summarize(pairs)
    gate = {"d1_all_distinctions": all(not s["d1_fail"] and not s["precondition_fail"] for s in summary),
            "ablations_exact": all(not s["ablation_fail"] for s in summary),
            "catalog": all(c["pass"] for c in catalog.values()),
            "functional": functional["pass"],
            "dimensions_equal": len({tuple(d) for d in dims.values()}) == 1}
    gate["pass"] = all(pair_["pass"] for pair_ in pairs) and all(gate.values())
    return {"gate": gate, "summary": summary, "pairs": pairs, "catalog_checks": catalog,
            "functional": functional, "dimensions": dims, "masks": masks,
            "noapp_kept_payload_facts": list(NOAPP_KEPT_FACTS), "need_bound_derived": list(NEED_BOUND_DERIVED)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/campaigns/extended-03/review/stage-b-preflight.json")
    ap.add_argument("--functional-seeds", type=int, default=6)
    args = ap.parse_args(argv)
    result = audit(args.functional_seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str) + "\n")
    for p in result["pairs"]:
        flag = "PASS" if p["pass"] else "FAIL"
        cells = " ".join(f"{v}={'D' if p['versions'][v]['distinguishable'] else '='}" for v in VERSIONS)
        d1 = p["versions"]["d1"]
        coords = d1.get("row_diff") or d1.get("observation_diff") or d1.get("first_difference") or ""
        print(f"{flag} {p['id']:<44} {cells}  {coords}")
    for s in result["summary"]:
        print(s)
    print("functional", {k: v for k, v in result["functional"].items()})
    print("catalog", result["catalog_checks"])
    print("dims", result["dimensions"])
    print("GATE", result["gate"])
    return result


if __name__ == "__main__":
    main()
