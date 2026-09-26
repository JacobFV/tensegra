"""EVALUATOR-ONLY step audit for the extended-03 P1 sealed evaluation.

``P1AuditedDepWorkshop`` is a ``DepWorkshop`` whose dynamics, observations,
catalogue, RNG use and ``evaluate()`` fields are unchanged. It additionally
logs, per step and *before* the action is applied, the raw facts the
pre-registered P1 metrics need (research/campaigns/extended-03/
protocol-P1-metrics.md), and exposes them under the new ``evaluate()`` key
``p1_audit``:

- the public need (which stage result the public state lacks) and the stage
  opening step (``_stage_open``, the teacher's own freshness boundary);
- the records that are *reusable* at that step: status success, applicable by
  the public teacher rule, created at or before the stage opened, and whose
  use_return the environment would accept right now and that leaves the
  episode completable (hidden-state check, ``completes_now``);
- for every use_return: the target's public applicability, creation step,
  pre-existence, hidden audit and whether it was reusable;
- for every call: the primitive of the called problem;
- at the event: which committed pieces were revoked or kept, and whether each
  kept piece can still be part of a deadline-feasible completion (hidden);
- the public attempt log (action keys, outcomes, reasons and the
  ``relevant_dependencies`` snapshot the d1 encoder compares against).

Hidden-state reads here are logging only. Nothing is fed back to an actor,
encoder, catalogue or reference, and nothing here consumes the handle RNG.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
from typing import Any

from .campaign03_depworld import (PRIMITIVE_OF_USE, USE_OF_PRIMITIVE, DepWorkshop, _dijkstra, _finish,
                                  _hidden_relations, _stage_open, _validate_assignment, _validate_selection,
                                  action_key, applicable)

AUDIT_VERSION = "p1-audit-v1"
NEED_PRIMITIVE = {"select": "constrained_subset", "assign": "csp", "route": "shortest_path"}


def public_need(o) -> str | None:
    """The stage result the public state lacks, in the dependency order."""
    if o.done:
        return None
    if o.selection_id is None:
        return "select"
    if o.assignment_id is None:
        return "assign"
    if o.position != o.goal["destination"]:
        return "route"
    return None


def _shortest(env) -> int | None:
    s = env._spec
    return _dijkstra([(u, v, w) for (u, v), w in env._edges.items()], s.locations, env._position, s.destination)[1]


def accepted_now(env, record, use: str) -> bool:
    """Hidden check: would use_return(record, use) be accepted by the environment
    in the current state (ignoring only the retrieve precondition)?"""
    try:
        if PRIMITIVE_OF_USE[use] != record["primitive"] or record["payload"] is None:
            return False
        if record["status"] not in ("success", "timeout") or not record["certificate_valid"]:
            return False
        payload, snap = record["payload"], record["problem_snapshot"]["problem"]
        if use == "select":
            if env._selected or not env._req_known:
                return False
            handles = [snap["handles"][i] for i in payload[0]]
            return _validate_selection(env._rows, env._spec.categories, env._req, handles)[0]
        if not env._selected:
            return False
        if use == "assign":
            if env._assignment is not None or sorted(snap["items"]) != sorted(env._selected):
                return False
            assignment = dict(zip(snap["items"], payload))
            if not _validate_assignment(env._rows, env._selected, env._req["closed_slots"], assignment)[0]:
                return False
            return _finish(env._rows, assignment) + env._travel <= env._req["deadline"]
        if env._assignment is None:
            return False
        path = list(payload[0])
        if len(path) < 2 or path[0] != env._position:
            return False
        if any((u, v) not in env._edges for u, v in zip(path, path[1:])):
            return False
        total = sum(env._edges[(u, v)] for u, v in zip(path, path[1:]))
        if _finish(env._rows, env._assignment) + env._travel + total > env._req["deadline"]:
            return False
        return env._travel + total <= env._spec.travel_limit
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def selection_completable(env, handles=None) -> bool:
    """Hidden: some assignment of the selection (default: the committed one; current
    closed slots) finishes early enough for a shortest route from here to meet the deadline."""
    items = list(env._selected if handles is None else handles)
    if not items:
        return False
    dist = _shortest(env)
    if dist is None:
        return False
    closed = set(env._req["closed_slots"])
    domains = [[s for s in env._rows[h]["slots"] if s not in closed] for h in items]
    for combo in itertools.product(*domains):
        assignment = dict(zip(items, combo))
        if (_validate_assignment(env._rows, items, env._req["closed_slots"], assignment)[0]
                and _finish(env._rows, assignment) + env._travel + dist <= env._req["deadline"]):
            return True
    return False


def assignment_completable(env, assignment=None) -> bool:
    """Hidden: the assignment (default: the committed one) is valid now and a shortest
    route from here still meets the deadline."""
    assignment = env._assignment if assignment is None else assignment
    if assignment is None or not env._selected:
        return False
    dist = _shortest(env)
    if dist is None:
        return False
    ok = _validate_assignment(env._rows, env._selected, env._req["closed_slots"], assignment)[0]
    return ok and _finish(env._rows, assignment) + env._travel + dist <= env._req["deadline"]


def completes_now(env, record, use: str) -> bool:
    """Hidden: use_return(record, use) would be accepted now AND leave the episode
    completable (select: some deadline-feasible assignment exists for the selection;
    assign: a shortest route still meets the deadline; route: it delivers)."""
    if not accepted_now(env, record, use):
        return False
    try:
        payload, snap = record["payload"], record["problem_snapshot"]["problem"]
        if use == "select":
            return selection_completable(env, [snap["handles"][i] for i in payload[0]])
        if use == "assign":
            return assignment_completable(env, dict(zip(snap["items"], payload)))
        return list(payload[0])[-1] == env._spec.destination
    except (KeyError, TypeError, ValueError, IndexError):
        return False


class P1AuditedDepWorkshop(DepWorkshop):
    """DepWorkshop plus an evaluator-only per-step audit (see module docstring)."""

    def __init__(self, *args, **kwargs):
        self._p1_steps: list[dict[str, Any]] = []
        self._p1_event: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def _p1_pre(self, o, action) -> dict[str, Any]:
        need = public_need(o)
        row: dict[str, Any] = {"step": o.step + 1, "kind": action.kind, "key": action_key(action), "need": need}
        if need is not None:
            primitive = NEED_PRIMITIVE[need]
            opened = _stage_open(o, primitive)
            row["open"] = opened
            applicable_any, applicable_pre, reusable = 0, [], []
            for r in o.records:
                if r["primitive"] != primitive or r["status"] != "success" or not applicable(o, r, primitive):
                    continue
                applicable_any += 1
                if r["created_step"] > opened:
                    continue
                applicable_pre.append(r["handle"])
                if completes_now(self, self._records[r["handle"]], need):
                    reusable.append(r["handle"])
            row.update(applicable_any=applicable_any, applicable_pre=applicable_pre, reusable=reusable)
        args = action.arguments
        if action.kind == "call":
            row["call_primitive"] = self._problems.get(args.get("problem"), {}).get("primitive")
        elif action.kind == "use_return":
            handle, use = args.get("handle"), args.get("as")
            record = self._records.get(handle)
            info: dict[str, Any] = {"handle": handle, "as": use, "exists": record is not None}
            if record is not None and use in PRIMITIVE_OF_USE:
                primitive = PRIMITIVE_OF_USE[use]
                opened = _stage_open(o, primitive)
                public = next((r for r in o.records if r["handle"] == handle), None)
                type_match = record["primitive"] == primitive
                hidden = (_hidden_relations(self, record["problem_snapshot"], record["depends_on"], primitive)
                          if type_match else {"request_match": False, "dependency_match": False, "applicable": False})
                info.update(primitive=record["primitive"], type_match=type_match, status=record["status"],
                            created_step=record["created_step"], pre_existing=record["created_step"] <= opened,
                            stage_open=opened, foreign=record["_label"] is not None, retrieved=handle in o.retrieved,
                            public_applicable=bool(public is not None and applicable(o, public, primitive)),
                            accepted_now=accepted_now(self, record, use),
                            completes_now=completes_now(self, record, use),
                            hidden_request_match=hidden["request_match"],
                            hidden_dependency_match=hidden["dependency_match"],
                            hidden_applicable=hidden["applicable"],
                            reusable=need == use and handle in row.get("reusable", ()))
            row["use"] = info
        return row

    def step(self, action):
        if self._done:
            return super().step(action)
        pre = self._p1_pre(self._cache, action)
        events_before = len(self._events)
        out = super().step(action)
        pre["outcome"] = self._feedback.get("status")
        pre["work"] = self._work
        if len(self._events) > events_before:
            ev = self._events[-1]
            revoked = list(ev["revoked"])
            kept = ([p for p, on in (("selection", bool(self._selected)), ("assignment", self._assignment is not None))
                     if on])
            self._p1_event = {"step": ev["step"], "kind": ev["kind"], "revoked": revoked, "kept": kept,
                              "kept_completable": {"selection": selection_completable(self) if "selection" in kept else None,
                                                   "assignment": assignment_completable(self) if "assignment" in kept else None},
                              "work_at_event": self._work}
            pre["event"] = True
        self._p1_steps.append(pre)
        return out

    def evaluate(self) -> dict[str, Any]:
        out = super().evaluate()
        out["p1_audit"] = {"version": AUDIT_VERSION, "steps": deepcopy(self._p1_steps),
                           "attempts": deepcopy(self._attempts), "event": deepcopy(self._p1_event),
                           "use_of_primitive": dict(USE_OF_PRIMITIVE)}
        return out
