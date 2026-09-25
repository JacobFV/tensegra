"""Public-observation workshop reference policies; no evaluator access in policy.

These are supplied schedules for measuring solvability and bootstrapping, not
learned planning. Every search is a charged primitive call. Local greedy scans
are measured as controller CPU by ``run_episode``.
"""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from .campaign02_world import Action, Observation


class ReferencePolicy:
    """Public heuristic over the learned actor's exact action catalogue.

    All search is explicit charged solver execution. Greedy local scans are
    timed as controller CPU. This supplied policy is not learned planning.
    """
    def __init__(self, mode="cheap_first", initial_budget=16):
        if mode not in {"cheap", "always_tool", "cheap_first"}:
            raise ValueError("unknown reference mode")
        self.mode, self.initial_budget = mode, initial_budget

    def choose_index(self, observation, candidates):
        return candidates.index(self.choose(observation, candidates))

    def choose(self, o: Observation, catalog=None) -> Action:
        from .campaign02_world import action_catalog
        actions = action_catalog(o) if catalog is None else catalog
        def pick(kind, **kwargs):
            return next((a for a in actions if a.kind == kind and
                         all(a.arguments.get(k) == v for k,v in kwargs.items())), None)
        if o.done:
            raise ValueError("cannot act after episode end")
        if o.delivered:
            return pick("verify")
        for row in o.item_inventory:
            if row["handle"] not in o.known_items:
                return pick("inspect", target=row["handle"])
        if not o.selected:
            attempted = any(p["primitive"] == "constrained_subset" for p in o.problems.values())
            if self.mode != "always_tool" and not attempted:
                chosen = self._greedy(o)
                if chosen is not None:
                    for handle in chosen:
                        if handle not in o.pending:
                            return pick("choose_item", item=handle)
                    if o.feedback.get("status") != "rejected":
                        return pick("commit_pending")
            if self.mode == "cheap":
                return pick("abstain")
            return self._solver(o, actions, "constrained_subset", "subset", pick)
        if o.known_edges is None:
            return pick("inspect", target="map")
        move, greedy_distance = self._greedy_route(o, pick)
        if self.mode == "cheap":
            return move or pick("abstain")
        if self.mode == "cheap_first" and move is not None:
            # Public lower bound: every route must pay one outgoing edge.
            # This is a value-of-computation heuristic, not an exact optimum.
            # Compare possible travel savings with building/calling/retrieving/
            # applying a route plus an estimated edge scan. All actual work is
            # still charged; this estimate never silently runs shortest path.
            lower = min(w for u,v,w in o.known_edges if u == o.position)
            possible_saving = max(0, greedy_distance-lower)*o.prices["travel"]
            overhead = 4*o.prices["action"]+len(o.known_edges)*o.prices["work"]
            if possible_saving <= overhead or o.remaining_work < 16:
                return move
        return self._solver(o, actions, "shortest_path", "route", pick)

    @staticmethod
    def _greedy_route(o, pick):
        """Follow locally cheapest forward edges in the public generated DAG.

        No shortest-path computation/backtracking. Local scans and arithmetic
        are measured controller CPU; the route is only a heuristic cost bound.
        """
        at, distance, first = o.position, 0, None
        visited = set()
        while at != o.goal["destination"] and at not in visited:
            visited.add(at)
            outgoing = [(w,v) for u,v,w in o.known_edges if u == at and v > at]
            if not outgoing:
                return None, 0
            w,v = min(outgoing, key=lambda pair:(pair[0],-pair[1]))
            distance += w
            if first is None:
                first = v
            at = v
        if at != o.goal["destination"] or distance > o.remaining_travel:
            return None, 0
        return pick("move", destination=first), distance

    def _solver(self, o, actions, primitive, role, pick):
        entries = [(h,p) for h,p in o.problems.items() if p["primitive"] == primitive
                   and p["state_version"] == o.state_version]
        if not entries:
            return pick("start_subset" if role == "subset" else "build_route")
        handle, entry = entries[-1]
        if role == "subset":
            for constraint in ("capacity", "funds", "incompatibility"):
                if constraint not in entry["problem"]["constraints"]:
                    return pick("add_constraint", problem=handle, constraint=constraint)
        records = [r for r in o.records if r.get("problem") == handle and
                   r.get("primitive") == primitive and r["state_version"] == o.state_version]
        if records:
            last = records[-1]
            if last.get("status") in {"success", "timeout"} and last.get("certificate_valid", False):
                if last["handle"] not in o.retrieved:
                    return pick("retrieve", handle=last["handle"])
                return pick("use_return", handle=last["handle"], **{"as":role})
            if last.get("status") not in {"timeout", "unknown"}:
                return pick("abstain")
        calls = [a for a in actions if a.kind == "call" and a.arguments["problem"] == handle]
        if not calls:
            return pick("abstain")
        calls.sort(key=lambda a:a.arguments["budget"])
        if self.mode == "always_tool":
            return calls[-1]
        used = max([r.get("work_units",0) for r in records], default=0)
        return next((a for a in calls if a.arguments["budget"] >= self.initial_budget
                     and a.arguments["budget"] > used), pick("abstain"))

    @staticmethod
    def _greedy(o):
        selected, weight, cost = [], 0, 0
        conflicts = {frozenset((a,b)) for a,b in o.incompatible}
        for category in o.goal["categories"]:
            candidates = [h for h,r in o.known_items.items() if r["category"] == category]
            candidates.sort(key=lambda h:(o.known_items[h]["weight"]+o.known_items[h]["price"],h))
            choice = next((h for h in candidates if
                weight+o.known_items[h]["weight"] <= o.goal["capacity"] and
                cost+o.known_items[h]["price"] <= o.goal["funds"] and
                all(frozenset((h,q)) not in conflicts for q in selected)), None)
            if choice is None:
                return None
            selected.append(choice)
            weight += o.known_items[choice]["weight"]
            cost += o.known_items[choice]["price"]
        return tuple(selected)


class FallbackReferencePolicy(ReferencePolicy):
    """Version2: route search failure may fall back to a public greedy walk.

    Uses the supplied catalogue unchanged. Optional remaining-budget actions
    are an independently versioned environment condition, not added here.
    """
    def __init__(self, mode="cheap_first", initial_budget=16):
        super().__init__(mode, initial_budget)
        self.reference_name = mode + "_fallback_v2"

    def _solver(self, o, actions, primitive, role, pick):
        action = super()._solver(o, actions, primitive, role, pick)
        if role == "route" and action.kind == "abstain":
            move, _ = self._greedy_route(o, pick)
            if move is not None:
                return move
        return action


def make_reference(name):
    if name.startswith("dep_"):
        from .campaign03_depworld import DepReference
        return DepReference(name.removeprefix("dep_"))
    if name.startswith("modular_"):
        from .campaign02_modular import ModularReference
        return ModularReference(name.removeprefix("modular_"))
    suffix = "_fallback_v2"
    if name.endswith(suffix):
        return FallbackReferencePolicy(name[:-len(suffix)])
    return ReferencePolicy(name)


def run_episode(world, policy, model_compute_tariff: float = 0.0) -> dict[str, Any]:
    """Harness receives environment; policy only receives copied observations.

    Actual policy CPU is separate from deterministic solver work units. Initial
    harness construction/generation is outside this function and must be timed
    by the experiment runner. No experiment launch occurs on import.
    """
    if model_compute_tariff < 0:
        raise ValueError("compute tariff must be nonnegative")
    cpu_start, wall_start = time.process_time(), time.monotonic()
    controller_cpu, trace = 0.0, []
    observation = world.observe()
    while not observation.done:
        if model_compute_tariff:
            world.charge_compute(model_compute_tariff)
        start = time.process_time()
        action = policy.choose(observation)
        controller_cpu += time.process_time()-start
        observation = world.step(action)
        trace.append({"action": asdict(action), "feedback": observation.feedback})
    result = world.evaluate()
    result.update(controller_cpu_seconds=controller_cpu,
                  episode_cpu_seconds=time.process_time()-cpu_start,
                  episode_wall_seconds=time.monotonic()-wall_start,
                  reference=getattr(policy,"reference_name",policy.mode), model_compute_tariff=model_compute_tariff, trace=trace)
    return result
