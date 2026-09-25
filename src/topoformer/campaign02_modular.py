"""Versioned modular workshop: publicly declared ordered stages over three primitives.

A goal is an ordered tuple of stages from STAGES. Each stage has a direct (greedy)
path and a charged solver path:
  select -> constrained_subset (as workshop-v1), route -> shortest_path,
  assign -> csp (tasks x slot domains with forbidden value pairs).
Stage-completing actions (commit_pending, commit_assignment, move, deliver,
use_return) are accepted only while their stage is the current (first
incomplete) stage; inspections, drafts and solver calls are allowed any time.
Hidden world state is evaluator-only; roster/map/item facts become public only
after the corresponding inspection. No schedule or stage-to-action mapping is
supplied to learned actors beyond the public goal and action catalogue.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import math
import random
import time
from typing import Any, Callable

from .campaign02_world import Action, Item, generate_world, validate_subset

VERSION = "workshop-modular-v1"
STAGES = ("select", "route", "assign")
PRIMITIVE = {"select": "constrained_subset", "route": "shortest_path", "assign": "csp"}
STAGE_OF_PRIMITIVE = {v: k for k, v in PRIMITIVE.items()}


@dataclass(frozen=True)
class ModularSpec:
    stages: tuple[str, ...]
    items: tuple[Item, ...]
    categories: tuple[int, ...]
    capacity: int
    funds: int
    incompatible: tuple[tuple[str, str], ...]
    edges: tuple[tuple[int, int, int], ...]
    start: int
    destination: int
    domains: tuple[tuple[int, ...], ...]
    forbidden: tuple[tuple[int, int, int, int], ...]
    step_limit: int = 64
    work_limit: int = 4096
    observation_price: float = .001
    action_price: float = .001
    work_price: float = .00001
    travel_price: float = .001
    travel_limit: int = 64
    compute_price: float = 0.0
    call_budgets: tuple[int, ...] = (16, 128, 1024)
    include_remaining_budget: bool = True
    # Public prior computation records from primitives NOT in this goal
    # (primitive, payload, problem_snapshot). Certificate-valid for their own
    # snapshot, irrelevant to this goal. Present from t=0; never charged.
    distractors: tuple = ()

    def __post_init__(self):
        if not self.stages or any(s not in STAGES for s in self.stages) or len(set(self.stages)) != len(self.stages):
            raise ValueError("stages must be distinct known stage names")
        if not isinstance(self.call_budgets, tuple) or not self.call_budgets or any(
                isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in self.call_budgets):
            raise ValueError("call_budgets must be a nonempty tuple of positive integers")


def generate_modular(seed: int, stages=("select", "route"), categories: int = 3, choices: int = 3,
                     locations: int = 7, tasks: int = 4, slots: int = 3, forbid: float = .3,
                     distractors: int = 0, **overrides: Any) -> ModularSpec:
    """Planted-feasible instances for every stage; planting is never public."""
    stages = tuple(stages)
    base = generate_world(seed, categories=categories, choices=choices, locations=locations, obstacle=False)
    rng = random.Random(f"modular-assign-{seed}")
    domains, planted = [], []
    for _ in range(tasks):
        size = rng.randint(2, slots)
        d = tuple(sorted(rng.sample(range(slots), size)))
        domains.append(d)
        planted.append(rng.choice(d))
    forbidden = []
    for i in range(tasks):
        for j in range(i+1, tasks):
            for x in domains[i]:
                for y in domains[j]:
                    if not (x == planted[i] and y == planted[j]) and rng.random() < forbid:
                        forbidden.append((i, x, j, y))
    params = dict(stages=stages, items=base.items, categories=base.categories, capacity=base.capacity,
                  funds=base.funds, incompatible=base.incompatible, edges=base.edges, start=base.start,
                  destination=base.destination, domains=tuple(domains), forbidden=tuple(forbidden))
    params["distractors"] = _distractors(seed, stages, base, domains, distractors)
    params.update(overrides)
    if "call_budgets" in params:
        params["call_budgets"] = tuple(params["call_budgets"])
    if "stages" in overrides:
        params["stages"] = tuple(overrides["stages"])
    return ModularSpec(**params)


def _distractors(seed, stages, base, domains, limit):
    """0..limit prior results of primitives absent from the goal (typed wrong for every stage)."""
    if limit <= 0:
        return ()
    rng = random.Random(f"modular-distractor-{seed}")
    absent = [p for st, p in PRIMITIVE.items() if st not in stages]
    rows = []
    for _ in range(rng.randint(0, limit) if absent else 0):
        primitive = rng.choice(absent)
        if primitive == "csp":
            d = [sorted(rng.sample(range(4), rng.randint(2, 4))) for _ in domains]
            payload = tuple(rng.choice(x) for x in d)
            snapshot = {"primitive": "csp", "problem": {"domains": d, "forbidden": [], "constraints": []}}
        elif primitive == "shortest_path":
            other = rng.randrange(1, base.destination)
            path = tuple(range(other, base.destination+1))
            w = {(a, b): c for a, b, c in base.edges}
            payload = (path, sum(w[(a, b)] for a, b in zip(path, path[1:])))
            snapshot = {"primitive": "shortest_path", "problem": {"n": base.destination+1,
                        "edges": [list(e) for e in base.edges], "start": other, "goal": base.destination}}
        else:
            handles = [x.handle for x in base.items]
            pick = [next(i for i, x in enumerate(base.items) if x.category == c and rng.random() < .6 or
                         (x.category == c and all(y.category != c for y in base.items[i+1:]))) for c in base.categories]
            payload = (tuple(pick), len(pick), sum(base.items[i].weight for i in pick), sum(base.items[i].price for i in pick))
            snapshot = {"primitive": "constrained_subset", "problem": {"items": [[x.category, x.weight, x.price, 1] for x in base.items],
                        "handles": handles, "constraints": []}}
        rows.append((primitive, payload, snapshot))
    return tuple(rows)


def validate_assignment(spec: ModularSpec, assignment: Any) -> tuple[bool, str]:
    """Independent evaluator check; never completes missing values."""
    if not isinstance(assignment, (list, tuple)) or len(assignment) != len(spec.domains):
        return False, "assignment_arity"
    if any(type(v) is not int or v not in d for v, d in zip(assignment, spec.domains)):
        return False, "domain"
    if any(assignment[i] == x and assignment[j] == y for i, x, j, y in spec.forbidden):
        return False, "forbidden_pair"
    return True, "valid"


@dataclass(frozen=True)
class ModularObservation:
    version: str
    state_version: int
    goal: dict[str, Any]
    completed: tuple[str, ...]
    current_stage: str | None
    item_inventory: tuple[dict[str, Any], ...]
    known_items: dict[str, dict[str, Any]]
    incompatible: tuple[tuple[str, str], ...]
    known_edges: tuple[tuple[int, int, int], ...] | None
    roster: dict[str, Any] | None
    records: tuple[dict[str, Any], ...]
    retrieved: dict[str, dict[str, Any]]
    problems: dict[str, dict[str, Any]]
    pending: tuple[str, ...]
    selected: tuple[str, ...]
    pending_assignment: dict[int, int]
    assigned: tuple[int, ...] | None
    position: int
    verified: bool
    done: bool
    remaining_steps: int
    remaining_travel: int
    remaining_work: int
    prices: dict[str, float]
    feedback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pending_assignment"] = {str(k): v for k, v in self.pending_assignment.items()}
        return d


class ModularWorkshop:
    """Actor surface observe()/step(); hidden spec is evaluator-only."""

    def __init__(self, spec: ModularSpec, executor: Callable | None = None, address_seed: int = 0):
        self._spec, self._executor = spec, executor
        self._items: dict[str, dict[str, Any]] = {}
        self._edges = {(a, b): w for a, b, w in spec.edges}
        self._map_known = self._roster_known = False
        self._records: dict[str, dict[str, Any]] = {}
        self._problems: dict[str, dict[str, Any]] = {}
        self._retrieved: dict[str, dict[str, Any]] = {}
        self._completed: list[str] = []
        self._selected: tuple[str, ...] = ()
        self._assigned: tuple[int, ...] | None = None
        self._pending: dict[int, str] = {}
        self._pending_assignment: dict[int, int] = {}
        self._position = spec.start
        self._verified = self._done = False
        self._steps = self._work = self._observations = self._travel = 0
        self._compute_units = 0.0
        self._solver_cpu = 0.0
        self._reductions: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._feedback: dict[str, Any] = {"status": "ready"}
        self._handle_rng = random.Random(address_seed)
        for i, (primitive, payload, snapshot) in enumerate(spec.distractors):
            self._record("computation", payload, primitive=primitive, problem=f"prior_{i}", status="success",
                         work_units=0, budget=0, problem_snapshot=deepcopy(snapshot), call_position=None,
                         certificate_valid=True, prior=True)

    def charge_compute(self, units: float) -> None:
        if isinstance(units, bool) or not isinstance(units, (int, float)) or not math.isfinite(units) or units < 0:
            raise ValueError("compute units must be finite nonnegative numbers")
        self._compute_units += units

    def _current(self) -> str | None:
        return next((s for s in self._spec.stages if s not in self._completed), None)

    def observe(self) -> ModularObservation:
        s = self._spec
        roster = None
        if self._roster_known:
            roster = {"domains": [list(d) for d in s.domains], "forbidden": [list(r) for r in s.forbidden]}
        return ModularObservation(
            VERSION, len(self._completed),
            {"stages": list(s.stages), "categories": list(s.categories), "capacity": s.capacity, "funds": s.funds,
             "destination": s.destination, "tasks": len(s.domains), "call_budgets": list(s.call_budgets),
             "include_remaining_budget": s.include_remaining_budget},
            tuple(self._completed), self._current(),
            tuple({"handle": x.handle, "category": x.category} for x in s.items), deepcopy(self._items),
            s.incompatible,
            tuple((a, b, w) for (a, b), w in sorted(self._edges.items())) if self._map_known else None,
            roster,
            tuple({k: deepcopy(v) for k, v in r.items() if k != "payload"} for r in self._records.values()),
            deepcopy(self._retrieved), deepcopy(self._problems), tuple(self._pending.values()), self._selected,
            dict(self._pending_assignment), self._assigned, self._position, self._verified, self._done,
            max(0, s.step_limit-self._steps), max(0, s.travel_limit-self._travel), max(0, s.work_limit-self._work),
            {"observation": s.observation_price, "action": s.action_price, "work": s.work_price,
             "travel": s.travel_price, "compute": s.compute_price},
            deepcopy(self._feedback))

    def _record(self, kind: str, payload: Any, **metadata: Any) -> str:
        handle = f"r{self._handle_rng.getrandbits(64):016x}"
        self._records[handle] = {"handle": handle, "kind": kind, "state_version": len(self._completed),
                                 "payload": deepcopy(payload), **metadata}
        return handle

    def step(self, action: Action) -> ModularObservation:
        if self._done:
            raise RuntimeError("episode finished")
        self._steps += 1
        try:
            self._feedback = self._apply(action)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self._feedback = {"status": "invalid_input", "reason": str(exc)}
        self._history.append({"action": asdict(action), "feedback": deepcopy(self._feedback),
                              "stage": self._current(), "work": self._work})
        if self._steps >= self._spec.step_limit:
            self._done = True
        return self.observe()

    def _stage_gate(self, stage: str) -> dict[str, Any] | None:
        if stage not in self._spec.stages:
            return {"status": "rejected", "reason": "stage_not_in_goal"}
        if self._current() != stage:
            return {"status": "rejected", "reason": "stage_order", "current": self._current()}
        return None

    def _complete(self, stage: str) -> None:
        self._completed.append(stage)

    def _apply(self, action: Action) -> dict[str, Any]:
        a, kind, s = action.arguments, action.kind, self._spec
        if kind == "inspect":
            target = a["target"]
            if target == "map":
                self._map_known = True
                payload = [(u, v, w) for (u, v), w in sorted(self._edges.items())]
            elif target == "roster":
                self._roster_known = True
                payload = {"domains": [list(d) for d in s.domains], "forbidden": [list(r) for r in s.forbidden]}
            else:
                item = next((x for x in s.items if x.handle == target), None)
                if item is None:
                    raise ValueError("unknown inspection target")
                payload = asdict(item)
                self._items[target] = payload
            self._observations += 1
            return {"status": "success", "return": self._record("observation", payload, source=target)}
        if kind == "start_subset":
            name = a["handle"]
            if name in self._problems or not self._items:
                raise ValueError("fresh problem handle and inspected domain required")
            rows = list(self._items.values())
            self._problems[name] = {"primitive": "constrained_subset", "state_version": len(self._completed),
                "problem": {"items": [[r["category"], r["weight"], r["price"], 1] for r in rows],
                            "handles": [r["handle"] for r in rows], "constraints": []}}
            return {"status": "success", "problem": name}
        if kind == "start_assign":
            name = a["handle"]
            if name in self._problems or not self._roster_known:
                raise ValueError("fresh problem handle and inspected roster required")
            self._problems[name] = {"primitive": "csp", "state_version": len(self._completed),
                "problem": {"domains": [list(d) for d in s.domains], "forbidden": [], "constraints": []}}
            return {"status": "success", "problem": name}
        if kind == "add_constraint":
            entry = self._problems[a["problem"]]
            p, c = entry["problem"], a["constraint"]
            if entry["primitive"] == "constrained_subset":
                if c not in ("capacity", "funds", "incompatibility"):
                    raise ValueError("unknown subset constraint")
                if c == "capacity":
                    p["capacity"] = s.capacity
                elif c == "funds":
                    p["max_cost"] = s.funds
                else:
                    p["forbidden_pairs"] = [[p["handles"].index(x), p["handles"].index(y)]
                        for x, y in s.incompatible if x in p["handles"] and y in p["handles"]]
            elif entry["primitive"] == "csp":
                if c != "forbidden":
                    raise ValueError("unknown assignment constraint")
                p["forbidden"] = [list(r) for r in s.forbidden]
            else:
                raise ValueError("constraints apply to subset or assignment drafts")
            if c not in p["constraints"]:
                p["constraints"].append(c)
            return {"status": "success", "problem": a["problem"], "constraint": c}
        if kind == "build_route":
            if not self._map_known:
                raise ValueError("map not inspected")
            name = a["handle"]
            if name in self._problems:
                raise ValueError("problem handle already exists")
            self._problems[name] = {"primitive": "shortest_path", "state_version": len(self._completed),
                "problem": {"n": 1+max(max(x, y) for x, y in self._edges),
                            "edges": [[x, y, w] for (x, y), w in sorted(self._edges.items())],
                            "start": a["start"], "goal": a["goal"]}}
            return {"status": "success", "problem": name}
        if kind == "choose_item":
            item = next((x for x in s.items if x.handle == a["item"]), None)
            if item is None:
                raise ValueError("unknown item")
            self._pending[item.category] = item.handle
            return {"status": "success", "pending": list(self._pending.values())}
        if kind == "choose_slot":
            task, slot = a["task"], a["slot"]
            if not self._roster_known or not 0 <= task < len(s.domains):
                raise ValueError("roster not inspected or unknown task")
            self._pending_assignment[task] = slot
            return {"status": "success"}
        if kind == "commit_pending":
            return self._commit_subset(list(self._pending.values()))
        if kind == "commit_assignment":
            if len(self._pending_assignment) != len(s.domains):
                return {"status": "rejected", "reason": "incomplete_assignment"}
            return self._commit_assignment([self._pending_assignment[i] for i in range(len(s.domains))])
        if kind == "call":
            entry = self._problems[a["problem"]]
            budget = a["budget"]
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise ValueError("budget must be positive integer")
            if budget > s.work_limit-self._work or self._executor is None:
                return {"status": "unavailable_resource"}
            start = time.process_time()
            result = self._executor(entry["primitive"], deepcopy(entry["problem"]), budget)
            self._solver_cpu += time.process_time()-start + float(result.get("child_cpu_seconds", 0))
            self._reductions.append({"primitive": entry["primitive"], "problem": a["problem"],
                "certificate_valid": result.get("certificate_valid"), "work_budget": budget,
                "correct": modular_reduction_matches(s, entry["primitive"], entry["problem"], self._edges, self._position)})
            used = result["work_units"]
            if not isinstance(used, int) or used < 0 or used > budget:
                raise RuntimeError("executor violated work contract")
            self._work += used
            handle = self._record("computation", result.get("payload"), primitive=entry["primitive"],
                problem=a["problem"], status=result["status"], work_units=used, budget=budget,
                problem_snapshot=deepcopy(entry), call_position=self._position,
                certificate_valid=bool(result.get("certificate_valid", False)))
            return {"status": result["status"], "return": handle}
        if kind == "retrieve":
            record = self._records[a["handle"]]
            self._retrieved[a["handle"]] = deepcopy(record)
            return {"status": "success", "record": deepcopy(record)}
        if kind == "use_return":
            record = self._records[a["handle"]]
            if record.get("status") not in ("success", "timeout") or not record.get("certificate_valid"):
                return {"status": "invalid_input", "reason": "no_valid_feasible_payload"}
            if a["handle"] not in self._retrieved:
                return {"status": "invalid_input", "reason": "retrieve_required"}
            use = a["as"]
            if STAGE_OF_PRIMITIVE.get(record.get("primitive")) != use:
                raise ValueError("return type mismatch")
            payload = record["payload"]
            if use == "select":
                p = record["problem_snapshot"]["problem"]
                return self._commit_subset([p["handles"][i] for i in payload[0]])
            if use == "route":
                if payload[0][0] != self._position:
                    return {"status": "invalid_input", "reason": "stale_result"}
                return self._walk(list(payload[0]))
            return self._commit_assignment(list(payload))
        if kind == "move":
            return self._walk([self._position, a["destination"]])
        if kind == "deliver":
            return self._walk(list(a["path"]))
        if kind == "verify":
            self._verified = len(self._completed) == len(s.stages)
            if self._verified:
                self._done = True
            return {"status": "success" if self._verified else "incomplete", "verified": self._verified}
        if kind == "think":
            return {"status": "success"}
        if kind == "abstain":
            self._done = True
            return {"status": "unknown"}
        raise ValueError("unknown action")

    def _commit_subset(self, handles):
        gate = self._stage_gate("select")
        if gate:
            return gate
        valid, reason = validate_subset(self._spec, handles)
        if not valid:
            return {"status": "rejected", "reason": reason}
        self._selected = tuple(handles)
        self._complete("select")
        return {"status": "success", "stage_completed": "select"}

    def _commit_assignment(self, values):
        gate = self._stage_gate("assign")
        if gate:
            return gate
        valid, reason = validate_assignment(self._spec, values)
        if not valid:
            return {"status": "rejected", "reason": reason}
        self._assigned = tuple(values)
        self._complete("assign")
        return {"status": "success", "stage_completed": "assign"}

    def _walk(self, path):
        gate = self._stage_gate("route")
        if gate:
            return gate
        if not isinstance(path, list) or len(path) < 2 or path[0] != self._position:
            raise ValueError("path starts at current location")
        if any((u, v) not in self._edges for u, v in zip(path, path[1:])):
            return {"status": "rejected", "reason": "missing_edge"}
        for u, v in zip(path, path[1:]):
            w = self._edges[(u, v)]
            if w > self._spec.travel_limit-self._travel:
                return {"status": "unavailable_resource", "reason": "travel_budget", "position": self._position}
            self._travel += w
            self._position = v
        if self._position == self._spec.destination:
            self._complete("route")
            return {"status": "success", "position": self._position, "stage_completed": "route"}
        return {"status": "success", "position": self._position}

    def evaluate(self) -> dict[str, Any]:
        s = self._spec
        cost = (self._steps*s.action_price+self._travel*s.travel_price+self._observations*s.observation_price
                + self._work*s.work_price+self._compute_units*s.compute_price)
        return {"verified_success": self._verified, "stages_completed": list(self._completed),
                "stage_completion": len(self._completed)/len(s.stages), "utility": float(self._verified)-cost,
                "steps": self._steps, "observations": self._observations, "work_units": self._work,
                "solver_cpu_seconds": self._solver_cpu, "travel_distance": self._travel,
                "compute_units": self._compute_units, "modeled_compute_cost": self._compute_units*s.compute_price,
                "cost": cost, "reductions": deepcopy(self._reductions), "history": deepcopy(self._history)}


def modular_reduction_matches(spec, primitive, problem, edges, position) -> bool:
    """Privileged evaluator-only reduction audit, never a mask."""
    try:
        if primitive == "csp":
            return ([tuple(d) for d in problem["domains"]] == [tuple(d) for d in spec.domains]
                    and {tuple(r) for r in problem.get("forbidden", [])} == set(spec.forbidden))
        from .campaign02_world import reduction_matches_world
        return reduction_matches_world(spec, primitive, problem, edges, position)
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def modular_executor(primitive: str, problem: dict[str, Any], max_work: int, *, execute_call=None) -> dict[str, Any]:
    """Supplied deterministic lowering; csp drafts lower to (domains, forbidden)."""
    from .campaign02_world import protocol_executor
    if primitive != "csp":
        return protocol_executor(primitive, problem, max_work, execute_call=execute_call)
    from .campaign02_protocol import Budget, Call, execute_isolated, validate_result
    args = (tuple(tuple(d) for d in problem["domains"]), tuple(tuple(r) for r in problem.get("forbidden", [])))
    call = Call("csp", args, budget=Budget(max_work))
    result = (execute_isolated if execute_call is None else execute_call)(call)
    return {"status": result.status, "payload": result.payload, "work_units": result.work_units,
            "cpu_seconds": result.cpu_seconds, "child_cpu_seconds": result.cpu_seconds,
            "certificate": result.certificate, "certificate_valid": validate_result(call, result)}


KINDS = ("inspect", "start_subset", "start_assign", "add_constraint", "build_route", "call", "retrieve",
         "choose_item", "choose_slot", "commit_pending", "commit_assignment", "use_return", "deliver", "move",
         "verify", "think", "abstain")
CONSTRAINTS = {"constrained_subset": ("capacity", "funds", "incompatibility"), "csp": ("forbidden",)}


def action_catalog(o: ModularObservation) -> list[Action]:
    """Public syntactic choices for every stage (including out-of-order ones)."""
    if o.done:
        return []
    budgets = tuple(dict.fromkeys(tuple(o.goal["call_budgets"]) + ((o.remaining_work,) if
                    o.goal.get("include_remaining_budget") and o.remaining_work > 0 else ())))
    acts = [Action("verify"), Action("think"), Action("abstain"), Action("commit_pending"), Action("commit_assignment"),
            Action("inspect", {"target": "map"}), Action("inspect", {"target": "roster"})]
    acts += [Action("inspect", {"target": r["handle"]}) for r in o.item_inventory]
    acts += [Action("choose_item", {"item": r["handle"]}) for r in o.item_inventory]
    name = f"problem_{len(o.problems)}"
    if o.known_items:
        acts.append(Action("start_subset", {"handle": name}))
    if o.roster is not None:
        acts.append(Action("start_assign", {"handle": name}))
        acts += [Action("choose_slot", {"task": t, "slot": v}) for t, d in enumerate(o.roster["domains"]) for v in d]
    if o.known_edges is not None:
        acts += [Action("move", {"destination": v}) for u, v, w in o.known_edges if u == o.position]
        acts.append(Action("build_route", {"handle": name, "start": o.position, "goal": o.goal["destination"]}))
    for handle, p in o.problems.items():
        acts += [Action("add_constraint", {"problem": handle, "constraint": c}) for c in CONSTRAINTS.get(p["primitive"], ())]
        acts += [Action("call", {"problem": handle, "budget": b}) for b in budgets if b <= o.remaining_work]
    for r in o.records:
        acts.append(Action("retrieve", {"handle": r["handle"]}))
        if r["kind"] == "computation":
            acts += [Action("use_return", {"handle": r["handle"], "as": st}) for st in STAGES]
    return acts


def _log(x, top=12.0):
    return math.log2(1+max(0.0, float(x)))/top


def encode_observation(o: ModularObservation) -> list[float]:
    stages = o.goal["stages"]
    cur = o.current_stage
    fb = o.feedback.get("status")
    return ([float(cur == s) for s in STAGES] + [float(cur is None)]
            + [float(s in stages and s not in o.completed) for s in STAGES]
            + [len(stages)/4, len(o.completed)/4, (len(stages)-len(o.completed))/4,
               float(o.roster is not None), float(o.known_edges is not None),
               len(o.known_items)/max(1, len(o.item_inventory)), len(o.item_inventory)/32, o.goal["tasks"]/8,
               len(o.pending)/8, len(o.pending_assignment)/8, o.position/32, o.goal["destination"]/32,
               _log(o.remaining_work), _log(o.remaining_steps, 7), o.remaining_travel/64,
               len(o.problems)/8, len(o.records)/16,
               o.prices["observation"], o.prices["action"], o.prices["work"]*100, o.prices["travel"]*10, o.prices["compute"]*100]
            + [float(fb == x) for x in ("success", "rejected", "timeout", "invalid_input", "unavailable_resource", "incomplete")]
            + [float(o.feedback.get("reason") == "stage_order")])


def _action_stage(o: ModularObservation, action: Action) -> str | None:
    a, k = action.arguments, action.kind
    if k in ("choose_item", "commit_pending", "start_subset") or (k == "inspect" and a.get("target") not in ("map", "roster")):
        return "select"
    if k in ("move", "deliver", "build_route") or (k == "inspect" and a.get("target") == "map"):
        return "route"
    if k in ("choose_slot", "commit_assignment", "start_assign") or (k == "inspect" and a.get("target") == "roster"):
        return "assign"
    if k == "use_return":
        return a["as"]
    if k in ("call", "add_constraint"):
        return STAGE_OF_PRIMITIVE.get(o.problems.get(a["problem"], {}).get("primitive"))
    if k == "retrieve":
        rec = next((r for r in o.records if r["handle"] == a["handle"]), {})
        return STAGE_OF_PRIMITIVE.get(rec.get("primitive"))
    return None


def encode_action(o: ModularObservation, action: Action) -> list[float]:
    """Stage-typed candidate features. No handle spelling; no solved labels."""
    a, k = action.arguments, action.kind
    stage = _action_stage(o, action)
    out = [float(k == x) for x in KINDS] + [float(stage == s) for s in STAGES]
    out += [float(stage is not None and stage == o.current_stage), float(stage in o.completed),
            float(stage is not None and stage in o.goal["stages"])]
    # item facts
    h = a.get("item", a.get("target"))
    row = next((r for r in o.item_inventory if r["handle"] == h), None)
    if row is not None:
        known = o.known_items.get(h, {})
        conflicts = [y if x == h else x for x, y in o.incompatible if h in (x, y)]
        cat = [r["handle"] for r in o.item_inventory if r["category"] == row["category"]]
        peers = [o.known_items[x]["weight"]+o.known_items[x]["price"] for x in cat if x in o.known_items]
        rank = sum(p < known["weight"]+known["price"] for p in peers) if known else 0
        pend_w = sum(o.known_items.get(x, {}).get("weight", 0) for x in o.pending)
        pend_p = sum(o.known_items.get(x, {}).get("price", 0) for x in o.pending)
        pend_cats = {r["category"] for r in o.item_inventory if r["handle"] in o.pending}
        out += [1.0, float(bool(known)), known.get("weight", 0)/16, known.get("price", 0)/16, float(h in o.pending),
                len(conflicts)/8, float(any(x in o.pending for x in conflicts)), rank/8, float(bool(known) and rank == 0),
                float(row["category"] in pend_cats), (o.goal["capacity"]-pend_w)/64, (o.goal["funds"]-pend_p)/64]
    else:
        out += [0.0]*12
    # assignment facts
    if k == "choose_slot" and o.roster is not None:
        t, v = a["task"], a["slot"]
        clash = sum(1 for i, x, j, y in o.roster["forbidden"]
                    if (i == t and x == v and o.pending_assignment.get(j) == y) or (j == t and y == v and o.pending_assignment.get(i) == x))
        involve = sum(1 for i, x, j, y in o.roster["forbidden"] if (i == t and x == v) or (j == t and y == v))
        out += [1.0, t/8, v/8, float(t in o.pending_assignment), float(o.pending_assignment.get(t) == v),
                float(clash > 0), involve/8, len(o.roster["domains"][t])/8,
                float(all(i in o.pending_assignment for i in range(t))), float(v == min(o.roster["domains"][t]))]
    else:
        out += [0.0]*10
    if k == "commit_assignment":
        out += [len(o.pending_assignment)/max(1, o.goal["tasks"]), float(len(o.pending_assignment) == o.goal["tasks"])]
    elif k == "commit_pending":
        out += [len(o.pending)/max(1, len(o.goal["categories"])), float(len(o.pending) == len(o.goal["categories"]))]
    else:
        out += [0.0, 0.0]
    # solver draft / budget facts
    problem = o.problems.get(a.get("problem")) if k in ("call", "add_constraint") else None
    if problem is not None:
        tried = [r for r in o.records if r.get("problem") == a["problem"] and r.get("problem_snapshot") == problem]
        budget = a.get("budget", 0)
        most = max((r.get("work_units", 0) for r in tried), default=0)
        cons = problem["problem"].get("constraints", [])
        needed = CONSTRAINTS.get(problem["primitive"], ())
        out += [1.0, _log(budget), float(k == "call" and budget > most),
                float(k == "call" and any(r.get("status") == "timeout" and r.get("budget", 0) >= budget for r in tried)),
                float(k == "call" and budget == o.remaining_work), sum(r.get("status") == "timeout" for r in tried)/4,
                float(any(r.get("status") == "success" for r in tried)), len(cons)/3,
                float(all(c in cons for c in needed)), float(a.get("constraint") in cons),
                _log(len(problem["problem"].get("items", problem["problem"].get("domains", []))), 5)]
    else:
        out += [0.0]*11
    # record facts
    rec = next((r for r in o.records if r["handle"] == a.get("handle")), None) if k in ("retrieve", "use_return") else None
    if rec is not None:
        out += [1.0] + [float(rec.get("status") == x) for x in ("success", "timeout", "infeasible", "invalid", "unknown")]
        out += [float(rec.get("certificate_valid", False)), float(a.get("handle") in o.retrieved),
                float(k == "use_return" and STAGE_OF_PRIMITIVE.get(rec.get("primitive")) == a.get("as")),
                float(rec.get("kind") == "computation")]
    else:
        out += [0.0]*10
    # movement facts
    if k == "move" and o.known_edges is not None:
        v = a["destination"]
        w = next((w for x, y, w in o.known_edges if x == o.position and y == v), 0)
        ahead = [w2 for x, y, w2 in o.known_edges if x == v]
        direct = next((w2 for x, y, w2 in o.known_edges if x == v and y == o.goal["destination"]), None)
        out += [1.0, w/32, float(v == o.goal["destination"]), min(ahead, default=0)/32, float(direct is not None),
                (direct or 0)/32, float(v > o.position)]
    else:
        out += [0.0]*7
    return out


def encode_public(o: ModularObservation, actions: list[Action]):
    return encode_observation(o), [encode_action(o, x) for x in actions]


class ModularReference:
    """Public cheap-first / cheap / always-tool heuristics per current stage.

    Supplied schedules for solvability, bootstrapping and comparison; charged
    identically; not learned planning.
    """
    def __init__(self, mode: str = "cheap_first", initial_budget: int = 16):
        if mode not in {"cheap", "always_tool", "cheap_first"}:
            raise ValueError("unknown modular reference mode")
        self.mode, self.initial_budget = mode, initial_budget
        self.reference_name = "modular_" + mode

    def choose(self, o: ModularObservation, catalog=None) -> Action:
        acts = action_catalog(o) if catalog is None else catalog
        def pick(kind, **kw):
            return next((x for x in acts if x.kind == kind and all(x.arguments.get(k2) == v for k2, v in kw.items())), None)
        if o.done:
            raise ValueError("cannot act after episode end")
        stage = o.current_stage
        if stage is None:
            return pick("verify")
        if stage == "select":
            for r in o.item_inventory:
                if r["handle"] not in o.known_items:
                    return pick("inspect", target=r["handle"])
            if self.mode != "always_tool" and not self._attempted(o, "constrained_subset"):
                chosen = self._greedy_subset(o)
                if chosen is not None:
                    for h in chosen:
                        if h not in o.pending:
                            return pick("choose_item", item=h)
                    if not (o.feedback.get("status") == "rejected" and o.feedback.get("reason") != "stage_order"):
                        return pick("commit_pending")
            if self.mode == "cheap":
                return pick("abstain")
            return self._solver(o, acts, "constrained_subset", pick)
        if stage == "assign":
            if o.roster is None:
                return pick("inspect", target="roster")
            if self.mode != "always_tool" and not self._attempted(o, "csp"):
                chosen = self._greedy_assign(o)
                if chosen is not None:
                    for t, v in enumerate(chosen):
                        if o.pending_assignment.get(t) != v:
                            return pick("choose_slot", task=t, slot=v)
                    if not (o.feedback.get("status") == "rejected" and o.feedback.get("reason") != "stage_order"):
                        return pick("commit_assignment")
            if self.mode == "cheap":
                return pick("abstain")
            return self._solver(o, acts, "csp", pick)
        if o.known_edges is None:
            return pick("inspect", target="map")
        move, distance = self._greedy_route(o, pick)
        if self.mode == "cheap":
            return move or pick("abstain")
        if self.mode == "cheap_first" and move is not None:
            lower = min(w for u, v, w in o.known_edges if u == o.position)
            saving = max(0, distance-lower)*o.prices["travel"]
            overhead = 4*o.prices["action"]+len(o.known_edges)*o.prices["work"]
            if saving <= overhead or o.remaining_work < 16:
                return move
        action = self._solver(o, acts, "shortest_path", pick)
        if action.kind == "abstain" and move is not None:
            return move
        return action

    @staticmethod
    def _attempted(o, primitive):
        return any(p["primitive"] == primitive for p in o.problems.values())

    @staticmethod
    def _greedy_subset(o):
        selected, weight, cost = [], 0, 0
        conflicts = {frozenset(p) for p in o.incompatible}
        for c in o.goal["categories"]:
            cands = sorted((h for h, r in o.known_items.items() if r["category"] == c),
                           key=lambda h: (o.known_items[h]["weight"]+o.known_items[h]["price"], h))
            choice = next((h for h in cands if weight+o.known_items[h]["weight"] <= o.goal["capacity"]
                           and cost+o.known_items[h]["price"] <= o.goal["funds"]
                           and all(frozenset((h, q)) not in conflicts for q in selected)), None)
            if choice is None:
                return None
            selected.append(choice)
            weight += o.known_items[choice]["weight"]
            cost += o.known_items[choice]["price"]
        return selected

    @staticmethod
    def _greedy_assign(o):
        chosen = []
        forbidden = {tuple(r) for r in o.roster["forbidden"]}
        for t, d in enumerate(o.roster["domains"]):
            v = next((v for v in d if all((i, chosen[i], t, v) not in forbidden for i in range(t))), None)
            if v is None:
                return None
            chosen.append(v)
        return chosen

    @staticmethod
    def _greedy_route(o, pick):
        at, distance, first, seen = o.position, 0, None, set()
        while at != o.goal["destination"] and at not in seen:
            seen.add(at)
            out = [(w, v) for u, v, w in o.known_edges if u == at and v > at]
            if not out:
                return None, 0
            w, v = min(out, key=lambda p: (p[0], -p[1]))
            distance += w
            first = v if first is None else first
            at = v
        if at != o.goal["destination"] or distance > o.remaining_travel:
            return None, 0
        return pick("move", destination=first), distance

    def _solver(self, o, acts, primitive, pick):
        stage = STAGE_OF_PRIMITIVE[primitive]
        entries = [(h, p) for h, p in o.problems.items() if p["primitive"] == primitive
                   and (primitive != "shortest_path" or p["problem"]["start"] == o.position)]
        if not entries:
            return pick({"constrained_subset": "start_subset", "csp": "start_assign", "shortest_path": "build_route"}[primitive])
        handle, entry = entries[-1]
        for c in CONSTRAINTS.get(primitive, ()):
            if c not in entry["problem"]["constraints"]:
                return pick("add_constraint", problem=handle, constraint=c)
        records = [r for r in o.records if r.get("problem") == handle and r.get("problem_snapshot") == entry]
        if records:
            last = records[-1]
            if last.get("status") in {"success", "timeout"} and last.get("certificate_valid", False):
                if last["handle"] not in o.retrieved:
                    return pick("retrieve", handle=last["handle"])
                return pick("use_return", handle=last["handle"], **{"as": stage})
            if last.get("status") not in {"timeout", "unknown"}:
                return pick("abstain")
        calls = sorted((x for x in acts if x.kind == "call" and x.arguments["problem"] == handle),
                       key=lambda x: x.arguments["budget"])
        if not calls:
            return pick("abstain")
        if self.mode == "always_tool":
            return calls[-1]
        used = max((r.get("work_units", 0) for r in records), default=0)
        return next((x for x in calls if x.arguments["budget"] >= self.initial_budget and x.arguments["budget"] > used),
                    pick("abstain"))
