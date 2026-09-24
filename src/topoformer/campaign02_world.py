"""Versioned, partially observed workshop. No actor method receives a solution.

Executors consume *submitted* problems, never the hidden world. Solver-result
validity and correspondence to the actual world are distinct audit questions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import random
import time
from typing import Any, Callable, Mapping

VERSION = "workshop-v1"


@dataclass(frozen=True)
class Item:
    handle: str
    category: int
    weight: int
    price: int


@dataclass(frozen=True)
class WorldSpec:
    items: tuple[Item, ...]
    categories: tuple[int, ...]
    capacity: int
    funds: int
    incompatible: tuple[tuple[str, str], ...]
    edges: tuple[tuple[int, int, int], ...]
    start: int
    destination: int
    blocked_edge: tuple[int, int] | None = None
    step_limit: int = 40
    work_limit: int = 4096
    observation_price: float = .001
    action_price: float = .001
    work_price: float = .00001


@dataclass(frozen=True)
class Action:
    kind: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    version: str
    state_version: int
    goal: dict[str, Any]
    item_inventory: tuple[dict[str, Any], ...]
    known_items: dict[str, dict[str, Any]]
    known_edges: tuple[tuple[int, int, int], ...] | None
    incompatible: tuple[tuple[str, str], ...]
    records: tuple[dict[str, Any], ...]
    retrieved: dict[str, dict[str, Any]]
    problems: dict[str, dict[str, Any]]
    pending: tuple[str, ...]
    selected: tuple[str, ...]
    position: int
    delivered: bool
    verified: bool
    done: bool
    remaining_steps: int
    remaining_work: int
    prices: dict[str, float]
    feedback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_world(seed: int, categories: int = 2, choices: int = 3,
                   locations: int = 5, obstacle: bool = False, **overrides: Any) -> WorldSpec:
    """Planted feasible instances; planting and generation seed are never public.

    Feasibility is a generator guarantee, not optimality or a hardness claim.
    Randomized item positions/handles avoid revealing which alternatives planted.
    """
    if min(categories, choices) < 1 or locations < 3:
        raise ValueError("positive category/choice counts and >=3 locations required")
    rng = random.Random(seed)
    handles = rng.sample(range(10_000, 999_999), categories * choices)
    items, planted = [], []
    for c in range(categories):
        group = [Item(f"i{handles[c*choices+j]}", c, rng.randint(1, 7), rng.randint(1, 9))
                 for j in range(choices)]
        items.extend(group)
        planted.append(rng.choice(group))
    rng.shuffle(items)
    planted_handles = {x.handle for x in planted}
    incompatible = tuple((x.handle, y.handle) for i, x in enumerate(items) for y in items[i+1:]
                         if x.category != y.category and not {x.handle,y.handle} <= planted_handles
                         and rng.random() < .18)
    # Chain remains open after optional shortcut obstruction.
    edges = {(i, i+1): rng.randint(1, 4) for i in range(locations-1)}
    edges[(0, locations-1)] = rng.randint(2, 6)
    for i in range(locations):
        for j in range(i+2, locations):
            if rng.random() < .25:
                edges.setdefault((i,j), rng.randint(1, 7))
    params = dict(items=tuple(items), categories=tuple(range(categories)),
                  capacity=sum(x.weight for x in planted)+rng.randint(0,2),
                  funds=sum(x.price for x in planted)+rng.randint(0,3),
                  incompatible=incompatible, edges=tuple((a,b,w) for (a,b),w in sorted(edges.items())),
                  start=0, destination=locations-1,
                  blocked_edge=(0,locations-1) if obstacle else None)
    params.update(overrides)
    return WorldSpec(**params)


def validate_subset(spec: WorldSpec, handles: Any) -> tuple[bool, str]:
    """Independent evaluator check. It never fills missing selections."""
    if not isinstance(handles, (list, tuple)) or not all(isinstance(h,str) for h in handles):
        return False, "invalid_handles"
    if len(set(handles)) != len(handles):
        return False, "duplicate_item"
    inventory = {x.handle:x for x in spec.items}
    if any(h not in inventory for h in handles):
        return False, "unknown_item"
    chosen = [inventory[h] for h in handles]
    if sorted(x.category for x in chosen) != sorted(spec.categories):
        return False, "category_coverage"
    if sum(x.weight for x in chosen) > spec.capacity:
        return False, "capacity"
    if sum(x.price for x in chosen) > spec.funds:
        return False, "funds"
    if any(a in handles and b in handles for a,b in spec.incompatible):
        return False, "compatibility"
    return True, "valid"


class Workshop:
    """Actor surface: observe()/step(). Hidden state lives only in simulator.

    Executor signature: executor(primitive, submitted_problem, max_work) -> dict.
    Required result fields: status, payload, work_units. CPU cost measured here.
    The harness must supply bounded executors; policies cannot supply callbacks.
    """
    def __init__(self, spec: WorldSpec, executor: Callable | None = None):
        self._spec, self._executor = spec, executor
        self._items: dict[str, dict[str, Any]] = {}
        self._edges = {(a,b):w for a,b,w in spec.edges}
        self._map_known = False
        self._records: dict[str, dict[str, Any]] = {}
        self._problems: dict[str, dict[str, Any]] = {}
        self._retrieved: dict[str, dict[str,Any]] = {}
        self._version = 0
        self._selected: tuple[str,...] = ()
        self._position = spec.start
        self._delivered = self._verified = self._done = False
        self._steps = self._work = self._observations = 0
        self._solver_cpu = 0.0
        self._feedback: dict[str,Any] = {"status":"ready"}
        self._history: list[dict[str,Any]] = []
        self._pending: dict[int,str] = {}
        self._handle_rng = random.Random("|".join(x.handle for x in spec.items))

    def observe(self) -> Observation:
        s = self._spec
        return Observation(VERSION, self._version,
            {"categories":list(s.categories), "capacity":s.capacity, "funds":s.funds,
             "destination":s.destination},
            tuple({"handle":x.handle,"category":x.category} for x in s.items),
            deepcopy(self._items),
            tuple((a,b,w) for (a,b),w in sorted(self._edges.items())) if self._map_known else None,
            s.incompatible,
            tuple({k:deepcopy(v) for k,v in r.items() if k != "payload"} for r in self._records.values()),
            deepcopy(self._retrieved),deepcopy(self._problems),tuple(self._pending.values()),self._selected,self._position,self._delivered,self._verified,self._done,
            max(0,s.step_limit-self._steps), max(0,s.work_limit-self._work),
            {"observation":s.observation_price,"action":s.action_price,"work":s.work_price},
            deepcopy(self._feedback))

    def _record(self, kind: str, payload: Any, **metadata: Any) -> str:
        handle = f"r{self._handle_rng.getrandbits(64):016x}"
        self._records[handle] = {"handle":handle,"kind":kind,"state_version":self._version,
                                 "payload":deepcopy(payload), **metadata}
        return handle

    def step(self, action: Action) -> Observation:
        if self._done:
            raise RuntimeError("episode finished")
        self._steps += 1
        try:
            self._feedback = self._apply(action)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self._feedback = {"status":"invalid_input","reason":str(exc)}
        self._history.append({"action":asdict(action),"feedback":deepcopy(self._feedback),
                              "state_version":self._version,"work":self._work})
        if self._steps >= self._spec.step_limit:
            self._done = True
        return self.observe()

    def _apply(self, action: Action) -> dict[str,Any]:
        a, kind = action.arguments, action.kind
        if kind == "inspect":
            target = a["target"]
            if target == "map":
                self._map_known = True
                payload = [(u,v,w) for (u,v),w in sorted(self._edges.items())]
            else:
                item = next((x for x in self._spec.items if x.handle == target),None)
                if item is None:
                    raise ValueError("unknown inspection target")
                payload = asdict(item)
                self._items[target] = payload
            self._observations += 1
            return {"status":"success","return":self._record("observation",payload,source=target)}
        if kind == "build":
            # Any typed problem may be submitted: no gold instance correction.
            name, primitive, problem = a["handle"],a["primitive"],a["problem"]
            if not isinstance(name,str) or name in self._problems or not isinstance(problem,dict):
                raise ValueError("problem handle must be fresh; problem must be a mapping")
            self._problems[name] = {"primitive":primitive,"problem":deepcopy(problem),"state_version":self._version}
            return {"status":"success","problem":name}
        if kind == "start_subset":
            name = a["handle"]
            if name in self._problems or not self._items:
                raise ValueError("fresh problem handle and inspected domain required")
            rows = list(self._items.values())
            self._problems[name] = {"primitive":"constrained_subset", "state_version":self._version,
                "problem":{"items":[[r["category"],r["weight"],r["price"],1] for r in rows],
                           "handles":[r["handle"] for r in rows], "constraints":[]}}
            return {"status":"success","problem":name}
        if kind == "add_constraint":
            entry = self._problems[a["problem"]]
            if entry["primitive"] != "constrained_subset":
                raise ValueError("constraint requires subset instance")
            constraint = a["constraint"]
            if constraint not in ("capacity","funds","incompatibility"):
                raise ValueError("unknown public constraint identifier")
            p = entry["problem"]
            if constraint not in p["constraints"]:
                p["constraints"].append(constraint)
            if constraint == "capacity":
                p["capacity"] = self._spec.capacity
            elif constraint == "funds":
                p["max_cost"] = self._spec.funds
            else:
                p["forbidden_pairs"] = [[p["handles"].index(x),p["handles"].index(y)]
                    for x,y in self._spec.incompatible if x in p["handles"] and y in p["handles"]]
            return {"status":"success","problem":a["problem"],"constraint":constraint}
        if kind == "build_route":
            if not self._map_known:
                raise ValueError("map not inspected")
            name = a["handle"]
            if name in self._problems:
                raise ValueError("problem handle already exists")
            self._problems[name] = {"primitive":"shortest_path","state_version":self._version,
                "problem":{"n":1+max(max(x,y) for x,y in self._edges),
                           "edges":[[x,y,w] for (x,y),w in sorted(self._edges.items())],
                           "start":a["start"],"goal":a["goal"]}}
            return {"status":"success","problem":name}
        if kind == "choose_item":
            item = next((x for x in self._spec.items if x.handle == a["item"]),None)
            if item is None:
                raise ValueError("unknown item")
            self._pending[item.category] = item.handle
            return {"status":"success","pending":list(self._pending.values())}
        if kind == "commit_pending":
            return self._apply(Action("commit_subset",{"items":list(self._pending.values())}))
        if kind == "use_return":
            record = self._records[a["handle"]]
            if record.get("primitive") == "shortest_path" and record["state_version"] != self._version:
                return {"status":"invalid_input","reason":"stale_result"}
            if record.get("status") != "success":
                return {"status":"invalid_input","reason":"not_success_result"}
            if a["handle"] not in self._retrieved:
                return {"status":"invalid_input","reason":"retrieve_required"}
            payload = record["payload"]
            if a["as"] == "subset":
                if record.get("primitive") != "constrained_subset":
                    raise ValueError("return type mismatch")
                p = self._problems[record["problem"]]["problem"]
                return self._apply(Action("commit_subset",{"items":[p["handles"][i] for i in payload[0]]}))
            if a["as"] == "route":
                if record.get("primitive") != "shortest_path":
                    raise ValueError("return type mismatch")
                return self._apply(Action("deliver",{"path":payload[0]}))
            raise ValueError("unknown return use")
        if kind == "call":
            entry = self._problems[a["problem"]]
            if entry["primitive"] == "shortest_path" and entry["state_version"] != self._version:
                return {"status":"invalid_input","reason":"stale_problem"}
            budget = a["budget"]
            if not isinstance(budget,int) or isinstance(budget,bool) or budget <= 0:
                raise ValueError("budget must be positive integer")
            if budget > self._spec.work_limit-self._work or self._executor is None:
                return {"status":"unavailable_resource"}
            start = time.process_time()
            result = self._executor(entry["primitive"],deepcopy(entry["problem"]),budget)
            self._solver_cpu += time.process_time()-start
            used = result["work_units"]
            if not isinstance(used,int) or used < 0 or used > budget:
                raise RuntimeError("executor violated work contract")
            self._work += used
            handle = self._record("computation",result.get("payload"),primitive=entry["primitive"],
                                  problem=a["problem"],status=result["status"],work_units=used)
            return {"status":result["status"],"return":handle}
        if kind == "retrieve":
            record = self._records[a["handle"]]
            self._retrieved[a["handle"]] = deepcopy(record)
            return {"status":"success","record":deepcopy(record),
                    "stale":record["state_version"] != self._version}
        if kind == "commit_subset":
            handles = a["items"]
            valid, reason = validate_subset(self._spec,handles)
            if not valid:
                return {"status":"rejected","reason":reason}
            self._selected = tuple(handles)
            return {"status":"success","selected":list(handles)}
        if kind == "deliver":
            if not self._selected:
                return {"status":"rejected","reason":"no_selection"}
            path = a["path"]
            if not isinstance(path,(list,tuple)) or len(path)<2 or path[0] != self._position:
                raise ValueError("route must start at current position")
            if path[-1] != self._spec.destination:
                return {"status":"rejected","reason":"wrong_destination"}
            # No teleportation/partial movement on malformed path.
            if any((u,v) not in self._edges for u,v in zip(path,path[1:])):
                return {"status":"rejected","reason":"missing_edge"}
            for u,v in zip(path,path[1:]):
                if (u,v) == self._spec.blocked_edge:
                    del self._edges[(u,v)]
                    self._version += 1
                    return {"status":"obstacle","edge":[u,v],"position":self._position}
                self._position = v
            self._delivered = True
            return {"status":"success","delivered":True}
        if kind == "verify":
            valid,_ = validate_subset(self._spec,self._selected)
            self._verified = bool(valid and self._delivered and self._position == self._spec.destination)
            if self._verified:
                self._done = True
            return {"status":"success" if self._verified else "incomplete","verified":self._verified}
        if kind == "think":
            return {"status":"success"}
        if kind == "abstain":
            self._done = True
            return {"status":"unknown"}
        raise ValueError("unknown action")

    def evaluate(self) -> dict[str,Any]:
        """Evaluator-only summary, never returned to actor during an episode."""
        s = self._spec
        cost = self._steps*s.action_price+self._observations*s.observation_price+self._work*s.work_price
        return {"verified_success":self._verified,"delivered":self._delivered,"utility":float(self._verified)-cost,
                "steps":self._steps,"observations":self._observations,"work_units":self._work,
                "solver_cpu_seconds":self._solver_cpu,"cost":cost,"history":deepcopy(self._history)}


ACTION_KINDS = ("inspect","start_subset","add_constraint","build_route","call","retrieve",
                "choose_item","commit_pending","use_return","deliver","verify","think","abstain")
CONSTRAINTS = ("capacity","funds","incompatibility")


def action_catalog(observation: Observation, budgets: tuple[int,...] = (16,128,1024)) -> list[Action]:
    """Public syntactic choices, including premature, stale, and wrong-return uses.

    Does not enumerate feasible subsets, shortest paths, or hidden-ready masks.
    Only a direct visible edge is offered as the cheap delivery candidate; an
    arbitrary directly supplied route remains accepted by step().
    """
    o = observation
    if o.done:
        return []
    actions = [Action("verify"),Action("think"),Action("abstain"),Action("commit_pending")]
    actions += [Action("inspect",{"target":r["handle"]}) for r in o.item_inventory]
    actions += [Action("choose_item",{"item":r["handle"]}) for r in o.item_inventory]
    actions.append(Action("inspect",{"target":"map"}))
    name = f"problem_{len(o.problems)}"
    if o.known_items:
        actions.append(Action("start_subset",{"handle":name}))
    if o.known_edges is not None:
        actions.append(Action("build_route",{"handle":name,"start":o.position,"goal":o.goal["destination"]}))
        if any(u == o.position and v == o.goal["destination"] for u,v,_ in o.known_edges):
            actions.append(Action("deliver",{"path":[o.position,o.goal["destination"]]}))
    for handle,p in o.problems.items():
        if p["primitive"] == "constrained_subset":
            actions += [Action("add_constraint",{"problem":handle,"constraint":c}) for c in CONSTRAINTS]
        actions += [Action("call",{"problem":handle,"budget":b}) for b in budgets if b <= o.remaining_work]
    for r in o.records:
        actions.append(Action("retrieve",{"handle":r["handle"]}))
        if r["kind"] == "computation":
            actions += [Action("use_return",{"handle":r["handle"],"as":t}) for t in ("subset","route")]
    return actions


def encode_observation(o: Observation) -> list[float]:
    """Small public summary; per-object facts remain in candidate vectors.

    Full JSON remains available equally to all policies. This summary is not
    claimed to be a lossless semantic encoder or sufficient statistic.
    """
    records = o.records
    return [len(o.item_inventory)/32,len(o.known_items)/32,len(o.goal["categories"])/8,
            o.goal["capacity"]/64,o.goal["funds"]/64,float(o.known_edges is not None),
            len(o.problems)/16,len(records)/32,len(o.pending)/8,len(o.selected)/8,
            o.position/32,o.goal["destination"]/32,float(o.delivered),float(o.verified),
            o.remaining_steps/64,o.remaining_work/4096,o.state_version/8,
            o.prices["observation"],o.prices["action"],o.prices["work"],
            float(o.feedback.get("status")=="success"),float(o.feedback.get("status")=="rejected"),
            float(o.feedback.get("status")=="obstacle"),float(o.feedback.get("status")=="timeout")]


def encode_action(o: Observation, action: Action) -> list[float]:
    """No handle spelling enters neural features; equality/provenance may enter."""
    a = action.arguments
    out = [float(action.kind == k) for k in ACTION_KINDS]
    h = a.get("item",a.get("target"))
    row = next((r for r in o.item_inventory if r["handle"]==h),{})
    known = o.known_items.get(h,{})
    p = o.problems.get(a.get("problem"),{})
    record = next((r for r in o.records if r["handle"]==a.get("handle")),{})
    if record:
        p = o.problems.get(record.get("problem"),{})
    constraints = p.get("problem",{}).get("constraints",[])
    out += [row.get("category",-1)/8,float(bool(known)),known.get("weight",0)/16,
            known.get("price",0)/16,float(h in o.pending),float(h in o.selected),
            a.get("budget",0)/4096,float(h=="map"),
            float(p.get("primitive")=="constrained_subset"),float(p.get("primitive")=="shortest_path")]
    out += [float(c in constraints) for c in CONSTRAINTS]
    out += [float(a.get("constraint")==c) for c in CONSTRAINTS]
    out += [float(record.get("status")==s) for s in ("success","timeout","infeasible","invalid_input","unknown")]
    out += [float(bool(record) and record.get("state_version") != o.state_version),
            float(a.get("as")=="subset"),float(a.get("as")=="route"),
            float(record.get("primitive")=="constrained_subset"),float(record.get("primitive")=="shortest_path"),
            len(p.get("problem",{}).get("items",[]))/32,
            float(a.get("handle") in o.retrieved)]
    return out


def protocol_executor(primitive: str, problem: dict[str, Any], max_work: int) -> dict[str, Any]:
    """Supplied deterministic lowering of explicit public drafts into protocol.

    Missing constraints remain unconstrained; missing items remain missing.
    This compiler does not inspect the world, complete constraints, or solve.
    """
    from .campaign02_protocol import Budget, Call, execute
    if primitive == "constrained_subset":
        rows = tuple(tuple(row) for row in problem["items"])
        args = (rows,problem.get("capacity",sum(row[1] for row in rows)),
                problem.get("max_cost",sum(row[2] for row in rows)),
                tuple(tuple(pair) for pair in problem.get("forbidden_pairs",[])))
    elif primitive == "shortest_path":
        args = (problem["n"],tuple(tuple(edge) for edge in problem["edges"]),problem["start"],problem["goal"])
    else:
        args = tuple(problem.get("arguments",()))
    result = execute(Call(primitive,args,budget=Budget(max_work)))
    return {"status":result.status,"payload":result.payload,"work_units":result.work_units,
            "cpu_seconds":result.cpu_seconds,"certificate":result.certificate}
