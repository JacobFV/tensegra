"""Versioned, pure, bounded computation over explicitly supplied public instances.

Work units are deterministic search/scan expansions, not CPU-time equivalences.
``execute_isolated`` is the hostile-input boundary; ``execute`` is for trusted,
size-capped simulator calls. No function evaluates generated host-language code.
"""
from dataclasses import dataclass, replace
from itertools import product
import heapq
import math
import multiprocessing as mp
import time


@dataclass(frozen=True)
class Budget:
    work_units: int = 10000
    wall_seconds: float = 2.0


@dataclass(frozen=True)
class Call:
    primitive: str
    arguments: tuple
    source_version: int = 0
    caller: str = "actor"
    budget: Budget = Budget()
    api_version: str = "1"


@dataclass(frozen=True)
class Result:
    status: str
    payload: object
    source_version: int
    caller: str
    primitive: str
    work_units: int
    cpu_seconds: float
    certificate: tuple = ()
    api_version: str = "1"


@dataclass(frozen=True)
class Descriptor:
    identity: str
    inputs: tuple
    output: str
    semantics: str
    guarantee: str
    version: str = "1"
    state_dependencies: str = "Only ordered supplied arguments; source_version is provenance"
    resource_controls: tuple = ("work_units", "wall_seconds")


DESCRIPTORS = {
    name: Descriptor(name, inputs, output, semantics, guarantee)
    for name, inputs, output, semantics, guarantee in (
        ("lookup", ("tuple", "index"), "value", "Read indexed immutable record", "exact"),
        ("filter", ("int tuple", "int"), "int tuple", "Keep values >= threshold", "exact"),
        ("intersection", ("int tuple", "int tuple"), "int tuple", "Sorted unique intersection", "exact"),
        ("add", ("int", "int"), "int", "Ordered integer addition", "exact"),
        ("sub", ("int", "int"), "int", "Ordered integer subtraction", "exact"),
        ("mul", ("int", "int"), "int", "Ordered integer multiplication", "exact"),
        ("compare", ("int", "int"), "int", "Sign of first minus second", "exact"),
        ("shortest_path", ("node count", "directed weighted edges", "start", "goal"), "(path,distance)", "Nonnegative integer shortest path", "optimal on success"),
        ("csp", ("domains", "forbidden ordered value pairs"), "assignment", "Finite domains with binary incompatibilities", "feasible on success; exhaustive infeasibility"),
        ("constrained_subset", ("items(category,weight,cost,value)", "capacity", "max_cost", "conflicts"), "(indices,value,weight,cost)", "Exactly one per supplied category; maximize value", "optimal on success"),
        ("subset", ("weights", "values", "capacity"), "(indices,value,weight)", "Maximize value subject to weight <= capacity", "optimal only on success; timeout incumbent feasible"),
    )
}
STATUSES = ("success", "infeasible", "invalid", "unavailable", "timeout", "unknown")


def _integer(x, low=None, high=None):
    if type(x) is not int or x.bit_length() > 64 or (low is not None and x < low) or (high is not None and x > high):
        raise ValueError("integer/domain violation")
    return x


def _tuple(x, limit=4096):
    if type(x) is not tuple or len(x) > limit:
        raise ValueError("immutable tuple/size required")
    return x


def _safe(x, depth=0, remaining=None):
    if remaining is None: remaining = [20000]
    remaining[0] -= 1
    if remaining[0] < 0: raise ValueError("total input size limit")
    if depth > 8:
        raise ValueError("nesting limit")
    if type(x) is int:
        _integer(x)
    elif type(x) is str:
        if len(x) > 256:
            raise ValueError("string limit")
    elif type(x) is tuple:
        for v in _tuple(x):
            _safe(v, depth + 1, remaining)
    else:
        raise ValueError("unsupported public value")


def _validate(call):
    if type(call) is not Call or type(call.budget) is not Budget:
        raise ValueError("call/budget type")
    _integer(call.source_version, 0)
    _integer(call.budget.work_units, 0, 1000000)
    if type(call.budget.wall_seconds) not in (int, float) or not math.isfinite(call.budget.wall_seconds) or not 0 < call.budget.wall_seconds <= 30:
        raise ValueError("wall budget")
    if type(call.caller) is not str or len(call.caller) > 128:
        raise ValueError("caller")
    if type(call.primitive) is not str or len(call.primitive) > 64: raise ValueError("primitive identity")
    _safe(call.arguments)
    a = _tuple(call.arguments, 4)
    p = call.primitive
    if p not in DESCRIPTORS or call.api_version != "1":
        return
    if len(a) != len(DESCRIPTORS[p].inputs):
        raise ValueError("arity")
    if p in ("add", "sub", "mul", "compare"):
        for x in a: _integer(x)
    elif p == "lookup":
        _tuple(a[0]); _integer(a[1], 0, len(a[0]) - 1)
    elif p in ("filter", "intersection"):
        for x in _tuple(a[0]): _integer(x)
        if p == "filter": _integer(a[1])
        else:
            for x in _tuple(a[1]): _integer(x)
    elif p == "shortest_path":
        n, edges, start, goal = a
        _integer(n, 1, 256); _integer(start, 0, n - 1); _integer(goal, 0, n - 1)
        for edge in _tuple(edges):
            if len(_tuple(edge, 3)) != 3: raise ValueError("edge arity")
            u, v, w = edge
            _integer(u, 0, n - 1); _integer(v, 0, n - 1); _integer(w, 0)
    elif p == "csp":
        domains, forbidden = a
        _tuple(domains, 16)
        for d in domains:
            _tuple(d, 16)
            if len(set(d)) != len(d): raise ValueError("duplicate domain value")
            for x in d: _integer(x)
        for row in _tuple(forbidden):
            if len(_tuple(row, 4)) != 4: raise ValueError("constraint arity")
            i, x, j, y = row
            _integer(i, 0, len(domains)-1); _integer(j, 0, len(domains)-1)
            if x not in domains[i] or y not in domains[j]: raise ValueError("constraint domain")
    elif p == "constrained_subset":
        items, cap, max_cost, conflicts = a
        _tuple(items, 20); _integer(cap, 0); _integer(max_cost, 0)
        for item in items:
            if len(_tuple(item, 4)) != 4: raise ValueError("item arity")
            for x in item[:3]: _integer(x, 0)
            _integer(item[3])
        for pair in _tuple(conflicts):
            if len(_tuple(pair, 2)) != 2: raise ValueError("pair arity")
            for i in pair: _integer(i, 0, len(items)-1)
    elif p == "subset":
        weights, values, cap = a
        _tuple(weights, 20); _tuple(values, 20); _integer(cap, 0)
        if len(weights) != len(values): raise ValueError("length mismatch")
        for x in weights: _integer(x, 0)
        for x in values: _integer(x)


class _Exhausted(Exception):
    pass


def execute(call):
    """Execute a trusted frozen Call. Certificate describes executor guarantees."""
    cpu, wall, used = time.process_time(), time.monotonic(), 0
    payload = None
    def result(status, certificate=()):
        return Result(status, payload, getattr(call, "source_version", 0), getattr(call, "caller", ""), getattr(call, "primitive", ""), used, time.process_time()-cpu, certificate)
    def tick():
        nonlocal used
        if used >= call.budget.work_units or time.monotonic()-wall >= call.budget.wall_seconds:
            raise _Exhausted
        used += 1
    try:
        _validate(call)
        if call.primitive not in DESCRIPTORS or call.api_version != "1": return result("unavailable")
        p, a = call.primitive, call.arguments
        if p in ("add", "sub", "mul", "compare", "lookup"):
            tick()
            x, y = a
            payload = {"add": lambda: x+y, "sub": lambda: x-y, "mul": lambda: x*y, "compare": lambda: (x>y)-(x<y), "lookup": lambda: x[y]}[p]()
        elif p in ("filter", "intersection"):
            selected = []
            other = set(a[1]) if p == "intersection" else None
            for x in a[0]:
                tick()
                if (x in other if other is not None else x >= a[1]): selected.append(x)
            payload = tuple(sorted(set(selected))) if other is not None else tuple(selected)
        elif p == "shortest_path":
            n, edges, start, goal = a
            graph = [[] for _ in range(n)]
            for u, v, w in edges:
                tick(); graph[u].append((v,w))
            heap, distance = [(0, start, (start,))], {start: 0}
            while heap:
                tick(); d, u, path = heapq.heappop(heap)
                if d != distance[u]: continue
                if u == goal:
                    payload = (path, d)
                    return result("success", (("optimal", True),))
                for v, w in graph[u]:
                    tick()
                    if v not in distance or d+w < distance[v]:
                        distance[v] = d+w
                        heapq.heappush(heap, (d+w,v,path+(v,)))
            return result("infeasible", (("exhaustive", True),))
        elif p == "csp":
            domains, forbidden = a
            for candidate in product(*domains):
                tick(); valid = True
                for i,x,j,y in forbidden:
                    tick()
                    if candidate[i] == x and candidate[j] == y:
                        valid = False; break
                if valid:
                    payload = candidate
                    return result("success", (("feasible", True),))
            return result("infeasible", (("exhaustive", True),))
        elif p == "constrained_subset":
            items, cap, max_cost, conflicts = a
            categories = sorted(set(item[0] for item in items))
            groups = [tuple(i for i,item in enumerate(items) if item[0] == c) for c in categories]
            for ids in product(*groups):
                tick()
                weight = sum(items[i][1] for i in ids)
                cost = sum(items[i][2] for i in ids)
                value = sum(items[i][3] for i in ids)
                if weight <= cap and cost <= max_cost and all(not(i in ids and j in ids) for i,j in conflicts):
                    if payload is None or value > payload[1]: payload = (ids,value,weight,cost)
            return result("success" if payload is not None else "infeasible", (("optimal", payload is not None), ("exhaustive", True)))
        elif p == "subset":
            weights, values, cap = a
            payload = ((), 0, 0)
            for mask in range(1 << len(weights)):
                tick()
                ids = tuple(i for i in range(len(weights)) if mask & (1 << i))
                weight, value = sum(weights[i] for i in ids), sum(values[i] for i in ids)
                if weight <= cap and value > payload[1]: payload = (ids, value, weight)
            return result("success", (("optimal", True), ("exhaustive", True)))
        return result("success", (("exact", True),))
    except _Exhausted:
        return result("timeout", (("optimal", False),))
    except (ValueError, TypeError, IndexError, KeyError):
        payload = None
        return result("invalid")


def validate_result(call, result):
    """Independently check positive payload feasibility, never trust certificates.

Does NOT certify infeasibility/optimality or correctness of a world reduction.
"""
    try:
        _validate(call)
        if (result.primitive, result.source_version, result.caller) != (call.primitive, call.source_version, call.caller): return False
        if result.status not in ("success", "timeout") or result.payload is None: return False
        p, a, out = call.primitive, call.arguments, result.payload
        if p == "csp":
            domains, forbidden = a
            return type(out) is tuple and len(out) == len(domains) and all(type(x) is int and x in d for x,d in zip(out,domains)) and all(not(out[i] == x and out[j] == y) for i,x,j,y in forbidden)
        if p == "constrained_subset":
            ids,value,weight,cost = out
            items,cap,max_cost,conflicts = a
            if type(ids) is not tuple or any(type(i) is not int or not 0 <= i < len(items) for i in ids): return False
            categories = [items[i][0] for i in ids]
            return len(set(ids)) == len(ids) and len(categories) == len(set(categories)) and set(categories) == {item[0] for item in items} and all(type(x) is int for x in (value,weight,cost)) and value == sum(items[i][3] for i in ids) and weight == sum(items[i][1] for i in ids) <= cap and cost == sum(items[i][2] for i in ids) <= max_cost and all(not(i in ids and j in ids) for i,j in conflicts)
        if p == "subset":
            ids, value, weight = out
            w, v, cap = a
            return type(ids) is tuple and len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < len(w) for i in ids) and type(value) is int and type(weight) is int and value == sum(v[i] for i in ids) and weight == sum(w[i] for i in ids) <= cap
        if p == "shortest_path":
            path, cost = out
            n, edges, start, goal = a
            if type(path) is not tuple or not path or path[0] != start or path[-1] != goal or type(cost) is not int or any(type(i) is not int or not 0 <= i < n for i in path): return False
            minimum = {}
            for u,v,w in edges: minimum[u,v] = min(minimum.get((u,v),w),w)
            return cost == sum(minimum[u,v] for u,v in zip(path,path[1:]))
        if result.status != "success": return False
        x,y = a
        expected = {"add":lambda:x+y,"sub":lambda:x-y,"mul":lambda:x*y,"compare":lambda:(x>y)-(x<y),"lookup":lambda:x[y],"filter":lambda:tuple(v for v in x if v>=y),"intersection":lambda:tuple(sorted(set(x)&set(y)))}[p]()
        return type(out) is type(expected) and out == expected
    except (ValueError, TypeError, KeyError, IndexError):
        return False


def _child(conn, call):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(call.budget.wall_seconds)), max(1, math.ceil(call.budget.wall_seconds))+1))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    try:
        result = execute(call)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        conn.send(replace(result, cpu_seconds=usage.ru_utime + usage.ru_stime))
    finally: conn.close()


def execute_isolated(call):
    """Spawn fixed executor code with a wall deadline and OS CPU limit.

This is not an arbitrary-code sandbox. Actor supplies frozen data only. Parent
must charge process launch/validation overhead separately from solver CPU.
"""
    try: _validate(call)
    except (ValueError, TypeError, IndexError, KeyError): return execute(call)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(False)
    process = ctx.Process(target=_child, args=(child, call))
    process.start(); child.close()
    try:
        if parent.poll(call.budget.wall_seconds + 0.25):
            try: return parent.recv()
            except EOFError: pass
        return Result("timeout", None, call.source_version, call.caller, call.primitive, call.budget.work_units, 0.0, (("process_killed_or_no_result", True), ("cpu_seconds_unknown", True), ("work_units_are_upper_bound", True)))
    finally:
        if process.is_alive(): process.terminate()
        process.join(); parent.close()
