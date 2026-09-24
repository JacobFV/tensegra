"""CPU-only frozen-source public-feature collision proof; no model inference.

Run from the repository with --output PATH. --verify reconstructs encodings and
teacher labels entirely from archived public observations, not hidden worlds.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def independently_feasible(obs, handles):
    rows = [obs["known_items"][h] for h in handles]
    return (len(handles) == len(set(handles))
        and sorted(r["category"] for r in rows) == sorted(obs["goal"]["categories"])
        and sum(r["weight"] for r in rows) <= obs["goal"]["capacity"]
        and sum(r["price"] for r in rows) <= obs["goal"]["funds"]
        and not any(a in handles and b in handles for a, b in obs["incompatible"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="a9337b71")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    source = subprocess.check_output(["git", "rev-parse", args.source], text=True).strip()
    files = {}
    with tempfile.TemporaryDirectory(prefix="e02-collision-") as tmp:
        pkg = Path(tmp) / "topoformer"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        for name in ("campaign02_world.py", "campaign02_references.py"):
            data = subprocess.check_output(["git", "show", f"{source}:src/topoformer/{name}"])
            (pkg / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
        sys.path.insert(0, tmp)
        w = importlib.import_module("topoformer.campaign02_world")
        r = importlib.import_module("topoformer.campaign02_references")
        teacher = r.make_reference("cheap_first_fallback_v2")
        def encode(o):
            actions = w.action_catalog(o)
            return {"observation": w.encode_observation(o),
                    "candidates": [w.encode_action(o, a) for a in actions]}
        def public(o):
            return {"observation": asdict(o), "actions": [asdict(a) for a in w.action_catalog(o)],
                    "encoded": encode(o), "feature_hash": digest(encode(o)),
                    "teacher": asdict(teacher.choose(o))}
        if args.verify:
            raw = json.loads(args.verify.read_text())
            assert raw["source"] == source and raw["source_hashes"] == files
            for name, pair in raw["pairs"].items():
                for side in ("left", "right"):
                    record = pair[side]
                    obs = w.Observation(**record["observation"])
                    assert digest(encode(obs)) == record["feature_hash"]
                    assert digest(encode(obs)) == digest(record["encoded"])
                    assert asdict(teacher.choose(obs)) == record["teacher"]
                    assert digest([asdict(a) for a in w.action_catalog(obs)]) == digest(record["actions"])
                assert pair["left"]["feature_hash"] == pair["right"]["feature_hash"]
                assert pair["left"]["teacher"]["kind"] != pair["right"]["teacher"]["kind"]
            conflict = raw["pairs"]["nonpending_incompatibility"]
            assert independently_feasible(conflict["left"]["observation"], ["a", "c"])
            assert independently_feasible(conflict["right"]["observation"], ["b", "c"])
            route = raw["pairs"]["downstream_route_weight"]
            # Direct independently written public-policy cost calculation.
            left, right = route["left"]["observation"], route["right"]["observation"]
            for obs, cheap in ((left, True), (right, False)):
                edges = {(u, v): cost for u, v, cost in obs["known_edges"]}
                greedy_total = edges[(0, 1)] + edges[(1, 2)]
                savings = (greedy_total - min(edges[(0, 1)], edges[(0, 2)])) * obs["prices"]["travel"]
                overhead = 4 * obs["prices"]["action"] + len(edges) * obs["prices"]["work"]
                assert (savings <= overhead) == cheap
            print(json.dumps({"verified": True, "pairs": 2, "source": source,
                "artifact_sha256": hashlib.sha256(args.verify.read_bytes()).hexdigest(),
                "wall_seconds": time.perf_counter()-started, "process_cpu_seconds": time.process_time()-cpu}))
            return
        if args.output is None:
            parser.error("--output or --verify required")
        def ready(spec):
            env = w.Workshop(spec)
            for item in spec.items:
                env.step(w.Action("inspect", {"target": item.handle}))
            return env
        base = w.WorldSpec((w.Item("a",0,1,1),w.Item("b",0,2,2),
            w.Item("c",1,1,1),w.Item("d",1,7,7)), (0,1),4,4,(),((0,1,1),(1,2,1)),0,2)
        conflict = ready(base), ready(replace(base, incompatible=(("a","c"),)))
        route = w.WorldSpec((w.Item("a",0,1,1),),(0,),10,10,(),
            ((0,1,1),(1,2,1),(0,2,3)),0,2,travel_price=.002)
        routes = ready(route), ready(replace(route,edges=((0,1,1),(1,2,4),(0,2,3))))
        for env in routes:
            env.step(w.Action("choose_item",{"item":"a"}))
            env.step(w.Action("commit_pending"))
            env.step(w.Action("inspect",{"target":"map"}))
        pairs = {name: {"left": public(a.observe()), "right": public(b.observe())}
            for name,(a,b) in (("nonpending_incompatibility",conflict),("downstream_route_weight",routes))}
        for pair in pairs.values():
            assert pair["left"]["feature_hash"] == pair["right"]["feature_hash"]
            assert pair["left"]["teacher"]["kind"] != pair["right"]["teacher"]["kind"]
        initial = w.Workshop(base).observe()
        actions = w.action_catalog(initial)
        target = actions.index(teacher.choose(initial))
        features = [w.encode_action(initial,a) for a in actions]
        ties = [asdict(a) for a,f in zip(actions,features) if f == features[target]]
        raw = {"source": source, "source_hashes": files,
            "teacher": "cheap_first_fallback_v2", "dimensions": [27,67], "pairs": pairs,
            "initial_inspection_tie": {"target": asdict(actions[target]), "identical_candidates": ties},
            "limits": "Constructive public-encoder collisions, not measured population frequency or task-unsolvability proof. No neural inference/training.",
            "wall_seconds": time.perf_counter()-started, "process_cpu_seconds": time.process_time()-cpu}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(raw,indent=2,sort_keys=True))
        print(json.dumps({"output":str(args.output),"pairs":len(pairs),"process_cpu_seconds":raw["process_cpu_seconds"]}))


if __name__ == "__main__":
    main()
