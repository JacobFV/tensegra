"""Independent reconstruction of E09 sealed metrics from raw episode rows.

Reads every `<condition>/<arm>.jsonl.gz` produced by campaign02_evaluate.py and
recomputes success, utility, cost, solver use and paired differences from the
raw evaluator outcomes (never from the evaluator's own summary). Arms are named
`<mode>-r<replicate>` for finalists, `bank0-r<replicate>` for the no-RL
bootstrap member, and `reference-<name>` for supplied schedules.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random

IID = ["iid_3x3", "iid_4x4", "iid_4x4_tight128", "iid_4x4_expensive_work", "iid_3x3_obstacle", "iid_4x4_obstacle", "iid_4x5", "iid_4x4_steps36"]


def load(root: Path):
    data = defaultdict(dict)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in folder.glob("*.jsonl.gz"):
            if path.name == "worlds.jsonl.gz":
                continue
            rows = {}
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    o = row["outcome"]
                    calls = [h for h in o["history"] if h["action"]["kind"] == "call"]
                    rows[row["seed"]] = {
                        "success": bool(o["verified_success"]), "utility": float(o["utility"]), "cost": float(o["cost"]),
                        "steps": o["steps"], "work": o["work_units"], "calls": len(calls),
                        "timeouts": sum(h["feedback"].get("status") == "timeout" for h in calls),
                        "abstain": any(h["action"]["kind"] == "abstain" for h in o["history"]),
                        "budgets": [h["action"]["arguments"].get("budget") for h in calls],
                        "correct_reductions": sum(bool(r["correct"]) for r in o.get("reductions", [])),
                        "reductions": len(o.get("reductions", []))}
            data[folder.name][path.name.removesuffix(".jsonl.gz")] = rows
    return data


def describe(rows):
    n = len(rows)
    v = list(rows.values())
    s = sum(r["success"] for r in v)
    return {"n": n, "success": s/n, "utility": sum(r["utility"] for r in v)/n, "cost": sum(r["cost"] for r in v)/n,
            "cost_per_success": sum(r["cost"] for r in v)/s if s else None,
            "calls_per_episode": sum(r["calls"] for r in v)/n, "timeouts_per_episode": sum(r["timeouts"] for r in v)/n,
            "work_per_episode": sum(r["work"] for r in v)/n, "abstain_rate": sum(r["abstain"] for r in v)/n,
            "steps_per_episode": sum(r["steps"] for r in v)/n,
            "reduction_correct_rate": (sum(r["correct_reductions"] for r in v)/max(1, sum(r["reductions"] for r in v)))}


def paired(data, conditions, a, b, *, draws=2000, seed=0):
    """Mean utility difference a-b over (condition, world) pairs, equal condition weight."""
    per = {}
    for c in conditions:
        x, y = data[c][a], data[c][b]
        assert set(x) == set(y), (c, a, b)
        per[c] = [x[k]["utility"]-y[k]["utility"] for k in sorted(x)]
    def stat(samples):
        return sum(sum(v)/len(v) for v in samples.values())/len(samples)
    point = stat(per)
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        boots.append(stat({c: [v[rng.randrange(len(v))] for _ in v] for c, v in per.items()}))
    boots.sort()
    wins = {c: sum(d > 0 for d in v) for c, v in per.items()}
    losses = {c: sum(d < 0 for d in v) for c, v in per.items()}
    return {"a": a, "b": b, "mean_difference": point, "ci95": [boots[int(.025*draws)], boots[int(.975*draws)-1]],
            "per_condition": {c: sum(v)/len(v) for c, v in per.items()}, "worlds_a_better": wins, "worlds_b_better": losses,
            "scope": "bootstrap over sealed worlds within condition; replicate fixed; equal condition weight"}


def pooled(data, conditions, pairs, *, draws=2000, seed=0):
    """Pooled paired difference across replicates on the shared sealed worlds."""
    per = defaultdict(list)
    for c in conditions:
        keys = sorted(data[c][pairs[0][0]])
        for k in keys:
            per[c].append(sum(data[c][a][k]["utility"]-data[c][b][k]["utility"] for a, b in pairs)/len(pairs))
    def stat(samples):
        return sum(sum(v)/len(v) for v in samples.values())/len(samples)
    rng = random.Random(seed)
    boots = sorted(stat({c: [v[rng.randrange(len(v))] for _ in v] for c, v in per.items()}) for _ in range(draws))
    return {"pairs": pairs, "mean_difference": stat(per), "ci95": [boots[int(.025*draws)], boots[int(.975*draws)-1]],
            "scope": "world-bootstrap of replicate-averaged paired differences; does not model replicate variance"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = defaultdict(dict)
    for root in a.roots:
        for c, arms in load(root).items():
            for arm, rows in arms.items():
                if arm in data[c]:
                    assert data[c][arm] == rows, f"conflicting duplicate arm {c}/{arm}"
                data[c][arm] = rows
    conditions = sorted(data)
    arms = sorted({arm for c in conditions for arm in data[c]})
    table = {c: {arm: describe(data[c][arm]) for arm in sorted(data[c])} for c in conditions}
    iid = [c for c in IID if c in data]
    transfer = [c for c in conditions if c.startswith("xfer_")]
    summary = {"conditions": conditions, "arms": arms, "table": table}
    def mean_over(cs, arm):
        return sum(table[c][arm]["utility"] for c in cs)/len(cs) if all(arm in table[c] for c in cs) else None
    summary["iid_mean_utility"] = {arm: mean_over(iid, arm) for arm in arms}
    summary["iid_mean_success"] = {arm: (sum(table[c][arm]["success"] for c in iid)/len(iid)) if all(arm in table[c] for c in iid) else None for arm in arms}
    summary["transfer_mean_utility"] = {arm: mean_over(transfer, arm) for arm in arms} if transfer else {}
    reps = sorted({arm.split("-r")[1] for arm in arms if arm.startswith(("pbt-r", "multistart-r", "single-r"))})
    comparisons = []
    for scope, cs in (("iid", iid), ("transfer", transfer)):
        if not cs:
            continue
        for r in reps:
            for x, y in (("pbt", "multistart"), ("pbt", "single"), ("multistart", "single"),
                         ("pbt", "bank0"), ("multistart", "bank0"), ("single", "bank0")):
                a_, b_ = f"{x}-r{r}", f"{y}-r{r}"
                if all(a_ in data[c] and b_ in data[c] for c in cs):
                    comparisons.append({"scope": scope, "replicate": r, **paired(data, cs, a_, b_)})
            if all("reference-cheap_first_fallback_v2" in data[c] for c in cs):
                for x in ("pbt", "multistart", "single", "bank0"):
                    if all(f"{x}-r{r}" in data[c] for c in cs):
                        comparisons.append({"scope": scope, "replicate": r,
                                            **paired(data, cs, f"{x}-r{r}", "reference-cheap_first_fallback_v2")})
        for x, y in (("pbt", "multistart"), ("pbt", "single"), ("multistart", "single"), ("pbt", "bank0"),
                     ("multistart", "bank0"), ("single", "bank0")):
            pairs = [(f"{x}-r{r}", f"{y}-r{r}") for r in reps if all(f"{x}-r{r}" in data[c] and f"{y}-r{r}" in data[c] for c in cs)]
            if len(pairs) == len(reps) and pairs:
                comparisons.append({"scope": scope, "replicate": "pooled", **pooled(data, cs, pairs)})
    summary["comparisons"] = comparisons
    a.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
