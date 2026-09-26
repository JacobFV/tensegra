#!/usr/bin/env python3
"""P2a auditor: development curves, replay (trace) identity, anchor logging and the
registered deployment rule, recomputed from raw development rows.

Standard library only. Written by the independent P2a auditor.

Usage:
    python campaign03_p2a_audit_runs.py <results dir holding p2a-c0-x1-r2, p2a-c1-x1-r2,
        p1-rl-x1-r2, p1-boot-x1-r2> <repo configs/campaign03 dir> --output runs.json
"""
import argparse
import gzip
import hashlib
import json
import os

RUNS = ["p2a-c0-x1-r2", "p2a-c1-x1-r2", "p1-rl-x1-r2", "p1-boot-x1-r2"]
TIMING_KEYS = {"timing", "neural_forward_wall_seconds_allocated", "solver_cpu_seconds", "batch_process_cpu_seconds",
               "batch_wall_seconds"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TIMING_KEYS and "seconds" not in k}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dev_file(path):
    succ = util = n = 0
    seeds = []
    h = hashlib.sha256()
    kinds = {}
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        o = r["outcome"]
        n += 1
        succ += bool(o["verified_success"])
        util += o["utility"]
        seeds.append(r["seed"])
        h.update(json.dumps(strip(r), sort_keys=True).encode())
        for x in o["history"]:
            kinds[x["action"]["kind"]] = kinds.get(x["action"]["kind"], 0) + 1
    tot = sum(kinds.values())
    return {"n": n, "success": succ / n, "utility": util / n, "seeds": seeds,
            "stripped_hash": h.hexdigest(), "action_mix": {k: v / tot for k, v in sorted(kinds.items())},
            "decisions_per_episode": tot / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("configs")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = {"runs": {}}
    for run in RUNS:
        st = json.load(open(os.path.join(a.results, run, "state.json")))
        rows = []
        for i, al in enumerate(st["allocations"]):
            dp = os.path.join(a.results, run, al["development_predictions"])
            d = dev_file(dp)
            rows.append({
                "attempt": i, "label": os.path.basename(al["checkpoint"])[:-3], "updates": al["cumulative_slot_updates"],
                "checkpoint_sha256": al["checkpoint_sha256"], "dev_file_sha_ok": sha_file(dp) == al["development_sha256"],
                "state_success": al["success"], "state_utility": al.get("utility"),
                "raw_success": d["success"], "raw_utility": d["utility"], "n": d["n"],
                "seeds_min": min(d["seeds"]), "seeds_max": max(d["seeds"]), "seeds": d["seeds"],
                "stripped_hash": d["stripped_hash"], "action_mix": d["action_mix"],
                "decisions_per_episode": d["decisions_per_episode"],
                "objective_parts": al.get("training_timing", {}).get("last", {}).get("objective_parts"),
                "anchor_logged": al.get("training_timing", {}).get("anchor"),
                "training_seed_interval": al.get("training_seed_interval"),
            })
        out["runs"][run] = {"allocations": rows, "anchor": st.get("anchor"),
                            "protocol_sha256": sha_file(os.path.join(a.results, run, "protocol.json")),
                            "protocol_hash_in_state": st.get("protocol_hash"),
                            "lineage0": st["lineage"][0]}
    c0, p1, c1 = (out["runs"][r]["allocations"] for r in ["p2a-c0-x1-r2", "p1-rl-x1-r2", "p2a-c1-x1-r2"])
    out["replay"] = {
        "checkpoint_sha_equal": [x["checkpoint_sha256"] == y["checkpoint_sha256"] for x, y in zip(c0, p1)],
        "dev_success_equal": [x["raw_success"] == y["raw_success"] for x, y in zip(c0, p1)],
        "dev_utility_equal": [abs(x["raw_utility"] - y["raw_utility"]) < 1e-12 for x, y in zip(c0, p1)],
        "dev_rows_equal_timing_stripped": [x["stripped_hash"] == y["stripped_hash"] for x, y in zip(c0, p1)],
        "training_seed_intervals_equal": [x["training_seed_interval"] == y["training_seed_interval"] for x, y in zip(c0, p1)],
        "c0_success": [x["raw_success"] for x in c0], "p1_success": [x["raw_success"] for x in p1],
        "c1_success": [x["raw_success"] for x in c1],
        "c0_utility": [x["raw_utility"] for x in c0], "c1_utility": [x["raw_utility"] for x in c1],
    }
    # configs
    cfg = {n: open(os.path.join(a.configs, n), "rb").read() for n in
           ["p1-rl-x1-r2.json", "p2a-c0-x1-r2.json", "p2a-c1-x1-r2.json"]}
    j0, j1 = json.loads(cfg["p2a-c0-x1-r2.json"]), json.loads(cfg["p2a-c1-x1-r2.json"])

    def diff(x, y, p=""):
        res = []
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    res.append(f"{p}{k}: {'added' if k not in x else 'removed'} -> {json.dumps(y.get(k))[:200]}")
                else:
                    res += diff(x[k], y[k], f"{p}{k}.")
        elif x != y:
            res.append(f"{p[:-1]}: {x} -> {y}")
        return res
    banks = json.load(open(os.path.join(a.configs, "p1-banks.json")))
    out["configs"] = {"c0_byte_identical_p1": cfg["p1-rl-x1-r2.json"] == cfg["p2a-c0-x1-r2.json"],
                      "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in cfg.items()},
                      "c1_vs_c0_diff": diff(j0, j1), "banks_mentions_74ff10a6": "74ff10a686c04b4face63ac2e666add7fd5a3c17509ec78c81f8aa954b5f2fc9" in json.dumps(banks)}
    # bootstrap development and deployment rule
    boot = out["runs"]["p1-boot-x1-r2"]["allocations"]
    bsel = [b for b in boot if b["checkpoint_sha256"] == "74ff10a686c04b4face63ac2e666add7fd5a3c17509ec78c81f8aa954b5f2fc9"]
    b = bsel[-1]
    dep = {"boot_attempt": b["attempt"], "boot_success": b["raw_success"], "boot_utility": b["raw_utility"],
           "boot_seeds_equal_run_seeds": all(b["seeds"] == x["seeds"] for x in c0 + c1)}
    for name, rows in [("c0", c0), ("c1", c1)]:
        q = [x for x in rows if x["raw_success"] >= b["raw_success"] - 0.02 and x["raw_utility"] >= b["raw_utility"] - 0.02]
        # float-safe re-check with a tiny epsilon to flag borderline cases
        border = [x["attempt"] for x in rows if abs(x["raw_success"] - (b["raw_success"] - 0.02)) < 1e-9
                  or abs(x["raw_utility"] - (b["raw_utility"] - 0.02)) < 1e-6]
        sel = q[-1] if q else None
        registered = json.load(open(os.path.join(a.configs, f"deployment-{name}.json")))
        dep[name] = {"qualifying": [x["attempt"] for x in q], "n_qualifying": len(q),
                     "deployed_attempt": sel["attempt"] if sel else "rollback",
                     "deployed_sha256": sel["checkpoint_sha256"] if sel else None,
                     "registered_deployed_label": registered["deployed"]["label"],
                     "registered_deployed_sha256": registered["deployed"]["sha256"],
                     "agrees": bool(sel) and registered["deployed"]["sha256"] == sel["checkpoint_sha256"],
                     "registered_boot": registered["bootstrap_development"],
                     "borderline_attempts": border,
                     "margin_to_threshold": {x["attempt"]: [x["raw_success"] - (b["raw_success"] - 0.02),
                                                            x["raw_utility"] - (b["raw_utility"] - 0.02)] for x in rows}}
    out["deployment"] = dep
    for r in out["runs"].values():
        for x in r["allocations"]:
            del x["seeds"]
    json.dump(out, open(a.output, "w"), indent=1, sort_keys=True)
    print(json.dumps({"replay": {k: (all(v) if isinstance(v, list) and v and isinstance(v[0], bool) else v)
                                 for k, v in out["replay"].items()}, "configs": out["configs"],
                      "deployment": {k: ({kk: vv for kk, vv in v.items() if kk != "margin_to_threshold"} if isinstance(v, dict) else v)
                                     for k, v in dep.items()}}, indent=1))


if __name__ == "__main__":
    main()
