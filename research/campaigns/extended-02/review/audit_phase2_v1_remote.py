"""Read-only check that the failed E09 v1 partial outputs agree with the E09v2 re-run.

For every (condition, arm) file present in both e09-sealed-* (v1, failed at condition 16/26) and
e09v2-sealed-*, compare per-seed (spec_hash, utility, verified_success, action sequence).
Identical outcomes mean the v2 re-run could not have been a selective re-draw. Prints JSON.
"""
import glob
import gzip
import hashlib
import json
import os
import resource
import time

RESULTS = os.path.expanduser("~/topoformer-campaign02/results")


def digest(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "outcome" not in r:  # worlds.jsonl.gz
                out[r["seed"]] = (r["spec_hash"], r.get("address_seed"))
                continue
            o = r["outcome"]
            acts = json.dumps([h["action"] for h in o.get("history", [])], sort_keys=True)
            out[r["seed"]] = (r["spec_hash"], o["utility"], o["verified_success"], hashlib.sha256(acts.encode()).hexdigest())
    return out


def main():
    t0 = time.time()
    report = {"pairs": 0, "identical_pairs": 0, "differing": [], "v1_only": []}
    for suffix in ("r0", "r1", "r2", "references"):
        v1, v2 = os.path.join(RESULTS, f"e09-sealed-{suffix}"), os.path.join(RESULTS, f"e09v2-sealed-{suffix}")
        for p1 in sorted(glob.glob(os.path.join(v1, "*", "*.jsonl.gz"))):
            rel = os.path.relpath(p1, v1)
            p2 = os.path.join(v2, rel)
            if not os.path.exists(p2):
                report["v1_only"].append(f"{suffix}/{rel}")
                continue
            a, b = digest(p1), digest(p2)
            report["pairs"] += 1
            if a == b:
                report["identical_pairs"] += 1
            else:
                report["differing"].append({"file": f"{suffix}/{rel}", "seeds_v1": len(a), "seeds_v2": len(b),
                                            "outcome_diffs": sum(1 for s in a if a[s] != b.get(s))})
    ru = resource.getrusage(resource.RUSAGE_SELF)
    report["audit_cost"] = {"user_s": ru.ru_utime, "sys_s": ru.ru_stime, "wall_s": time.time() - t0}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
