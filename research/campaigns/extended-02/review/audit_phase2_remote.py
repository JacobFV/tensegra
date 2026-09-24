"""Independent phase-2 audit: read-only extraction on the results host (stdlib only).

Reads sealed E09v2 outputs, population state/protocol files, checkpoints (hash only) and job
receipts under ~/topoformer-campaign02/results and writes one JSON document to argv[1]
(default /tmp/audit-phase2-extract.json). Never writes inside results/.
Written independently of research/tools/campaign02_e09_analysis.py and friends.
"""
import glob
import gzip
import hashlib
import json
import os
import resource
import sys
import time

RESULTS = os.path.expanduser("~/topoformer-campaign02/results")
ROOTS = ["e09v2-sealed-r0", "e09v2-sealed-r1", "e09v2-sealed-r2", "e09v2-sealed-references",
         "e09v2-sealed-e11", "e09v2-sealed-e12"]
V1_ROOTS = ["e09-sealed-r0", "e09-sealed-r1", "e09-sealed-r2", "e09-sealed-references"]
RUNS = (["e07-legacy-v2", "e07-recurrent-v2"]
        + [f"e08-bank-r{r}" for r in range(3)]
        + [f"e08-main-{m}-r{r}" for m in ("pbt", "multistart", "single") for r in range(3)]
        + ["e11-arch-lightweight", "e11-arch-recurrent"]
        + [f"e12-curriculum-pbt-r{r}" for r in range(3)]
        + [f"e13-robust-fitness-pbt-r{r}" for r in range(3)])
GREEDY = {"choose_item", "commit_pending"}
TOOL = {"call", "start_subset"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def episode(row):
    o = row["outcome"]
    kinds = [h["action"]["kind"] for h in o.get("history", [])]
    first_greedy = next((i for i, k in enumerate(kinds) if k in GREEDY), None)
    first_tool = next((i for i, k in enumerate(kinds) if k in TOOL), None)
    greedy_first = first_greedy is not None and (first_tool is None or first_greedy < first_tool)
    return {
        "seed": row["seed"], "spec_hash": row["spec_hash"], "sem": row.get("semantic_spec_hash"),
        "u": o["utility"], "s": bool(o["verified_success"]), "c": o["cost"],
        "steps": o.get("steps"), "hist": len(kinds), "trunc": row.get("truncated"),
        "gf": greedy_first, "tool": first_tool is not None,
        "abstain": "abstain" in kinds, "deliver": "deliver" in kinds,
        "last": kinds[-1] if kinds else None,
    }


def extract_eval(root):
    base = os.path.join(RESULTS, root)
    out = {"conditions": {}, "summary": None}
    summ_path = os.path.join(base, "summary.json")
    if os.path.exists(summ_path):
        s = json.load(open(summ_path))
        out["summary"] = {
            "config": s.get("config"), "config_sha256": s.get("config_sha256"), "sources": s.get("sources"),
            "checkpoints": s.get("checkpoints"),
            "arm_meta": [{k: r.get(k) for k in ("condition", "arm", "kind", "examples", "success", "artifact",
                                                   "artifact_sha256", "policy_config", "training_config",
                                                   "checkpoint_updates", "checkpoint_presentations")}
                         | {"mean_utility": r.get("means", {}).get("utility")}
                         for r in s.get("results", [])],
        }
    for cdir in sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)):
        cond = os.path.basename(cdir)
        cinfo = {"worlds": None, "arms": {}}
        wpath = os.path.join(cdir, "worlds.jsonl.gz")
        if os.path.exists(wpath):
            w = [(r["seed"], r["spec_hash"], r["semantic_spec_hash"], canonical(r["spec"]) == r["spec_hash"],
                  r.get("spec", {}).get("step_limit")) for r in rows(wpath)]
            cinfo["worlds"] = {"n": len(w), "seeds": [x[0] for x in w], "spec_hash": [x[1] for x in w],
                               "sem": [x[2] for x in w], "spec_hash_recomputed_ok": sum(x[3] for x in w),
                               "step_limits": sorted(set(x[4] for x in w))}
        for apath in sorted(glob.glob(os.path.join(cdir, "*.jsonl.gz"))):
            arm = os.path.basename(apath)[:-len(".jsonl.gz")]
            if arm == "worlds":
                continue
            eps = [episode(r) for r in rows(apath)]
            cinfo["arms"][arm] = {"file_sha256": sha256_file(apath), "episodes": eps}
        out["conditions"][cond] = cinfo
    return out


def slim_state(state):
    def strip(a):
        return {k: v for k, v in a.items() if k not in ("training_timing", "scores")}
    return {
        "allocations": [strip(a) for a in state.get("allocations", [])],
        "lineage": [strip(l) for l in state.get("lineage", [])],
        "members": [strip(m) for m in state.get("members", [])],
        "finalist": strip(state["finalist"]) if state.get("finalist") else None,
        "selection_rule": state.get("selection_rule"), "status": state.get("status"),
        "round": state.get("round"), "source_hashes": state.get("source_hashes"),
        "feature_dimensions": state.get("feature_dimensions"), "protocol_hash": state.get("protocol_hash"),
    }


def main():
    t0 = time.time()
    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit-phase2-extract.json"
    doc = {"eval": {}, "v1": {}, "runs": {}, "checkpoint_hashes": {}, "receipts": {}}
    for root in ROOTS:
        doc["eval"][root] = extract_eval(root)
    for root in V1_ROOTS:  # only config/binding identity and which files exist; outcomes not read
        base = os.path.join(RESULTS, root)
        doc["v1"][root] = {"exists": os.path.isdir(base),
                           "conditions": sorted(os.path.basename(d) for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)),
                           "summary_exists": os.path.exists(os.path.join(base, "summary.json"))}
    paths = set()
    for root in ROOTS:
        s = doc["eval"][root]["summary"]
        for c in (s or {}).get("checkpoints", []) or []:
            paths.add(c["path"])
    for run in RUNS:
        base = os.path.join(RESULTS, run)
        entry = {}
        for name in ("state.json", "protocol.json", "population_lineage.json"):
            p = os.path.join(base, name)
            if os.path.exists(p):
                entry[name + ".sha256"] = sha256_file(p)
        if os.path.exists(os.path.join(base, "state.json")):
            entry["state"] = slim_state(json.load(open(os.path.join(base, "state.json"))))
            fin = entry["state"]["finalist"]
            if fin:
                paths.add(os.path.join(base, fin["checkpoint"]))
        if os.path.exists(os.path.join(base, "protocol.json")):
            entry["protocol"] = json.load(open(os.path.join(base, "protocol.json")))
        doc["runs"][run] = entry
    for p in sorted(paths):
        doc["checkpoint_hashes"][p] = sha256_file(p) if os.path.exists(p) else None
    for pdir in sorted(glob.glob(os.path.join(RESULTS, "*-process"))):
        rec = {}
        for name in ("launch.json", "occupancy.json"):
            p = os.path.join(pdir, name)
            if os.path.exists(p):
                try:
                    rec[name] = json.load(open(p))
                except ValueError as e:
                    rec[name] = {"_error": str(e)}
        doc["receipts"][os.path.basename(pdir)[:-len("-process")]] = rec
    ru = resource.getrusage(resource.RUSAGE_SELF)
    doc["audit_cost"] = {"user_s": ru.ru_utime, "sys_s": ru.ru_stime, "wall_s": time.time() - t0,
                         "peak_rss_kib": ru.ru_maxrss}
    with open(target, "w") as f:
        json.dump(doc, f)
    print(json.dumps(doc["audit_cost"]))


if __name__ == "__main__":
    main()
