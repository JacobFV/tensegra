#!/usr/bin/env python3
"""Independent raw-row reconstruction of the extended-03 P2a screening results.

Written by the independent P2a auditor from protocol-P2a.md and the registered
no-progress operationalization in decisions.md (2026-09-26T14:28Z), WITHOUT reading
research/tools/campaign03_p2a_analysis.py or campaign03_p2a_collapse_audit.py.
Standard library only. Correct-reuse windows reuse the P1 independent auditor's
function (campaign03_p1_audit_reconstruct.correct_reuse), not the root's code.

Usage:
    python research/tools/campaign03_p2a_audit_reconstruct.py <p2a-screening dir> \
        --output recon.json [--window 6] [--workers 4]

Decision state (registered): draft (problems incl. each draft's depends_on
requirement versions, pending choices, pending assignment), commitments
(selection, assignment, finish time, verified), retrieved set, position.
Breakers: a solver call, a new inspection (first inspect of that target in the
episode), or a step on which an event fired.
- idempotent repeat: accepted action whose action_key equals the previous accepted
  action's key and whose post-state equals its pre-state;
- short cycle: accepted action whose post-state equals a state among the states at
  the preceding `window` decision points (pre-state of this action included) with no
  breaker in between; post == pre is a length-1 cycle;
- no-progress: either (per decision); per-episode counts; any-episode fraction.
The post-state of the final action is not in the trace; if an earlier step had the
same pre-state and the same action key, its (deterministic) post-state is used,
a successful retrieve of an
already-retrieved record is a no-op; otherwise the final action is left unclassified and counted in `unknown_last`.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from campaign03_p1_audit_reconstruct import correct_reuse  # noqa: E402

CONDITIONS = ["iid_f0", "iid_f2", "events_train_kinds_p1", "foreign4"]
IID = ["iid_f0", "iid_f2"]
POLICY_OF_FILE = {
    "p2a-bootstrap-x1-r2": "bootstrap", "p2a-c0-final": "c0_final", "p2a-c0-deployed": "c0_deployed",
    "p2a-c1-final": "c1_final", "p2a-c1-deployed": "c1_deployed", "p1-rl-x1-r2": "p1_rl",
    "reference-dep_reuse": "dep_reuse",
}
WINDOW = 6


def canon(x):
    return hashlib.sha1(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decision_state(ob):
    return canon({
        "problems": ob.get("problems"),
        "pending": ob.get("pending"),
        "pending_assignment": ob.get("pending_assignment"),
        "selected": ob.get("selected"), "selection_id": ob.get("selection_id"),
        "assignment": ob.get("assignment"), "assignment_id": ob.get("assignment_id"),
        "finish_time": ob.get("finish_time"), "verified": ob.get("verified"),
        "retrieved": sorted((ob.get("retrieved") or {}).keys()),
        "position": ob.get("position"),
    })


def loops(row, window=WINDOW):
    trace = row["trace"]
    o = row["outcome"]
    keys = [s["key"] for s in o["p1_audit"]["steps"]]
    n = len(trace)
    assert n == len(keys) == o["steps"], (n, len(keys), o["steps"])
    pre = [decision_state(t["observation"]) for t in trace]
    nev = [len(t["observation"].get("events") or []) for t in trace]
    post = [pre[k + 1] if k + 1 < n else None for k in range(n)]
    unknown_last = 0
    if n:
        k = n - 1
        for j in range(n - 1):
            if pre[j] == pre[k] and keys[j] == keys[k]:
                post[k] = post[j]
                break
        a = trace[k]["action"]
        if post[k] is None and a["kind"] == "retrieve" and trace[k]["feedback"].get("status") == "success" \
                and a["arguments"].get("handle") in (trace[k]["observation"].get("retrieved") or {}):
            post[k] = pre[k]  # re-retrieving an already retrieved record has no effect on the decision state
    # breakers
    inspected = set()
    breaker = []
    total_events = len(o.get("events") or [])
    for k, t in enumerate(trace):
        a = t["action"]
        b = False
        if a["kind"] == "call":
            b = True
        if a["kind"] == "inspect":
            tgt = json.dumps(a.get("arguments"), sort_keys=True)
            if tgt not in inspected:
                b = True
                inspected.add(tgt)
        after = nev[k + 1] if k + 1 < n else total_events
        if after > nev[k]:
            b = True
        breaker.append(b)
    idem = cyc = 0
    last_accepted_key = None
    last_break = -1  # index of last breaker action
    for k, t in enumerate(trace):
        if breaker[k]:
            last_break = k
        accepted = t["feedback"].get("status") == "success"
        if accepted and post[k] is None:
            unknown_last += 1
        elif accepted:
            is_idem = last_accepted_key == keys[k] and post[k] == pre[k]
            is_cyc = False
            if not breaker[k]:
                lo = max(k - window + 1, last_break + 1, 0)
                is_cyc = any(post[k] == pre[m] for m in range(lo, k + 1))
            idem += is_idem
            cyc += is_cyc or is_idem
        if accepted:
            last_accepted_key = keys[k]
    return {"decisions": n, "idem": idem, "cyc": cyc, "np": cyc, "unknown_last": unknown_last}


def process_file(args):
    path, window = args
    cond_dir = os.path.basename(os.path.dirname(path))
    mode = "sampled" if cond_dir.endswith("-sampled") else "greedy"
    cond = cond_dir[:-len("-sampled")] if mode == "sampled" else cond_dir
    pol = POLICY_OF_FILE[os.path.basename(path)[:-len(".jsonl.gz")]]
    c = collections.Counter()
    seeds, samp = [], []
    for line in gzip.open(path, "rt"):
        row = json.loads(line)
        o = row["outcome"]
        seeds.append(row["seed"])
        samp.append((row.get("policy_mode"), row.get("sampling_seed"), row.get("sampling_rng_seed")))
        c["episodes"] += 1
        c["success"] += bool(o["verified_success"])
        c["utility"] += o["utility"]
        c["work"] += o["work_units"]
        c["cap"] += o["steps"] >= 96
        c["truncated"] += bool(row.get("truncated"))
        for h in o["history"]:
            c["kind/" + h["action"]["kind"]] += 1
        if "trace" in row:
            lp = loops(row, window)
        else:  # references carry no trace; registered: "The references score 0."
            c["no_trace"] += 1
            lp = {"decisions": o["steps"], "idem": 0, "cyc": 0, "np": 0, "unknown_last": 0}
        c["decisions"] += lp["decisions"]
        c["idem"] += lp["idem"]
        c["cyc"] += lp["cyc"]
        c["np"] += lp["np"]
        c["np_any_ep"] += lp["np"] > 0
        c["unknown_last"] += lp["unknown_last"]
        for w in correct_reuse(o["p1_audit"]["steps"]):
            c["opp"] += w["opp"]
            c["correct"] += w["correct"]
        ra = o.get("reuse_audit") or []
        c["audited"] += len(ra)
        c["invalid"] += sum(1 for u in ra if not u.get("applicable_hidden"))
    return {"cond": cond, "mode": mode, "policy": pol, "counts": dict(c), "seeds": seeds, "sampling": samp}


def metrics(c):
    e = c["episodes"]
    d = c["decisions"]
    return {
        "success": c["success"] / e, "utility": c["utility"] / e,
        "work_per_success": c["work"] / c["success"] if c["success"] else None,
        "idempotent_repeat_rate": c.get("idem", 0) / d, "short_cycle_rate": c.get("cyc", 0) / d,
        "no_progress_rate": c.get("np", 0) / d, "no_progress_per_episode": c.get("np", 0) / e,
        "no_progress_any_episode": c.get("np_any_ep", 0) / e,
        "steps_to_cap_rate": c.get("cap", 0) / e,
        "correct_reuse_rate": c.get("correct", 0) / c["opp"] if c.get("opp") else None,
        "invalid_reuse_rate": c.get("invalid", 0) / c["audited"] if c.get("audited") else None,
        "truncated": c.get("truncated", 0) / e, "no_trace_episodes": c.get("no_trace", 0), "unknown_last": c.get("unknown_last", 0),
        "episodes": e, "decisions": d,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("screening")
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    for cd in sorted(os.listdir(a.screening)):
        p = os.path.join(a.screening, cd)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".jsonl.gz") and f != "worlds.jsonl.gz":
                jobs.append((os.path.join(p, f), a.window))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(process_file, jobs))
    cells, integrity = {}, {}
    worlds = {}
    for cd in sorted(os.listdir(a.screening)):
        wp = os.path.join(a.screening, cd, "worlds.jsonl.gz")
        if os.path.exists(wp):
            worlds[cd] = [json.loads(l)["seed"] for l in gzip.open(wp, "rt")]
    seeds_by = collections.defaultdict(dict)
    samp_by = collections.defaultdict(dict)
    for r in res:
        key = f"{r['policy']}/{r['mode']}/{r['cond']}"
        cells[key] = metrics(r["counts"])
        cells[key]["action_mix"] = {k[5:]: v / r["counts"]["decisions"] for k, v in r["counts"].items() if k.startswith("kind/")}
        seeds_by[(r["cond"], r["mode"])][r["policy"]] = r["seeds"]
        if r["mode"] == "sampled":
            samp_by[r["cond"]][r["policy"]] = r["sampling"]
    # integrity: identical world order across policies and modes
    ident = {}
    all_seeds = set()
    for (cond, mode), d in seeds_by.items():
        lists = list(d.values())
        ident[f"{cond}/{mode}"] = all(l == lists[0] for l in lists) and lists[0] == worlds.get(cond if mode == "greedy" else cond + "-sampled")
        all_seeds.update(lists[0])
    same_worlds_modes = {c: seeds_by[(c, "greedy")]["bootstrap"] == seeds_by[(c, "sampled")]["bootstrap"] for c in CONDITIONS}
    samp_fixed = {}
    for c, d in samp_by.items():
        vals = list(d.values())
        samp_fixed[c] = {"identical_across_policies": all(v == vals[0] for v in vals),
                         "sampling_seed": sorted({v[1] for v in vals[0]}),
                         "distinct_rng_seeds": len({v[2] for v in vals[0]}),
                         "modes": sorted({str(v[0]) for v in vals[0]})}
    integrity = {"worlds_identical_within_condition_mode": ident, "greedy_sampled_same_worlds": same_worlds_modes,
                 "sampling": samp_fixed, "seed_min": min(all_seeds), "seed_max": max(all_seeds),
                 "n_seeds": len(all_seeds),
                 "seed_ranges": {c: [min(seeds_by[(c, 'greedy')]['bootstrap']), max(seeds_by[(c, 'greedy')]['bootstrap']),
                                     len(set(seeds_by[(c, 'greedy')]['bootstrap']))] for c in CONDITIONS}}
    groups = {}
    for key in sorted({k.rsplit("/", 1)[0] for k in cells}):
        g = {}
        for m in ["success", "utility", "work_per_success", "idempotent_repeat_rate", "short_cycle_rate",
                  "no_progress_rate", "no_progress_per_episode", "no_progress_any_episode", "steps_to_cap_rate",
                  "correct_reuse_rate", "invalid_reuse_rate"]:
            g[m] = mean([cells[f"{key}/{c}"][m] for c in IID if f"{key}/{c}" in cells])
        groups[key] = g
    b, c1 = groups["bootstrap/greedy"], groups["c1_final/greedy"]
    readings = {
        "S1": {"holds": c1["success"] >= b["success"] - 0.02 and c1["no_progress_rate"] <= b["no_progress_rate"] + 0.02,
               "c1_success": c1["success"], "boot_success": b["success"],
               "c1_no_progress": c1["no_progress_rate"], "boot_no_progress": b["no_progress_rate"]},
        "S2": {"holds": c1["utility"] >= b["utility"] + 0.01 or (
            c1["work_per_success"] <= 0.9 * b["work_per_success"] and c1["success"] >= b["success"] - 0.02),
               "c1_utility": c1["utility"], "boot_utility": b["utility"],
               "c1_wps": c1["work_per_success"], "boot_wps": b["work_per_success"],
               "wps_ratio": c1["work_per_success"] / b["work_per_success"]},
        "S3": {arm: {"final_greedy_success": groups[f"{arm}_final/greedy"]["success"],
                     "deployed_greedy_success": groups[f"{arm}_deployed/greedy"]["success"],
                     "final_sampled_success": groups[f"{arm}_final/sampled"]["success"],
                     "deployed_sampled_success": groups[f"{arm}_deployed/sampled"]["success"]} for arm in ["c0", "c1"]},
    }
    json.dump({"window": a.window, "cells": cells, "groups": groups, "readings": readings, "integrity": integrity},
              open(a.output, "w"), indent=1, sort_keys=True)
    print(json.dumps(readings, indent=1))
    print(json.dumps(integrity, indent=1)[:3000])


if __name__ == "__main__":
    main()
