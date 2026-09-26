"""Independent F1/F1b reconstruction from raw depworld eval rows (extended-04 F audit).

Written without reading the campaign's F scorers. Reads only raw rows, worlds files,
configs/summary and receipts. Prints JSON to stdout.
"""
import glob, gzip, hashlib, json, os, sys
from collections import Counter, defaultdict

R = os.path.expanduser("~/tensegra-campaign04/results")
LIN = ["f-boot-x1-r3", "f-boot-x1-r4", "f-boot-x1-r5"]
CONDS = ["iid_f0", "iid_f2", "events_train_kinds_p1", "foreign4"]
MODES = ["greedy", "r_mask"]
START = {"f1": 170_000_000, "f1b": 180_000_000}


def rows(path, slim=False):
    out = []
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            if slim:
                o = r["outcome"]
                r = {"seed": r["seed"], "spec_hash": r["spec_hash"], "truncated": r.get("truncated"),
                     "progress": r["progress"],
                     "outcome": {"steps": o["steps"], "utility": o["utility"], "cost": o["cost"],
                                 "verified_success": o["verified_success"],
                                 "history": [{"action": h["action"]} for h in o["history"]]}}
            out.append(r)
    return out


NPC = set("CD")  # campaign04_progress CODES: short_cycle=C, idempotent=D


def first_np(classes):
    return next((i for i, ch in enumerate(classes) if ch in NPC), None)


def summarise(rs):
    n = len(rs)
    steps = sum(r["outcome"]["steps"] for r in rs)
    np_steps = sum(r["progress"]["no_progress"] for r in rs)
    return {
        "n": n,
        "success": sum(bool(r["outcome"]["verified_success"]) for r in rs) / n,
        "utility": sum(r["outcome"]["utility"] for r in rs) / n,
        "cost": sum(r["outcome"]["cost"] for r in rs) / n,
        "steps": steps,
        "np_steps": np_steps,
        "np_step_rate": np_steps / steps,
        "np_step_rate_mean_of_eps": sum(r["progress"]["no_progress"] / r["outcome"]["steps"] for r in rs) / n,
        "np_episode_rate": sum(r["progress"]["no_progress"] > 0 for r in rs) / n,
        "flag_episode_rate": sum(r["progress"]["flags"] > 0 for r in rs) / n,
        "interventions_per_ep": sum(len(r["progress"]["interventions"]) for r in rs) / n,
        "truncated": sum(bool(r.get("truncated")) for r in rs),
    }


def actions(r):
    return [json.dumps(h["action"], sort_keys=True) for h in r["outcome"]["history"]]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


out = {"experiments": {}, "integrity": {}}
class_letters = Counter()
spec_by_exp = {}
for exp in ["f1", "f1b"]:
    E = out["experiments"][exp] = {"cells": {}, "iid": {}, "mechanism": {}, "worlds": {}}
    specs = {}  # (cond) -> {seed: spec_hash}
    world_problems = []
    for lin in LIN + ["references"]:
        d = f"{R}/{exp}-{lin}"
        for cond in CONDS:
            for mode in (MODES if lin != "references" else ["greedy"]):
                cdir = f"{d}/{cond}-{mode}"
                fn = glob.glob(f"{cdir}/*.jsonl.gz")
                fn = [x for x in fn if not x.endswith("worlds.jsonl.gz")]
                assert len(fn) == 1, (cdir, fn)
                rs = rows(fn[0], slim=True)
                ws = rows(f"{cdir}/worlds.jsonl.gz")
                seeds = [r["seed"] for r in rs]
                exp_start = START[exp] + 100_000 * CONDS.index(cond)
                ok_seeds = sorted(seeds) == list(range(exp_start, exp_start + 512))
                if not ok_seeds:
                    world_problems.append((lin, cond, mode, "seed set", min(seeds), max(seeds), len(seeds)))
                wmap = {w["seed"]: w["spec_hash"] for w in ws}
                for r in rs:
                    if wmap.get(r["seed"]) != r["spec_hash"]:
                        world_problems.append((lin, cond, mode, "row/world spec mismatch", r["seed"]))
                    prev = specs.setdefault(cond, {}).setdefault(r["seed"], r["spec_hash"])
                    if prev != r["spec_hash"]:
                        world_problems.append((lin, cond, mode, "spec differs across policies/modes", r["seed"]))
                    for ch in r["progress"]["classes"]:
                        class_letters[ch] += 1
                    if sum(ch in NPC for ch in r["progress"]["classes"]) != r["progress"]["no_progress"]:
                        class_letters["_np_count_mismatch"] += 1
                    if len(r["progress"]["classes"]) != r["outcome"]["steps"]:
                        class_letters["_len_mismatch"] += 1
                key = f"{lin}/{mode}/{cond}"
                E["cells"][key] = summarise(rs)
                E["cells"][key]["_rows"] = {r["seed"]: r for r in rs}
    E["worlds"] = {"problems": world_problems[:20], "n_problems": len(world_problems),
                   "distinct_specs_per_cond": {c: len(set(v.values())) for c, v in specs.items()},
                   "seeds_per_cond": {c: len(v) for c, v in specs.items()}}
    spec_by_exp[exp] = {c: set(v.values()) for c, v in specs.items()}
    # IID group and mechanism
    for lin in LIN + ["references"]:
        for mode in (MODES if lin != "references" else ["greedy"]):
            rs = []
            for cond in ("iid_f0", "iid_f2"):
                rs += list(E["cells"][f"{lin}/{mode}/{cond}"]["_rows"].values())
            E["iid"][f"{lin}/{mode}"] = summarise(rs)
            if lin != "references":
                E["iid"][f"{lin}/{mode}"]["mean_of_cond_np_step_rate"] = sum(
                    E["cells"][f"{lin}/{mode}/{c}"]["np_step_rate"] for c in ("iid_f0", "iid_f2")) / 2
    verdict = {}
    for lin in LIN:
        g, m = E["iid"][f"{lin}/greedy"], E["iid"][f"{lin}/r_mask"]
        c_u = m["utility"] >= g["utility"] - 0.005
        c_s = m["success"] >= g["success"] - 0.01
        c_ep = m["np_episode_rate"] <= 0.5 * g["np_episode_rate"]
        c_st = m["np_step_rate"] <= 0.5 * g["np_step_rate"]
        verdict[lin] = {"dU": m["utility"] - g["utility"], "dS": m["success"] - g["success"],
                        "ep_greedy": g["np_episode_rate"], "ep_rmask": m["np_episode_rate"],
                        "step_greedy": g["np_step_rate"], "step_rmask": m["np_step_rate"],
                        "step_ratio": g["np_step_rate"] / max(m["np_step_rate"], 1e-12),
                        "utility_clause": c_u, "success_clause": c_s,
                        "episode_clause": c_ep, "step_clause": c_st}
    E["verdict_per_lineage"] = verdict
    E["F_episode_H"] = all(v["utility_clause"] and v["success_clause"] and v["episode_clause"] for v in verdict.values())
    E["F_step_H"] = all(v["utility_clause"] and v["success_clause"] and v["step_clause"] for v in verdict.values())
    # floor: r_mask success >= .8 * dep_reuse success per condition
    E["floor"] = {f"{lin}/{cond}": E["cells"][f"{lin}/r_mask/{cond}"]["success"] >= 0.8 * E["cells"][f"references/greedy/{cond}"]["success"]
                  for lin in LIN for cond in CONDS}
    # mechanism: episode sets and prefix identity
    mech = {}
    for lin in LIN:
        for cond in CONDS:
            G = E["cells"][f"{lin}/greedy/{cond}"]["_rows"]
            M = E["cells"][f"{lin}/r_mask/{cond}"]["_rows"]
            gset = {s for s, r in G.items() if r["progress"]["no_progress"] > 0}
            mset = {s for s, r in M.items() if r["progress"]["no_progress"] > 0}
            prefix_ok = 0; prefix_bad = []; first_step_same = 0; identical_full = 0
            first_int_after_first_np = 0; first_int_total = 0
            for s in gset & mset:
                ga, ma = actions(G[s]), actions(M[s])
                gc, mc = G[s]["progress"]["classes"], M[s]["progress"]["classes"]
                # first no-progress step index: position of first non-progress class letter
                gi, mi = first_np(gc), first_np(mc)
                if gi is not None and gi == mi:
                    first_step_same += 1
                if gi is not None and ga[:gi + 1] == ma[:gi + 1]:
                    prefix_ok += 1
                else:
                    prefix_bad.append(s)
                if ga == ma:
                    identical_full += 1
                ints = M[s]["progress"]["interventions"]
                if ints:
                    first_int_total += 1
                    st = ints[0].get("step") if isinstance(ints[0], dict) else None
                    if st is not None and mi is not None and st > mi:
                        first_int_after_first_np += 1
            # episodes without any NP in greedy: are r_mask trajectories identical?
            same_traj_clean = sum(actions(G[s]) == actions(M[s]) for s in G if s not in gset)
            mech[f"{lin}/{cond}"] = {
                "np_eps_greedy": len(gset), "np_eps_rmask": len(mset),
                "sets_identical": gset == mset, "only_greedy": len(gset - mset), "only_rmask": len(mset - gset),
                "prefix_identical_through_first_np": prefix_ok, "prefix_mismatch": len(prefix_bad),
                "first_np_index_same": first_step_same, "fully_identical_np_eps": identical_full,
                "clean_eps": len(G) - len(gset), "clean_eps_identical_trajectory": same_traj_clean,
                "rmask_eps_with_intervention": first_int_total,
                "first_intervention_step_after_first_np": first_int_after_first_np,
            }
    E["mechanism"] = mech
    for k in E["cells"]:
        del E["cells"][k]["_rows"]

out["class_letters"] = dict(class_letters)
# cross-experiment world freshness
out["integrity"]["f1_f1b_spec_overlap"] = {c: len(spec_by_exp["f1"][c] & spec_by_exp["f1b"][c]) for c in CONDS}
allf1b = set().union(*spec_by_exp["f1b"].values())
allf1 = set().union(*spec_by_exp["f1"].values())
other = {}
for wf in glob.glob(f"{R}/*/*/worlds.jsonl.gz"):
    top = wf.split("/")[-3]
    if top.startswith("f1-") or top.startswith("f1b-"):
        continue
    try:
        hs = {w["spec_hash"] for w in rows(wf)}
    except Exception as e:
        continue
    o1b, o1 = len(hs & allf1b), len(hs & allf1)
    k = top.split("-")[0]
    rec = other.setdefault(k, {"files": 0, "worlds": 0, "overlap_f1b": 0, "overlap_f1": 0})
    rec["files"] += 1; rec["worlds"] += len(hs); rec["overlap_f1b"] += o1b; rec["overlap_f1"] += o1
out["integrity"]["other_eval_world_overlap"] = other
# checkpoints
ck = {}
for lin in LIN:
    files = sorted(glob.glob(f"{R}/{lin}/checkpoints/*.pt"), key=os.path.getmtime)
    ck[lin] = {"latest_by_mtime": os.path.basename(files[-1]), "n": len(files), "sha256_latest": sha(files[-1]),
               "files": [os.path.basename(f) for f in files]}
    for exp in ["f1", "f1b"]:
        s = json.load(open(f"{R}/{exp}-{lin}/summary.json"))
        c = s["config"]["checkpoints"][0]
        ck[lin][f"{exp}_config_path"] = os.path.basename(c["path"])
        ck[lin][f"{exp}_config_sha"] = c["sha256"]
    try:
        st = json.load(open(f"{R}/{lin}/state.json"))
        ck[lin]["state_keys"] = list(st)[:30]
        ck[lin]["state_excerpt"] = json.dumps(st)[:600]
    except Exception as e:
        ck[lin]["state_err"] = str(e)
out["integrity"]["checkpoints"] = ck
# receipts
rec = {}
for p in sorted(glob.glob(f"{R}/f-boot-x1-r*-process") + glob.glob(f"{R}/f1-*-process") + glob.glob(f"{R}/f1b-*-process")
                + glob.glob(f"{R}/f-b*-process") + glob.glob(f"{R}/b2-*-process")):
    try:
        o = json.load(open(f"{p}/occupancy.json")); l = json.load(open(f"{p}/launch.json"))
        rec[os.path.basename(p)] = {"exit": o["exit_code"], "cpu": round(o["cpu_core_seconds"], 1), "wall": round(o["wall_seconds"], 1),
                                    "start": l["started_unix"], "end": o["ended_unix"], "cwd": os.path.basename(l["cwd"]),
                                    "cmd": " ".join(l["command"][-12:])}
    except Exception as e:
        rec[os.path.basename(p)] = {"error": str(e)}
out["integrity"]["receipts"] = rec
json.dump(out, sys.stdout, indent=1, default=str)
