#!/usr/bin/env python3
"""Phase-3b independent raw audit (E19/E20/E21/E22 sealed rows + run seed ranges).

Stdlib only. Written fresh for this audit; shares no code with earlier audit
scripts. Read-only: it opens result files under RESULTS and prints one JSON
document on stdout. Intended invocation:

    ssh gb10-direct 'python3 - ' < phase3b_raw_remote.py > phase3b-raw.json
"""
import glob
import gzip
import json
import os
import resource
import sys
import time

RESULTS = os.path.expanduser("~/topoformer-campaign02/results")
PRIM = {"select": "constrained_subset", "route": "shortest_path", "assign": "csp"}
STAGE_OF = {v: k for k, v in PRIM.items()}
COMPLETION = {"commit_pending", "commit_assignment", "use_return", "deliver", "move"}
BAD = {"rejected", "invalid_input"}
T0 = time.process_time()


def cut_parse(line):
    """Parse only the prefix up to the top-level timing key (skips the large trace)."""
    i = line.find(', "timing": ')
    if i > 0:
        try:
            return json.loads(line[:i] + "}"), False
        except ValueError:
            pass
    return json.loads(line), True


def rows(path, full=False):
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if full:
                yield json.loads(line)
            else:
                yield cut_parse(line)[0]


def worlds(root, cond):
    out = {}
    for d in rows(os.path.join(RESULTS, root, cond, "worlds.jsonl.gz"), full=True):
        out[d["seed"]] = d
    return out


def episode(d):
    o = d["outcome"]
    h = o.get("history", [])
    rej = sum(1 for x in h if x["action"]["kind"] in COMPLETION and x["feedback"].get("status") in BAD)
    own = set()
    prior_uses, prior_use_success = 0, 0
    for x in h:
        k, fb = x["action"]["kind"], x["feedback"]
        if k in ("call", "inspect") and fb.get("return"):
            own.add(fb["return"])
    for x in h:
        if x["action"]["kind"] == "use_return":
            hd = x["action"]["arguments"].get("handle")
            if hd not in own:
                prior_uses += 1
                prior_use_success += fb_ok(x["feedback"])
    return {"seed": d["seed"], "spec_hash": d.get("spec_hash"), "ok": bool(o.get("verified_success")),
            "utility": o.get("utility"), "steps": len(h), "rej": rej, "prior_uses": prior_uses,
            "prior_use_success": prior_use_success, "stages_completed": o.get("stages_completed"),
            "kinds": [x["action"]["kind"] for x in h],
            "calls": sum(1 for x in h if x["action"]["kind"] == "call")}


def fb_ok(fb):
    return int(fb.get("status") == "success")


def arm_eps(root, cond, arm):
    p = os.path.join(RESULTS, root, cond, arm + ".jsonl.gz")
    return [episode(d) for d in rows(p)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


# ---------- distractor validity vs the goal (independent re-implementation) ----------
def subset_valid(spec, handles):
    inv = {x["handle"]: x for x in spec["items"]}
    if len(set(handles)) != len(handles) or any(h not in inv for h in handles):
        return False
    ch = [inv[h] for h in handles]
    if sorted(x["category"] for x in ch) != sorted(spec["categories"]):
        return False
    if sum(x["weight"] for x in ch) > spec["capacity"] or sum(x["price"] for x in ch) > spec["funds"]:
        return False
    return not any(a in handles and b in handles for a, b in spec["incompatible"])


def assign_valid(spec, vals):
    if len(vals) != len(spec["domains"]):
        return False
    if any(v not in d for v, d in zip(vals, spec["domains"])):
        return False
    return not any(vals[i] == x and vals[j] == y for i, x, j, y in spec["forbidden"])


def distractor_facts(spec):
    goal_prims = {PRIM[s] for s in spec["stages"]}
    out = []
    for prim, payload, snap in spec["distractors"]:
        f = {"prim": prim, "in_goal": prim in goal_prims, "snap_prim_match": snap.get("primitive") == prim}
        if prim == "constrained_subset":
            hs = [snap["problem"]["handles"][i] for i in payload[0]]
            f["accidentally_valid"] = subset_valid(spec, hs)
            f["same_item_set"] = sorted(snap["problem"]["handles"]) == sorted(x["handle"] for x in spec["items"])
        elif prim == "csp":
            f["accidentally_valid"] = assign_valid(spec, list(payload))
        else:
            f["accidentally_valid"] = payload[0][0] == spec["start"]  # usable only from its own start
            f["path_start"] = payload[0][0]
        out.append(f)
    return out


def e19():
    root = "e19-sealed"
    conds = sorted(os.listdir(os.path.join(RESULTS, root)))
    conds = [c for c in conds if os.path.isdir(os.path.join(RESULTS, root, c))]
    arms = ["e17rl-r0", "e17rl-r1", "e17rl-r2", "reference-modular_cheap_first", "reference-modular_cheap",
            "reference-modular_always_tool"]
    res = {"conditions": {}, "arm_mean": {}}
    per_arm = {a: [] for a in arms}
    xt = {a: {"zero": [0, 0], "some": [0, 0]} for a in arms}
    dist_summary = {"n_worlds": 0, "n_distractors": 0, "not_in_goal": 0, "snap_mismatch": 0,
                    "accidentally_valid": 0, "by_prim": {}, "count_hist": {}, "subset_same_item_set": 0,
                    "subset_total": 0}
    prior_use = {a: {"success_in_distractor_worlds": 0, "those_using_prior": 0, "failures_using_prior": 0,
                     "failures": 0} for a in arms}
    for c in conds:
        W = worlds(root, c)
        facts = {}
        for s, w in W.items():
            fs = distractor_facts(w["spec"])
            facts[s] = fs
            dist_summary["n_worlds"] += 1
            dist_summary["count_hist"][len(fs)] = dist_summary["count_hist"].get(len(fs), 0) + 1
            for f in fs:
                dist_summary["n_distractors"] += 1
                dist_summary["not_in_goal"] += (not f["in_goal"])
                dist_summary["snap_mismatch"] += (not f["snap_prim_match"])
                dist_summary["accidentally_valid"] += f["accidentally_valid"]
                dist_summary["by_prim"][f["prim"]] = dist_summary["by_prim"].get(f["prim"], 0) + 1
                if f["prim"] == "constrained_subset":
                    dist_summary["subset_total"] += 1
                    dist_summary["subset_same_item_set"] += f["same_item_set"]
        cres = {}
        for a in arms:
            eps = arm_eps(root, c, a)
            assert len(eps) == 256 and {e["seed"] for e in eps} == set(W), (c, a)
            assert all(e["spec_hash"] == W[e["seed"]]["spec_hash"] for e in eps)
            sr = mean(e["ok"] for e in eps)
            cres[a] = sr
            per_arm[a].append(sr)
            for e in eps:
                k = "zero" if not facts[e["seed"]] else "some"
                xt[a][k][0] += e["ok"]
                xt[a][k][1] += 1
                if k == "some" and e["ok"]:
                    prior_use[a]["success_in_distractor_worlds"] += 1
                    prior_use[a]["those_using_prior"] += e["prior_uses"] > 0
                if not e["ok"]:
                    prior_use[a]["failures"] += 1
                    prior_use[a]["failures_using_prior"] += e["prior_uses"] > 0
        n0 = sum(1 for s in W if not facts[s])
        cres["zero_distractor_worlds"] = n0
        cres["worlds_with_accidentally_valid_distractor"] = sum(1 for s in W if any(f["accidentally_valid"] for f in facts[s]))
        res["conditions"][c] = cres
    for a in arms:
        res["arm_mean"][a] = mean(per_arm[a])
    res["success_by_distractor_presence"] = {a: {k: {"success": v[0], "n": v[1], "rate": v[0] / v[1]}
                                                 for k, v in xt[a].items()} for a in arms}
    res["distractors"] = dist_summary
    res["prior_record_use"] = prior_use
    return res


def success_table(root, arms, conds):
    out = {}
    for c in conds:
        out[c] = {}
        for a in arms:
            eps = arm_eps(root, c, a)
            out[c][a] = eps
    return out


def group_of(c):
    if c.startswith("same_"):
        return "same"
    if c.startswith("distr_"):
        return "wrong_type"
    if c.startswith("iid_") and "-" in c:
        return "iid_pairs"
    if c.startswith("iid_"):
        return "iid_singles"
    if c.startswith("heldpair_"):
        return "heldpairs"
    if c.startswith("triple_"):
        return "triples"
    if c.startswith("hard"):
        return "hard"
    return "other"


def summarize(tab, arms):
    per = {c: {a: mean(e["ok"] for e in tab[c][a]) for a in arms} for c in tab}
    groups = {}
    for c in tab:
        groups.setdefault(group_of(c), []).append(c)
    gm = {g: {a: mean(per[c][a] for c in cs) for a in arms} for g, cs in groups.items()}
    return per, gm, groups


def failure_detail(tab, arms, exclude_hard=True):
    out = {}
    for a in arms:
        fails = []
        for c, byarm in tab.items():
            if exclude_hard and c.startswith("hard"):
                continue
            for e in byarm[a]:
                if not e["ok"]:
                    fails.append({"cond": c, "seed": e["seed"], "rej": e["rej"], "prior_uses": e["prior_uses"],
                                  "steps": e["steps"], "stages_completed": e["stages_completed"]})
        out[a] = {"failures": len(fails), "perseveration": sum(f["rej"] >= 10 for f in fails),
                  "min_rej": min((f["rej"] for f in fails), default=None),
                  "max_rej": max((f["rej"] for f in fails), default=None),
                  "failures_using_prior_record": sum(f["prior_uses"] > 0 for f in fails),
                  "by_condition": {}, "list": fails[:200]}
        for f in fails:
            out[a]["by_condition"][f["cond"]] = out[a]["by_condition"].get(f["cond"], 0) + 1
    return out


def world_hashes(root, cond):
    return {s: w["spec_hash"] for s, w in worlds(root, cond).items()}


def e20_e22():
    conds = sorted(c for c in os.listdir(os.path.join(RESULTS, "e20-sealed"))
                   if os.path.isdir(os.path.join(RESULTS, "e20-sealed", c)))
    a20 = ["e20rl-r0", "e20rl-r1", "e20rl-r2"]
    a22 = ["e22rl-r0", "e22rl-r1", "e22rl-r2"]
    a17 = ["e17rl-r0", "e17rl-r1", "e17rl-r2"]
    t20 = success_table("e20-sealed", a20, conds)
    t22 = success_table("e22-sealed", a22, conds)
    c17 = [c for c in conds if os.path.isdir(os.path.join(RESULTS, "e17-sealed", c))]
    t17 = success_table("e17-sealed", a17, c17)
    same = [c for c in conds if c.startswith("same_")]
    t17.update(success_table("e19-sealed", a17, same))
    # identical worlds across roots
    hash_check = {}
    wrong_type = {"n": 0, "in_goal": 0}
    for c in conds:
        h20 = world_hashes("e20-sealed", c)
        h22 = world_hashes("e22-sealed", c)
        other = "e19-sealed" if c.startswith("same_") else "e17-sealed"
        h17 = world_hashes(other, c)
        hash_check[c] = {"n": len(h20), "e22_equal": h20 == h22, "e17_or_e19_equal": h20 == h17}
        if c.startswith("distr_"):
            for s, w in worlds("e20-sealed", c).items():
                for f in distractor_facts(w["spec"]):
                    wrong_type["n"] += 1
                    wrong_type["in_goal"] += f["in_goal"]
        # rows' spec hashes equal worlds' hashes
        for t, arms in ((t20, a20), (t22, a22), (t17, a17)):
            for a in arms:
                if c in t:
                    assert all(e["spec_hash"] == h20[e["seed"]] for e in t[c][a]), (c, a)
    p20, g20, groups = summarize(t20, a20)
    p22, g22, _ = summarize(t22, a22)
    p17, g17, _ = summarize(t17, a17)
    nonhard = [c for c in conds if not c.startswith("hard")]
    fd20 = failure_detail(t20, a20)
    fd22 = failure_detail(t22, a22)
    # rejected completion count distribution in E20 r2 failed triples
    r2 = fd20["e20rl-r2"]["list"]
    triple_rej = sorted(f["rej"] for f in r2 if f["cond"].startswith("triple_"))
    # which kinds are rejected in E20 r2 failures
    # retention: E20 - E17 per group
    delta_same = {f"r{i}": g20["same"][a20[i]] - g17["same"][a17[i]] for i in range(3)}
    retention = {g: {f"r{i}": g20[g][a20[i]] - g17[g][a17[i]] for i in range(3)} for g in g17}
    e22_vs_e20 = {g: {f"r{i}": g22[g][a22[i]] - g20[g][a20[i]] for i in range(3)} for g in g20}
    per_cond_drop = []
    for c in nonhard:
        for i in range(3):
            d = p22[c][a22[i]] - p20[c][a20[i]]
            per_cond_drop.append((d, c, i))
    per_cond_drop.sort()
    return {"conditions": conds, "nonhard_count": len(nonhard),
            "world_identity": hash_check, "wrong_type_distractors": wrong_type,
            "per_condition": {"e17": p17, "e20": p20, "e22": p22},
            "groups": {"e17": g17, "e20": g20, "e22": g22}, "group_members": groups,
            "failures_nonhard": {"e20": fd20, "e22": fd22},
            "e20_r2_triple_failure_rejections": triple_rej,
            "delta_same_e20_minus_e17": delta_same, "retention_e20_minus_e17": retention,
            "e22_minus_e20": e22_vs_e20, "worst_condition_drops_e22_vs_e20": per_cond_drop[:6]}, t20, t22


def m3_trace_check(t22):
    """Recompute m3 counters from the actor-visible history and compare to logged observations."""
    root = "e22-sealed"
    checked = mismatches = eps = 0
    examples = []
    budget = 1200
    for c, byarm in t22.items():
        for a, lst in byarm.items():
            want = {e["seed"] for e in lst if e["rej"] > 0}
            if not want:
                continue
            p = os.path.join(RESULTS, root, c, a + ".jsonl.gz")
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    if eps >= budget:
                        break
                    i = line.find('"seed": ')
                    if int(line[i + 8:line.find(",", i)]) not in want:
                        continue
                    d = json.loads(line)
                    if d.get("truncated"):
                        continue
                    eps += 1
                    rej = calls = 0
                    ncomp = 0
                    for step in d["trace"]:
                        ob = step["observation"]
                        checked += 1
                        if (ob["stage_rejections"], ob["stage_solver_calls"]) != (rej, calls):
                            mismatches += 1
                            if len(examples) < 5:
                                examples.append({"cond": c, "arm": a, "seed": d["seed"], "logged": [ob["stage_rejections"], ob["stage_solver_calls"]], "recomputed": [rej, calls]})
                        k, fb = step["action"]["kind"], step["feedback"]
                        if "stage_completed" in fb:
                            rej = calls = 0
                        else:
                            if k in COMPLETION and fb.get("status") in BAD:
                                rej += 1
                            if k == "call":
                                calls += 1
    return {"episodes": eps, "steps_checked": checked, "mismatches": mismatches, "examples": examples}


def e21():
    iid = ["iid_3x3", "iid_4x4", "iid_4x4_tight128", "iid_4x4_expensive_work", "iid_3x3_obstacle",
           "iid_4x4_obstacle", "iid_4x5", "iid_4x4_steps36"]
    xfer = sorted(c for c in os.listdir(os.path.join(RESULTS, "e09v2-sealed-e21")) if c.startswith("xfer_"))
    out = {}
    for root, stem in (("e09v2-sealed-e21", "halving-robust"), ("e09v2-sealed-e15", "halving-plain")):
        for r in range(3):
            a = f"{stem}-r{r}"
            iu = [mean(e["utility"] for e in arm_eps(root, c, a)) for c in iid]
            xu = [mean(e["utility"] for e in arm_eps(root, c, a)) for c in xfer]
            ctl = arm_eps(root, "int_4x4_control", a)
            gf, has = 0, 0
            for e in ctl:
                first = next((k for k in e["kinds"] if k in ("choose_item", "commit_pending", "call", "start_subset")), None)
                if first is not None:
                    has += 1
                    gf += first in ("choose_item", "commit_pending")
            nt = arm_eps(root, "int_4x4_no_tools", a)
            out[a] = {"iid_utility": mean(iu), "iid_per_condition": dict(zip(iid, iu)),
                      "transfer_utility": mean(xu), "n_transfer_conditions": len(xu),
                      "greedy_first_all": gf / len(ctl), "greedy_first_among_decided": gf / has if has else None,
                      "control_decided": has, "calls_per_control_episode": mean(e["calls"] for e in ctl),
                      "no_tools_success": mean(e["ok"] for e in nt)}
    same_worlds = {c: world_hashes("e09v2-sealed-e21", c) == world_hashes("e09v2-sealed-e15", c)
                   for c in iid + ["int_4x4_control", "int_4x4_no_tools"]}
    out["worlds_identical_e21_vs_e15"] = same_worlds
    out["transfer_conditions"] = xfer
    return out


def seed_ranges():
    runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RESULTS, "e1[79]-*")) +
                  glob.glob(os.path.join(RESULTS, "e2[0-2]-*")) if os.path.isdir(p)
                  and os.path.exists(os.path.join(p, "state.json")))
    out = {}
    for r in runs:
        st = json.load(open(os.path.join(RESULTS, r, "state.json")))
        pr = json.load(open(os.path.join(RESULTS, r, "protocol.json"))) if os.path.exists(os.path.join(RESULTS, r, "protocol.json")) else {}
        ivs = []
        for al in st.get("allocations", []):
            iv = al.get("training_seed_interval")
            if iv:
                ivs.append(tuple(iv))
        dev_seeds = set()
        dev_files = sorted(glob.glob(os.path.join(RESULTS, r, "development", "*.jsonl.gz")))
        pick = dev_files[:1] + dev_files[-1:] if dev_files else []
        per_file = []
        for f in pick:
            s = set()
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    i = line.find('"seed": ')
                    j = line.find(",", i)
                    s.add(int(line[i + 8:j].strip(" }")))
            per_file.append(s)
            dev_seeds |= s
        out[r] = {"training_min": min((a for a, b in ivs), default=None),
                  "training_max": max((b for a, b in ivs), default=None), "n_intervals": len(ivs),
                  "development_min": min(dev_seeds) if dev_seeds else None,
                  "development_max": max(dev_seeds) if dev_seeds else None,
                  "development_n": len(dev_seeds), "dev_files_checked": len(pick), "n_dev_files": len(dev_files),
                  "dev_sets_equal_first_last": (per_file[0] == per_file[-1]) if per_file else None,
                  "protocol_training_seed_start": pr.get("training_seed_start"),
                  "protocol_development_seed_start": pr.get("development_seed_start")}
    sealed = [(90000000, 90050256), (91000000, 91020256), (92000000, 92050256), (94000000, 94010256),
              (95000000, 95050256), (96000000, 96070256), (70000000, 71700256), (75000000, 75100256)]
    overlaps = []
    for r, v in out.items():
        spans = []
        if v["training_min"] is not None:
            spans.append(("training", v["training_min"], v["training_max"]))
        if v["development_min"] is not None:
            spans.append(("development", v["development_min"], v["development_max"]))
        for kind, lo, hi in spans:
            for a, b in sealed:
                if lo < b and a <= hi:
                    overlaps.append((r, kind, lo, hi, a, b))
    return {"runs": out, "sealed_spans": sealed, "overlaps": overlaps}


def main():
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out["e19"] = e19()
    out["e19_cpu"] = time.process_time() - T0
    e2022, t20, t22 = e20_e22()
    out["e20_e22"] = e2022
    out["m3_trace_check"] = m3_trace_check(t22)
    out["e21"] = e21()
    out["seeds"] = seed_ranges()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    out["cpu_seconds_remote"] = ru.ru_utime + ru.ru_stime
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    main()
