#!/usr/bin/env python3
"""Independent raw-row reconstruction of the extended-03 P1 sealed results.

Written by the independent auditor from the registered definitions in
research/campaigns/extended-03/protocol-P1-metrics.md, WITHOUT reading
research/tools/campaign03_p1_analysis.py. Standard library only.

Usage:
    python research/tools/campaign03_p1_audit_reconstruct.py <p1-sealed dir> --output <json> [--workers 4]

Per (condition, arm) it recomputes success, utility, work per success,
correct reuse (per stage window), invalid/stale use, identical retry
(primary + secondaries), revision quality, plus descriptive behaviour
statistics (action mix, episode endings) used for the collapse
characterisation; then the R1-R5 verdicts, per-lineage values and
per-condition readings; and the non-registered confound readings
(bootstrap endpoints; lineage r1 only).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor

CONDITIONS = ["iid_f0", "iid_f2", "noevent_f2", "events_train_kinds_p1", "heldout_deadline_moved",
              "foreign4", "larger_s4", "work_price_x4"]
IID = ["iid_f0", "iid_f2"]
R3_GROUP = ["foreign4", "events_train_kinds_p1", "heldout_deadline_moved"]
NEED_PRIMITIVE = {"select": "constrained_subset", "assign": "csp", "route": "shortest_path"}
PRIMARY_KINDS = {"commit_pending", "commit_assignment", "use_return", "move"}
PIECE_OF_TARGET = {"select": "selection", "assign": "assignment"}
PRIMITIVE_OF_PIECE = {"selection": "constrained_subset", "assignment": "csp"}


def arm_name(fname: str) -> str:
    base = fname[:-len(".jsonl.gz")]
    if base.startswith("p1-"):
        _, endpoint, arm, lin = base.split("-")
        return f"{arm}-{endpoint}-{lin}"
    return base.replace("reference-", "")


# ---------------------------------------------------------------- per episode

def windows_of(steps):
    """Stage windows keyed by (need, open) in first-seen order."""
    order, members = [], collections.defaultdict(list)
    for s in steps:
        need = s.get("need")
        if need is None:
            continue
        key = (need, s["open"])
        if key not in members:
            order.append(key)
        members[key].append(s)
    return order, members


def correct_reuse(steps):
    """Returns per-window facts: opportunity/correct under the registered definition
    (reusable = pre-existing, publicly applicable, completes), the applicable-pre
    secondary, and details of each counted correct reuse."""
    order, members = windows_of(steps)
    out = []
    for key in order:
        need, opened = key
        prim = NEED_PRIMITIVE[need]
        ws = members[key]
        opp = any(s.get("reusable") for s in ws)
        opp_pre = any(s.get("applicable_pre") for s in ws)
        called = called_success = False
        correct = correct_pre = False
        correct_anycall_ok = False  # variant: only *successful* earlier calls block
        detail = None
        for s in ws:
            if s["kind"] == "call" and s.get("call_primitive") == prim:
                called = True
                if s.get("outcome") == "success":
                    called_success = True
            if s["kind"] == "use_return" and s.get("outcome") == "success":
                use = s.get("use") or {}
                h = use.get("handle")
                if use.get("as") == need:
                    if h in (s.get("reusable") or []):
                        if not called and not correct:
                            correct = True
                            detail = {"need": need, "open": opened, "step": s["step"], "handle": h,
                                      "foreign": use.get("foreign"), "created_step": use.get("created_step")}
                        if not called_success:
                            correct_anycall_ok = True
                    if h in (s.get("applicable_pre") or []) and not called:
                        correct_pre = True
        out.append({"need": need, "open": opened, "opp": opp, "correct": opp and correct,
                    "correct_succcall": opp and correct_anycall_ok,
                    "opp_pre": opp_pre, "correct_pre": opp_pre and correct_pre, "detail": detail if opp and correct else None})
    return out


def retries(attempts):
    """Identical-retry counts (primary and secondaries)."""
    res = collections.Counter()
    n = len(attempts)
    for j, a in enumerate(attempts):
        kind = a["action_kind"]
        status = a["outcome_status"]
        nxt = next((b for b in attempts[j + 1:] if b["action_key"] == a["action_key"]), None)
        ident = nxt is not None and nxt["dependency_versions_at_attempt"] == a["dependency_versions_at_attempt"]
        if kind in PRIMARY_KINDS:
            res["primary_attempts"] += 1
            if status == "rejected":
                res["rej"] += 1
                res[f"rej/{kind}"] += 1
                if ident:
                    res["ident"] += 1
                    res[f"ident/{kind}"] += 1
                if a.get("reason") != "missing_dependency":
                    res["rej_nomd"] += 1
                    if ident:
                        res["ident_nomd"] += 1
        if status != "success":
            res["fail_all"] += 1
            if ident:
                res["ident_all"] += 1
    res["attempts_all"] = n
    return res


def revision(outcome):
    ev = outcome["p1_audit"].get("event")
    if not ev:
        return None
    e = ev["step"]
    valid = {p for p in ev["kept"] if (ev.get("kept_completable") or {}).get(p)}
    over = False
    for a in outcome["p1_audit"]["attempts"]:
        if a["step"] <= e:
            continue
        if a["action_kind"] == "uncommit" and a["outcome_status"] == "success":
            piece = PIECE_OF_TARGET.get(a["arguments"].get("target"))
            if piece in valid:
                over = True
    for s in outcome["p1_audit"]["steps"]:
        if s["step"] <= e or s["kind"] != "call":
            continue
        for p in valid:
            if s.get("call_primitive") == PRIMITIVE_OF_PIECE[p]:
                over = True
    return {"over": over, "success": bool(outcome["verified_success"]),
            "good": bool(outcome["verified_success"]) and not over}


def behaviour(row):
    o = row["outcome"]
    hist = o["history"]
    kinds = collections.Counter(h["action"]["kind"] for h in hist)
    last = hist[-1]["action"]["kind"] if hist else None
    steps = o["steps"]
    # stage progress from the audit's public need at each step
    needs = [s.get("need") for s in o["p1_audit"]["steps"]]
    reached = "none"
    if any(n in ("assign", "route") for n in needs) or any(n is None for n in needs):
        reached = "select_committed"
    if any(n == "route" for n in needs) or any(n is None for n in needs):
        reached = "assign_committed"
    if any(n is None for n in needs):
        reached = "at_destination"
    keys = collections.Counter(s["key"] for s in o["p1_audit"]["steps"])
    if o["verified_success"]:
        end = "verified"
    elif last == "abstain":
        end = "abstain"
    elif steps >= 96:
        end = "step_cap_96"
    elif last == "verify":
        end = "verify_failed_end"
    else:
        end = f"other:{last}"
    return {"kinds": kinds, "end": end, "reached": reached, "steps": steps,
            "max_key_repeat": max(keys.values()) if keys else 0, "truncated": bool(row.get("truncated")),
            "last": last, "uses_unique_frac": None}


def process_file(path):
    cond = os.path.basename(os.path.dirname(path))
    arm = arm_name(os.path.basename(path))
    agg = collections.Counter()
    kinds = collections.Counter()
    ends = collections.Counter()
    reached = collections.Counter()
    retry = collections.Counter()
    reuse_details = []
    seeds = []
    per_ep = []
    for line in gzip.open(path, "rt"):
        row = json.loads(line)
        o = row["outcome"]
        seeds.append(row["seed"])
        succ = bool(o["verified_success"])
        agg["episodes"] += 1
        agg["success"] += succ
        agg["utility"] += o["utility"]
        agg["work_units"] += o["work_units"]
        agg["steps"] += o["steps"]
        agg["truncated"] += bool(row.get("truncated"))
        # correct reuse
        ws = correct_reuse(o["p1_audit"]["steps"])
        for w in ws:
            agg["windows"] += 1
            agg["opp"] += w["opp"]
            agg["correct"] += w["correct"]
            agg["correct_succcall"] += w["correct_succcall"]
            agg["opp_pre"] += w["opp_pre"]
            agg["correct_pre"] += w["correct_pre"]
            agg[f"opp/{w['need']}"] += w["opp"]
            agg[f"correct/{w['need']}"] += w["correct"]
            if w["detail"]:
                d = dict(w["detail"])
                d["seed"] = row["seed"]
                reuse_details.append(d)
        # invalid / stale
        audit = o.get("reuse_audit") or []
        agg["audited"] += len(audit)
        agg["invalid"] += sum(1 for u in audit if not u["applicable_hidden"])
        agg["stale"] += sum(1 for u in audit if not u["dependency_match_hidden"])
        agg["invalid_foreign"] += sum(1 for u in audit if not u["applicable_hidden"] and u.get("foreign"))
        agg["invalid_ep"] += any(not u["applicable_hidden"] for u in audit)
        agg["use_attempts"] += sum(1 for s in o["p1_audit"]["steps"] if s["kind"] == "use_return")
        # retries
        retry.update(retries(o["p1_audit"]["attempts"]))
        # revision
        rv = revision(o)
        if rv is not None:
            agg["events"] += 1
            agg["rev_good"] += rv["good"]
            agg["rev_over"] += rv["over"]
        b = behaviour(row)
        kinds.update(b["kinds"])
        ends[b["end"]] += 1
        reached[b["reached"]] += 1
        agg["max_key_repeat_sum"] += b["max_key_repeat"]
        per_ep.append({"seed": row["seed"], "success": succ, "end": b["end"], "steps": b["steps"],
                       "reached": b["reached"]})
    return {"condition": cond, "arm": arm, "agg": dict(agg), "retry": dict(retry), "kinds": dict(kinds),
            "ends": dict(ends), "reached": dict(reached), "seeds": seeds, "reuse_details": reuse_details,
            "per_episode": per_ep}


# ---------------------------------------------------------------- metrics

def ratio(n, d):
    return None if not d else n / d


def metrics(r):
    a, t = collections.Counter(r["agg"]), collections.Counter(r["retry"])
    n = a["episodes"]
    return {
        "episodes": n,
        "success": a["success"] / n,
        "utility": a["utility"] / n,
        "work_per_success": ratio(a["work_units"], a["success"]),
        "correct_reuse_rate": ratio(a["correct"], a["opp"]),
        "correct_reuse_rate_successful_calls_only_block": ratio(a["correct_succcall"], a["opp"]),
        "correct_reuse_rate_applicable_pre": ratio(a["correct_pre"], a["opp_pre"]),
        "reuse_opportunities": a["opp"],
        "correct_reuse_windows": a["correct"],
        "invalid_reuse_rate": ratio(a["invalid"], a["audited"]),
        "stale_use_rate": ratio(a["stale"], a["audited"]),
        "audited_uses": a["audited"],
        "use_attempts": a["use_attempts"],
        "identical_retry_rate": ratio(t.get("ident", 0), t.get("rej", 0)),
        "identical_retry_per_attempt": ratio(t.get("ident", 0), t.get("primary_attempts", 0)),
        "identical_retry_all_kinds": ratio(t.get("ident_all", 0), t.get("fail_all", 0)),
        "identical_retry_excl_missing_dependency": ratio(t.get("ident_nomd", 0), t.get("rej_nomd", 0)),
        "rejections": t.get("rej", 0),
        "events": a["events"],
        "revision_quality": ratio(a["rev_good"], a["events"]),
        "over_revision_rate": ratio(a["rev_over"], a["events"]),
        "mean_steps": a["steps"] / n,
        "truncated": a["truncated"],
    }


def gmean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def group(M, arm, key, conds):
    return gmean([M[c][arm][key] for c in conds])


def paired_group(M, a, b, key, conds):
    both = [c for c in conds if M[c][a][key] is not None and M[c][b][key] is not None]
    if not both:
        return None, None, []
    return (sum(M[c][a][key] for c in both) / len(both), sum(M[c][b][key] for c in both) / len(both), both)


def rules(M, iid=IID, r3=R3_GROUP, r4=CONDITIONS, x1="rl", x2pair=("boot", "rl"), lineages=(0, 1, 2),
          other="rl"):
    L = [f"r{i}" for i in lineages]
    need = (len(lineages) * 2 + 2) // 3  # 2 of 3; 1 of 1
    out = {}
    # R1
    ref = group(M, "dep_reuse", "utility", iid)
    per = {}
    for l in L:
        arm = f"x1-{x1}-{l}"
        s, u = group(M, arm, "success", iid), group(M, arm, "utility", iid)
        per[l] = {"iid_success": s, "iid_utility": u, "success_pass": s is not None and s >= 0.95}
    mean_u = gmean([per[l]["iid_utility"] for l in L])
    npass = sum(per[l]["success_pass"] for l in L)
    out["R1"] = {"per_lineage": per, "dep_reuse_iid_utility": ref, "x1_lineage_mean_utility": mean_u,
                 "utility_pass": mean_u >= ref - 0.02, "success_passes": npass,
                 "supported": npass >= need and mean_u >= ref - 0.02}
    # R2
    per = {}
    b, r = x2pair
    for l in L:
        ab, ar = f"x2-{b}-{l}", f"x2-{r}-{l}"
        cr_r, cr_b, both = paired_group(M, ar, ab, "correct_reuse_rate", iid)
        gain = None if cr_r is None else cr_r - cr_b
        w_b, w_r = group(M, ab, "work_per_success", iid), group(M, ar, "work_per_success", iid)
        if w_r is None:
            wpass = False
        elif w_b is None:
            wpass = True
        else:
            wpass = w_r < w_b
        s_b, s_r = group(M, ab, "success", iid), group(M, ar, "success", iid)
        spass = s_r >= s_b - 0.02
        per[l] = {"reuse_boot": cr_b, "reuse_rl": cr_r, "gain": gain, "wps_boot": w_b, "wps_rl": w_r,
                  "success_boot": s_b, "success_rl": s_r, "gain_pass": gain is not None and gain >= 0.25,
                  "work_pass": wpass, "success_pass": spass}
        per[l]["pass"] = per[l]["gain_pass"] and wpass and spass
    out["R2"] = {"per_lineage": per, "supported": sum(per[l]["pass"] for l in L) >= need}
    # R3
    per = {}
    for l in L:
        a3, a1 = f"x3-{other}-{l}", f"x1-{x1}-{l}"
        i3, i1, _ = paired_group(M, a3, a1, "invalid_reuse_rate", r3)
        s3, s1, _ = paired_group(M, a3, a1, "stale_use_rate", r3)
        di = None if i3 is None else i3 - i1
        ds = None if s3 is None else s3 - s1
        p = (di is not None and di >= 0.05) or (ds is not None and ds >= 0.05)
        per[l] = {"invalid_x3": i3, "invalid_x1": i1, "invalid_diff": di, "stale_x3": s3, "stale_x1": s1,
                  "stale_diff": ds, "pass": p}
    out["R3"] = {"per_lineage": per, "supported": sum(per[l]["pass"] for l in L) >= need}
    # R4
    per = {}
    for l in L:
        a4, a1 = f"x4-{other}-{l}", f"x1-{x1}-{l}"
        r4v, r1v, _ = paired_group(M, a4, a1, "identical_retry_rate", r4)
        d = None if r4v is None else r4v - r1v
        s4, s1 = group(M, a4, "success", r4), group(M, a1, "success", r4)
        per[l] = {"retry_x4": r4v, "retry_x1": r1v, "retry_diff": d, "retry_pass": d is not None and d >= 0.05,
                  "success_x4": s4, "success_x1": s1, "success_gap_x1_minus_x4": s1 - s4}
    gap = gmean([per[l]["success_gap_x1_minus_x4"] for l in L])
    npass = sum(per[l]["retry_pass"] for l in L)
    out["R4"] = {"per_lineage": per, "retry_passes": npass, "lineage_mean_success_gap": gap,
                 "retry_clause": npass >= need, "success_clause": gap >= 0.02,
                 "supported": npass >= need or gap >= 0.02}
    # R5
    per = {}
    for l in L:
        arm = f"x1-{x1}-{l}"
        s = group(M, arm, "success", iid)
        h = M["heldout_deadline_moved"][arm]["success"]
        g = M["larger_s4"][arm]["success"]
        ratio_h = None if not s else h / s
        per[l] = {"iid_success": s, "heldout_success": h, "ratio": ratio_h,
                  "larger_ratio": None if not s else g / s, "pass": ratio_h is not None and ratio_h >= 0.9}
    out["R5"] = {"per_lineage": per, "supported": sum(per[l]["pass"] for l in L) >= need}
    return out


def per_condition_readings(M):
    res = {}
    for c in IID:
        rr = rules(M, iid=[c])
        res.setdefault("R1", {})[c] = rr["R1"]["supported"]
        res.setdefault("R2", {})[c] = rr["R2"]["supported"]
        res.setdefault("R5", {})[c] = rr["R5"]["supported"]
    for c in R3_GROUP:
        res.setdefault("R3", {})[c] = rules(M, r3=[c])["R3"]["supported"]
    for c in CONDITIONS:
        res.setdefault("R4", {})[c] = rules(M, r4=[c])["R4"]["supported"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sealed")
    ap.add_argument("--output", required=True)
    ap.add_argument("--details", help="optional path for per-episode/behaviour detail JSON")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    paths = []
    for c in CONDITIONS:
        d = os.path.join(args.sealed, c)
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".jsonl.gz") and f != "worlds.jsonl.gz"]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        raw = list(ex.map(process_file, paths))
    M = collections.defaultdict(dict)
    for r in raw:
        M[r["condition"]][r["arm"]] = metrics(r)
    registered = rules(M)
    readings = per_condition_readings(M)
    confound = {
        "bootstrap_endpoints": {k: rules(M, x1="boot", other="boot")[k] for k in ("R3", "R4")},
        "r1_only_rl": {k: rules(M, lineages=(1,))[k] for k in ("R3", "R4")},
        "r1_only_boot": {k: rules(M, x1="boot", other="boot", lineages=(1,))[k] for k in ("R3", "R4")},
    }
    seeds_ok = {}
    for c in CONDITIONS:
        ss = {r["arm"]: r["seeds"] for r in raw if r["condition"] == c}
        ref = next(iter(ss.values()))
        seeds_ok[c] = {"identical_across_arms": all(v == ref for v in ss.values()), "n": len(ref),
                       "min": min(ref), "max": max(ref), "distinct": len(set(ref))}
    out = {"version": "p1-independent-audit-v1", "metrics": M, "verdicts": registered,
           "per_condition_readings": readings, "confound_readings_not_registered": confound, "seeds": seeds_ok}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    if args.details:
        det = {f"{r['condition']}/{r['arm']}": {k: r[k] for k in ("kinds", "ends", "reached", "reuse_details",
                                                                  "per_episode", "retry")} for r in raw}
        with open(args.details, "w") as f:
            json.dump(det, f)
    for k, v in registered.items():
        print(k, "SUPPORTED" if v["supported"] else "not supported")


if __name__ == "__main__":
    main()
