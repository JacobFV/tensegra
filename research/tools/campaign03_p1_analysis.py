"""extended-03 P1 analysis: registered metrics and rules R1-R5 from a sealed evaluation.

Definitions are registered in research/campaigns/extended-03/protocol-P1-metrics.md
(before any P1 result existed). This tool reads only the evaluator output
directory (summary.json, <condition>/worlds.jsonl.gz, <condition>/<arm>.jsonl.gz),
recomputes every metric from raw per-episode rows, and writes JSON + markdown.
Standard library only (no torch); deterministic.

    python research/tools/campaign03_p1_analysis.py <eval-dir> --output <dir> [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re

ANALYSIS_VERSION = "p1-analysis-v1"
CONDITIONS = ("iid_f0", "iid_f2", "noevent_f2", "events_train_kinds_p1", "heldout_deadline_moved", "foreign4",
              "larger_s4", "work_price_x4")
GROUPS = {
    "iid": ("iid_f0", "iid_f2"),
    # R3: "the heavier-foreign-records and event conditions" (the two p_event=1 conditions).
    "r3": ("foreign4", "events_train_kinds_p1", "heldout_deadline_moved"),
    # R4: attempt memory can matter wherever a rejection can occur: every sealed condition.
    "all": CONDITIONS,
}
HELDOUT, LARGER = "heldout_deadline_moved", "larger_s4"
ARMS = ("x1", "x2", "x3", "x4")
LINEAGES = (0, 1, 2)
REFERENCES = ("dep_greedy", "dep_recompute", "dep_reuse", "dep_naive_reuse", "dep_reuse_norevise")
NEED_PRIMITIVE = {"select": "constrained_subset", "assign": "csp", "route": "shortest_path"}
PRIMARY_RETRY_KINDS = ("commit_pending", "commit_assignment", "use_return", "move")
ALL_ATTEMPT_KINDS = ("commit_pending", "commit_assignment", "use_return", "move", "call", "uncommit", "verify")
NAME_PATTERN = re.compile(r"^p1-(boot|rl)-(x[1-4])-r(\d+)$")
THRESHOLDS = {"r1_success": 0.95, "r1_utility_margin": 0.02, "r2_reuse_gain": 0.25, "r2_success_margin": 0.02,
              "r3_gap": 0.05, "r4_retry_gap": 0.05, "r4_success_gap": 0.02, "r5_ratio": 0.9, "majority": 2}


# ---------------------------------------------------------------------------
# Per-episode reductions (raw row -> numerators/denominators)
# ---------------------------------------------------------------------------

def _reuse_windows(steps):
    """Stage windows keyed by (need, stage-open step); see metrics doc section 3."""
    windows = {}
    for s in steps:
        need = s.get("need")
        if need is None:
            continue
        w = windows.setdefault((need, s["open"]), {"need": need, "opp": False, "opp_pre": False, "calls": 0,
                                                     "correct": False, "correct_pre": False, "resolution": None})
        if s.get("reusable"):
            w["opp"] = True
        if s.get("applicable_pre"):
            w["opp_pre"] = True
        use = s.get("use")
        if s["kind"] == "use_return" and use and use.get("as") == need and w["calls"] == 0:
            if use.get("reusable") and s.get("outcome") == "success" and not w["correct"]:
                w["correct"] = True
                w["resolution"] = w["resolution"] or "reused"
            if use.get("handle") in (s.get("applicable_pre") or ()):
                w["correct_pre"] = True
        if s["kind"] == "call" and s.get("call_primitive") == NEED_PRIMITIVE[need]:
            if w["resolution"] is None:
                w["resolution"] = "recomputed"
            w["calls"] += 1
    return list(windows.values())


def _retries(attempts, kinds, failure):
    """(rejections, identically_retried_rejections, attempts, identical_retry_attempts)."""
    rejections = retried = total = identical = 0
    last = {}
    rows = [a for a in attempts if a["action_kind"] in kinds]
    for i, a in enumerate(rows):
        total += 1
        prev = last.get(a["action_key"])
        if prev is not None and failure(prev) and prev["dependency_versions_at_attempt"] == a["dependency_versions_at_attempt"]:
            identical += 1
        last[a["action_key"]] = a
        if failure(a):
            rejections += 1
            nxt = next((b for b in rows[i + 1:] if b["action_key"] == a["action_key"]), None)
            retried += int(nxt is not None and nxt["dependency_versions_at_attempt"] == a["dependency_versions_at_attempt"])
    return rejections, retried, total, identical


def episode_metrics(row, prices=None):
    """Numerators/denominators for one raw evaluator row (learned or reference)."""
    o = row["outcome"]
    m = {"seed": row["seed"], "episodes": 1, "success": int(bool(o["verified_success"])), "utility": float(o["utility"]),
         "cost": float(o["cost"]), "work_units": int(o.get("work_units", 0)), "steps": int(o.get("steps", 0)),
         "truncated": int(bool(row.get("truncated", False))), "calls": int(o.get("calls", 0))}
    if prices is not None:
        m.update(cost_action=o["steps"] * prices["action_price"],
                 cost_observation=o["observations"] * prices["observation_price"],
                 cost_travel=o["travel_distance"] * prices["travel_price"],
                 cost_work=o["work_units"] * prices["work_price"],
                 cost_compute=o["compute_units"] * prices["compute_price"])
    uses = o.get("reuse_audit", [])
    history = o.get("history", [])
    m.update(use_attempts=sum(h["action"]["kind"] == "use_return" for h in history), audited_uses=len(uses),
             invalid_uses=sum(not u["applicable_hidden"] for u in uses),
             stale_uses=sum(not u["dependency_match_hidden"] for u in uses),
             invalid_foreign_uses=sum(u["foreign"] and not u["applicable_hidden"] for u in uses),
             invalid_own_uses=sum(not u["foreign"] and not u["applicable_hidden"] for u in uses),
             invalid_use_episode=int(any(not u["applicable_hidden"] for u in uses)),
             stale_use_episode=int(any(not u["dependency_match_hidden"] for u in uses)))
    audit = o.get("p1_audit")
    if audit is None:
        return m
    steps, attempts = audit["steps"], audit["attempts"]
    if len(steps) != len(history):
        raise ValueError("P1 audit and history lengths differ")
    windows = _reuse_windows(steps)
    m["audit"] = 1
    m["reuse_opportunities"] = sum(w["opp"] for w in windows)
    m["reuse_correct"] = sum(w["opp"] and w["correct"] for w in windows)
    m["reuse_recomputed"] = sum(w["opp"] and not w["correct"] and w["resolution"] == "recomputed" for w in windows)
    m["reuse_opportunities_applicable_pre"] = sum(w["opp_pre"] for w in windows)
    m["reuse_correct_applicable_pre"] = sum(w["opp_pre"] and w["correct_pre"] for w in windows)
    for need in NEED_PRIMITIVE:
        m[f"reuse_opportunities_{need}"] = sum(w["opp"] and w["need"] == need for w in windows)
        m[f"reuse_correct_{need}"] = sum(w["opp"] and w["correct"] and w["need"] == need for w in windows)
    m["reuse_opportunity_episode"] = int(m["reuse_opportunities"] > 0)
    m["reuse_decisions"] = sum(bool(s.get("reusable")) for s in steps)
    rejected = lambda a: a["outcome_status"] == "rejected"
    nonsuccess = lambda a: a["outcome_status"] != "success"
    for label, kinds, failure in (("", PRIMARY_RETRY_KINDS, rejected), ("_all", ALL_ATTEMPT_KINDS, nonsuccess)):
        rej, ret, tot, ident = _retries(attempts, kinds, failure)
        m[f"retry_rejections{label}"], m[f"retry_identical{label}"] = rej, ret
        m[f"retry_attempts{label}"], m[f"retry_identical_attempts{label}"] = tot, ident
    no_md = [a for a in attempts if a.get("reason") != "missing_dependency"]
    rej, ret, _, _ = _retries(no_md, PRIMARY_RETRY_KINDS, rejected)
    m["retry_rejections_no_missing_dependency"], m["retry_identical_no_missing_dependency"] = rej, ret
    for kind in PRIMARY_RETRY_KINDS:
        rej, ret, _, _ = _retries(attempts, (kind,), rejected)
        m[f"retry_rejections_{kind}"], m[f"retry_identical_{kind}"] = rej, ret
    event = audit.get("event")
    m["event_episodes"] = int(event is not None)
    if event is not None:
        valid = {p for p in event["kept"] if event["kept_completable"].get(p)}
        over = False
        for s, h in zip(steps, history):
            if s["step"] <= event["step"]:
                continue
            target = (h["action"].get("arguments") or {}).get("target")
            if s["kind"] == "uncommit" and s.get("outcome") == "success" and (
                    (target == "select" and "selection" in valid) or (target == "assign" and "assignment" in valid)):
                over = True
            if s["kind"] == "call" and ((s.get("call_primitive") == "constrained_subset" and "selection" in valid)
                                        or (s.get("call_primitive") == "csp" and "assignment" in valid)):
                over = True
        m["event_success"] = m["success"]
        m["over_revision"] = int(over)
        m["revision_good"] = int(m["success"] and not over)
        m["post_event_work"] = m["work_units"] - event["work_at_event"]
        m["event_revoked"] = int(bool(event["revoked"]))
    return m


# ---------------------------------------------------------------------------
# Condition-level aggregation (pooled within a condition)
# ---------------------------------------------------------------------------

def _ratio(a, b):
    return a / b if b else None


RATES = {  # name: (numerator, denominator); pooled ratio of sums over the condition's episodes
    "success": ("success", "episodes"),
    "utility": ("utility", "episodes"),
    "cost": ("cost", "episodes"),
    "work_per_episode": ("work_units", "episodes"),
    "work_per_success": ("work_units", "success"),
    "steps": ("steps", "episodes"),
    "truncation": ("truncated", "episodes"),
    "correct_reuse_rate": ("reuse_correct", "reuse_opportunities"),
    "correct_reuse_rate_applicable_pre": ("reuse_correct_applicable_pre", "reuse_opportunities_applicable_pre"),
    "reuse_recompute_rate": ("reuse_recomputed", "reuse_opportunities"),
    "correct_reuse_rate_select": ("reuse_correct_select", "reuse_opportunities_select"),
    "correct_reuse_rate_assign": ("reuse_correct_assign", "reuse_opportunities_assign"),
    "correct_reuse_rate_route": ("reuse_correct_route", "reuse_opportunities_route"),
    "reuse_opportunities_per_episode": ("reuse_opportunities", "episodes"),
    "invalid_reuse_rate": ("invalid_uses", "audited_uses"),
    "stale_use_rate": ("stale_uses", "audited_uses"),
    "invalid_reuse_episode_rate": ("invalid_use_episode", "episodes"),
    "stale_use_episode_rate": ("stale_use_episode", "episodes"),
    "audited_uses_per_episode": ("audited_uses", "episodes"),
    "identical_retry_rate": ("retry_identical", "retry_rejections"),
    "identical_retry_per_attempt": ("retry_identical_attempts", "retry_attempts"),
    "identical_retry_rate_all_kinds": ("retry_identical_all", "retry_rejections_all"),
    "identical_retry_rate_no_missing_dependency": ("retry_identical_no_missing_dependency",
                                                   "retry_rejections_no_missing_dependency"),
    "revision_quality": ("revision_good", "event_episodes"),
    "over_revision_rate": ("over_revision", "event_episodes"),
    "event_success": ("event_success", "event_episodes"),
    "post_event_work": ("post_event_work", "event_episodes"),
}
for _kind in PRIMARY_RETRY_KINDS:
    RATES[f"identical_retry_rate_{_kind}"] = (f"retry_identical_{_kind}", f"retry_rejections_{_kind}")
for _c in ("action", "observation", "travel", "work", "compute"):
    RATES[f"cost_{_c}"] = (f"cost_{_c}", "episodes")
SUPPORTS = ("episodes", "success", "reuse_opportunities", "audited_uses", "retry_rejections", "event_episodes", "audit")


def aggregate(metrics):
    totals = defaultdict(float)
    for m in metrics:
        for k, v in m.items():
            if k != "seed":
                totals[k] += v
    out = {name: _ratio(totals.get(a, 0.0), totals.get(b, 0.0)) for name, (a, b) in RATES.items()
           if b in totals or a in totals}
    out["supports"] = {k: totals.get(k, 0) for k in SUPPORTS}
    return out


def bootstrap_ci(a_rows, b_rows, metric, group, resamples, seed, combine):
    """Percentile 95% interval of combine(group statistic of a, of b), resampling worlds
    (seeds) with replacement within each condition. a and b were evaluated on the same
    sealed worlds, so one index draw is applied to both (paired)."""
    if resamples <= 0 or not a_rows or not b_rows:
        return None
    num, den = RATES[metric]
    conds = [c for c in group if a_rows.get(c) and b_rows.get(c)]
    for c in conds:
        if [m["seed"] for m in a_rows[c]] != [m["seed"] for m in b_rows[c]]:
            raise ValueError(f"unpaired worlds in {c}")
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        va, vb = {}, {}
        for c in conds:
            ra, rb = a_rows[c], b_rows[c]
            idx = [rng.randrange(len(ra)) for _ in ra]
            va[c] = _ratio(sum(ra[j].get(num, 0) for j in idx), sum(ra[j].get(den, 0) for j in idx))
            vb[c] = _ratio(sum(rb[j].get(num, 0) for j in idx), sum(rb[j].get(den, 0) for j in idx))
        common = {c for c in conds if va[c] is not None and vb[c] is not None}
        v = combine(group_mean(va, conds, common)[0], group_mean(vb, conds, common)[0])
        if v is not None and math.isfinite(v):
            values.append(v)
    if not values:
        return None
    values.sort()
    return [values[int(0.025 * (len(values) - 1))], values[int(math.ceil(0.975 * (len(values) - 1)))]]


def group_mean(values, group, require=None):
    """Equal-weighted mean over the group's conditions with a defined value. `require`
    restricts to conditions defined for every paired arm (same condition set)."""
    used = [c for c in group if values.get(c) is not None and (require is None or c in require)]
    if not used:
        return None, used
    return sum(values[c] for c in used) / len(used), used


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arm_identity(result):
    """(kind, key) where key is ('ref', name) or (endpoint, arm, lineage)."""
    if result["kind"] == "supplied_schedule":
        return ("ref", result["arm"].removeprefix("reference-"))
    binding = result.get("checkpoint_binding") or {}
    if {"endpoint", "arm", "lineage"} <= set(binding):
        return (binding["endpoint"], binding["arm"], int(binding["lineage"]))
    m = NAME_PATTERN.match(result["arm"])
    if m:
        return (m.group(1), m.group(2), int(m.group(3)))
    return ("other", result["arm"])


def load(eval_dir: Path, allow_partial=False, verify=True):
    path = eval_dir / "summary.json"
    if not path.exists():
        if not allow_partial:
            raise FileNotFoundError("summary.json missing (evaluation incomplete); use --allow-partial for smoke only")
        path = eval_dir / "summary.partial.json"
    summary = json.loads(path.read_text())
    prices = {}
    episodes = {}  # identity -> condition -> list of per-episode metric dicts
    for result in summary["results"]:
        c = result["condition"]
        if c not in prices:
            prices[c] = {w["seed"]: {k: w["spec"][k] for k in ("action_price", "observation_price", "travel_price",
                                                               "work_price", "compute_price")}
                         for w in read_rows(eval_dir / c / "worlds.jsonl.gz")}
        artifact = eval_dir / result["artifact"]
        if verify and sha256(artifact) != result["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact}")
        rows = read_rows(artifact)
        identity = arm_identity(result)
        episodes.setdefault(identity, {})[c] = [episode_metrics(r, prices[c][r["seed"]]) for r in rows]
    return summary, episodes


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _count(flags):
    return sum(1 for f in flags if f)


def rules(table, episodes, resamples=0, seed=0):
    """table[identity][condition] -> aggregate dict. Returns rule verdicts with per-lineage values,
    the aggregate reading and per-condition readings."""
    T = THRESHOLDS

    def val(identity, metric, c):
        return (table.get(identity, {}).get(c) or {}).get(metric)

    def per_c(identity, metric, group):
        return {c: val(identity, metric, c) for c in group}

    def avail(identity):
        return identity in table

    out = {}

    # R1 -------------------------------------------------------------------
    def r1(group):
        lineages, flags, utils = [], [], []
        for r in LINEAGES:
            ident = ("rl", "x1", r)
            if not avail(ident):
                lineages.append({"lineage": r, "missing": True})
                flags.append(False)
                continue
            s, used = group_mean(per_c(ident, "success", group), group)
            u, _ = group_mean(per_c(ident, "utility", group), group)
            flags.append(s is not None and s >= T["r1_success"])
            utils.append(u)
            lineages.append({"lineage": r, "iid_success": s, "iid_utility": u, "conditions": used,
                             "success_pass": flags[-1]})
        ref, _ = group_mean(per_c(("ref", "dep_reuse"), "utility", group), group)
        mean_u = sum(utils) / len(utils) if utils and None not in utils and len(utils) == len(LINEAGES) else None
        util_pass = mean_u is not None and ref is not None and mean_u >= ref - T["r1_utility_margin"]
        return {"lineages": lineages, "lineage_mean_utility": mean_u, "dep_reuse_utility": ref,
                "success_lineages_passing": _count(flags), "utility_pass": util_pass,
                "supported": _count(flags) >= T["majority"] and util_pass}
    out["R1"] = {"conditions": GROUPS["iid"], "aggregate": r1(GROUPS["iid"]),
                 "per_condition": {c: r1((c,)) for c in GROUPS["iid"]}}

    # R2 -------------------------------------------------------------------
    def r2(group, with_ci=False):
        lineages, flags = [], []
        for r in LINEAGES:
            b, l = ("boot", "x2", r), ("rl", "x2", r)
            if not (avail(b) and avail(l)):
                lineages.append({"lineage": r, "missing": True})
                flags.append(False)
                continue
            cr_b, cr_l = per_c(b, "correct_reuse_rate", group), per_c(l, "correct_reuse_rate", group)
            common = {c for c in group if cr_b[c] is not None and cr_l[c] is not None}
            rb, used = group_mean(cr_b, group, common)
            rl_, _ = group_mean(cr_l, group, common)
            gain = None if rb is None or rl_ is None else rl_ - rb
            wb, _ = group_mean(per_c(b, "work_per_success", group), group)
            wl, _ = group_mean(per_c(l, "work_per_success", group), group)
            wb_all = all(val(b, "work_per_success", c) is not None for c in group)
            wl_all = all(val(l, "work_per_success", c) is not None for c in group)
            work_dec = wl_all and (not wb_all or wl < wb)
            sb, _ = group_mean(per_c(b, "success", group), group)
            sl, _ = group_mean(per_c(l, "success", group), group)
            succ_ok = sb is not None and sl is not None and sl >= sb - T["r2_success_margin"]
            flag = gain is not None and gain >= T["r2_reuse_gain"] and work_dec and succ_ok
            flags.append(flag)
            row = {"lineage": r, "reuse_boot": rb, "reuse_rl": rl_, "reuse_gain": gain, "reuse_conditions": used,
                   "work_per_success_boot": wb if wb_all else None, "work_per_success_rl": wl if wl_all else None,
                   "work_decreases": work_dec, "success_boot": sb, "success_rl": sl, "success_ok": succ_ok,
                   "pass": flag}
            if with_ci and resamples:
                row["reuse_gain_ci95"] = bootstrap_ci(episodes.get(l), episodes.get(b), "correct_reuse_rate",
                                                      tuple(sorted(common, key=group.index)), resamples,
                                                      seed + r, lambda x, y: None if x is None or y is None else x - y)
            lineages.append(row)
        return {"lineages": lineages, "lineages_passing": _count(flags), "supported": _count(flags) >= T["majority"]}
    out["R2"] = {"conditions": GROUPS["iid"], "aggregate": r2(GROUPS["iid"], True),
                 "per_condition": {c: r2((c,)) for c in GROUPS["iid"]}}

    # R3 -------------------------------------------------------------------
    def paired(arm, metric, group, gap, with_ci=False, ci_seed=0):
        pairs = []
        for r in LINEAGES:
            a, x1 = ("rl", arm, r), ("rl", "x1", r)
            if not (avail(a) and avail(x1)):
                pairs.append({"lineage": r, "missing": True, "diff": None, "pass": False})
                continue
            va, vx = per_c(a, metric, group), per_c(x1, metric, group)
            common = {c for c in group if va[c] is not None and vx[c] is not None}
            ma, used = group_mean(va, group, common)
            mx, _ = group_mean(vx, group, common)
            diff = None if ma is None or mx is None else ma - mx
            row = {"lineage": r, arm: ma, "x1": mx, "diff": diff, "conditions": used,
                   "pass": diff is not None and diff >= gap}
            if with_ci and resamples and common:
                row["diff_ci95"] = bootstrap_ci(episodes.get(a), episodes.get(x1), metric,
                                                tuple(c for c in group if c in common), resamples, ci_seed + r,
                                                lambda p, q: None if p is None or q is None else p - q)
            pairs.append(row)
        return pairs

    def r3(group, with_ci=False):
        inv = paired("x3", "invalid_reuse_rate", group, T["r3_gap"], with_ci, seed + 100)
        stale = paired("x3", "stale_use_rate", group, T["r3_gap"], with_ci, seed + 200)
        flags = [i["pass"] or s["pass"] for i, s in zip(inv, stale)]
        return {"invalid_reuse": inv, "stale_use": stale, "pairs_passing": _count(flags),
                "supported": _count(flags) >= T["majority"]}
    out["R3"] = {"conditions": GROUPS["r3"], "aggregate": r3(GROUPS["r3"], True),
                 "per_condition": {c: r3((c,)) for c in GROUPS["r3"]}}

    # R4 -------------------------------------------------------------------
    def r4(group, with_ci=False):
        retry = paired("x4", "identical_retry_rate", group, T["r4_retry_gap"], with_ci, seed + 300)
        retry_pass = _count(p["pass"] for p in retry) >= T["majority"]
        s1, s4, per_pair = [], [], []
        for r in LINEAGES:
            a, x1 = ("rl", "x4", r), ("rl", "x1", r)
            if not (avail(a) and avail(x1)):
                per_pair.append({"lineage": r, "missing": True})
                continue
            m4, _ = group_mean(per_c(a, "success", group), group)
            m1, _ = group_mean(per_c(x1, "success", group), group)
            s1.append(m1)
            s4.append(m4)
            per_pair.append({"lineage": r, "x4": m4, "x1": m1, "diff": None if m4 is None or m1 is None else m1 - m4})
        complete = len(s1) == len(LINEAGES) and None not in s1 and None not in s4
        gap = (sum(s1) - sum(s4)) / len(LINEAGES) if complete else None
        success_pass = gap is not None and gap >= T["r4_success_gap"]
        return {"identical_retry": retry, "retry_pairs_passing": _count(p["pass"] for p in retry),
                "retry_pass": retry_pass, "success_by_pair": per_pair, "lineage_mean_success_gap_x1_minus_x4": gap,
                "success_pass": success_pass, "supported": retry_pass or success_pass,
                "secondary_all_kinds": paired("x4", "identical_retry_rate_all_kinds", group, T["r4_retry_gap"]),
                "secondary_per_attempt": paired("x4", "identical_retry_per_attempt", group, T["r4_retry_gap"]),
                "secondary_no_missing_dependency": paired("x4", "identical_retry_rate_no_missing_dependency", group,
                                                          T["r4_retry_gap"])}
    out["R4"] = {"conditions": GROUPS["all"], "aggregate": r4(GROUPS["all"], True),
                 "per_condition": {c: r4((c,)) for c in GROUPS["all"]}}

    # R5 -------------------------------------------------------------------
    def r5(group, target=HELDOUT):
        lineages, flags = [], []
        for r in LINEAGES:
            ident = ("rl", "x1", r)
            if not avail(ident):
                lineages.append({"lineage": r, "missing": True})
                flags.append(False)
                continue
            iid, _ = group_mean(per_c(ident, "success", group), group)
            held = val(ident, "success", target)
            ratio = None if iid in (None, 0) or held is None else held / iid
            flags.append(ratio is not None and ratio >= T["r5_ratio"])
            lineages.append({"lineage": r, "iid_success": iid, "target_success": held, "ratio": ratio,
                             "pass": flags[-1]})
        return {"target": target, "lineages": lineages, "lineages_passing": _count(flags),
                "supported": _count(flags) >= T["majority"]}
    out["R5"] = {"conditions": GROUPS["iid"] + (HELDOUT,), "aggregate": r5(GROUPS["iid"]),
                 "per_condition": {c: r5((c,)) for c in GROUPS["iid"]},
                 "secondary_larger_size": r5(GROUPS["iid"], LARGER)}
    for rule in out.values():
        agg = rule["aggregate"]["supported"]
        rule["supported"] = agg
        rule["per_condition_differs"] = {c: v["supported"] for c, v in rule["per_condition"].items()
                                         if v["supported"] != agg}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def label(identity):
    if identity[0] == "ref":
        return identity[1]
    if identity[0] == "other":
        return identity[1]
    return f"{identity[1]}-{identity[0]}-r{identity[2]}"


def analyze(eval_dir: Path, resamples=1000, seed=20260926, allow_partial=False):
    summary, episodes = load(eval_dir, allow_partial)
    table = {ident: {c: aggregate(ms) for c, ms in by_c.items()} for ident, by_c in episodes.items()}
    conditions = [c for c in CONDITIONS if any(c in v for v in table.values())]
    conditions += sorted({c for v in table.values() for c in v} - set(conditions))
    verdicts = rules(table, episodes, resamples, seed)
    lineage_means = {}
    for endpoint in ("boot", "rl"):
        for arm in ARMS:
            ids = [(endpoint, arm, r) for r in LINEAGES if (endpoint, arm, r) in table]
            if not ids:
                continue
            lineage_means[f"{arm}-{endpoint}"] = {
                c: {k: (sum(table[i][c][k] for i in ids) / len(ids)
                        if all(c in table[i] and table[i][c].get(k) is not None for i in ids) else None)
                    for k in RATES} | {"lineages": len(ids)}
                for c in conditions if all(c in table[i] for i in ids)}
    flags = []
    for ident, by_c in table.items():
        for c, agg in by_c.items():
            if agg["supports"]["audit"] < agg["supports"]["episodes"]:
                flags.append(f"{label(ident)}/{c}: P1 audit missing on some rows")
            if agg.get("truncation"):
                flags.append(f"{label(ident)}/{c}: truncated episodes {agg['truncation']:.3f}")
    missing = [f"{e}-{a}-r{r}" for e in ("boot", "rl") for a in ARMS for r in LINEAGES if (e, a, r) not in table]
    missing += [f"reference {x}" for x in REFERENCES if ("ref", x) not in table]
    return {"analysis_version": ANALYSIS_VERSION, "evaluation_dir": str(eval_dir),
            "evaluation_config_sha256": summary.get("config_sha256"), "sources": summary.get("sources"),
            "checkpoints": summary.get("checkpoints"), "partial": not (eval_dir / "summary.json").exists(),
            "thresholds": THRESHOLDS, "groups": {k: list(v) for k, v in GROUPS.items()},
            "bootstrap": {"resamples": resamples, "seed": seed, "unit": "world (seed) within condition, paired across arms",
                          "scope": "descriptive; verdicts use point estimates"},
            "rules": verdicts, "conditions": conditions,
            "table": {label(i): v for i, v in sorted(table.items(), key=lambda kv: label(kv[0]))},
            "lineage_means": lineage_means, "missing": missing, "flags": flags}


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    return str(x)


KEY_METRICS = ("success", "utility", "work_per_success", "correct_reuse_rate", "invalid_reuse_rate", "stale_use_rate",
               "identical_retry_rate", "revision_quality")


def markdown(result):
    R = result["rules"]
    lines = [f"# P1 analysis ({result['analysis_version']})", "",
             f"Evaluation: `{result['evaluation_dir']}` (config sha256 `{result['evaluation_config_sha256']}`)"
             + (" **PARTIAL**" if result["partial"] else ""), "",
             "Definitions: research/campaigns/extended-03/protocol-P1-metrics.md. Verdicts use point estimates; "
             "bootstrap intervals are descriptive.", "", "## Verdicts", "",
             "| Rule | Aggregate | Per-condition readings that differ |", "|---|---|---|"]
    for name, rule in R.items():
        diff = ", ".join(f"{c}: {_f(v)}" for c, v in rule["per_condition_differs"].items()) or "none"
        lines.append(f"| {name} | {'SUPPORTED' if rule['supported'] else 'not supported'} | {diff} |")
    a = R["R1"]["aggregate"]
    lines += ["", "## R1 (acquisition; X1 RL, IID group)", "",
              f"dep_reuse IID utility {_f(a['dep_reuse_utility'])}; X1 lineage-mean utility "
              f"{_f(a['lineage_mean_utility'])}; utility criterion {_f(a['utility_pass'])}; "
              f"success >= 0.95 in {a['success_lineages_passing']}/3.", "",
              "| lineage | IID success | IID utility | success pass |", "|---|---|---|---|"]
    lines += [f"| r{x['lineage']} | {_f(x.get('iid_success'))} | {_f(x.get('iid_utility'))} | {_f(x.get('success_pass'))} |"
              for x in a["lineages"]]
    a = R["R2"]["aggregate"]
    lines += ["", "## R2 (emergent reuse; X2 bootstrap vs RL, IID group)", "",
              "| lineage | reuse boot | reuse RL | gain [95% CI] | work/success boot | RL | success boot | RL | pass |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| r{x['lineage']} | {_f(x.get('reuse_boot'))} | {_f(x.get('reuse_rl'))} | {_f(x.get('reuse_gain'))} "
              f"{_f(x.get('reuse_gain_ci95'))} | {_f(x.get('work_per_success_boot'), 1)} | {_f(x.get('work_per_success_rl'), 1)} | "
              f"{_f(x.get('success_boot'))} | {_f(x.get('success_rl'))} | {_f(x.get('pass'))} |" for x in a["lineages"]]
    a = R["R3"]["aggregate"]
    lines += ["", f"## R3 (applicability input; X3 vs X1 RL on {', '.join(R['R3']['conditions'])})", "",
              "| lineage | invalid X3 | X1 | diff [95% CI] | stale X3 | X1 | diff [95% CI] |", "|---|---|---|---|---|---|---|"]
    for i, s in zip(a["invalid_reuse"], a["stale_use"]):
        lines.append(f"| r{i['lineage']} | {_f(i.get('x3'))} | {_f(i.get('x1'))} | {_f(i.get('diff'))} {_f(i.get('diff_ci95'))} | "
                     f"{_f(s.get('x3'))} | {_f(s.get('x1'))} | {_f(s.get('diff'))} {_f(s.get('diff_ci95'))} |")
    a = R["R4"]["aggregate"]
    lines += ["", "## R4 (attempt memory; X4 vs X1 RL, all conditions)", "",
              f"Retry criterion {_f(a['retry_pass'])} ({a['retry_pairs_passing']}/3 pairs); lineage-mean success gap "
              f"X1-X4 {_f(a['lineage_mean_success_gap_x1_minus_x4'])} (criterion {_f(a['success_pass'])}). "
              "F1: the retry comparison uses `relevant_dependencies`, which ignores requirement inspection.", "",
              "| lineage | identical-retry X4 | X1 | diff [95% CI] | all kinds diff | per-attempt diff |", "|---|---|---|---|---|---|"]
    for p, q, w in zip(a["identical_retry"], a["secondary_all_kinds"], a["secondary_per_attempt"]):
        lines.append(f"| r{p['lineage']} | {_f(p.get('x4'))} | {_f(p.get('x1'))} | {_f(p.get('diff'))} {_f(p.get('diff_ci95'))} | "
                     f"{_f(q.get('diff'))} | {_f(w.get('diff'))} |")
    a = R["R5"]["aggregate"]
    lines += ["", "## R5 (transfer; X1 RL)", "", "| lineage | IID success | held-out success | ratio | larger_s4 ratio |",
              "|---|---|---|---|---|"]
    for x, y in zip(a["lineages"], R["R5"]["secondary_larger_size"]["lineages"]):
        lines.append(f"| r{x['lineage']} | {_f(x.get('iid_success'))} | {_f(x.get('target_success'))} | {_f(x.get('ratio'))} | "
                     f"{_f(y.get('ratio'))} |")
    lines += ["", "## Per-condition readings (every arm/endpoint/lineage and reference)", ""]
    for c in result["conditions"]:
        lines += [f"### {c}", "", "| arm | " + " | ".join(KEY_METRICS) + " | reuse opp. | audited uses | rejections | events |",
                  "|---" * (len(KEY_METRICS) + 5) + "|"]
        for name, by_c in result["table"].items():
            if c not in by_c:
                continue
            agg = by_c[c]
            s = agg["supports"]
            lines.append(f"| {name} | " + " | ".join(_f(agg.get(k), 1 if k == "work_per_success" else 3) for k in KEY_METRICS)
                         + f" | {int(s['reuse_opportunities'])} | {int(s['audited_uses'])} | {int(s['retry_rejections'])} | "
                           f"{int(s['event_episodes'])} |")
        lines.append("")
    if result["missing"]:
        lines += ["## Missing arms", "", ", ".join(result["missing"]), ""]
    if result["flags"]:
        lines += ["## Flags", ""] + [f"- {x}" for x in result["flags"]] + [""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evaluation", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--allow-partial", action="store_true", help="smoke only: read summary.partial.json")
    a = p.parse_args()
    result = analyze(a.evaluation, a.bootstrap, a.seed, a.allow_partial)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "p1-analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (a.output / "p1-analysis.md").write_text(markdown(result))
    print(json.dumps({k: v["supported"] for k, v in result["rules"].items()}))


if __name__ == "__main__":
    main()
