"""Track C analysis summary across bases (extended-04 trackc.md §9; v2 revision 8).

    python research/tools/campaign04_c_analysis.py --eval results/c-eval-* \
        [--train results/c-train-*] [--labels results/c-labels-*] --output trackc-analysis.json

Reads campaign04_c_eval outputs (summary.json + per-arm row files; several jobs per base
are merged by (condition, arm, meta seed)), campaign04_c_train records and label stats.
Reports, per base lineage (never pooled across lineages for a claim):

- every arm x condition: success, utility (meta charged / not), cost, no-progress,
  intervention rate; the IID group {iid_f0, iid_f2} pooled at world level;
- paired (same worlds) utility differences of learned control (seed-mean per world and
  per seed) vs every fixed rule, the best fixed rule (by IID mean utility), D0, the
  matched-rate random arm and the threshold rule, with paired SE;
- the registered criterion on the IID group, split by base type (v2 rev. 8):
  loop-prone (P1 RL finals): learned - best fixed >= .02;
  competent (bootstraps): learned - best fixed >= -.01 AND lower cost or no-progress than D0;
  passes need >= 2/3 lineages per base type. F10 strata (P1 r1 = stable) are also shown;
- calibration (appraisal_only ECE; branch-evaluated Q MAE on controller-induced states);
- causal arms (zero/shuffle/clamp/retrained) vs learned (paired);
- CPU by stage from the tools' own records (job receipts are authoritative).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

FIXED = ("fixed:greedy", "fixed:sampled", "fixed:r_mask", "fixed:r_sample")
IID = ("iid_f0", "iid_f2")
LOOP_MARGIN, COMPETENT_TOLERANCE = 0.02, 0.01


def read_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def paired(a, b):
    """Mean and SE of per-world a - b (dicts world -> value on the same worlds)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = [a[k] - b[k] for k in keys]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1) if len(d) > 1 else 0.0
    se = math.sqrt(var / len(d))
    return {"n": len(d), "mean": m, "se": se, "ci95": [m - 1.96 * se, m + 1.96 * se]}


def load_eval(dirs):
    """base -> {"meta": summary fields, "results": [...], "rows": {(cond, arm, seed): {world: row}}}"""
    bases = {}
    for d in dirs:
        d = Path(d)
        s = json.loads((d / "summary.json").read_text())
        b = bases.setdefault(s["base"], {"base_type": s["base_type"], "f10_stratum": s["f10_stratum"],
                                         "results": [], "rows": {}, "jobs": []})
        b["jobs"].append({"dir": str(d), "cpu": s.get("resources", {}).get("parent_cpu_seconds")})
        for r in s["results"]:
            b["results"].append(r)
            b["rows"][(r["condition"], r["arm"], r["meta_seed"])] = {x["seed"]: x for x in read_rows(d / r["artifact"])}
    return bases


def world_values(rows_by_key, cond_list, arm, seeds, field="utility"):
    """{(cond, world): value} averaged over the given meta seeds (None = fixed arm)."""
    out = {}
    for cond in cond_list:
        per = [rows_by_key.get((cond, arm, s)) for s in seeds]
        if any(p is None for p in per):
            return None
        for w in per[0]:
            vals = [p[w][field] for p in per if w in p]
            out[(cond, w)] = sum(float(v) for v in vals) / len(vals)
    return out


def base_report(b, seeds):
    rows = b["rows"]
    conds = sorted({k[0] for k in rows})
    arms = sorted({k[1] for k in rows})
    rep = {"base_type": b["base_type"], "f10_stratum": b["f10_stratum"], "conditions": conds, "table": [],
           "iid": {}, "paired_iid": {}, "criterion": None}
    for r in b["results"]:
        rep["table"].append({k: r.get(k) for k in ("condition", "arm", "meta_seed", "examples", "success", "utility",
                                                    "utility_no_meta_charge", "utility_with_diagnostic_cpu_charged",
                                                    "cost", "no_progress_steps", "no_progress_episode_rate",
                                                    "intervention_rate", "u_counts")})
    if not all(c in conds for c in IID):
        return rep
    def arm_seeds(arm):
        return [None] if arm.startswith("fixed:") else [s for s in seeds if (IID[0], arm, s) in rows]
    means = {}
    for arm in arms:
        ss = arm_seeds(arm)
        if not ss:
            continue
        vals = {f: world_values(rows, IID, arm, ss, f) for f in ("utility", "success", "cost", "no_progress")}
        if vals["utility"] is None:
            continue
        n = len(vals["utility"])
        means[arm] = {f: sum(v.values()) / n for f, v in vals.items()} | {"worlds": n, "seeds": ss}
    rep["iid"] = means
    fixed = [a for a in FIXED if a in means]
    if not fixed or "learned" not in means:
        return rep
    best = max(fixed, key=lambda a: means[a]["utility"])
    learned = world_values(rows, IID, "learned", arm_seeds("learned"))
    comps = {a: world_values(rows, IID, a, arm_seeds(a)) for a in list(fixed) + ["random_matched", "threshold",
                                                                                 "appraisal_only"] if a in means}
    rep["paired_iid"] = {f"learned - {a}": paired(learned, v) for a, v in comps.items()}
    rep["paired_iid_per_seed"] = {
        str(s): {f"learned - {a}": paired(world_values(rows, IID, "learned", [s]), v) for a, v in comps.items()}
        for s in arm_seeds("learned")}
    for arm in arms:
        if arm.startswith(("zero:", "clamp:", "model:")) or arm == "shuffle_telemetry":
            ss = arm_seeds(arm)
            v = world_values(rows, IID, arm, ss) if ss else None
            if v:
                rep["paired_iid"][f"{arm} - learned(same seeds)"] = paired(v, world_values(rows, IID, "learned", ss))
    diff = rep["paired_iid"][f"learned - {best}"]["mean"]
    d0 = means.get("fixed:r_mask")
    if b["base_type"] == "loop_prone":
        ok = diff >= LOOP_MARGIN
        rule = f"learned - best fixed ({best}) >= {LOOP_MARGIN}"
    else:
        cheaper = d0 is not None and (means["learned"]["cost"] < d0["cost"]
                                      or means["learned"]["no_progress"] < d0["no_progress"])
        ok = diff >= -COMPETENT_TOLERANCE and cheaper
        rule = f"learned - best fixed ({best}) >= -{COMPETENT_TOLERANCE} and (cost or no-progress below D0)"
    rep["criterion"] = {"rule": rule, "best_fixed": best, "difference": diff, "pass": ok,
                        "paired": rep["paired_iid"][f"learned - {best}"]}
    return rep


def calibration(b):
    out = {}
    for r in b["results"]:
        key = f"{r['condition']}/{r['arm']}/s{r['meta_seed']}"
        if r["arm"] == "appraisal_only" and "calibration_steps" in r:
            c = r["calibration_steps"]
            out[key] = {"ece_steps": c["ece"], "ece_first_decision": c.get("ece_first_decision"),
                        "q0_mae": c["q0_mae"], "bins_used": c["bins_used"]}
        if "branch_eval" in r:
            out.setdefault(key, {})["branch_eval"] = r["branch_eval"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--train", nargs="*", default=[])
    p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    bases = load_eval(a.eval)
    report = {"bases": {}, "calibration": {}, "criterion_by_type": {}, "training": {}, "labels": {}, "cpu": {}}
    for name, b in sorted(bases.items()):
        report["bases"][name] = base_report(b, a.seeds)
        report["calibration"][name] = calibration(b)
        report["cpu"].setdefault("eval_parent_cpu", 0.0)
        report["cpu"]["eval_parent_cpu"] += sum(j["cpu"] or 0.0 for j in b["jobs"])
    by_type = defaultdict(list)
    for name, r in report["bases"].items():
        if r["criterion"] is not None:
            by_type[r["base_type"]].append((name, r["criterion"]["pass"]))
    for t, xs in by_type.items():
        report["criterion_by_type"][t] = {"lineages": dict(xs), "passes": sum(ok for _, ok in xs),
                                          "required": math.ceil(2 * len(xs) / 3),
                                          "pass": sum(ok for _, ok in xs) >= math.ceil(2 * len(xs) / 3)}
    for d in a.train:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            report["training"].setdefault(r["base"], {})[f.stem] = {
                "margin": r["registration"]["margin"], "tau": r["registration"]["tau"],
                "dev_gain": r["registration"]["margin_dev_gain"], "dev": {k: r["dev"][k] for k in (
                    "ece_success_steps", "q0_mae_steps", "adv_mae_points", "dev_points")},
                "parameters": r["parameters"], "actor_parameters": r["actor_parameters"], "cpu": r["process_cpu_seconds"]}
            report["cpu"]["train_process_cpu"] = report["cpu"].get("train_process_cpu", 0.0) + r["process_cpu_seconds"]
    for d in a.labels:
        for f in sorted(Path(d).glob("labels-c*.json")):
            r = json.loads(f.read_text())
            lab = report["labels"].setdefault(r["base"], {"chunks": 0, "episodes": 0, "points": 0, "branches": 0,
                                                          "cpu": 0.0, "default_checks": 0, "default_check_matches": 0})
            lab["chunks"] += 1
            for k in ("episodes", "points", "branches", "default_checks", "default_check_matches"):
                lab[k] += r["stats"][k]
            lab["cpu"] += r["process_cpu_seconds"]
            report["cpu"]["label_process_cpu"] = report["cpu"].get("label_process_cpu", 0.0) + r["process_cpu_seconds"]
    for lab in report["labels"].values():
        lab["cpu_per_point"] = lab["cpu"] / lab["points"] if lab["points"] else None
    a.output.write_text(json.dumps(report, indent=1, default=str))
    for name, r in report["bases"].items():
        c = r["criterion"]
        print(name, r["base_type"], None if c is None else {k: c[k] for k in ("best_fixed", "difference", "pass")})
    print(json.dumps(report["criterion_by_type"]))


if __name__ == "__main__":
    main()
