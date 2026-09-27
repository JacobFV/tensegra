"""Extended-07 Phase 2: cost/RSS table of the metered smokes (receipts under results/dev) and the unit costs of each
batch type read from the smoke runs' own train_meta / summaries.  Pure Python (runs anywhere the files are).

  python research/tools/campaign07_p2_costs.py DEV_RESULTS_DIR SMOKE_DIR [--prefix e07-dev-p2-smoke-]
"""
import glob
import json
import os
import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    dev, smoke = argv[0], argv[1]
    prefix = argv[3] if len(argv) > 3 and argv[2] == "--prefix" else "e07-dev-p2-smoke-"
    out = {"receipts": [], "runs": {}, "eval": None}
    for f in sorted(glob.glob(os.path.join(dev, prefix + "*", "occupancy.json"))):
        o = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        out["receipts"].append({"job": name.rsplit("-", 2)[0], "exit": o["exit_code"], "cpu": round(o["cpu_core_seconds"], 2),
                                "wall": round(o["wall_seconds"], 2), "rss_mib": round(o["max_child_rss_kib"] / 1024)})
    for f in sorted(glob.glob(os.path.join(smoke, "*", "run", "train_meta.json"))):
        m = json.load(open(f))
        p = m["p2"]
        b = p.get("bank", {})
        out["runs"][f.split(os.sep)[-3]] = {
            "bank_updates": b.get("updates", 0), "cpu_bank": round(p["cpu_s_bank"], 3),
            "cpu_per_bank_update": round(p["cpu_s_bank"] / b["updates"], 4) if b.get("updates") else None,
            "onpolicy_updates": p["onpolicy_updates"], "cpu_onpolicy": round(p["cpu_s_onpolicy"], 3),
            "cpu_per_onpolicy_update": round(p["cpu_s_onpolicy"] / p["onpolicy_updates"], 4) if p["onpolicy_updates"] else None,
            "cpu_label_load": round(m["cpu_s_label_load"], 2), "cpu_total": round(m["cpu_s_total"], 2),
            "batch": m["batch"], "decisions_seen": b.get("decisions_seen")}
    ev = glob.glob(os.path.join(smoke, "eval", "diag-v1-summary-*.json"))
    if ev:
        s = json.load(open(ev[0]))
        out["eval"] = {"shards": s["shards"], "cf": s["cf"], "cpu_total": s["cpu_s_total"],
                       "B_summary": {k: {kk: round(v[kk], 3) for kk in ("n", "regret", "gap_regret")}
                                     for k, v in s["B_summary"].items()}}
    for f in glob.glob(os.path.join(smoke, "*.pkl.json")) + glob.glob(os.path.join(smoke, "oof.pt.json")):
        d = json.load(open(f))
        out[os.path.basename(f)] = {k: d[k] for k in ("episodes", "decisions", "cpu_s", "combo_episodes", "group_nmae",
                                                      "mse", "sources") if k in d}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
