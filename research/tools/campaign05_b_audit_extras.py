#!/usr/bin/env python3
"""Independent audit (Track B) extras, from raw episode rows (pure Python):
  * B-Q: all-pairs Q-ranking accuracy on U+C (pairs with |dQ*| >= eps; Q-head ties count incorrect) and a
    decision-relevant variant (Q-head argmax eps-optimal at s0 on the probe-uniquely-optimal configurations);
  * B-X / B-XC: s0 accuracy, s0 gap vs later-decision gap, probe propensity (overall first-probe rate), and a
    discrimination index P(probe | uniquely optimal) - P(probe | not optimal);
  * cross-rule checks: would the B-X data have passed the (later) B-XC rule, and does B-XC pass the original
    registered B-X primary (>= +.15 in 3/3, not-optimal <= .10, regret NI 1.0 from paired config bootstrap)?
Usage: campaign05_b_audit_extras.py DATA_DIR AUDIT_JSON OUT_JSON
"""
import gzip
import itertools
import json
import os
import sys

EPS = 0.5


def rows(data, run, base):
    out = []
    with gzip.open(os.path.join(data, run, "run", "eval_episodes.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if "decisions" in r and base + 50_000_000 <= r["world_seed"] < base + 54_000_000:
                out.append(r)
    return out


def s0_stats(rs):
    n = len(rs)
    s0acc = sum(r["decisions"][0]["a"] in r["decisions"][0]["opt"] for r in rs) / n
    s0gap = sum(r["decisions"][0]["delta"] for r in rs) / n
    later = sum(sum(d["delta"] for d in r["decisions"][1:]) for r in rs) / n
    probe = sum(r["decisions"][0]["a"] == 0 for r in rs) / n
    u = [r for r in rs if r["decisions"][0]["opt"] == [0]]
    no = [r for r in rs if 0 not in r["decisions"][0]["opt"]]
    pu = sum(r["decisions"][0]["a"] == 0 for r in u) / len(u)
    pn = sum(r["decisions"][0]["a"] == 0 for r in no) / len(no)
    extra_wrong = sum(r["decisions"][0]["a"] == 0 for r in no)
    return {"s0_acc": s0acc, "s0_gap": s0gap, "later_gap": later, "first_probe_rate": probe,
            "discrimination": pu - pn, "n_probe_when_not_opt": extra_wrong, "n_probe_when_unique": sum(
                r["decisions"][0]["a"] == 0 for r in u)}


def qrank(rs):
    n = c = 0
    s0n = s0c = 0
    for r in rs:
        for j, d in enumerate(r["decisions"]):
            q, qh = d["Q"], d["qhat"]
            for x, y in itertools.combinations(range(len(q)), 2):
                dq = q[x] - q[y]
                if abs(dq) < EPS:
                    continue
                n += 1
                dh = qh[x] - qh[y]
                c += (dq > 0 and dh > 0) or (dq < 0 and dh < 0)
            if j == 0 and d["opt"] == [0]:
                s0n += 1
                am = max(range(len(qh)), key=lambda i: qh[i])
                s0c += d["avail"][am] in d["opt"]
    return {"q_rank_acc": c / n, "n_pairs": n, "s0_unique_qhead_argmax_eps_opt": s0c / s0n, "n_s0_unique": s0n}


def main(data, audit, out):
    A = json.load(open(audit))
    res = {"bq": {}, "bx_s0": {}, "bxc_s0": {}, "cross_rules": {}}
    for s in (10, 11, 12):
        for arm, pat in (("B0-L4", "bx-bql4-b0-s{}"), ("BX1-L4", "bx-bql4-bx1-s{}")):
            res["bq"][f"{arm}-s{s}"] = qrank(rows(data, pat.format(s), 5_600_000_000))
        for arm, pat in (("B0", "bx-b0-s{}"), ("BX1", "bx-bx1-s{}"), ("BX2", "bx-bx2-s{}"), ("BX3", "bx-bx3-s{}")):
            res["bx_s0"][f"{arm}-s{s}"] = s0_stats(rows(data, pat.format(s), 5_600_000_000))
    for s in (20, 21, 22):
        for arm, pat in (("B0", "bx-xc-b0-s{}"), ("BX1", "bx-xc-bx1-s{}")):
            res["bxc_s0"][f"{arm}-s{s}"] = s0_stats(rows(data, pat.format(s), 5_900_000_000))
    # cross rules
    bx1 = A["bx_primary_uc"]["BX1"]["pairs"]
    lifts = [p["lift"] for p in bx1]
    gaps = [p["gap_diff"] for p in bx1]
    res["cross_rules"]["BX_data_under_BXC_rule"] = {
        "mean_lift": sum(lifts) / 3, "all_pos": all(x > 0 for x in lifts), "gap_lower_3of3": all(g < 0 for g in gaps),
        "would_pass": sum(lifts) / 3 >= .10 and all(x > 0 for x in lifts) and all(g < 0 for g in gaps)}
    pc = A["bxc"]["b5c_hold_uc"]["pairs"]
    res["cross_rules"]["BXC_data_under_original_BX_rule"] = {
        "per_pair": [{"gain": p["gain"], "BX1_not_opt": p["BX1"]["probe_not_opt"],
                      "regret_diff_ci95_upper": p["diff_ci95"]["regret"][1]} for p in pc],
        "would_pass": all(p["gain"] >= .15 - 1e-12 and p["BX1"]["probe_not_opt"] <= .10
                          and p["diff_ci95"]["regret"][1] <= 1.0 for p in pc)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
