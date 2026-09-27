#!/usr/bin/env python3
"""Independent audit of extended-05 Track B (B-HR, B-X, B-Q, B-XC, B-LOC) from raw episode rows.

Written by the independent auditor WITHOUT reading campaign05_bx_score.py / campaign05_bloc.py.  Pure Python
(+ random); inputs are local read-only copies of the pro6000 run directories (eval_episodes.jsonl.gz, train_meta.json,
eval.json), the hold s0 label files, and the root score JSONs (only for the final comparison).

Definitions reconstructed from design.md / registry.json / decisions.md:
  * s0 = the first decision of the episode (decisions[0]).  Probe = action 0.
  * uniquely-optimal first-probe rate: over episodes whose s0 eps-optimal set (eps = .5) is exactly {probe},
    the fraction whose first action is probe.  Support = number of such configurations (1 world / configuration).
  * not-optimal probe rate: over episodes whose s0 eps-optimal set excludes probe, the fraction that probe first.
  * realized regret = V*(s0) - U (one outcome draw);  gap regret = sum_t delta_t, delta_t = V*(I_t) - Q*(I_t, a_t).
  * CIs: 95% percentile bootstrap over configurations (1,000 resamples; seed fixed); paired differences resample
    the same configurations in both runs (the runs share configurations and worlds).
Usage:  campaign05_b_audit.py DATA_DIR OUT_JSON
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys

EPS = 0.5
PROBE = 0
NBOOT = 1000
WORLD_OFF = 50_000_000
BASES = {"b5_hold_uc": 5_600_000_000, "b5_hold_se": 5_700_000_000, "b5c_hold_uc": 5_900_000_000,
         "heldout_comp": 4_600_000_000}


def load_rows(run):
    rows = []
    with gzip.open(os.path.join(run, "eval_episodes.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if "decisions" in r:
                rows.append(r)
    return rows


def split_of(r):
    if r.get("split"):
        return r["split"]
    ws = r["world_seed"]
    for name, base in BASES.items():
        if base + WORLD_OFF <= ws < base + WORLD_OFF + 4000 * 1000:
            return name
    return None


def cfg_idx(r, split):
    return (r["world_seed"] - BASES[split] - WORLD_OFF) // 1000


def ep_stats(r):
    d0 = r["decisions"][0]
    opt0 = set(d0["opt"])
    return {
        "uniq": opt0 == {PROBE},
        "notopt": PROBE not in opt0,
        "probe": d0["a"] == PROBE,
        "regret": r["V_star"] - r["U"],
        "gap": sum(d["delta"] for d in r["decisions"]),
        "success": r["success"],
        "k": r["k"],
    }


def metrics(eps):
    u = [e for e in eps if e["uniq"]]
    n = [e for e in eps if e["notopt"]]
    m = len(eps)
    return {
        "episodes": m,
        "probe_unique": (sum(e["probe"] for e in u) / len(u)) if u else None,
        "n_probe_unique": len(u),
        "probe_not_opt": (sum(e["probe"] for e in n) / len(n)) if n else None,
        "n_probe_not_opt": len(n),
        "regret": sum(e["regret"] for e in eps) / m,
        "gap_regret": sum(e["gap"] for e in eps) / m,
        "success": sum(e["success"] for e in eps) / m,
    }


def pct(xs, lo=0.025, hi=0.975):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return [xs[int(lo * (len(xs) - 1))], xs[int(math.ceil(hi * (len(xs) - 1)))]]


def boot_single(by_cfg, keys, rng):
    cfgs = sorted(by_cfg)
    out = {k: [] for k in keys}
    for _ in range(NBOOT):
        sample = [by_cfg[cfgs[rng.randrange(len(cfgs))]] for _ in cfgs]
        m = metrics(sample)
        for k in keys:
            out[k].append(m[k])
    return {k: pct(v) for k, v in out.items()}


def boot_paired(a, b, keys, rng):
    """a, b: dict cfg -> episode stats (same configurations); CI of metric(b) - metric(a)."""
    cfgs = sorted(set(a) & set(b))
    out = {k: [] for k in keys}
    for _ in range(NBOOT):
        idx = [cfgs[rng.randrange(len(cfgs))] for _ in cfgs]
        ma = metrics([a[c] for c in idx])
        mb = metrics([b[c] for c in idx])
        for k in keys:
            if ma[k] is None or mb[k] is None:
                continue
            out[k].append(mb[k] - ma[k])
    return {k: pct(v) for k, v in out.items()}


def run_groups(run, s0_sens):
    """-> {group: {cfg: stats}} for the hold groups present in a run."""
    rows = load_rows(run)
    groups = {}
    for r in rows:
        sp = split_of(r)
        if sp not in BASES or sp == "heldout_comp":
            continue
        c = cfg_idx(r, sp)
        st = ep_stats(r)
        st["world_rep"] = (r["world_seed"] - BASES[sp] - WORLD_OFF) % 1000
        groups.setdefault(sp, {})[c] = st
        if sp in ("b5_hold_uc", "b5c_hold_uc"):
            if s0_sens[sp][c]["flag_sensitive"]:
                groups.setdefault(sp + ":flag_sensitive", {})[c] = st
            groups.setdefault(sp + (":k1" if r["k"] == 1 else ":k>1"), {})[c] = st
    return groups


def s0_consistency(groups, s0):
    """Episode-row s0 opt sets agree with the label job's s0 file (probe_unique flags)."""
    bad = 0
    for sp in ("b5_hold_uc", "b5_hold_se", "b5c_hold_uc"):
        if sp not in groups:
            continue
        for c, st in groups[sp].items():
            bad += st["uniq"] != s0[sp][c]["probe_unique"]
    return bad


def main(data, out):
    rng = random.Random(20260927)
    s0 = {}
    for sp, f in (("b5_hold_uc", "bx-labels/labels/b5_hold_uc_s0.json"),
                  ("b5_hold_se", "bx-labels/labels/b5_hold_se_s0.json"),
                  ("b5c_hold_uc", "bxc-labels/labels/b5c_hold_uc_s0.json")):
        s0[sp] = {x["idx"]: x for x in json.load(open(os.path.join(data, f)))}
    res = {"auditor_tool": "campaign05_b_audit", "eps": EPS, "n_boot": NBOOT, "sizing": {}, "bx": {}, "bxc": {},
           "bq": {}, "regret_defs": {}}

    # ---- sizing rule re-applied to the label job's own s0 records (prefix rule)
    for sp in s0:
        recs = [s0[sp][i] for i in range(len(s0[sp]))]
        c = cs = 0
        n_at = None
        for n, x in enumerate(recs, 1):
            c += x["probe_unique"]
            cs += bool(x["probe_unique"] and x.get("flag_sensitive"))
            tgt_s = 20 if "uc" in sp else 0
            if c >= 60 and cs >= tgt_s:
                n_at = n
                break
        res["sizing"][sp] = {"pool": len(recs), "prefix_rule_n": n_at, "eligible": c, "eligible_flag_sensitive": cs,
                             "matches": n_at == len(recs)}

    arms = {"B0": "bx-b0-s1{}", "BX1": "bx-bx1-s1{}", "BX2": "bx-bx2-s1{}", "BX3": "bx-bx3-s1{}",
            "B0-L4": "bx-bql4-b0-s1{}", "BX1-L4": "bx-bql4-bx1-s1{}"}
    G = {}
    for arm, pat in arms.items():
        for s in (0, 1, 2):
            run = os.path.join(data, pat.format(s), "run")
            G[(arm, 10 + s)] = run_groups(run, s0)
            meta = json.load(open(os.path.join(run, "train_meta.json")))
            res["bx"].setdefault(arm, {})[10 + s] = {"meta": {k: meta.get(k) for k in
                                                              ("rung", "seed", "train_split", "inputs", "arch",
                                                               "updates", "batch", "params")},
                                                     "s0_mismatch_vs_label_file": s0_consistency(G[(arm, 10 + s)], s0)}
    keys = ("probe_unique", "probe_not_opt", "regret", "gap_regret", "success")
    for (arm, seed), g in G.items():
        for grp in ("b5_hold_uc", "b5_hold_se", "b5_hold_uc:flag_sensitive", "b5_hold_uc:k1", "b5_hold_uc:k>1"):
            m = metrics(list(g[grp].values()))
            if grp in ("b5_hold_uc", "b5_hold_uc:flag_sensitive"):
                m["ci95"] = boot_single(g[grp], keys, rng)
            res["bx"][arm][seed][grp] = m
    # B-HR from B0 only (registered: no_failure >= .80 w/ support >= 20; tooling: evaluable <= .70 in >= 2/3)
    bhr = {}
    for grp in ("b5_hold_uc", "b5_hold_se", "b5_hold_uc:flag_sensitive"):
        per = [res["bx"]["B0"][s][grp] for s in (10, 11, 12)]
        nf = sum(p["probe_unique"] >= .80 and p["n_probe_unique"] >= 20 for p in per)
        ev = sum(p["probe_unique"] <= .70 for p in per)
        bhr[grp] = {"per_seed": [(p["probe_unique"], p["n_probe_unique"]) for p in per],
                    "verdict": "no_failure" if nf >= 2 else ("evaluable" if ev >= 2 else "ambiguous")}
    res["bhr"] = bhr
    # B-X primary per arm on U+C (and paired CIs)
    prim = {}
    for arm in ("BX1", "BX2", "BX3"):
        pairs = []
        for s in (10, 11, 12):
            a, b = G[("B0", s)]["b5_hold_uc"], G[(arm, s)]["b5_hold_uc"]
            ma, mb = metrics(list(a.values())), metrics(list(b.values()))
            ci = boot_paired(a, b, keys, rng)
            pairs.append({"seed": s, "lift": mb["probe_unique"] - ma["probe_unique"],
                          "arm_not_opt": mb["probe_not_opt"], "regret_diff": mb["regret"] - ma["regret"],
                          "gap_diff": mb["gap_regret"] - ma["gap_regret"], "diff_ci95": ci,
                          "pass_lift": mb["probe_unique"] >= ma["probe_unique"] + .15,
                          "pass_notopt": mb["probe_not_opt"] <= .10,
                          "pass_regret_ni": ci["regret"][1] <= 1.0})
        n_pass = sum(p["pass_lift"] and p["pass_notopt"] and p["pass_regret_ni"] for p in pairs)
        prim[arm] = {"pairs": pairs, "pairs_passing_all": n_pass, "lift_passes": sum(p["pass_lift"] for p in pairs),
                     "supported": n_pass == 3}
    res["bx_primary_uc"] = prim
    # B-Q (L4) readings on U+C
    for s in (10, 11, 12):
        a, b = G[("B0-L4", s)]["b5_hold_uc"], G[("BX1-L4", s)]["b5_hold_uc"]
        res["bq"][s] = {"B0-L4": metrics(list(a.values())), "BX1-L4": metrics(list(b.values()))}

    # ---- B-XC
    GX = {}
    for arm, pat in (("B0", "bx-xc-b0-s2{}"), ("BX1", "bx-xc-bx1-s2{}")):
        for s in (0, 1, 2):
            run = os.path.join(data, pat.format(s), "run")
            rows = load_rows(run)
            meta = json.load(open(os.path.join(run, "train_meta.json")))
            ev = json.load(open(os.path.join(run, "eval.json")))
            splits = {split_of(r) for r in rows}
            reps = {(r["world_seed"] - BASES["b5c_hold_uc"] - WORLD_OFF) % 1000 for r in rows}
            cfgs = sorted(cfg_idx(r, "b5c_hold_uc") for r in rows)
            GX[(arm, 20 + s)] = run_groups(run, s0)
            res["bxc"].setdefault("validity", {})[f"{arm}-s{20 + s}"] = {
                "seed": meta["seed"], "train_split": meta["train_split"], "rung": meta["rung"],
                "inputs": meta.get("inputs"), "arch": meta.get("arch"), "updates": meta["updates"],
                "eval_splits": sorted(ev["splits"]), "row_splits": sorted(map(str, splits)),
                "world_reps": sorted(reps), "n_rows": len(rows), "cfgs_0_to_n": cfgs == list(range(len(cfgs))),
                "s0_mismatch_vs_label_file": s0_consistency(GX[(arm, 20 + s)], s0)}
    for grp in ("b5c_hold_uc", "b5c_hold_uc:flag_sensitive", "b5c_hold_uc:k1", "b5c_hold_uc:k>1"):
        pairs = []
        for s in (20, 21, 22):
            a, b = GX[("B0", s)][grp], GX[("BX1", s)][grp]
            ma, mb = metrics(list(a.values())), metrics(list(b.values()))
            pairs.append({"seed": s, "B0": ma, "BX1": mb, "gain": mb["probe_unique"] - ma["probe_unique"],
                          "gap_diff": mb["gap_regret"] - ma["gap_regret"],
                          "regret_diff": mb["regret"] - ma["regret"],
                          "notopt_diff": mb["probe_not_opt"] - ma["probe_not_opt"],
                          "diff_ci95": boot_paired(a, b, keys, rng) if ":k" not in grp else None})
        mean_gain = sum(p["gain"] for p in pairs) / 3
        # CI for the mean paired gain: configurations resampled jointly across the 6 runs
        cf = sorted(GX[("B0", 20)][grp])
        bs = []
        for _ in range(NBOOT):
            idx = [cf[rng.randrange(len(cf))] for _ in cf]
            g = []
            for s in (20, 21, 22):
                ma = metrics([GX[("B0", s)][grp][c] for c in idx])["probe_unique"]
                mb = metrics([GX[("BX1", s)][grp][c] for c in idx])["probe_unique"]
                if ma is not None and mb is not None:
                    g.append(mb - ma)
            if len(g) == 3:
                bs.append(sum(g) / 3)
        res["bxc"][grp] = {"pairs": pairs, "mean_gain": mean_gain, "mean_gain_ci95": pct(bs),
                           "all_pairs_pos": all(p["gain"] > 0 for p in pairs),
                           "gap_lower_3of3": all(p["gap_diff"] < 0 for p in pairs)}
    g = res["bxc"]["b5c_hold_uc"]
    res["bxc"]["primary_confirmed"] = g["mean_gain"] >= .10 and g["all_pairs_pos"] and g["gap_lower_3of3"]

    # ---- regret definitions: realized - gap = martingale residual (mean 0 in expectation)
    for key, runs in (("B-X U+C (12 runs)", [G[(a, s)]["b5_hold_uc"] for a in ("B0", "BX1", "BX2", "BX3")
                                            for s in (10, 11, 12)]),
                      ("B-XC (6 runs)", [GX[(a, s)]["b5c_hold_uc"] for a in ("B0", "BX1") for s in (20, 21, 22)])):
        per = []
        for g in runs:
            eps = list(g.values())
            n = len(eps)
            reg = [e["regret"] for e in eps]
            gap = [e["gap"] for e in eps]
            res_ = [x - y for x, y in zip(reg, gap)]

            def sd(v):
                mu = sum(v) / len(v)
                return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
            per.append({"n": n, "regret": sum(reg) / n, "gap": sum(gap) / n, "resid_mean": sum(res_) / n,
                        "resid_se": sd(res_) / math.sqrt(n), "regret_se": sd(reg) / math.sqrt(n),
                        "gap_se": sd(gap) / math.sqrt(n), "frac_negative_realized": sum(x < 0 for x in reg) / n,
                        "min_gap": min(gap)})
        res["regret_defs"][key] = per
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
