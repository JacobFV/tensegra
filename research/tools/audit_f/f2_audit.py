"""Independent F2 recomputation from eval.json + f-brefs.json (extended-04 F audit).

Written before reading campaign04_f2_score.py. Usage: f2_audit.py F2_DIR [B1_DIR]
F2_DIR holds f-btrain-{L0,L1,L4}-s{3,4,5}/run/eval.json and f-brefs.json.
"""
import json, sys, os, hashlib

D = sys.argv[1]
HO = ["heldout_price", "heldout_k", "heldout_comp"]
SEEDS = [3, 4, 5]
refs = json.load(open(os.path.join(D, "f-brefs.json")))


def ev(rung, s):
    p = os.path.join(D, f"f-btrain-{rung}-s{s}", "run", "eval.json")
    e = json.load(open(p))
    assert e["rung"] == rung and e["seed"] == s, p
    return e


def summ(e, sp):
    return e["splits"][sp]["free_running_greedy"]["summary"]


out = {"H1": {}, "H2": {}, "H3": {}, "H4": {}, "per_split": {}}
E = {(r, s): ev(r, s) for r in ("L0", "L1", "L4") for s in SEEDS}
for (r, s), e in E.items():
    assert sorted(e["splits"]) == sorted(HO), (r, s, list(e["splits"]))

# ---- H1: held-out mean regret L1, L4 <= .8 x L0 in 3/3 pairs + transfer floor per split
h1 = {}
for s in SEEDS:
    m = {r: sum(summ(E[(r, s)], sp)["regret"] for sp in HO) / 3 for r in ("L0", "L1", "L4")}
    mg = {r: sum(summ(E[(r, s)], sp)["gap_regret"] for sp in HO) / 3 for r in ("L0", "L1", "L4")}
    floor = {r: {sp: summ(E[(r, s)], sp)["success"] >= 0.8 * refs[sp]["pi_star"]["success"] for sp in HO} for r in ("L1", "L4")}
    h1[s] = {"regret_mean": m, "gap_regret_mean": mg,
             "L1_pass": m["L1"] <= 0.8 * m["L0"] and all(floor["L1"].values()),
             "L4_pass": m["L4"] <= 0.8 * m["L0"] and all(floor["L4"].values()), "floor": floor,
             "per_split_regret": {r: {sp: summ(E[(r, s)], sp)["regret"] for sp in HO} for r in ("L0", "L1", "L4")},
             "per_split_success": {r: {sp: summ(E[(r, s)], sp)["success"] for sp in HO} for r in ("L0", "L1", "L4")}}
out["H1"] = {"per_seed": h1, "supported": all(v["L1_pass"] and v["L4_pass"] for v in h1.values()),
             "ref_regret": {sp: {k: refs[sp][k]["regret"] for k in refs[sp]} for sp in HO},
             "pi_star_success": {sp: refs[sp]["pi_star"]["success"] for sp in HO}}

# ---- H2: L4 effective-rho build-rate deviation from pi* on heldout_k and heldout_price
h2 = {}
for s in SEEDS:
    row = {}
    for sp in ("heldout_k", "heldout_price"):
        bins = E[("L4", s)]["splits"][sp]["free_running_greedy"]["rho_curve"]["by_rho_eff"]
        nz = [b for b in bins if b["n"] > 0]
        unw = sum(abs(b["build_rate"] - b["build_rate_pi_star"]) for b in nz) / len(nz)
        w = sum(b["n"] * abs(b["build_rate"] - b["build_rate_pi_star"]) for b in nz) / sum(b["n"] for b in nz)
        # also against the f-brefs pi* curve (same worlds) as a cross-check of the pi* column
        rb = refs[sp]["pi_star"].get("rho_curve", {}).get("by_rho_eff")
        same_pi = rb is not None and [round(b["build_rate_pi_star"], 9) for b in bins] == [round(b["build_rate"], 9) for b in rb]
        row[sp] = {"unweighted": unw, "n_weighted": w, "bins": len(bins), "nonempty": len(nz), "pi_star_col_matches_refs": same_pi}
    row["pass"] = all(row[sp]["unweighted"] <= 0.15 for sp in ("heldout_k", "heldout_price"))
    row["pass_weighted"] = all(row[sp]["n_weighted"] <= 0.15 for sp in ("heldout_k", "heldout_price"))
    h2[s] = row
out["H2"] = {"per_seed": h2, "supported": sum(v["pass"] for v in h2.values()) >= 2}

# ---- H3: first-probe rates, pooled held-out (n-weighted) and per split
h3 = {}
for s in SEEDS:
    S = [summ(E[("L4", s)], sp) for sp in HO]
    nn = sum(x["n_probe_not_opt"] for x in S); nu = sum(x["n_probe_unique_opt"] for x in S)
    pn = sum(x["first_probe_rate_when_not_opt"] * x["n_probe_not_opt"] for x in S) / nn
    pu = sum(x["first_probe_rate_when_unique_opt"] * x["n_probe_unique_opt"] for x in S) / nu
    per = {sp: {"not_opt": x["first_probe_rate_when_not_opt"], "n_not_opt": x["n_probe_not_opt"],
                "unique_opt": x["first_probe_rate_when_unique_opt"], "n_unique_opt": x["n_probe_unique_opt"],
                "pass": x["first_probe_rate_when_not_opt"] <= 0.10 and x["first_probe_rate_when_unique_opt"] >= 0.80}
           for sp, x in zip(HO, S)}
    h3[s] = {"pooled_not_opt": pn, "pooled_unique_opt": pu, "n_not_opt": nn, "n_unique_opt": nu,
             "pass": pn <= 0.10 and pu >= 0.80, "per_split": per}
out["H3"] = {"per_seed": h3, "supported": sum(v["pass"] for v in h3.values()) >= 2}

# ---- H4: unjustified switch fraction and justified-after-failed-probe ratio vs pi*
h4 = {}
for s in SEEDS:
    S = {sp: summ(E[("L4", s)], sp) for sp in HO}
    def unjust_frac(xs):
        j = sum(x[f"switch_just_{c}"] for x in xs for c in "abcd")
        u = sum(x[f"switch_unjust_{c}"] for x in xs for c in "abcd")
        return u / (u + j)
    def ratio_b(xs, sps):  # per-episode justified case-(b) switches, model / pi* (same worlds)
        return sum(x["switch_just_b"] for x in xs) / sum(refs[sp]["pi_star"]["switch_just_b"] for sp in sps)
    def ratio_b_per_case(xs, sps):  # justified switches per case-(b) failed probe, model / pi*
        m = sum(x["switch_just_b"] for x in xs) / sum(x["case_b"] for x in xs)
        p = sum(refs[sp]["pi_star"]["switch_just_b"] for sp in sps) / sum(refs[sp]["pi_star"]["case_b"] for sp in sps)
        return m / p
    xs = list(S.values())
    pooled = {"unjust_frac": unjust_frac(xs), "ratio_b": ratio_b(xs, HO), "ratio_b_per_case": ratio_b_per_case(xs, HO)}
    per = {sp: {"unjust_frac": unjust_frac([S[sp]]), "ratio_b": ratio_b([S[sp]], [sp]),
                "ratio_b_per_case": ratio_b_per_case([S[sp]], [sp]),
                "pass": unjust_frac([S[sp]]) <= 0.10 and ratio_b([S[sp]], [sp]) >= 0.8} for sp in HO}
    h4[s] = {"pooled": pooled, "pass": pooled["unjust_frac"] <= 0.10 and pooled["ratio_b"] >= 0.8,
             "pass_per_case_variant": pooled["unjust_frac"] <= 0.10 and pooled["ratio_b_per_case"] >= 0.8, "per_split": per}
out["H4"] = {"per_seed": h4, "supported": sum(v["pass"] for v in h4.values()) >= 2,
             "supported_per_case_variant": sum(v["pass_per_case_variant"] for v in h4.values()) >= 2}

# hashes of inputs
out["input_sha256"] = {os.path.relpath(p, D): hashlib.sha256(open(p, "rb").read()).hexdigest()
                       for p in [os.path.join(D, "f-brefs.json")] + [os.path.join(D, f"f-btrain-{r}-s{s}", "run", "eval.json") for r, s in E]}
json.dump(out, sys.stdout, indent=1)
