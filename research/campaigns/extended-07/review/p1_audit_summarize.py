import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
blocks = {"b6c": [k for k in d if "b6c" in k], "dev": [k for k in d if "dev" in k]}


def pooled(tags, fam, f):
    """sum-weighted pooled mean over seeds, and per-seed list"""
    vals = [f(d[t][fam]) for t in tags if fam in d[t]]
    return vals


for blk, tags in blocks.items():
    for fam in ("SCE", "UCE"):
        if not any(fam in d[t] for t in tags):
            continue
        print(f"===== {blk} {fam}")
        R = [d[t][fam] for t in tags if fam in d[t]]
        print("checks: hist_mismatch", [r["hist_mismatch"] for r in R], "target_mm", [r["target_mismatch"] for r in R],
              "flag_agree", [r["flag_agree"] for r in R], "q5 same across seeds", len({json.dumps(r["q5"], sort_keys=True) for r in R}) == 1)
        print("q5", R[0]["q5"], R[0]["q5_flip_types"])
        for arm in ("LRN", "SUP", "RAWF"):
            row = {}
            for s in ("flip", "near_miss", "inv", "all", "none", "single", "pair"):
                num = sum(r["acc"][arm][s][0] * r["acc"][arm][s][1] for r in R)
                den = sum(r["acc"][arm][s][1] for r in R)
                row[s] = (round(num / den, 4), den)
            print(arm, row, "flip per seed", [round(r["acc"][arm]["flip"][0], 3) for r in R])
        # q2
        for s in ("flip", "near_miss", "inv", "all"):
            for iv in ("exact", "mirror", "gauss_pred:1"):
                k = f"{s}|{iv}"
                tot = {m: sum(r["q2"][k][m] for r in R) for m in ("n", "changed", "rescue", "harm", "rescue_in")}
                print("Q2", k, tot, "dacc", round((tot["rescue"] - tot["harm"]) / tot["n"], 4), "chg", round(tot["changed"] / tot["n"], 4))
        # q3 (equal weight per seed since n_dec equal)
        for arm in ("LRN", "SUP"):
            print("Q3", arm, "fid", [round(r["q3"][arm]["fidelity"], 5) for r in R], "share", [round(r["q3"][arm]["contrib_var_share"], 3) for r in R],
                  "zperm", round(np.mean([r["q3"][arm]["zperm_change_all"] for r in R]), 3), "base_flip", round(np.mean([r["q3"][arm]["base_acc_flip"] for r in R]), 3))
            for p in ("zero", "mean", "perm", "g1", "g4", "alt_none", "alt_single", "exact", "lrn_pred"):
                if p not in R[0]["q3"][arm]:
                    continue
                m = {k: round(float(np.mean([r["q3"][arm][p][k] for r in R])), 3) for k in R[0]["q3"][arm][p]}
                ps = [round(r["q3"][arm][p]["acc_flip"], 3) for r in R]
                print("   ", p, m, "acc_flip per seed", ps)
        # nmae
        nm = {}
        for s in R[0]["nmae"]:
            num = sum(r["nmae"][s][0] * r["nmae"][s][1] for r in R if s in r["nmae"])
            den = sum(r["nmae"][s][1] for r in R if s in r["nmae"])
            nm[s] = round(num / den, 3)
        print("nMAE DEC", nm, "flip full-single per seed", [round(r["nmae"]["flip_full_minus_single"][0], 3) for r in R])
        st = {}
        for k in R[0]["stale"]:
            n = sum(r["stale"][k]["n"] for r in R if k in r["stale"])
            s_ = sum(r["stale"][k]["stale"] * r["stale"][k]["n"] for r in R if k in r["stale"])
            cm = np.mean([r["stale"][k]["captured_median"] for r in R if k in r["stale"]])
            st[k] = (n, round(s_ / n, 3), round(float(cm), 3))
        print("stale", st)
