"""Score protocol-B2 (own-continuation value head) as registered.

Usage: python campaign04_b2_score.py B2_DIR B1_DIR [--out FILE] [--models]

  B2_DIR: contains b2-train-{L1,L4}-s{0,1,2}/run/{eval.json,failure_records.json,train_meta.json}
  B1_DIR: contains b-train-{L1,L4}-s{0,1,2}/run/eval.json (and model.pt for --models)
  --models: also compare every B1 parameter tensor of the B2 model.pt with the B1 model.pt (needs torch).
"""
import json
import random
import sys
from pathlib import Path

HELD = ("heldout_price", "heldout_k", "heldout_comp")
IID = ("dev", "test_iid")
RUNGS = ("L4", "L1")
SEEDS = (0, 1, 2)
THRESH = 0.05  # R, as B-H6
PREDICTORS = ("v_own", "v", "q_head_taken", "qstar_taken", "vstar")


def auroc(pos, neg):
    if not pos or not neg:
        return None
    neg = sorted(neg)
    import bisect
    tot = sum(bisect.bisect_left(neg, x) + 0.5 * (bisect.bisect_right(neg, x) - bisect.bisect_left(neg, x)) for x in pos)
    return tot / (len(pos) * len(neg))


def pooled_auroc(recs, key, n_boot=1000, seed=0):
    """Score = -prediction (low value predicts a wrong commit).  Stratified percentile bootstrap 95% CI."""
    pos = [-r[key] for r in recs if r["wrong"]]
    neg = [-r[key] for r in recs if not r["wrong"]]
    a = auroc(pos, neg)
    if a is None:
        return {"auroc": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    boots = sorted(auroc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(n_boot))
    return {"auroc": a, "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1]}


def fr(e, split):
    return e["splits"][split]["free_running_greedy"]


def score(b2, b1, models=False):
    out = {"per_run": {}, "hypotheses": {}}
    for r in RUNGS:
        for s in SEEDS:
            d2 = b2 / f"b2-train-{r}-s{s}/run"
            e2 = json.loads((d2 / "eval.json").read_text())
            e1 = json.loads((b1 / f"b-train-{r}-s{s}/run/eval.json").read_text())
            recs = json.loads((d2 / "failure_records.json").read_text())
            row = {"splits": {}}
            for sp in HELD + IID:
                f2, f1 = fr(e2, sp), fr(e1, sp)
                row["splits"][sp] = {
                    "v_own_rel_err": f2["v_own_calibration_own_return"]["reliability_error"],
                    "v_rel_err_own": f2["value_calibration_own_return"]["reliability_error"],
                    "v_rel_err_vstar": f2["value_vs_vstar"]["reliability_error"],
                    "v_own_rel_err_vstar": f2["v_own_vs_vstar"]["reliability_error"],
                    "vstar_rel_err_own_floor": f2["vstar_as_predictor_own_return"]["reliability_error"],
                    "v_own_mae": f2["v_own_calibration_own_return"]["mae"],
                    "v_mae_own": f2["value_calibration_own_return"]["mae"],
                    "regret": f2["summary"]["regret"], "regret_b1": f1["summary"]["regret"],
                    "regret_se_b1": f1["summary"]["regret_se"],
                    # stop-gradient => every B1 eval key should be bit-identical
                    "b1_keys_identical": all(f2.get(k) == v for k, v in f1.items()),
                }
            pooled = [x for sp in HELD for x in recs[sp]["free_running_greedy"]["commit"]]
            pooled_ep = [x for sp in HELD for x in recs[sp]["free_running_greedy"]["episode"]]
            row["failure_commit"] = {"n_pos": sum(x["wrong"] for x in pooled), "n": len(pooled),
                                     **{k: pooled_auroc(pooled, k) for k in PREDICTORS}}
            row["failure_episode"] = {"n_pos": sum(x["wrong"] for x in pooled_ep), "n": len(pooled_ep),
                                      **{k: pooled_auroc(pooled_ep, k) for k in PREDICTORS}}
            row["own_value_meta"] = json.loads((d2 / "train_meta.json").read_text())["own_value"]
            if models:
                import torch
                m2 = torch.load(d2 / "model.pt")
                m1 = torch.load(b1 / f"b-train-{r}-s{s}/run/model.pt")
                row["b1_params_bit_identical"] = (set(m1) <= set(m2)) and all(torch.equal(m1[k], m2[k]) for k in m1)
            out["per_run"][f"{r}-s{s}"] = row
    H = out["hypotheses"]
    for r in RUNGS:
        runs = [out["per_run"][f"{r}-s{s}"] for s in SEEDS]
        seed_pass = [all(x["splits"][sp]["v_own_rel_err"] <= THRESH for sp in HELD) for x in runs]
        v_pass = [all(x["splits"][sp]["v_rel_err_own"] <= THRESH for sp in HELD) for x in runs]
        iid_pass = [all(x["splits"][sp]["v_own_rel_err"] <= THRESH for sp in IID) for x in runs]
        H[f"B2-primary-{r}"] = {"v_own_heldout_pass_per_seed": seed_pass, "V_head_heldout_pass_per_seed": v_pass,
                               "v_own_iid_pass_per_seed": iid_pass, "supported": sum(seed_pass) >= 2}
        H[f"B2-policy-unchanged-{r}"] = {
            "b1_eval_keys_identical_all_splits": [all(x["splits"][sp]["b1_keys_identical"] for sp in HELD + IID) for x in runs],
            "b1_params_bit_identical": [x.get("b1_params_bit_identical") for x in runs],
            "max_abs_regret_diff_heldout": [max(abs(x["splits"][sp]["regret"] - x["splits"][sp]["regret_b1"]) for sp in HELD) for x in runs]}
        fp = []
        for x in runs:
            c = x["failure_commit"]
            if c["n_pos"] < 10:
                fp.append(None)  # insufficient positives (registered)
            else:
                v = c["v_own"]
                fp.append(v["auroc"] >= 0.70 and v["lo"] > 0.5)
        H[f"B2-failure-prediction-{r}"] = {"commit_level_pass_per_seed": fp,
                                          "predicts_failure": (sum(bool(p) for p in fp) >= 2) if sum(p is not None for p in fp) >= 2 else "insufficient"}
    # registered reading (L4)
    p4 = H["B2-primary-L4"]
    if p4["supported"]:
        reading = "continuation mismatch supported: own-return targets calibrate where the jointly trained V head does not"
    elif sum(p4["v_own_iid_pass_per_seed"]) >= 2:
        reading = "not continuation: v_own calibrated in-distribution but not held out (generalization of the value estimate)"
    else:
        reading = "not continuation: v_own miscalibrated even in-distribution (fit/capacity; see V* floor)"
    H["reading_L4"] = reading
    return out


if __name__ == "__main__":
    b2, b1 = Path(sys.argv[1]), Path(sys.argv[2])
    res = score(b2, b1, models="--models" in sys.argv)
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(res, indent=1))
    for k, v in res["hypotheses"].items():
        print(k, json.dumps(v))
    for run, row in res["per_run"].items():
        print(run, {sp: (round(d["v_own_rel_err"], 3), round(d["v_rel_err_own"], 3), round(d["vstar_rel_err_own_floor"], 3))
                    for sp, d in row["splits"].items()},
              "commit AUROC v_own", row["failure_commit"]["v_own"], "n_pos", row["failure_commit"]["n_pos"])
