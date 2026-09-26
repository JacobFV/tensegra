"""Remote checks for F2 (world freshness, inputs) and B2 (primary, policy-unchanged) -- extended-04 F audit.

Written before reading campaign04_f2_score.py / campaign04_b2_score.py. CPU only. Prints JSON.
"""
import glob, hashlib, json, os, pickle, sys

HOME = os.path.expanduser("~/tensegra-campaign04")
R = f"{HOME}/results"
SRC = f"{HOME}/source-7655c0b0"  # the source tree the F2 evals ran from
sys.path.insert(0, f"{SRC}/src")
import torch  # noqa: E402
from tensegra import campaign04_probeworld as pw  # noqa: E402

HO = ["heldout_price", "heldout_k", "heldout_comp"]
ALL = ["dev", "test_iid"] + HO
out = {}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------- F2: inputs, seeds, world freshness
f2 = {"eval_sha256": {}, "train_meta": {}}
for p in sorted(glob.glob(f"{R}/f-btrain-*/run/eval.json")):
    f2["eval_sha256"][p.split("/")[-3]] = sha(p)
    tm = json.load(open(p.replace("eval.json", "train_meta.json")))
    f2["train_meta"][p.split("/")[-3]] = {k: tm.get(k) for k in ("rung", "seed", "updates", "batch", "params", "continuation")}
f2["f_brefs_sha256"] = sha(f"{R}/f-brefs.json")
labels = f"{R}/b-labels/labels"
fresh = {}
for sp in HO:
    with open(f"{labels}/{sp}.pkl", "rb") as f:
        pool = pickle.load(f)
    idxs = [x[0] for x in pool]
    same_world = 0; same_theta = 0; same_out = 0; total = 0; seed_overlap = 0
    s500 = set(); s700 = set()
    for idx, cfg, V, Q in pool:
        for r in range(4):
            a, b = pw.world_seed(sp, idx, 500 + r), pw.world_seed(sp, idx, 700 + r)
            s500.add(a); s700.add(b)
            ea, eb = pw.Episode(cfg, a), pw.Episode(cfg, b)
            total += 1
            same_theta += (ea.thetas == eb.thetas and ea.z == eb.z)
            # continue each world's outcome stream: identical hidden state + identical next draws => same world
            da = [ea.rng.random() for _ in range(8)]; db = [eb.rng.random() for _ in range(8)]
            same_out += (da == db)
            same_world += (ea.thetas == eb.thetas and ea.z == eb.z and da == db)
    fresh[sp] = {"n_configs": len(pool), "idx_min": min(idxs), "idx_max": max(idxs), "episodes": total,
                 "seed_overlap_500_700": len(s500 & s700), "same_hidden_theta_z": same_theta,
                 "same_outcome_stream": same_out, "identical_world": same_world,
                 "labels_pkl_sha256": sha(f"{labels}/{sp}.pkl")}
f2["world_freshness"] = fresh
# B1 eval commands and F2 eval commands: same labels dir?
cmds = {}
for p in sorted(glob.glob(f"{R}/b-train-*-process/launch.json") + glob.glob(f"{R}/b-eval-*-process/launch.json")
                + glob.glob(f"{R}/f-beval-*-process/launch.json") + glob.glob(f"{R}/f-btrain-*-process/launch.json")
                + glob.glob(f"{R}/b-refs*-process/launch.json") + glob.glob(f"{R}/f-brefs-process/launch.json")):
    l = json.load(open(p))
    c = l["command"]
    lab = c[c.index("--labels") + 1] if "--labels" in c else None
    off = c[c.index("--world-offset") + 1] if "--world-offset" in c else "default(500)"
    cmds[p.split("/")[-2]] = {"labels": lab, "world_offset": off, "cwd": os.path.basename(l["cwd"]),
                              "sub": [x for x in c if x in ("train", "eval", "references")]}
f2["commands"] = cmds
out["F2"] = f2

# ---------------- B2
b2 = {"primary": {}, "L1": {}, "policy_unchanged": {}, "eval_keys": {}, "failure": {}}


def reliab(e, sp, key):
    try:
        return e["splits"][sp]["free_running_greedy"][key]["reliability_error"]
    except KeyError:
        return None


for rung in ("L4", "L1"):
    for s in (0, 1, 2):
        run2 = f"{R}/b2-train-{rung}-s{s}/run"
        run1 = f"{R}/b-train-{rung}-s{s}/run"
        e2 = json.load(open(f"{run2}/eval.json")); e1 = json.load(open(f"{run1}/eval.json"))
        cal = {sp: {"v_own": reliab(e2, sp, "v_own_calibration_own_return"),
                    "V": reliab(e2, sp, "value_calibration_own_return"),
                    "V_b1": reliab(e1, sp, "value_calibration_own_return"),
                    "vstar_floor": reliab(e2, sp, "vstar_as_predictor_own_return"),
                    "v_own_vs_vstar": reliab(e2, sp, "v_own_vs_vstar")} for sp in ALL if sp in e2["splits"]}
        passes = all(cal[sp]["v_own"] <= 0.05 for sp in HO)
        iid_pass = all(cal[sp]["v_own"] <= 0.05 for sp in ("dev", "test_iid"))
        b2["primary" if rung == "L4" else "L1"][f"s{s}"] = {"cal": cal, "heldout_pass": passes, "iid_pass": iid_pass,
                                                            "V_heldout_pass": all(cal[sp]["V"] <= 0.05 for sp in HO)}
        # policy unchanged: parameter bit identity
        sd2 = torch.load(f"{run2}/model.pt", map_location="cpu"); sd1 = torch.load(f"{run1}/model.pt", map_location="cpu")
        k2 = set(sd2); k1 = set(sd1)
        extra = sorted(k2 - k1); missing = sorted(k1 - k2)
        diff = [k for k in sorted(k1 & k2) if not (sd1[k].dtype == sd2[k].dtype and sd1[k].shape == sd2[k].shape and torch.equal(sd1[k], sd2[k]))]
        b2["policy_unchanged"][f"{rung}-s{s}"] = {"n_b1_tensors": len(k1), "n_common": len(k1 & k2), "extra_in_b2": extra,
                                                  "missing_in_b2": missing, "differing": diff,
                                                  "n_params_b1": int(sum(v.numel() for v in sd1.values())),
                                                  "n_params_b2": int(sum(v.numel() for v in sd2.values()))}
        # B1 eval keys identical (recursive, over keys present in B1 eval)
        mism = []

        def cmp(a, b, path):
            if isinstance(a, dict):
                if not isinstance(b, dict):
                    mism.append(path); return
                for k in a:
                    if k not in b:
                        mism.append(path + "/" + k + "(missing)")
                    else:
                        cmp(a[k], b[k], path + "/" + k)
            elif isinstance(a, list):
                if not isinstance(b, list) or len(a) != len(b):
                    mism.append(path); return
                for i, (x, y) in enumerate(zip(a, b)):
                    cmp(x, y, f"{path}[{i}]")
            else:
                if a != b and not (path.endswith("cpu_s")):
                    mism.append(path)
        cmp(e1, e2, "")
        b2["eval_keys"][f"{rung}-s{s}"] = {"n_mismatch": len(mism), "first": mism[:10],
                                           "regret_diff": {sp: e2["splits"][sp]["free_running_greedy"]["summary"]["regret"]
                                                           - e1["splits"][sp]["free_running_greedy"]["summary"]["regret"] for sp in ALL}}
        fp = e2["splits"]
        pooled = None
        b2["failure"][f"{rung}-s{s}"] = {sp: {lvl: {"n_pos": v.get("n_pos"), "auroc_v_own": (v.get("auroc") or {}).get("v_own")}
                                              for lvl, v in fp[sp]["free_running_greedy"].get("failure_prediction", {}).items()
                                              if isinstance(v, dict)} for sp in HO}
out["B2"] = b2
out["B2"]["primary_supported"] = sum(v["heldout_pass"] for v in b2["primary"].values()) >= 2
out["B2"]["iid_pass_count_L4"] = sum(v["iid_pass"] for v in b2["primary"].values())
json.dump(out, sys.stdout, indent=1, default=str)
