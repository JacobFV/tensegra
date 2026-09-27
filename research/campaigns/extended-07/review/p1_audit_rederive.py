"""Independent re-derivation of extended-07 P1 headline numbers from raw diag-v1 logs (audit; read-only inputs).
Written without importing campaign07_p1analysis.  Usage: python audit_p1.py OUT.json DIR...
"""
import base64, gzip, json, sys, time
from collections import defaultdict
from itertools import combinations
import numpy as np
import torch

t0 = time.process_time()
OUT = sys.argv[1]
DIRS = sys.argv[2:]
FAMS = ("SCE", "UCE")
res = {}


def load(path):
    with gzip.open(path, "rt") as f:
        head = json.loads(f.readline())["_meta"]
        recs = [json.loads(l) for l in f if l.strip()]
    return head, [r for r in recs if r.get("kind") == "decision"]


def zdec(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float64)


def weights(run):
    sd = torch.load(run + "/model.pt", map_location="cpu")
    g = lambda k: sd[k].numpy().astype(np.float64)
    return g("fuse.weight"), g("fuse.bias"), g("pi.weight"), g("pi.bias")


def act(w, Z, PHI, masks):
    W, b, P, pb = w
    h = np.tanh(np.concatenate([Z, PHI], 1) @ W.T + b)
    lg = h @ P.T + pb
    lg = np.where(masks, lg, -np.inf)
    return lg.argmax(1)


def pi_of(q):
    v = max(q.values())
    return min(int(a) for a, x in q.items() if x >= v - 1e-9)


for d in DIRS:
    seed = int(d.rstrip("/").split("-s")[-1])
    tag = d.rstrip("/").split("/")[-1]
    R = res.setdefault(tag, {})
    for fam in FAMS:
        files = {k: f"{d}/diag-v1-cf-{fam}-{k}-s{seed}.jsonl.gz" for k in ("LRN", "SUP", "RAWF")}
        try:
            data = {k: load(p) for k, p in files.items()}
        except FileNotFoundError:
            continue
        head = data["LRN"][0]
        feats = head["features"]
        FI = {f: j for j, f in enumerate(feats)}
        groups = head["groups"]
        DEC = np.array([FI[c] for c in groups["G1"] + groups["G2"] + groups["G3"]])
        sup_ref = json.load(open(head["models"][f"LRN-s{seed}"]["support_ref"]))
        tstd = np.asarray(sup_ref["target_std"])
        sig = np.asarray(sup_ref["err_rms"])
        runs = {m["kind"]: m["run"] for m in head["models"].values()}
        W = {k: weights(runs[k]) for k in ("LRN", "SUP")}
        # ---- key uniqueness / identical histories
        by = {}
        for k in ("LRN", "SUP", "RAWF"):
            recs = data[k][1]
            keys = [(r["cfg_id"], r["cf"]["type"]) for r in recs]
            assert len(set(keys)) == len(keys), (tag, fam, k, "duplicate (cfg_id,type) keys")
            by[k] = dict(zip(keys, recs))
        assert set(by["LRN"]) == set(by["SUP"]) == set(by["RAWF"])
        keys = sorted(by["LRN"])
        hist_mismatch = sum(len({by[k][x]["hist_id"] for k in by}) > 1 for x in keys)
        targ_mismatch = sum(not np.allclose(by["LRN"][x]["target"], by["SUP"][x]["target"], atol=2e-6) for x in keys)
        sup_pred_is_target = sum(np.allclose(by["SUP"][x]["pred"], by["SUP"][x]["target"], atol=2e-6) for x in keys)
        # ---- own membership from Q (units: octet x type)
        units = defaultdict(dict)
        for x in keys:
            r = by["LRN"][x]
            c = r["cf"]
            units[(c["octet"], c["type"])][c["member"] if c["role"] != "near_miss" else "near_miss"] = x
        pairs = ["".join(p) for p in combinations(fam, 2)]
        need = [fam, "0"] + list(fam) + pairs
        mem = {}
        q5 = defaultdict(int)
        q5_types = defaultdict(int)
        flag_agree = [0, 0]
        for u, m in units.items():
            if any(k not in m for k in need):
                q5["incomplete"] += 1
                continue
            Q = {k: by["LRN"][m[k]]["Q"] for k in need}
            full = by["LRN"][m[fam]]
            O = set(full["opt"])
            uniq = len(O) == 1
            pis = {k: pi_of(Q[k]) for k in need}
            fs = all(pis[f] not in O for f in fam) and uniq
            fp = all(pis[p] not in O for p in pairs) and uniq
            flag_agree[0] += 1
            flag_agree[1] += fs == (full["cf"]["flip"] and full["cf"]["unique_full"])
            q5["units"] += 1
            q5["unique_full"] += uniq
            q5["flip_vs_singles"] += fs
            q5["flip_vs_singles_and_pairs"] += fs and fp
            q5["inherited_any_pair"] += fs and not fp and any(pis[p] in O for p in pairs)
            for p in pairs:
                q5[f"inherited_{p}"] += fs and pis[p] in O
            if fs:
                q5_types[u[1]] += 1
            acts = sorted(Q[fam], key=int)
            qadd = {a: sum(Q[p][a] for p in pairs) - sum(Q[f][a] for f in fam) + Q["0"][a] for a in acts}
            best = max(qadd, key=lambda a: (qadd[a], -int(a)))
            q5["pairQ_hits_all"] += int(best) in O
            if fs:
                q5["pairQ_hits_flip"] += int(best) in O
                q5["pairQ_hits_flip_vs_pairs"] += (int(best) in O) and fp
            # additive-in-singles (first-order) extrapolation for comparison
            q1add = {a: sum(Q[f][a] for f in fam) - 2 * Q["0"][a] for a in acts}
            b1 = max(q1add, key=lambda a: (q1add[a], -int(a)))
            q5["singleQ_hits_all"] += int(b1) in O
            if fs:
                q5["singleQ_hits_flip"] += int(b1) in O
            for k in need:
                lv = "none" if k == "0" else {1: "single", 2: "pair"}.get(len(k), "full")
                mem[m[k]] = lv
            mem[m[fam]] = "flip" if fs else "inv"
            if "near_miss" in m:
                mem[m["near_miss"]] = "near_miss"
        # ---- accuracies
        acc = defaultdict(lambda: defaultdict(list))
        for x in keys:
            s = mem.get(x)
            for k in by:
                acc[k][s].append(by[k][x]["ok"])
                acc[k]["all"].append(by[k][x]["ok"])
        accs = {k: {s: [float(np.mean(v)), len(v)] for s, v in d2.items()} for k, d2 in acc.items()}
        # ---- Q2 exact replacement (logged iv on LRN)
        q2 = defaultdict(lambda: defaultdict(int))
        for x in keys:
            r = by["LRN"][x]
            s = mem.get(x)
            for iv in ("exact", "mirror", "gauss_pred:1"):
                e = r["iv"][iv]
                for ss in (s, "all"):
                    q2[f"{ss}|{iv}"]["n"] += 1
                    q2[f"{ss}|{iv}"]["changed"] += e["a"] != r["a"]
                    q2[f"{ss}|{iv}"]["rescue"] += (not r["ok"]) and e["ok"]
                    q2[f"{ss}|{iv}"]["harm"] += r["ok"] and not e["ok"]
                    q2[f"{ss}|{iv}"]["rescue_in"] += (not r["ok"]) and e["ok"] and bool(e["support"]["in"])
        # ---- Q3 recomputation
        n = len(keys)
        masks = np.zeros((n, 10), bool)
        opt = np.zeros((n, 10), bool)
        for i, x in enumerate(keys):
            masks[i, by["LRN"][x]["avail"]] = True
            opt[i, by["LRN"][x]["opt"]] = True
        Z = {k: np.stack([zdec(by[k][x]["z_pre"]) for x in keys]) for k in ("LRN", "SUP")}
        PH = {k: np.array([by[k][x]["pred"] for x in keys], float) for k in ("LRN", "SUP")}
        TG = np.array([by["LRN"][x]["target"] for x in keys], float)
        logged = {k: np.array([by[k][x]["a"] for x in keys]) for k in ("LRN", "SUP")}
        isflip = np.array([mem.get(x) == "flip" for x in keys])
        isinv = np.array([mem.get(x) == "inv" for x in keys])
        isnm = np.array([mem.get(x) == "near_miss" for x in keys])
        # within-support alternative phi: the same arm's phi at the NONE member / a SINGLE member of the same unit+type
        alt_none = {k: PH[k].copy() for k in PH}
        alt_single = {k: PH[k].copy() for k in PH}
        has_alt = np.zeros(n, bool)
        idx = {x: i for i, x in enumerate(keys)}
        rng = np.random.default_rng(1000 + seed)
        for u, m in units.items():
            if "0" not in m or fam not in m:
                continue
            for mm, x in m.items():
                if mm in ("0",):
                    continue
                i = idx[x]
                has_alt[i] = True
                sgl = [f for f in fam if f in m and f != mm]
                j0 = idx[m["0"]]
                js = idx[m[sgl[rng.integers(len(sgl))]]] if sgl else j0
                for k in PH:
                    alt_none[k][i] = PH[k][j0]
                    alt_single[k][i] = PH[k][js]
        q3 = {}
        for k in ("LRN", "SUP"):
            w = W[k]
            a0 = act(w, Z[k], PH[k], masks)
            fid = float((a0 == logged[k]).mean())
            Wz, Wf = w[0][:, :Z[k].shape[1]], w[0][:, Z[k].shape[1]:]
            cz, cfv = Z[k] @ Wz.T, PH[k] @ Wf.T
            share = float(cfv.var(0).sum() / (cz.var(0).sum() + cfv.var(0).sum()))
            perts = {"zero": np.zeros_like(PH[k]), "mean": np.broadcast_to(PH[k].mean(0), PH[k].shape),
                     "perm": PH[k][rng.permutation(n)], "g1": PH[k] + sig * rng.standard_normal(PH[k].shape),
                     "g4": PH[k] + 4 * sig * rng.standard_normal(PH[k].shape),
                     "alt_none": alt_none[k], "alt_single": alt_single[k]}
            if k == "LRN":
                perts["exact"] = TG
            else:
                perts["lrn_pred"] = PH["LRN"]
                perts["lrn_pred_as_err"] = PH["SUP"] + (PH["LRN"] - TG)  # same thing (SUP phi == target)
            out = {"fidelity": fid, "contrib_var_share": share,
                   "base_acc_flip": float(opt[isflip, a0[isflip]].mean())}
            for pn, P in perts.items():
                a = act(w, Z[k], P, masks)
                ch = a != a0
                okp = opt[np.arange(n), a]
                out[pn] = {"change_all": float(ch.mean()), "change_flip": float(ch[isflip].mean()),
                           "change_alt_rows": float(ch[has_alt].mean()),
                           "acc_flip": float(okp[isflip].mean()), "acc_inv": float(okp[isinv].mean()),
                           "acc_nm": float(okp[isnm].mean()), "acc_all": float(okp.mean())}
            q3[k] = out
        # z-permutation reference
        for k in ("LRN", "SUP"):
            a0 = act(W[k], Z[k], PH[k], masks)
            a = act(W[k], Z[k][rng.permutation(n)], PH[k], masks)
            q3[k]["zperm_change_all"] = float((a != a0).mean())
        # ---- Q1 nMAE by level, stale not-H belief
        err = np.abs(PH["LRN"] - TG) / np.maximum(tstd, 1e-3)
        nonc = tstd > 1e-6
        decm = np.zeros(len(feats), bool); decm[DEC] = True
        decm &= nonc
        nm = err[:, decm].mean(1)
        lv = defaultdict(list)
        for i, x in enumerate(keys):
            s = mem.get(x)
            if s is not None:
                lv[s].append(nm[i])
                if s in ("flip", "inv"):
                    lv["full"].append(nm[i])
        nmae = {s: [float(np.mean(v)), len(v)] for s, v in lv.items()}
        # paired full - single on flip units
        diffs = []
        for u, m in units.items():
            if fam in m and mem.get(m[fam]) == "flip":
                sg = [nm[idx[m[f]]] for f in fam if f in m]
                diffs.append(nm[idx[m[fam]]] - np.mean(sg))
        nmae["flip_full_minus_single"] = [float(np.mean(diffs)), len(diffs)]
        bH = FI["belief_H"]
        stale = defaultdict(lambda: [0, 0, []])
        for u, m in units.items():
            for mm, x in m.items():
                if mm == "near_miss" or "C" not in mm:
                    continue
                r = by["LRN"][x]
                if r["ctx"] not in ("query2_after_H", "query2_after_notH"):
                    continue
                noc = mm.replace("C", "") or "0"
                if noc not in m:
                    continue
                y = by["LRN"][m[noc]]
                if y["ctx"] != r["ctx"]:
                    continue
                tO, tN, p = r["target"][bH], y["target"][bH], r["pred"][bH]
                if abs(tO - tN) < 1e-6:
                    continue
                lvl = {1: "single", 2: "pair"}.get(len(mm), "full")
                s = stale[f"{r['ctx']}|{lvl}"]
                s[0] += 1
                s[1] += abs(p - tN) < abs(p - tO)
                s[2].append((p - tN) / (tO - tN))
        stale_out = {k: {"n": v[0], "stale": v[1] / v[0], "captured_mean": float(np.mean(v[2])),
                         "captured_median": float(np.median(v[2]))} for k, v in stale.items()}
        R[fam] = {"n_dec": n, "hist_mismatch": hist_mismatch, "target_mismatch": targ_mismatch,
                  "sup_pred_is_target": int(sup_pred_is_target), "flag_agree": flag_agree,
                  "q5": dict(q5), "q5_flip_types": dict(q5_types), "acc": accs,
                  "q2": {k: dict(v) for k, v in q2.items()}, "q3": q3, "nmae": nmae, "stale": stale_out}
        del data, by, Z
    print(json.dumps({"dir": d, "cpu": round(time.process_time() - t0, 1)}), flush=True)

json.dump(res, open(OUT, "x"), indent=1)
print("done", round(time.process_time() - t0, 1))
