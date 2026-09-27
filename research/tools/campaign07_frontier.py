"""campaign07_frontier: discrimination vs bias on the P2-SCREEN counterfactual octets (extended-07 analyst task).

READ ONLY on the per-decision logs (diag-v1-cf-{SCE,UCE}-<model>[@phi=<c>].jsonl.gz of e07-p2-eval-s4i /
e07-p2-evalX-s4i).  Seed index i pairs P2 seed 40+i with the historical lineage 35+i (evalX design).

Definitions (fixed BEFORE any output of this script was looked at)
-------------------------------------------------------------------
unit      = (octet, decision type); every member of the octet (the 2^3 sub-combinations of the family, keyed "0", "S",
            "SC", ..., full = family) is decided after the SAME visible history, so a model's action can differ between
            two members only through the configuration (public vector / factor channel), never through the history.
unique    = the member's eps-optimal set (label 'opt', eps .5) has exactly one action.
Accuracy endpoints (identical to campaign07_diagscore.membership, the registered screen endpoints):
  flip       = full member of a unit with the campaign06 flip flag and a unique full optimum; ok = greedy action in opt
  near_miss  = the near-miss full-family configuration of a flip unit (unique optimum, inherited from a single, differs
               from the full member's optimum); ok = greedy in opt
  invariance = full member of a non-flip unit
Discrimination (signal-detection style, over member PAIRS with a unique optimum at BOTH ends):
  optchg(p) = the two unique optima differ;  chg(p) = the model's greedy actions differ.
  H  = P(chg | optchg)            (hit rate: the model changes its action where the optimum changes)
  FA = P(chg | not optchg)        (false-alarm rate: it changes where the optimum does not)
  J  = H - FA                     (Youden J; pi* has J = 1 by construction; a policy that changes its action at a
                                   rate independent of whether the optimum changes has J = 0 whatever its rate)
  bias = P(chg) over the same pairs (overall action-change propensity).
  Pair sets (the octet's own ablation structure):
    'sub'  (PRIMARY): full member vs each of its pairs and singles (6 pairs per unit)  -- composition discrimination
    'edge' (secondary): every one-factor edge of the cube, member without f vs member with f (12 per unit, incl. "0")
BA_nm = (flip accuracy on flip units that HAVE a near-miss + near-miss accuracy) / 2  (balanced accuracy on the
        flip/near-miss sets; = (TPR + TNR)/2 of 'act as the full combination requires' vs 'act as the constituent
        does'; 2 BA_nm - 1 is the matching Youden index).
POST HOC (added after the v1 tables were seen, labelled as such in the report):
  H_nm = P(chg between the full member and its near-miss) on flip units with a near-miss (the optimum ALWAYS changes
         there; pi* = 1); J_nm = H_nm - FA_sub (magnitude discrimination: does the model tell the full combination
         from a milder full-family configuration more often than it changes where nothing changes?).
Statistics: two-level bootstrap (seed indices with replacement x whole-octet multinomial weights; the draws of
campaign07_diagscore.draws: 20000 draws, seed 7); each statistic is computed per seed from weighted sums and averaged
over the drawn seeds; paired contrasts use the same draws for both models (seed index paired).
Meaningful margin for discrimination (J_sub, BA_nm): .03 absolute -- the registered P2-SCREEN minimum flip gain.
Reference policies (deterministic, no seed dimension; CI over octets only):
  pi*        : greedy on exact Q* of each member (J = 1).
  add1       : first-order additive: every member with >= 2 factors decided by argmax_a sum_{f in m} Q_f(a) - (|m|-1)Q_0(a)
               (singles and "0" exact).
  add2       : pairwise-additive (P1 Q5): full member by argmax_a sum_pairs Q_p - sum_singles Q_f + Q_0; pairs, singles,
               "0" exact.  Both use exact labels of lower-order members (oracle on the constituents, not a learnable
               policy); no near-miss decision exists for them (its constituents are not labelled).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign07_diagscore as DS  # noqa: E402

TYPES = ("first", "after_probe_solved", "after_probe_failed", "after_b1_timeout", "q2_after_H", "q2_after_notH")
FNAME = re.compile(r"diag-v1-cf-(SCE|UCE)-(.+?)(?:@phi=(\w+))?\.jsonl\.gz$")
MARGIN = 0.03


def base_name(model):
    return re.sub(r"predLRN\d+", "predLRN", re.sub(r"-s\d+$", "", model))


def seed_index(d):
    m = re.search(r"-s4(\d)$", str(Path(d).name))
    if not m:
        raise SystemExit(f"cannot parse seed index from {d}")
    return int(m.group(1))


def members_of(fam):
    return ["".join(T) or "0" for m in range(len(fam) + 1) for T in combinations(fam, m)]


def pair_sets(fam):
    subs = [m for m in members_of(fam) if m not in ("0", fam)]
    sub = [(fam, m) for m in subs]
    edge = []
    for f in fam:
        for m in members_of(fam):
            if f in m:
                continue
            with_f = "".join(x for x in fam if x in m or x == f)
            edge.append((m, with_f))
    return {"sub": sub, "edge": edge}


# ------------------------------------------------------------------------------------------------ extraction

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extract(path, want_q=False):
    """-> {(octet, type, member): (a, opt tuple)}, unit flags, Q (optional)."""
    dec, flags, Q = {}, {}, {}
    with gzip.open(path, "rt") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            c = r["cf"]
            key = (c["octet"], c["type"], c["member"])
            dec[key] = (r["a"], tuple(r["opt"]))
            u = (c["octet"], c["type"])
            if u not in flags:
                flags[u] = (bool(c["flip"]), bool(c["unique_full"]), bool(c["has_near_miss"]))
            if want_q:
                Q[key] = {int(k): v for k, v in r["Q"].items()}
    return dec, flags, Q


# ------------------------------------------------------------------------------------------------ reference policies

def pi_of(q):
    v = max(q.values())
    return min(a for a, x in q.items() if x >= v - 1e-9)


def ref_policies(fam, Q, labels):
    """{'pistar'|'add1'|'add2': {(octet,type,member): (a, opt)}} (no near-miss)."""
    out = {"pistar": {}, "add1": {}, "add2": {}}
    units = defaultdict(dict)
    for (o, t, m), q in Q.items():
        if m != "near_miss":
            units[(o, t)][m] = q
    pairs = ["".join(p) for p in combinations(fam, 2)]
    for (o, t), u in units.items():
        for m, q in u.items():
            opt = labels[(o, t, m)][1]
            a_star = pi_of(q)
            out["pistar"][(o, t, m)] = (a_star, opt)
            # add1
            if m == "0" or len(m) == 1 or any(x not in u for x in list(m) + ["0"]):
                a1 = a_star if (m == "0" or len(m) == 1) else None
            else:
                acts = [a for a in q if all(a in u[x] for x in list(m) + ["0"])]
                a1 = pi_of({a: sum(u[x][a] for x in m) - (len(m) - 1) * u["0"][a] for a in acts}) if acts else None
            if a1 is not None:
                out["add1"][(o, t, m)] = (a1, opt)
            # add2
            if m != fam:
                a2 = a_star
            elif all(x in u for x in pairs + list(fam) + ["0"]):
                acts = [a for a in q if all(a in u[x] for x in pairs + list(fam) + ["0"])]
                a2 = pi_of({a: sum(u[p][a] for p in pairs) - sum(u[f][a] for f in fam) + u["0"][a] for a in acts})
            else:
                a2 = None
            if a2 is not None:
                out["add2"][(o, t, m)] = (a2, opt)
    return out


# ------------------------------------------------------------------------------------------------ cells

ENDPOINT_PARTS = {
    # endpoint: list of (numerator-count, denominator-count) names; combined by COMBINE
    "acc_flip": [("flip_ok", "flip_n")],
    "acc_near_miss": [("nm_ok", "nm_n")],
    "acc_invariance": [("inv_ok", "inv_n")],
    "BA_nm": [("flipnm_ok", "flipnm_n"), ("nm_ok", "nm_n")],
    "H_sub": [("sub_hit", "sub_pos")],
    "FA_sub": [("sub_fa", "sub_neg")],
    "J_sub": [("sub_hit", "sub_pos"), ("sub_fa", "sub_neg")],
    "bias_sub": [("sub_chg", "sub_all")],
    "base_sub": [("sub_pos", "sub_all")],
    "H_edge": [("edge_hit", "edge_pos")],
    "FA_edge": [("edge_fa", "edge_neg")],
    "J_edge": [("edge_hit", "edge_pos"), ("edge_fa", "edge_neg")],
    "bias_edge": [("edge_chg", "edge_all")],
    "acc_sub_pairs": [("sub_bothok", "sub_all")],
    "H_nm": [("nmpair_chg", "nmpair_n")],
    "J_nm": [("nmpair_chg", "nmpair_n"), ("sub_fa", "sub_neg")],
}
COMBINE = {"BA_nm": lambda r: (r[0] + r[1]) / 2, "J_sub": lambda r: r[0] - r[1], "J_edge": lambda r: r[0] - r[1],
           "J_nm": lambda r: r[0] - r[1]}
CONTRAST_ENDPOINTS = ("J_sub", "bias_sub", "H_sub", "FA_sub", "BA_nm", "J_edge", "bias_edge", "acc_flip",
                      "acc_near_miss", "H_nm", "J_nm")


def counts(dec, flags, fam, psets):
    """{stratum: {octet: Counter}} for one model x seed (stratum = 'all' and each decision type)."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def add(t, o, k, v=1.0):
        out["all"][o][k] += v
        out[t][o][k] += v

    units = defaultdict(dict)
    for (o, t, m), v in dec.items():
        units[(o, t)][m] = v
    for (o, t), u in units.items():
        fl = flags.get((o, t))
        if fl is not None and fam in u:
            flip, uniq, has_nm = fl
            a, opt = u[fam]
            if flip and uniq:
                add(t, o, "flip_n"); add(t, o, "flip_ok", a in opt)
                if has_nm and "near_miss" in u:
                    add(t, o, "flipnm_n"); add(t, o, "flipnm_ok", a in opt)
                    add(t, o, "nmpair_n"); add(t, o, "nmpair_chg", a != u["near_miss"][0])
            else:
                add(t, o, "inv_n"); add(t, o, "inv_ok", a in opt)
        if "near_miss" in u:
            a, opt = u["near_miss"]
            add(t, o, "nm_n"); add(t, o, "nm_ok", a in opt)
        for ps, pairs in psets.items():
            for m1, m2 in pairs:
                if m1 not in u or m2 not in u:
                    continue
                (a1, o1), (a2, o2) = u[m1], u[m2]
                if len(o1) != 1 or len(o2) != 1:
                    continue
                pos, chg = o1 != o2, a1 != a2
                add(t, o, f"{ps}_all"); add(t, o, f"{ps}_chg", chg)
                if pos:
                    add(t, o, f"{ps}_pos"); add(t, o, f"{ps}_hit", chg)
                else:
                    add(t, o, f"{ps}_neg"); add(t, o, f"{ps}_fa", chg)
                if ps == "sub":
                    add(t, o, "sub_bothok", (a1 in o1) and (a2 in o2))
    return out


# ------------------------------------------------------------------------------------------------ bootstrap

def mats(cells_by_seed, octets, keys):
    """cells_by_seed: list over seed idx of {octet: counter} -> {key: (S, C) array}."""
    oi = {o: j for j, o in enumerate(octets)}
    S = len(cells_by_seed)
    M = {k: np.zeros((S, len(octets))) for k in keys}
    for s, cells in enumerate(cells_by_seed):
        for o, cnt in cells.items():
            for k in keys:
                M[k][s, oi[o]] = cnt.get(k, 0.0)
    return M


def stat_fn(M, endpoint):
    parts = ENDPOINT_PARTS[endpoint]
    comb = COMBINE.get(endpoint, lambda r: r[0])

    def per_seed_point():
        rs = []
        for n, d in parts:
            num, den = M[n].sum(1), M[d].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rs.append(num / np.where(den > 0, den, np.nan))
        return comb(rs)

    def f(W):
        rs = []
        for n, d in parts:
            dd = W @ M[d].T
            with np.errstate(invalid="ignore", divide="ignore"):
                rs.append((W @ M[n].T) / np.where(dd > 0, dd, np.nan))
        return comb(rs)
    return per_seed_point, f


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def endpoint(M, e, n_boot, seed):
    point, f = stat_fn(M, e)
    per = point()
    ok = ~np.isnan(per)
    res = {"mean": _r(per[ok].mean()) if ok.any() else None, "per_seed": [_r(x) for x in per],
           "support": {k: float(M[k].sum()) for pr in ENDPOINT_PARTS[e] for k in pr}}
    if ok.any() and n_boot > 0:
        v = DS.two_level(f, M[ENDPOINT_PARTS[e][0][1]].shape[0], M[ENDPOINT_PARTS[e][0][1]].shape[1], n_boot, seed)
        res["ci"], res["mc_check"] = DS._ci(v, n_boot)
    return res


def contrast(MA, MB, e, n_boot, seed):
    pa, fa = stat_fn(MA, e)
    pb, fb = stat_fn(MB, e)
    per = pa() - pb()
    ok = ~np.isnan(per)
    res = {"mean": _r(per[ok].mean()) if ok.any() else None, "per_pair": [_r(x) for x in per],
           "signs": {"positive": int((per[ok] > 0).sum()), "negative": int((per[ok] < 0).sum())}}
    if ok.any() and n_boot > 0:
        S, C = MA[ENDPOINT_PARTS[e][0][1]].shape
        v = DS.two_level(lambda W: fa(W) - fb(W), S, C, n_boot, seed)
        res["ci"], res["mc_check"] = DS._ci(v, n_boot)
    return res


# ------------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="e07-p2-eval-s4i / e07-p2-evalX-s4i directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=DS.N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=7)
    ap.add_argument("--families", default="SCE,UCE")
    ap.add_argument("--only", default=None, help="regex on model@phi (smoke)")
    ap.add_argument("--no-per-type-contrasts", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.process_time()
    fams = a.families.split(",")
    files = defaultdict(dict)  # (fam, arm) -> {seed idx: path}
    inputs = {}
    for d in a.dirs:
        si = seed_index(d)
        for p in sorted(Path(d).glob("diag-v1-cf-*.jsonl.gz")):
            m = FNAME.search(p.name)
            if not m or m.group(1) not in fams:
                continue
            arm = f"{base_name(m.group(2))}@{m.group(3) or 'own'}"
            if a.only and not re.search(a.only, arm):
                continue
            if si in files[(m.group(1), arm)]:
                raise SystemExit(f"duplicate {arm} seed {si}: {p}")
            files[(m.group(1), arm)][si] = p
    res = {"version": "e07-frontier-v2", "definitions": __doc__, "margin": MARGIN, "n_boot": a.n_boot,
           "boot_seed": a.boot_seed, "families": {}}
    for fam in fams:
        psets = pair_sets(fam)
        arms = sorted(k[1] for k in files if k[0] == fam)
        C = {}  # arm -> list over seeds of {stratum: {octet: counter}}
        labels = flags0 = Q0 = None
        octets = set()
        for arm in arms:
            per = files[(fam, arm)]
            if sorted(per) != [0, 1, 2, 3, 4]:
                raise SystemExit(f"{fam} {arm}: seeds {sorted(per)}")
            C[arm] = []
            for si in range(5):
                p = per[si]
                inputs[str(p)] = sha256(p)
                dec, flags, Q = extract(p, want_q=Q0 is None)
                if Q0 is None:
                    Q0, flags0, labels = Q, flags, dec
                else:  # labels must agree across files
                    assert flags == flags0, (p, "unit flags differ")
                octets |= {k[0] for k in dec}
                C[arm].append(counts(dec, flags, fam, psets))
            print(json.dumps({"fam": fam, "arm": arm, "cpu_s": round(time.process_time() - t0, 1)}), flush=True)
        refs = ref_policies(fam, Q0, labels)
        for rname, dec in refs.items():
            cnt = counts(dec, flags0, fam, psets)
            C[f"REF-{rname}"] = [cnt] * 5
        octets = sorted(octets)
        strata = ["all"] + list(TYPES)
        keys = sorted({k for pr in ENDPOINT_PARTS.values() for p in pr for k in p})
        fr = {"octets": len(octets), "pair_sets": {k: [list(p) for p in v] for k, v in psets.items()}, "arms": {},
              "contrasts": {}}
        Ms = {arm: {s: mats([c[s] for c in C[arm]], octets, keys) for s in strata} for arm in C}
        for arm in C:
            fr["arms"][arm] = {s: {e: endpoint(Ms[arm][s], e, a.n_boot, a.boot_seed) for e in ENDPOINT_PARTS}
                               for s in strata}
        for refarm in ("S1R1-bank@own", "S0R0-bank@own"):
            if refarm not in C:
                continue
            for arm in C:
                if arm == refarm:
                    continue
                ss = ["all"] if a.no_per_type_contrasts or fam != "SCE" else strata
                fr["contrasts"][f"{arm} - {refarm}"] = {
                    s: {e: contrast(Ms[arm][s], Ms[refarm][s], e, a.n_boot, a.boot_seed)
                        for e in (CONTRAST_ENDPOINTS if s == "all" else ("J_sub", "bias_sub"))}
                    for s in ss}
        # phi contrasts within a model: own - zero, own - mean, exact - own (pooled)
        fr["phi_contrasts"] = {}
        for arm in C:
            if not arm.endswith("@own"):
                continue
            b = arm[:-4]
            for other in ("zero", "mean", "exact"):
                o = f"{b}@{other}"
                if o in C:
                    fr["phi_contrasts"][f"{arm} - {o}"] = {
                        e: contrast(Ms[arm]["all"], Ms[o]["all"], e, a.n_boot, a.boot_seed)
                        for e in ("J_sub", "bias_sub", "H_sub", "FA_sub", "BA_nm", "acc_flip", "acc_near_miss", "H_nm",
                                  "J_nm")}
        res["families"][fam] = fr
        print(json.dumps({"fam": fam, "done": True, "cpu_s": round(time.process_time() - t0, 1)}), flush=True)
    res["inputs_sha256"] = inputs
    res["cpu_s"] = round(time.process_time() - t0, 1)
    (out / "frontier.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"out": str(out), "cpu_s": res["cpu_s"], "files": len(inputs)}))


if __name__ == "__main__":
    main()
