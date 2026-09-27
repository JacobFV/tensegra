"""Extended-07 Phase 1 mechanism analysis (P1-DIAG readouts Q1-Q6) over campaign07_diag.py per-decision logs.

Read-only on every input (diag-v1-*.jsonl.gz, support references, frozen checkpoints, train_meta.json); writes one new
JSON (+ markdown) into --out and refuses to overwrite.  The sha256 of every input file consumed is recorded.

  python research/tools/campaign07_p1analysis.py --dirs DIR_s35 ... --out OUTDIR [--tag b6c] [--code-sha SHA]
      [--n-boot 20000] [--boot-seed 7] [--no-reliance]

Populations (denominators are never mixed):
  cf:<FAM>     balanced counterfactual octets (every member x decision type + near-miss); identical visible histories
               in every model (the cf path IS the common-history protocol for octets)
  pool:A-pistar  pi* histories on the held-out pool replayed through every model (common history, natural states)
  pool:B       LRN's own free-running states (prediction quality only)
Membership (octet decisions): flip = full member of a unit (octet x decision type) with a unique full optimum that
  differs from every single-factor member's (the campaign06 flag); near_miss; inv = full member of a non-flip unit;
  sub-members by level (pair / single / none), split into flip units vs non-flip units.

Q1 prediction accuracy of LRN's 23-d factor prediction: per group (G1..G4 of the factor contract, DEC = G1+G2+G3) the
   tolerance hit (all coordinates of the group within tol) and nMAE (|err| / the support reference's target std;
   constant coordinates excluded), sign error of build_value_rel, argmin error of the one-query strategy costs,
   out-of-range (pred outside LRN's own training-prediction range), by membership x level x context; paired
   within-octet contrasts full - single and full - pair; interaction-carrying coordinates (IX: the full member's exact
   value differs by > tol from EVERY single member's); interaction reproduction (2nd/3rd-order contrasts of pred vs
   target); nearest-member attraction of the full prediction; stale belief after the verifier's reveal (members with C
   at query2_after_H / notH: is the predicted belief_H closer to the no-C member's exact value?); attenuation slopes.
   Tolerances: the registered semantic ones (primary) and LRN's own training error RMS (sensitivity).
Q2 immediate replacement on the same frozen LRN (logged interventions): rescue / harm / net accuracy change /
   action-change rate / change of probability on the optimal set, split by replaced-vector support membership.
Q3 consumer reliance (standalone, read-only checkpoints): policy = pi(tanh(W [z; phi])) recomputed from the logged
   pre-fusion latent z (float16) and phi (LRN: its prediction; SUP: its supplied inputs); perturbations of phi at the
   LRN's own training-error scale (x .5, 1, 2, 4; the SAME per-seed sigma for LRN and SUP), phi -> population mean,
   phi -> another decision's phi (permutation), phi -> 0, and a trunk reference z -> another decision's z; fusion weight
   shares (raw Frobenius, input-scaled, contribution variance).
Q4 flip-decision cross-tab: [LRN pred within tol on the relevant set] x [LRN correct] x [exact replacement fixes /
   breaks / same] x [SUP correct at the identical history] (+ RAWF), on octet flips, near-misses and pi* pool decisions.
Q5 interaction order of the octet labels: per decision type, full optimum vs every single / every pair / none.
Q6 = Q1/Q2/Q4/Q5 on any further family present (UCE on b6c) + training exposure of each arm to the family's pairs.
Statistics: two-level bootstrap (seed with replacement x shared clusters: whole octets for cf, configurations for pool),
campaign07_diagscore.boot_endpoint / boot_contrast (percentile 95% CI, half-split MC check), per-seed values.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign07_diagscore as DS  # noqa: E402

VERSION = "p1analysis-v1.1"  # v1.1: Q4 cross-tab strata per population (bug fix after the v1 b6c run)
KINDS = ("LRN", "SUP", "RAWF")
IVS = ("exact", "iso:G1", "iso:G2", "iso:G3", "iso:G4", "dep:G1", "dep:G2", "dep:G3", "dep:G4", "up:G3", "up:G4",
       "keep:G1", "keep:G2", "keep:G3", "keep:G4", "gauss:1", "gauss_pred:1", "scale_err:0.5", "mirror", "pred")
GROUPS_EXTRA = ("DEC", "ALL")
STRAT = DS.STRAT
PERTURB = ("gauss0.5", "gauss1", "gauss2", "gauss4", "mean", "perm", "zero", "z_perm", "exact", "lrn_pred")
_SHA: dict = {}
REMAP: list = []  # (old prefix, new prefix): local development copies only


def rp(path):
    path = str(path)
    for old, new in REMAP:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def sha256(p):
    p = str(p)
    if p not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA[p] = h.hexdigest()
    return _SHA[p]


def _r(x, d=6):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


# ------------------------------------------------------------------------------------------------ loading

def level_of(member):
    if member == "near_miss":
        return "near_miss"
    if member == "0":
        return "none"
    return {1: "single", 2: "pair"}.get(len(member), "full")


def compact(r, head, keep_z):
    """Keep only what the analysis needs from one decision record."""
    opt = r["opt"]
    probs = r["probs"]
    c = {"cid": r["cfg_id"], "ctx": r["ctx"], "t": r["t"], "ws": r.get("ws"), "ok": bool(r["ok"]), "a": r["a"],
         "opt": opt, "avail": r["avail"], "p_opt": sum(probs[b] for b in opt), "target": np.asarray(r["target"]),
         "pred": np.asarray(r["pred"]) if r.get("pred") is not None else None, "margin": r["margin"],
         "Q": {int(k): v for k, v in r["Q"].items()}}
    sp = r.get("support_pred")
    c["oob"] = None if sp is None else bool(sp["n_out"] > 0)
    c["pred_in"] = None if sp is None else bool(sp["in"])
    if keep_z and "z_pre" in r:
        c["z"] = r["z_pre"]
    cf = r.get("cf")
    if cf is not None:
        c.update(fam=cf["family"], octet=cf["octet"], type=cf["type"], member=cf["member"], role=cf["role"],
                 flip=bool(cf["flip"]), flip_abl=bool(cf.get("flip_ablations")), unique=bool(cf["unique_full"]),
                 level=level_of(cf["member"]), cluster=f"{cf['family']}:{cf['octet']}")
    else:
        c["cluster"] = r["cfg_id"]
    if r.get("iv"):
        c["iv"] = {n: (v["a"], bool(v["ok"]), v.get("support", {}).get("in"), v["p_opt"]) for n, v in r["iv"].items()}
    return c


def load(path, keep_z=False):
    with gzip.open(path, "rt") as f:
        head = json.loads(f.readline())["_meta"]
        recs = []
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "decision":
                recs.append(compact(r, head, keep_z))
    return head, recs


def find_files(d):
    """{(population, kind): path} for one seed directory."""
    out = {}
    for f in sorted(Path(d).iterdir()):
        m = re.match(r"diag-v1-(cf-[A-Z]+|A-pistar|B)-(LRN|SUP|RAWF)-s(\d+)\.jsonl\.gz$", f.name)
        if m:
            proto, kind, seed = m.groups()
            pop = f"cf:{proto[3:]}" if proto.startswith("cf-") else f"pool:{proto}"
            if pop == "pool:B" and kind != "LRN":
                continue
            out[(pop, kind)] = f
    return out


# ------------------------------------------------------------------------------------------------ accumulators

class Acc:
    """population -> stratum -> endpoint -> {(seed, cluster): [num, den]}; point sums for descriptive tables."""

    def __init__(self):
        self.e = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(2)))))
        self.universe = defaultdict(lambda: [set(), set()])

    def add(self, pop, stratum, endpoint, seed, cluster, num, den=1.0):
        v = self.e[pop][stratum][endpoint][(seed, cluster)]
        v[0] += num
        v[1] += den
        u = self.universe[pop]
        u[0].add(seed)
        u[1].add(cluster)


def membership(c):
    if "role" not in c:
        return "pool"
    if c["role"] == "near_miss":
        return "near_miss"
    unit_flip = c["flip"] and c["unique"]
    if c["role"] == "full":
        return "flip" if unit_flip else "inv"
    return ("flipunit_" if unit_flip else "nonflip_") + c["level"]


# ------------------------------------------------------------------------------------------------ Q1

class Ctx:
    def __init__(self, feats, groups, tol_sem, sup):
        self.feats = feats
        self.fi = {f: j for j, f in enumerate(feats)}
        self.groups = dict(groups)
        self.groups["DEC"] = groups["G1"] + groups["G2"] + groups["G3"]
        self.groups["ALL"] = list(feats)
        self.gix = {g: np.array([self.fi[c] for c in cs]) for g, cs in self.groups.items()}
        self.tol_sem = tol_sem
        self.tstd = np.asarray(sup["target_std"])
        self.nonconst = self.tstd > 1e-6
        self.tol_err = np.asarray(sup["err_rms"])  # sensitivity: LRN's own training error scale
        self.lo, self.hi = np.asarray(sup["min"]), np.asarray(sup["max"])
        self.bv = self.fi["build_value_rel"]
        self.strat = [self.fi[c] for c in STRAT]
        self.probs = [self.fi[c] for c in DS.PROB_COORDS]
        self.bh = self.fi["belief_H"]


def pred_metrics(c, X):
    """Per-decision prediction metrics (dict endpoint -> (num, den))."""
    p, t = c["pred"], c["target"]
    err = np.abs(p - t)
    nerr = err / np.maximum(X.tstd, 1e-3)
    out = {}
    for tolname, tol in (("", X.tol_sem), ("E", X.tol_err)):
        hit = err <= tol
        for g, ix in X.gix.items():
            out[f"hit{tolname}_{g}"] = (float(hit[ix].all()), 1.0)
            ixn = ix[X.nonconst[ix]]
            out[f"chit{tolname}_{g}"] = (float(hit[ixn].mean()), 1.0)  # share of (non-constant) coordinates in tol
    for g, ix in X.gix.items():
        ixn = ix[X.nonconst[ix]]
        out[f"nmae_{g}"] = (float(nerr[ixn].mean()), 1.0)
    j = X.bv
    if abs(t[j]) > X.tol_sem[j]:
        out["sign_bv"] = (float((p[j] > 0) != (t[j] > 0)), 1.0)
    tt = t[X.strat]
    srt = np.sort(tt)
    if srt[1] - srt[0] > X.tol_sem[X.strat[0]]:
        out["argmin_strat"] = (float(np.argmin(p[X.strat]) != np.argmin(tt)), 1.0)
    out["prob_invalid"] = (float(((p[X.probs] < -1e-6) | (p[X.probs] > 1 + 1e-6)).any()), 1.0)
    out["target_at_clip"] = (float((np.abs(t) >= 4.999).any()), 1.0)  # +-5 clip of the relative cost coordinates
    out["oob_range"] = (float(((p < X.lo - 1e-9) | (p > X.hi + 1e-9)).any()), 1.0)
    if c.get("pred_in") is not None:
        out["pred_out_support"] = (float(not c["pred_in"]), 1.0)
    out["acc"] = (float(c["ok"]), 1.0)
    return out


COORD_STRATA = ("flip", "near_miss", "inv", "level=full", "level=pair", "level=single", "level=none", "all")


def q1_add(acc, pop, seed, c, X, strata, dcoord):
    m = pred_metrics(c, X)
    for s in strata:
        for e, (n, d) in m.items():
            acc.add(pop, s, e, seed, c["cluster"], n, d)
        if s in COORD_STRATA:
            err = np.abs(c["pred"] - c["target"])
            k = dcoord[(pop, s)]
            k["n"] += 1
            k["abs"] = k.get("abs", 0) + err
            k["hit"] = k.get("hit", 0) + (err <= X.tol_sem)
            k["hitE"] = k.get("hitE", 0) + (err <= X.tol_err)
            k["bias"] = k.get("bias", 0) + (c["pred"] - c["target"])
    return m


def cf_units(recs):
    """(octet, type) -> {member: rec} (near-miss under 'near_miss')."""
    u = defaultdict(dict)
    for c in recs:
        u[(c["octet"], c["type"])][c["member"]] = c
    return u


def interaction_coords(unit, fam, X):
    """IX: coordinates whose full-member exact value differs by > tol from every single member's."""
    full = unit.get(fam)
    singles = [unit.get(f) for f in fam]
    if full is None or any(s is None for s in singles):
        return None
    d = np.min(np.stack([np.abs(full["target"] - s["target"]) for s in singles]), 0)
    return np.where(d > X.tol_sem)[0]


def contrast_terms(unit, fam):
    """Signed coefficients of the 3rd-order and 2nd-order (each pair) interaction contrasts over the members."""
    out = {}
    subsets = [("".join(T) or "0") for m in range(len(fam) + 1) for T in combinations(fam, m)]
    if any(s not in unit for s in subsets):
        return None
    out["3rd"] = {s: (-1) ** (len(fam) - (0 if s == "0" else len(s))) for s in subsets}
    for pair in combinations(fam, 2):
        a, b = pair
        out["2nd:" + a + b] = {a + b: 1, a: -1, b: -1, "0": 1}
    return out


def q1_cf_extra(acc, pop, seed, recs, fam, X, desc):
    """IX errors, interaction reproduction, attraction, stale belief, attenuation (LRN on cf records)."""
    units = cf_units(recs)
    for (octet, typ), unit in units.items():
        full = unit.get(fam)
        if full is None:
            continue
        mem = membership(full)
        cl = full["cluster"]
        ix = interaction_coords(unit, fam, X)
        if ix is not None:
            desc["ix_size"][mem].append(len(ix))
            if len(ix):
                err = np.abs(full["pred"] - full["target"])[ix]
                for s in (mem, "full_all"):
                    acc.add(pop, s, "hit_IX", seed, cl, float((err <= X.tol_sem[ix]).all()))
                    acc.add(pop, s, "hitE_IX", seed, cl, float((err <= X.tol_err[ix]).all()))
                    nx = ix[X.nonconst[ix]]
                    if len(nx):
                        acc.add(pop, s, "nmae_IX", seed, cl,
                                float((np.abs(full["pred"] - full["target"])[nx] / np.maximum(X.tstd[nx], 1e-3)).mean()))
                for j in ix:
                    desc["ix_coord"][mem][X.feats[j]] += 1
        # nearest-member attraction (every member; DEC coordinates, normalized L1; ties count as own): is a member's
        # prediction closest to its OWN exact vector, or to another member's (e.g. the full member's to a pair's)?
        ixd = X.gix["DEC"][X.nonconst[X.gix["DEC"]]]
        sc = np.maximum(X.tstd[ixd], 1e-3)
        for km, cm in unit.items():
            if km == "near_miss":
                continue
            dists = {k: float((np.abs(cm["pred"][ixd] - m["target"][ixd]) / sc).mean()) for k, m in unit.items()
                     if k != "near_miss"}
            dmin = min(dists.values())
            own = dists[km] <= dmin + 1e-12
            near = km if own else min(dists, key=dists.get)
            lv = level_of(near)
            for s in (f"att|{membership(cm)}", f"att|level={cm['level']}") + ((mem, "full_all") if km == fam else ()):
                acc.add(pop, s, "attract_own", seed, cl, float(own))
                acc.add(pop, s, "attract_full", seed, cl, float((not own) and lv == "full"))
                acc.add(pop, s, "attract_pair", seed, cl, float((not own) and lv == "pair"))
                acc.add(pop, s, "attract_single", seed, cl, float((not own) and lv in ("single", "none")))
        # interaction reproduction: contrasts of pred vs target
        terms = contrast_terms(unit, fam)
        if terms:
            for name, coef in terms.items():
                It = sum(w * unit[s]["target"] for s, w in coef.items())
                Ip = sum(w * unit[s]["pred"] for s, w in coef.items())
                d = desc["contrast"][name]
                big = np.abs(It) > X.tol_sem
                d["n"] += 1
                d["sum_t2"] += np.where(big, It * It, 0.0)
                d["sum_e2"] += np.where(big, (Ip - It) ** 2, 0.0)
                d["sum_tp"] += np.where(big, It * Ip, 0.0)
                d["nbig"] += big
                d["abs_t"] += np.abs(It)
    # stale belief after the verifier's reveal: members with C at query2 contexts
    if "C" in fam:
        for (octet, typ), unit in units.items():
            if typ not in ("q2_after_H", "q2_after_notH"):
                continue
            for k, c in unit.items():
                if k == "near_miss" or "C" not in k:
                    continue
                k0 = k.replace("C", "") or "0"
                if k0 not in unit:
                    continue
                t, t0, p = c["target"][X.bh], unit[k0]["target"][X.bh], c["pred"][X.bh]
                if abs(t - t0) <= 0.05:
                    continue
                s = f"{typ}|{c['level']}"
                acc.add(pop, s, "stale", seed, c["cluster"], float(abs(p - t0) < abs(p - t)))
                acc.add(pop, s, "update_captured", seed, c["cluster"], float(np.clip((p - t0) / (t - t0), -1, 2)))
                acc.add(pop, s, "err_bH", seed, c["cluster"], float(abs(p - t)))
                acc.add(pop, s, "acc", seed, c["cluster"], float(c["ok"]))
    # attenuation (pooled, descriptive): per level, least-squares slope of pred on target per coordinate
    for c in recs:
        mem = membership(c)
        d = desc["slope"][mem]
        d["n"] += 1
        d["st"] += c["target"]
        d["sp"] += c["pred"]
        d["stt"] += c["target"] ** 2
        d["stp"] += c["target"] * c["pred"]


def new_desc(nf):
    z = lambda: np.zeros(nf)  # noqa: E731
    return {"ix_size": defaultdict(list), "ix_coord": defaultdict(lambda: defaultdict(int)),
            "contrast": defaultdict(lambda: {"n": 0, "sum_t2": z(), "sum_e2": z(), "sum_tp": z(), "nbig": z(),
                                              "abs_t": z()}),
            "slope": defaultdict(lambda: {"n": 0, "st": z(), "sp": z(), "stt": z(), "stp": z()})}


def q1_strata(c):
    mem = membership(c)
    ctx = c["ctx"]
    if mem == "pool":
        return ["all", f"ctx={ctx}"]
    s = [mem, f"level={c['level']}", f"ctx={ctx}|{mem}", "all"]
    if mem == "flip":
        s.append("flip_vs_pairs" if c["flip_abl"] else "flip_pair_inherited")
    if mem in ("flip", "inv"):
        s += ["full_all", f"ctx={ctx}|full_all"]
    if mem.startswith("flipunit_") or mem == "flip":
        s.append(f"flipunit|level={'full' if mem == 'flip' else c['level']}")
    return s


# ------------------------------------------------------------------------------------------------ Q2

def q2_add(acc, pop, seed, c, strata, counts):
    for name, (a, ok, sin, popt) in (c.get("iv") or {}).items():
        base_ok = c["ok"]
        changed = a != c["a"]
        sup = "in" if sin else ("out" if sin is not None else "na")
        for s in strata:
            acc.add(pop, s, f"dacc:{name}", seed, c["cluster"], float(ok) - float(base_ok))
            acc.add(pop, s, f"change:{name}", seed, c["cluster"], float(changed))
            acc.add(pop, s, f"dpopt:{name}", seed, c["cluster"], popt - c["p_opt"])
            acc.add(pop, s, f"dacc_{sup}:{name}", seed, c["cluster"], float(ok) - float(base_ok))
            k = counts[(pop, s, name)]
            k["n"] += 1
            k[f"n_{sup}"] += 1
            k["changed"] += changed
            k[f"changed_{sup}"] += changed
            if not base_ok and ok:
                k["rescue"] += 1
                k[f"rescue_{sup}"] += 1
            if base_ok and not ok:
                k["harm"] += 1
                k[f"harm_{sup}"] += 1
            k[f"seed{seed}_net"] += float(ok) - float(base_ok)
            k[f"seed{seed}_n"] += 1


# ------------------------------------------------------------------------------------------------ Q3

def load_weights(run):
    import torch
    sd = torch.load(Path(run) / "model.pt", map_location="cpu")
    return {"W": sd["fuse.weight"].numpy().astype(np.float64), "b": sd["fuse.bias"].numpy().astype(np.float64),
            "P": sd["pi.weight"].numpy().astype(np.float64), "pb": sd["pi.bias"].numpy().astype(np.float64)}


def policy(w, Z, PHI, mask):
    h = np.tanh(np.concatenate([Z, PHI], 1) @ w["W"].T + w["b"])
    lg = h @ w["P"].T + w["pb"]
    lg = np.where(mask, lg, -1e9)
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    return lg, p


def decode_z(b64s):
    import base64
    return np.stack([np.frombuffer(base64.b64decode(s), dtype=np.float16) for s in b64s]).astype(np.float64)


def reliance(recs, w, sigma, rng, kind, alt_phi=None):
    """Per-decision perturbation outcomes for one model on one population. Returns dict name -> arrays.
    alt_phi (SUP only): the same-seed LRN's predicted factors at the identical history ('lrn_pred': SUP's consumer
    reading LRN's estimates instead of its exact inputs)."""
    Z = decode_z([c["z"] for c in recs])
    PHI = np.stack([c["pred"] for c in recs])
    T = np.stack([c["target"] for c in recs])
    n_act = w["P"].shape[0]
    mask = np.zeros((len(recs), n_act), bool)
    for i, c in enumerate(recs):
        mask[i, c["avail"]] = True
    opt = np.zeros((len(recs), n_act), bool)
    for i, c in enumerate(recs):
        opt[i, c["opt"]] = True
    lg0, p0 = policy(w, Z, PHI, mask)
    a0 = lg0.argmax(1)
    rec_a = np.array([c["a"] for c in recs])
    ok0 = opt[np.arange(len(recs)), a0]
    perm = rng.permutation(len(recs))
    out = {"fidelity": (a0 == rec_a).astype(float), "base_ok": ok0.astype(float)}
    nz = w["W"].shape[1] - PHI.shape[1]
    # weight shares
    Wz, Wf = w["W"][:, :nz], w["W"][:, nz:]
    sz, sf = Z.std(0), PHI.std(0)
    cz, cf_ = Z @ Wz.T, PHI @ Wf.T
    shares = {"fro_factor_share": float((Wf ** 2).sum() / (w["W"] ** 2).sum()),
              "scaled_factor_share": float(((Wf * sf) ** 2).sum() / (((Wz * sz) ** 2).sum() + ((Wf * sf) ** 2).sum())),
              "contrib_var_factor_share": float(cf_.var(0).sum() / (cz.var(0).sum() + cf_.var(0).sum())),
              "n_factor_cols": int(Wf.shape[1]), "n_trunk_cols": int(nz)}
    for name in PERTURB:
        if (name == "exact" and kind != "LRN") or (name == "lrn_pred" and alt_phi is None):
            continue
        Zp, PHIp = Z, PHI
        if name.startswith("gauss"):
            s = float(name[5:])
            PHIp = PHI + s * sigma * rng.standard_normal(PHI.shape)
        elif name == "mean":
            PHIp = np.broadcast_to(PHI.mean(0), PHI.shape)
        elif name == "perm":
            PHIp = PHI[perm]
        elif name == "zero":
            PHIp = np.zeros_like(PHI)
        elif name == "z_perm":
            Zp = Z[perm]
        elif name == "exact":
            PHIp = T
        elif name == "lrn_pred":
            PHIp = alt_phi
        lg, p = policy(w, Zp, PHIp, mask)
        a = lg.argmax(1)
        ok = opt[np.arange(len(recs)), a]
        out[f"change:{name}"] = (a != a0).astype(float)
        out[f"dacc:{name}"] = ok.astype(float) - ok0
        out[f"acc:{name}"] = ok.astype(float)
        out[f"tv:{name}"] = 0.5 * np.abs(p - p0).sum(1)
        dl = np.where(mask, lg - lg0, 0.0)
        out[f"dlogit:{name}"] = np.abs(dl - dl.sum(1, keepdims=True) / mask.sum(1, keepdims=True) * mask).max(1)
    return out, shares


# ------------------------------------------------------------------------------------------------ Q4

def q4_xtab(lrn, sup, rawf, X, fam, units_by_seed, seed, tabs, want):
    """lrn/sup/rawf: key -> rec at identical histories.  want(c) -> list of strata (may be empty)."""
    for key, c in lrn.items():
        ss = want(c)
        if not ss or key not in sup:
            continue
        err = np.abs(c["pred"] - c["target"])
        rel = {g: X.gix[g] for g in ("DEC", "G1", "G2", "G3")}
        if fam and "octet" in c and c["role"] == "full":
            ix = interaction_coords(units_by_seed[(c["octet"], c["type"])], fam, X)
            rel["IX"] = ix if ix is not None else None
        ex = c.get("iv", {}).get("exact")
        fix = "na" if ex is None else ("fixes" if (not c["ok"] and ex[1]) else ("breaks" if (c["ok"] and not ex[1]) else "same"))
        for rname, ix in rel.items():
            if ix is None:
                continue
            tin = "vacuous" if len(ix) == 0 else ("in" if (err[ix] <= X.tol_sem[ix]).all() else "out")
            tinE = "vacuous" if len(ix) == 0 else ("in" if (err[ix] <= X.tol_err[ix]).all() else "out")
            for tn, tv in (("sem", tin), ("errscale", tinE)):
                k = (f"pred_{tv}|lrn_{'ok' if c['ok'] else 'wrong'}|exact_{fix}|sup_{'ok' if sup[key]['ok'] else 'wrong'}"
                     + (f"|rawf_{'ok' if rawf[key]['ok'] else 'wrong'}" if key in rawf else ""))
                for s in ss:
                    tabs[(s, rname, tn)][k] += 1
                    tabs[(s, rname, tn)][f"_seed{seed}_n"] += 1


def q4_strata(c):
    mem = membership(c)
    if mem == "flip":
        return ["flip", "flip_vs_pairs" if c["flip_abl"] else "flip_pair_inherited"]
    return [mem] if mem in ("near_miss", "inv") else []


def q4_summary(tab):
    """Rows (pred in/out/vacuous, LRN ok/wrong) -> n, exact fixes / breaks, SUP ok, RAWF ok."""
    rows = defaultdict(lambda: defaultdict(int))
    for k, n in tab.items():
        if k.startswith("_"):
            continue
        f = dict(x.split("_", 1) for x in k.split("|"))
        r = rows[f"pred_{f['pred']}|lrn_{f['lrn']}"]
        r["n"] += n
        r["exact_fixes"] += n * (f["exact"] == "fixes")
        r["exact_breaks"] += n * (f["exact"] == "breaks")
        r["sup_ok"] += n * (f["sup"] == "ok")
        r["rawf_ok"] += n * (f.get("rawf") == "ok")
        r["sup_ok_and_exact_same"] += n * (f["sup"] == "ok" and f["exact"] == "same")
    return {k: dict(v) for k, v in sorted(rows.items())}


# ------------------------------------------------------------------------------------------------ Q5

def pi_of(q):
    v = max(q.values())
    return min(a for a, x in q.items() if x >= v - 1e-9)


def q5(recs, fam, eps=0.5):
    units = cf_units(recs)
    pairs = ["".join(p) for p in combinations(fam, 2)]
    rows = defaultdict(lambda: defaultdict(int))
    agree = [0, 0]
    for (octet, typ), u in units.items():
        need = [fam, "0"] + list(fam) + pairs
        if any(k not in u for k in need):
            rows[typ]["incomplete"] += 1
            continue
        full = u[fam]
        O = set(full["opt"])
        pis = {k: pi_of(u[k]["Q"]) for k in need}
        flip_s = all(pis[f] not in O for f in fam)
        flip_p = all(pis[p] not in O for p in pairs)
        agree[0] += 1
        agree[1] += (flip_s == full["flip"]) and (flip_p == full["flip_abl"])
        r = rows[typ]
        r["units"] += 1
        uniq = len(O) == 1
        r["unique_full"] += uniq
        r["changed_vs_none"] += pis["0"] not in O
        r["flip_vs_singles"] += flip_s and uniq
        r["flip_vs_pairs"] += flip_p and uniq
        r["flip_vs_singles_and_pairs"] += flip_s and flip_p and uniq
        r["flip_vs_singles_not_pairs"] += flip_s and (not flip_p) and uniq
        r["flip_vs_pairs_not_singles"] += flip_p and (not flip_s) and uniq
        for p in pairs:
            if flip_s and uniq and pis[p] in O:
                r[f"inherited_from_{p}"] += 1
        # disjoint eps-optimal sets (stricter): no single / pair eps-optimal action is eps-optimal for the full member
        dis_s = all(not (set(u[f]["opt"]) & O) for f in fam)
        dis_p = all(not (set(u[p]["opt"]) & O) for p in pairs)
        r["strict_flip_vs_singles"] += dis_s and uniq
        r["strict_flip_vs_singles_and_pairs"] += dis_s and dis_p and uniq
        # pair-level (2nd-order) flips: a pair's optimum differs from both of its singles
        r["any_pair_flips_vs_its_singles"] += any(pis[p[0]] not in set(u[p]["opt"]) and pis[p[1]] not in set(u[p]["opt"])
                                                 and len(u[p]["opt"]) == 1 for p in pairs)
        # full optimum differs from the purely additive-in-Q prediction (Q_S + Q_C + Q_E - Q_SC - ... + Q_0 inverse)
        acts = sorted(full["Q"])
        qadd = {a: sum(u[p]["Q"][a] for p in pairs) - sum(u[f]["Q"][a] for f in fam) + u["0"]["Q"][a] for a in acts
                if all(a in u[k]["Q"] for k in need)}
        if qadd:
            r["pairwise_Q_extrapolation_opt"] += max(qadd, key=lambda a: (qadd[a], -a)) in O
            r["pairwise_Q_extrapolation_n"] += 1
    return {t: dict(v) for t, v in rows.items()}, {"checked": agree[0], "agree_with_recorded_flags": agree[1]}


# ------------------------------------------------------------------------------------------------ main pass

def analyse(dirs, n_boot, boot_seed, do_reliance):
    t0 = time.process_time()
    acc = Acc()
    counts2 = defaultdict(lambda: defaultdict(float))
    tabs4 = defaultdict(lambda: defaultdict(int))
    q5res, exposure, rel_shares = {}, {}, defaultdict(dict)
    desc = {}
    training_ref = {}
    dcoord = defaultdict(lambda: {"n": 0})
    inputs = {}
    fams = set()
    meta = None
    for d in dirs:
        files = find_files(d)
        heads = {}
        # support reference + contract context from the LRN header
        pops = sorted({p for p, _ in files})
        for key, path in files.items():
            with gzip.open(path, "rt") as f:
                heads[key] = json.loads(f.readline())["_meta"]
        anyh = next(iter(heads.values()))
        lrn_name = next(n for n, m in anyh["models"].items() if m["kind"] == "LRN")
        seed = anyh["models"][lrn_name]["seed"]
        sup_path = rp(anyh["models"][lrn_name]["support_ref"])
        sup = json.loads(Path(sup_path).read_text())
        inputs[sup_path] = sha256(sup_path)
        feats, groups = anyh["features"], anyh["groups"]
        tol = DS.semantic_tol(feats)
        X = Ctx(feats, groups, np.array([tol[f] for f in feats]), sup)
        meta = meta or {"features": feats, "groups": X.groups, "group_source": anyh["group_source"],
                        "tol_semantic": tol, "tstd_by_seed": {}}
        meta["tstd_by_seed"][seed] = sup["target_std"]
        tr = {g: _r(np.mean((np.asarray(sup["err_mae"])[ix] / np.maximum(X.tstd[ix], 1e-3))[X.nonconst[ix]]))
              for g, ix in X.gix.items()}
        training_ref[seed] = {"nmae_train_states": tr, "n_states": sup["n_states"], "pool": sup["pool"]}
        # training exposure of every arm
        for n, m in anyh["models"].items():
            tm = Path(rp(m["run"])) / "train_meta.json"
            if not tm.exists():
                exposure[n] = {"kind": m["kind"], "missing": str(tm)}
                continue
            inputs[str(tm)] = sha256(tm)
            mt = json.loads(tm.read_text())
            exposure[n] = {"kind": m["kind"], "train_split": mt.get("train_split"),
                           "train_combo_counts": mt.get("train_combo_counts"), "model_sha256": m["model_sha256"]}
        weights = {}
        if do_reliance:
            for n, m in anyh["models"].items():
                if m["kind"] in ("LRN", "SUP"):
                    mp = Path(rp(m["run"])) / "model.pt"
                    inputs[str(mp)] = sha256(mp)
                    assert inputs[str(mp)] == m["model_sha256"], f"checkpoint changed: {mp}"
                    weights[m["kind"]] = load_weights(rp(m["run"]))
        for pop in pops:
            data = {}
            for kind in KINDS:
                if (pop, kind) in files:
                    p = files[(pop, kind)]
                    inputs[str(p)] = sha256(p)
                    _, data[kind] = load(p, keep_z=do_reliance and kind in ("LRN", "SUP") and pop != "pool:B")
            lrn = data.get("LRN", [])
            fam = None
            if pop.startswith("cf:"):
                fam = pop[3:]
                fams.add(fam)
                desc.setdefault(pop, new_desc(len(feats)))
            # Q1
            for c in lrn:
                q1_add(acc, "Q1|" + pop, seed, c, X, q1_strata(c), dcoord)
            if fam:
                q1_cf_extra(acc, "Q1x|" + pop, seed, lrn, fam, X, desc[pop])
            # Q2
            if pop != "pool:B":
                for c in lrn:
                    mem = membership(c)
                    st = ["all"] + ([mem] if mem in ("flip", "near_miss", "inv") else []) + (
                        [("flip_vs_pairs" if c["flip_abl"] else "flip_pair_inherited")] if mem == "flip" else []) + (
                        [f"ctx={c['ctx']}|{mem}"] if mem in ("flip", "near_miss") else [])
                    q2_add(acc, "Q2|" + pop, seed, c, st, counts2)
            # Q4
            if pop != "pool:B" and "SUP" in data:
                keyf = (lambda c: (c["cid"], c["type"])) if fam else (lambda c: (c["cid"], c["ws"], c["t"]))
                L = {keyf(c): c for c in lrn}
                S = {keyf(c): c for c in data["SUP"]}
                R = {keyf(c): c for c in data.get("RAWF", [])}
                units = cf_units(lrn) if fam else None
                # v1.1 fix: strata carry the population (v1 pooled cf:SCE and cf:UCE decisions in one cross-tab)
                want = (lambda c, _p=pop: [f"{_p}:{s}" for s in q4_strata(c)]) if fam else (lambda c, _p=pop: [f"{_p}:pistar_all"])
                q4_xtab(L, S, R, X, fam, units, seed, tabs4, want)
                # decision accuracy of the three arms on the same strata (seed-paired contrasts later)
                for kind, recs in data.items():
                    for c in recs:
                        mem = membership(c)
                        for s in (["all", mem] if mem in ("flip", "near_miss", "inv") else ["all"]) + (
                                [("flip_vs_pairs" if c["flip_abl"] else "flip_pair_inherited")] if mem == "flip" else []):
                            acc.add(f"ACC|{pop}|{kind}", s, "acc", seed, c["cluster"], float(c["ok"]))
            # Q5
            if fam and lrn:
                q5res[f"{pop}|s{seed}"] = q5(lrn, fam)
            # Q3
            if do_reliance and pop != "pool:B":
                rng_base = [seed, sorted(pops).index(pop)]
                sigma = np.asarray(sup["err_rms"])
                keyq = (lambda c: (c["cid"], c["type"])) if fam else (lambda c: (c["cid"], c["ws"], c["t"]))
                lrn_by = {keyq(c): c for c in lrn}
                for kind in ("LRN", "SUP"):
                    if kind not in data or kind not in weights:
                        continue
                    recs = data[kind]
                    subsets = {"all": recs}
                    if fam:
                        subsets["flip"] = [c for c in recs if membership(c) == "flip"]
                    for sname, rs in subsets.items():
                        if not rs:
                            continue
                        rng = np.random.default_rng(rng_base + [KINDS.index(kind), ["all", "flip"].index(sname)])
                        alt = None
                        if kind == "SUP":
                            alt_c = [lrn_by.get(keyq(c)) for c in rs]
                            assert all(a is not None for a in alt_c), "SUP decision without an LRN twin"
                            mism = sum(not np.allclose(a["target"], c["target"], atol=2e-6) for a, c in zip(alt_c, rs))
                            assert mism == 0, f"{mism} SUP/LRN twins differ in their exact factors"
                            alt = np.stack([a["pred"] for a in alt_c])
                        out, shares = reliance(rs, weights[kind], sigma, rng, kind, alt)
                        cls = [c["cluster"] for c in rs]
                        for mname, vals in out.items():
                            for cl, v in zip(cls, vals):
                                acc.add(f"Q3|{pop}|{sname}|{kind}", "x", mname, seed, cl, float(v))
                        if sname == "all":
                            rel_shares[f"{pop}|{kind}"][seed] = shares
            del data, lrn
        print(json.dumps({"dir": str(d), "seed": seed, "cpu_s": round(time.process_time() - t0, 1)}), flush=True)
    meta["tstd_mean"] = np.mean(list(meta["tstd_by_seed"].values()), 0).tolist()
    return {"acc": acc, "counts2": counts2, "tabs4": tabs4, "q5": q5res, "exposure": exposure, "shares": rel_shares,
            "desc": desc, "dcoord": dcoord, "training_ref": training_ref, "inputs": inputs, "fams": sorted(fams), "meta": meta,
            "cpu_pass": time.process_time() - t0}


# ------------------------------------------------------------------------------------------------ bootstrap / output

def boot_all(acc, n_boot, boot_seed, only=None):
    """Every endpoint of every (population, stratum), over the population's full seed x cluster universe."""
    out = {}
    for pop in sorted(acc.e):
        seeds, cls = sorted(acc.universe[pop][0]), sorted(acc.universe[pop][1])
        for s in sorted(acc.e[pop]):
            for e in sorted(acc.e[pop][s]):
                if only and not only(pop, s, e):
                    continue
                out[f"{pop}||{s}||{e}"] = DS.boot_endpoint(acc.e[pop][s][e], n_boot, boot_seed, seeds, cls)
    return out


def boot_pairs(acc, pairs, n_boot, boot_seed):
    """Paired contrasts A - B: (name, (popA, stratumA, endpointA), (popB, stratumB, endpointB)); seeds common,
    clusters the union (shared weights)."""
    out = {}
    for name, a, b in pairs:
        ca = acc.e[a[0]].get(a[1], {}).get(a[2])
        cb = acc.e[b[0]].get(b[1], {}).get(b[2])
        if not ca or not cb:
            continue
        out[name] = DS.boot_contrast(ca, cb, n_boot, boot_seed)
    return out


def contrast_list(R, fams):
    pairs = []
    for fam in fams:
        pop = f"Q1|cf:{fam}"
        for g in ("G1", "G2", "G3", "G4", "DEC"):
            for e in (f"nmae_{g}", f"chit_{g}", f"chitE_{g}"):
                pairs.append((f"{pop}|flipunit full-single|{e}", (pop, "flipunit|level=full", e),
                              (pop, "flipunit|level=single", e)))
                pairs.append((f"{pop}|flipunit full-pair|{e}", (pop, "flipunit|level=full", e),
                              (pop, "flipunit|level=pair", e)))
                pairs.append((f"{pop}|all full-single|{e}", (pop, "level=full", e), (pop, "level=single", e)))
                pairs.append((f"{pop}|all full-pair|{e}", (pop, "level=full", e), (pop, "level=pair", e)))
                pairs.append((f"{pop}|flip-inv|{e}", (pop, "flip", e), (pop, "inv", e)))
                pairs.append((f"{pop}|flip-near_miss|{e}", (pop, "flip", e), (pop, "near_miss", e)))
        # Q2: exact vs gauss_pred / mirror action-change rates on flips / all
        p2 = f"Q2|cf:{fam}"
        for s in ("flip", "near_miss", "inv", "all"):
            for other in ("gauss_pred:1", "mirror", "gauss:1"):
                pairs.append((f"{p2}|{s}|change exact-{other}", (p2, s, "change:exact"), (p2, s, f"change:{other}")))
        # arm accuracy contrasts on the same octet decisions
        for s in ("flip", "flip_vs_pairs", "flip_pair_inherited", "near_miss", "inv", "all"):
            for a, b in (("LRN", "RAWF"), ("SUP", "RAWF"), ("LRN", "SUP")):
                pairs.append((f"ACC|cf:{fam}|{s}|{a}-{b}", (f"ACC|cf:{fam}|{a}", s, "acc"), (f"ACC|cf:{fam}|{b}", s, "acc")))
        # Q3 LRN - SUP reliance
        for sname in ("all", "flip"):
            for pn in PERTURB:
                if pn in ("exact", "lrn_pred"):
                    continue
                for m in ("change", "tv", "dacc"):
                    pairs.append((f"Q3|cf:{fam}|{sname}|{m}:{pn} LRN-SUP", (f"Q3|cf:{fam}|{sname}|LRN", "x", f"{m}:{pn}"),
                                  (f"Q3|cf:{fam}|{sname}|SUP", "x", f"{m}:{pn}")))
    for pn in PERTURB:
        if pn not in ("exact", "lrn_pred"):
            for m in ("change", "tv"):
                pairs.append((f"Q3|pool:A-pistar|all|{m}:{pn} LRN-SUP", ("Q3|pool:A-pistar|all|LRN", "x", f"{m}:{pn}"),
                              ("Q3|pool:A-pistar|all|SUP", "x", f"{m}:{pn}")))
    for a, b in (("LRN", "RAWF"), ("SUP", "RAWF"), ("LRN", "SUP")):
        pairs.append((f"ACC|pool:A-pistar|all|{a}-{b}", (f"ACC|pool:A-pistar|{a}", "all", "acc"),
                      (f"ACC|pool:A-pistar|{b}", "all", "acc")))
    return pairs


def finish_desc(desc, feats):
    out = {}
    for pop, d in desc.items():
        o = {"ix_size": {m: {"n": len(v), "mean": _r(np.mean(v)) if v else None,
                             "frac_nonempty": _r(np.mean([x > 0 for x in v])) if v else None} for m, v in d["ix_size"].items()},
             "ix_coord_counts": {m: dict(sorted(v.items(), key=lambda kv: -kv[1])) for m, v in d["ix_coord"].items()},
             "interaction_reproduction": {}, "attenuation_slope": {}}
        for name, c in d["contrast"].items():
            with np.errstate(invalid="ignore", divide="ignore"):
                r2 = 1 - c["sum_e2"] / c["sum_t2"]
                slope = c["sum_tp"] / c["sum_t2"]
            o["interaction_reproduction"][name] = {
                "units": c["n"], "coords": {f: {"n_big": int(c["nbig"][j]), "mean_abs_contrast": _r(c["abs_t"][j] / max(c["n"], 1)),
                                                "R2_pred_vs_exact": _r(r2[j]) if c["nbig"][j] else None,
                                                "slope_pred_on_exact": _r(slope[j]) if c["nbig"][j] else None}
                                            for j, f in enumerate(feats)}}
        for mem, c in d["slope"].items():
            n = c["n"]
            with np.errstate(invalid="ignore", divide="ignore"):
                cov = c["stp"] / n - (c["st"] / n) * (c["sp"] / n)
                var = c["stt"] / n - (c["st"] / n) ** 2
                sl = np.where(var > 1e-10, cov / var, np.nan)
            o["attenuation_slope"][mem] = {"n": n, **{f: _r(sl[j], 4) for j, f in enumerate(feats)}}
        out[pop] = o
    return out


FACTOR_LETTER = {"none": "0", "unreliable": "U", "side_effect": "S", "correlated": "C", "events": "E", "deadline": "D",
                 "hard_exact_cost": "T"}


def exposure_summary(exposure, fams):
    """Per arm: training combos (letters) and which singles / pairs / full combination of each family were trained."""
    out = {}
    for n, e in exposure.items():
        cc = e.get("train_combo_counts")
        if cc is None:
            out[n] = {"kind": e["kind"], "missing": True}
            continue
        combos = {"".join(FACTOR_LETTER[x] for x in k.split("+")): v for k, v in cc.items()}
        fe = {}
        for fam in fams:
            subs = {"".join(T): combos.get("".join(T), 0) for m in (1, 2, 3) for T in combinations(fam, m)}
            fe[fam] = {"singles": {k: v for k, v in subs.items() if len(k) == 1},
                       "pairs": {k: v for k, v in subs.items() if len(k) == 2}, "full": subs[fam]}
        out[n] = {"kind": e["kind"], "train_split": e.get("train_split"), "train_combos": combos, "family_exposure": fe}
    return out


def finish_dcoord(dcoord, feats, tstd):
    out = {}
    for (pop, s), k in sorted(dcoord.items()):
        n = k["n"]
        out[f"{pop}|{s}"] = {"n": n, **{f: {"mae": _r(k["abs"][j] / n), "nmae": _r(k["abs"][j] / n / tstd[j]) if tstd[j] > 1e-6 else None,
                                            "bias": _r(k["bias"][j] / n), "hit": _r(k["hit"][j] / n), "hitE": _r(k["hitE"][j] / n)}
                                        for j, f in enumerate(feats)}}
    return out


def finish_counts2(counts2):
    out = defaultdict(dict)
    for (pop, s, name), k in sorted(counts2.items()):
        out[f"{pop}|{s}"][name] = {kk: (int(v) if float(v).is_integer() else _r(v)) for kk, v in sorted(k.items())}
    return out


def finish_tabs(tabs):
    out = {}
    for (s, rel, tn), t in sorted(tabs.items()):
        out[f"{s}|{rel}|{tn}"] = dict(sorted(t.items()))
    return out


def summarize_q5(q5res):
    tot = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for key, (rows, agree) in q5res.items():
        pop = key.split("|")[0]
        for typ, r in rows.items():
            for k, v in r.items():
                tot[pop][typ][k] += v
        tot[pop]["_agree"]["checked"] += agree["checked"]
        tot[pop]["_agree"]["agree"] += agree["agree_with_recorded_flags"]
    # labels are identical across seeds (same octet files): report one seed's counts and check equality
    first = {}
    for key, (rows, agree) in sorted(q5res.items()):
        pop = key.split("|")[0]
        if pop not in first:
            first[pop] = {"from": key, "per_type": rows, "flag_agreement": agree}
        else:
            first[pop].setdefault("identical_across_seeds", True)
            if rows != first[pop]["per_type"]:
                first[pop]["identical_across_seeds"] = False
    return first


def markdown(res):
    L = [f"# {VERSION} ({res['tag']}; code {res['code_sha']}; n_boot {res['n_boot']}, seed {res['boot_seed']})", ""]
    B = res["boot"]

    def fmt(k):
        v = B.get(k)
        if not v or v.get("mean") is None:
            return "—"
        ci = v.get("ci")
        ps = ",".join("—" if x is None else f"{x:.3f}" for x in v["per_seed"].values())
        return f"{v['mean']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] ({ps}) n={int(v['support']['den'])}" if ci else f"{v['mean']:.3f} n={int(v['support']['den'])}"

    def fmtc(k):
        v = res["contrasts"].get(k)
        if not v or v.get("mean") is None:
            return "—"
        ci = v.get("ci")
        pp = ",".join("—" if x is None else f"{x:+.3f}" for x in v["per_pair"])
        return f"{v['mean']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] ({pp})" if ci else f"{v['mean']:+.3f}"

    for fam in res["fams"]:
        pop = f"Q1|cf:{fam}"
        L += [f"## Q1 cf:{fam} prediction quality by membership", "",
              "| stratum | acc | chitE G1 | chitE G2 | chitE G3 | chitE G4 | nMAE G1 | nMAE G2 | nMAE G3 | nMAE G4 | sign bv | argmin | oob |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in ("flip", "flip_vs_pairs", "flip_pair_inherited", "near_miss", "inv", "flipunit_pair", "flipunit_single",
                  "flipunit_none", "nonflip_pair",
                  "nonflip_single", "nonflip_none", "level=full", "level=pair", "level=single", "level=none", "all"):
            L.append(f"| {s} | " + " | ".join(fmt(f"{pop}||{s}||{e}") for e in (
                "acc", "chitE_G1", "chitE_G2", "chitE_G3", "chitE_G4", "nmae_G1", "nmae_G2", "nmae_G3", "nmae_G4",
                "sign_bv", "argmin_strat", "oob_range")) + " |")
        L += ["", "| contrast | nMAE DEC | nMAE G1 | nMAE G2 | nMAE G3 | nMAE G4 | chitE DEC |", "|---|---|---|---|---|---|---|"]
        for cn in ("flipunit full-single", "flipunit full-pair", "all full-single", "all full-pair", "flip-inv",
                   "flip-near_miss"):
            L.append(f"| {cn} | " + " | ".join(fmtc(f"{pop}|{cn}|{e}") for e in (
                "nmae_DEC", "nmae_G1", "nmae_G2", "nmae_G3", "nmae_G4", "chitE_DEC")) + " |")
        L += ["", f"### Q1 cf:{fam} by context (flip / inv / single)", "", "| ctx | stratum | acc | nMAE G1 | nMAE G2 | nMAE G3 | chitE DEC |",
              "|---|---|---|---|---|---|---|"]
        ctxs = sorted({k.split("||")[1].split("|")[0] for k in B if k.startswith(pop + "||ctx=")})
        for cx in ctxs:
            for m in ("flip", "inv", "near_miss", "flipunit_single", "flipunit_pair"):
                k = f"{pop}||{cx}|{m}"
                if f"{k}||acc" in B:
                    L.append(f"| {cx[4:]} | {m} | " + " | ".join(fmt(f"{k}||{e}") for e in (
                        "acc", "nmae_G1", "nmae_G2", "nmae_G3", "chitE_DEC")) + " |")
        px = f"Q1x|cf:{fam}"
        L += ["", f"### Q1 cf:{fam} interaction-carrying coords, attraction", "",
              "| stratum | hit IX | hitE IX | nMAE IX | nearest=own | nearest=pair | nearest=single/none |", "|---|---|---|---|---|---|---|"]
        for s in ("flip", "inv", "full_all"):
            L.append(f"| {s} | " + " | ".join(fmt(f"{px}||{s}||{e}") for e in (
                "hit_IX", "hitE_IX", "nmae_IX", "attract_own", "attract_pair", "attract_single")) + " |")
        L += ["", "| attraction (every member) | nearest=own | =full | =pair | =single/none |", "|---|---|---|---|---|"]
        for k in sorted({k.split("||")[1] for k in B if k.startswith(px + "||att|")}):
            L.append(f"| {k[4:]} | " + " | ".join(fmt(f"{px}||{k}||{e}") for e in (
                "attract_own", "attract_full", "attract_pair", "attract_single")) + " |")
        L += ["", "| stale belief stratum | stale | update captured | |err bH| | acc |", "|---|---|---|---|---|"]
        for k in sorted({k.split("||")[1] for k in B if k.startswith(px + "||q2_")}):
            L.append(f"| {k} | " + " | ".join(fmt(f"{px}||{k}||{e}") for e in ("stale", "update_captured", "err_bH", "acc")) + " |")
        p2 = f"Q2|cf:{fam}"
        L += ["", f"## Q2 cf:{fam} immediate replacement (LRN)", "",
              "| stratum | iv | n | changed | rescue (in/out) | harm (in/out) | Δacc | change rate | Δp_opt |", "|---|---|---|---|---|---|---|---|---|"]
        for s in ("flip", "flip_vs_pairs", "flip_pair_inherited", "near_miss", "inv", "all"):
            for iv in IVS:
                k = res["q2_counts"].get(f"{p2}|{s}", {}).get(iv)
                if not k:
                    continue
                L.append(f"| {s} | {iv} | {k['n']} | {k['changed']} | {k.get('rescue', 0)} ({k.get('rescue_in', 0)}/{k.get('rescue_out', 0)}) | "
                         f"{k.get('harm', 0)} ({k.get('harm_in', 0)}/{k.get('harm_out', 0)}) | {fmt(f'{p2}||{s}||dacc:{iv}')} | "
                         f"{fmt(f'{p2}||{s}||change:{iv}')} | {fmt(f'{p2}||{s}||dpopt:{iv}')} |")
        L += ["", "| contrast | CI |", "|---|---|"]
        for s in ("flip", "near_miss", "inv", "all"):
            for o in ("gauss_pred:1", "mirror", "gauss:1"):
                L.append(f"| {s} change exact-{o} | {fmtc(f'{p2}|{s}|change exact-{o}')} |")
        L += ["", f"## arm accuracy on cf:{fam} (identical histories)", "", "| stratum | LRN | SUP | RAWF | LRN-RAWF | SUP-RAWF | LRN-SUP |",
              "|---|---|---|---|---|---|---|"]
        for s in ("flip", "flip_vs_pairs", "flip_pair_inherited", "near_miss", "inv", "all"):
            L.append(f"| {s} | " + " | ".join(fmt(f"ACC|cf:{fam}|{k}||{s}||acc") for k in ("LRN", "SUP", "RAWF")) + " | " +
                     " | ".join(fmtc(f"ACC|cf:{fam}|{s}|{c}") for c in ("LRN-RAWF", "SUP-RAWF", "LRN-SUP")) + " |")
    L += ["", "## Q1 pool (natural distribution)", "", "| pop | stratum | acc | chitE DEC | chit DEC | nMAE G1 | nMAE G2 | nMAE G3 | nMAE G4 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pop in ("Q1|pool:A-pistar", "Q1|pool:B"):
        for s in sorted({k.split("||")[1] for k in B if k.startswith(pop + "||")}):
            L.append(f"| {pop[3:]} | {s} | " + " | ".join(fmt(f"{pop}||{s}||{e}") for e in (
                "acc", "chitE_DEC", "chit_DEC", "nmae_G1", "nmae_G2", "nmae_G3", "nmae_G4")) + " |")
    L += ["", "training-state nMAE (support reference): " + json.dumps(res["training_ref"])]
    L += ["", "## Q2 pool A-pistar", "", "| iv | n | changed | rescue | harm | Δacc | change rate |", "|---|---|---|---|---|---|---|"]
    for iv in IVS:
        k = res["q2_counts"].get("Q2|pool:A-pistar|all", {}).get(iv)
        if k:
            L.append(f"| {iv} | {k['n']} | {k['changed']} | {k.get('rescue', 0)} | {k.get('harm', 0)} | "
                     f"{fmt(f'Q2|pool:A-pistar||all||dacc:{iv}')} | {fmt(f'Q2|pool:A-pistar||all||change:{iv}')} |")
    L += ["", "## Q3 reliance (recomputed policy; action-change rate / TV / Δacc)", "",
          "| population | subset | arm | fidelity | " + " | ".join(p for p in PERTURB) + " |", "|---|---|---|---|" + "---|" * len(PERTURB)]
    for key in sorted({k.split("||")[0] for k in B if k.startswith("Q3|")}):
        _, pop, sname, kind = key.split("|")
        L.append(f"| {pop} | {sname} | {kind} | {fmt(key + '||x||fidelity')} | " +
                 " | ".join(fmt(f"{key}||x||change:{p}") for p in PERTURB) + " |")
    L += ["", "| population | subset | arm | TV " + " | TV ".join(PERTURB) + " |", "|---|---|---|" + "---|" * len(PERTURB)]
    for key in sorted({k.split("||")[0] for k in B if k.startswith("Q3|")}):
        _, pop, sname, kind = key.split("|")
        L.append(f"| {pop} | {sname} | {kind} | " + " | ".join(fmt(f"{key}||x||tv:{p}") for p in PERTURB) + " |")
    L += ["", "| population | subset | arm | base acc | acc exact (LRN) | acc lrn_pred (SUP) | acc zero | acc mean | acc perm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted({k.split("||")[0] for k in B if k.startswith("Q3|")}):
        _, pop, sname, kind = key.split("|")
        L.append(f"| {pop} | {sname} | {kind} | {fmt(key + '||x||base_ok')} | " + " | ".join(
            fmt(f"{key}||x||acc:{p}") for p in ("exact", "lrn_pred", "zero", "mean", "perm")) + " |")
    L += ["", "| contrast LRN-SUP | CI |", "|---|---|"]
    for k in sorted(c for c in res["contrasts"] if c.startswith("Q3|")):
        L.append(f"| {k} | {fmtc(k)} |")
    L += ["", "fusion weight shares: " + json.dumps(res["q3_shares"])]
    L += ["", "## Q4 cross-tabs (counts summed over seeds; full cells in the JSON)", "",
          "| stratum | relevant set | tol | pred | LRN | n | exact fixes | exact breaks | SUP ok | SUP ok & exact same | RAWF ok |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, t in res["q4"].items():
        st, rel, tn = k.split("|")
        if rel not in ("DEC", "IX", "G1") or st.rsplit(":", 1)[-1].startswith(("inv", "pistar")) and rel != "DEC":
            continue
        for rk, r in q4_summary(t).items():
            pr, lr = rk.split("|")
            L.append(f"| {st} | {rel} | {tn} | {pr[5:]} | {lr[4:]} | {r['n']} | {r['exact_fixes']} | {r['exact_breaks']} | "
                     f"{r['sup_ok']} | {r['sup_ok_and_exact_same']} | {r['rawf_ok']} |")
    L += ["", "## Q5 interaction order (octet labels)", "", "```", json.dumps(res["q5"], indent=1), "```"]
    L += ["", "## Q6 exposure", "", "```", json.dumps(res["exposure_summary"], indent=1), "```"]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="dev")
    p.add_argument("--code-sha", default="uncommitted")
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--boot-seed", type=int, default=7)
    p.add_argument("--no-reliance", dest="reliance", action="store_false")
    p.add_argument("--remap", nargs="*", default=[], metavar="OLD=NEW", help="path prefix remap (local dev copies)")
    a = p.parse_args(argv)
    REMAP[:] = [tuple(x.split("=", 1)) for x in a.remap]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fj, fm = out / f"{VERSION}-{a.tag}.json", out / f"{VERSION}-{a.tag}.md"
    for f in (fj, fm):
        if f.exists():
            raise SystemExit(f"refusing to overwrite {f}")
    t0 = time.process_time()
    R = analyse(a.dirs, a.n_boot, a.boot_seed, a.reliance)
    boot = boot_all(R["acc"], a.n_boot, a.boot_seed)
    contrasts = boot_pairs(R["acc"], contrast_list(R, R["fams"]), a.n_boot, a.boot_seed)
    res = {"version": VERSION, "tag": a.tag, "code_sha": a.code_sha, "script_sha256": sha256(__file__),
           "diagscore_sha256": sha256(DS.__file__), "n_boot": a.n_boot, "boot_seed": a.boot_seed, "dirs": a.dirs,
           "fams": R["fams"], "meta": R["meta"], "inputs_sha256": dict(sorted(R["inputs"].items())),
           "training_ref": R["training_ref"], "exposure": R["exposure"],
           "exposure_summary": exposure_summary(R["exposure"], R["fams"]), "q3_shares": R["shares"],
           "q2_counts": finish_counts2(R["counts2"]), "q4": finish_tabs(R["tabs4"]),
           "q4_summary": {k: q4_summary(t) for k, t in finish_tabs(R["tabs4"]).items()}, "q5": summarize_q5(R["q5"]),
           "q1_desc": finish_desc(R["desc"], R["meta"]["features"]),
           "q1_coords": finish_dcoord(R["dcoord"], R["meta"]["features"], R["meta"]["tstd_mean"]), "boot": boot, "contrasts": contrasts,
           "cpu_s": {"pass": round(R["cpu_pass"], 1), "total": None}}
    res["cpu_s"]["total"] = round(time.process_time() - t0, 1)
    txt = json.dumps(res, indent=1, default=DS._np)
    md = markdown(res)
    with open(fj, "x") as f:
        f.write(txt)
    with open(fm, "x") as f:
        f.write(md)
    print(json.dumps({"out": str(fj), "cpu_s": res["cpu_s"], "n_endpoints": len(boot), "n_contrasts": len(contrasts)}))


if __name__ == "__main__":
    main()
