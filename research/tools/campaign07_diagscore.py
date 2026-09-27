"""Extended-07 Phase 1 scorer for campaign07_diag.py logs (diag-v1-*.jsonl.gz).  Read-only on its inputs; writes one
new JSON (+ optional markdown) and refuses to overwrite.

Per (protocol, arm) -- the protocol of an own-history replay is normalized to 'A-own_<source arm>' so seed-paired arms
share it -- and never mixing protocols (denominators are never mixed):
  prediction quality (models with predictions): per coordinate and per group n, MAE, RMSE, bias, normalized MAE
      (MAE / target std; the target std of the support reference when --support-ref is given, else of the pooled
      records), tolerance hit rate (|err| <= tol); calibration of probability coordinates (10 equal-width bins, ECE);
      sign errors of signed coordinates (|target| > tol); order errors of the one-query strategy costs (argmin mismatch
      where the exact argmin is unique by > tol; pairwise discordance for pairs whose exact difference > tol); invalid
      probability outputs (a probability coordinate outside [0, 1]; beliefs not summing to 1 within .02)
  breakdowns by combination, decision context, optimal-action margin bin (Q* best - second best: <.5, .5-2, 2-10,
      >=10) and counterfactual membership (flip = full member of a flip unit with a unique optimum; near_miss;
      invariance = full member of a non-flip unit; sub = other members); joint combination x context x margin table
  decisions: accuracy (greedy action eps-optimal), later-decision accuracy (decision index > 0), mean action gap
  interventions (immediate, LRN): per intervention accuracy, rescue (policy wrong -> right), harm (right -> wrong),
      cross-tab prediction right/wrong (replaced coordinates within tol) x policy right/wrong x replacement fixes / does
      not; share of replaced values inside the consumer's training support ('in' of the support distances)
  bootstrap: two-level (seed with replacement, then shared configuration / whole-octet clusters by multinomial
      weights), n_boot default 20000, fixed --boot-seed; percentile 95% CI of the mean over seeds, and a Monte Carlo
      resolution check (the bounds of the two independent halves of the draws).  --contrast A B: seed-paired A - B.

  python research/tools/campaign07_diagscore.py --files DIR/diag-v1-*.jsonl.gz --out score.json [--md score.md] \
      [--contrast LRN RAWF] [--support-ref LRN-s35=FILE ...] [--tol T | --tol-coord C=T ...] [--n-boot 20000] [--boot-seed 7]
  (or --dirs DIR ... --include 'diag-v1-cf-*' instead of --files; tolerances default to semantic per-coordinate values)

  --by-model-phi (extended-07 P2-SCREEN addendum_2): the arm is '<model>@<phi>' instead of the model kind: <model> =
      the model name without its trailing '-s<seed>' (and 'predLRN<lineage>' -> 'predLRN'), so seeds pool into one
      arm per model; <phi> = the record's factor-channel condition (own when absent: the default files).  Adds, per
      protocol, the seed-paired contrasts <model>@own - <model>@<phi> for every other phi present, and a
      'phi_readout' table (pool / flip / near-miss / invariance accuracy per model x phi, with CIs).
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION = "diagscore-v1"
N_BOOT = 20000
PROB_COORDS = ("belief_H", "belief_M", "belief_F", "belief_X", "p_probe_resolves", "p_probe_false_solved",
               "p_H_given_solved", "p_b1_resolves", "event_hazard_active", "candidate_trust")
BELIEFS = ("belief_H", "belief_M", "belief_F", "belief_X")
SIGNED = ("build_value_rel",)
STRAT = ("cost_exact_b2_rel", "cost_probe_first_rel", "cost_b1_first_rel", "cost_use_rel")
COST_UNIT = ("exp_side_cost_rel", "exp_hard_cost_rel", "cost_exact_b2_rel", "cost_probe_first_rel", "cost_b1_first_rel",
             "cost_use_rel", "best_strategy_cost_rel", "build_value_rel")
EPS_REL = 0.005  # decision tolerance eps = 0.5 price units in the cost coordinates' units (price / R; factor contract)


def semantic_tol(feats):
    """Registered-before-use per-coordinate tolerances from the coordinates' semantics (factor contract): cost-unit
    coordinates eps_rel = .005 (the decision tolerance); probabilities .05; remaining_queries_rel half a query (1/16);
    binary bookkeeping flags .5 (right side); steps_left_rel half a step (.25)."""
    out = {}
    for f in feats:
        if f in COST_UNIT:
            out[f] = EPS_REL
        elif f in PROB_COORDS:
            out[f] = 0.05
        elif f == "remaining_queries_rel":
            out[f] = 1.0 / 16
        elif f == "steps_left_rel":
            out[f] = 0.25
        else:
            out[f] = 0.5
    return out


MARGIN_BINS = ((0.0, 0.5, "<.5"), (0.5, 2.0, ".5-2"), (2.0, 10.0, "2-10"), (10.0, math.inf, ">=10"))


def margin_bin(m):
    return next(n for lo, hi, n in MARGIN_BINS if lo <= m < hi)


def membership(r):
    c = r.get("cf")
    if c is None:
        return "pool"
    if c["role"] == "near_miss":
        return "near_miss"
    if c["role"] == "full":
        return "flip" if (c["flip"] and c["unique_full"]) else "invariance"
    return "sub"


def model_base(name):
    """Model name without its trailing seed ('S1R1-bank-s40' -> 'S1R1-bank'; 'CONS-mix-predLRN35-s40' ->
    'CONS-mix-predLRN'): the seed-pooled arm of --by-model-phi."""
    return re.sub(r"predLRN\d+", "predLRN", re.sub(r"-s\d+$", "", name))


def phi_arm(name, phi):
    return f"{model_base(name)}@{phi or 'own'}"


PHI_ENDPOINTS = ("acc", "acc_flip", "acc_near_miss", "acc_invariance")


def cluster_of(r):
    """Configuration cluster (pool config) or whole octet (counterfactual)."""
    c = r.get("cf")
    return f"{c['family']}:{c['octet']}" if c else r["cfg_id"]


def read(path):
    with gzip.open(path, "rt") as f:
        head = json.loads(f.readline())["_meta"]
        for line in f:
            if line.strip():
                yield head, json.loads(line)


# ------------------------------------------------------------------------------------------------ accumulators

class Stat:
    """Running per-coordinate error statistics."""

    def __init__(self, feats):
        self.feats = feats
        n = len(feats)
        self.n = 0
        self.abs = np.zeros(n)
        self.sq = np.zeros(n)
        self.bias = np.zeros(n)
        self.tsum = np.zeros(n)
        self.tsq = np.zeros(n)
        self.hit = np.zeros(n)

    def add(self, pred, target, tol):
        p, t = np.asarray(pred), np.asarray(target)
        e = p - t
        self.n += 1
        self.abs += np.abs(e)
        self.sq += e * e
        self.bias += e
        self.tsum += t
        self.tsq += t * t
        self.hit += np.abs(e) <= tol

    def result(self, groups, tstd_ref=None):
        if self.n == 0:
            return None
        n = self.n
        mae, rmse, bias = self.abs / n, np.sqrt(self.sq / n), self.bias / n
        tstd = np.sqrt(np.maximum(self.tsq / n - (self.tsum / n) ** 2, 0.0))
        scale = np.asarray(tstd_ref) if tstd_ref is not None else tstd
        nmae = np.where(scale > 1e-9, mae / np.maximum(scale, 1e-9), np.nan)
        coords = {f: {"mae": _r(mae[j]), "rmse": _r(rmse[j]), "bias": _r(bias[j]), "nmae": _r(nmae[j]),
                      "hit": _r(self.hit[j] / n), "target_std": _r(tstd[j])} for j, f in enumerate(self.feats)}
        grp = {}
        for g, cs in groups.items():
            ix = [self.feats.index(c) for c in cs]
            grp[g] = {"mae": _r(mae[ix].mean()), "nmae": _r(np.nanmean(nmae[ix])) if not np.all(np.isnan(nmae[ix])) else None,
                      "hit": _r((self.hit[ix] / n).mean())}
        return {"n": n, "coords": coords, "groups": grp}


def _r(x, d=6):
    x = float(x)
    return None if math.isnan(x) else round(x, d)


class Calib:
    def __init__(self):
        self.b = defaultdict(lambda: [0, 0.0, 0.0])

    def add(self, p, t):
        k = min(max(int(p * 10), 0), 9)
        c = self.b[k]
        c[0] += 1
        c[1] += p
        c[2] += t

    def result(self):
        n = sum(c[0] for c in self.b.values())
        if not n:
            return None
        bins = [{"bin": k, "n": c[0], "pred": _r(c[1] / c[0]), "target": _r(c[2] / c[0])} for k, c in sorted(self.b.items())]
        ece = sum(c[0] * abs(c[1] / c[0] - c[2] / c[0]) for c in self.b.values()) / n
        return {"n": n, "ece": _r(ece), "bins": bins}


def order_errors(pred, target, fi, tol):
    ix = [fi[c] for c in STRAT]
    pt, tt = [pred[j] for j in ix], [target[j] for j in ix]
    srt = sorted(tt)
    argmin_ok = None
    if srt[1] - srt[0] > tol:
        argmin_ok = int(np.argmin(pt)) == int(np.argmin(tt))
    n = d = 0
    for a in range(4):
        for b in range(a + 1, 4):
            if abs(tt[a] - tt[b]) > tol:
                n += 1
                d += (tt[a] - tt[b]) * (pt[a] - pt[b]) <= 0
    return argmin_ok, n, d


# ------------------------------------------------------------------------------------------------ scoring

class Cell:
    """Everything accumulated for one (protocol, arm[, breakdown key])."""

    def __init__(self, feats):
        self.feats = feats
        self.stat = Stat(feats)
        self.n = 0
        self.ok = 0
        self.gap = 0.0
        self.later_n = self.later_ok = 0

    def add(self, r, tol):
        self.n += 1
        self.ok += r["ok"]
        self.gap += r["gap"]
        if r["t"] > 0:
            self.later_n += 1
            self.later_ok += r["ok"]
        if r.get("pred") is not None:
            self.stat.add(r["pred"], r["target"], tol)

    def result(self, groups, tstd_ref=None):
        out = {"n": self.n, "acc": _r(self.ok / self.n) if self.n else None,
               "later_acc": _r(self.later_ok / self.later_n) if self.later_n else None, "later_n": self.later_n,
               "gap": _r(self.gap / self.n) if self.n else None}
        pq = self.stat.result(groups, tstd_ref)
        if pq is not None:
            out["pred_groups"] = pq["groups"]
        return out


def score(files, tol=None, supports=None, n_boot=N_BOOT, boot_seed=7, contrasts=(), tol_coord=None, by_model_phi=False):
    """tol None: semantic per-coordinate tolerances (semantic_tol); a number: uniform; tol_coord overrides per coord."""
    supports = supports or {}
    heads = {}
    feats = groups = None
    full = {}  # (proto, arm) -> dict of accumulators
    # (proto, arm) -> endpoint -> {(seed, cluster): [num, den]}
    boot = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(2))))
    seeds_of = defaultdict(set)
    for path in files:
        for head, r in read(path):
            if r.get("kind") != "decision":
                continue
            feats = feats or head["features"]
            groups = groups or head["groups"]
            heads[path] = head
            info = head["models"][r["model"]]
            arm, seed = info["kind"], info["seed"]
            if by_model_phi:
                arm = phi_arm(r["model"], r.get("phi"))
            proto = r["protocol"]
            if proto.startswith("A-own_"):
                src = proto[len("A-own_"):]
                proto = "A-own_" + (model_base(src) if by_model_phi else head["models"][src]["kind"])
            fi = {f: j for j, f in enumerate(feats)}
            base_tol = semantic_tol(feats) if tol is None else {f: tol for f in feats}
            tolv = np.array([(tol_coord or {}).get(f, base_tol[f]) for f in feats])
            key = (proto, arm)
            A = full.get(key)
            if A is None:
                A = full[key] = {"all": Cell(feats), "by": defaultdict(lambda: Cell(feats)), "joint": defaultdict(lambda: Cell(feats)),
                                 "calib": defaultdict(Calib), "sign": [0, 0], "order": [0, 0, 0, 0], "invalid": [0, 0, 0],
                                 "iv": defaultdict(lambda: {"n": 0, "ok": 0, "ok_none": 0, "rescue": 0, "harm": 0, "in": 0,
                                                            "in_n": 0, "xtab": defaultdict(int)}),
                                 "support_pred": [0, 0], "tstd": _tstd(supports, r["model"])}
            seeds_of[key].add(seed)
            A["all"].add(r, tolv)
            mb = margin_bin(r["margin"])
            mem = membership(r)
            for dim, val in (("combo", r.get("combo")), ("ctx", r["ctx"]), ("margin", mb), ("membership", mem),
                             ("k", r.get("k"))):
                A["by"][f"{dim}={val}"].add(r, tolv)
            A["joint"][f"{r.get('combo')}|{r['ctx']}|{mb}"].add(r, tolv)
            cl = cluster_of(r)
            B = boot[key]
            _acc(B, "acc", seed, cl, r["ok"], 1)
            _acc(B, "gap", seed, cl, r["gap"], 1)
            if r["t"] > 0:
                _acc(B, "later_acc", seed, cl, r["ok"], 1)
            if mem in ("flip", "near_miss", "invariance"):
                _acc(B, f"acc_{mem}", seed, cl, r["ok"], 1)
            pred, target = r.get("pred"), r["target"]
            if pred is not None:
                err = np.abs(np.asarray(pred) - np.asarray(target))
                for g, cs in groups.items():
                    ix = [fi[c] for c in cs]
                    if A["tstd"] is not None:  # normalized by the support reference's target std
                        _acc(B, f"nmae_{g}", seed, cl, float((err[ix] / np.maximum(np.asarray(A["tstd"])[ix], 1e-3)).mean()), 1)
                    else:
                        _acc(B, f"mae_{g}", seed, cl, float(err[ix].mean()), 1)
                    _acc(B, f"hit_{g}", seed, cl, float((err[ix] <= tolv[ix]).all()), 1)
                for c in PROB_COORDS:
                    A["calib"][c].add(pred[fi[c]], target[fi[c]])
                for c in SIGNED:
                    j = fi[c]
                    if abs(target[j]) > tolv[j]:
                        A["sign"][0] += 1
                        A["sign"][1] += (pred[j] > 0) != (target[j] > 0)
                am, npair, disc = order_errors(pred, target, fi, float(tolv[fi[STRAT[0]]]))
                if am is not None:
                    A["order"][0] += 1
                    A["order"][1] += not am
                A["order"][2] += npair
                A["order"][3] += disc
                A["invalid"][0] += 1
                A["invalid"][1] += any(pred[fi[c]] < -1e-6 or pred[fi[c]] > 1 + 1e-6 for c in PROB_COORDS)
                A["invalid"][2] += abs(sum(pred[fi[c]] for c in BELIEFS) - 1.0) > 0.02
                sp = r.get("support_pred")
                if sp is not None:
                    A["support_pred"][0] += 1
                    A["support_pred"][1] += sp["in"]
            for name, iv in (r.get("iv") or {}).items():
                S = A["iv"][name]
                S["n"] += 1
                S["ok"] += iv["ok"]
                S["ok_none"] += r["ok"]
                S["rescue"] += (not r["ok"]) and iv["ok"]
                S["harm"] += r["ok"] and not iv["ok"]
                if "support" in iv:
                    S["in_n"] += 1
                    S["in"] += iv["support"]["in"]
                if pred is not None:
                    cs = _iv_coords(name, groups)
                    ix = [fi[c] for c in cs] if cs else list(range(len(feats)))
                    pok = bool((np.abs(np.asarray(pred)[ix] - np.asarray(target)[ix]) <= tolv[ix]).all())
                    S["xtab"][f"pred_{'right' if pok else 'wrong'}|policy_{'right' if r['ok'] else 'wrong'}|"
                              f"{'fixes' if (not r['ok'] and iv['ok']) else ('breaks' if (r['ok'] and not iv['ok']) else 'same')}"] += 1
                _acc(B, f"iv_acc_delta:{name}", seed, cl, iv["ok"] - r["ok"], 1)
                if not r["ok"]:
                    _acc(B, f"iv_rescue:{name}", seed, cl, iv["ok"], 1)
                else:
                    _acc(B, f"iv_harm:{name}", seed, cl, not iv["ok"], 1)
    out = {"version": VERSION, "tol": tol if tol is not None else "semantic", "tol_used": (
               {f: float(v) for f, v in zip(feats, tolv)} if feats else None), "tol_coord": tol_coord or {},
           "n_boot": n_boot, "boot_seed": boot_seed,
           "files": [str(f) for f in files], "groups": groups,
           "group_source": next(iter(heads.values()))["group_source"] if heads else None, "cells": {}, "contrasts": {}}
    for (proto, arm), A in sorted(full.items()):
        tref = A["tstd"]
        ent = {"seeds": sorted(seeds_of[(proto, arm)]), "overall": A["all"].result(groups, tref)}
        pq = A["all"].stat.result(groups, tref)
        if pq is not None:
            ent["prediction"] = pq
            ent["calibration"] = {c: A["calib"][c].result() for c in PROB_COORDS}
            ent["sign_errors"] = {"n": A["sign"][0], "rate": _r(A["sign"][1] / A["sign"][0]) if A["sign"][0] else None}
            ent["order_errors"] = {"argmin_n": A["order"][0], "argmin_err": _r(A["order"][1] / A["order"][0]) if A["order"][0] else None,
                                   "pairs": A["order"][2], "discordance": _r(A["order"][3] / A["order"][2]) if A["order"][2] else None}
            ent["invalid_prob"] = {"n": A["invalid"][0], "out_of_range": _r(A["invalid"][1] / A["invalid"][0]),
                                   "belief_sum": _r(A["invalid"][2] / A["invalid"][0])}
            if A["support_pred"][0]:
                ent["pred_in_support"] = _r(A["support_pred"][1] / A["support_pred"][0])
        ent["by"] = {k: c.result(groups, tref) for k, c in sorted(A["by"].items())}
        ent["joint_combo_ctx_margin"] = {k: c.result(groups, tref) for k, c in sorted(A["joint"].items())}
        if A["iv"]:
            ent["interventions"] = {n: {"n": S["n"], "acc": _r(S["ok"] / S["n"]), "acc_none": _r(S["ok_none"] / S["n"]),
                                        "rescue": S["rescue"], "harm": S["harm"],
                                        "in_support": _r(S["in"] / S["in_n"]) if S["in_n"] else None,
                                        "xtab": dict(sorted(S["xtab"].items()))} for n, S in sorted(A["iv"].items())}
        useeds = sorted(seeds_of[(proto, arm)])
        ucls = sorted({c for cells in boot[(proto, arm)].values() for _, c in cells})
        ent["bootstrap"] = {e: boot_endpoint(boot[(proto, arm)][e], n_boot, boot_seed, useeds, ucls)
                            for e in sorted(boot[(proto, arm)])}
        out["cells"][f"{proto}|{arm}"] = ent
    contrasts = list(contrasts)
    if by_model_phi:  # own - phi, per model (seed-paired; the same seeds and clusters)
        arms = {a for _, a in full}
        for a_ in sorted(arms):
            m, _, ph = a_.rpartition("@")
            if ph != "own" and f"{m}@own" in arms and (f"{m}@own", a_) not in contrasts:
                contrasts.append((f"{m}@own", a_))
    for a_arm, b_arm in contrasts:
        for proto in sorted({p for p, _ in full}):
            if (proto, a_arm) in full and (proto, b_arm) in full:
                Ba, Bb = boot[(proto, a_arm)], boot[(proto, b_arm)]
                common = sorted(seeds_of[(proto, a_arm)] & seeds_of[(proto, b_arm)])
                ucls = sorted({c for B_ in (Ba, Bb) for cells in B_.values() for s_, c in cells if s_ in common})
                out["contrasts"][f"{proto}|{a_arm}-{b_arm}"] = {
                    e: boot_contrast(Ba[e], Bb[e], n_boot, boot_seed, common, ucls) for e in sorted(set(Ba) & set(Bb))
                    if not e.startswith(("nmae_", "mae_", "hit_", "iv_"))}
    if by_model_phi:
        out["phi_readout"] = phi_readout(out)
    return out


def phi_readout(res):
    """{protocol: {model: {phi: {endpoint: {mean, ci, per_seed}}}}} for pool / flip / near-miss / invariance
    accuracy, plus the own - phi contrasts (mean, ci, per pair)."""
    out = {}
    for key, ent in res["cells"].items():
        proto, arm = key.split("|", 1)
        m, _, ph = arm.rpartition("@")
        bs = ent.get("bootstrap", {})
        row = {e: {k: bs[e].get(k) for k in ("mean", "ci", "per_seed")} for e in PHI_ENDPOINTS if e in bs}
        row["n"] = ent["overall"]["n"]
        out.setdefault(proto, {}).setdefault(m, {})[ph] = row
        c = res["contrasts"].get(f"{proto}|{m}@own-{arm}")
        if c:
            row["own_minus_this"] = {e: {k: c[e].get(k) for k in ("mean", "ci", "per_pair")}
                                     for e in PHI_ENDPOINTS if c.get(e)}
    return out


def _tstd(supports, model):
    s = supports.get(model)
    return s["target_std"] if s else None


def _iv_coords(name, groups):
    base = name.split(":")[0]
    if base in ("iso", "dep", "up", "keep"):
        g = name.split(":", 1)[1]
        cs = groups.get(g, [])
        if base == "keep":
            return None  # everything else exact: judge the full vector
        return cs  # dep: the closure includes cs; judged on the named group
    return None


def _acc(B, e, seed, cl, num, den):
    v = B[e][(seed, cl)]
    v[0] += num
    v[1] += den


# ------------------------------------------------------------------------------------------------ bootstrap

def _matrix(cells, seeds=None, cls=None):
    """(seeds, clusters, num, den) over the given universes (a cell's full seed / cluster sets: every endpoint of a
    cell then shares the same bootstrap draws; clusters absent from an endpoint carry zero weight in it)."""
    seeds = seeds if seeds is not None else sorted({s for s, _ in cells})
    cls = cls if cls is not None else sorted({c for _, c in cells})
    si, ci = {s: j for j, s in enumerate(seeds)}, {c: j for j, c in enumerate(cls)}
    num, den = np.zeros((len(seeds), len(cls))), np.zeros((len(seeds), len(cls)))
    for (s, c), (n, d) in cells.items():
        if s in si:
            num[si[s], ci[c]], den[si[s], ci[c]] = n, d
    return seeds, cls, num, den


_DRAWS: dict = {}


def draws(S, C, n_boot, seed):
    """Fixed-seed two-level draws: seed indices with replacement (n_boot, S) and multinomial cluster weights
    (n_boot, C) shared by the drawn seeds.  Cached: endpoints over the same universe share the draws."""
    key = (S, C, n_boot, seed)
    if key not in _DRAWS:
        rng = np.random.default_rng(seed)
        W = rng.multinomial(C, np.full(C, 1.0 / C), size=n_boot).astype(np.float32)
        pix = rng.integers(0, S, size=(n_boot, S))
        _DRAWS.clear()
        _DRAWS[key] = (W, pix)
    return _DRAWS[key]


def two_level(ratio_fn, S, C, n_boot, seed, chunk=4000):
    """ratio_fn(W) -> (nb, S) per-seed statistic under cluster weights W (nb, C); statistic = mean over the drawn
    seeds (nan-safe).  Returns the finite bootstrap values."""
    W, pix = draws(S, C, n_boot, seed)
    out = []
    for b0 in range(0, n_boot, chunk):
        R = ratio_fn(W[b0:b0 + chunk].astype(float))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out.append(np.nanmean(np.take_along_axis(R, pix[b0:b0 + chunk], 1), 1))
    v = np.concatenate(out)
    return v[~np.isnan(v)]


def _ci(v, n_boot):
    if len(v) < n_boot // 2:
        return None, None
    half = len(v) // 2
    q = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]  # noqa: E731
    return q(v), {"half1": q(v[:half]), "half2": q(v[half:])}


def _ratio(num, den):
    def f(W):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (W @ num.T) / np.where((W @ den.T) > 0, W @ den.T, np.nan)
    return f


def boot_endpoint(cells, n_boot, seed, seeds=None, cls=None):
    seeds, cls, num, den = _matrix(cells, seeds, cls)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = num.sum(1) / np.where(den.sum(1) > 0, den.sum(1), np.nan)
    ok = ~np.isnan(per)
    point = float(per[ok].mean()) if ok.any() else None
    res = {"per_seed": {str(s): _r(p) for s, p in zip(seeds, per)}, "mean": _r(point) if point is not None else None,
           "support": {"clusters": len(cls), "den": float(den.sum())}}
    if point is None or n_boot <= 0:
        return res
    v = two_level(_ratio(num, den), len(seeds), len(cls), n_boot, seed)
    res["ci"], res["mc_check"] = _ci(v, n_boot)
    return res


def boot_contrast(ca, cb, n_boot, seed, common=None, cls=None):
    """Seed-paired A - B (seeds present in both; clusters = union, shared weights)."""
    common = common if common is not None else sorted({s for s, _ in ca} & {s for s, _ in cb})
    if not common:
        return None
    cls = cls if cls is not None else sorted({c for s, c in list(ca) + list(cb) if s in common})
    ci_ = {c: j for j, c in enumerate(cls)}
    mats = []
    for cells in (ca, cb):
        num, den = np.zeros((len(common), len(cls))), np.zeros((len(common), len(cls)))
        for (s, c), (n, d) in cells.items():
            if s in common:
                num[common.index(s), ci_[c]], den[common.index(s), ci_[c]] = n, d
        mats.append((num, den))
    (na, da), (nb_, db) = mats
    fa, fb = _ratio(na, da), _ratio(nb_, db)
    per = [(na[j].sum() / da[j].sum() - nb_[j].sum() / db[j].sum()) if da[j].sum() and db[j].sum() else None
           for j in range(len(common))]
    vals = [p for p in per if p is not None]
    res = {"seeds": common, "per_pair": [_r(p) if p is not None else None for p in per],
           "mean": _r(sum(vals) / len(vals)) if vals else None,
           "signs": {"negative": sum(p < 0 for p in vals), "positive": sum(p > 0 for p in vals)},
           "support": {"clusters": len(cls)}}
    if vals and n_boot > 0:
        v = two_level(lambda W: fa(W) - fb(W), len(common), len(cls), n_boot, seed)
        res["ci"], res["mc_check"] = _ci(v, n_boot)
    return res


# ------------------------------------------------------------------------------------------------ report

def markdown(res):
    lines = [f"# diagscore ({res['version']}; tol {res['tol']}; n_boot {res['n_boot']}, seed {res['boot_seed']}; "
             f"groups: {res['group_source']})", ""]
    lines += ["| protocol | arm | n | acc | later acc | gap | " + " | ".join(f"nMAE {g}" for g in res["groups"] or {}) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(res["groups"] or {})]
    for key, ent in res["cells"].items():
        p, a = key.split("|")
        o = ent["overall"]
        pg = o.get("pred_groups") or {}
        lines.append(f"| {p} | {a} | {o['n']} | {o['acc']} | {o['later_acc']} | {o['gap']} | " +
                     " | ".join(str((pg.get(g) or {}).get("nmae", "—")) for g in res["groups"] or {}) + " |")
    for key, ent in res["cells"].items():
        if "interventions" in ent:
            lines += ["", f"## interventions {key}", "", "| iv | n | acc | acc none | rescue | harm | in support |",
                      "|---|---|---|---|---|---|---|"]
            for n, s in ent["interventions"].items():
                lines.append(f"| {n} | {s['n']} | {s['acc']} | {s['acc_none']} | {s['rescue']} | {s['harm']} | {s['in_support']} |")
    if res.get("phi_readout"):
        lines += ["", "## factor-channel conditions (model x phi; mean [95% CI]; own-phi = seed-paired own minus this)",
                  "", "| protocol | model | phi | n | " + " | ".join(PHI_ENDPOINTS) + " | own-phi |",
                  "|---|---|---|---|" + "---|" * len(PHI_ENDPOINTS) + "---|"]
        for proto, ms in res["phi_readout"].items():
            for m, phis in ms.items():
                for ph in sorted(phis, key=lambda x: ("own", "exact", "zero", "mean").index(x)
                                 if x in ("own", "exact", "zero", "mean") else 9):
                    row = phis[ph]
                    cells = []
                    for e in PHI_ENDPOINTS:
                        v = row.get(e)
                        cells.append("—" if not v or v["mean"] is None else
                                     f"{v['mean']:.3f}" + (f" [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if v.get("ci") else ""))
                    d = row.get("own_minus_this") or {}
                    dd = "; ".join(f"{e} {v['mean']:+.3f}" for e, v in d.items() if v.get("mean") is not None)
                    lines.append(f"| {proto} | {m} | {ph} | {row['n']} | " + " | ".join(cells) + f" | {dd or '—'} |")
    if res["contrasts"]:
        lines += ["", "## contrasts (seed-paired, two-level CI)", "", "| contrast | endpoint | mean | CI | per pair |",
                  "|---|---|---|---|---|"]
        for key, ent in res["contrasts"].items():
            for e, v in ent.items():
                if v:
                    lines.append(f"| {key} | {e} | {v['mean']} | {v.get('ci')} | {v['per_pair']} |")
    return "\n".join(lines) + "\n"


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--dirs", nargs="*", default=[], help="diag output dirs (files matched by --include; no shell globs)")
    p.add_argument("--include", nargs="*", default=["diag-v1-*.jsonl.gz"], metavar="PATTERN")
    p.add_argument("--out", required=True)
    p.add_argument("--md", default=None)
    p.add_argument("--tol", type=float, default=None, help="uniform tolerance (default: semantic per coordinate)")
    p.add_argument("--tol-coord", nargs="*", default=None, metavar="COORD=TOL")
    p.add_argument("--support-ref", nargs="*", default=None, metavar="MODEL=FILE")
    p.add_argument("--contrast", nargs=2, action="append", default=[], metavar=("A", "B"))
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--boot-seed", type=int, default=7)
    p.add_argument("--by-model-phi", action="store_true",
                   help="arm = <model without seed>@<phi> (factor-channel conditions; adds own - phi contrasts)")
    a = p.parse_args(argv)
    for f in [a.out] + ([a.md] if a.md else []):
        if Path(f).exists():
            raise SystemExit(f"refusing to overwrite {f}")
    sup = {}
    for x in a.support_ref or []:
        k, v = x.split("=", 1)
        sup[k] = json.loads(Path(v).read_text())
    tc = {k: float(v) for k, v in (x.split("=", 1) for x in a.tol_coord or [])}
    files = list(a.files)
    for d in a.dirs:
        files += sorted(str(f) for f in Path(d).iterdir() if any(fnmatch.fnmatch(f.name, pt) for pt in a.include)
                        and not f.name.startswith("diag-v1-hist-") and f.name.endswith(".jsonl.gz"))
    if not files:
        raise SystemExit("no diag files")
    res = score(files, a.tol, sup, a.n_boot, a.boot_seed, [tuple(c) for c in a.contrast], tc, a.by_model_phi)
    txt = json.dumps(res, indent=1, default=_np)  # serialize before creating the file (no partial outputs)
    with open(a.out, "x") as f:
        f.write(txt)
    if a.md:
        with open(a.md, "x") as f:
            f.write(markdown(res))


if __name__ == "__main__":
    main()
