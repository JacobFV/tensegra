"""Track C label packing and appraisal training (extended-04; review F8, F11, F16).

Targets are **regression / proper-scoring targets only** (review F8):

- step level (D0 base episodes, every decision; the main line after t is D0):
  success (BCE), remaining cost (squared error), ΔU under default (squared error);
- point level (branch labels): ΔU(default) (non-D0 episodes), success / remaining
  cost under the default branch, and the paired advantages ΔU(u) − ΔU(default) for
  u ∈ {sample (per draw), mask_top, stop} (squared error; common world across u).

No classification target is derived from per-world branch comparisons. Inputs are
the telemetry sequence only (``pack`` stores targets in separate tensors and the
model's forward never sees them; tested). ``shuffle_targets`` is the shuffled-target
control: step-level target rows are permuted across all valid (episode, step)
positions and point target rows across points, breaking any input-target relation.

The margin m (learned controller) and τ (automatic threshold rule) are registered on
the **development split of the label episodes** (a hash split by world seed; never
evaluation worlds) by the realized one-step gain of the rule over default on the
dev label points: gain = ΔU(u chosen) − ΔU(default), with the K-draw mean for sample.
Ties go to the more conservative value (larger m, smaller τ).
"""
from __future__ import annotations

import math
import random

import torch
from torch import nn

from .campaign04_meta import ADV_US, BASE_KINDS, LABEL_VERSION, MetaSpec, digest_seed, make_model
from .campaign04_telemetry import DIM, INTERVENTIONS, TELEMETRY_VERSION
from .campaign04_progress import CLASSES

STRATA = ("flagged", "low_margin", "other")
MARGIN_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, math.inf)
TAU_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NAN = float("nan")


def _f(x):
    return NAN if x is None else float(x)


def pack(episodes, points, meta=None) -> dict:
    """Compact tensor form of ``generate_labels`` output (one chunk)."""
    lengths = [len(e["feats"]) for e in episodes]
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    feats = [v for e in episodes for v in e["feats"]]
    k = max((len(p["dU_sample"]) for p in points), default=0)
    return {
        "version": LABEL_VERSION, "telemetry_version": TELEMETRY_VERSION, "meta": dict(meta or {}),
        "feats": torch.tensor(feats, dtype=torch.float32).reshape(-1, DIM),
        "ep_offsets": torch.tensor(offsets, dtype=torch.long),
        "ep_seed": torch.tensor([e["seed"] for e in episodes], dtype=torch.long),
        "ep_kind": torch.tensor([BASE_KINDS.index(e["base_kind"]) for e in episodes], dtype=torch.int8),
        "ep_success": torch.tensor([e["final"]["success"] for e in episodes], dtype=torch.bool),
        "ep_utility": torch.tensor([e["final"]["utility"] for e in episodes], dtype=torch.float64),
        "ep_cost": torch.tensor([e["final"]["cost"] for e in episodes], dtype=torch.float64),
        "ep_truncated": torch.tensor([e["truncated"] for e in episodes], dtype=torch.bool),
        "ep_capped": torch.tensor([e["capped_points"] for e in episodes], dtype=torch.int16),
        "U": torch.tensor([u for e in episodes for u in e["U"]], dtype=torch.float64),
        "u": torch.tensor([INTERVENTIONS.index(u) for e in episodes for u in e["u"]], dtype=torch.int8),
        "cls": torch.tensor([CLASSES.index(c) for e in episodes for c in e["cls"]], dtype=torch.int8),
        "pt_episode": torch.tensor([p["episode"] for p in points], dtype=torch.long),
        "pt_step": torch.tensor([p["step"] for p in points], dtype=torch.long),
        "pt_stratum": torch.tensor([STRATA.index(p["stratum"]) for p in points], dtype=torch.int8),
        "pt_p": torch.tensor([p["p_include"] for p in points], dtype=torch.float32),
        "pt_u_taken": torch.tensor([INTERVENTIONS.index(p["u_taken"]) for p in points], dtype=torch.int8),
        "pt_dU": torch.tensor([[_f(p["dU"][u]) for u in ("default", "mask_top", "stop")] for p in points],
                              dtype=torch.float64).reshape(-1, 3),
        "pt_dU_sample": torch.tensor([[_f(x) for x in p["dU_sample"]] for p in points],
                                     dtype=torch.float64).reshape(-1, k),
        "pt_sample_is_default": torch.tensor([p["sample_is_default"] for p in points], dtype=torch.bool).reshape(-1, k),
        "pt_success_d": torch.tensor([_f(p["success_default"]) for p in points], dtype=torch.float32),
        "pt_cost_d": torch.tensor([_f(p["cost_default"]) for p in points], dtype=torch.float64),
        "pt_check": torch.tensor([p["default_check"] for p in points], dtype=torch.int8),
        "pt_branches": torch.tensor([p["branches"] for p in points], dtype=torch.int16),
    }


def merge(packs) -> dict:
    """Concatenate chunks (episode indices of points are re-based)."""
    if not packs:
        raise ValueError("no label chunks")
    out = {"version": packs[0]["version"], "telemetry_version": packs[0]["telemetry_version"],
           "meta": {"chunks": [p["meta"] for p in packs]}}
    for p in packs:
        if p["version"] != LABEL_VERSION or p["telemetry_version"] != TELEMETRY_VERSION:
            raise ValueError("label/telemetry version mismatch")
    n_eps, n_steps, offsets, pt_eps = 0, 0, [torch.zeros(1, dtype=torch.long)], []
    for p in packs:
        offsets.append(p["ep_offsets"][1:] + n_steps)
        pt_eps.append(p["pt_episode"] + n_eps)
        n_eps += len(p["ep_seed"])
        n_steps += int(p["ep_offsets"][-1])
    out["ep_offsets"] = torch.cat(offsets)
    out["pt_episode"] = torch.cat(pt_eps)
    for key in packs[0]:
        if key in out or key in ("version", "telemetry_version", "meta"):
            continue
        out[key] = torch.cat([p[key] for p in packs])
    return out


def split(data, dev_fraction=0.2, salt="c-dev-split"):
    """Episode index lists (train, dev) by a hash of the world seed."""
    train, dev = [], []
    for i, seed in enumerate(data["ep_seed"].tolist()):
        (dev if digest_seed(salt, seed) / 2 ** 64 < dev_fraction else train).append(i)
    return train, dev


class Tensors:
    """Padded training tensors. Inputs (``x``) and targets are separate objects."""

    def __init__(self, data):
        off = data["ep_offsets"].tolist()
        self.lengths = [b - a for a, b in zip(off, off[1:])]
        E, T = len(self.lengths), max(self.lengths)
        self.x = torch.zeros(E, T, DIM)
        # step-level targets (D0 episodes only)
        self.s_valid = torch.zeros(E, T, dtype=torch.bool)
        self.s_success = torch.zeros(E, T)
        self.s_cost = torch.zeros(E, T)
        self.s_q0 = torch.zeros(E, T)
        kinds = data["ep_kind"].tolist()
        U = data["U"]
        for e, (a, b) in enumerate(zip(off, off[1:])):
            n = b - a
            self.x[e, :n] = data["feats"][a:b]
            if BASE_KINDS[kinds[e]] == "d0":
                self.s_valid[e, :n] = True
                self.s_success[e, :n] = float(data["ep_success"][e])
                self.s_q0[e, :n] = (data["ep_utility"][e] - U[a:b]).float()
                self.s_cost[e, :n] = (data["ep_cost"][e] + U[a:b]).float()   # cost_t = -U_t
        # point-level targets
        self.p_e = data["pt_episode"]
        self.p_t = data["pt_step"]
        d = data["pt_dU"]
        self.p_default_valid = torch.tensor([BASE_KINDS[kinds[e]] != "d0" for e in self.p_e.tolist()],
                                            dtype=torch.bool)
        self.p_q0 = d[:, 0].float()
        self.p_success = data["pt_success_d"]
        self.p_cost = data["pt_cost_d"].float()
        adv = torch.full((len(self.p_e), len(ADV_US), data["pt_dU_sample"].shape[1] or 1), NAN)
        ks = data["pt_dU_sample"].shape[1]
        if ks:
            adv[:, 0, :ks] = (data["pt_dU_sample"] - d[:, :1]).float()
        adv[:, 1, 0] = (d[:, 1] - d[:, 0]).float()
        adv[:, 2, 0] = (d[:, 2] - d[:, 0]).float()
        self.p_adv = adv          # [L, 3, K] (NaN = no target)
        self.by_episode = {}
        for i, e in enumerate(self.p_e.tolist()):
            self.by_episode.setdefault(e, []).append(i)

    def shuffle_targets(self, episodes, rng: random.Random):
        """Shuffled-target control: permute step-level target rows across all valid positions of
        ``episodes`` and point target rows across their points."""
        pos = [(e, t) for e in episodes for t in range(self.lengths[e]) if self.s_valid[e, t]]
        perm = list(range(len(pos)))
        rng.shuffle(perm)
        vals = [(self.s_success[e, t].item(), self.s_cost[e, t].item(), self.s_q0[e, t].item()) for e, t in pos]
        for (e, t), j in zip(pos, perm):
            self.s_success[e, t], self.s_cost[e, t], self.s_q0[e, t] = vals[j]
        pts = [i for e in episodes for i in self.by_episode.get(e, [])]
        perm = pts[:]
        rng.shuffle(perm)
        idx, src = torch.tensor(pts, dtype=torch.long), torch.tensor(perm, dtype=torch.long)
        if len(pts):
            for name in ("p_q0", "p_success", "p_cost", "p_default_valid", "p_adv"):
                t = getattr(self, name)
                t[idx] = t[src].clone()


def _nanmse(pred, target):
    valid = ~torch.isnan(target)
    if not valid.any():
        return pred.sum() * 0.0, 0
    return ((pred[valid] - target[valid]) ** 2).mean(), int(valid.sum())


def loss_terms(model, tensors: Tensors, episodes):
    idx = torch.tensor(episodes, dtype=torch.long)
    T = max(tensors.lengths[e] for e in episodes)
    out = model.sequence(tensors.x[idx, :T])
    scale = model.spec.cost_scale
    valid = tensors.s_valid[idx, :T]
    terms = {}
    if valid.any():
        terms["s_success"] = nn.functional.binary_cross_entropy_with_logits(out["logit"][valid],
                                                                            tensors.s_success[idx, :T][valid])
        terms["s_cost"] = (((out["cost"][valid] - tensors.s_cost[idx, :T][valid]) * scale) ** 2).mean()
        terms["s_q0"] = ((out["q0"][valid] - tensors.s_q0[idx, :T][valid]) ** 2).mean()
    local = {e: j for j, e in enumerate(episodes)}
    pts = [i for e in episodes for i in tensors.by_episode.get(e, [])]
    if pts:
        pi = torch.tensor(pts, dtype=torch.long)
        rows = torch.tensor([local[e] for e in tensors.p_e[pi].tolist()], dtype=torch.long)
        steps = tensors.p_t[pi]
        dv = tensors.p_default_valid[pi]
        if dv.any():
            terms["p_q0"] = _nanmse(out["q0"][rows, steps][dv], tensors.p_q0[pi][dv])[0]
            ps = tensors.p_success[pi][dv]
            ok = ~torch.isnan(ps)
            if ok.any():
                terms["p_success"] = nn.functional.binary_cross_entropy_with_logits(
                    out["logit"][rows, steps][dv][ok], ps[ok])
            terms["p_cost"] = _nanmse(out["cost"][rows, steps][dv] * scale, tensors.p_cost[pi][dv] * scale)[0]
        adv_pred = out["adv"][rows, steps]                     # [P, 3]
        target = tensors.p_adv[pi]                              # [P, 3, K]
        terms["p_adv"] = _nanmse(adv_pred[:, :, None].expand_as(target), target)[0]
    return terms


def train(tensors: Tensors, train_eps, spec: MetaSpec, *, seed=0, epochs=30, lr=3e-3, batch=64, log=None):
    torch.manual_seed(seed)
    rng = random.Random(digest_seed("c-train-order", seed))
    model = make_model(spec)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        order = list(train_eps)
        rng.shuffle(order)
        total, n = 0.0, 0
        for start in range(0, len(order), batch):
            terms = loss_terms(model, tensors, order[start:start + batch])
            loss = sum(terms.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        history.append(total / max(1, n))
        if log is not None:
            log(epoch, history[-1])
    return model, history


@torch.no_grad()
def predict(model, tensors: Tensors, episodes):
    """Heads at every step of the given episodes: dict of [E, T(, 3)] tensors (padded)."""
    idx = torch.tensor(episodes, dtype=torch.long)
    T = max(tensors.lengths[e] for e in episodes)
    out = model.sequence(tensors.x[idx, :T])
    return {k: v for k, v in out.items() if k != "logit"}


def ece(pred, outcome, bins=10, min_count=20):
    """Reliability error: count-weighted mean |mean prediction - frequency| over equal-width bins
    with at least ``min_count`` examples (review F11). Returns (error, bins used, table)."""
    table = []
    pred, outcome = list(pred), list(outcome)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, p in enumerate(pred) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not sel:
            continue
        mp = sum(pred[i] for i in sel) / len(sel)
        fr = sum(outcome[i] for i in sel) / len(sel)
        table.append({"bin": [lo, hi], "count": len(sel), "mean_pred": mp, "frequency": fr})
    used = [r for r in table if r["count"] >= min_count]
    total = sum(r["count"] for r in used)
    err = sum(r["count"] * abs(r["mean_pred"] - r["frequency"]) for r in used) / total if total else None
    return err, len(used), table


def point_view(model, tensors: Tensors, data, episodes):
    """Per dev point: predictions and realized labels (sample = K-draw mean)."""
    preds = predict(model, tensors, episodes)
    local = {e: j for j, e in enumerate(episodes)}
    rows = []
    for i in [i for e in episodes for i in tensors.by_episode.get(e, [])]:
        e, t = int(tensors.p_e[i]), int(tensors.p_t[i])
        j = local[e]
        d = data["pt_dU"][i].tolist()
        draws = [x for x in data["pt_dU_sample"][i].tolist() if not math.isnan(x)]
        rows.append({"p": float(preds["p_success"][j, t]), "q0": float(preds["q0"][j, t]),
                     "adv": [float(x) for x in preds["adv"][j, t]],
                     "dU_default": d[0], "gain": {"sample": (sum(draws) / len(draws) - d[0]) if draws else NAN,
                                                  "mask_top": d[1] - d[0] if not math.isnan(d[1]) else NAN,
                                                  "stop": d[2] - d[0]},
                     "success_default": float(data["pt_success_d"][i]), "stratum": STRATA[int(data["pt_stratum"][i])]})
    return rows


def rule_gain(rows, margin):
    total = 0.0
    chosen = {u: 0 for u in ADV_US}
    for r in rows:
        avail = {u: r["adv"][k] for k, u in enumerate(ADV_US) if not math.isnan(r["gain"][u])}
        if not avail:
            continue
        best = max(avail, key=lambda u: (avail[u], -ADV_US.index(u)))
        if avail[best] > margin:
            total += r["gain"][best]
            chosen[best] += 1
    return total / max(1, len(rows)), chosen


def threshold_gain(rows, tau):
    sel = [r for r in rows if r["p"] < tau and not math.isnan(r["gain"]["sample"])]
    return sum(r["gain"]["sample"] for r in sel) / max(1, len(rows)), len(sel)


def register(rows):
    """Margin m and threshold τ from dev label points (ties: larger m, smaller τ)."""
    margins = [(m, *rule_gain(rows, m)) for m in MARGIN_GRID]
    best_m = max(margins, key=lambda x: (round(x[1], 12), x[0]))
    taus = [(t, *threshold_gain(rows, t)) for t in TAU_GRID]
    best_t = max(taus, key=lambda x: (round(x[1], 12), -x[0]))
    return {"margin": best_m[0], "margin_dev_gain": best_m[1],
            "margin_table": [{"margin": m, "gain": g, "chosen": c} for m, g, c in margins],
            "tau": best_t[0], "tau_dev_gain": best_t[1],
            "tau_table": [{"tau": t, "gain": g, "interventions": c} for t, g, c in taus],
            "rule": "maximize mean realized one-step gain over default on dev label points (base-state distribution)"}


def dev_metrics(model, tensors: Tensors, data, dev):
    rows = point_view(model, tensors, data, dev)
    preds = predict(model, tensors, dev)
    p, y, q_err = [], [], []
    for j, e in enumerate(dev):
        for t in range(tensors.lengths[e]):
            if tensors.s_valid[e, t]:
                p.append(float(preds["p_success"][j, t]))
                y.append(float(tensors.s_success[e, t]))
                q_err.append(abs(float(preds["q0"][j, t]) - float(tensors.s_q0[e, t])))
    err, used, table = ece(p, y) if p else (None, 0, [])
    adv_mae = {}
    for k, u in enumerate(ADV_US):
        errs = [abs(r["adv"][k] - r["gain"][u]) for r in rows if not math.isnan(r["gain"][u])]
        adv_mae[u] = sum(errs) / len(errs) if errs else None
    return {"dev_points": len(rows), "dev_steps_d0": len(p), "ece_success_steps": err, "ece_bins_used": used,
            "q0_mae_steps": sum(q_err) / len(q_err) if q_err else None, "adv_mae_points": adv_mae,
            "reliability": table}, rows
