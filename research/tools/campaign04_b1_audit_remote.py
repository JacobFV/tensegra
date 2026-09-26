"""Independent B1 audit, remote (torch, CPU only, 1 thread, read-only on results).

  (a) matching: train_log V_star/U streams across rungs per seed; model.pt weights differ across seeds.
  (b) leakage: capture the model input x at every step of a greedy batch and rebuild it independently from the
      public config fields + visible history only; compare element-wise.  Also: the pickled labels' configs equal
      split_config(split, i), and a fresh DP reproduces the pickled V* table for a few configs; the train pool
      contains no held-out generator parameter.
  (c) L0 characterization: greedy action statistics of L0 s0/s1/s2 (+ L4 s0 and pi*) on all eval splits; per-
      episode trajectory identity between seeds; logit agreement.
  (d) B-H6: value-head calibration vs own greedy return, vs V*, vs the return of the SAMPLED policy the head was
      trained on, and the L4 Q-head's max_a Q-hat as a value estimate.

usage: python campaign04_b1_audit_remote.py --results R --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import campaign04_probeworld_train as T  # noqa: E402
from tensegra import campaign04_probeworld as pw  # noqa: E402

SPLITS = ("dev", "test_iid", "heldout_price", "heldout_k", "heldout_comp")
HELD = ("heldout_price", "heldout_k", "heldout_comp")


def load_model(R, rung, seed):
    run = Path(R) / f"b-train-{rung}-s{seed}/run"
    meta = json.loads((run / "train_meta.json").read_text())
    m = T.ProbeNet(meta["hidden"])
    m.load_state_dict(torch.load(run / "model.pt"))
    m.eval()
    return m


# ------------------------------------------------------------------------------------------------ (a)

def matching(R):
    out = {}
    for s in (0, 1, 2):
        logs = {r: json.loads((Path(R) / f"b-train-{r}-s{s}/run/train_log.json").read_text()) for r in T.RUNGS}
        ups = {r: [row["update"] for row in logs[r]] for r in T.RUNGS}
        same_updates = all(ups[r] == ups["L0"] for r in T.RUNGS)
        vs_equal = all([row["V_star"] for row in logs[r]] == [row["V_star"] for row in logs["L0"]] for r in T.RUNGS)
        u0 = {r: logs[r][0]["U"] for r in T.RUNGS}
        out[s] = {"same_logged_updates": same_updates, "last_update": ups["L0"][-1],
                  "V_star_stream_identical_all_logged_updates": vs_equal,
                  "update0_U_identical": len(set(u0.values())) == 1, "update0_U": u0,
                  "final_train_regret": {r: logs[r][-1]["regret"] for r in T.RUNGS}}
    sd = {s: torch.load(Path(R) / f"b-train-L0-s{s}/run/model.pt") for s in (0, 1, 2)}
    diffs = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        diffs[f"L0 s{a} vs s{b}"] = max(float((sd[a][k] - sd[b][k]).abs().max()) for k in sd[a])
    out["L0_weight_max_abs_diff"] = diffs
    return out


# ------------------------------------------------------------------------------------------------ (b)

def rebuild_x(cfg, hist, n_done):
    """Independent rebuild of the model input from public information only (probeworld.md section 5)."""
    Rr = cfg.R
    pub = [cfg.c_probe / Rr, cfg.c_b1 / Rr, cfg.c_b2 / Rr, cfg.c_inspect / Rr, cfg.c_prop / Rr, cfg.L / Rr,
           min(cfg.C_build / Rr, 10.0) / 10.0, cfg.C_execute / Rr, cfg.C_return / Rr, cfg.C_verify / Rr,
           cfg.D_side / Rr, cfg.q, cfg.corr, cfg.p_event, cfg.eta, cfg.p_conflict, cfg.k / 8.0,
           math.log2(cfg.k) / 3.0,
           math.log(max(cfg.c_b2 / (cfg.C_build / cfg.k + cfg.C_execute + cfg.C_return + cfg.C_verify), 1e-3)) / 3.0]
    pub += [float(cfg.q < 1), float(cfg.D_side > 0), float(cfg.corr > 0), float(cfg.p_event > 0)] + list(cfg.prior)
    a1 = [0.0] * 11; o1 = [0.0] * 13; r1 = [0.0] * 5; ev = 0.0
    if not hist:
        a1[10] = o1[12] = r1[4] = 1.0
    else:
        a, o, e, rv = hist[-1]
        a1[a] = 1.0; o1[o] = 1.0; ev = float(e); r1[4 if rv is None else rv] = 1.0
    st = pw.public_state_from_history(cfg, hist)  # public transition only
    m = [0.0] * 10
    for a in pw.available(cfg, st):
        m[a] = 1.0
    return pub + a1 + o1 + [ev] + r1 + m + [st[0][0] / cfg.k]


def leakage(R, model, labels):
    res = {}
    for split in ("heldout_comp", "heldout_k"):
        pool = T.load_pool(labels, split)
        items = T.eval_items(pool, split, 4)[:128]
        captured = []
        orig = model.step

        def hook(x, h):
            captured.append(x.clone())
            return orig(x, h)
        model.step = hook
        with torch.no_grad():
            eps, steps, ep_steps = T.run_batch(model, items, "greedy", need_labels=False)
        model.step = orig
        # captured[t] rows correspond to steps[t]["idx"] (active episodes at step t)
        n = bad = 0
        maxdiff = 0.0
        for t, rec in enumerate(steps):
            for j, i in enumerate(rec["idx"].tolist()):
                hist = eps[i].history[:t]
                x2 = torch.tensor(rebuild_x(items[i][0], hist, None), dtype=torch.float32)
                d = float((captured[t][j] - x2).abs().max())
                maxdiff = max(maxdiff, d)
                bad += d > 1e-6
                n += 1
        res[split] = {"steps_checked": n, "mismatches": bad, "max_abs_diff": maxdiff, "in_dim": captured[0].shape[1]}
    return res


def label_pickles(R, labels):
    out = {}
    for split in HELD:
        with open(Path(labels) / f"{split}.pkl", "rb") as f:
            pool = pickle.load(f)
        cfg_ok = all(cfg == pw.split_config(split, idx) for idx, cfg, _, _ in pool)
        fresh = []
        for idx, cfg, V, Q in pool[:3]:
            s = pw.ExactSolver(cfg)
            s.value(pw.initial_state(cfg))
            same_keys = set(s._V) == set(V)
            maxd = max(abs(s._V[k] - V[k]) for k in V) if same_keys else None
            fresh.append({"idx": idx, "n_states": len(V), "same_state_set": same_keys, "max_abs_V_diff": maxd})
        out[split] = {"n": len(pool), "configs_equal_split_config": cfg_ok, "fresh_dp": fresh}
        del pool
    t0 = time.process_time()
    with open(Path(labels) / "train.pkl", "rb") as f:
        pool = pickle.load(f)
    bad = []
    for idx, cfg, _, _ in pool:
        cell, k, combo, _ = pw.generator_params("train", idx)
        if cfg != pw.split_config("train", idx):
            bad.append((idx, "cfg"))
        if tuple(cell) == (1, 1) or cfg.k == 4 or tuple(cfg.flags) in pw.HELDOUT_COMBOS or pw.price_cell(cfg) == (1, 1):
            bad.append((idx, "heldout_param"))
    out["train"] = {"n": len(pool), "problems": bad[:10], "n_problems": len(bad),
                    "ks": sorted({c.k for _, c, _, _ in pool}),
                    "cells": sorted({tuple(pw.price_cell(c)) for _, c, _, _ in pool}, key=str),
                    "load_cpu_s": time.process_time() - t0}
    del pool
    return out


# ------------------------------------------------------------------------------------------------ (c), (d)

def action_stats(eps, ep_steps):
    first = {}
    counts = {a: 0 for a in pw.ACTIONS}
    first_by_q = {}
    for ep, infos in zip(eps, ep_steps):
        acts = [inf["a"] for inf in infos]
        first[pw.ACTIONS[acts[0]]] = first.get(pw.ACTIONS[acts[0]], 0) + 1
        for a in acts:
            counts[pw.ACTIONS[a]] += 1
        # per-query opening action and per-query action sequence
        q_open = True
        for a in acts:
            if q_open:
                first_by_q[pw.ACTIONS[a]] = first_by_q.get(pw.ACTIONS[a], 0) + 1
                q_open = False
            if a in pw.TERMINAL:
                q_open = True
    n = len(eps)
    return {"first_action": first, "per_episode_action_counts": {a: c / n for a, c in counts.items() if c},
            "query_opening_action": first_by_q}


def top_sequences(ep_steps, n=8):
    d = {}
    for infos in ep_steps:
        seq, cur = [], []
        for inf in infos:
            cur.append(pw.ACTIONS[inf["a"]])
            if inf["a"] in pw.TERMINAL:
                seq.append(">".join(cur)); cur = []
        for q in seq:
            d[q] = d.get(q, 0) + 1
    return sorted(d.items(), key=lambda kv: -kv[1])[:n]


def calib(pairs):
    return T.value_calibration(pairs)


def rollout_values(model, items, mode, rng=None):
    """Greedy/sampled rollouts; returns (vhat, return-to-go, V*, max Q-hat over available) per visited step."""
    rows = []
    eps_all, ep_steps_all = [], []
    for c0 in range(0, len(items), 256):
        part = items[c0:c0 + 256]
        with torch.no_grad():
            eps, steps, ep_steps = T.run_batch(model, part, mode, rng=rng, need_labels=True)
            heads = [(model.v(rec["z"]).squeeze(-1), model.q(rec["z"]).masked_fill(~rec["mask"], -1e9).max(-1).values)
                     for rec in steps]
        for i, infos in enumerate(ep_steps):
            g = 0.0
            for inf in reversed(infos):
                g += inf["reward"]
                t, j = inf["step_row"], inf["row"]
                rows.append((float(heads[t][0][j]), g / 100.0, float(steps[t]["vstar"][j]) / 100.0,
                             float(heads[t][1][j]), part[i][0].k))
        eps_all += eps
        ep_steps_all += ep_steps
    return rows, eps_all, ep_steps_all


def bias(pairs):
    return sum(p - t for p, t in pairs) / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    R = a.results
    labels = str(Path(R) / "b-labels/labels")
    t0 = time.process_time()
    out = {"matching": matching(R)}
    print("matching", json.dumps(out["matching"])[:1500], flush=True)
    m_l4 = load_model(R, "L4", 0)
    out["leakage"] = leakage(R, m_l4, labels)
    print("leakage", out["leakage"], flush=True)
    out["label_pickles"] = label_pickles(R, labels)
    print("pickles", json.dumps(out["label_pickles"])[:1500], flush=True)
    models = {("L0", s): load_model(R, "L0", s) for s in (0, 1, 2)}
    for r in ("L1", "L2", "L3", "L4"):
        for s in (0, 1, 2):
            models[(r, s)] = load_model(R, r, s)
    out["l0"] = {}
    out["b_h6"] = {}
    for split in SPLITS:
        pool = T.load_pool(labels, split)
        items = T.eval_items(pool, split, 4)
        ent = {}
        seqs = {}
        # pi*
        eps, ep_steps = T.reference_policy_eval(items, "pi_star")
        ent["pi_star"] = action_stats(eps, ep_steps)
        ent["pi_star"]["top_query_sequences"] = top_sequences(ep_steps)
        for key in (("L0", 0), ("L0", 1), ("L0", 2), ("L4", 0)):
            rows, eps, ep_steps = rollout_values(models[key], items, "greedy")
            name = f"{key[0]}-s{key[1]}"
            ent[name] = action_stats(eps, ep_steps)
            ent[name]["top_query_sequences"] = top_sequences(ep_steps)
            ent[name]["mean_U"] = sum(e.utility for e in eps) / len(eps)
            seqs[name] = [tuple(inf["a"] for inf in infos) for infos in ep_steps]
        for x, y in (("L0-s0", "L0-s1"), ("L0-s0", "L0-s2"), ("L0-s1", "L0-s2")):
            same = sum(p == q for p, q in zip(seqs[x], seqs[y]))
            ent[f"identical_trajectories_{x}_vs_{y}"] = same / len(seqs[x])
        out["l0"][split] = ent
        print("l0", split, json.dumps({k: (v["first_action"] if isinstance(v, dict) else v) for k, v in ent.items()}),
              flush=True)
        if split in ("dev", "heldout_price", "heldout_k", "heldout_comp"):
            h6 = {}
            for key in [(r, s) for r in T.RUNGS for s in (0, 1, 2)] if split != "dev" else [("L0", 0), ("L4", 0)]:
                rows_g, _, _ = rollout_values(models[key], items, "greedy")
                rng = random.Random(123 + key[1])
                rows_s, _, _ = rollout_values(models[key], items, "sample", rng=rng)
                h6[f"{key[0]}-s{key[1]}"] = {
                    "greedy_vhat_vs_own_return": {k: v for k, v in calib([(r[0], r[1]) for r in rows_g]).items() if k != "bins"},
                    "greedy_vhat_vs_vstar": {k: v for k, v in calib([(r[0], r[2]) for r in rows_g]).items() if k != "bins"},
                    "greedy_own_return_vs_vstar_bias": bias([(r[1], r[2]) for r in rows_g]),
                    "greedy_vhat_bias_vs_own_return": bias([(r[0], r[1]) for r in rows_g]),
                    "sampled_vhat_vs_own_return": {k: v for k, v in calib([(r[0], r[1]) for r in rows_s]).items() if k != "bins"},
                    "sampled_vhat_bias_vs_own_return": bias([(r[0], r[1]) for r in rows_s]),
                    "greedy_maxQhat_vs_own_return": {k: v for k, v in calib([(r[3], r[1]) for r in rows_g]).items() if k != "bins"},
                    "greedy_maxQhat_vs_vstar": {k: v for k, v in calib([(r[3], r[2]) for r in rows_g]).items() if k != "bins"},
                    "n_steps_greedy": len(rows_g), "n_steps_sampled": len(rows_s),
                    "greedy_bins_vs_own_return": calib([(r[0], r[1]) for r in rows_g])["bins"]}
                print("h6", split, key, round(h6[f"{key[0]}-s{key[1]}"]["greedy_vhat_vs_own_return"]["reliability_error"], 4),
                      round(h6[f"{key[0]}-s{key[1]}"]["greedy_vhat_vs_vstar"]["reliability_error"], 4),
                      round(h6[f"{key[0]}-s{key[1]}"]["sampled_vhat_vs_own_return"]["reliability_error"], 4),
                      round(h6[f"{key[0]}-s{key[1]}"]["greedy_maxQhat_vs_own_return"]["reliability_error"], 4), flush=True)
            out["b_h6"][split] = h6
        del pool
    out["cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print("cpu", out["cpu_s"])


if __name__ == "__main__":
    main()
