"""Extended-05 B-X scorer (registry B-X, design v2 revisions 11-15): B-HR headroom gate and B-X endpoints on the new
holds, with configuration-clustered bootstrap CIs.

Pure Python + numpy.  Reads the per-episode decision records written by
``campaign04_probeworld_train.py eval --split-set b5 --episode-rows`` (``<run>/eval_episodes.jsonl.gz``).

Endpoints (per arm x seed, per group; per family = flag combo and k):
  probe_unique      s0 (first decision of the episode) probe rate where probing is the unique eps-optimal first
                    action; support = CONFIGURATIONS (the s0 endpoint is configuration-level; holds use 1 world each)
  probe_unique_qf   per-query-first-decision variant (review F11b): every query's first decision on the model's own
                    trajectory where probe is uniquely eps-optimal
  probe_not_opt     s0 probe rate where probing is not eps-optimal
  decision_acc      a_t in A*(I_t); mean delta_t; reported by DECISION DEPTH (revision 14): s0 / later query-first /
                    within-query (step >= 1)
  boundary          decision accuracy / mean delta_t by rho_eff bin ('near' = [0.5, 2)) and first-decision accuracy
                    by |probe margin| bin
  q_rank            Q-head pairwise ranking accuracy over available actions with |dQ*| >= eps (n/a unless weights.q > 0)
  regret            realized V*(I0) - U and gap regret sum_t delta_t per episode
  success           absolute success; floor success >= .8 x pi* success on the same worlds
Rates with support < 20 configurations are marked insufficient.  CIs: 95% percentile bootstrap over configurations
(clusters = (split, cfg_idx)), 1,000 resamples, fixed seed; arm - B0 differences are paired.

Groups: new_holds (U+C and S+E pooled), b5_hold_uc, b5_hold_se, uc_flag_sensitive (registered U+C subset where the
correlated flag changes V*, revision 12), uc_k1 / uc_k_gt1 strata, challenge (historical U+E, S+C), dev, test_iid,
heldout_price, heldout_k.

B-HR gate (revision 13, review F13), per hold cell (U+C, S+E), from B0 only:
  no_failure  B0 probe_unique >= .80 with >= 20 eligible configurations in >= 2/3 seeds -> nothing to repair
  evaluable   B0 probe_unique <= .70 in >= 2/3 seeds (support >= 20) -> the B-X primary is evaluable on that cell
  otherwise   ambiguous (reported; primary not evaluable)
B-X primary (per evaluable cell and on the pooled holds): arm probe_unique >= B0 + .15 in 3/3 seed pairs, arm
probe_not_opt <= .10, and realized regret non-inferior to B0 (upper 95% CI of arm - B0 <= --ni-regret, a tolerance
to be registered before scoring; default 1.0 per episode).

B-XC (registry entry B-XC, adaptive confirmation; eval --split-set b5c): groups b5c_hold_uc (fresh U+C configurations,
seed base 5.9e9), b5c_uc_flag_sensitive, b5c_uc_k1 / b5c_uc_k_gt1 -- same metrics as b5_hold_uc.  ``--bxc`` scores the
registered primary exactly: mean paired (BX1 - B0) s0 uniquely-optimal first-probe rate >= +.10 with all 3 seed-pair
differences > 0, AND gap regret lower for BX1 in 3/3 pairs; secondaries: BX1 not-optimal probe <= .10, success floor,
flag-sensitive subset readings, configuration-clustered CIs (per pair and for the mean paired difference).

Usage: campaign05_bx_score.py --arm B0 RUN RUN RUN --arm BX1 RUN... [--arm BX2 ...] [--arm BX3 ...] --out FILE
       campaign05_bx_score.py --bhr-only --arm B0 RUN RUN RUN --out FILE      (headroom gate before training arms)
       campaign05_bx_score.py --bxc --arm B0 RUN RUN RUN --arm BX1 RUN RUN RUN --out FILE   (B-XC)
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

EPS = 0.5
RHO_BINS = (0.0, 0.5, 1.0, 2.0, 4.0, float("inf"))
NEAR = (0.5, 2.0)
MARGIN_BINS = (0.0, 2.0, 5.0, 10.0, float("inf"))
MIN_SUPPORT = 20
NO_FAILURE, EVALUABLE = 0.80, 0.70
LIFT = 0.15
NOT_OPT_MAX = 0.10
PROBE = 0  # action index of 'probe' (campaign04_probeworld.ACTIONS)
DEPTHS = ("s0", "query_first", "in_query")

GROUPS = {
    "new_holds": lambda r: r["split"] in ("b5_hold_uc", "b5_hold_se"),
    "b5_hold_uc": lambda r: r["split"] == "b5_hold_uc",
    "b5_hold_se": lambda r: r["split"] == "b5_hold_se",
    "uc_flag_sensitive": lambda r: r["split"] == "b5_hold_uc" and bool(r.get("flag_sensitive")),
    "uc_k1": lambda r: r["split"] == "b5_hold_uc" and r["k"] == 1,
    "uc_k_gt1": lambda r: r["split"] == "b5_hold_uc" and r["k"] > 1,
    "challenge": lambda r: r["split"] == "heldout_comp",
    "challenge_ue": lambda r: r["split"] == "heldout_comp" and r["combo"] == "unreliable+events",
    "challenge_sc": lambda r: r["split"] == "heldout_comp" and r["combo"] == "side_effect+correlated",
    "dev": lambda r: r["split"] == "b5_dev",
    "test_iid": lambda r: r["split"] == "b5_test_iid",
    "heldout_price": lambda r: r["split"] == "b5_heldout_price",
    "heldout_k": lambda r: r["split"] == "b5_heldout_k",
    # B-XC (fresh U+C configurations, eval --split-set b5c); these groups select no b5 row, so b5 outputs are unchanged
    "b5c_hold_uc": lambda r: r["split"] == "b5c_hold_uc",
    "b5c_uc_flag_sensitive": lambda r: r["split"] == "b5c_hold_uc" and bool(r.get("flag_sensitive")),
    "b5c_uc_k1": lambda r: r["split"] == "b5c_hold_uc" and r["k"] == 1,
    "b5c_uc_k_gt1": lambda r: r["split"] == "b5c_hold_uc" and r["k"] > 1,
}
CI_GROUPS = ("new_holds", "b5_hold_uc", "b5_hold_se", "uc_flag_sensitive", "challenge", "b5c_hold_uc",
             "b5c_uc_flag_sensitive")
PRIMARY_CELLS = ("b5_hold_uc", "b5_hold_se")
# B-XC (registry entry B-XC): BX1 vs B0, L1, fresh seeds 20-22 paired, on b5c_hold_uc at world offset 900
BXC_CELL = "b5c_hold_uc"
BXC_GROUPS = ("b5c_hold_uc", "b5c_uc_flag_sensitive", "b5c_uc_k1", "b5c_uc_k_gt1")
BXC_LIFT = 0.10
BXC_SEEDS = (20, 21, 22)
BXC_WORLD_OFFSET = 900
BXC_TRAIN_SPLITS = {"B0": "b5_train", "BX1": "b5x_train"}


def read_rows(run: Path, name="eval_episodes.jsonl.gz"):
    with gzip.open(Path(run) / name, "rt") as f:
        head = json.loads(f.readline())["_meta"]
        rows = [json.loads(line) for line in f]
    return head, rows


def _bin(x, bins):
    for j, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= x < hi:
            return j
    return len(bins) - 2


def episode_stats(r: dict, q_trained: bool) -> dict:
    """Additive numerators/denominators of every endpoint for one episode."""
    st = {"ep": 1.0, "regret": r["V_star"] - r["U"], "gap": sum(d["delta"] for d in r["decisions"]),
          "success": r["success"], "pi_success": r["pi_star_success"],
          "pu_n": float(r["probe_unique_opt"]), "pu_k": float(r["probe_unique_opt"] and r["first_probe"]),
          "no_n": float(not r["probe_eps_opt"]), "no_k": float((not r["probe_eps_opt"]) and r["first_probe"])}
    rb = _bin(r["rho_eff"], RHO_BINS)
    near = NEAR[0] <= r["rho_eff"] < NEAR[1]
    for t, d in enumerate(r["decisions"]):
        ok = float(d["a"] in d["opt"])
        depth = "s0" if t == 0 else ("query_first" if d.get("step_in_query", 1) == 0 else "in_query")
        for key in ("dec", f"rho{rb}", "dep_" + depth) + (("near",) if near else ()):
            st[key + "_n"] = st.get(key + "_n", 0.0) + 1.0
            st[key + "_c"] = st.get(key + "_c", 0.0) + ok
            st[key + "_d"] = st.get(key + "_d", 0.0) + d["delta"]
        if d.get("step_in_query", 1) == 0 and d["opt"] == [PROBE]:
            st["puq_n"] = st.get("puq_n", 0.0) + 1.0
            st["puq_k"] = st.get("puq_k", 0.0) + float(d["a"] == PROBE)
        if q_trained:
            qs, qh = d["Q"], d["qhat"]
            for x in range(len(qs)):
                for y in range(x + 1, len(qs)):
                    dq = qs[x] - qs[y]
                    if abs(dq) < EPS:
                        continue
                    dh = qh[x] - qh[y]
                    st["qr_n"] = st.get("qr_n", 0.0) + 1.0
                    st["qr_c"] = st.get("qr_c", 0.0) + float((dq > 0 and dh > 0) or (dq < 0 and dh < 0))
    d0 = r["decisions"][0]
    if d0.get("probe_margin") is not None:
        mb = _bin(abs(d0["probe_margin"]), MARGIN_BINS)
        st[f"m{mb}_n"] = 1.0
        st[f"m{mb}_c"] = float(d0["a"] in d0["opt"])
    return st


def _add(acc, st, w=1.0):
    for k, v in st.items():
        acc[k] = acc.get(k, 0.0) + w * v


def metrics(tot: dict, q_trained: bool, n_cfg_pu: int | None = None) -> dict:
    g = lambda k: tot.get(k, 0.0)
    rate = lambda num, den: (g(num) / g(den)) if g(den) > 0 else None
    n = g("ep")
    out = {"episodes": n, "regret": g("regret") / n if n else None, "gap_regret": g("gap") / n if n else None,
           "success": g("success") / n if n else None, "pi_star_success": g("pi_success") / n if n else None,
           "probe_unique": rate("pu_k", "pu_n"), "n_probe_unique": g("pu_n"),
           "probe_unique_qf": rate("puq_k", "puq_n"), "n_probe_unique_qf": g("puq_n"),
           "probe_not_opt": rate("no_k", "no_n"), "n_probe_not_opt": g("no_n"),
           "decision_acc": rate("dec_c", "dec_n"), "mean_delta_per_decision": rate("dec_d", "dec_n"),
           "n_decisions": g("dec_n"),
           "near_boundary_acc": rate("near_c", "near_n"), "near_boundary_mean_delta": rate("near_d", "near_n"),
           "n_near_boundary_decisions": g("near_n"),
           "q_rank_acc": rate("qr_c", "qr_n") if q_trained else "n/a", "n_q_rank_pairs": g("qr_n") if q_trained else 0}
    out["by_depth"] = {dp: {"n": g(f"dep_{dp}_n"), "acc": rate(f"dep_{dp}_c", f"dep_{dp}_n"),
                            "mean_delta": rate(f"dep_{dp}_d", f"dep_{dp}_n")} for dp in DEPTHS}
    out["success_floor_pass"] = (out["success"] >= 0.8 * out["pi_star_success"]) if n else None
    out["rho_bins"] = [{"lo": lo, "hi": hi, "n_decisions": g(f"rho{j}_n"), "acc": rate(f"rho{j}_c", f"rho{j}_n"),
                        "mean_delta": rate(f"rho{j}_d", f"rho{j}_n")}
                       for j, (lo, hi) in enumerate(zip(RHO_BINS[:-1], RHO_BINS[1:]))]
    out["probe_margin_bins"] = [{"lo": lo, "hi": hi, "n_episodes": g(f"m{j}_n"),
                                 "first_decision_acc": rate(f"m{j}_c", f"m{j}_n")}
                                for j, (lo, hi) in enumerate(zip(MARGIN_BINS[:-1], MARGIN_BINS[1:]))]
    if n_cfg_pu is not None:
        out["n_probe_unique_configs"] = n_cfg_pu
    out["insufficient"] = sorted(k for k, nk in (("probe_unique", "pu_n"), ("probe_unique_qf", "puq_n"),
                                                 ("probe_not_opt", "no_n"), ("near_boundary_acc", "near_n"))
                                 if g(nk) < MIN_SUPPORT)
    if n_cfg_pu is not None and n_cfg_pu < MIN_SUPPORT and "probe_unique" not in out["insufficient"]:
        out["insufficient"] = sorted(out["insufficient"] + ["probe_unique"])
    return out


SCALARS = ("probe_unique", "probe_unique_qf", "probe_not_opt", "decision_acc", "near_boundary_acc", "regret",
           "gap_regret", "success", "q_rank_acc")


def clusters_of(rows, q_trained, sel):
    cl = {}
    for r in rows:
        if sel(r):
            _add(cl.setdefault((r["split"], r["cfg_idx"]), {}), episode_stats(r, q_trained))
    return cl


def _total(cl):
    tot = {}
    for v in cl.values():
        _add(tot, v)
    return tot


def _n_cfg_pu(cl):
    return sum(1 for v in cl.values() if v.get("pu_n", 0) > 0)


def _boot_draws(cl_list, q_list, n_boot, seed):
    """Joint configuration resampling (common cluster keys): per clusters dict, the SCALARS of every draw."""
    import numpy as np
    keys = sorted(set.intersection(*[set(c) for c in cl_list]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    W = np.stack([np.bincount(row, minlength=len(keys)) for row in idx]).astype(float)
    draws = []
    for i, cl in enumerate(cl_list):
        fields = sorted({f for k in keys for f in cl[k]})
        M = np.array([[cl[k].get(f, 0.0) for f in fields] for k in keys])
        T = W @ M
        draws.append([{s: m[s] for s in SCALARS} for m in (metrics(dict(zip(fields, row)), q_list[i]) for row in T)])
    return keys, draws


def _ci(vals):
    vals = sorted(v for v in vals if isinstance(v, (int, float)))
    if not vals:
        return None
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def bootstrap(cl_list, q_list, n_boot=1000, seed=20260927):
    """Percentile CIs of SCALARS for each clusters dict in cl_list (common cluster keys, resampled jointly) and of
    the paired differences cl_list[i] - cl_list[0].  numpy for the resampling sums."""
    keys, draws = _boot_draws(cl_list, q_list, n_boot, seed)
    ci = _ci
    out = []
    for i in range(len(cl_list)):
        ent = {s: ci([d[s] for d in draws[i]]) for s in SCALARS}
        if i > 0:
            ent["diff_vs_first"] = {s: ci([(d[s] - d0[s]) if isinstance(d[s], (int, float)) and
                                           isinstance(d0[s], (int, float)) else None
                                           for d, d0 in zip(draws[i], draws[0])]) for s in SCALARS}
        out.append(ent)
    return out, len(keys)


def load_arms(arms: dict, name="eval_episodes.jsonl.gz"):
    data = {}
    for arm, runs in arms.items():
        lst = []
        for run in runs:
            head, rows = read_rows(run, name)
            lst.append((head["seed"], head, rows))
        data[arm] = sorted(lst, key=lambda x: x[0])
    return data


def arm_groups(lst, n_boot):
    groups = {}
    for gname, sel in GROUPS.items():
        per_seed = []
        for seed, head, rows in lst:
            q = head["q_head_trained"]
            cl = clusters_of(rows, q, sel)
            if not cl:
                continue
            m = metrics(_total(cl), q, _n_cfg_pu(cl))
            fam = {}
            for r in rows:
                if sel(r):
                    _add(fam.setdefault(r["combo"], {}), episode_stats(r, q))
                    _add(fam.setdefault(f"k={r['k']}", {}), episode_stats(r, q))
            m["by_family"] = {k: {s: metrics(v, q)[s] for s in
                                  ("episodes", "probe_unique", "n_probe_unique", "probe_not_opt", "n_probe_not_opt",
                                   "decision_acc", "regret", "gap_regret", "success")} for k, v in sorted(fam.items())}
            m["n_clusters"] = len(cl)
            if gname in CI_GROUPS and n_boot:
                (ci,), _ = bootstrap([cl], [q], n_boot)
                m["ci95"] = ci
            per_seed.append({"seed": seed, **m})
        if per_seed:
            groups[gname] = per_seed
    return groups


def bhr_gate(b0_groups) -> dict:
    """B-HR headroom gate per primary cell, from B0's per-seed s0 uniquely-optimal probe rates."""
    out = {}
    for cell in PRIMARY_CELLS + ("uc_flag_sensitive",):
        per = b0_groups.get(cell, [])
        vals = [(p["probe_unique"], p.get("n_probe_unique_configs", p["n_probe_unique"])) for p in per]
        ok = [(v, n) for v, n in vals if v is not None and n >= MIN_SUPPORT]
        n_seeds = len(per)
        need = -(-2 * n_seeds // 3) if n_seeds else 0  # >= 2/3 of seeds
        nofail = sum(v >= NO_FAILURE for v, _ in ok)
        evaluable = sum(v <= EVALUABLE for v, _ in ok)
        verdict = ("insufficient" if len(ok) < need or not n_seeds else
                   "no_failure" if nofail >= need else "evaluable" if evaluable >= need else "ambiguous")
        out[cell] = {"per_seed": [{"probe_unique": v, "support_configs": n} for v, n in vals], "verdict": verdict,
                     "rule": f"no_failure if >= {NO_FAILURE} (support >= {MIN_SUPPORT}) in >= 2/3 seeds; evaluable "
                             f"if <= {EVALUABLE} in >= 2/3 seeds"}
    return out


def score(arms: dict, n_boot=1000, ni_regret=1.0) -> dict:
    data = load_arms(arms)
    res = {"tool": "campaign05_bx_score", "eps": EPS, "min_support": MIN_SUPPORT, "n_boot": n_boot,
           "rho_bins": RHO_BINS, "near_boundary": NEAR, "ni_regret": ni_regret, "arms": {}}
    for arm, lst in data.items():
        h = lst[0][1]
        res["arms"][arm] = {"seeds": [s for s, _, _ in lst], "inputs": h["inputs"], "arch": h.get("arch", "flat"),
                            "params": h.get("params"), "train_split": h["train_split"],
                            "q_head_trained": h["q_head_trained"], "groups": arm_groups(lst, n_boot)}
    if "B0" not in data:
        return res
    res["B_HR"] = bhr_gate(res["arms"]["B0"]["groups"])
    res["vs_B0"] = {}
    for arm in data:
        if arm == "B0":
            continue
        pairs = []
        for (s0, h0, r0), (s1, h1, r1) in zip(data["B0"], data[arm]):
            ent = {"seed_B0": s0, "seed_arm": s1}
            for gname in ("new_holds",) + PRIMARY_CELLS + ("uc_flag_sensitive", "challenge"):
                c0 = clusters_of(r0, h0["q_head_trained"], GROUPS[gname])
                c1 = clusters_of(r1, h1["q_head_trained"], GROUPS[gname])
                if not c0 or not c1:
                    continue
                m0 = metrics(_total(c0), h0["q_head_trained"], _n_cfg_pu(c0))
                m1 = metrics(_total(c1), h1["q_head_trained"], _n_cfg_pu(c1))
                cis, _ = bootstrap([c0, c1], [h0["q_head_trained"], h1["q_head_trained"]], n_boot)
                d = {s: (m1[s] - m0[s]) if isinstance(m1[s], (int, float)) and isinstance(m0[s], (int, float))
                     else None for s in SCALARS}
                reg_ci = cis[1]["diff_vs_first"]["regret"]
                e = {"B0": {s: m0[s] for s in SCALARS}, arm: {s: m1[s] for s in SCALARS}, "diff": d,
                     "diff_ci95": cis[1]["diff_vs_first"],
                     "support_configs": {"B0": m0["n_probe_unique_configs"], arm: m1["n_probe_unique_configs"]},
                     "by_depth": {"B0": m0["by_depth"], arm: m1["by_depth"]}}
                e["pair_pass"] = bool(d["probe_unique"] is not None and d["probe_unique"] >= LIFT and
                                      m1["probe_not_opt"] is not None and m1["probe_not_opt"] <= NOT_OPT_MAX and
                                      reg_ci is not None and reg_ci[1] <= ni_regret and
                                      min(m0["n_probe_unique_configs"], m1["n_probe_unique_configs"]) >= MIN_SUPPORT)
                ent[gname] = e
            pairs.append(ent)
        res["vs_B0"][arm] = pairs
    prim = {}
    for arm in res["vs_B0"]:
        if arm == "BO":
            continue  # diagnostic arm, never a claim
        prim[arm] = {}
        for cell in ("new_holds",) + PRIMARY_CELLS:
            if cell != "new_holds" and res["B_HR"][cell]["verdict"] != "evaluable":
                prim[arm][cell] = {"evaluable": False, "B_HR": res["B_HR"][cell]["verdict"]}
                continue
            passes = [p.get(cell, {}).get("pair_pass", False) for p in res["vs_B0"][arm]]
            prim[arm][cell] = {"evaluable": True, "seed_pairs": len(passes), "passes": sum(passes),
                               "supported": len(passes) >= 3 and all(passes)}
    res["BX_primary"] = prim
    return res


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def bxc_score(arms: dict, n_boot=1000, name="eval_episodes.jsonl.gz") -> dict:
    """Registry entry B-XC (adaptive confirmation), scored exactly as registered on b5c_hold_uc:
      primary    mean over the 3 seed pairs of (BX1 - B0) s0 uniquely-optimal first-probe rate >= +.10, all 3 pair
                 differences > 0, AND gap regret (per episode) lower for BX1 in 3/3 pairs
      secondary  BX1 not-optimal probe <= .10; success >= .8 x pi* success (same worlds); flag-sensitive subset
                 readings; configuration-clustered 95% CIs (per pair, and for the mean paired difference)
    Validity checks (reported; the primary is only 'confirmed' when valid): arms B0 and BX1 with 3 runs each, paired
    seeds {20, 21, 22}, train splits b5_train / b5x_train, world offset 900, every row on b5c_hold_uc with 1 world per
    configuration, and the same configurations in every run."""
    data = load_arms({k: arms[k] for k in ("B0", "BX1")}, name)
    b0, bx = data["B0"], data["BX1"]
    q = {arm: [h["q_head_trained"] for _, h, _ in lst] for arm, lst in data.items()}
    cfgsets = [frozenset((r["split"], r["cfg_idx"]) for r in rows) for lst in (b0, bx) for _, _, rows in lst]
    validity = {
        "three_pairs": len(b0) == 3 and len(bx) == 3,
        "seeds_paired": [s for s, _, _ in b0] == [s for s, _, _ in bx],
        "seeds_registered": sorted(s for s, _, _ in b0) == list(BXC_SEEDS) and sorted(s for s, _, _ in bx) == list(BXC_SEEDS),
        "train_splits": all(h["train_split"] == BXC_TRAIN_SPLITS[arm] for arm, lst in data.items() for _, h, _ in lst),
        "rung_L1_public_flat": all(h["rung"] == "L1" and h.get("inputs", "public") == "public"
                                   and h.get("arch", "flat") == "flat" for lst in data.values() for _, h, _ in lst),
        "world_offset_900": all(h.get("world_offset") == BXC_WORLD_OFFSET for lst in data.values() for _, h, _ in lst),
        "rows_b5c_only_one_world": all(r["split"] == BXC_CELL and r["rep"] == 0 for lst in data.values()
                                       for _, _, rows in lst for r in rows)
        and all(len(rows) == len({r["cfg_idx"] for r in rows}) for lst in data.values() for _, _, rows in lst),
        "same_configurations": len(set(cfgsets)) == 1 and bool(cfgsets[0]),
    }
    out = {"registry": "B-XC", "cell": BXC_CELL, "lift": BXC_LIFT, "validity": validity,
           "valid": all(validity.values()), "n_configurations": len(cfgsets[0]) if cfgsets else 0, "groups": {}}
    for gname in BXC_GROUPS:
        sel = GROUPS[gname]
        pairs, cls0, cls1 = [], [], []
        for (s0, h0, r0), (s1, h1, r1) in zip(b0, bx):
            c0, c1 = clusters_of(r0, h0["q_head_trained"], sel), clusters_of(r1, h1["q_head_trained"], sel)
            if not c0 or not c1:
                continue
            cls0.append(c0)
            cls1.append(c1)
            m0 = metrics(_total(c0), h0["q_head_trained"], _n_cfg_pu(c0))
            m1 = metrics(_total(c1), h1["q_head_trained"], _n_cfg_pu(c1))
            e = {"seed_B0": s0, "seed_BX1": s1,
                 "B0": {s: m0[s] for s in SCALARS + ("pi_star_success", "success_floor_pass")},
                 "BX1": {s: m1[s] for s in SCALARS + ("pi_star_success", "success_floor_pass")},
                 "diff": {s: (m1[s] - m0[s]) if _num(m1[s]) and _num(m0[s]) else None for s in SCALARS},
                 "support_configs": {"B0": m0["n_probe_unique_configs"], "BX1": m1["n_probe_unique_configs"]},
                 "n_clusters": len(c0), "insufficient": {"B0": m0["insufficient"], "BX1": m1["insufficient"]},
                 "by_depth": {"B0": m0["by_depth"], "BX1": m1["by_depth"]}}
            if n_boot:
                cis, _ = bootstrap([c0, c1], [h0["q_head_trained"], h1["q_head_trained"]], n_boot)
                e["ci95"] = {"B0": {s: cis[0][s] for s in SCALARS}, "BX1": {s: cis[1][s] for s in SCALARS},
                             "diff": cis[1]["diff_vs_first"]}
            pairs.append(e)
        if not pairs:
            continue
        d_pu = [p["diff"]["probe_unique"] for p in pairs]
        d_gap = [p["diff"]["gap_regret"] for p in pairs]
        g = {"pairs": pairs,
             "mean_diff": {s: (sum(p["diff"][s] for p in pairs) / len(pairs))
                           if all(_num(p["diff"][s]) for p in pairs) else None for s in SCALARS}}
        if n_boot and len(cls0) == len(pairs):  # CI of the mean paired difference, configurations resampled jointly
            _, draws = _boot_draws(cls0 + cls1, q["B0"][:len(cls0)] + q["BX1"][:len(cls1)], n_boot, 20260927)
            n = len(cls0)
            g["mean_diff_ci95"] = {s: _ci([sum(draws[n + i][j][s] - draws[i][j][s] for i in range(n)) / n
                                           if all(_num(draws[n + i][j][s]) and _num(draws[i][j][s]) for i in range(n))
                                           else None for j in range(n_boot)]) for s in SCALARS}
        g["probe_unique_gain"] = {"per_pair": d_pu, "mean": g["mean_diff"]["probe_unique"],
                                  "all_pairs_positive": len(d_pu) == 3 and all(_num(d) and d > 0 for d in d_pu)}
        g["gap_regret_lower"] = {"per_pair": d_gap, "n_lower": sum(_num(d) and d < 0 for d in d_gap)}
        g["BX1_not_opt_le_10"] = [(_num(p["BX1"]["probe_not_opt"]) and p["BX1"]["probe_not_opt"] <= NOT_OPT_MAX)
                                  for p in pairs]
        g["success_floor_pass"] = {"B0": [p["B0"]["success_floor_pass"] for p in pairs],
                                   "BX1": [p["BX1"]["success_floor_pass"] for p in pairs]}
        out["groups"][gname] = g
    cell = out["groups"].get(BXC_CELL)
    if cell is None:
        out["primary"] = {"evaluable": False}
        return out
    mean = cell["probe_unique_gain"]["mean"]
    prim = {"mean_probe_unique_gain": mean, "per_pair_gain": cell["probe_unique_gain"]["per_pair"],
            "mean_gain_ge_lift": _num(mean) and mean >= BXC_LIFT,
            "all_3_pairs_positive": cell["probe_unique_gain"]["all_pairs_positive"],
            "gap_regret_lower_3_of_3": len(cell["pairs"]) == 3 and cell["gap_regret_lower"]["n_lower"] == 3,
            "per_pair_gap_regret_diff": cell["gap_regret_lower"]["per_pair"],
            "rule": "mean paired (BX1 - B0) s0 uniquely-optimal first-probe rate >= +.10 with all 3 seed-pair "
                    "differences > 0, AND gap regret lower for BX1 in 3/3 pairs"}
    prim["pass"] = bool(prim["mean_gain_ge_lift"] and prim["all_3_pairs_positive"] and prim["gap_regret_lower_3_of_3"])
    prim["verdict"] = "invalid" if not out["valid"] else ("confirmed" if prim["pass"] else "not_confirmed")
    out["primary"] = prim
    sens = out["groups"].get("b5c_uc_flag_sensitive", {})
    out["secondary"] = {
        "BX1_not_opt_le_10_all_seeds": all(cell["BX1_not_opt_le_10"]) and len(cell["BX1_not_opt_le_10"]) == 3,
        "BX1_not_opt_per_seed": [p["BX1"]["probe_not_opt"] for p in cell["pairs"]],
        "success_floor_BX1_all_seeds": all(cell["success_floor_pass"]["BX1"]),
        "success_floor_B0_all_seeds": all(cell["success_floor_pass"]["B0"]),
        "flag_sensitive": {k: sens.get(k) for k in ("probe_unique_gain", "gap_regret_lower", "mean_diff",
                                                    "mean_diff_ci95")} if sens else None,
        "flag_sensitive_support_configs": [p["support_configs"] for p in sens.get("pairs", [])],
        "mean_diff_ci95": cell.get("mean_diff_ci95"),
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "RUN"))
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--ni-regret", type=float, default=1.0, help="regret non-inferiority tolerance (register first)")
    p.add_argument("--bhr-only", action="store_true", help="B-HR gate from B0 alone (before training other arms)")
    p.add_argument("--bxc", action="store_true",
                   help="registry B-XC: B0 vs BX1 (seeds 20-22) on b5c_hold_uc (eval --split-set b5c)")
    p.add_argument("--episodes-name", default="eval_episodes.jsonl.gz",
                   help="episode-rows file inside each run dir (eval --out-name X.json -> X_episodes.jsonl.gz)")
    a = p.parse_args(argv)
    arms = {x[0]: x[1:] for x in a.arm}
    if a.bxc:
        data = load_arms({k: arms[k] for k in ("B0", "BX1")}, a.episodes_name)
        res = {"tool": "campaign05_bx_score --bxc", "eps": EPS, "min_support": MIN_SUPPORT, "n_boot": a.n_boot,
               "episodes_name": a.episodes_name,
               "runs": {k: [str(r) for r in arms[k]] for k in ("B0", "BX1")},
               "arms": {arm: {"seeds": [s for s, _, _ in lst], "train_split": lst[0][1]["train_split"],
                              "groups": {g: v for g, v in arm_groups(lst, a.n_boot).items() if g in BXC_GROUPS}}
                        for arm, lst in data.items()},
               "B_XC": bxc_score(arms, a.n_boot, a.episodes_name)}
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("B-XC validity:", res["B_XC"]["validity"])
        print("B-XC primary:", {k: v for k, v in res["B_XC"]["primary"].items() if k != "rule"})
        return
    if a.bhr_only:
        data = load_arms({"B0": arms["B0"]})
        groups = arm_groups(data["B0"], a.n_boot)
        res = {"tool": "campaign05_bx_score --bhr-only", "B_HR": bhr_gate(groups), "B0_groups": groups}
    else:
        res = score(arms, a.n_boot, a.ni_regret)
    Path(a.out).write_text(json.dumps(res, indent=1))
    if "B_HR" in res:
        print("B-HR:", {c: v["verdict"] for c, v in res["B_HR"].items()})
    for arm, ent in res.get("arms", {}).items():
        for s in ent["groups"].get("new_holds", []):
            print(arm, s["seed"], {k: s[k] for k in ("probe_unique", "n_probe_unique", "probe_not_opt", "regret")})
    if "BX_primary" in res:
        print("BX primary:", res["BX_primary"])


if __name__ == "__main__":
    main()
