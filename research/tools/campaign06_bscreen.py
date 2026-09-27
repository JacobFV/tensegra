"""Extended-06 Track B: exact structural screening of probeworld factor families (model-independent; pure Python).

For every candidate family F (a set of 2 or 3 factors; single factors are screened as a reference) and a sample of
configurations drawn from the family's own screening stream (probeworld-v3 generator, train price cells, k in V3_K),
the exact DP of the configuration and of every sub-combination (each factor subset switched off, same prices) gives:

  per-factor decision relevance (f in F; c_-f = c with f switched off, a valid public configuration)
    first     : pi*_{c-f}(I0) is NOT eps-optimal in c   (the optimal first decision changes when f is removed)
    set       : A*_eps(c-f, I0) != A*_eps(c, I0)          (weaker: the optimal set changes)
    any       : following pi*_c, some reachable decision exists where pi*_{c-f} (own belief tracking) is
                eps-suboptimal in c; also the probability that a pi*_c trajectory meets one
    regret    : V*(c) - V^{pi*_{c-f}}(c) > eps              (ignoring f costs more than eps in expectation)
  interaction residual at I0 (Q-vectors of all sub-combinations at the same public initial state)
    additive order 1: Q_add = Q_0 + sum_f (Q_f - Q_0); residual max_a |Q_F - Q_add|; additive argmax not eps-opt
    (triples) order 2: pairwise inclusion-exclusion; the same two quantities
    joint_flip: pi*_{c-f}(I0) is eps-suboptimal in c for EVERY f in F (no single ablation predicts the decision)
  global-shift solvability (GS): the family's best score achievable by a TRAINING-SUPPORT-OPTIMAL policy up to ONE
    global action bias.  Reference policies pi_r = pi*_{c-f} for f in F (for a held-out pair or triple the
    one-factor ablations are exactly the training-support configurations nearest to c).  Policy class:
        pi_{r,a,b}(I) = argmax_x [ Q*_{c-f}(I~, x) + b * 1{x = a} ]      (I~ = c-f's own public state)
    with a single (r, a, b) shared by every configuration of the family (b in price units, R = 100).
      GS_first  = (acc_best - acc_0) / (1 - acc_0), acc = fraction of eligible configurations whose first action is
                  eps-optimal in c; acc_0 = best unbiased reference; b on a 0.25 grid in [-100, 100], all actions
      GS_regret = 1 - Reg_best / Reg_0, Reg = mean exact regret of the whole-episode policy (bias applied at every
                  decision) over the eligible configurations; b on GS_EP_GRID, actions GS_EP_ACTIONS
      GS_vec    = like GS_first with a free per-action bias VECTOR (coordinate ascent; a global action-rate prior)
    Low GS means a global probe/exact/gather rate shift cannot stand in for composition.
  eligible: V*(c, I0) > eps (attempting the task is worthwhile).  Support counts are reported per family.

Registered selection rule (committed before the screening run; no model exists): see SELECTION below and
research/campaigns/extended-06/trackb-screen.md section 1.

Subcommands:
  run        --families F1 F2 ... --n N [--n-triple N3] --out FILE.jsonl [--workers W] [--start I]
  summarize  --inputs FILE.jsonl ... --out screen.json [--md table.md]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:
    from tensegra import campaign04_probeworld as pw  # noqa: E402
except ImportError:  # the package __init__ imports torch; load the pure-Python modules directly
    import types
    for _k in [k for k in sys.modules if k == "tensegra" or k.startswith("tensegra.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("tensegra")
    _pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "tensegra")]
    sys.modules["tensegra"] = _pkg
    from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign06_probeworld as pw6  # noqa: E402

SCREEN_VERSION = "bscreen-v1"
EPS = pw.EPS
GS_FIRST_GRID = [x / 4.0 for x in range(-400, 401)]  # price units
GS_EP_ACTIONS = (pw.A_PROBE, pw.A_B1, pw.A_B2, pw.A_INSPECT, pw.A_PROP)
GS_EP_GRID = (-30.0, -15.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 15.0, 30.0)
SELECTION = {
    "rel_regret_min": 0.30,   # every factor: fraction of eligible configs where ignoring it costs > eps
    "rel_first_min": 0.15,    # every factor: fraction where its removal changes the optimal first decision
    "gs_max": 0.35,           # GS_first and GS_regret (global shift recovers <= 35% of what ignoring a factor loses)
    "joint_flip_per_1000_min": 15.0,  # support: first decisions no single ablation predicts, per 1,000 draws
    "additive_wrong_min": 0.10,       # interaction: order-1 additive Q prediction's first action not eps-optimal
    "rank": "min_f rel_regret(f) * (1 - max(GS_first, GS_regret))",
}
# design v2 revisions 1-2 (registry B-SCREEN; supersedes SELECTION, which is still reported): among UNTOUCHED
# triples, primary = the largest minimum per-factor decision relevance ('any': the optimal first or a later decision
# changes when the factor is removed, following pi*) subject to global-shift closable fraction (GS_regret: the share of
# the regret gap of the nearest training-support-optimal policy closed by one shared per-action bias) <= .5; the next
# passing triple is the secondary holdout.
SELECTION_V2 = {"relevance": "any", "gs_closable_max": 0.5, "families": "untouched triples",
                "untouched": ("USC", "USE", "UCE", "SCE"), "extension_triples_considered": "only if they clear the "
                "rule more clearly (larger minimum relevance) than every untouched triple"}


def _sub(full):
    """All subsets of the family's factor tuple, as sorted tuples in FACTORS order."""
    return [T for m in range(len(full) + 1) for T in itertools.combinations(full, m)]


def _rq(q):
    return {int(a): round(v, 6) for a, v in sorted(q.items())}


def screen_one(task):
    family, idx = task
    t0 = time.process_time()
    fam = pw6.canon(family)
    full = tuple(fam)
    cfg = pw6.screen_config(fam, idx)
    solvers = {}
    for T in _sub(full):
        c = pw6.restrict(cfg, T)
        s = pw6.ExactSolver(c)
        s.value(pw.initial_state(c))
        solvers[T] = s
    sF = solvers[full]
    s0 = pw.initial_state(cfg)
    qF = sF.q_values(s0)
    V = sF.value(s0)
    optF = pw6.eps_set(qF)
    rec = {"family": fam, "idx": idx, "seed": pw6.screen_seed(fam, idx), "k": cfg.k, "cell": list(pw.price_cell(cfg)),
           "V": V, "eligible": V > EPS, "opt": sorted(optF), "unique": len(optF) == 1, "pi": sF.pi_star(s0),
           "Q": _rq(qF), "states": {"".join(T) or "0": len(s._V) for T, s in solvers.items()}}
    qs = {T: s.q_values(pw.initial_state(s.cfg)) for T, s in solvers.items()}
    per = {}
    for f in full:
        T = tuple(x for x in full if x != f)
        sf = solvers[T]
        pf = sf.pi_star(pw.initial_state(sf.cfg))
        val = pw6.foreign_policy_value(sF, sf)
        ex, pmeet = pw6.any_decision_change(sF, sf)
        ep = {}
        for a in GS_EP_ACTIONS:
            if a not in qF:
                continue
            ep[a] = [round(V - (val if b == 0.0 else pw6.foreign_policy_value(sF, sf, {a: b})), 6) for b in GS_EP_GRID]
        per[f] = {"first": pf not in optF, "set": pw6.eps_set(qs[T]) != optF, "pi_ablated": pf,
                  "any": ex, "p_meet": round(pmeet, 6), "regret": round(V - val, 6), "Q_ablated": _rq(qs[T]),
                  "ep_regret_grid": {str(a): r for a, r in ep.items()}}
    rec["factors"] = per
    rec["joint_flip"] = all(per[f]["first"] for f in full) and len(full) >= 2
    inter = {}
    if len(full) >= 2:
        for order in range(1, len(full)):
            qa = pw6.mobius_additive(qs, full, order)
            inter[f"order{order}"] = {"resid_q": round(max(abs(qF[a] - qa[a]) for a in qF), 6),
                                      "argmax_wrong": pw6.argmax_q(qa) not in optF, "Q_add": _rq(qa)}
        vs = {T: s.value(pw.initial_state(s.cfg)) for T, s in solvers.items()}
        v_add = vs[()] + sum(vs[(f,)] - vs[()] for f in full)
        inter["resid_V_order1"] = round(V - v_add, 6)
    rec["interaction"] = inter
    rec["cpu_s"] = round(time.process_time() - t0, 3)
    return rec


def cmd_run(a):
    tasks = []
    for fam in a.families:
        n = a.n_triple if (len(pw6.canon(fam)) >= 3 and a.n_triple) else a.n
        tasks += [(pw6.canon(fam), i) for i in range(a.start, a.start + n)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "w") as f:
        if a.workers > 1:
            with mp.Pool(a.workers) as pool:
                for rec in pool.imap(screen_one, tasks, chunksize=1):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
        else:
            for t in tasks:
                f.write(json.dumps(screen_one(t)) + "\n")
    print(json.dumps({"tasks": len(tasks), "wall_s": round(time.time() - t0, 1)}))


# ------------------------------------------------------------------------------------------------ summary

def _acc_first(rows, qkey, a_b=None, b=0.0):
    ok = 0
    for r in rows:
        q = qkey(r)
        x = pw6.argmax_q(q, {a_b: b} if a_b is not None else None)
        ok += x in r["_opt"]
    return ok / len(rows)


def _q(d):
    return {int(a): v for a, v in d.items()}


def gs_first(rows, full):
    """Best single global bias (reference, action, b) on the first decision; also the free bias-vector variant."""
    refs = {f: (lambda r, f=f: r["_qa"][f]) for f in full}
    acc0 = {f: _acc_first(rows, refs[f]) for f in full}
    best = (max(acc0.values()), max(acc0, key=acc0.get), None, 0.0)
    base0 = best[0]
    for f in full:
        for a in range(pw.N_ACTIONS):
            # the argmax of Q + b e_a changes only at breakpoints; evaluate the grid
            for b in GS_FIRST_GRID:
                if b == 0.0:
                    continue
                acc = _acc_first(rows, refs[f], a, b)
                if acc > best[0] + 1e-12:
                    best = (acc, f, a, b)
    # free bias vector (coordinate ascent from 0; a global action-rate prior on the best reference)
    vbest = (base0, None, None)
    grid = GS_FIRST_GRID[::4]
    for f in full:
        vec = {a: 0.0 for a in range(pw.N_ACTIONS)}
        cur = sum(pw6.argmax_q(r["_qa"][f], vec) in r["_opt"] for r in rows) / len(rows)
        for _ in range(3):
            improved = False
            for a in range(pw.N_ACTIONS):
                keep = vec[a]
                for b in grid:
                    vec[a] = b
                    acc = sum(pw6.argmax_q(r["_qa"][f], vec) in r["_opt"] for r in rows) / len(rows)
                    if acc > cur + 1e-12:
                        cur, keep, improved = acc, b, True
                vec[a] = keep
            if not improved:
                break
        if cur > vbest[0]:
            vbest = (cur, f, dict(vec))
    # nothing to recover (an unbiased support policy is already eps-optimal everywhere) = fully solvable: GS = 1
    rec = lambda x, x0: (x - x0) / (1.0 - x0) if x0 < 1.0 - 1e-12 else 1.0
    return {"acc_0": base0, "acc_0_by_ref": acc0, "acc_best": best[0], "best": {"ref_without": best[1],
            "action": pw.ACTIONS[best[2]] if best[2] is not None else None, "bias": best[3]},
            "GS_first": rec(best[0], base0), "acc_vec": vbest[0], "GS_vec": rec(vbest[0], base0),
            "vec_ref_without": vbest[1], "vec": ({pw.ACTIONS[a]: b for a, b in vbest[2].items() if b} if vbest[2] else {})}


def gs_regret(rows, full):
    n = len(rows)
    reg0 = {f: sum(r["factors"][f]["regret"] for r in rows) / n for f in full}
    f0 = min(reg0, key=reg0.get)
    best = (reg0[f0], f0, None, 0.0)
    for f in full:
        acts = set.intersection(*[set(r["factors"][f]["ep_regret_grid"]) for r in rows])
        for a in acts:
            for j, b in enumerate(GS_EP_GRID):
                m = sum(r["factors"][f]["ep_regret_grid"][a][j] for r in rows) / n
                if m < best[0] - 1e-12:
                    best = (m, f, int(a), b)
    R0 = reg0[f0]
    return {"regret_0": R0, "regret_0_by_ref": reg0, "regret_best": best[0],
            "best": {"ref_without": best[1], "action": pw.ACTIONS[best[2]] if best[2] is not None else None,
                     "bias": best[3]}, "GS_regret": (1.0 - best[0] / R0) if R0 > 1e-9 else 1.0}


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_family(fam, rows_all):
    full = tuple(fam)
    rows = [r for r in rows_all if r["eligible"]]
    for r in rows:
        r["_opt"] = set(r["opt"])
        r["_qa"] = {f: _q(r["factors"][f]["Q_ablated"]) for f in full}
    n, ne = len(rows_all), len(rows)
    out = {"family": fam, "factors": list(full), "n_sampled": n, "n_eligible": ne,
           "n_unique_first": sum(r["unique"] for r in rows),
           "cpu_s": round(sum(r["cpu_s"] for r in rows_all), 1),
           "k_counts": {k: sum(r["k"] == k for r in rows) for k in pw6.V3_K},
           "first_action_counts": {pw.ACTIONS[a]: sum(r["pi"] == a for r in rows) for a in range(pw.N_ACTIONS)
                                   if any(r["pi"] == a for r in rows)}}
    if not rows:
        return out
    rel = {}
    for f in full:
        rel[f] = {"first": _mean(r["factors"][f]["first"] for r in rows),
                  "set": _mean(r["factors"][f]["set"] for r in rows),
                  "any": _mean(r["factors"][f]["any"] for r in rows),
                  "any_p_meet": _mean(r["factors"][f]["p_meet"] for r in rows),
                  "regret": _mean(r["factors"][f]["regret"] > EPS for r in rows),
                  "mean_regret": _mean(r["factors"][f]["regret"] for r in rows),
                  "n_first": sum(r["factors"][f]["first"] for r in rows),
                  "n_regret": sum(r["factors"][f]["regret"] > EPS for r in rows)}
    out["relevance"] = rel
    out["joint_flip"] = {"n": sum(r["joint_flip"] for r in rows), "frac_eligible": _mean(r["joint_flip"] for r in rows),
                         "per_1000_draws": 1000.0 * sum(r["joint_flip"] for r in rows) / n}
    out["all_factors_any"] = {"n": sum(all(r["factors"][f]["any"] for f in full) for r in rows),
                              "frac_eligible": _mean(all(r["factors"][f]["any"] for f in full) for r in rows)}
    out["all_factors_regret_relevant"] = {
        "n": sum(all(r["factors"][f]["regret"] > EPS for f in full) for r in rows),
        "frac_eligible": _mean(all(r["factors"][f]["regret"] > EPS for f in full) for r in rows)}
    if len(full) >= 2:
        inter = {}
        for order in range(1, len(full)):
            key = f"order{order}"
            res = sorted(r["interaction"][key]["resid_q"] for r in rows)
            inter[key] = {"argmax_wrong": _mean(r["interaction"][key]["argmax_wrong"] for r in rows),
                          "n_argmax_wrong": sum(r["interaction"][key]["argmax_wrong"] for r in rows),
                          "resid_q_median": res[len(res) // 2], "resid_q_mean": _mean(res),
                          "resid_q_gt_eps": _mean(x > EPS for x in res)}
        inter["resid_V_order1_mean_abs"] = _mean(abs(r["interaction"]["resid_V_order1"]) for r in rows)
        inter["inherited_first"] = _mean(not r["joint_flip"] for r in rows)
        out["interaction"] = inter
        out["gs_first"] = gs_first(rows, full)
        out["gs_regret"] = gs_regret(rows, full)
        out["selection"] = select(out)
    for r in rows:
        r.pop("_opt", None)
        r.pop("_qa", None)
    return out


def select(s):
    rel = s["relevance"]
    checks = {
        "rel_regret": min(v["regret"] for v in rel.values()) >= SELECTION["rel_regret_min"],
        "rel_first": min(v["first"] for v in rel.values()) >= SELECTION["rel_first_min"],
        "gs_first": s["gs_first"]["GS_first"] <= SELECTION["gs_max"],
        "gs_regret": s["gs_regret"]["GS_regret"] <= SELECTION["gs_max"],
        "support": s["joint_flip"]["per_1000_draws"] >= SELECTION["joint_flip_per_1000_min"],
        "interaction": s["interaction"]["order1"]["argmax_wrong"] >= SELECTION["additive_wrong_min"],
    }
    score = min(v["regret"] for v in rel.values()) * (1.0 - max(s["gs_first"]["GS_first"], s["gs_regret"]["GS_regret"]))
    return {"checks": checks, "pass": all(checks.values()), "rank_score": score}


def select_v2(res):
    """Design-v2 registered rule over the summarized families (triples only)."""
    cand = []
    for fam, s in res["families"].items():
        if len(fam) != 3 or "relevance" not in s:
            continue
        mn = min(v["any"] for v in s["relevance"].values())
        gs = s["gs_regret"]["GS_regret"]
        cand.append({"family": fam, "min_relevance_any": mn, "gs_closable": gs, "passes": gs <= SELECTION_V2["gs_closable_max"],
                     "untouched": fam in SELECTION_V2["untouched"], "n_eligible": s["n_eligible"],
                     "per_factor_any": {f: v["any"] for f, v in s["relevance"].items()},
                     "all_factors_any": s.get("all_factors_any", {}).get("frac_eligible")})
    cand.sort(key=lambda c: -c["min_relevance_any"])
    base = [c for c in cand if c["untouched"] and c["passes"]]
    ext = [c for c in cand if not c["untouched"] and c["passes"]]
    primary = base[0] if base else None
    if primary and ext and ext[0]["min_relevance_any"] > primary["min_relevance_any"]:
        primary = dict(ext[0], via_extension=True)
    rest = [c for c in (base + ext) if primary is None or c["family"] != primary["family"]]
    rest.sort(key=lambda c: -c["min_relevance_any"])
    return {"rule": SELECTION_V2, "candidates": cand, "primary": primary["family"] if primary else None,
            "secondary": rest[0]["family"] if rest else None}


def _open(p):
    import gzip
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def cmd_summarize(a):
    rows = []
    for p in a.inputs:
        with _open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    res = {"version": SCREEN_VERSION, "generator": pw6.GENERATOR_VERSION, "env": pw6.VERSION, "eps": EPS,
           "selection_rule": SELECTION, "k_support": list(pw6.V3_K), "cells": "train cells (8; centre held out)",
           "gs_first_grid": [GS_FIRST_GRID[0], GS_FIRST_GRID[-1], 0.25],
           "gs_ep_grid": list(GS_EP_GRID), "gs_ep_actions": [pw.ACTIONS[x] for x in GS_EP_ACTIONS],
           "cpu_s_total": round(sum(r["cpu_s"] for r in rows), 1), "families": {}}
    order = {f: i for i, f in enumerate(pw6.SCREEN_FAMILY_ORDER)}
    for fam in sorted(fams, key=lambda f: order.get(f, 999)):
        res["families"][fam] = summarize_family(fam, fams[fam])
        print(fam, json.dumps({k: v for k, v in res["families"][fam].items()
                               if k in ("n_eligible", "selection")}), flush=True)
    passing = sorted((f for f, s in res["families"].items() if s.get("selection", {}).get("pass")),
                     key=lambda f: -res["families"][f]["selection"]["rank_score"])
    res["passing_ranked"] = passing
    res["selection_v2"] = select_v2(res)
    print("selection_v2", json.dumps({k: res["selection_v2"][k] for k in ("primary", "secondary")}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.md:
        Path(a.md).write_text(markdown_table(res))


def markdown_table(res):
    lines = ["| family | n (elig.) | relevance first / any / regret>eps, per factor | joint flip (/1000) | "
             "additive wrong o1 (o2) | resid Q med | GS_first (vec) | GS_regret | pass |",
             "|---|---|---|---|---|---|---|---|---|"]
    for fam, s in res["families"].items():
        if "relevance" not in s:
            continue
        rel = "; ".join(f"{f} {v['first']:.2f}/{v['any']:.2f}/{v['regret']:.2f}" for f, v in s["relevance"].items())
        if "interaction" not in s:
            lines.append(f"| {fam} (single) | {s['n_sampled']} ({s['n_eligible']}) | {rel} | - | - | - | - | - | - |")
            continue
        it = s["interaction"]
        o2 = f" ({it['order2']['argmax_wrong']:.2f})" if "order2" in it else ""
        lines.append(
            f"| {fam} | {s['n_sampled']} ({s['n_eligible']}) | {rel} | {s['joint_flip']['n']} "
            f"({s['joint_flip']['per_1000_draws']:.0f}) | {it['order1']['argmax_wrong']:.2f}{o2} | "
            f"{it['order1']['resid_q_median']:.2f} | {s['gs_first']['GS_first']:.2f} ({s['gs_first']['GS_vec']:.2f}) | "
            f"{s['gs_regret']['GS_regret']:.2f} | {'PASS' if s['selection']['pass'] else '-'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--families", nargs="+", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--n-triple", type=int, default=None)
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("summarize")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--md", default=None)
    a = p.parse_args(argv)
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
