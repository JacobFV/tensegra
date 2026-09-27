"""Extended-07 Phase 0a: machine-readable factor contract + static/numerical audit of the 23 derived factor
coordinates (campaign06_probeworld.FACTOR_FEATURES / factor_features).

Pure Python (no torch).  Deterministic.  New configurations and world seeds come ONLY from the extended-06 Track B
dev_smoke sub-range [6.8e9, 6.9e9) (block 6.87e9 + i: configurations; 6.875e9 + i: episode worlds; 6.877e9 + i:
Q*-comparison configurations; never protocol).

Subcommands
  contract --out factor-contract.json        write the contract (spec checked against the code's name order)
  audit    --out audit.json [--n N]          sampled-decision numerical audit (ranges, clip binding, sums,
                                             dependency identities, target-variance shares)
  qstar    --out qstar.json [--n N]          closed-form strategy costs vs exact Q* (k = 1: exact per-query
                                             comparison; k = 2: argmin agreement only)

Design: research/campaigns/extended-07/design.md (P0a); report: research/campaigns/extended-07/factor-contract.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    from tensegra import campaign04_probeworld as pw
except ImportError:  # no torch: the package __init__ imports torch; load the pure-Python modules directly
    import types
    for _k in [k for k in sys.modules if k == "tensegra" or k.startswith("tensegra.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("tensegra")
    _pkg.__path__ = [str(ROOT / "src" / "tensegra")]
    sys.modules["tensegra"] = _pkg
    from tensegra import campaign04_probeworld as pw
from tensegra import campaign06_probeworld as pw6  # noqa: E402

DEV_CFG_BASE = 6_870_000_000
DEV_WORLD_BASE = 6_875_000_000
DEV_Q_BASE = 6_877_000_000
assert all(pw6.SUBRANGES["dev_smoke"][0] <= b < b + 1_000_000 <= pw6.SUBRANGES["dev_smoke"][1]
           for b in (DEV_CFG_BASE, DEV_WORLD_BASE, DEV_Q_BASE))
CLIP = 5.0

# ------------------------------------------------------------------------------------------------ contract
# kind: exact = exact function of the public information state (Bayes posterior or one-step outcome probability);
#       bookkeeping = exact deterministic readout of public state/config;
#       approx_myopic = closed-form ONE-QUERY cost of a FIXED hand-written continuation (not Q*, not pi*, not the
#       agent's policy; ignores information-gathering, abstention, deadline, cross-query value of information).
# Nothing is policy-dependent: every coordinate is a function of (public config, public information state).
# status: SUP = supplied at inference in the SUP arm (all 23); in LRN every coordinate is a TARGET only (the policy
#       reads the stop-gradient prediction); in RAWF none is used.
G1, G2, G3, G4 = ("G1_posterior_outcome_probabilities", "G2_side_event_costs", "G3_one_query_strategy_costs",
                  "G4_amortized_build_bookkeeping")
SPEC = [
    # name, group, code block, kind, range(v1 factors U,S,C,E), normalization, depends_on, clipped, formula
    ("belief_H", G1, "posterior", "exact", [0, 1], "probability", ["history", "prior", "q", "eta", "p_conflict", "corr"], False,
     "P(theta=H | public history); pw.theta_prior (revealed-type evidence under C) x Bayes updates, rounded 1e-12"),
    ("belief_M", G1, "posterior", "exact", [0, 1], "probability", ["history", "prior", "q", "eta", "p_conflict", "corr"], False,
     "P(theta=M | public history)"),
    ("belief_F", G1, "posterior", "exact", [0, 1], "probability", ["history", "prior", "q", "eta", "p_conflict", "corr"], False,
     "P(theta=F | public history)"),
    ("belief_X", G1, "posterior", "exact", [0, 1], "probability", ["history", "prior", "q", "eta", "p_conflict", "corr"], False,
     "P(theta=X | public history)"),
    ("p_probe_resolves", G1, "outcome", "exact", [0, 1], "probability", ["f0"], False,
     "bH (IDENTICAL to belief_H): P(a probe now yields a correct candidate)"),
    ("p_probe_false_solved", G1, "outcome", "exact", [0, 0.7], "probability", ["f0", "q"], False,
     "(1 - bH)(1 - q): joint P(theta != H and probe reports 'solved')"),
    ("p_H_given_solved", G1, "outcome", "exact", [0, 1], "probability (0 if P(solved)=0)", ["f0", "f5"], False,
     "bH / (bH + (1-bH)(1-q)); 0 when bH = 0 and q = 1 (undefined -> 0)"),
    ("p_b1_resolves", G1, "outcome", "exact", [0, 1], "probability", ["f0", "f1", "f2", "reduced"], False,
     "bH + bM + bF * [reduced]: P(b1 now returns 'solved')"),
    ("exp_side_cost_rel", G2, "factor_costs", "exact", [0, 0.4], "price / R", ["f0", "D_side"], False,
     "D_side (1 - bH) / R: expected side-effect charge of a probe now"),
    ("exp_hard_cost_rel", G2, "factor_costs", "exact", [0, 3.125], "price / R (UNCLIPPED)", ["f2", "f3", "t_hard", "c_b2"], False,
     "t_hard c_b2 (bF + bX) / R: expected T surcharge of an exact call now (0 whenever T is off: every b6/b6c pool)"),
    ("event_hazard_active", G2, "factor_costs", "exact", [0, 0.6], "probability", ["p_event", "ev"], False,
     "p_event if no event yet in this query else 0: P(an event follows the next non-terminal action)"),
    ("cost_exact_b2_rel", G3, "strategy_costs", "approx_myopic", [0, 2.0], "price / R, clip [-5, 5]", ["c_b2", "f9", "f10"], True,
     "(c_b2 + hard)(1 + hazard) / R: b2 then commit, one re-run if an event invalidates (re-runs even on X)"),
    ("cost_probe_first_rel", G3, "strategy_costs", "approx_myopic", [0, 3.1], "price / R, clip [-5, 5]",
     ["c_probe", "f8", "f0", "q", "f11", "f5", "L", "f10"], True,
     "c_probe + side + q(1-bH) b2_route + (1-bH)(1-q) L + hazard c_probe, / R (forgone reward R on a false 'solved' NOT counted)"),
    ("cost_b1_first_rel", G3, "strategy_costs", "approx_myopic", [0, 2.5], "price / R, clip [-5, 5]", ["c_b1", "f9", "f7", "f11"], True,
     "c_b1 + hard + (1 - p_b1) b2_route, / R (no event re-run term, unlike b2/probe)"),
    ("cost_use_rel", G3, "strategy_costs", "approx_myopic", [0, 5.0], "price / R, clip [-5, 5] (BINDS)",
     ["C_build", "f17", "f18", "c_use"], True,
     "(C_build / max(k - i, 1) if not built else 0) + c_use, / R: build now, amortized over the remaining queries"),
    ("best_strategy_cost_rel", G3, "strategy_costs", "approx_myopic", [0, 2.0], "price / R, clip [-5, 5]",
     ["f11", "f12", "f13", "f14"], True, "min(b2, probe, b1, use) / R == min(f11..f14) (clip is monotone)"),
    ("build_value_rel", G4, "build", "approx_myopic", [-5, 5], "price / R, clip [-5, 5] (BINDS), 0 once built",
     ["f11", "f12", "f13", "c_use", "C_build", "f17", "f18"], True,
     "(rem (min(b2, probe, b1) - c_use) - C_build) / R if not built else 0; rem = max(k - i, 1)"),
    ("remaining_queries_rel", G4, "build", "bookkeeping", [0.125, 1], "rem / 8", ["k", "i"], False,
     "max(k - i, 1) / 8 (also a function of the public vector k/8 and the qfrac input)"),
    ("built", G4, "build", "bookkeeping", [0, 1], "indicator", ["built"], False, "structure built"),
    ("candidate_valid", G4, "candidate", "bookkeeping", [0, 1], "indicator", ["cand", "cand_valid"], False,
     "a candidate exists and no event invalidated it"),
    ("candidate_exact", G4, "candidate", "bookkeeping", [0, 1], "indicator", ["cand", "cand_valid"], False,
     "valid candidate from b1 / b2 / use (always correct on commit)"),
    ("candidate_trust", G4, "candidate", "approx_myopic", [0, 1], "probability-like (0 if no valid candidate)",
     ["f19", "f20", "f6"], False,
     "1 if exact; p_H_given_solved if probe candidate; else 0.  Under U (q < 1) this applies the probe likelihood a "
     "SECOND time (belief already conditions on 'solved'): trust > P(commit correct) = bH"),
    ("steps_left_rel", G4, "steps", "bookkeeping", [0, 1], "(deadline - steps) / 2; 1 when D off", ["deadline", "usage"], False,
     "remaining deadline budget of the query (constant 1 in every b6/b6c pool: D never on)"),
]
ANALYTIC_DEPS = {  # identities checked on every sampled decision (f = factor vector, c = config, st = state)
    "f4 == f0": lambda f, c, st: f[4] - f[0],
    "f5 == (1-f0)(1-q)": lambda f, c, st: f[5] - (1 - f[0]) * (1 - c.q),
    "f6 == f0/(f0+f5) (0 if denominator 0)": lambda f, c, st: f[6] - (f[0] / (f[0] + f[5]) if f[0] + f[5] > 0 else 0.0),
    "f7 == f0+f1+f2*reduced": lambda f, c, st: f[7] - (f[0] + f[1] + (f[2] if st[1][2] else 0.0)),
    "f8 == D_side/R (1-f0)": lambda f, c, st: f[8] - c.D_side / c.R * (1 - f[0]),
    "f15 == min(f11..f14)": lambda f, c, st: f[15] - min(f[11:15]),
    "f20 <= f19": lambda f, c, st: max(0.0, f[20] - f[19]),
    "f21 == f20 + (f19 - f20) f6": lambda f, c, st: f[21] - (f[20] + (f[19] - f[20]) * f[6]),
    "sum(f0..f3) == 1": lambda f, c, st: sum(f[:4]) - 1.0,
}


def contract() -> dict:
    names = [s[0] for s in SPEC]
    assert tuple(names) == pw6.FACTOR_FEATURES, "spec out of sync with campaign06_probeworld.FACTOR_FEATURES"
    rows = []
    for i, (name, group, block, kind, rng, norm, dep, clipped, formula) in enumerate(SPEC):
        rows.append({"index": i, "name": name, "group": group, "code_block": block, "kind": kind,
                     "range_v1_factors": rng, "normalization": norm, "clipped_pm5": clipped,
                     "depends_on": dep, "formula": formula,
                     "arms": {"SUP": "supplied input (exact) to the fusion layer only (not the GRU)",
                              "LRN": "target only; policy reads stop-gradient prediction", "RAWF": "unused (phi = 0)"},
                     "public_history_reconstructible": True,
                     "constant_in_b6": name in ("exp_hard_cost_rel", "steps_left_rel"),
                     "duplicate_of": "belief_H" if name == "p_probe_resolves" else None})
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r["index"])
    return {"version": "e07-factor-contract-v1", "source": "src/tensegra/campaign06_probeworld.py:factor_features",
            "n": len(rows), "groups": groups, "clip": [-CLIP, CLIP],
            "dependency_notation": "fN = coordinate N; other names are public config fields or public-state fields "
                                   "(history, reduced, ev, built, cand, cand_valid, usage, i)",
            "replacement_closure": replacement_closure(rows),
            "core_state": {"variables": ["belief(4; 3 dof)", "reduced", "ev (event this query)", "i (query index)",
                                         "built", "cand_ok", "cand_exact", "deadline steps used"],
                           "note": "given the public config, all 23 coordinates are a deterministic function of these "
                                   "public-state variables; a dependency-consistent replacement edits these and "
                                   "recomputes factor_features"},
            "decision_scale": {"eps_price_units": pw.EPS, "eps_in_rel_units": pw.EPS / 100.0,
                               "note": "cost coordinates are price / R (R = 100): the decision tolerance eps = 0.5 "
                                       "is 0.005 in these units"},
            "b6_constants": {"exp_hard_cost_rel": 0.0, "steps_left_rel": 1.0},
            "coordinates": rows}


def replacement_closure(rows) -> dict:
    """For each coordinate, every coordinate that is (transitively) a function of it: replacing f_j by its exact value
    is dependency-consistent only if every coordinate in closure[j] is recomputed too."""
    child = {r["index"]: set() for r in rows}
    for r in rows:
        for d in r["depends_on"]:
            if d.startswith("f") and d[1:].isdigit():
                child[int(d[1:])].add(r["index"])
    out = {}
    for j in child:
        seen, stack = set(), [j]
        while stack:
            for c in child[stack.pop()]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        out[str(j)] = sorted(seen)
    return out


# ------------------------------------------------------------------------------------------------ sampling

AUDIT_COMBOS = tuple(pw6.TRAIN_TYPES) + ("SCE", "UCE", "T", "D", "UT", "ST", "DT")


def dev_config(i: int, combos=AUDIT_COMBOS, ks=pw6.V3_K, base=DEV_CFG_BASE):
    return pw6.stream_config(base, i, pw.TRAIN_CELLS, ks, combos)


def random_episode_states(cfg, world_seed: int, policy_rng: random.Random, solver=None):
    """(state, history) at every decision of one episode; uniform-random policy (or pi* if solver given)."""
    ep = pw6.Episode(cfg, world_seed)
    out = []
    while not ep.done:
        out.append((ep.state, tuple(ep.history)))
        av = ep.available()
        a = solver.pi_star(ep.state) if solver is not None else policy_rng.choice(av)
        ep.step(a)
    return out


def cmd_audit(a):
    t0 = time.process_time()
    N = pw6.N_FACTOR_FEATURES
    prng = random.Random(DEV_WORLD_BASE)
    per_scope = {}
    dep_max = {k: 0.0 for k in ANALYTIC_DEPS}
    trust_excess = []
    for i in range(a.n):
        cfg, (cell, k, cmb, seed) = dev_config(i)
        scope = "b6_like" if pw6.flags_of(cfg)[4:] == (False, False) else "with_D_or_T"
        for st, hist in random_episode_states(cfg, DEV_WORLD_BASE + i, prng):
            f = pw6.factor_features(cfg, st)
            raw = raw_strategy_costs(cfg, st)
            d = per_scope.setdefault(scope, {"n": 0, "min": [math.inf] * N, "max": [-math.inf] * N,
                                             "s1": [0.0] * N, "s2": [0.0] * N, "clip_bind": [0] * N,
                                             "nonfinite": 0, "prob_out_of_range": 0})
            d["n"] += 1
            for j, x in enumerate(f):
                if not math.isfinite(x):
                    d["nonfinite"] += 1
                d["min"][j] = min(d["min"][j], x)
                d["max"][j] = max(d["max"][j], x)
                d["s1"][j] += x
                d["s2"][j] += x * x
            for j, x in raw.items():
                if abs(x) > CLIP:
                    d["clip_bind"][j] += 1
            for j in (0, 1, 2, 3, 4, 5, 6, 7, 10, 21):
                if not (-1e-12 <= f[j] <= 1 + 1e-12):
                    d["prob_out_of_range"] += 1
            for name, fn in ANALYTIC_DEPS.items():
                dep_max[name] = max(dep_max[name], abs(fn(f, cfg, st)))
            local = st[1]
            if local[3] == pw.C_PROBE and local[4]:
                trust_excess.append(f[21] - f[0])  # trust vs the true P(probe candidate correct) = bH
    out = {"n_configs": a.n, "config_seed_block": [DEV_CFG_BASE, DEV_CFG_BASE + a.n], "combos": AUDIT_COMBOS,
           "policy": "uniform random over available actions", "world_seed_block": [DEV_WORLD_BASE, DEV_WORLD_BASE + a.n],
           "dependency_max_abs_residual": dep_max, "scopes": {}}
    for scope, d in per_scope.items():
        n = d["n"]
        mean = [s / n for s in d["s1"]]
        var = [max(s2 / n - m * m, 0.0) for s2, m in zip(d["s2"], mean)]
        tot = sum(var)
        out["scopes"][scope] = {
            "n_decisions": n, "nonfinite": d["nonfinite"], "prob_out_of_range": d["prob_out_of_range"],
            "coords": [{"index": j, "name": pw6.FACTOR_FEATURES[j], "min": d["min"][j], "max": d["max"][j],
                        "mean": mean[j], "sd": math.sqrt(var[j]), "var_share": var[j] / tot if tot else 0.0,
                        "clip_bind_rate": d["clip_bind"][j] / n} for j in range(N)]}
    if trust_excess:
        out["candidate_trust_minus_bH_on_probe_candidates"] = {
            "n": len(trust_excess), "mean": sum(trust_excess) / len(trust_excess), "max": max(trust_excess),
            "frac_gt_1e-9": sum(x > 1e-9 for x in trust_excess) / len(trust_excess)}
    out["cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"cpu_s": out["cpu_s"], "scopes": {s: v["n_decisions"] for s, v in out["scopes"].items()}}))


def raw_strategy_costs(cfg, st) -> dict:
    """Unclipped values of the clipped coordinates 11..16 (mirrors factor_features; used for clip-binding rates)."""
    (i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev) = st
    R = cfg.R
    bH, bM, bF, bX = b
    p_false = (1 - bH) * (1 - cfg.q)
    p_solved = bH + p_false
    p_b1 = bH + bM + (bF if reduced else 0.0)
    hazard = 0.0 if ev else cfg.p_event
    hard = getattr(cfg, "t_hard", 0.0) * cfg.c_b2 * (bF + bX)
    side = cfg.D_side * (1 - bH)
    b2 = (cfg.c_b2 + hard) * (1 + hazard)
    pr = cfg.c_probe + side + (1 - p_solved) * b2 + p_false * cfg.L + hazard * cfg.c_probe
    b1 = cfg.c_b1 + hard + (1 - p_b1) * b2
    rem = max(cfg.k - i, 1)
    use = (0.0 if built else cfg.C_build / rem) + cfg.c_use
    best_ns = min(b2, pr, b1)
    bv = 0.0 if built else rem * (best_ns - cfg.c_use) - cfg.C_build
    return {11: b2 / R, 12: pr / R, 13: b1 / R, 14: use / R, 15: min(best_ns, use) / R, 16: bv / R}


# ------------------------------------------------------------------------------------------------ Q* comparison

STRATEGY_FIRST = {11: pw.A_B2, 12: pw.A_PROBE, 13: pw.A_B1, 14: None}  # 14: build (not built) / use (built)


def cmd_qstar(a):
    """k = 1: at every decision of pi* and random trajectories, compare closed-form cost f_j * R with the exact
    cost-equivalent R - Q*(s, first action of strategy j) (utility = R * success - costs, so R - Q* is the expected
    cost incl. the forgone reward of a wrong commit, under a-then-pi*).  Also the probe cost with the forgone reward
    added (utility-consistent closed form).  k = 2: the closed-form argmin strategy's first action vs Q*'s eps-set."""
    t0 = time.process_time()
    prng = random.Random(DEV_WORLD_BASE + 500_000)
    combos = tuple(pw6.TRAIN_TYPES) + ("SCE",)
    res = {"k1": {j: [] for j in (11, 12, 13, 14)}, "k1_best": [], "k1_probe_utility_consistent": [],
           "argmin_in_eps_set": {1: [0, 0], 2: [0, 0]}, "argmin_equals_pistar_class": {1: [0, 0], 2: [0, 0]}}
    for i in range(a.n):
        k = 1 if i % 2 == 0 else 2
        cfg, meta = pw6.stream_config(DEV_Q_BASE, i, pw.TRAIN_CELLS, (k,),
                                      tuple(c for c in combos if k >= 2 or "C" not in c))
        s = pw6.ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        states = random_episode_states(cfg, DEV_WORLD_BASE + 500_000 + i, prng, solver=s) \
            + random_episode_states(cfg, DEV_WORLD_BASE + 600_000 + i, prng)
        for st, _ in states:
            f = pw6.factor_features(cfg, st)
            raw = raw_strategy_costs(cfg, st)
            q = s.q_values(st)
            first = dict(STRATEGY_FIRST)
            first[14] = pw.A_USE if st[0][1] else pw.A_BUILD
            avail = {j: aa for j, aa in first.items() if aa in q}
            if not avail:
                continue
            jbest = min(avail, key=lambda j: raw[j])
            eps_set = pw6.eps_set(q)
            res["argmin_in_eps_set"][k][0] += avail[jbest] in eps_set
            res["argmin_in_eps_set"][k][1] += 1
            if eps_set & set(avail.values()):  # some strategy's first action is eps-optimal here
                res["argmin_equals_pistar_class"][k][0] += avail[jbest] in eps_set
                res["argmin_equals_pistar_class"][k][1] += 1
            if k == 1:
                for j, aa in avail.items():
                    res["k1"][j].append((raw[j] * cfg.R, cfg.R - q[aa]))
                if pw.A_PROBE in q:
                    bH = st[1][0][0]
                    res["k1_probe_utility_consistent"].append(
                        (raw[12] * cfg.R + (1 - bH) * (1 - cfg.q) * cfg.R, cfg.R - q[pw.A_PROBE]))
                res["k1_best"].append((min(raw[j] for j in avail) * cfg.R, cfg.R - max(q.values())))
    out = {"n_configs": a.n, "config_seed_block": [DEV_Q_BASE, DEV_Q_BASE + a.n], "eps": pw.EPS,
           "cpu_s": None, "k1": {}, "argmin": {}}
    names = {11: "cost_exact_b2", 12: "cost_probe_first", 13: "cost_b1_first", 14: "cost_use"}

    def summ(pairs):
        d = [cf - ex for cf, ex in pairs]
        return {"n": len(d), "mean_closed_minus_exact": sum(d) / len(d) if d else None,
                "mean_abs": sum(abs(x) for x in d) / len(d) if d else None,
                "frac_abs_gt_eps": sum(abs(x) > pw.EPS for x in d) / len(d) if d else None,
                "frac_closed_below_exact": sum(x < -1e-9 for x in d) / len(d) if d else None,
                "max_abs": max((abs(x) for x in d), default=None)}
    for j, pairs in res["k1"].items():
        out["k1"][names[j]] = summ(pairs)
    out["k1"]["cost_probe_first+forgone_R"] = summ(res["k1_probe_utility_consistent"])
    out["k1"]["best_strategy_cost vs R - V*"] = summ(res["k1_best"])
    for k in (1, 2):
        h, n = res["argmin_in_eps_set"][k]
        h2, n2 = res["argmin_equals_pistar_class"][k]
        out["argmin"][f"k{k}"] = {"n": n, "closed_form_argmin_first_action_in_eps_optimal_set": h / n if n else None,
                                  "n_where_some_strategy_first_action_is_eps_optimal": n2,
                                  "agreement_there": h2 / n2 if n2 else None}
    out["cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contract")
    c.add_argument("--out", required=True)
    s = sub.add_parser("audit")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=600)
    q = sub.add_parser("qstar")
    q.add_argument("--out", required=True)
    q.add_argument("--n", type=int, default=120)
    a = p.parse_args(argv)
    if a.cmd == "contract":
        Path(a.out).write_text(json.dumps(contract(), indent=1) + "\n")
    elif a.cmd == "audit":
        cmd_audit(a)
    else:
        cmd_qstar(a)


if __name__ == "__main__":
    main()
