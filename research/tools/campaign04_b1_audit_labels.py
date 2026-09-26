"""Independent B1 audit, label integrity (pure Python; runs locally on CPU).

1. Split disjointness by generator parameters, re-derived from the generated Config itself (price cell from
   prices, k, flags) as well as from generator_params, for train 0..383 and every eval split 0..127.
2. Brute-force expectimax over RAW visible histories (no information-state compression, no belief rounding,
   posterior by likelihood enumeration over hidden theta; auditor's own re-coding of the action/outcome
   rules from probeworld.md) for k = 1 held-out configs: V*(root), Q*(root, a) for all a, and V*/Q* at
   depth-1/2 histories, compared with ExactSolver.
3. Monte-Carlo rejection check (k >= 2, events, unreliable, correlated): for visible histories taken from
   pi* rollouts on held-out eval worlds, sample hidden worlds from the SIMULATOR conditioned on the visible
   history (replay + reject), force action a, continue with pi*, and compare the mean utility-to-go with
   Q*(I, a) (z-score).
4. Labels are a function of the visible history: random-policy rollouts on held-out configs; identical
   histories -> identical (V*, Q*, A*); simulator state == public_state_from_history(history).

usage: python research/tools/campaign04_b1_audit_labels.py [--out FILE] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

try:  # the tensegra package __init__ imports torch; load the pure-Python module directly when torch is absent
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from tensegra import campaign04_probeworld as pw  # noqa: E402
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "campaign04_probeworld", Path(__file__).resolve().parents[2] / "src/tensegra/campaign04_probeworld.py")
    pw = importlib.util.module_from_spec(_spec)
    sys.modules["campaign04_probeworld"] = pw
    _spec.loader.exec_module(pw)

H, M, F, X = range(4)
PROBE, B1, B2, INSPECT, PROP, BUILD, USE, COMMIT, CINF, ABSTAIN = range(10)
TERM = {COMMIT, CINF, ABSTAIN}


# ----------------------------------------------------------------------------------------- 1. splits

def audit_splits():
    held_cell, held_k = (1, 1), 4
    held_combos = {(True, False, False, True), (False, True, True, False)}
    rows = {}
    seen = {}
    problems = []
    for split, n in [("train", 384), ("dev", 128), ("test_iid", 128), ("heldout_price", 128), ("heldout_k", 128),
                     ("heldout_comp", 128)]:
        cells, ks, combos = set(), set(), set()
        for i in range(n):
            gcell, gk, gcombo, _ = pw.generator_params(split, i)
            cfg = pw.split_config(split, i)
            cell = pw.price_cell(cfg)
            # independent recovery of the cell from prices (x = c_probe/c_b1, y = C_build/c_b2)
            x, y = cfg.c_probe / cfg.c_b1, cfg.C_build / cfg.c_b2
            cx = 0 if x < 0.05 else (1 if x < 0.2 else 2)
            cy = 0 if y < 1.0 else (1 if y < 3.0 else 2)
            if (cx, cy) != tuple(gcell) or tuple(cell) != tuple(gcell):
                # rounding at a bin edge is tolerated only within 1e-3 relative
                edge = min(abs(x - e) / e for e in (0.05, 0.2)) < 1e-3 or min(abs(y - e) / e for e in (1.0, 3.0)) < 1e-3
                if not edge:
                    problems.append((split, i, "cell", (cx, cy), gcell))
            if cfg.k != gk or tuple(cfg.flags) != tuple(gcombo):
                problems.append((split, i, "k/flags", cfg.k, gk, cfg.flags, gcombo))
            cells.add(tuple(gcell)); ks.add(cfg.k); combos.add(tuple(cfg.flags))
            key = pw.config_dict(cfg).__repr__()
            if key in seen:
                problems.append((split, i, "duplicate_config_of", seen[key]))
            seen[key] = (split, i)
        rows[split] = {"cells": sorted(cells), "ks": sorted(ks),
                       "combos": sorted("+".join(nm for nm, f in zip(pw.FLAG_NAMES, c) if f) or "none" for c in combos)}
    tr = rows["train"]
    checks = {
        "train_excludes_held_cell": tuple(held_cell) not in map(tuple, tr["cells"]),
        "train_excludes_k4": held_k not in tr["ks"],
        "train_excludes_held_combos": not any(c in {"unreliable+events", "side_effect+correlated"} for c in tr["combos"]),
        "heldout_price_only_centre_cell": rows["heldout_price"]["cells"] == [held_cell],
        "heldout_k_only_k4": rows["heldout_k"]["ks"] == [4],
        "heldout_comp_only_held_combos": set(rows["heldout_comp"]["combos"]) <= {"unreliable+events", "side_effect+correlated"},
        "heldout_price_k_and_combos_train_like": set(rows["heldout_price"]["ks"]) <= {1, 2, 8}
        and not (set(rows["heldout_price"]["combos"]) & {"unreliable+events", "side_effect+correlated"}),
        "heldout_k_cells_combos_train_like": held_cell not in map(tuple, rows["heldout_k"]["cells"])
        and not (set(rows["heldout_k"]["combos"]) & {"unreliable+events", "side_effect+correlated"}),
        "heldout_comp_cells_k_train_like": held_cell not in map(tuple, rows["heldout_comp"]["cells"])
        and 4 not in rows["heldout_comp"]["ks"],
        "each_train_flag_seen_singly": all(nm in tr["combos"] for nm in pw.FLAG_NAMES),
    }
    # world seeds: eval = base + 5e7 + 1000 i + 500 + r (r < 4); train = 8e9 + 1e8 s + n (n < 256000)
    ev_ranges = []
    for sp in ("dev", "test_iid", "heldout_price", "heldout_k", "heldout_comp"):
        ev_ranges.append((pw.world_seed(sp, 0, 500), pw.world_seed(sp, 127, 503)))
    tr_ranges = [(8_000_000_000 + s * 100_000_000, 8_000_000_000 + s * 100_000_000 + 255_999) for s in range(3)]
    allr = sorted(ev_ranges + tr_ranges)
    checks["world_seed_ranges_disjoint"] = all(a[1] < b[0] for a, b in zip(allr, allr[1:]))
    checks["config_seeds_disjoint_from_world_seeds"] = all(
        not (lo <= pw.SPLITS[sp][3] + 383 and pw.SPLITS[sp][3] <= hi) for sp in pw.SPLITS for lo, hi in allr)
    return {"per_split": rows, "checks": checks, "problems": problems[:20], "n_problems": len(problems),
            "eval_world_seed_ranges": ev_ranges, "train_world_seed_ranges": tr_ranges}


# ----------------------------------------------------------------------------------------- 2. brute force

class Brute:
    """Expectimax over raw visible histories (single query, k = 1).  Auditor's own rules from probeworld.md."""

    def __init__(self, cfg, memo=False):
        assert cfg.k == 1
        self.c = cfg
        # memo=False: pure expectimax over raw histories.  memo=True (needed for event configs, where raw
        # histories explode): memoize on the auditor's own replay summary (posterior from likelihood
        # enumeration, rounded to 10 digits, + public facts) -- still independent of pw.advance/update_belief.
        self.memo = {} if memo else None
        p = list(cfg.prior)
        if cfg.corr > 0:  # k = 1: mixture over z of the shifted prior
            mix = [0.0] * 4
            for sg in (1, -1):
                ph = min(max(p[H] + sg * cfg.corr, 0.0), 1.0)
                rest = 1 - p[H]
                q = [ph] + [(1 - ph) * p[t] / rest for t in (M, F, X)]
                for t in range(4):
                    mix[t] += 0.5 * q[t]
            p = mix
        self.prior = p
        self.nodes = 0

    def odist(self, a, th, reduced):
        c = self.c
        if a == PROBE:
            return {"solved": 1.0} if th == H else ({"failed": c.q, "solved": 1 - c.q} if c.q < 1 else {"failed": 1.0})
        if a == B1:
            return {"solved": 1.0} if (th in (H, M) or (th == F and reduced)) else {"timeout": 1.0}
        if a in (B2, USE):
            return {"infeasible": 1.0} if th == X else {"solved": 1.0}
        if a == INSPECT:
            e = th in (H, M)
            return {"easy": 1 - c.eta, "hard": c.eta} if e else {"hard": 1 - c.eta, "easy": c.eta}
        if a == PROP:
            return {"conflict": c.p_conflict, "reduced": 1 - c.p_conflict} if th == X else {"reduced": 1.0}
        if a == BUILD:
            return {"built": 1.0}

    def replay(self, hist):
        """Public facts from the raw history: usage in current epoch, inspect used, built, reduced, candidate,
        event fired; plus the likelihood of the outcomes under each theta."""
        usage, insp, built, reduced, cand, cvalid, ev = set(), False, False, False, None, False, False
        lik = list(self.prior)
        for a, o, e in hist:
            if a != BUILD:
                for t in range(4):
                    lik[t] *= self.odist(a, t, reduced).get(o, 0.0)
            if a == BUILD:
                built = True
            elif a == INSPECT:
                insp = True
            else:
                usage.add(a)
            if a in (B1, B2, USE) and o == "solved":
                cand, cvalid = "exact", True
            elif a == PROBE and o == "solved" and not (cand == "exact" and cvalid):
                cand, cvalid = "probe", True
            if a == PROP and o == "reduced":
                reduced = True
            if e:
                cvalid, reduced, ev = False, False, True
                usage = set()
        z = sum(lik)
        post = [x / z for x in lik]
        return dict(usage=usage, insp=insp, built=built, reduced=reduced, cand=cand, cvalid=cvalid, ev=ev, post=post)

    def avail(self, s):
        u = s["usage"]
        acts = []
        if PROBE not in u: acts.append(PROBE)
        if B1 not in u and B2 not in u: acts.append(B1)
        if B2 not in u: acts.append(B2)
        if not s["insp"]: acts.append(INSPECT)
        if PROP not in u: acts.append(PROP)
        if not s["built"]: acts.append(BUILD)
        if s["built"] and USE not in u: acts.append(USE)
        if s["cand"] is not None: acts.append(COMMIT)
        acts += [CINF, ABSTAIN]
        return acts

    def cost(self, a):
        c = self.c
        return {PROBE: c.c_probe, B1: c.c_b1, B2: c.c_b2, INSPECT: c.c_inspect, PROP: c.c_prop,
                BUILD: c.C_build, USE: c.C_execute + c.C_return + c.C_verify}.get(a, 0.0)

    def Q(self, hist, a, s=None):
        c = self.c
        s = s or self.replay(hist)
        post = s["post"]
        if a == ABSTAIN:
            return 0.0
        if a == CINF:
            return post[X] * c.R - (1 - post[X]) * c.L
        if a == COMMIT:
            if not s["cvalid"]:
                return -c.L
            pc = 1.0 if s["cand"] == "exact" else post[H]
            return pc * c.R - (1 - pc) * c.L
        val = -self.cost(a)
        if a == PROBE and c.D_side > 0:
            val -= c.D_side * (1 - post[H])
        pe = c.p_event if (c.p_event > 0 and not s["ev"]) else 0.0
        po = {}
        for t in range(4):
            if post[t] > 0:
                for o, p in self.odist(a, t, s["reduced"]).items():
                    if p > 0:
                        po[o] = po.get(o, 0.0) + post[t] * p
        for o, p in po.items():
            for e, pv in ((False, 1 - pe), (True, pe)):
                if pv > 0:
                    val += p * pv * self.V(hist + ((a, o, e),))
        return val

    def V(self, hist):
        s = self.replay(hist)
        if self.memo is not None:
            key = (tuple(round(x, 10) for x in s["post"]), frozenset(s["usage"]), s["insp"], s["built"],
                   s["reduced"], s["cand"], s["cvalid"], s["ev"])
            v = self.memo.get(key)
            if v is not None:
                return v
        self.nodes += 1
        v = max(self.Q(hist, a, s) for a in self.avail(s))
        if self.memo is not None:
            self.memo[key] = v
        return v


OUT_CODE = {"solved": pw.O_SOLVED, "failed": pw.O_FAILED, "timeout": pw.O_TIMEOUT, "infeasible": pw.O_INFEASIBLE,
            "easy": pw.O_EASY, "hard": pw.O_HARD, "conflict": pw.O_CONFLICT, "reduced": pw.O_REDUCED,
            "built": pw.O_BUILT}


def brute_force_checks(n_cfg, time_budget):
    out = []
    t0 = time.process_time()
    for split in ("heldout_price", "heldout_comp"):
        found = 0
        for i in range(128):
            cfg = pw.split_config(split, i)
            if cfg.k != 1:
                continue
            if time.process_time() - t0 > time_budget:
                break
            b = Brute(cfg, memo=cfg.p_event > 0)
            s = pw.ExactSolver(cfg)
            st0 = pw.initial_state(cfg)
            q = s.q_values(st0)
            t1 = time.process_time()
            maxerr = 0.0
            cmp = []
            for a in q:
                qb = b.Q((), a)
                maxerr = max(maxerr, abs(qb - q[a]))
                cmp.append((pw.ACTIONS[a], q[a], qb))
            vb = b.V(())
            # depth-1/2 histories along the modal outcome of the two cheapest informative actions
            deep = []
            for a1 in (PROBE, INSPECT, PROP):
                for o1, _p in [(o, p) for o, p in b.odist(a1, M, False).items()] + [(o, p) for o, p in b.odist(a1, H, False).items()]:
                    h1 = ((a1, o1, False),)
                    try:
                        st1 = pw.public_state_from_history(cfg, [(a1, OUT_CODE[o1], False, None)])
                    except ValueError:
                        continue
                    bs1 = b.replay(h1)
                    if sum(bs1["post"]) == 0:
                        continue
                    v1b = b.V(h1)
                    v1 = s.value(st1)
                    deep.append((pw.ACTIONS[a1], o1, v1, v1b))
                    maxerr = max(maxerr, abs(v1 - v1b))
            vs = s.value(st0)
            maxerr = max(maxerr, abs(vb - vs))
            out.append({"split": split, "idx": i, "memoized_brute": b.memo is not None, "flags": [nm for nm, f in zip(pw.FLAG_NAMES, cfg.flags) if f],
                        "V_dp": vs, "V_brute": vb, "Q_root": cmp, "deep": deep, "max_abs_err": maxerr,
                        "brute_nodes": b.nodes, "cpu_s": time.process_time() - t1})
            found += 1
            if found >= n_cfg:
                break
    return out


# ----------------------------------------------------------------------------------------- 3. Monte Carlo

def mc_check(split, idx, rep, depth, n_acc, max_tries, rng):
    cfg = pw.split_config(split, idx)
    s = pw.ExactSolver(cfg)
    # the visible history: pi* on the eval world, truncated at `depth` steps
    ep = pw.Episode(cfg, pw.world_seed(split, idx, 500 + rep))
    while not ep.done and len(ep.history) < depth:
        ep.step(s.pi_star(ep.state))
    if ep.done:
        return None
    hist = list(ep.history)
    st = ep.state
    assert st == pw.public_state_from_history(cfg, hist)
    q = s.q_values(st)
    # sample hidden worlds consistent with hist: fresh simulator worlds, replay the actions, reject mismatches.
    # Each accepted world then continues with its own (post-conditioning) RNG stream.
    acc = []
    tries = 0
    while len(acc) < n_acc and tries < max_tries:
        tries += 1
        seed = rng.randrange(10 ** 15)
        e2 = pw.Episode(cfg, seed)
        ok = True
        for (a, o, ev, rv) in hist:
            r = e2.step(a)
            if r != (a, o, ev, rv):
                ok = False
                break
        if ok:
            acc.append(seed)
    if len(acc) < 50:
        return {"split": split, "idx": idx, "depth": depth, "accepted": len(acc), "tries": tries, "skipped": True}
    res = {}
    for a in q:
        vals = []
        for seed in acc:
            e2 = pw.Episode(cfg, seed)
            for (a0, _, _, _) in hist:
                e2.step(a0)
            u0 = e2.utility
            e2.step(a)
            while not e2.done:
                e2.step(s.pi_star(e2.state))
            vals.append(e2.utility - u0)
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))
        se = sd / math.sqrt(len(vals))
        z = (m - q[a]) / se if se > 1e-9 else (0.0 if abs(m - q[a]) < 1e-6 else float("inf"))
        res[pw.ACTIONS[a]] = {"Q_star": q[a], "mc_mean": m, "se": se, "z": z}
    cfg_flags = [nm for nm, f in zip(pw.FLAG_NAMES, cfg.flags) if f]
    return {"split": split, "idx": idx, "k": cfg.k, "flags": cfg_flags, "depth": depth, "query": st[0][0],
            "history": [(pw.ACTIONS[a], pw.OUTCOMES[o], ev, rv) for a, o, ev, rv in hist],
            "accepted": len(acc), "tries": tries, "V_star": s.value(st), "actions": res,
            "max_abs_z": max(abs(v["z"]) for v in res.values())}


def belief_check(split, idx, rep, depth, n_acc, rng):
    """Large-sample follow-up: empirical posterior of the current query's hidden theta among simulator worlds
    consistent with the visible history vs the DP belief, and MC of Q*(commit_infeasible) (terminal, so it is
    exactly P(X) R - (1 - P(X)) L + E[V* of the next query])."""
    cfg = pw.split_config(split, idx)
    s = pw.ExactSolver(cfg)
    ep = pw.Episode(cfg, pw.world_seed(split, idx, 500 + rep))
    while not ep.done and len(ep.history) < depth:
        ep.step(s.pi_star(ep.state))
    hist, st = list(ep.history), ep.state
    counts = [0, 0, 0, 0]
    vals = []
    tries = 0
    while sum(counts) < n_acc:
        tries += 1
        e2 = pw.Episode(cfg, rng.randrange(10 ** 15))
        if all(e2.step(a) == (a, o, ev, rv) for a, o, ev, rv in hist):
            counts[e2.thetas[e2.query]] += 1
            if pw.A_COMMIT_INF in e2.available():
                u0 = e2.utility
                e2.step(pw.A_COMMIT_INF)
                while not e2.done:
                    e2.step(s.pi_star(e2.state))
                vals.append(e2.utility - u0)
    n = sum(counts)
    emp = [c / n for c in counts]
    b = s.belief(st)
    se = [math.sqrt(max(p * (1 - p), 1e-12) / n) for p in b]
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    q = s.q_values(st).get(pw.A_COMMIT_INF)
    return {"split": split, "idx": idx, "depth": depth, "accepted": n, "tries": tries, "dp_belief": b,
            "empirical_belief": emp, "z_belief": [(e - p) / s_ for e, p, s_ in zip(emp, b, se)],
            "Q_commit_inf": q, "mc_commit_inf": m, "z_commit_inf": (m - q) / (sd / math.sqrt(len(vals)))}


# ----------------------------------------------------------------------------------------- 4. visible-history

def history_function_check(n_cfg, eps_per_cfg, rng):
    n_hist = n_conflict = n_state_mismatch = 0
    for split in ("heldout_price", "heldout_k", "heldout_comp"):
        for i in range(n_cfg):
            cfg = pw.split_config(split, i)
            s = pw.ExactSolver(cfg)
            table = {}
            for w in range(eps_per_cfg):
                ep = pw.Episode(cfg, rng.randrange(10 ** 12))
                while not ep.done:
                    h = tuple(ep.history)
                    if ep.state != pw.public_state_from_history(cfg, h):
                        n_state_mismatch += 1
                    lab = (round(s.value(ep.state), 9), tuple(sorted((a, round(v, 9)) for a, v in s.q_values(ep.state).items())),
                           tuple(sorted(s.opt_set(ep.state))))
                    if h in table and table[h] != lab:
                        n_conflict += 1
                    table[h] = lab
                    ep.step(rng.choice(ep.available()))
            n_hist += len(table)
    return {"distinct_histories": n_hist, "label_conflicts": n_conflict, "state_mismatches": n_state_mismatch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/campaigns/extended-04/review/b1-audit-labels.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.process_time()
    rng = random.Random(20260926)
    out = {"splits": audit_splits()}
    print("splits", json.dumps(out["splits"]["checks"]), out["splits"]["n_problems"], flush=True)
    out["brute_force"] = brute_force_checks(n_cfg=2 if a.quick else 4, time_budget=120 if a.quick else 600)
    for r in out["brute_force"]:
        print("brute", r["split"], r["idx"], r["flags"], round(r["V_dp"], 6), round(r["V_brute"], 6),
              f"maxerr={r['max_abs_err']:.2e} nodes={r['brute_nodes']} cpu={r['cpu_s']:.1f}", flush=True)
    mc = []
    cases = [("heldout_k", i, 0, d) for i in (0, 1, 2, 3, 5) for d in (2, 5)] + \
            [("heldout_comp", i, 1, d) for i in (0, 1, 2, 3, 4, 6) for d in (1, 3)] + \
            [("heldout_price", i, 2, d) for i in (0, 1, 2) for d in (1, 3)]
    if a.quick:
        cases = cases[:4]
    for sp, i, rep, d in cases:
        r = mc_check(sp, i, rep, d, n_acc=400, max_tries=40000, rng=rng)
        if r is None:
            continue
        mc.append(r)
        print("mc", sp, i, d, r.get("flags"), r.get("history"), "acc", r["accepted"], "max|z|",
              round(r.get("max_abs_z", float("nan")), 2), flush=True)
    out["monte_carlo"] = mc
    zs = [v["z"] for r in mc if not r.get("skipped") for v in r["actions"].values()]
    out["monte_carlo_summary"] = {"n_histories": sum(1 for r in mc if not r.get("skipped")), "n_Q_tests": len(zs),
                                  "max_abs_z": max(abs(z) for z in zs) if zs else None,
                                  "n_abs_z_gt_3": sum(abs(z) > 3 for z in zs),
                                  "n_abs_z_gt_4": sum(abs(z) > 4 for z in zs),
                                  "mean_z": sum(zs) / len(zs) if zs else None,
                                  "mean_z2": sum(z * z for z in zs) / len(zs) if zs else None}
    print("mc summary", out["monte_carlo_summary"], flush=True)
    # follow-up on the largest |z| cells (all commit_infeasible): 10x larger samples + direct belief check
    out["belief_followup"] = []
    for sp, i, rep, d in (("heldout_k", 1, 0, 5), ("heldout_comp", 0, 1, 3), ("heldout_price", 0, 2, 3)):
        r = belief_check(sp, i, rep, d, n_acc=4000 if a.quick else 20000, rng=rng)
        out["belief_followup"].append(r)
        print("belief", sp, i, d, [round(x, 4) for x in r["dp_belief"]], [round(x, 4) for x in r["empirical_belief"]],
              "z", [round(x, 2) for x in r["z_belief"]], "Qcinf", round(r["Q_commit_inf"], 3),
              round(r["mc_commit_inf"], 3), "z", round(r["z_commit_inf"], 2), flush=True)
    out["history_function"] = history_function_check(n_cfg=4 if a.quick else 8, eps_per_cfg=60, rng=rng)
    print("history", out["history_function"], flush=True)
    out["local_cpu_s"] = time.process_time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print("cpu", out["local_cpu_s"])


if __name__ == "__main__":
    main()
