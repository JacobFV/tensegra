#!/usr/bin/env python3
"""Independent audit (Track B): split v2 / b5c integrity, own exact DP, episode re-simulation, sizing rule.

Pure Python.  Uses only the ENVIRONMENT spec from campaign04_probeworld (Config, make_config, outcome_dist, advance,
available, commit_correct, Episode) -- not ExactSolver, not campaign05_probeworld, not the Track B scorers.  The
registered split table (seed bases, cells, k, combos) is hard-coded here from trackb-tooling.md / registry.json.
Inputs: pools.json (configs actually stored in the label pickles, dumped by campaign05_b_audit_pools.py), the local
copies of run directories (episode rows) and the hold s0 label files.
Usage: campaign05_b_audit_split.py POOLS_JSON DATA_DIR OUT_JSON [--full-b5c]
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
import random
import sys
import time

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(  # load the module file directly (the package __init__ imports torch)
    "campaign04_probeworld", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "tensegra",
                                          "campaign04_probeworld.py"))
pw = importlib.util.module_from_spec(_spec)  # environment spec only
sys.modules["campaign04_probeworld"] = pw
_spec.loader.exec_module(pw)

EPS = 0.5
N, U_, S_, C_, E_ = (False,) * 4, (True, False, False, False), (False, True, False, False), \
    (False, False, True, False), (False, False, False, True)


def cmb(*fs):
    return tuple(any(f[j] for f in fs) for j in range(4))


UC, SE, UE, SC, US, CE = cmb(U_, C_), cmb(S_, E_), cmb(U_, E_), cmb(S_, C_), cmb(U_, S_), cmb(C_, E_)
TRAIN7 = (N, U_, S_, C_, E_, US, CE)
TRAIN_CELLS = tuple((x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1))
REG = {  # registered split table v2 + b5c (trackb-tooling.md section 2 / 9.1)
    "b5_train": (TRAIN_CELLS, (1, 2, 8), TRAIN7, 5_100_000_000),
    "b5_dev": (TRAIN_CELLS, (1, 2, 8), TRAIN7, 5_200_000_000),
    "b5_test_iid": (TRAIN_CELLS, (1, 2, 8), TRAIN7, 5_300_000_000),
    "b5_heldout_price": (((1, 1),), (1, 2, 8), TRAIN7, 5_400_000_000),
    "b5_heldout_k": (TRAIN_CELLS, (4,), TRAIN7, 5_500_000_000),
    "b5_hold_uc": (TRAIN_CELLS, (1, 2, 8), (UC,), 5_600_000_000),
    "b5_hold_se": (TRAIN_CELLS, (1, 2, 8), (SE,), 5_700_000_000),
    "b5x_train": (TRAIN_CELLS, (1, 2, 8), TRAIN7 + (UE, SC), 5_800_000_000),
    "b5c_hold_uc": (TRAIN_CELLS, (1, 2, 8), (UC,), 5_900_000_000),
}
TRAINING_OR_SELECTION = ("b5_train", "b5x_train", "b5_dev", "b5_test_iid", "b5_heldout_price", "b5_heldout_k")


def regen(split, i):
    cells, ks, combos, base = REG[split]
    rng = random.Random(base + i)
    cell, k, c = rng.choice(cells), rng.choice(ks), rng.choice(combos)
    return pw.make_config(cell, k, c, random.Random((base + i) * 7 + 1))


def cfg_key(cd):
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(cd.items()))


# ------------------------------------------------------------------ own exact DP (independent of ExactSolver)
class DP:
    def __init__(self, cfg):
        self.cfg = cfg
        self.V = {}
        self.Q = {}

    def branches(self, st, a):
        cfg = self.cfg
        (i, built, nH, nN), (b, usage, reduced, cand, cvalid, ev) = st
        if a in pw.TERMINAL:
            out = []
            for t in range(4):
                if b[t] > 0:
                    if a == pw.A_ABSTAIN:
                        o, r = pw.O_ABSTAINED, 0.0
                    elif pw.commit_correct(a, t, cand, cvalid):
                        o, r = pw.O_CORRECT, cfg.R
                    else:
                        o, r = pw.O_WRONG, -cfg.L
                    out.append((b[t], r, pw.advance(cfg, st, a, o, False, t)))
            return out
        cost = {pw.A_PROBE: cfg.c_probe, pw.A_B1: cfg.c_b1, pw.A_B2: cfg.c_b2, pw.A_INSPECT: cfg.c_inspect,
                pw.A_PROP: cfg.c_prop, pw.A_BUILD: cfg.C_build,
                pw.A_USE: cfg.C_execute + cfg.C_return + cfg.C_verify}[a]
        if a == pw.A_PROBE and cfg.D_side > 0:
            cost += cfg.D_side * (1.0 - b[0])  # charged iff theta != H; expectation under the public belief
        po = {}
        for t in range(4):
            if b[t] > 0:
                for p, o in pw.outcome_dist(cfg, a, t, reduced):
                    if p > 0:
                        po[o] = po.get(o, 0.0) + b[t] * p
        pe = cfg.p_event if (cfg.p_event > 0 and not ev) else 0.0
        out = []
        for o, p in po.items():
            for e, pr in ((False, 1 - pe), (True, pe)):
                if pr > 0:
                    out.append((p * pr, -cost, pw.advance(cfg, st, a, o, e, None)))
        return out

    def q(self, st):
        q = self.Q.get(st)
        if q is None:
            q = {a: sum(p * (r + self.v(ns)) for p, r, ns in self.branches(st, a)) for a in pw.available(self.cfg, st)}
            self.Q[st] = q
        return q

    def v(self, st):
        x = self.V.get(st)
        if x is None:
            x = 0.0 if st[0][0] >= self.cfg.k else max(self.q(st).values())
            self.V[st] = x
        return x


def s0_facts(cfg):
    d = DP(cfg)
    st = pw.initial_state(cfg)
    q = d.q(st)
    v = max(q.values())
    opt = sorted(a for a, x in q.items() if x >= v - EPS)
    out = {"V": v, "opt": opt, "probe_unique": opt == [pw.A_PROBE]}
    if cfg.corr > 0:
        c0 = dataclasses.replace(cfg, corr=0.0)
        v0 = DP(c0).v(pw.initial_state(c0))
        out["V_without_correlated"] = v0
        out["flag_sensitive"] = abs(v - v0) > 1e-6
    return out, d


def main(pools_path, data, out, full_b5c):
    t0 = time.process_time()
    pools = json.load(open(pools_path))
    res = {"tool": "campaign05_b_audit_split", "checks": {}}
    ck = res["checks"]
    by_split = {}
    for path, ents in pools.items():
        name = os.path.basename(path).split(".")[0]
        tag = ("e04:" if "campaign04" in path else "") + name
        by_split.setdefault(tag, []).extend(ents)
    for v in by_split.values():
        v.sort(key=lambda e: e["idx"])
    # 1. stored pools == registered generator (every configuration)
    regen_ok = {}
    for sp in REG:
        if sp not in by_split:
            continue
        ents = by_split[sp]
        regen_ok[sp] = {"n": len(ents), "idx_contiguous": [e["idx"] for e in ents] == list(range(len(ents))),
                        "all_equal_regenerated": all(
                            cfg_key(e["cfg"]) == cfg_key(json.loads(json.dumps(dataclasses.asdict(regen(sp, e["idx"])))))
                            for e in ents)}
    ck["pools_equal_registered_generator"] = regen_ok
    # 2. new holds absent from every training / selection / eval-family pool (by the stored configs' flags)
    flags_count = {}
    for sp, ents in by_split.items():
        c = {}
        for e in ents:
            key = "+".join(n for n, f in zip(pw.FLAG_NAMES, e["flags"]) if f) or "none"
            c[key] = c.get(key, 0) + 1
        flags_count[sp] = c
    ck["flag_counts_per_pool"] = flags_count
    ck["new_holds_absent_from_training_selection"] = {
        sp: {"UC": flags_count[sp].get("unreliable+correlated", 0), "SE": flags_count[sp].get("side_effect+events", 0)}
        for sp in TRAINING_OR_SELECTION}
    ck["historical_pairs_only_in_b5x"] = {
        sp: {"UE": flags_count[sp].get("unreliable+events", 0), "SC": flags_count[sp].get("side_effect+correlated", 0)}
        for sp in TRAINING_OR_SELECTION}
    # 3. b5c freshness: no b5c configuration equals any earlier configuration (stored pools + first 4,000 of the
    #    b5 hold streams + extended-04 pools); config seeds and eval world seeds disjoint
    earlier = set()
    for sp, ents in by_split.items():
        if sp != "b5c_hold_uc":
            earlier.update(cfg_key(e["cfg"]) for e in ents)
    for sp in ("b5_hold_uc", "b5_hold_se"):
        for i in range(4000):
            earlier.add(cfg_key(json.loads(json.dumps(dataclasses.asdict(regen(sp, i))))))
    b5c = [cfg_key(e["cfg"]) for e in by_split["b5c_hold_uc"]]
    b5c_prices = {tuple(dict(k)[f] for f in ("c_probe", "c_b1", "c_b2", "C_build", "L")) for k in b5c}
    earlier_prices = {tuple(dict(k)[f] for f in ("c_probe", "c_b1", "c_b2", "C_build", "L")) for k in earlier}
    ck["b5c_fresh"] = {"n_b5c": len(b5c), "n_earlier_compared": len(earlier),
                       "exact_config_collisions": sum(k in earlier for k in b5c),
                       "price_tuple_collisions": len(b5c_prices & earlier_prices),
                       "b5c_unique": len(set(b5c)) == len(b5c),
                       "config_seed_range": [5_900_000_000, 5_900_000_000 + len(b5c) - 1],
                       "config_seed_overlap_other_bases": any(
                           b <= 5_900_000_000 + len(b5c) - 1 and 5_900_000_000 <= b + 3999
                           for s, (_, _, _, b) in REG.items() if s != "b5c_hold_uc")}
    # 4. episode rows: re-simulate with the world seed and recorded actions; own DP labels on the public state
    #    rebuilt from the visible history only
    sims = {"episodes": 0, "decisions": 0, "rec_mismatch": 0, "U_mismatch": 0, "Q_maxabs": 0.0, "opt_mismatch": 0,
            "V_star_maxabs": 0.0, "hidden_info_used": False}
    rng = random.Random(7)
    samples = []
    for run, sp, n in (("bx-xc-b0-s20", "b5c_hold_uc", 20), ("bx-xc-bx1-s21", "b5c_hold_uc", 20),
                       ("bx-b0-s10", "b5_hold_uc", 15), ("bx-bx1-s12", "b5_hold_se", 10)):
        rows = [json.loads(x) for x in gzip.open(os.path.join(data, run, "run", "eval_episodes.jsonl.gz"), "rt")]
        base = REG[sp][3]
        rows = [r for r in rows if "decisions" in r and base + 50_000_000 <= r["world_seed"] < base + 50_000_000 + 4_000_000]
        samples += [(sp, r) for r in rng.sample(rows, n)]
    for sp, r in samples:
        i = (r["world_seed"] - REG[sp][3] - 50_000_000) // 1000
        cfg = regen(sp, i)
        ep = pw.Episode(cfg, r["world_seed"])
        dp = DP(cfg)
        hist = []
        for d in r["decisions"]:
            st = pw.public_state_from_history(cfg, hist)  # visible history only
            q = dp.q(st)
            av = sorted(q)
            if av != d["avail"]:
                sims["opt_mismatch"] += 1
            v = max(q.values())
            sims["Q_maxabs"] = max(sims["Q_maxabs"], max(abs(q[a] - x) for a, x in zip(av, d["Q"])))
            opt = sorted(a for a in av if q[a] >= v - EPS)
            sims["opt_mismatch"] += opt != d["opt"]
            rec = ep.step(d["a"])
            sims["rec_mismatch"] += list(rec) != [d["rec"][0], d["rec"][1], d["rec"][2], d["rec"][3]]
            hist.append(tuple(d["rec"]))
            sims["decisions"] += 1
        sims["U_mismatch"] += abs(ep.utility - r["U"]) > 1e-3
        sims["V_star_maxabs"] = max(sims["V_star_maxabs"], abs(dp.v(pw.initial_state(cfg)) - r["V_star"]))
        sims["episodes"] += 1
    ck["resimulation_and_own_dp"] = sims
    # 5. sizing rule: own DP s0 facts vs the label job's s0 file; full prefix recomputation for b5c if asked
    s0 = {x["idx"]: x for x in json.load(open(os.path.join(data, "bxc-labels/labels/b5c_hold_uc_s0.json")))}
    s0uc = {x["idx"]: x for x in json.load(open(os.path.join(data, "bx-labels/labels/b5_hold_uc_s0.json")))}
    s0se = {x["idx"]: x for x in json.load(open(os.path.join(data, "bx-labels/labels/b5_hold_se_s0.json")))}
    if full_b5c:
        idxs = list(range(len(s0)))
    else:
        elig = [i for i, x in s0.items() if x["probe_unique"]]
        idxs = sorted(set(elig) | set(random.Random(3).sample(range(len(s0)), 120)) | {len(s0) - 1})
    mism = {"probe_unique": 0, "flag_sensitive": 0, "V_maxabs": 0.0}
    mine = {}
    for i in idxs:
        f, _ = s0_facts(regen("b5c_hold_uc", i))
        mine[i] = f
        mism["probe_unique"] += f["probe_unique"] != s0[i]["probe_unique"]
        mism["flag_sensitive"] += f["flag_sensitive"] != s0[i]["flag_sensitive"]
        mism["V_maxabs"] = max(mism["V_maxabs"], abs(f["V"] - s0[i]["V"]))
    size = None
    if full_b5c:
        c = cs = 0
        for n in range(len(s0)):
            c += mine[n]["probe_unique"]
            cs += mine[n]["probe_unique"] and mine[n]["flag_sensitive"]
            if c >= 60 and cs >= 20:
                size = n + 1
                break
    # extra spot check of the b5 hold s0 files (every eligible configuration + the last one)
    m2 = 0
    n2 = 0
    for sp, s0x in (("b5_hold_uc", s0uc), ("b5_hold_se", s0se)):
        for i in sorted([i for i, x in s0x.items() if x["probe_unique"]] + [len(s0x) - 1]):
            f, _ = s0_facts(regen(sp, i))
            m2 += f["probe_unique"] != s0x[i]["probe_unique"] or (sp == "b5_hold_uc" and f["flag_sensitive"] != s0x[i]["flag_sensitive"])
            n2 += 1
    ck["sizing_b5c"] = {"n_checked": len(idxs), "full_prefix": full_b5c, "mismatch": mism, "own_prefix_size": size,
                        "label_job_pool": len(s0),
                        "eligible_by_own_dp_in_checked": sum(f["probe_unique"] for f in mine.values()),
                        "flag_sensitive_eligible_by_own_dp": sum(f["probe_unique"] and f["flag_sensitive"] for f in mine.values())}
    ck["sizing_b5_holds_eligible_spotcheck"] = {"n_checked": n2, "mismatch": m2}
    res["local_cpu_s"] = round(time.process_time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["checks"], indent=1)[:6000])
    print("cpu", res["local_cpu_s"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--full-b5c" in sys.argv)
