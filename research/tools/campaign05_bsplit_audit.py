"""Extended-05 B-SPLIT support audit and BX2 triviality assessment (pure Python; run before any B-X model run).

  campaign05_bsplit_audit.py audit --out FILE [--n-pair 32] [--n-hist 8]
  campaign05_bsplit_audit.py triviality --out FILE [--n-hold 40] [--n-train 80] [--worlds 8]
  campaign05_bsplit_audit.py sizing --out FILE [--n-uc 480] [--n-se 160]

audit:
  A. pair definability under the flag semantics (every pair generable; composition interaction of the first decision)
  B. generator-parameter sets: families disjoint; new holds and historical pairs absent from training support
     (historical pairs only in the BX1 exposure pool); every hold flag seen singly in training
  C. generated configurations (full label-pool sizes): recovered (cell, k, flags) = generator parameters; held-out
     pairs never co-occur in any training/selection configuration; no duplicate configuration across splits;
     configuration and world seed ranges disjoint (also from every extended-04 range)
  D. labels (Q*, V*, A*) and the BX2/BO supplied features are functions of the visible history: random-policy
     rollouts, identical visible histories -> identical labels/features; simulator state == public state rebuilt
     from the history
sizing (design v2 revision 11, estimate only; the registered rule is applied by the labels job):
  s0 facts (probe uniquely optimal, registered U+C flag sensitivity) on a prefix of each hold's configuration stream;
  projected pool size for HOLD_TARGET_ELIGIBLE eligible configurations, and projected label CPU.
triviality:
  Does supplying the exact belief (BX2) make the decision trivial?  Over pi*-visited decisions, how often is the
  eps-optimal set determined by (i) bookkeeping alone, (ii) the belief alone, (iii) belief + bookkeeping,
  (iv) everything BX2 supplies except the continuous prices (belief + bookkeeping + flags + remaining horizon) --
  measured as transfer of a majority-action lookup table from training-support decisions to new-hold decisions,
  and as within-key purity on the holds (keys shared by >= 2 configurations).  The remainder requires the
  price/horizon/reliability trade-off.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
try:
    from tensegra import campaign04_probeworld as pw  # noqa: E402
except ImportError:  # the package __init__ imports torch; bypass it for these pure-Python modules
    import types
    for _k in [k for k in sys.modules if k == "tensegra" or k.startswith("tensegra.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("tensegra")
    _pkg.__path__ = [str(REPO / "src" / "tensegra")]
    sys.modules["tensegra"] = _pkg
    from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign05_probeworld as pw5  # noqa: E402

POOL_SIZES = {"b5_train": 384, "b5x_train": 768, "b5_dev": 128, "b5_test_iid": 128, "b5_heldout_price": 128,
              "b5_heldout_k": 128, "b5_hold_uc": pw5.HOLD_MAX_CONFIGS, "b5_hold_se": pw5.HOLD_MAX_CONFIGS}
TRAINING_OR_SELECTION = ("b5_train", "b5_dev", "b5x_train")


def cell_consistent(cfg, cell) -> bool:
    """Public price ratios lie in the generator cell's ranges (prices are rounded to 4 decimals, so a ratio can sit
    up to ~0.1% outside a bin edge; pw.price_cell returns the FIRST matching bin and so can report the neighbour)."""
    (xlo, xhi), (ylo, yhi) = pw.X_BINS[cell[0]], pw.Y_BINS[cell[1]]
    x, y = cfg.c_probe / cfg.c_b1, cfg.C_build / cfg.c_b2
    return xlo * 0.999 <= x <= xhi * 1.001 and ylo * 0.999 <= y <= yhi * 1.001


def check(name, ok, detail=None):
    return {"check": name, "pass": bool(ok), **({"detail": detail} if detail is not None else {})}


def audit(a):
    t0 = time.process_time()
    checks = []
    names = pw5.combo_name
    # ---------------------------------------------------------------- A. pair definability
    pairs = {}
    for cmb in pw5.ALL_PAIRS:
        rng = random.Random(7_000_000 + sum(1 << j for j, f in enumerate(cmb) if f))
        rows = []
        for n in range(a.n_pair):
            cell = rng.choice(pw.TRAIN_CELLS)
            k = 2  # k = 2 keeps the audit DP cheap and lets the correlated flag act (it is inert at k = 1)
            cfg = pw.make_config(cell, k, cmb, random.Random(rng.randrange(1 << 30)))
            assert cfg.flags == cmb
            rows.append(pw5.composition_interaction(cfg))
        role = ("new_hold" if cmb in pw5.NEW_HOLD_COMBOS else "historical_challenge" if cmb in pw5.HIST_COMBOS
                else "training_pair")
        pairs[names(cmb)] = {"role": role, "generable": True, "configs": len(rows),
                             "interaction_rate": sum(r["interaction"] for r in rows) / len(rows),
                             "first_decision_probe_unique_rate": sum(r["probe_unique"] for r in rows) / len(rows),
                             "first_decision_differs_from_ablation": {
                                 f: sum(r["ablations"][f] != r["opt"] for r in rows) / len(rows)
                                 for f in rows[0]["ablations"]},
                             "naive_policy_regret_mean": {
                                 f: sum(r["naive_regret"][f] for r in rows) / len(rows) for f in rows[0]["naive_regret"]},
                             "naive_policy_regret_gt_eps_frac": {
                                 f: sum(r["naive_regret"][f] > pw.EPS for r in rows) / len(rows)
                                 for f in rows[0]["naive_regret"]},
                             "naive_policy_regret_gt_eps_both_frac": sum(
                                 all(v > pw.EPS for v in r["naive_regret"].values()) for r in rows) / len(rows),
                             "V_star_mean": sum(r["V_star"] for r in rows) / len(rows)}
    checks.append(check("A1 all 6 pairs generable (flags sampled independently by make_config)",
                        len(pairs) == 6 and all(p["generable"] for p in pairs.values())))
    checks.append(check("A2 new-hold pairs require both conditions: acting optimally for either single-flag ablation "
                        "loses > eps on a non-zero fraction of pair configurations",
                        all(pairs[names(c)]["naive_policy_regret_gt_eps_both_frac"] > 0 for c in pw5.NEW_HOLD_COMBOS),
                        {names(c): pairs[names(c)]["naive_policy_regret_gt_eps_both_frac"] for c in pw5.NEW_HOLD_COMBOS}))
    # ---------------------------------------------------------------- B. generator-parameter sets
    b5 = list(pw5.SPLITS5)
    fams = pw5.family_params([s for s in b5 if s != "b5x_train"])
    disjoint = all(not (fams[x] & fams[y]) for x, y in itertools.combinations(fams, 2))
    checks.append(check("B1 b5 split families (excluding the BX1 exposure pool) disjoint by generator parameters",
                        disjoint, sorted(fams)))
    train_combos = {s: set(pw5.SPLITS5[s][2]) for s in TRAINING_OR_SELECTION + ("b5_test_iid",)}
    checks.append(check("B2 new holds absent from every training/selection combo set",
                        all(not (set(pw5.NEW_HOLD_COMBOS) & c) for c in train_combos.values())))
    checks.append(check("B3 historical pairs absent from b5_train/b5_dev/b5_test_iid; present only in b5x_train",
                        all(not (set(pw5.HIST_COMBOS) & train_combos[s]) for s in ("b5_train", "b5_dev", "b5_test_iid"))
                        and set(pw5.HIST_COMBOS) <= train_combos["b5x_train"]))
    singles_ok = all(tuple(jj == j for jj in range(4)) in pw5.SPLITS5["b5_train"][2]
                     for c in pw5.NEW_HOLD_COMBOS + pw5.HIST_COMBOS for j, f in enumerate(c) if f)
    checks.append(check("B4 every flag of every held-out pair is seen singly in b5_train", singles_ok))
    hold_params = pw5.family_params(pw5.NEW_HOLD_SPLITS)
    x_params = pw5.family_params(["b5x_train"])["b5x_train"]
    checks.append(check("B5 b5x_train (BX1 exposure) disjoint from the new holds by generator parameters",
                        all(not (v & x_params) for v in hold_params.values())))
    checks.append(check("B6 training support = none + 4 singles + U+S + C+E (7 combos); exposure adds U+E, S+C",
                        [names(c) for c in pw5.B5_TRAIN_COMBOS] == ["none", "unreliable", "side_effect", "correlated",
                                                                   "events", "unreliable+side_effect",
                                                                   "correlated+events"]
                        and len(pw5.B5X_TRAIN_COMBOS) == 9, [names(c) for c in pw5.B5X_TRAIN_COMBOS]))
    # ---------------------------------------------------------------- C. generated configurations
    configs, combos_in, problems, boundary = {}, {}, [], []
    for split, n in POOL_SIZES.items():
        cnt = {}
        for idx in range(n):
            cell, k, cmb, seed = pw5.generator_params(split, idx)
            cfg = pw5.split_config(split, idx)
            if not cell_consistent(cfg, cell) or cfg.k != k or cfg.flags != tuple(cmb):
                problems.append((split, idx))
            if pw.price_cell(cfg) != tuple(cell):
                boundary.append((split, idx))
            configs[(split, idx)] = cfg
            cnt[names(cfg.flags)] = cnt.get(names(cfg.flags), 0) + 1
        combos_in[split] = dict(sorted(cnt.items()))
    checks.append(check("C1 generated configs land in their generator parameters (cell ranges / k / flags checked on "
                        "the public prices; +-0.1% rounding tolerance)", not problems,
                        {"problems": problems[:5], "rounding_boundary_configs": boundary}))
    hold_names = {names(c) for c in pw5.NEW_HOLD_COMBOS}
    co = {s: [c for c in combos_in[s] if c in hold_names] for s in TRAINING_OR_SELECTION + ("b5_test_iid",)}
    checks.append(check("C2 held-out pairs never co-occur in any training/selection configuration",
                        not any(co.values()), co))
    hist_names = {names(c) for c in pw5.HIST_COMBOS}
    leaks = [(s, i) for s in TRAINING_OR_SELECTION + ("b5_test_iid", "b5_heldout_price", "b5_heldout_k")
             for i in range(POOL_SIZES[s])
             if pw5.split_of_params(*pw5.generator_params(s, i)[:3]) & set(pw5.NEW_HOLD_SPLITS)]
    checks.append(check("C2b split_of_params (table v2): no training/dev/test/price/k configuration lies in a new-hold "
                        "family", not leaks, leaks[:5]))
    checks.append(check("C3 historical pairs occur in b5x_train only",
                        all(not [c for c in combos_in[s] if c in hist_names] for s in ("b5_train", "b5_dev", "b5_test_iid"))
                        and hist_names <= set(combos_in["b5x_train"])))
    seen, dup = {}, []
    for key, cfg in configs.items():
        if cfg in seen:
            dup.append((key, seen[cfg]))
        seen[cfg] = key
    # also against the extended-04 pools the b5 evaluation reuses / the B1 models were trained on
    for split, n in (("train", 384), ("heldout_comp", 128), ("dev", 128)):
        for idx in range(n):
            cfg = pw.split_config(split, idx)
            if cfg in seen:
                dup.append(((split, idx), seen[cfg]))
    checks.append(check("C4 no duplicate configuration across b5 splits (and vs extended-04 train/dev/heldout_comp)",
                        not dup, dup[:5]))
    cseeds = [pw5.generator_params(s, i)[3] for s, n in POOL_SIZES.items() for i in range(n)]
    old = [(b, b + 10_000) for (_, _, _, b) in pw.SPLITS.values()]
    checks.append(check("C5 configuration seeds unique and disjoint from extended-04 seed ranges",
                        len(cseeds) == len(set(cseeds)) and not any(lo <= x < hi for x in cseeds for lo, hi in old)))
    ws = [pw5.world_seed(s, i, r) for s, n in POOL_SIZES.items() if s not in TRAINING_OR_SELECTION
          for i in range(n) for r in (0, 3, 500, 503, 700, 703, 900, 903)]
    ws_old = {pw.world_seed(s, i, r) for s in pw.SPLITS for i in range(128) for r in (500, 503, 700, 703, 900, 903)}
    train_band = (8_000_000_000, 8_000_000_000 + 100_000_000 * 64)
    checks.append(check("C6 eval world seeds unique, disjoint from extended-04 eval worlds and the training band",
                        len(ws) == len(set(ws)) and not (set(ws) & ws_old)
                        and not any(train_band[0] <= x < train_band[1] for x in ws)))
    # ---------------------------------------------------------------- D. labels / features function of visible info
    n_hist, conflicts, state_mismatch, feat_conflicts = 0, 0, 0, 0
    seen_lab = {}
    rng = random.Random(4242)
    for split in ("b5_hold_uc", "b5_hold_se", "b5_train", pw5.CHALLENGE_SPLIT):
        for idx in range(a.n_hist):
            cfg = pw5.split_config(split, idx)
            if cfg.k > 2:
                continue  # keep the audit DP cheap
            s = pw.ExactSolver(cfg)
            for w in range(40):
                ep = pw.Episode(cfg, 9_000_000 + 1000 * idx + w)
                while not ep.done:
                    st = ep.state
                    if pw.public_state_from_history(cfg, ep.history) != st:
                        state_mismatch += 1
                    key = (split, idx, tuple(ep.history))
                    lab = (round(s.value(st), 9), tuple(sorted((a_, round(v, 9)) for a_, v in s.q_values(st).items())),
                           tuple(sorted(s.opt_set(st))))
                    feats = (tuple(pw5.supplied_features(cfg, st, "bx2")), tuple(pw5.supplied_features(cfg, st, "belief")))
                    if key in seen_lab:
                        conflicts += seen_lab[key][0] != lab
                        feat_conflicts += seen_lab[key][1] != feats
                    else:
                        seen_lab[key] = (lab, feats)
                        n_hist += 1
                    ep.step(rng.choice(ep.available()))
    checks.append(check("D1 labels are functions of the visible history (0 conflicts)", conflicts == 0,
                        {"distinct_histories": n_hist, "conflicts": conflicts}))
    checks.append(check("D2 supplied BX2/BO features are functions of the visible history (0 conflicts)",
                        feat_conflicts == 0, {"conflicts": feat_conflicts}))
    checks.append(check("D3 simulator state == public state rebuilt from the visible history", state_mismatch == 0,
                        {"mismatches": state_mismatch}))
    res = {"tool": "campaign05_bsplit_audit", "version": pw5.VERSION, "all_pass": all(c["pass"] for c in checks),
           "checks": checks, "pairs": pairs,
           "split_definition": {s: {"cells": [list(c) for c in v[0]], "k": list(v[1]), "combos": [names(c) for c in v[2]],
                                    "config_seed_base": v[3], "pool_size": POOL_SIZES[s]}
                                for s, v in pw5.SPLITS5.items()},
           "challenge_split": {"name": pw5.CHALLENGE_SPLIT, "combos": [names(c) for c in pw.SPLITS["heldout_comp"][2]],
                               "note": "extended-04 configurations (B1/F2 evaluated on them): inspected data"},
           "combo_counts": combos_in, "cpu_s": time.process_time() - t0}
    return res


# ------------------------------------------------------------------------------------------------ triviality

def _keys(cfg, st):
    (i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev) = st
    book = (usage, reduced, cand, cand_valid, ev, built)
    bel = tuple(round(x, 3) for x in b)
    return {"bookkeeping_only": book, "belief_only": bel, "belief+bookkeeping": bel + book,
            "bx2_discrete": bel + book + cfg.flags + (cfg.k - i,)}


def _decisions(split, n, worlds, first_only=False):
    out = []
    for idx in range(n):
        cfg = pw5.split_config(split, idx)
        s = pw.ExactSolver(cfg)
        for w in range(worlds):
            ep = pw.Episode(cfg, pw5.world_seed(split, idx, 50 + w))
            t = 0
            while not ep.done:
                st = ep.state
                opt = s.opt_set(st)
                out.append({"cfg": (split, idx), "keys": _keys(cfg, st), "opt": opt, "pi": s.pi_star(st), "t": t,
                            "n_avail": len(pw.available(cfg, st))})
                ep.step(s.pi_star(st))
                t += 1
                if first_only:
                    break
    return out


def triviality(a):
    t0 = time.process_time()
    train = _decisions("b5_train", a.n_train, a.worlds)
    holds = _decisions("b5_hold_uc", a.n_hold, a.worlds) + _decisions("b5_hold_se", a.n_hold, a.worlds)
    res = {"tool": "campaign05_bsplit_audit triviality", "n_train_decisions": len(train),
           "n_hold_decisions": len(holds), "keys": {}}
    for scope in ("all_decisions", "first_decision"):
        tr = train if scope == "all_decisions" else [d for d in train if d["t"] == 0]
        ho = holds if scope == "all_decisions" else [d for d in holds if d["t"] == 0]
        ent = {"n_hold": len(ho), "singleton_opt_set": sum(len(d["opt"]) == 1 for d in ho) / len(ho),
               "one_available_action": sum(d["n_avail"] == 1 for d in ho) / len(ho)}
        for kname in ("bookkeeping_only", "belief_only", "belief+bookkeeping", "bx2_discrete"):
            table = {}
            for d in tr:
                table.setdefault(d["keys"][kname], {}).setdefault(d["pi"], 0)
                table[d["keys"][kname]][d["pi"]] += 1
            cov = [d for d in ho if d["keys"][kname] in table]
            acc = [max(table[d["keys"][kname]].items(), key=lambda kv: (kv[1], -kv[0]))[0] in d["opt"] for d in cov]
            # within-key purity on the holds (keys shared by >= 2 hold configurations)
            groups = {}
            for d in ho:
                groups.setdefault(d["keys"][kname], []).append(d)
            pure_n = pure_c = 0
            for g in groups.values():
                if len({d["cfg"] for d in g}) < 2:
                    continue
                votes = {}
                for d in g:
                    votes[d["pi"]] = votes.get(d["pi"], 0) + 1
                maj = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                pure_n += len(g)
                pure_c += sum(maj in d["opt"] for d in g)
            ent[kname] = {"transfer_coverage": len(cov) / len(ho),
                          "transfer_accuracy_on_covered": (sum(acc) / len(acc)) if acc else None,
                          "determined_fraction_of_all": sum(acc) / len(ho),
                          "hold_shared_key_decisions": pure_n,
                          "hold_shared_key_purity": (pure_c / pure_n) if pure_n else None}
        res["keys"][scope] = ent
    res["reading"] = ("fraction of new-hold decisions whose eps-optimal action is determined by the key without the "
                      "continuous prices; 1 - determined = decisions requiring the price/horizon/reliability "
                      "trade-off (or unseen keys)")
    res["cpu_s"] = time.process_time() - t0
    return res


def sizing(a):
    t0 = time.process_time()
    res = {"tool": "campaign05_bsplit_audit sizing", "target_eligible": pw5.HOLD_TARGET_ELIGIBLE, "splits": {}}
    for split, n in (("b5_hold_uc", a.n_uc), ("b5_hold_se", a.n_se)):
        recs = []
        for idx in range(n):
            t1 = time.process_time()
            cfg = pw5.split_config(split, idx)
            s = pw.ExactSolver(cfg)
            r = pw5.s0_record(s)
            if split in pw5.SENSITIVITY_FLAGS:
                r.update(pw5.flag_sensitivity(cfg, r["V"], pw5.SENSITIVITY_FLAGS[split]))
            r["cpu_s"] = time.process_time() - t1
            recs.append(r)
        elig = sum(r["probe_unique"] for r in recs)
        rate = elig / n
        size = pw5.hold_pool_size(recs)
        cpu_cfg = sum(r["cpu_s"] for r in recs) / n
        ent = {"sample": n, "eligible": elig, "eligible_rate": rate, "pool_size_if_reached_in_sample": size,
               "projected_pool_size": size or (round(pw5.HOLD_TARGET_ELIGIBLE / rate) if rate else None),
               "projected_pool_size_range_rate_pm_2se": None, "label_cpu_s_per_config": cpu_cfg,
               "by_k": {k: {"n": sum(r["k"] == k for r in recs),
                            "eligible": sum(r["k"] == k and r["probe_unique"] for r in recs)} for k in (1, 2, 8)}}
        if rate:
            se = (rate * (1 - rate) / n) ** 0.5
            lo, hi = max(rate - 2 * se, 1e-9), rate + 2 * se
            ent["projected_pool_size_range_rate_pm_2se"] = [round(pw5.HOLD_TARGET_ELIGIBLE / hi),
                                                            round(pw5.HOLD_TARGET_ELIGIBLE / lo)]
        ent["projected_label_cpu_s"] = (ent["projected_pool_size"] or 0) * cpu_cfg
        if split in pw5.SENSITIVITY_FLAGS:
            ent["flag_sensitive"] = sum(r["flag_sensitive"] for r in recs)
            ent["flag_sensitive_eligible"] = sum(r["flag_sensitive"] and r["probe_unique"] for r in recs)
            ent["flag_sensitive_by_k"] = {k: sum(r["k"] == k and r["flag_sensitive"] for r in recs) for k in (1, 2, 8)}
        res["splits"][split] = ent
    res["cpu_s"] = time.process_time() - t0
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit")
    s.add_argument("--out", required=True)
    s.add_argument("--n-pair", type=int, default=32)
    s.add_argument("--n-hist", type=int, default=8)
    s = sub.add_parser("triviality")
    s.add_argument("--out", required=True)
    s.add_argument("--n-hold", type=int, default=40)
    s.add_argument("--n-train", type=int, default=80)
    s.add_argument("--worlds", type=int, default=8)
    s = sub.add_parser("sizing")
    s.add_argument("--out", required=True)
    s.add_argument("--n-uc", type=int, default=480)
    s.add_argument("--n-se", type=int, default=160)
    a = p.parse_args(argv)
    res = {"audit": audit, "triviality": triviality, "sizing": sizing}[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=list))
    if a.cmd == "audit":
        for c in res["checks"]:
            print(("PASS " if c["pass"] else "FAIL ") + c["check"])
        print(json.dumps(res["pairs"], indent=1))
    elif a.cmd == "triviality":
        print(json.dumps(res["keys"], indent=1))
    else:
        print(json.dumps(res["splits"], indent=1))
    print("cpu_s", round(res["cpu_s"], 1))


if __name__ == "__main__":
    main()
