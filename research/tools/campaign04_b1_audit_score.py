"""Independent B1 audit: recompute B-H1..B-H6 from eval.json + b-refs.json (protocol-B1.md wording).

Written by the independent auditor without reading research/tools/campaign04_b1_score.py.
Pure Python; reads only committed results.  Every ambiguous protocol clause is computed under each
reasonable reading ("variants") and the reading used for the headline verdict is named.

usage: python research/tools/campaign04_b1_audit_score.py [--res DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNGS = ("L0", "L1", "L2", "L3", "L4")
SEEDS = (0, 1, 2)
HELD = ("heldout_price", "heldout_k", "heldout_comp")
ALL = ("dev", "test_iid") + HELD


def load(res):
    E = {}
    for r in RUNGS:
        for s in SEEDS:
            E[r, s] = json.loads((res / f"b-train-{r}-s{s}/run/eval.json").read_text())
            assert E[r, s]["rung"] == r and E[r, s]["seed"] == s
    refs = json.loads((res / "b-refs.json").read_text())
    return E, refs


def summ(E, r, s, sp):
    return E[r, s]["splits"][sp]["free_running_greedy"]["summary"]


def fr(E, r, s, sp):
    return E[r, s]["splits"][sp]["free_running_greedy"]


def held_mean(E, r, s, key):
    return sum(summ(E, r, s, sp)[key] for sp in HELD) / len(HELD)


def floor_ok(E, refs, r, s, splits=HELD):
    return {sp: summ(E, r, s, sp)["success"] >= 0.8 * refs[sp]["pi_star"]["success"] for sp in splits}


def bh1(E, refs, key):
    ratio = {s: held_mean(E, "L4", s, key) / held_mean(E, "L0", s, key) for s in SEEDS}
    a = all(ratio[s] <= 0.8 for s in SEEDS)
    per_split = {sp: sum(summ(E, "L4", s, sp)[key] <= summ(E, "L0", s, sp)[key] for s in SEEDS) for sp in HELD}
    b = all(v >= 2 for v in per_split.values())
    fl = {s: floor_ok(E, refs, "L4", s) for s in SEEDS}
    fl_all = all(all(v.values()) for v in fl.values())
    return {"metric": key, "held_mean": {r: [held_mean(E, r, s, key) for s in SEEDS] for r in RUNGS},
            "ratio_L4_over_L0": ratio, "all_3_pairs_le_0.8": a, "per_split_L4_le_L0_seed_count": per_split,
            "each_split_ge_2of3": b, "transfer_floor_L4": fl, "transfer_floor_all": fl_all,
            "verdict": "SUPPORTED" if (a and b and fl_all) else "NOT SUPPORTED"}


def bh2(E, key):
    out = {}
    for lo, hi in zip(RUNGS[:-1], RUNGS[1:]):
        rel = [held_mean(E, hi, s, key) / held_mean(E, lo, s, key) for s in SEEDS]
        n = sum(x <= 0.9 for x in rel)
        out[f"{lo}->{hi}"] = {"ratio_per_seed": rel, "n_seeds_ge_10pct_reduction": n, "adds_value": n >= 2}
    return {"metric": key, "steps": out}


def bh3_one(E, refs, r, s, sp):
    curve = fr(E, r, s, sp)["rho_curve"]["by_rho_eff"]
    ref_curve = {(c["lo"], c["hi"]): c for c in refs[sp]["pi_star"]["rho_curve"]["by_rho_eff"]}
    d = [abs(c["build_rate"] - c["build_rate_pi_star"]) for c in curve]
    n = [c["n"] for c in curve]
    for c in curve:  # model file's pi* column must equal the reference file's pi* on the same worlds
        rc = ref_curve[(c["lo"], c["hi"])]
        assert rc["n"] == c["n"] and abs(rc["build_rate_pi_star"] - c["build_rate_pi_star"]) < 1e-12
    return {"unweighted": sum(d) / len(d), "n_weighted": sum(x * m for x, m in zip(d, n)) / sum(n),
            "bins": [(c["lo"], c["hi"], c["n"], c["build_rate"], c["build_rate_pi_star"]) for c in curve]}


def bh3(E, refs):
    splits = ("heldout_k", "heldout_price")
    per = {r: {s: {sp: bh3_one(E, refs, r, s, sp) for sp in splits} for s in SEEDS} for r in RUNGS}
    verdicts = {}
    for variant in ("unweighted", "n_weighted"):
        # each split separately must be <= .15 (strict reading); seed passes if both splits pass
        n_each = sum(all(per["L4"][s][sp][variant] <= 0.15 for sp in splits) for s in SEEDS)
        # pooled reading: mean over the two splits
        n_pool = sum((sum(per["L4"][s][sp][variant] for sp in splits) / 2) <= 0.15 for s in SEEDS)
        verdicts[variant] = {"seeds_pass_each_split": n_each, "seeds_pass_split_mean": n_pool}
    fl = all(all(floor_ok(E, refs, "L4", s, splits).values()) for s in SEEDS)
    head = verdicts["unweighted"]["seeds_pass_each_split"] >= 2 and fl
    return {"per": {r: {s: {sp: {k: v for k, v in per[r][s][sp].items() if k != "bins"} for sp in splits}
                        for s in SEEDS} for r in RUNGS},
            "L4_bins": {s: {sp: per["L4"][s][sp]["bins"] for sp in splits} for s in SEEDS},
            "variants": verdicts, "transfer_floor": fl,
            "headline_reading": "unweighted mean over non-empty registered rho_eff bins, each split <= .15",
            "verdict": "SUPPORTED" if head else "NOT SUPPORTED",
            "all_variants_agree": all(v >= 2 for vv in verdicts.values() for v in vv.values()) == head}


def bh4_seed(E, r, s, splits):
    num_n = den_n = num_u = den_u = 0.0
    per = {}
    for sp in splits:
        m = summ(E, r, s, sp)
        pn, nn = m["first_probe_rate_when_not_opt"], m["n_probe_not_opt"]
        pu, nu = m["first_probe_rate_when_unique_opt"], m["n_probe_unique_opt"]
        per[sp] = {"not_opt": pn, "n_not_opt": nn, "unique": pu, "n_unique": nu,
                   "pass": (pn is not None and pn <= 0.10) and (pu is not None and pu >= 0.80)}
        if pn is not None:
            num_n += pn * nn
            den_n += nn
        if pu is not None:
            num_u += pu * nu
            den_u += nu
    pooled_n = num_n / den_n if den_n else None
    pooled_u = num_u / den_u if den_u else None
    return per, {"not_opt": pooled_n, "n_not_opt": den_n, "unique": pooled_u, "n_unique": den_u,
                 "pass": (pooled_n is not None and pooled_n <= 0.10) and (pooled_u is not None and pooled_u >= 0.80)}


def bh4(E):
    out = {}
    for label, splits in (("held_out", HELD), ("all_5", ALL)):
        res = {r: {s: bh4_seed(E, r, s, splits) for s in SEEDS} for r in RUNGS}
        out[label] = {
            "per_seed": {r: {s: {"per_split": res[r][s][0], "pooled": res[r][s][1]} for s in SEEDS} for r in RUNGS},
            "L4_seeds_pass_pooled": sum(res["L4"][s][1]["pass"] for s in SEEDS),
            "L4_seeds_pass_every_split": sum(all(v["pass"] for v in res["L4"][s][0].values()) for s in SEEDS),
            "L4_seeds_pass_per_split": {sp: sum(res["L4"][s][0][sp]["pass"] for s in SEEDS) for sp in splits}}
    h = out["held_out"]
    out["verdict_pooled_heldout"] = "SUPPORTED" if h["L4_seeds_pass_pooled"] >= 2 else "NOT SUPPORTED"
    out["verdict_each_heldout_split"] = "SUPPORTED" if h["L4_seeds_pass_every_split"] >= 2 else "NOT SUPPORTED"
    return out


SW_J = [f"switch_just_{c}" for c in "abcd"]
SW_U = [f"switch_unjust_{c}" for c in "abcd"]


def bh5_seed(E, refs, r, s, splits):
    # per-episode means with n = 512 in every split, so the sum of means is the n-weighted pool
    j = sum(summ(E, r, s, sp)[k] for sp in splits for k in SW_J)
    u = sum(summ(E, r, s, sp)[k] for sp in splits for k in SW_U)
    frac_unjust = u / (j + u) if j + u else None
    per_split_unjust = {}
    for sp in splits:
        jj = sum(summ(E, r, s, sp)[k] for k in SW_J)
        uu = sum(summ(E, r, s, sp)[k] for k in SW_U)
        per_split_unjust[sp] = uu / (jj + uu) if jj + uu else None
    m_jb = sum(summ(E, r, s, sp)["switch_just_b"] for sp in splits)
    p_jb = sum(refs[sp]["pi_star"]["switch_just_b"] for sp in splits)
    m_b = sum(summ(E, r, s, sp)["case_b"] for sp in splits)
    p_b = sum(refs[sp]["pi_star"]["case_b"] for sp in splits)
    ratio_per_ep = m_jb / p_jb
    ratio_cond = (m_jb / m_b) / (p_jb / p_b) if m_b else None
    per_split_ratio = {sp: summ(E, r, s, sp)["switch_just_b"] / refs[sp]["pi_star"]["switch_just_b"]
                       for sp in splits}
    return {"frac_unjust": frac_unjust, "frac_unjust_per_split": per_split_unjust,
            "just_b_ratio_per_episode": ratio_per_ep, "just_b_ratio_conditional_on_case_b": ratio_cond,
            "just_b_ratio_per_split": per_split_ratio,
            "pass_pooled": frac_unjust is not None and frac_unjust <= 0.10 and ratio_per_ep >= 0.8,
            "pass_each_split": all(v is not None and v <= 0.10 for v in per_split_unjust.values())
            and all(v >= 0.8 for v in per_split_ratio.values()),
            "pass_pooled_conditional": frac_unjust is not None and frac_unjust <= 0.10
            and ratio_cond is not None and ratio_cond >= 0.8}


def bh5(E, refs):
    out = {}
    for label, splits in (("held_out", HELD), ("all_5", ALL)):
        per = {r: {s: bh5_seed(E, refs, r, s, splits) for s in SEEDS} for r in RUNGS}
        out[label] = {"per_seed": per,
                      "L4_seeds_pass_pooled": sum(per["L4"][s]["pass_pooled"] for s in SEEDS),
                      "L4_seeds_pass_each_split": sum(per["L4"][s]["pass_each_split"] for s in SEEDS),
                      "L4_seeds_pass_pooled_conditional": sum(per["L4"][s]["pass_pooled_conditional"] for s in SEEDS)}
    out["verdict_pooled_heldout"] = "SUPPORTED" if out["held_out"]["L4_seeds_pass_pooled"] >= 2 else "NOT SUPPORTED"
    return out


def bh6(E):
    fields = ("value_calibration_own_return", "value_vs_vstar", "q_head_vs_qstar_taken")
    per = {r: {s: {f: {sp: fr(E, r, s, sp)[f]["reliability_error"] for sp in ALL} for f in fields} for s in SEEDS}
           for r in RUNGS}
    summ_ = {}
    for r in RUNGS:
        v = [per[r][s]["value_calibration_own_return"][sp] for s in SEEDS for sp in HELD]
        summ_[r] = {"own_return_heldout_min": min(v), "own_return_heldout_max": max(v),
                    "own_return_heldout_mean_per_seed": [
                        sum(per[r][s]["value_calibration_own_return"][sp] for sp in HELD) / 3 for s in SEEDS],
                    "seeds_all_heldout_splits_le_.05": sum(
                        all(per[r][s]["value_calibration_own_return"][sp] <= 0.05 for sp in HELD) for s in SEEDS),
                    "seeds_heldout_mean_le_.05": sum(
                        sum(per[r][s]["value_calibration_own_return"][sp] for sp in HELD) / 3 <= 0.05 for s in SEEDS)}
    # the only rung with Q/V supervision in the trainer is L4 (Q* regression); V head is never V*-supervised
    ok = summ_["L4"]["seeds_all_heldout_splits_le_.05"] >= 2
    return {"per": per, "summary": summ_,
            "verdict_L4_only": "SUPPORTED" if ok else "NOT SUPPORTED",
            "verdict_L1_L4": "SUPPORTED" if all(summ_[r]["seeds_all_heldout_splits_le_.05"] >= 2
                                              for r in RUNGS[1:]) else "NOT SUPPORTED"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", default="research/results/campaign-04/b1")
    p.add_argument("--out", default="research/campaigns/extended-04/review/b1-audit-score.json")
    a = p.parse_args()
    E, refs = load(Path(a.res))
    out = {"B-H1": {"realized_regret": bh1(E, refs, "regret"), "gap_regret": bh1(E, refs, "gap_regret")},
           "B-H2": {"realized_regret": bh2(E, "regret"), "gap_regret": bh2(E, "gap_regret")},
           "B-H3": bh3(E, refs), "B-H4": bh4(E), "B-H5": bh5(E, refs), "B-H6": bh6(E),
           "transfer_floor_all_runs": {f"{r}-s{s}": floor_ok(E, refs, r, s) for r in RUNGS for s in SEEDS}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({
        "B-H1": [out["B-H1"][k]["verdict"] for k in out["B-H1"]],
        "B-H2": {k: {s: v["adds_value"] for s, v in out["B-H2"][k]["steps"].items()} for k in out["B-H2"]},
        "B-H3": [out["B-H3"]["verdict"], out["B-H3"]["variants"]],
        "B-H4": [out["B-H4"]["verdict_pooled_heldout"], out["B-H4"]["verdict_each_heldout_split"],
                 out["B-H4"]["held_out"]["L4_seeds_pass_per_split"]],
        "B-H5": [out["B-H5"]["verdict_pooled_heldout"],
                 {k: out["B-H5"]["held_out"][k] for k in out["B-H5"]["held_out"] if k.startswith("L4")}],
        "B-H6": [out["B-H6"]["verdict_L4_only"], out["B-H6"]["verdict_L1_L4"],
                 {r: (round(v["own_return_heldout_min"], 3), round(v["own_return_heldout_max"], 3))
                  for r, v in out["B-H6"]["summary"].items()}]}, indent=1))


if __name__ == "__main__":
    main()
