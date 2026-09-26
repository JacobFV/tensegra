"""Score protocol-B1 hypotheses B-H1..B-H6 from ladder eval.json files + b-refs.json (as registered).

Usage: python campaign04_b1_score.py <dir with b-train-*/run/eval.json and b-refs.json> [--out file]
"""
import json, sys
from pathlib import Path

HELD = ("heldout_price", "heldout_k", "heldout_comp")


def load(d):
    ev = {(r, s): json.loads((d / f"b-train-{r}-s{s}/run/eval.json").read_text())
          for r in ("L0", "L1", "L2", "L3", "L4") for s in range(3)}
    return ev, json.loads((d / "b-refs.json").read_text())


def fr(e, split):
    return e["splits"][split]["free_running_greedy"]


def score(ev, refs):
    out = {"per_seed": {}, "hypotheses": {}}
    reg = {(r, s): {sp: fr(ev[r, s], sp)["summary"]["regret"] for sp in HELD} for (r, s) in ev}
    for k, v in reg.items():
        v["mean"] = sum(v[sp] for sp in HELD) / 3
    out["per_seed"]["heldout_regret"] = {f"{r}-s{s}": v for (r, s), v in reg.items()}
    succ_floor = {f"{r}-s{s}": {sp: fr(ev[r, s], sp)["summary"]["success"] >= .8 * refs[sp]["pi_star"]["success"] for sp in HELD}
                  for (r, s) in ev}
    out["per_seed"]["transfer_floor"] = succ_floor
    # B-H1
    pairs = [reg["L4", s]["mean"] <= .8 * reg["L0", s]["mean"] for s in range(3)]
    per_split = {sp: sum(reg["L4", s][sp] <= reg["L0", s][sp] for s in range(3)) >= 2 for sp in HELD}
    floor4 = all(all(succ_floor[f"L4-s{s}"].values()) for s in range(3))
    out["hypotheses"]["B-H1"] = {"pairs_L4_le_0.8xL0": pairs, "per_split_2of3": per_split, "L4_transfer_floor_all": floor4,
                                 "supported": all(pairs) and all(per_split.values()) and floor4}
    # B-H2 descriptive
    rungs = ["L0", "L1", "L2", "L3", "L4"]
    out["hypotheses"]["B-H2"] = {f"{a}->{b}": {"seeds_reducing_ge_10pct": sum(reg[b, s]["mean"] <= .9 * reg[a, s]["mean"] for s in range(3)),
                                               "adds_value": sum(reg[b, s]["mean"] <= .9 * reg[a, s]["mean"] for s in range(3)) >= 2}
                                 for a, b in zip(rungs, rungs[1:])}
    # B-H3 (L4; heldout_k and heldout_price; unweighted over populated effective-rho bins)
    def rho_dev(e, sp):
        bins = [b for b in fr(e, sp)["rho_curve"]["by_rho_eff"] if b["n"] > 0]
        return sum(abs(b["build_rate"] - b["build_rate_pi_star"]) for b in bins) / len(bins)
    h3 = {s: {sp: rho_dev(ev["L4", s], sp) for sp in ("heldout_k", "heldout_price")} for s in range(3)}
    out["hypotheses"]["B-H3"] = {"L4_mad": h3, "all_rungs_mad": {f"{r}-s{s}": {sp: rho_dev(ev[r, s], sp) for sp in ("heldout_k", "heldout_price")} for (r, s) in ev},
                                 "supported": sum(all(v <= .15 for v in h3[s].values()) for s in range(3)) >= 2}
    # B-H4 (L4; held-out splits pooled by episode count)
    def h4(e):
        n1 = sum(fr(e, sp)["summary"]["n_probe_not_opt"] for sp in HELD)
        r1 = sum(fr(e, sp)["summary"]["first_probe_rate_when_not_opt"] * fr(e, sp)["summary"]["n_probe_not_opt"] for sp in HELD) / max(n1, 1)
        n2 = sum(fr(e, sp)["summary"]["n_probe_unique_opt"] for sp in HELD)
        r2 = sum(fr(e, sp)["summary"]["first_probe_rate_when_unique_opt"] * fr(e, sp)["summary"]["n_probe_unique_opt"] for sp in HELD) / max(n2, 1)
        return {"probe_rate_not_opt": r1, "n": n1, "probe_rate_unique_opt": r2, "n_unique": n2, "pass": r1 <= .10 and r2 >= .80}
    hh4 = {f"{r}-s{s}": h4(ev[r, s]) for (r, s) in ev}
    out["hypotheses"]["B-H4"] = {"by_run": hh4, "supported": sum(hh4[f"L4-s{s}"]["pass"] for s in range(3)) >= 2}
    # B-H5 (L4; held-out pooled)
    def h5(e):
        just = unjust = jb = 0.0; pjb = 0.0
        for sp in HELD:
            sm = fr(e, sp)["summary"]; n = sm["n"]
            just += sum(sm[f"switch_just_{c}"] for c in "abcd") * n
            unjust += sum(sm[f"switch_unjust_{c}"] for c in "abcd") * n
            jb += sm["switch_just_b"] * n; pjb += refs[sp]["pi_star"]["switch_just_b"] * n
        frac = unjust / max(just + unjust, 1e-9)
        return {"unjustified_fraction": frac, "just_after_b_ratio_to_pi_star": jb / max(pjb, 1e-9),
                "pass": frac <= .10 and jb >= .8 * pjb}
    hh5 = {f"{r}-s{s}": h5(ev[r, s]) for (r, s) in ev}
    out["hypotheses"]["B-H5"] = {"by_run": hh5, "supported": sum(hh5[f"L4-s{s}"]["pass"] for s in range(3)) >= 2}
    # B-H6 (value head reliability error, units of R, held-out splits)
    hh6 = {f"{r}-s{s}": {sp: fr(ev[r, s], sp)["value_calibration_own_return"]["reliability_error"] for sp in HELD} for (r, s) in ev}
    out["hypotheses"]["B-H6"] = {"reliability_error_R": hh6,
                                 "supported_L4": sum(all(v <= .05 for v in hh6[f"L4-s{s}"].values()) for s in range(3)) >= 2}
    out["refs_heldout_regret"] = {sp: {n: refs[sp][n]["regret"] for n in refs[sp]} for sp in HELD}
    return out


if __name__ == "__main__":
    d = Path(sys.argv[1]); ev, refs = load(d); res = score(ev, refs)
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v.get("supported", v.get("supported_L4", "descriptive")) for k, v in res["hypotheses"].items()}))
    for r in ("L0", "L1", "L2", "L3", "L4"):
        print(r, [round(res["per_seed"]["heldout_regret"][f"{r}-s{s}"]["mean"], 2) for s in range(3)],
              "floor", [all(res["per_seed"]["transfer_floor"][f"{r}-s{s}"].values()) for s in range(3)],
              "H4", [(round(res["hypotheses"]["B-H4"]["by_run"][f"{r}-s{s}"]["probe_rate_not_opt"], 3), round(res["hypotheses"]["B-H4"]["by_run"][f"{r}-s{s}"]["probe_rate_unique_opt"], 3)) for s in range(3)],
              "H5", [round(res["hypotheses"]["B-H5"]["by_run"][f"{r}-s{s}"]["unjustified_fraction"], 3) for s in range(3)],
              "H6", [round(max(res["hypotheses"]["B-H6"]["reliability_error_R"][f"{r}-s{s}"].values()), 3) for s in range(3)])
    print("H3 L4", res["hypotheses"]["B-H3"]["L4_mad"]); print("H2", {k: v["adds_value"] for k, v in res["hypotheses"]["B-H2"].items()})
    print("refs", {sp: {n: round(v, 1) for n, v in d2.items()} for sp, d2 in res["refs_heldout_regret"].items()})
