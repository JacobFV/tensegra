"""Score protocol-F F2 (registered 2026-09-26T21:50Z) from f-btrain-{L0,L1,L4}-s{3,4,5}/run/eval.json + f-brefs.json.
Reuses the B1 metric helpers (campaign04_b1_score) with the F2 rung/seed set; per-split readings are printed."""
import importlib.util, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("b1s", Path(__file__).with_name("campaign04_b1_score.py"))
b1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b1)
HELD, SEEDS = b1.HELD, (3, 4, 5)


def main():
    d = Path(sys.argv[1])
    ev = {(r, s): json.loads((d / f"f-btrain-{r}-s{s}/run/eval.json").read_text()) for r in ("L0", "L1", "L4") for s in SEEDS}
    refs = json.loads((d / "f-brefs.json").read_text())
    fr = b1.fr
    reg = {k: {sp: fr(e, sp)["summary"]["regret"] for sp in HELD} for k, e in ev.items()}
    for v in reg.values(): v["mean"] = sum(v[sp] for sp in HELD) / 3
    floor = {k: all(fr(e, sp)["summary"]["success"] >= .8 * refs[sp]["pi_star"]["success"] for sp in HELD) for k, e in ev.items()}
    h1 = {r: [reg[r, s]["mean"] <= .8 * reg["L0", s]["mean"] and floor[r, s] for s in SEEDS] for r in ("L1", "L4")}
    def rho_dev(e, sp):
        bins = [x for x in fr(e, sp)["rho_curve"]["by_rho_eff"] if x["n"] > 0]
        return sum(abs(x["build_rate"] - x["build_rate_pi_star"]) for x in bins) / len(bins)
    h2 = [all(rho_dev(ev["L4", s], sp) <= .15 for sp in ("heldout_k", "heldout_price")) for s in SEEDS]
    def probe(e, splits):
        n1 = sum(fr(e, sp)["summary"]["n_probe_not_opt"] for sp in splits); n2 = sum(fr(e, sp)["summary"]["n_probe_unique_opt"] for sp in splits)
        r1 = sum(fr(e, sp)["summary"]["first_probe_rate_when_not_opt"] * fr(e, sp)["summary"]["n_probe_not_opt"] for sp in splits) / max(n1, 1)
        r2 = sum((fr(e, sp)["summary"]["first_probe_rate_when_unique_opt"] or 0) * fr(e, sp)["summary"]["n_probe_unique_opt"] for sp in splits) / max(n2, 1)
        return round(r1, 3), round(r2, 3), r1 <= .10 and r2 >= .80
    def switch(e, splits):
        just = unj = jb = pjb = 0.0
        for sp in splits:
            sm = fr(e, sp)["summary"]; n = sm["n"]
            just += sum(sm[f"switch_just_{c}"] for c in "abcd") * n; unj += sum(sm[f"switch_unjust_{c}"] for c in "abcd") * n
            jb += sm["switch_just_b"] * n; pjb += refs[sp]["pi_star"]["switch_just_b"] * n
        f = unj / max(just + unj, 1e-9); r = jb / max(pjb, 1e-9)
        return round(f, 3), round(r, 3), f <= .10 and r >= .8
    h3 = {s: {"pooled": probe(ev["L4", s], HELD), **{sp: probe(ev["L4", s], (sp,)) for sp in HELD}} for s in SEEDS}
    h4 = {s: {"pooled": switch(ev["L4", s], HELD), **{sp: switch(ev["L4", s], (sp,)) for sp in HELD}} for s in SEEDS}
    out = {"regret": {f"{r}-s{s}": v for (r, s), v in reg.items()}, "floor": {f"{r}-s{s}": v for (r, s), v in floor.items()},
           "F2-H1": {"per_seed": h1, "supported": all(h1["L1"]) and all(h1["L4"])},
           "F2-H2": {"per_seed": h2, "rho_dev": {s: {sp: rho_dev(ev["L4", s], sp) for sp in ("heldout_k", "heldout_price")} for s in SEEDS}, "supported": sum(h2) >= 2},
           "F2-H3": {"per_seed": h3, "supported": sum(h3[s]["pooled"][2] for s in SEEDS) >= 2},
           "F2-H4": {"per_seed": h4, "supported": sum(h4[s]["pooled"][2] for s in SEEDS) >= 2},
           "refs_regret": {sp: {n: refs[sp][n]["regret"] for n in refs[sp]} for sp in HELD}}
    if "--out" in sys.argv: Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(out, indent=1))
    for r in ("L0", "L1", "L4"): print(r, [round(reg[r, s]["mean"], 2) for s in SEEDS], "floor", [floor[r, s] for s in SEEDS])
    for k in ("F2-H1", "F2-H2", "F2-H3", "F2-H4"): print(k, out[k]["supported"], json.dumps(out[k]["per_seed"]))
    print("rho_dev", out["F2-H2"]["rho_dev"]); print("refs", out["refs_regret"])


if __name__ == "__main__":
    main()
