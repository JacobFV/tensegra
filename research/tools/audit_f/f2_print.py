"""Pretty-print f2_audit.py output."""
import json, sys
o = json.load(open(sys.argv[1]))
r = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
for s, v in o["H1"]["per_seed"].items():
    print("H1", s, r(v["regret_mean"]), r(v["gap_regret_mean"]), v["L1_pass"], v["L4_pass"])
print("H1 refs regret", {sp: {k: round(x, 2) for k, x in d.items()} for sp, d in o["H1"]["ref_regret"].items()})
for s, v in o["H2"]["per_seed"].items():
    print("H2", s, {sp: r(v[sp]) for sp in ("heldout_k", "heldout_price")}, v["pass"])
for s, v in o["H3"]["per_seed"].items():
    print("H3", s, round(v["pooled_not_opt"], 4), round(v["pooled_unique_opt"], 4), v["pass"],
          {sp: (round(x["not_opt"], 3), round(x["unique_opt"], 3), x["pass"]) for sp, x in v["per_split"].items()})
for s, v in o["H4"]["per_seed"].items():
    print("H4", s, r(v["pooled"]), v["pass"],
          {sp: (round(x["unjust_frac"], 3), round(x["ratio_b"], 3), round(x["ratio_b_per_case"], 3)) for sp, x in v["per_split"].items()})
print({k: o[k]["supported"] for k in ["H1", "H2", "H3", "H4"]}, "H4 per-case variant", o["H4"]["supported_per_case_variant"])
for s, v in o["H1"]["per_seed"].items():
    print("succ", s, {rr: {sp: round(x, 3) for sp, x in d.items()} for rr, d in v["per_split_success"].items()})
print("pi* success", o["H1"]["pi_star_success"])
