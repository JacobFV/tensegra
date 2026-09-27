#!/usr/bin/env python3
"""Summarize campaign05_b_audit_bloc_control.py output: BO vs public-input-model 'resolve' rates at first errors.

Usage: campaign05_b_audit_bloc_control_summary.py BLOC_CONTROL_JSON OUT_JSON
"""
import itertools
import json
import sys

BO = ["bx-bo-s0", "bx-bo-s1", "bx-bo-s2"]


def main(inp, out):
    rows = json.load(open(inp))["rows"]
    mism = sum(r["ok"][b][0] != r["bo_argmax"][i] for r in rows for i, b in enumerate(BO))
    res = {"n_first_errors": len(rows), "bo_replay_argmax_mismatch_vs_records": mism, "control": {}}
    for scope in ("all", "s0", "later"):
        for rung in ("L1", "L4", "both"):
            sel = [r for r in rows if (rung == "both" or f"-{rung}-" in r["model"])
                   and (scope == "all" or (scope == "s0") == (r["t"] == 0))]
            if not sel:
                continue
            n = len(sel)
            bo_maj = sum(sum(r["ok"][b][1] for b in BO) >= 2 for r in sel)
            pub = pub_each = 0.0
            for r in sel:
                others = sorted(k for k in r["ok"] if not k.startswith("bx-bo"))
                subs = list(itertools.combinations(others, 3))  # exact expectation over 3-model subsets
                pub += sum(sum(r["ok"][o][1] for o in s) >= 2 for s in subs) / len(subs)
                pub_each += sum(r["ok"][o][1] for o in others) / len(others)
            res["control"][f"{scope}/{rung}"] = {
                "n": n, "bo_majority3_resolves": round(bo_maj / n, 3),
                "public_L1_other_seeds_majority3_resolves": round(pub / n, 3),
                "bo_each": round(sum(r["ok"][b][1] for r in sel for b in BO) / (3 * n), 3),
                "public_L1_other_seeds_each": round(pub_each / n, 3)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
