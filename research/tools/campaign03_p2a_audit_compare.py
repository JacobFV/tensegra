#!/usr/bin/env python3
"""Compare the P2a auditor reconstruction with the root's p2a-analysis.json cell by cell.

Usage: python campaign03_p2a_audit_compare.py <recon.json> <p2a-analysis.json> [--tol 0.001]
"""
import argparse
import json


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}/"))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = float(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recon")
    ap.add_argument("root")
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--show-root-keys", action="store_true")
    a = ap.parse_args()
    mine = json.load(open(a.recon))
    root = json.load(open(a.root))
    if a.show_root_keys:
        print(list(root.keys()))
        for k, v in root.items():
            print(k, json.dumps(v)[:800])
        return
    # root layout discovered at runtime: cells keyed policy/mode/condition and groups keyed policy/mode
    rc = root.get("table")
    rg = root.get("groups")
    compared = diffs = 0
    rows = []
    for name, mine_d, root_d in [("cell", mine["cells"], rc), ("group", mine["groups"], rg)]:
        for key, mv in mine_d.items():
            rv = root_d.get(key) if isinstance(root_d, dict) else None
            if rv is None:
                rows.append((name, key, "missing in root", None, None))
                continue
            for m, x in mv.items():
                if not isinstance(x, (int, float)) or isinstance(x, bool) or m not in rv:
                    continue
                y = rv[m]
                if y is None or not isinstance(y, (int, float)):
                    if x is not None:
                        rows.append((name, key, m, x, y))
                    continue
                compared += 1
                if abs(x - y) > a.tol * (max(1.0, abs(y)) if m == "work_per_success" else 1.0):
                    diffs += 1
                    rows.append((name, key, m, x, y))
    print(f"compared {compared} values; {diffs} differ by > {a.tol}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
