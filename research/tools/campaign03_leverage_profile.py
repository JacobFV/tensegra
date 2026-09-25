"""Stage A leverage profile for depworld-v1 (CPU only, in-process solver).

Runs the five supplied references on N development worlds per condition over a
small grid (sizes x p_event x foreign_records) and reports success, utility,
cost, work, calls, revisions per reference. Utility is re-priced for several
work_price values from the recorded work units (references do not read the
work price, so their behavior is identical across prices).

Seeds are development seeds >= 2_000_000_000 (never the 70M-100M band).
No learned policy, no GPU, no subprocess solver.

    PYTHONPATH=src python research/tools/campaign03_leverage_profile.py --examples 64
"""
from __future__ import annotations

import argparse
from functools import partial
import json
import time

from topoformer.campaign02_protocol import execute
from topoformer.campaign03_depworld import (REFERENCE_MODES, DepReference, DepWorkshop, depworld_executor,
                                            generate_depworld)

SEED_BASE = 2_000_000_000
SIZES = {"s3": dict(categories=3, choices=3, locations=7, slots=6),
         "s4": dict(categories=4, choices=3, locations=8, slots=7)}
P_EVENT = (0.0, 0.5, 1.0)
FOREIGN = (0, 2, 4)
WORK_PRICES = (1e-4, 2e-4, 5e-4)


def run_episode(spec, mode, address_seed=0):
    env = DepWorkshop(spec, executor=partial(depworld_executor, execute_call=execute), address_seed=address_seed)
    ref = DepReference(mode)
    o = env.observe()
    while not o.done:
        o = env.step(ref.choose(o))
    return env.evaluate(), spec


def summarize(outcomes, spec_prices):
    n = len(outcomes)
    out = {"n": n, "success": sum(r["verified_success"] for r in outcomes) / n,
           "cost": sum(r["cost"] for r in outcomes) / n, "utility": sum(r["utility"] for r in outcomes) / n,
           "work": sum(r["work_units"] for r in outcomes) / n, "calls": sum(r["calls"] for r in outcomes) / n,
           "steps": sum(r["steps"] for r in outcomes) / n,
           "revisions": sum(sum(r["revisions"].values()) for r in outcomes) / n,
           "revised_episodes": sum(sum(r["revisions"].values()) > 0 for r in outcomes) / n,
           "revocation_episodes": sum(bool(r["revocations"]) for r in outcomes) / n,
           "event_fired": sum(bool(r["events"]) for r in outcomes) / n,
           "invalid_uses": sum(r["reuse"]["invalid_uses"] for r in outcomes) / n,
           "foreign_uses": sum(r["reuse"]["foreign_uses"] for r in outcomes) / n,
           "foreign_applicable_uses": sum(r["reuse"]["foreign_applicable_uses"] for r in outcomes) / n}
    for price in WORK_PRICES:
        out[f"utility@{price:g}"] = sum(r["utility"] + r["work_units"] * (spec_prices - price)
                                        for r in outcomes) / n
        out[f"cost@{price:g}"] = sum(r["cost"] - r["work_units"] * (spec_prices - price) for r in outcomes) / n
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--json", help="optional output path for the full table")
    parser.add_argument("--world-json", default="{}", help="generator overrides applied to every cell")
    args = parser.parse_args()
    overrides = json.loads(args.world_json)
    rows = []
    start = time.time()
    cell = 0
    for size in args.sizes.split(","):
        for p_event in P_EVENT:
            for foreign in FOREIGN:
                seeds = [SEED_BASE + cell * 100_000 + i for i in range(args.examples)]
                cell += 1
                specs = [generate_depworld(s, **SIZES[size], p_event=p_event, foreign_records=foreign, **overrides)
                         for s in seeds]
                for mode in REFERENCE_MODES:
                    outcomes = [run_episode(spec, mode, address_seed=seed)[0] for seed, spec in zip(seeds, specs)]
                    rows.append({"size": size, "p_event": p_event, "foreign": foreign, "reference": mode,
                                 **summarize(outcomes, specs[0].work_price)})
    wp = [f"utility@{p:g}" for p in WORK_PRICES]
    header = ("size", "p_ev", "forgn", "reference", "succ", "util", *[w.replace("utility", "u") for w in wp],
              "cost", "work", "calls", "steps", "revis", "rev_ep", "evFire", "revoke", "badUse", "frgnUse")
    print(" ".join(f"{h:>10}" for h in header))
    for r in rows:
        vals = (r["size"], r["p_event"], r["foreign"], r["reference"], f"{r['success']:.3f}", f"{r['utility']:.4f}",
                *[f"{r[w]:.4f}" for w in wp], f"{r['cost']:.4f}", f"{r['work']:.1f}", f"{r['calls']:.2f}",
                f"{r['steps']:.1f}", f"{r['revisions']:.2f}", f"{r['revised_episodes']:.2f}",
                f"{r['event_fired']:.2f}", f"{r['revocation_episodes']:.2f}", f"{r['invalid_uses']:.2f}",
                f"{r['foreign_uses']:.2f}")
        print(" ".join(f"{v:>10}" for v in vals))
    print(f"# wall seconds {time.time() - start:.1f}; examples per cell {args.examples}; seeds from {SEED_BASE}")
    gate(rows)
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(rows, stream, indent=2)


def gate(rows):
    """Print the Stage A gate clauses, pooled over conditions and per cell."""
    def pool(pred, ref):
        sel = [r for r in rows if pred(r) and r["reference"] == ref]
        n = sum(r["n"] for r in sel)
        return ({k: sum(r[k] * r["n"] for r in sel) / n for k in ("success", "cost", "utility")} if n else None)
    every = lambda r: True
    reuse, rec = pool(every, "reuse"), pool(every, "recompute")
    print(f"# G1 pooled: reuse success {reuse['success']:.3f} vs recompute {rec['success']:.3f} "
          f"(need >= {rec['success'] - .01:.3f}); cost {reuse['cost']:.4f} vs {rec['cost']:.4f} "
          f"({(reuse['cost'] / rec['cost'] - 1) * 100:+.1f}%)")
    for price in WORK_PRICES:
        key = f"cost@{price:g}"
        a = sum(r[key] * r["n"] for r in rows if r["reference"] == "reuse")
        b = sum(r[key] * r["n"] for r in rows if r["reference"] == "recompute")
        print(f"# G1 pooled cost ratio at work_price {price:g}: {(a / b - 1) * 100:+.1f}%")
    cells = {(r["size"], r["p_event"], r["foreign"]) for r in rows}
    worst = []
    for c in sorted(cells):
        get = {r["reference"]: r for r in rows if (r["size"], r["p_event"], r["foreign"]) == c}
        worst.append((get["reuse"]["success"] - get["recompute"]["success"], c,
                      get["reuse"]["cost"] / get["recompute"]["cost"] - 1))
    d, c, rc = min(worst)
    print(f"# G1 per cell: worst reuse-recompute success gap {d:+.3f} at {c}; cost ratio range "
          f"{min(w[2] for w in worst) * 100:+.1f}% .. {max(w[2] for w in worst) * 100:+.1f}%")
    for label, pred in (("foreign>0", lambda r: r["foreign"] > 0), ("p_event>0", lambda r: r["p_event"] > 0),
                        ("p_event>0,foreign=0", lambda r: r["p_event"] > 0 and r["foreign"] == 0),
                        ("foreign=0,p_event=0", lambda r: r["foreign"] == 0 and r["p_event"] == 0)):
        n, u = pool(pred, "naive_reuse"), pool(pred, "reuse")
        print(f"# G2 naive vs reuse [{label}]: success {n['success']:.3f} vs {u['success']:.3f}; "
              f"utility {n['utility']:.4f} vs {u['utility']:.4f}")
    g, u = pool(every, "greedy"), pool(every, "reuse")
    print(f"# G3 greedy vs reuse pooled: success {g['success']:.3f} vs {u['success']:.3f}")
    nr = pool(every, "reuse_norevise")
    print(f"# G4 reuse without revision vs reuse pooled: success {nr['success']:.3f} vs {u['success']:.3f}")
    events = [r for r in rows if r["p_event"] > 0]
    for ref in REFERENCE_MODES:
        sel = [r for r in events if r["reference"] == ref]
        n = sum(r["n"] for r in sel)
        fired = sum(r["event_fired"] * r["n"] for r in sel) / n
        revoked = sum(r["revocation_episodes"] * r["n"] for r in sel) / n
        per_p = {p: sum(r["event_fired"] * r["n"] for r in sel if r["p_event"] == p) /
                 sum(r["n"] for r in sel if r["p_event"] == p) for p in sorted({r["p_event"] for r in sel})}
        print(f"# exposure {ref}: event fired {fired:.3f} of p_event>0 episodes "
              f"({', '.join(f'p={p:g}: {v:.3f}' for p, v in per_p.items())}); revocation episodes {revoked:.3f}")


if __name__ == "__main__":
    main()
