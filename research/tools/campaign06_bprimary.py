"""Extended-06 B-ARMS registered primary (registry.json B-ARMS.primary_registered), pooled over seed pairs.

campaign06_bscore.py reports each seed pair separately; the registered rule is on the seed-paired MEAN with a
configuration-clustered 95% CI.  Here one bootstrap draw resamples the held-out configurations (and, for the
counterfactual endpoints, whole octets) once and applies the same draw to every pair, so the CI of the mean over pairs
is configuration-clustered and pair-paired.

  P1  gap regret on b6_hold_SCE:          mean_pairs(B2 - B0) < 0 and 95% CI upper bound < 0
  P2  later-decision accuracy (all decisions after the first, pooled over the context classes) on b6_hold_SCE:
                                          mean_pairs(B2 - B0) > 0 and 95% CI lower bound > 0
  P3  not a global shift (SCE octets):    mean_pairs(near-miss acc B2 - B0) >= -.02 and
                                          mean_pairs(false-change rate B2 - B0) <= +.02  (paired means)
  SUPPORTED iff P1 and P2 and P3.  Other arms vs the reference are reported with the same statistics (descriptive).

  python research/tools/campaign06_bprimary.py --ref B0 RUN.. --arm B2 RUN.. [--arm B1 RUN..] --out primary.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign06_bscore as bs  # noqa: E402

GROUP = "b6_hold_SCE"
CTXS = ("query_first", "after_probe_solved", "after_probe_failed", "after_b1_timeout", "other")


def later_acc(S):
    n = sum(s["ctx"].get(c, (0, 0))[0] for s in S for c in CTXS)
    k = sum(s["ctx"].get(c, (0, 0))[1] for s in S for c in CTXS)
    return k / n if n else None


def gap(S):
    return bs._mean(s["gap"] for s in S)


CF = bs.cf_funcs()
ENDPOINTS = {"gap_regret": ("ep", gap), "later_acc": ("ep", later_acc), "first_acc": ("ep", bs.endpoint_funcs()["first_acc"]),
             "cf_near_miss_acc": ("cf", bs._flat(CF["cf_near_miss_acc"])),
             "iv_false_change": ("cf", bs._flat(CF["iv_change_given_opt_same"])),
             "cf_flip_full_acc": ("cf", bs._flat(CF["cf_flip_full_acc"])),
             "cf_balanced_acc": ("cf", bs._flat(CF["cf_balanced_acc"]))}


def load(run):
    head, rows = bs.read_rows(run)
    g = bs.groups_of(rows)[GROUP]
    keys = sorted(g)
    S = {k: bs.config_stats(g[k]) for k in keys}
    cf = bs.read_cf(run)
    U = bs.cf_units(cf, None) if cf is not None else []
    clusters, ckeys = bs._cluster(U)
    fam = {k: c for k, c in zip(ckeys, clusters)}
    return {"seed": head["seed"], "S": S, "C": fam}


def contrast(A, R, n_boot, seed):
    """A, R: aligned lists of loaded runs (pairs).  Returns per-pair diffs, mean and pooled CI per endpoint."""
    keys = sorted(set.intersection(*[set(x["S"]) for x in A + R]))
    ckeys = sorted(set.intersection(*[set(x["C"]) for x in A + R]))
    out = {"n_pairs": len(A), "seeds": [[a["seed"], r["seed"]] for a, r in zip(A, R)],
           "support": {"configs": len(keys), "octets": len(ckeys)}, "endpoints": {}}

    def value(kind, fn, run, ix, cix):
        if kind == "ep":
            return fn([run["S"][keys[i]] for i in ix])
        return fn([run["C"][ckeys[i]] for i in cix])

    full_ix, full_cix = list(range(len(keys))), list(range(len(ckeys)))
    rng = random.Random(seed)
    draws = [([rng.randrange(len(keys)) for _ in keys], [rng.randrange(len(ckeys)) for _ in ckeys]) for _ in range(n_boot)]
    for name, (kind, fn) in ENDPOINTS.items():
        per = []
        for a, r in zip(A, R):
            va, vr = value(kind, fn, a, full_ix, full_cix), value(kind, fn, r, full_ix, full_cix)
            per.append(None if va is None or vr is None else va - vr)
        if any(p is None for p in per):
            out["endpoints"][name] = {"per_pair": per, "mean": None, "ci": None}
            continue
        mean = sum(per) / len(per)
        boots = []
        for ix, cix in draws:
            ds = []
            for a, r in zip(A, R):
                va, vr = value(kind, fn, a, ix, cix), value(kind, fn, r, ix, cix)
                if va is None or vr is None:
                    break
                ds.append(va - vr)
            else:
                boots.append(sum(ds) / len(ds))
        boots.sort()
        ci = [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if len(boots) >= n_boot // 2 else None
        out["endpoints"][name] = {"per_pair": per, "mean": mean, "ci": ci,
                                  "ref_mean": sum(value(kind, fn, r, full_ix, full_cix) for r in R) / len(R)}
    return out


def verdict(c):
    e = c["endpoints"]
    p1 = e["gap_regret"]["ci"] is not None and e["gap_regret"]["mean"] < 0 and e["gap_regret"]["ci"][1] < 0
    p2 = e["later_acc"]["ci"] is not None and e["later_acc"]["mean"] > 0 and e["later_acc"]["ci"][0] > 0
    p3a = e["cf_near_miss_acc"]["mean"] is not None and e["cf_near_miss_acc"]["mean"] >= -0.02
    p3b = e["iv_false_change"]["mean"] is not None and e["iv_false_change"]["mean"] <= 0.02
    return {"P1": p1, "P2": p2, "P3_near_miss": p3a, "P3_false_change": p3b, "P3": p3a and p3b,
            "SUPPORTED": p1 and p2 and p3a and p3b}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ref", nargs="+", required=True, help="NAME RUN [RUN ...]")
    p.add_argument("--arm", nargs="+", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--md", default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)
    ref_name, ref_runs = a.ref[0], [load(r) for r in a.ref[1:]]
    res = {"version": "bprimary-v1", "group": GROUP, "ref": ref_name, "n_boot": a.n_boot, "contrasts": {}}
    lines = [f"# B-ARMS registered primary ({GROUP}; pooled over seed pairs; configuration/octet-clustered 95% CI)", ""]
    for j, spec in enumerate(a.arm):
        name, runs = spec[0], [load(r) for r in spec[1:]]
        R = ref_runs[:len(runs)]
        assert [x["seed"] for x in runs] == [x["seed"] for x in R], "pairs must share seeds"
        c = contrast(runs, R, a.n_boot, seed=1000 + j)
        c["verdict"] = verdict(c)
        res["contrasts"][f"{name}-{ref_name}"] = c
        lines += [f"## {name} − {ref_name} ({c['n_pairs']} pairs; {c['support']['configs']} configs, {c['support']['octets']} octets)",
                  "", "| endpoint | ref mean | mean diff | 95% CI | per pair |", "|---|---|---|---|---|"]
        for en, v in c["endpoints"].items():
            f = lambda x: "—" if x is None else f"{x:+.3f}"
            ci = "—" if v["ci"] is None else f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
            lines.append(f"| {en} | {v.get('ref_mean', 0):.3f} | {f(v['mean'])} | {ci} | {' / '.join(f(x) for x in v['per_pair'])} |")
        lines += ["", "verdict (registered rule applied to this contrast): " + json.dumps(c["verdict"]), ""]
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.md:
        Path(a.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
