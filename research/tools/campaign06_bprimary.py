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

B-FACT-C (registry B-FACT-C.primary; additive, the defaults above are unchanged and tested bit for bit):
  --group b6c_hold_SCE   evaluate another held-out group; a b6c group reads eval_b6c_episodes.jsonl.gz and
                         cf_eval_b6c.json (eval --split-set b6c) unless --eval-name / --cf-name are given
  --two-level            one bootstrap draw resamples the seed pairs with replacement, then the configurations and octets
                         (one configuration/octet draw shared by the chosen pairs); the CI is of the mean over pairs.
                         Rule (reference = any arm name; primary: --ref RAWF, --arm LRN):
    Q1  gap regret:      mean < 0, two-level 95% upper bound < 0, >= 4 of 5 pairs negative
    Q2  later-decision accuracy: mean > 0, lower bound > 0, >= 4 of 5 pairs positive
    Q3  near-miss accuracy mean >= -.02 and one-factor false-change rate mean <= +.02 (paired means)
    SUPPORTED iff Q1 and Q2 and Q3.

  python research/tools/campaign06_bprimary.py --group b6c_hold_SCE --two-level --ref RAWF RUN x5 --arm LRN RUN x5 \
      [--arm SUP RUN x5] --out bfactc.json --md bfactc.md
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


def file_names(group, eval_name=None, cf_name=None):
    b6c = group.startswith("b6c_")
    return (eval_name or ("eval_b6c_episodes.jsonl.gz" if b6c else "eval_episodes.jsonl.gz"),
            cf_name or ("cf_eval_b6c.json" if b6c else "cf_eval.json"))


def load(run, group=GROUP, eval_name=None, cf_name=None):
    ev, cfn = file_names(group, eval_name, cf_name)
    head, rows = bs.read_rows(run, ev)
    g = bs.groups_of(rows)[group]
    keys = sorted(g)
    S = {k: bs.config_stats(g[k]) for k in keys}
    cf = bs.read_cf(run, cfn)
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


def contrast_two_level(A, R, n_boot, seed):
    """B-FACT-C: as contrast(), but each bootstrap draw first resamples the seed pairs with replacement, then the
    configurations and octets (one configuration/octet draw shared by every chosen pair); the statistic is the mean
    over the drawn pairs of the paired difference.  Also reports pair sign counts."""
    keys = sorted(set.intersection(*[set(x["S"]) for x in A + R]))
    ckeys = sorted(set.intersection(*[set(x["C"]) for x in A + R]))
    P = len(A)
    out = {"n_pairs": P, "seeds": [[a["seed"], r["seed"]] for a, r in zip(A, R)],
           "support": {"configs": len(keys), "octets": len(ckeys)},
           "bootstrap": "two-level: seed pairs with replacement, then configurations/octets (shared by the drawn pairs)",
           "endpoints": {}}

    def value(kind, fn, run, ix, cix):
        if kind == "ep":
            return fn([run["S"][keys[i]] for i in ix])
        return fn([run["C"][ckeys[i]] for i in cix])

    full_ix, full_cix = list(range(len(keys))), list(range(len(ckeys)))
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pix = [rng.randrange(P) for _ in range(P)]
        draws.append((pix, [rng.randrange(len(keys)) for _ in keys], [rng.randrange(len(ckeys)) for _ in ckeys]))
    for name, (kind, fn) in ENDPOINTS.items():
        per = []
        for a, r in zip(A, R):
            va, vr = value(kind, fn, a, full_ix, full_cix), value(kind, fn, r, full_ix, full_cix)
            per.append(None if va is None or vr is None else va - vr)
        if any(p is None for p in per):
            out["endpoints"][name] = {"per_pair": per, "mean": None, "ci": None, "signs": None}
            continue
        mean = sum(per) / len(per)
        boots = []
        for pix, ix, cix in draws:
            d = {}
            for q in sorted(set(pix)):
                va, vr = value(kind, fn, A[q], ix, cix), value(kind, fn, R[q], ix, cix)
                if va is None or vr is None:
                    break
                d[q] = va - vr
            else:
                boots.append(sum(d[q] for q in pix) / P)
        boots.sort()
        ci = [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if len(boots) >= n_boot // 2 else None
        out["endpoints"][name] = {"per_pair": per, "mean": mean, "ci": ci,
                                  "signs": {"negative": sum(x < 0 for x in per), "positive": sum(x > 0 for x in per),
                                            "zero": sum(x == 0 for x in per)},
                                  "ref_mean": sum(value(kind, fn, r, full_ix, full_cix) for r in R) / len(R)}
    return out


def verdict_two_level(c, min_sign=4):
    """B-FACT-C registered rule (registry B-FACT-C.primary); min_sign = 4 of the 5 registered pairs."""
    e = c["endpoints"]
    g, la = e["gap_regret"], e["later_acc"]
    q1 = g["ci"] is not None and g["mean"] < 0 and g["ci"][1] < 0 and g["signs"]["negative"] >= min_sign
    q2 = la["ci"] is not None and la["mean"] > 0 and la["ci"][0] > 0 and la["signs"]["positive"] >= min_sign
    q3a = e["cf_near_miss_acc"]["mean"] is not None and e["cf_near_miss_acc"]["mean"] >= -0.02
    q3b = e["iv_false_change"]["mean"] is not None and e["iv_false_change"]["mean"] <= 0.02
    return {"Q1": q1, "Q2": q2, "Q3_near_miss": q3a, "Q3_false_change": q3b, "Q3": q3a and q3b,
            "SUPPORTED": q1 and q2 and q3a and q3b, "min_sign_pairs": min_sign, "n_pairs": c["n_pairs"]}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ref", nargs="+", required=True, help="NAME RUN [RUN ...] (any arm name; B-FACT-C: RAWF)")
    p.add_argument("--arm", nargs="+", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--md", default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--group", default=GROUP, help="held-out group (default b6_hold_SCE; B-FACT-C: b6c_hold_SCE)")
    p.add_argument("--eval-name", default=None, help="episode-row file in each run dir (default by --group)")
    p.add_argument("--cf-name", default=None, help="counterfactual record file in each run dir (default by --group)")
    p.add_argument("--two-level", action="store_true", help="B-FACT-C: pair-then-configuration bootstrap and Q1-Q3")
    p.add_argument("--min-sign", type=int, default=4, help="--two-level: pairs with the required sign (registered 4/5)")
    a = p.parse_args(argv)
    if not a.two_level and a.group == GROUP and a.eval_name is None and a.cf_name is None:
        return _main_b_arms(a)
    ld = lambda r: load(r, a.group, a.eval_name, a.cf_name)  # noqa: E731
    ref_name, ref_runs = a.ref[0], [ld(r) for r in a.ref[1:]]
    ev, cfn = file_names(a.group, a.eval_name, a.cf_name)
    res = {"version": "bprimary-v1" + ("+two-level" if a.two_level else ""), "group": a.group, "ref": ref_name,
           "n_boot": a.n_boot, "files": [ev, cfn], "contrasts": {}}
    if a.two_level:
        res["rule"] = "B-FACT-C (Q1 gap, Q2 later-decision accuracy, Q3 not a global shift)"
        res["min_sign_pairs"] = a.min_sign
    title = "B-FACT-C registered primary" if a.two_level else "B-ARMS registered primary"
    ci_kind = "two-level (seed pair, then configuration/octet) 95% CI" if a.two_level else \
        "configuration/octet-clustered 95% CI"
    lines = [f"# {title} ({a.group}; pooled over seed pairs; {ci_kind})", ""]
    for j, spec in enumerate(a.arm):
        name, runs = spec[0], [ld(r) for r in spec[1:]]
        R = ref_runs[:len(runs)]
        assert [x["seed"] for x in runs] == [x["seed"] for x in R], "pairs must share seeds"
        if a.two_level:
            c = contrast_two_level(runs, R, a.n_boot, seed=1000 + j)
            c["verdict"] = verdict_two_level(c, a.min_sign)
        else:
            c = contrast(runs, R, a.n_boot, seed=1000 + j)
            c["verdict"] = verdict(c)
        res["contrasts"][f"{name}-{ref_name}"] = c
        lines += [f"## {name} − {ref_name} ({c['n_pairs']} pairs; {c['support']['configs']} configs, {c['support']['octets']} octets)",
                  "", "| endpoint | ref mean | mean diff | 95% CI | pairs −/+ | per pair |", "|---|---|---|---|---|---|"]
        for en, v in c["endpoints"].items():
            f = lambda x: "—" if x is None else f"{x:+.3f}"  # noqa: E731
            ci = "—" if v["ci"] is None else f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
            sg = v.get("signs")
            sg = "—" if not sg else f"{sg['negative']}/{sg['positive']}"
            lines.append(f"| {en} | {v.get('ref_mean', 0):.3f} | {f(v['mean'])} | {ci} | {sg} | "
                         f"{' / '.join(f(x) for x in v['per_pair'])} |")
        lines += ["", "verdict (registered rule applied to this contrast): " + json.dumps(c["verdict"]), ""]
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.md:
        Path(a.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _main_b_arms(a):
    """The B-ARMS tool exactly as registered (defaults; bit-identical output, tested)."""
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
