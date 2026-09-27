"""Extended-06 Track B scorer (pure Python): held-out-family endpoints, balanced counterfactual conditional accuracy
and on-manifold one-factor interventions, per arm and seed, with configuration-clustered bootstrap CIs.

Inputs are run directories evaluated with
  campaign04_probeworld_train.py eval --split-set b6 --episode-rows --cf ...
(eval_episodes.jsonl.gz: one row per episode with exact per-decision labels; cf_eval.json: the model's choice at every
member x decision type of every balanced counterfactual set).

Endpoints (per run, per group = eval split, plus the registered held-out subsets 'all factors regret-relevant' and
'joint flip at s0'):
  first_acc            P(first action eps-optimal)                                   (configurations)
  first_acc_by_class   P(first action = the unique optimum | unique optimum in class), classes probe / exact (b1, b2) /
                       gather (inspect, prop) / structure (build) / terminal (commit_infeasible, abstain)
  probe_unique         P(first action = probe | probe uniquely optimal)  (the extended-04/05 endpoint)
  ctx_acc              decision accuracy by visible context in the model's own trajectory: query_first (later
                       queries), after_probe_solved, after_probe_failed, after_b1_timeout, other
  gap_regret           sum_t [V*(I_t) - Q*(I_t, a_t)] per episode; realized regret V*(I0) - U
  success, success_floor   success rate and the floor success >= .8 x pi* success on the same worlds
  cf (per decision type and pooled): on FLIP sets (the full family's unique optimum is not eps-optimal for any
       one-factor ablation's optimal action): full-member accuracy (conditional accuracy), all-members accuracy,
       near-miss accuracy (full-family configuration whose decision is inherited), balanced = mean(full, near-miss)
       on sets that have a near-miss; non-flip full-member accuracy
  intervention: one-factor member pairs with unique optima at both ends: P(model changes | optimum changes),
       P(model changes | optimum unchanged) (false change), P(new choice optimal | optimum changes)
CIs: 95% percentile bootstrap over configurations (1,000 resamples); paired arm - reference differences resample the
same configurations (same seed pairs).  Support < 20 is flagged 'insufficient'.

  python research/tools/campaign06_bscore.py --arm B0 RUN RUN RUN --arm B2 RUN RUN RUN [--ref B0] --out score.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

ACTIONS = ("probe", "exact_b1", "exact_b2", "inspect", "prop", "build", "use", "commit", "commit_infeasible", "abstain")
CLASSES = {"probe": {0}, "exact": {1, 2}, "gather": {3, 4}, "structure": {5}, "terminal": {8, 9}}
A_PROBE, A_B1, O_SOLVED, O_FAILED, O_TIMEOUT = 0, 1, 0, 1, 2
TERMINAL = {7, 8, 9}
MIN_SUPPORT = 20
N_BOOT = 1000


def read_rows(run, name="eval_episodes.jsonl.gz"):
    with gzip.open(Path(run) / name, "rt") as f:
        head = json.loads(f.readline())["_meta"]
        rows = [json.loads(line) for line in f if line.strip()]
    return head, rows


def read_cf(run, name="cf_eval.json"):
    p = Path(run) / name
    return json.loads(p.read_text()) if p.exists() else None


def _ctx(prev):
    if prev is None:
        return None
    a, o = prev["rec"][0], prev["rec"][1]
    if a == A_PROBE and o == O_SOLVED:
        return "after_probe_solved"
    if a == A_PROBE and o == O_FAILED:
        return "after_probe_failed"
    if a == A_B1 and o == O_TIMEOUT:
        return "after_b1_timeout"
    return "other"


def config_stats(row, eps=0.5):
    """Per-configuration (episode) statistics used by every endpoint (one world per configuration)."""
    ds = row["decisions"]
    d0 = ds[0]
    opt0 = set(d0["opt"])
    st = {"first_ok": d0["a"] in opt0, "unique": len(opt0) == 1, "opt0": d0["opt"][0] if len(opt0) == 1 else None,
          "a0": d0["a"], "gap": row["gap_regret"], "regret": row["V_star"] - row["U"], "success": row["success"],
          "pi_success": row["pi_star_success"], "ctx": {}}
    prev = None
    for d in ds:
        if prev is not None:
            c = "query_first" if d["step_in_query"] == 0 else _ctx(prev)
            ok = d["delta"] <= eps
            n, k = st["ctx"].get(c, (0, 0))
            st["ctx"][c] = (n + 1, k + ok)
        prev = d
    return st


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def endpoint_funcs():
    """name -> function(list of config stats) -> value (None if unsupported)."""
    f = {
        "first_acc": lambda S: _mean(s["first_ok"] for s in S),
        "probe_unique": lambda S: _mean(s["a0"] == A_PROBE for s in S if s["unique"] and s["opt0"] == A_PROBE),
        "gap_regret": lambda S: _mean(s["gap"] for s in S),
        "regret": lambda S: _mean(s["regret"] for s in S),
        "success": lambda S: _mean(s["success"] for s in S),
        "pi_star_success": lambda S: _mean(s["pi_success"] for s in S),
        "unique_first_acc": lambda S: _mean(s["a0"] == s["opt0"] for s in S if s["unique"]),
    }
    for c, acts in CLASSES.items():
        f[f"first_acc_{c}"] = (lambda acts: lambda S: _mean(s["a0"] == s["opt0"] for s in S
                                                            if s["unique"] and s["opt0"] in acts))(acts)
    for c in ("query_first", "after_probe_solved", "after_probe_failed", "after_b1_timeout", "other"):
        f[f"ctx_{c}"] = (lambda c: lambda S: (lambda n, k: k / n if n else None)(
            sum(s["ctx"].get(c, (0, 0))[0] for s in S), sum(s["ctx"].get(c, (0, 0))[1] for s in S)))(c)
    return f


def support(S):
    out = {"configs": len(S), "unique_first": sum(s["unique"] for s in S)}
    for c, acts in CLASSES.items():
        out[f"unique_{c}"] = sum(s["unique"] and s["opt0"] in acts for s in S)
    for c in ("query_first", "after_probe_solved", "after_probe_failed", "after_b1_timeout", "other"):
        out[f"ctx_{c}"] = sum(s["ctx"].get(c, (0, 0))[0] for s in S)
    return out


def boot_ci(stats_list, fn, n_boot=N_BOOT, seed=0, paired=None):
    """Percentile CI of fn over configurations (resampled jointly across the aligned lists in stats_list; paired:
    fn(list_a) - fn(list_b))."""
    rng = random.Random(seed)
    n = len(stats_list[0])
    vals = []
    for _ in range(n_boot):
        ix = [rng.randrange(n) for _ in range(n)]
        if paired:
            a = fn([stats_list[0][i] for i in ix])
            b = fn([stats_list[1][i] for i in ix])
            v = None if a is None or b is None else a - b
        else:
            v = fn([stats_list[0][i] for i in ix])
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def groups_of(rows):
    """{group: {cfg key: row}} keyed by (split, cfg_idx); held-out subsets from the label job's s0 facts."""
    g = {}
    for r in rows:
        key = (r["split"], r["cfg_idx"], r["rep"])
        g.setdefault(r["split"], {})[key] = r
        if r.get("eligible"):
            g.setdefault(r["split"] + ":eligible", {})[key] = r
        for sub in ("all_any_relevant", "all_regret_relevant", "joint_flip"):
            if r.get(sub):
                g.setdefault(r["split"] + ":" + sub, {})[key] = r
    return g


# ------------------------------------------------------------------------------------------------ counterfactual

def cf_units(cf, h_filter=None):
    """One unit per (family, set, decision type) with its per-member correctness (for clustered resampling: units of
    the same set are resampled together through the set key)."""
    units = []
    for fam, sets in cf["families"].items():
        for s in sets:
            for h, t in s["types"].items():
                if h_filter and h != h_filter:
                    continue
                full = t["members"].get(fam)
                if full is None:
                    continue
                units.append({"set": (fam, s["index"]), "fam": fam, "h": h, "flip": t["flip"] and t["unique"],
                              "full_ok": full["ok"], "all_ok": all(m["ok"] for m in t["members"].values()),
                              "nm_ok": None if t["near_miss"] is None else t["near_miss"]["ok"],
                              "pairs": _pairs(fam, t["members"])})
    return units


def _pairs(fam, mem):
    out = []
    for key, m in mem.items():
        for f in fam:
            if f in key:
                continue
            on = "".join(x for x in fam if x in key or x == f)
            m2 = mem.get(on)
            if m2 is None or not (m["unique"] and m2["unique"]):
                continue
            opt_change = m["opt"] != m2["opt"]
            out.append({"f": f, "opt_change": opt_change, "model_change": m["a"] != m2["a"],
                        "new_ok": m2["ok"]})
    return out


def cf_funcs():
    return {
        "cf_flip_full_acc": lambda U: _mean(u["full_ok"] for u in U if u["flip"]),
        "cf_flip_all_members_acc": lambda U: _mean(u["all_ok"] for u in U if u["flip"]),
        "cf_near_miss_acc": lambda U: _mean(u["nm_ok"] for u in U if u["flip"] and u["nm_ok"] is not None),
        "cf_balanced_acc": lambda U: (lambda a, b: None if a is None or b is None else (a + b) / 2)(
            _mean(u["full_ok"] for u in U if u["flip"] and u["nm_ok"] is not None),
            _mean(u["nm_ok"] for u in U if u["flip"] and u["nm_ok"] is not None)),
        "cf_nonflip_full_acc": lambda U: _mean(u["full_ok"] for u in U if not u["flip"]),
        "iv_change_given_opt_change": lambda U: _mean(p["model_change"] for u in U for p in u["pairs"] if p["opt_change"]),
        "iv_change_given_opt_same": lambda U: _mean(p["model_change"] for u in U for p in u["pairs"] if not p["opt_change"]),
        "iv_new_ok_given_opt_change": lambda U: _mean(p["new_ok"] for u in U for p in u["pairs"] if p["opt_change"]),
    }


def cf_support(U):
    return {"units": len(U), "flip": sum(u["flip"] for u in U),
            "flip_with_near_miss": sum(u["flip"] and u["nm_ok"] is not None for u in U),
            "iv_opt_change": sum(p["opt_change"] for u in U for p in u["pairs"]),
            "iv_opt_same": sum(not p["opt_change"] for u in U for p in u["pairs"])}


def _cluster(U):
    """Group units by set so the bootstrap resamples whole sets."""
    by = {}
    for u in U:
        by.setdefault(u["set"], []).append(u)
    keys = sorted(by)
    return [by[k] for k in keys], keys


def _flat(fn):
    return lambda clusters: fn([u for c in clusters for u in c])


# ------------------------------------------------------------------------------------------------ scoring

def score(arms: dict, ref: str | None = None, n_boot: int = N_BOOT, eval_name: str = "eval_episodes.jsonl.gz",
          cf_name: str = "cf_eval.json") -> dict:
    ep_f = endpoint_funcs()
    runs = {}
    for arm, dirs in arms.items():
        for d in dirs:
            head, rows = read_rows(d, eval_name)
            runs[(arm, d)] = {"head": head, "groups": groups_of(rows), "cf": read_cf(d, cf_name)}
    out = {"version": "bscore-v1", "n_boot": n_boot, "min_support": MIN_SUPPORT, "arms": {}, "paired": {}}
    for arm, dirs in arms.items():
        A = out["arms"][arm] = {"runs": [], "groups": {}}
        for j, d in enumerate(dirs):
            r = runs[(arm, d)]
            A["runs"].append({"dir": str(d), "seed": r["head"]["seed"], "train_split": r["head"].get("train_split"),
                              "inputs": r["head"].get("inputs"), "arch": r["head"].get("arch")})
            for g, rows in r["groups"].items():
                keys = sorted(rows)
                S = [config_stats(rows[k]) for k in keys]
                ent = A["groups"].setdefault(g, {"per_seed": [], "support": support(S)})
                vals = {}
                for name, fn in ep_f.items():
                    v = fn(S)
                    vals[name] = {"value": v, "ci": boot_ci([S], fn, n_boot, seed=j) if v is not None else None}
                vals["success_floor"] = {"value": (vals["success"]["value"] >= 0.8 * vals["pi_star_success"]["value"])
                                         if vals["success"]["value"] is not None else None}
                ent["per_seed"].append({"seed": r["head"]["seed"], **vals})
            if r["cf"] is not None:
                for h in [None] + list(r["cf"]["_meta"]["decision_types"]):
                    U = cf_units(r["cf"], h)
                    if not U:
                        continue
                    clusters, _ = _cluster(U)
                    g = "cf:" + (h or "pooled")
                    ent = A["groups"].setdefault(g, {"per_seed": [], "support": cf_support(U)})
                    vals = {}
                    for name, fn in cf_funcs().items():
                        v = fn(U)
                        vals[name] = {"value": v, "ci": boot_ci([clusters], _flat(fn), n_boot, seed=j) if v is not None else None}
                    ent["per_seed"].append({"seed": r["head"]["seed"], **vals})
        for g, ent in A["groups"].items():
            ent["mean"] = {}
            for name in ent["per_seed"][0]:
                if name == "seed":
                    continue
                xs = [p[name]["value"] for p in ent["per_seed"] if p[name]["value"] is not None]
                ent["mean"][name] = (sum(xs) / len(xs)) if xs and not isinstance(xs[0], bool) else (
                    sum(xs) if xs else None)
            ent["insufficient_support"] = _insufficient(ent["support"])
    if ref:
        for arm in arms:
            if arm == ref:
                continue
            out["paired"][f"{arm}-{ref}"] = paired(arms[arm], arms[ref], runs, arm, ref, n_boot)
    return out


def _insufficient(sup):
    return sorted(k for k, v in sup.items() if isinstance(v, int) and v < MIN_SUPPORT)


def paired(dirs_a, dirs_b, runs, arm, ref, n_boot):
    """Seed-paired differences (i-th run of arm vs i-th run of ref; same eval configurations)."""
    ep_f = endpoint_funcs()
    cf_f = cf_funcs()
    res = {}
    for j, (da, db) in enumerate(zip(dirs_a, dirs_b)):
        ra, rb = runs[(arm, da)], runs[(ref, db)]
        pair = {"seeds": [ra["head"]["seed"], rb["head"]["seed"]], "groups": {}}
        for g in ra["groups"]:
            if g not in rb["groups"]:
                continue
            keys = sorted(set(ra["groups"][g]) & set(rb["groups"][g]))
            Sa = [config_stats(ra["groups"][g][k]) for k in keys]
            Sb = [config_stats(rb["groups"][g][k]) for k in keys]
            ent = {}
            for name, fn in ep_f.items():
                a, b = fn(Sa), fn(Sb)
                if a is None or b is None:
                    continue
                ent[name] = {"diff": a - b, "ci": boot_ci([Sa, Sb], fn, n_boot, seed=100 + j, paired=True)}
            pair["groups"][g] = ent
        if ra["cf"] is not None and rb["cf"] is not None:
            for h in [None] + list(ra["cf"]["_meta"]["decision_types"]):
                Ua, Ub = cf_units(ra["cf"], h), cf_units(rb["cf"], h)
                if not Ua:
                    continue
                ca, ka = _cluster(Ua)
                cb, kb = _cluster(Ub)
                assert ka == kb, "counterfactual sets differ between runs"
                ent = {}
                for name, fn in cf_f.items():
                    a, b = fn(Ua), fn(Ub)
                    if a is None or b is None:
                        continue
                    ent[name] = {"diff": a - b, "ci": boot_ci([ca, cb], _flat(fn), n_boot, seed=200 + j, paired=True)}
                pair["groups"]["cf:" + (h or "pooled")] = ent
        res[f"pair{j}"] = pair
    return res


def markdown(res, groups=None, names=("first_acc", "unique_first_acc", "probe_unique", "gap_regret", "success")):
    lines = []
    for arm, A in res["arms"].items():
        for g, ent in A["groups"].items():
            if groups and g not in groups:
                continue
            cells = []
            for n in names + ("cf_flip_full_acc", "cf_balanced_acc", "cf_near_miss_acc"):
                if n in ent["mean"] and ent["mean"][n] is not None:
                    per = "/".join(f"{p[n]['value']:.2f}" if p[n]["value"] is not None else "-" for p in ent["per_seed"])
                    cells.append(f"{n} {per}")
            lines.append(f"| {arm} | {g} | {'; '.join(cells)} |")
    return "\n".join(["| arm | group | per-seed values |", "|---|---|---|"] + lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "RUN"))
    p.add_argument("--ref", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--md", default=None)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--b6c", action="store_true",
                   help="B-FACT-C: read eval_b6c_episodes.jsonl.gz / cf_eval_b6c.json (eval --split-set b6c)")
    a = p.parse_args(argv)
    arms = {x[0]: x[1:] for x in a.arm}
    res = score(arms, a.ref, a.n_boot, *(("eval_b6c_episodes.jsonl.gz", "cf_eval_b6c.json") if a.b6c else ()))
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.md:
        Path(a.md).write_text(markdown(res))


if __name__ == "__main__":
    main()
