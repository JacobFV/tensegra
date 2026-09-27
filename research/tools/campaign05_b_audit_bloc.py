#!/usr/bin/env python3
"""Independent audit (Track B): B-LOC first-consequential-error classification, re-derived from the decision
records (bloc-final/records.jsonl.gz), plus a model-variation control for the 'belief-dependent' class.

Written without reading campaign05_bloc.py.  Classification re-derived from design.md section 2 (B-LOC):
  first consequential error = first decision with delta > eps (.5);
  deployment   : the chosen action was an eps-optimal tie (tie_opt) -- never observed;
  ranking      : the L4 Q-head argmax (over available actions) is eps-optimal but the policy chose otherwise;
  belief-dep.  : a majority (2 of 3) of the oracle-belief BO models choose an eps-optimal action at that same information state;
  value-est.   : the Q-head argmax is not eps-optimal;
  unlocalized  : L1 (no trained Q head) and not belief-dependent.
Pure Python.  Usage: campaign05_b_audit_bloc.py DATA_DIR OUT_JSON
"""
import gzip
import json
import os
import random
import sys
from collections import Counter

EPS = 0.5


def main(data, out):
    recs = [json.loads(x) for x in gzip.open(os.path.join(data, "bloc-final/records.jsonl.gz"), "rt")]
    bloc = json.load(open(os.path.join(data, "bloc-final/bloc.json")))
    res = {"n_records": len(recs)}
    # --- 1. recompute first error, Q-head argmax, BO eps-optimality; find the BO aggregation rule
    agg_rules = {"any": any, "all": all, "majority": lambda v: sum(v) >= 2}
    agree = Counter()
    fe_mismatch = 0
    classes = {}
    by_model = {}
    s0_consistent = {}
    s0_state = {}  # (cfg_idx) -> {model: eps-opt at s0}
    for r in recs:
        ds = r["decisions"]
        t = next((j for j, d in enumerate(ds) if d["delta"] > EPS), None)
        fe = r.get("first_error")
        fe_mismatch += (t is None) != (fe is None) or (fe is not None and fe["t"] != t)
        rung = "L4" if "-L4-" in r["model"] else "L1"
        key = (r["cfg_idx"], r["model"])  # s0 = the initial information state: a function of the configuration only
        s0_consistent.setdefault(key, set()).add(ds[0]["a"])
        s0_state.setdefault(r["cfg_idx"], {})[r["model"]] = ds[0]["a"] in ds[0]["opt"]
        if t is None:
            continue
        d = ds[t]
        qa = max(range(len(d["avail"])), key=lambda j: d["qhat"][j])
        q_eps = d["avail"][qa] in d["opt"]
        bo = [a in d["opt"] for a in fe["bo_argmax"]]
        for name, f in agg_rules.items():
            agree[name] += f(bo) == fe["bo_eps_opt"]
        mine = ("deployment" if d.get("tie_opt") else
                "ranking" if rung == "L4" and q_eps else
                "belief_dependent" if sum(bo) >= 2 else  # majority of the 3 BO models (matches the records)
                "value_estimate" if rung == "L4" else "unlocalized")
        classes.setdefault(r["model"], Counter())[mine] += 1
        by_model.setdefault(r["model"], []).append((r, t, mine, fe["class"], bo))
    res["first_error_index_mismatch"] = fe_mismatch
    res["s0_action_constant_within_model_config"] = all(len(v) == 1 for v in s0_consistent.values())
    res["bo_aggregation_agreement"] = dict(agree)
    n_err = sum(len(v) for v in by_model.values())
    res["class_agreement_with_records"] = sum(m == c for v in by_model.values() for _, _, m, c, _ in v) / n_err
    # compare to bloc.json per-model counts
    diff = 0
    for m, c in classes.items():
        root = bloc["models"][m]["all"]["first_error_class_counts"]
        diff += sum(abs(root.get(k, 0) - c.get(k, 0)) for k in set(root) | set(c))
    res["class_count_abs_diff_vs_bloc_json"] = diff
    pooled = {}
    for m, c in classes.items():
        rung = "L4" if "-L4-" in m else "L1"
        tot = sum(c.values())
        pooled.setdefault(rung, []).append({k: round(v / tot, 3) for k, v in c.items()} | {"model": m, "n": tot})
    res["per_model_class_fractions"] = pooled
    # --- 2. belief-dependent errors at the first decision: at s0 the exact belief equals the declared prior, which
    #        is already a public input, so BO cannot add belief information there.
    s0b = Counter()
    for v in by_model.values():
        for r, t, m, c, bo in v:
            if m == "belief_dependent":
                s0b["at_s0" if t == 0 else "later"] += 1
    res["belief_dependent_at_first_decision"] = dict(s0b)
    # --- 3. model-variation control at s0: for first errors at t = 0, how often does ANOTHER public-input model
    #        (no belief input; same configuration and world, hence the same initial information state) choose an
    #        eps-optimal first action?  Compare with how often BO (any of 3) does.
    ctrl = {"n_s0_errors": 0, "bo_majority_resolves": 0, "bo_each_resolves": 0.0,
            "other_public_L1_majority3_resolves": 0, "other_public_L1_each_resolves": 0.0}
    rng = random.Random(5)
    for m, v in by_model.items():
        others = sorted(o for o in s0_state[next(iter(s0_state))] if o != m and "-L1-" in o)
        for r, t, mine, c, bo in v:
            if t != 0:
                continue
            ctrl["n_s0_errors"] += 1
            ctrl["bo_majority_resolves"] += sum(bo) >= 2
            ctrl["bo_each_resolves"] += sum(bo) / 3
            pick = rng.sample(others, 3)
            ok = [s0_state[r["cfg_idx"]][o] for o in pick]
            ctrl["other_public_L1_majority3_resolves"] += sum(ok) >= 2
            ctrl["other_public_L1_each_resolves"] += sum(s0_state[r["cfg_idx"]][o] for o in others) / len(others)
    n = ctrl["n_s0_errors"]
    ctrl.update({k + "_rate": round(ctrl[k] / n, 3) for k in ("bo_majority_resolves", "bo_each_resolves",
                                                             "other_public_L1_majority3_resolves",
                                                             "other_public_L1_each_resolves")})
    res["s0_model_variation_control"] = ctrl
    # --- 4. sample of first errors for the report (stratified by class)
    rng = random.Random(11)
    sample = []
    allv = [x for v in by_model.values() for x in v]
    for cls in ("ranking", "value_estimate", "belief_dependent", "unlocalized"):
        pool = [x for x in allv if x[2] == cls]
        for r, t, mine, c, bo in rng.sample(pool, min(4, len(pool))):
            d = r["decisions"][t]
            qa = max(range(len(d["avail"])), key=lambda j: d["qhat"][j])
            sample.append({"model": r["model"], "world_seed": r["world_seed"], "combo": r["combo"], "t": t,
                           "chosen": d["a"], "opt": d["opt"], "delta": round(d["delta"], 3),
                           "qhat_argmax": d["avail"][qa], "bo_argmax": r["first_error"]["bo_argmax"],
                           "belief": d["belief"], "mine": mine, "recorded": c})
    res["sample"] = sample
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "sample"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
