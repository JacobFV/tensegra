"""extended-05 A-PI-T: learned delegation controller (registry A-PI-T). train (numpy) | eval (CPU, chunked) | score.

    python research/tools/campaign05_apit.py train --branch results/e05-hr2/branch \
        --output research/results/campaign-05/a-pi-t/controller.json          # numpy only; registers family + m
    CUDA_VISIBLE_DEVICES= python research/tools/campaign05_apit.py eval --base x1-r0 --condition iid_f0 \
        --chunk 0 1 2 3 --controller research/results/campaign-05/a-pi-t/controller.json --output results/e05-apit/eval
    python research/tools/campaign05_apit.py score --eval results/e05-apit/eval --output results/e05-apit/score.json

Training data (registry A-PI-T): the A-HR2 branch labels (option T = delegate to dep_reuse until the next
successful commit or 12 steps, then D) on the 220M screening worlds, lineages x1-r0..r2, restricted to the
trigger anchors {call, reuse_recompute, commit_revise}. Target = the delegate advantage dU(delegate) - q_d.
(The A-HR single-step labels carry no delegate option; their main lines were checked identical to A-HR2's
in the A-HR2 analysis pairing.) Model families (fixed here, before any evaluation): per-anchor ridge on the
G1b public features with lambda in {1 (the G1b value), 10, 100}, and a small per-anchor tanh MLP (16 hidden,
standardized inputs, L2 1e-3, 400 full-batch Adam steps, seed 0). Selection (development only): 5 outer
folds by WORLD (the same world across lineages is one unit); out-of-fold predictions for every family. The
PRIMARY criterion for (family, m) is the G1b single-deviation value (the estimator that passed the A-HR2 gate):
per held-out episode, the TRUE delegate advantage at the first sampled trigger point whose held-out prediction
exceeds m (0 if none), averaged over episodes; this is the exact (unbiased) value of a deployable policy
("delegate once, at the first firing sampled point, then D") and so ranks margins by the precision of the
controller where it first fires. Ties prefer the earlier family in FAMILIES and the larger m. The chosen
family is refit on all training points and frozen with m (controller.json, hashed); pi_T then applies the
same m at every eligible decision (multiple delegations). SECONDARY (reported, not used): the inverse-
inclusion-probability-weighted per-episode sum of true delegate advantages at every firing trigger point, a
first-order estimate that ignores overlapping delegations. It was the first draft rule and was replaced
before any evaluation because it overcounts (it credits ~.3 utility per episode at m = 0, an order of
magnitude above the always-teacher vs D gap) and therefore always selects m = 0 (delegate almost always).

Evaluation worlds (research/campaigns/extended-05/seed-ranges.json; disjointness asserted at start):
- ``--worlds apit`` (default for eval): 240,000,000 + 100,000*i + n, i = condition index in CONDITIONS
  (iid_f0 0, iid_f2 1 = the IID group; events_train_kinds_p1 2, foreign4 3 = secondary); 512 per condition
  = chunks 0..15 of 32; the same worlds for every base (address namespace e05apit).
- ``--worlds dev``: 2,250,500,000 + 100,000*i + n (non-protocol smokes/tests; namespace e05apit-dev).
Bases x1-r0..r2 only (r3-r5 are reserved for confirmation and refused). Device: CPU, 1 thread.

Per chunk and base, on the same worlds with one shared exact solver cache: D; pi_T (the controller at its
registered m); R1 (delegate at every eligible call anchor); R2 (every eligible commit_revise anchor); random
delegation at pi_T's matched per-eligible-decision rate on the same chunk (seeded per world and base); and,
for base x1-r0 only (``--teacher auto``; the teacher is lineage-independent), always-teacher = the dep_reuse
reference evaluation (``run_episode(world, DepReference('reuse'), model_compute_tariff=1.0)``). Delegation
regret is branch-evaluated on a 1/16 world subsample (hash 'e05-apit-regret'; the same worlds for every base):
at every eligible pi_T decision (max 16 per episode) the other choice is branched and continued with pi_T.
Outputs never overwrite (refuse): <stem>.json.gz (episode rows, regret rows) and <stem>.meta.json.

A-PI-C (registry A-PI-C; delegation cost c per teacher-controlled step; additive -- every A-PI-T/A-CF-T
command, default and output above is unchanged):

    python research/tools/campaign05_apit.py train --branch results/e05-hr2/branch --delegation-cost 0.001 \
        --output research/results/campaign-05/a-pi-c/controller-c0.001.json
    CUDA_VISIBLE_DEVICES= python research/tools/campaign05_apit.py eval --base x1-r0 --condition iid_f0 \
        --worlds apic --chunk 0 ... 15 --cost-controllers 0.001=<controller-c0.001.json> \
        0.003=<controller-c0.003.json> --output results/e05-apic/eval
    python research/tools/campaign05_apit.py score --cost --eval results/e05-apic/eval --output ...

- train --delegation-cost c: the SAME points, families, folds, m grid and G1b single-deviation selection
  rule as A-PI-T; only the target changes, exactly: (dU(delegate) - c x teacher_steps) - q_d, where
  teacher_steps is the delegate branch's teacher step count RECORDED per branch by the A-HR2 labeller
  (campaign05_options.run_branches: g["steps"], option field ``teacher_steps``; D branches have none).
  A delegate option without a recorded count is refused (no recomputation is needed or done). The
  controller dict gains ``delegation_cost`` (so its hash differs from A-PI-T's); c in DELEGATION_COSTS
  unless --dev.
- eval --cost-controllers C=PATH ...: one job evaluates every registered cost on the same worlds (one
  shared solver cache). D, R1, R2 and always-teacher do not depend on c (the cost is external; it never
  changes behaviour), so they run ONCE and their rows are charged per c; pi_T (the c controller) and
  random (matched to that pi_T's rate on the chunk) run per c. Every output row carries delegation_cost,
  delegation_charge (= c x teacher_steps: D 0; R1/R2/random/pi_T their teacher steps; always-teacher every
  step of the episode) and utility_gross; ``utility`` is net. Delegation regret (1/16 worlds) per c, net.
  Stems ``apic-<base>-<condition>-<worlds>-c<chunk>`` (never read by the A-PI-T scorer).
  World kinds: ``apic`` = 270,000,000 + 100,000*i (screen; x1-r0..r2 only) and ``acfc`` = 280,000,000 +
  100,000*i (confirmation, frozen controllers; x1-r3..r5 only), 512 per condition, own address namespaces
  (e05apic / e05acfc), each inside its own registered seed-range entry ("A-PI-C" / "A-CF-C"); protocol
  worlds require protocol controller artifacts whose delegation_cost equals their key; ``--cost-controllers``
  is refused on apit/acf worlds and required on apic/acfc. --teacher auto = x1-r0 / x1-r3.
- score --cost: per cost and lineage, PI-C-2 (primary): utility(pi_T) >= max(D, always-teacher, R1, R2,
  random-matched) + .005 AND success(pi_T) >= success(D) - .01 on the IID group; the registered claim needs
  3/3 lineages. PI-C-1 (reported): utility >= D + .015 and success >= D - .01. Per-condition readings,
  delegated-step fractions, delegation charges, world-clustered CIs.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
from functools import partial  # noqa: E402
import gzip  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONDITIONS = ("iid_f0", "iid_f2", "events_train_kinds_p1", "foreign4")
IID = ("iid_f0", "iid_f2")
WORLD_BASES = {"apit": 240_000_000, "acf": 260_000_000, "dev": 2_250_500_000,  # acf: A-CF-T sealed confirmation (r3-r5 only)
               "apic": 270_000_000, "acfc": 280_000_000}   # A-PI-C screen (r0-r2) / A-CF-C confirmation (r3-r5)
STRIDE = 100_000
NAMESPACES = {"apit": "e05apit", "acf": "e05acf", "dev": "e05apit-dev", "apic": "e05apic", "acfc": "e05acfc"}
PROTOCOL_KINDS = ("apit", "acf", "apic", "acfc")
COST_KINDS = ("apic", "acfc")
CONFIRMATION_KINDS = ("acf", "acfc")
OWN_ENTRY = {"apit": "A-PI-T", "acf": "A-CF-T", "apic": "A-PI-C", "acfc": "A-CF-C"}
CHUNK = 32
PER_CONDITION = 512
SCREEN_BASES = ("x1-r0", "x1-r1", "x1-r2")
RESERVED_FOR_CONFIRMATION = ("x1-r3", "x1-r4", "x1-r5")
TRAIN_VERSION = "e05-apit-train-v1"
FAMILIES = ({"name": "ridge_l1", "kind": "ridge", "lam": 1.0}, {"name": "ridge_l10", "kind": "ridge", "lam": 10.0},
            {"name": "ridge_l100", "kind": "ridge", "lam": 100.0},
            {"name": "mlp_h16", "kind": "mlp", "hidden": 16, "l2": 1e-3, "epochs": 400, "lr": 1e-2, "seed": 0})
M_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
PI_T_1 = {"utility_margin": 0.015, "success_tolerance": -0.01}
PI_T_2 = {"utility_margin": 0.005}
PI_C_1 = {"utility_margin": 0.015, "success_tolerance": -0.01}   # reported (vs D)
PI_C_2 = {"utility_margin": 0.005, "success_tolerance": -0.01}   # primary, per cost, 3/3 lineages
DEFAULT_REGRET_P = 1 / 16


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HR = {}


def hr():
    """research/tools/campaign05_hr.py (bases, loader, world construction, seed ranges, bootstrap CI)."""
    if not _HR:
        _HR["m"] = _load("campaign05_hr")
    return _HR["m"]


# --- seed ranges ------------------------------------------------------------------------

def check_seed_ranges(path=None):
    """Pairwise disjointness of every registered range (campaign05_hr.check_seed_ranges) and this tool's
    world ranges inside a registered extended-05 entry; the protocol range must be its own entry."""
    h = hr()
    rs = h.check_seed_ranges(path or h.SEED_RANGES)
    for kind, base in WORLD_BASES.items():
        lo, hi = base, base + STRIDE * len(CONDITIONS)
        inside = [r for r in rs if r["campaign"] == "extended-05" and r["lo"] <= lo and hi <= r["hi"]]
        if not inside:
            raise ValueError(f"{kind} worlds [{lo}, {hi}) are not inside a registered extended-05 range")
        if kind in OWN_ENTRY and not any(OWN_ENTRY[kind] in r["name"] for r in inside):
            raise ValueError(f"the {OWN_ENTRY[kind]} range ({kind}) must be registered under its own entry")
    return rs


def world_seeds(kind, condition, chunk, chunk_size, episodes=None):
    i = CONDITIONS.index(condition)
    if chunk < 0 or (chunk + 1) * chunk_size > (PER_CONDITION if kind in PROTOCOL_KINDS else STRIDE):
        raise SystemExit("chunk outside the registered range")
    first = WORLD_BASES[kind] + STRIDE * i + chunk * chunk_size
    n = chunk_size if episodes is None else min(episodes, chunk_size)
    return list(range(first, first + n))


# --- training (numpy) ----------------------------------------------------------------------

def training_points(dirs, dev=False, cost=None):
    """[(x, adv, weight, world key, episode key, anchor, sampled)] from A-HR2 multi branch files, and the
    episode keys (for per-episode normalization) and file hashes. ``cost`` (A-PI-C): target
    (dU - cost x recorded teacher_steps) - q_d (``cost_adjusted_label``); None = A-PI-T (dU - q_d)."""
    from tensegra.campaign05_apit import TRIGGERS, cost_adjusted_label, feature_vector
    h = hr()
    rows, episodes, files = [], {}, {}
    for d in dirs:
        for p in sorted(Path(d).glob("branch-*.json.gz")):
            files[p.name] = h.file_hash(p)
    for meta, data in h._read(dirs, "branch"):
        if meta.get("option_class") != "multi" or (meta.get("worlds") != "hr" and not dev):
            raise SystemExit("training expects A-HR2 (--option-class multi) branch files on the 220M hr worlds")
        if meta["base"] not in SCREEN_BASES and not dev:
            raise SystemExit(f"{meta['base']} is not a screening lineage")
        for e in data["episodes"]:
            episodes[(meta["base"], meta["condition"], e["seed"])] = h._world(meta["condition"], e["seed"])
        for p in data["points"]:
            if p["anchor"] not in TRIGGERS:
                continue
            dl = [o for o in p["options"] if o["type"] == "delegate"]
            if not dl:
                continue
            o = dl[0]
            if cost is None:
                y = o["dU"] - p["q_d"]
            else:
                if "teacher_steps" not in o:
                    raise SystemExit("delegate option without a recorded teacher step count: A-PI-C needs A-HR2 "
                                     "branch files written by run_branches (teacher_steps per branch)")
                y = cost_adjusted_label(o["dU"], p["q_d"], o["teacher_steps"], cost)
            rows.append({"x": feature_vector(p["telemetry"], o.get("features") or {}), "y": y,
                         "w": 1.0 / p["p_include"], "world": h._world(meta["condition"], p["seed"]),
                         "episode": (meta["base"], meta["condition"], p["seed"]), "anchor": p["anchor"],
                         "sampled": p["sampled"], "step": p["step"],
                         **({"teacher_steps": o["teacher_steps"], "y_free": o["dU"] - p["q_d"], "dU": o["dU"], "q_d": p["q_d"]}
                            if cost is not None else {})})
    return rows, episodes, files


def fit_family(fam, rows):
    import numpy as np
    from tensegra.campaign05_apit import TRIGGERS, feature_names, fit_mlp, fit_ridge
    models = {}
    for a in TRIGGERS:
        sub = [r for r in rows if r["anchor"] == a]
        if not sub:   # (only possible in tiny smokes) predict 0 = never delegate at m >= 0
            models[a] = {"kind": "ridge", "lam": fam.get("lam", 1.0), "w": [0.0] * len(feature_names()),
                         "fallback": "no training points"}
            continue
        X, y = np.array([r["x"] for r in sub]), np.array([r["y"] for r in sub])
        if fam["kind"] == "ridge":
            models[a] = fit_ridge(X, y, fam["lam"])
        else:
            models[a] = fit_mlp(X, y, fam["hidden"], fam["l2"], fam["epochs"], fam["lr"], fam["seed"])
    return models


def predict_rows(models, rows):
    import numpy as np
    from tensegra.campaign05_apit import predict_model
    out = np.zeros(len(rows))
    for a, m in models.items():
        idx = [i for i, r in enumerate(rows) if r["anchor"] == a]
        if idx:
            out[idx] = predict_model(m, [rows[i]["x"] for i in idx])
    return out


def objective(rows, pred, m, episodes):
    """Per-episode IPW sum of true delegate advantages at firing points (pred > m); world-clustered CI."""
    per = {k: 0.0 for k in episodes}
    fires = 0.0
    for r, p in zip(rows, pred):
        if p > m:
            per[r["episode"]] += r["w"] * r["y"]
            fires += r["w"]
    keys = sorted(per)
    ci = hr()._mean_ci([per[k] for k in keys], [episodes[k] for k in keys])
    return {**ci, "ipw_delegations_per_episode": fires / max(1, len(keys))}


def first_firing(rows, pred, m, episodes):
    """G1b-style secondary criterion: exact single-deviation value (first firing SAMPLED trigger point)."""
    first = {}
    for r, p in sorted(zip(rows, pred), key=lambda t: (t[0]["episode"], t[0]["step"])):
        if r["sampled"] and p > m and r["episode"] not in first:
            first[r["episode"]] = r["y"]
    keys = sorted(episodes)
    return hr()._mean_ci([first.get(k, 0.0) for k in keys], [episodes[k] for k in keys])


def cmd_train(a):
    import numpy as np
    from tensegra.campaign05_apit import (APIT_VERSION, DELEGATION_COSTS, FEATURE_VERSION, TRIGGERS, canonical_hash,
                                          check_cost, feature_names, model_parameters)
    t0, c0 = time.perf_counter(), time.process_time()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    cost = getattr(a, "delegation_cost", None)
    if cost is not None:
        cost = check_cost(cost)
        if cost not in DELEGATION_COSTS and not a.dev:
            raise SystemExit(f"delegation cost {cost} is not registered (A-PI-C: {DELEGATION_COSTS})")
    rows, episodes, files = training_points(a.branch, a.dev, cost)
    if not rows:
        raise SystemExit("no training points")
    h = hr()
    fold = h._folds(list(episodes.values()), a.folds, a.seed)
    oof = {}
    mse = {}
    for fam in FAMILIES:
        pred = np.zeros(len(rows))
        for f in range(a.folds):
            train = [r for r in rows if fold[r["world"]] != f]
            test = [i for i, r in enumerate(rows) if fold[r["world"]] == f]
            if not test:
                continue
            models = fit_family(fam, train)
            pred[test] = predict_rows(models, [rows[i] for i in test])
        oof[fam["name"]] = pred
        y = np.array([r["y"] for r in rows])
        mse[fam["name"]] = {"all": float(np.mean((pred - y) ** 2)),
                            **{an: (float(np.mean(v)) if (v := [(pred[i] - rows[i]["y"]) ** 2 for i in range(len(rows))
                                                                 if rows[i]["anchor"] == an]) else None)
                               for an in TRIGGERS}}
    y = np.array([r["y"] for r in rows])
    mse["constant_zero"] = {"all": float(np.mean(y ** 2))}
    table, secondary = {}, {}
    for fam in FAMILIES:
        table[fam["name"]] = {str(m): first_firing(rows, oof[fam["name"]], m, episodes) for m in M_GRID}
        secondary[fam["name"]] = {str(m): objective(rows, oof[fam["name"]], m, episodes) for m in M_GRID}
    order = {f["name"]: i for i, f in enumerate(FAMILIES)}
    best = max(((f, m) for f in order for m in M_GRID),
               key=lambda fm: (round(table[fm[0]][str(fm[1])]["mean"], 12), -order[fm[0]], fm[1]))
    fam = next(f for f in FAMILIES if f["name"] == best[0])
    models = fit_family(fam, rows)
    chosen_oof = oof[fam["name"]]
    controller = {"version": APIT_VERSION, "feature_version": FEATURE_VERSION, "feature_names": feature_names(),
                  "triggers": list(TRIGGERS), "family": fam, "margin": best[1], "models": models,
                  "target": "delegate advantage dU(delegate) - q_d (A-HR2 option T; true-world branch labels)"}
    if cost is not None:
        controller["delegation_cost"] = cost
        controller["target"] = ("cost-adjusted delegate advantage (dU(delegate) - c x teacher_steps) - q_d, c = "
                                f"{cost} per teacher-controlled step (A-PI-C; A-HR2 option T; true-world branch labels)")
    n_params = sum(model_parameters(m) for m in models.values())
    by_anchor = {an: {"points": sum(r["anchor"] == an for r in rows),
                      "oof_fire_rate": (float(np.mean(v)) if (v := [p > best[1] for p, r in zip(chosen_oof, rows)
                                                                    if r["anchor"] == an]) else None),
                      "mean_adv": (float(np.mean(v)) if (v := [r["y"] for r in rows if r["anchor"] == an]) else None),
                      "mean_adv_where_fired_oof": (float(np.mean(fired)) if (fired := [
                          r["y"] for p, r in zip(chosen_oof, rows) if r["anchor"] == an and p > best[1]]) else None)}
                 for an in TRIGGERS}
    artifact = {
        "controller": controller, "controller_sha256": canonical_hash(controller),
        "protocol": not a.dev,
        "n_parameters": n_params,
        "charge_rule": "each controller evaluation is charged neural_work_per_forward x n_parameters / actor "
                       "parameters (computed at evaluation from the frozen actor); each teacher step is charged one "
                       "forward (neural_work_per_forward) like any decision",
        "selection": {
            "version": TRAIN_VERSION, "registered_before_evaluation": True,
            "rule": "max over (family, m) of the out-of-fold (5 world folds) G1b single-deviation value: per "
                    "episode the true delegate advantage at the first sampled trigger point with prediction > m "
                    "(0 if none); ties: earlier family, larger m",
            "rule_history": "first draft (IPW per-episode sum of true delegate advantages at every firing point; "
                            "now 'secondary_objective_table') replaced BEFORE any evaluation: it ignores overlapping "
                            "delegations, credits ~.3/episode at m=0 and always selects m=0",
            "families": list(FAMILIES), "m_grid": list(M_GRID), "folds": a.folds, "fold_seed": a.seed,
            "chosen": {"family": fam["name"], "margin": best[1],
                       "objective": table[fam["name"]][str(best[1])],
                       "secondary_ipw_sum": secondary[fam["name"]][str(best[1])],
                       "by_anchor": by_anchor},
            "objective_table": table, "secondary_objective_table": secondary, "oof_mse": mse},
        "training_data": {"source": "A-HR2 branch labels (e05-hr2/branch; option T), 220M screening worlds, "
                                    "x1-r0..r2, iid_f0/iid_f2", "files": files, "points": len(rows),
                          "episodes": len(episodes), "worlds": len(set(episodes.values())),
                          "points_by_anchor": {an: sum(r["anchor"] == an for r in rows) for an in TRIGGERS},
                          "privilege": "true-world branch labels are training TARGETS only; never inputs"},
        "sources": {"campaign05_apit.py(src)": h.file_hash(REPO / "src/tensegra/campaign05_apit.py"),
                    "campaign05_apit.py(tool)": h.file_hash(Path(__file__)),
                    "campaign05_options.py": h.file_hash(REPO / "src/tensegra/campaign05_options.py"),
                    "campaign04_telemetry.py": h.file_hash(REPO / "src/tensegra/campaign04_telemetry.py")},
        "cpu": {"process_s": time.process_time() - c0, "wall_s": time.perf_counter() - t0}}
    if cost is not None:
        artifact["delegation_cost"] = cost_training_report(rows, chosen_oof, best[1], episodes, cost, table, fam)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(artifact, indent=1))
    print(json.dumps({"output": str(a.output), "controller_sha256": artifact["controller_sha256"],
                      "chosen": {k: v for k, v in artifact["selection"]["chosen"].items()},
                      "n_parameters": n_params, "points": len(rows), "episodes": len(episodes),
                      "objective_by_family_at_best_m": {f: max(t.values(), key=lambda v: v["mean"])["mean"]
                                                        for f, t in table.items()},
                      "oof_mse": mse, "cpu": artifact["cpu"],
                      **({"delegation_cost": {k: v for k, v in artifact["delegation_cost"].items()
                                              if k in ("c", "dev_estimate", "implied_firing")}}
                         if cost is not None else {})}, indent=1))


def cost_training_report(rows, oof, m, episodes, cost, table, fam):
    """A-PI-C artifact block: the exact label adjustment (checked), teacher-step facts, the development
    single-deviation estimate at the chosen (family, m) and the implied firing rates (out-of-fold)."""
    import numpy as np
    ts = [r["teacher_steps"] for r in rows]
    adj = [r["y_free"] - r["y"] for r in rows]
    exact = all(r["y"] == (r["dU"] - cost * r["teacher_steps"]) - r["q_d"]
                and abs((r["y_free"] - r["y"]) - cost * r["teacher_steps"]) <= 1e-12 for r in rows)
    sampled = [(r, p) for r, p in zip(rows, oof) if r["sampled"]]
    first = {}
    for r, p in sorted(zip(rows, oof), key=lambda t: (t[0]["episode"], t[0]["step"])):
        if r["sampled"] and p > m and r["episode"] not in first:
            first[r["episode"]] = r
    return {
        "c": cost, "unit": "external utility per teacher-controlled step",
        "label": "y_c = (dU(delegate) - c x teacher_steps) - q_d; teacher_steps = the delegate branch's recorded "
                 "teacher step count (campaign05_options.run_branches g['steps']; option field teacher_steps); "
                 "the D branch has no teacher steps",
        "teacher_steps_source": "recorded per branch in the A-HR2 branch files (verified present on every delegate "
                                "option used; none recomputed)",
        "label_adjustment_exact": bool(exact),
        "teacher_steps": {"points": len(ts), "mean": float(np.mean(ts)), "min": int(min(ts)), "max": int(max(ts))},
        "mean_adjustment": float(np.mean(adj)),
        "mean_label": {"free": float(np.mean([r["y_free"] for r in rows])), "cost": float(np.mean([r["y"] for r in rows]))},
        "dev_estimate": {"family": fam["name"], "margin": m, "g1b_single_deviation": table[fam["name"]][str(m)],
                         "note": "out-of-fold (5 world folds) G1b single-deviation value of the cost-adjusted "
                                 "advantage per episode (development only, 220M training worlds)"},
        "implied_firing": {
            "oof_fire_rate_all_points": float(np.mean([p > m for p in oof])),
            "oof_fire_rate_sampled_points": float(np.mean([p > m for _, p in sampled])) if sampled else None,
            "oof_fire_rate_ipw": float(sum(r["w"] * (p > m) for r, p in zip(rows, oof)) / sum(r["w"] for r in rows)),
            "episodes_with_a_firing_sampled_point": len(first) / max(1, len(episodes)),
            "mean_teacher_steps_where_first_fired": (float(np.mean([r["teacher_steps"] for r in first.values()]))
                                                    if first else None)}}


# --- evaluation (torch) ---------------------------------------------------------------------

def _children_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _stem(a):
    return a.output / f"apit-{a.base}-{a.condition}-{a.worlds}-c{a.chunk:03d}"


def run_chunk(actor, train_cfg, info, controller, seeds, condition, namespace, *, policies, teacher, regret_p,
              regret_check=False, solver_cls=None, base_name=""):
    """All policies on one chunk of worlds (one shared exact solver cache). Returns (rows, regret, stats)."""
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign02_references import make_reference, run_episode
    from tensegra.campaign03_depworld import depworld_executor
    from tensegra.campaign04_branch import SolverCache
    from tensegra.campaign05_apit import Delegator, RegretHook, episode_row, run_delegator, run_regret_branches
    from tensegra.campaign05_options import d_rollouts, HRConfig
    h = hr()
    nwpf = train_cfg.neural_work_per_forward
    cap = train_cfg.max_steps
    units = nwpf * controller.n_parameters / info["parameters"] if controller is not None else 0.0
    rows, regret, stats = [], [], {"cpu_s": {}, "controller_units": units, "matched_rate": None}
    with (solver_cls or BoundedSolver)() as solver:
        cache = SolverCache(partial(depworld_executor, execute_call=solver.execute))
        make = lambda: [h.world(s, condition, cache, namespace) for s in seeds]  # noqa: E731
        order = [p for p in ("d", "pit", "r1", "r2", "random", "always") if p in policies]
        for pol in order:
            c0 = time.process_time()
            if pol == "d":
                eps = d_rollouts(actor, make(), seeds, HRConfig(max_steps=cap), neural_work_per_forward=nwpf)
            else:
                kw = {}
                if pol == "pit":
                    kw = {"controller": controller, "controller_units": units, "record_predictions": True}
                if pol == "random":
                    pit = [r for r in rows if r["policy"] == "pit"]
                    if not pit:
                        raise SystemExit("random needs pi_T on the same chunk (matched rate)")
                    rate = sum(r["delegations"] for r in pit) / max(1, sum(r["eligible"] for r in pit))
                    stats["matched_rate"] = rate
                    kw = {"rate": rate, "rng_key": base_name}
                dl = Delegator(pol, **kw)
                hook = RegretHook(regret_p) if pol == "pit" and regret_p > 0 else None
                eps = run_delegator(actor, make(), seeds, dl, cap=cap, neural_work_per_forward=nwpf, hook=hook)
                if hook is not None and hook.points:
                    rc = time.process_time()
                    regret = run_regret_branches(actor, dl, hook.points, neural_work_per_forward=nwpf,
                                                 check=regret_check)
                    stats["cpu_s"]["regret_branches"] = time.process_time() - rc
                    stats["regret_points"] = len(hook.points)
            rows += [episode_row(ep, pol) for ep in eps]
            stats["cpu_s"][pol] = time.process_time() - c0 - stats["cpu_s"].get("regret_branches", 0.0) * (pol == "pit")
        if teacher:
            c0 = time.process_time()
            for s, env in zip(seeds, make()):
                r = run_episode(env, make_reference("dep_reuse"), model_compute_tariff=nwpf)
                rows.append({"seed": s, "policy": "teacher", "success": bool(r["verified_success"]),
                             "utility": r["utility"], "cost": r["cost"], "steps": r["steps"],
                             "compute_units": r["compute_units"], "decisions": len(r["trace"]),
                             "truncated": False, "unsupported": None, "delegations": 1, "eligible": 1,
                             "teacher_steps": len(r["trace"])})
            stats["cpu_s"]["teacher"] = time.process_time() - c0
        stats["solver_cache"] = cache.stats()
    return rows, regret, stats


def _policy_summary(rows, net=False):
    out = {}
    for pol in sorted(set(r["policy"] for r in rows)):
        rs = [r for r in rows if r["policy"] == pol]
        n = len(rs)
        out[pol] = {"episodes": n, "utility": sum(r["utility"] for r in rs) / n,
                    "success": sum(r["success"] for r in rs) / n, "cost": sum(r["cost"] for r in rs) / n,
                    "delegation_rate_per_eligible": sum(r["delegations"] for r in rs) / max(1, sum(r["eligible"] for r in rs)),
                    "delegations_per_episode": sum(r["delegations"] for r in rs) / n,
                    "delegated_step_fraction": sum(r["teacher_steps"] for r in rs) / max(1, sum(r["decisions"] for r in rs))}
        if net:
            out[pol]["utility_gross"] = sum(r["utility_gross"] for r in rs) / n
            out[pol]["delegation_charge"] = sum(r["delegation_charge"] for r in rs) / n
    return out


def cmd_eval(a):
    import torch
    from tensegra.campaign05_apit import APIT_VERSION, FEATURE_VERSION, Controller
    if a.cost_controllers or a.worlds in COST_KINDS:
        return cmd_eval_cost(a)
    if "pit" in a.policies and a.controller is None:
        raise SystemExit("pi_T needs --controller")
    torch.set_num_threads(a.threads)
    check_seed_ranges()
    h = hr()
    if a.base in RESERVED_FOR_CONFIRMATION and a.binding is None and a.worlds != "acf":
        raise SystemExit(f"{a.base} is reserved for confirmation (design v2 revision 4); A-PI-T screens x1-r0..r2")
    if a.worlds == "acf" and a.base not in RESERVED_FOR_CONFIRMATION:
        raise SystemExit("A-CF-T sealed worlds are for the fresh lineages x1-r3..r5 only")
    binding = h.bases()[a.base] if a.binding is None else a.binding
    actor, train_cfg, info = a.load_actor(binding, verify=not a.no_verify_hash)
    controller = Controller.from_file(a.controller, margin=a.margin_override) if "pit" in a.policies else None
    if controller is not None and a.worlds in ("apit", "acf") and not controller.artifact.get("protocol"):
        raise SystemExit("protocol worlds need the registered (protocol) controller artifact")
    if "random" in a.policies and "pit" not in a.policies:
        raise SystemExit("random is matched to pi_T's rate on the same chunk: include pit")
    teacher = {"auto": a.base in ("x1-r0", "x1-r3"), "yes": True, "no": False}[a.teacher]
    a.output.mkdir(parents=True, exist_ok=True)
    stem = _stem(a)
    if stem.with_suffix(".json.gz").exists():
        raise SystemExit(f"refusing to overwrite {stem}.json.gz")
    seeds = world_seeds(a.worlds, a.condition, a.chunk, a.chunk_size, a.episodes)
    wall0, cpu0, ch0 = time.perf_counter(), time.process_time(), _children_cpu()
    rows, regret, stats = run_chunk(actor, train_cfg, info, controller, seeds, a.condition, NAMESPACES[a.worlds],
                                    policies=a.policies, teacher=teacher, regret_p=a.regret_p,
                                    regret_check=a.regret_check, solver_cls=a.solver, base_name=binding["name"])
    for r in rows:
        r["condition"] = a.condition
    cpu = {"process_s": time.process_time() - cpu0, "children_s": _children_cpu() - ch0,
           "wall_s": time.perf_counter() - wall0}
    total = cpu["process_s"] + cpu["children_s"]
    summary = _policy_summary(rows)
    n = len(seeds)
    unit = {"core_s_per_world_all_policies": total / n,
            **{f"core_s_per_episode_{k}": v / n for k, v in stats["cpu_s"].items() if k != "regret_branches"},
            "solver_children_core_s_per_world": cpu["children_s"] / n}
    if stats.get("regret_points"):
        unit["core_s_per_regret_branch"] = stats["cpu_s"]["regret_branches"] / (
            stats["regret_points"] * (2 if a.regret_check else 1))
    ctrl = None
    if controller is not None:
        ctrl = {"path": str(a.controller), "file_sha256": h.file_hash(a.controller),
                "controller_sha256": controller.artifact["controller_sha256"], "family": controller.family["name"]
                if isinstance(controller.family, dict) else controller.family, "margin": controller.margin,
                "margin_override": a.margin_override, "n_parameters": controller.n_parameters,
                "controller_units_per_evaluation": stats["controller_units"]}
    meta = {"version": APIT_VERSION, "feature_version": FEATURE_VERSION, "command": "eval", "base": binding["name"],
            "base_sha256": binding["sha256"], "condition": a.condition, "worlds": a.worlds,
            "protocol_worlds": a.worlds != "dev", "namespace": NAMESPACES[a.worlds], "chunk": a.chunk,
            "seeds": [seeds[0], seeds[-1] + 1], "device": "cpu", "threads": a.threads, "max_steps": train_cfg.max_steps,
            "neural_work_per_forward": train_cfg.neural_work_per_forward, "actor": info, "controller": ctrl,
            "policies": list(a.policies) + (["teacher"] if teacher else []), "matched_random_rate": stats["matched_rate"],
            "regret": {"p_world": a.regret_p, "points": stats.get("regret_points", 0), "check": a.regret_check},
            "summary": summary, "solver_cache": stats["solver_cache"], "cpu": cpu, "cpu_by_policy_s": stats["cpu_s"],
            "unit_costs": unit, "sources": {**h.source_hashes(), "campaign05_apit(src)":
                                            h.file_hash(REPO / "src/tensegra/campaign05_apit.py"),
                                            "campaign05_apit(tool)": h.file_hash(Path(__file__))}}
    with gzip.open(stem.with_suffix(".json.gz"), "wt") as f:
        json.dump({"rows": rows, "regret": regret}, f, separators=(",", ":"))
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if regret and a.regret_check:
        meta["regret_check_all_equal"] = all(r["check_equal"] for r in regret)
        stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({"stem": str(stem), "summary": summary, "matched_rate": stats["matched_rate"],
                      "regret_points": stats.get("regret_points", 0),
                      **({"regret_check_all_equal": meta["regret_check_all_equal"]} if "regret_check_all_equal" in meta
                         else {}), "unit_costs": unit, "cpu": cpu}, indent=1))


# --- scoring (numpy) ---------------------------------------------------------------------------

# --- A-PI-C evaluation (delegation cost) -------------------------------------------------------

def parse_cost_controllers(items):
    """["0.001=path", ...] -> {cost: Path} (costs distinct)."""
    from tensegra.campaign05_apit import check_cost
    out = {}
    for it in items or []:
        c, sep, path = it.partition("=")
        if not sep or not path:
            raise SystemExit(f"--cost-controllers expects COST=PATH, got {it!r}")
        c = check_cost(c)
        if c in out:
            raise SystemExit(f"cost {c} given twice")
        out[c] = Path(path)
    return dict(sorted(out.items()))


def run_chunk_cost(actor, train_cfg, info, controllers, seeds, condition, namespace, *, teacher, regret_p,
                   regret_check=False, solver_cls=None, base_name=""):
    """A-PI-C: every cost on one chunk of worlds (one shared exact solver cache). D, R1, R2 and the teacher run
    once (the cost is external and never changes their behaviour) and are charged per cost; pi_T (that cost's
    controller) and random (matched to that pi_T's rate on this chunk) run per cost. Returns (rows, regret,
    stats); every row carries delegation_cost / delegation_charge / utility_gross (utility is net)."""
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign02_references import make_reference, run_episode
    from tensegra.campaign03_depworld import depworld_executor
    from tensegra.campaign04_branch import SolverCache
    from tensegra.campaign05_apit import (Delegator, RegretHook, apply_delegation_cost, episode_row, run_delegator,
                                          run_regret_branches)
    from tensegra.campaign05_options import d_rollouts, HRConfig
    h = hr()
    nwpf = train_cfg.neural_work_per_forward
    cap = train_cfg.max_steps
    base_rows, rows, regret = [], [], []
    stats = {"cpu_s": {}, "controller_units": {}, "matched_rate": {}, "regret_points": {}}
    with (solver_cls or BoundedSolver)() as solver:
        cache = SolverCache(partial(depworld_executor, execute_call=solver.execute))
        make = lambda: [h.world(s, condition, cache, namespace) for s in seeds]  # noqa: E731
        for pol in ("d", "r1", "r2"):
            c0 = time.process_time()
            if pol == "d":
                eps = d_rollouts(actor, make(), seeds, HRConfig(max_steps=cap), neural_work_per_forward=nwpf)
            else:
                eps = run_delegator(actor, make(), seeds, Delegator(pol), cap=cap, neural_work_per_forward=nwpf)
            base_rows += [episode_row(ep, pol) for ep in eps]
            stats["cpu_s"][pol] = time.process_time() - c0
        if teacher:
            c0 = time.process_time()
            for s, env in zip(seeds, make()):
                r = run_episode(env, make_reference("dep_reuse"), model_compute_tariff=nwpf)
                base_rows.append({"seed": s, "policy": "teacher", "success": bool(r["verified_success"]),
                                  "utility": r["utility"], "cost": r["cost"], "steps": r["steps"],
                                  "compute_units": r["compute_units"], "decisions": len(r["trace"]),
                                  "truncated": False, "unsupported": None, "delegations": 1, "eligible": 1,
                                  "teacher_steps": len(r["trace"])})   # always-teacher: every step is a teacher step
            stats["cpu_s"]["teacher"] = time.process_time() - c0
        for c, controller in controllers.items():
            key = str(c)
            rows += [apply_delegation_cost(r, c) for r in base_rows]
            units = nwpf * controller.n_parameters / info["parameters"]
            stats["controller_units"][key] = units
            c0 = time.process_time()
            dl = Delegator("pit", controller=controller, controller_units=units, record_predictions=True,
                           delegation_cost=c)
            hook = RegretHook(regret_p) if regret_p > 0 else None
            eps = run_delegator(actor, make(), seeds, dl, cap=cap, neural_work_per_forward=nwpf, hook=hook)
            pit = [episode_row(ep, "pit") for ep in eps]
            rows += pit
            rc = time.process_time()
            if hook is not None and hook.points:
                regret += run_regret_branches(actor, dl, hook.points, neural_work_per_forward=nwpf, check=regret_check)
                stats["regret_points"][key] = len(hook.points)
            stats["cpu_s"][f"regret_branches@{key}"] = time.process_time() - rc
            stats["cpu_s"][f"pit@{key}"] = rc - c0
            c0 = time.process_time()
            rate = sum(r["delegations"] for r in pit) / max(1, sum(r["eligible"] for r in pit))
            stats["matched_rate"][key] = rate
            eps = run_delegator(actor, make(), seeds, Delegator("random", rate=rate, rng_key=base_name,
                                                                delegation_cost=c),
                                cap=cap, neural_work_per_forward=nwpf)
            rows += [episode_row(ep, "random") for ep in eps]
            stats["cpu_s"][f"random@{key}"] = time.process_time() - c0
        stats["solver_cache"] = cache.stats()
    return rows, regret, stats


def cmd_eval_cost(a):
    import torch
    from tensegra.campaign05_apit import APIT_VERSION, DELEGATION_COSTS, FEATURE_VERSION, Controller
    if a.worlds in ("apit", "acf"):
        raise SystemExit("--cost-controllers are for the A-PI-C worlds (apic/acfc) or dev only")
    if not a.cost_controllers:
        raise SystemExit(f"{a.worlds} worlds are A-PI-C worlds: --cost-controllers COST=PATH ... is required")
    if a.controller is not None:
        raise SystemExit("A-PI-C evaluation takes --cost-controllers, not --controller")
    if list(a.policies) != ["d", "pit", "r1", "r2", "random"]:
        raise SystemExit("A-PI-C evaluation runs the fixed set D, pi_T, R1, R2, random (+ teacher)")
    torch.set_num_threads(a.threads)
    check_seed_ranges()
    h = hr()
    if a.base in RESERVED_FOR_CONFIRMATION and a.binding is None and a.worlds != "acfc":
        raise SystemExit(f"{a.base} is reserved for confirmation; the A-PI-C screen uses x1-r0..r2")
    if a.worlds == "acfc" and a.base not in RESERVED_FOR_CONFIRMATION:
        raise SystemExit("A-CF-C sealed worlds are for the fresh lineages x1-r3..r5 only")
    paths = parse_cost_controllers(a.cost_controllers)
    protocol = a.worlds in PROTOCOL_KINDS
    if protocol and set(paths) != set(DELEGATION_COSTS):
        raise SystemExit(f"protocol A-PI-C worlds evaluate exactly the registered costs {DELEGATION_COSTS}")
    controllers = {}
    for c, path in paths.items():
        ctrl = Controller.from_file(path, margin=a.margin_override)
        if ctrl.delegation_cost != c:
            raise SystemExit(f"{path}: controller trained under delegation cost {ctrl.delegation_cost}, keyed {c}")
        if protocol and not ctrl.artifact.get("protocol"):
            raise SystemExit("protocol worlds need the registered (protocol) controller artifacts")
        controllers[c] = ctrl
    binding = h.bases()[a.base] if a.binding is None else a.binding
    actor, train_cfg, info = a.load_actor(binding, verify=not a.no_verify_hash)
    teacher = {"auto": a.base in ("x1-r0", "x1-r3"), "yes": True, "no": False}[a.teacher]
    a.output.mkdir(parents=True, exist_ok=True)
    stem = a.output / f"apic-{a.base}-{a.condition}-{a.worlds}-c{a.chunk:03d}"
    if stem.with_suffix(".json.gz").exists():
        raise SystemExit(f"refusing to overwrite {stem}.json.gz")
    seeds = world_seeds(a.worlds, a.condition, a.chunk, a.chunk_size, a.episodes)
    wall0, cpu0, ch0 = time.perf_counter(), time.process_time(), _children_cpu()
    rows, regret, stats = run_chunk_cost(actor, train_cfg, info, controllers, seeds, a.condition,
                                         NAMESPACES[a.worlds], teacher=teacher, regret_p=a.regret_p,
                                         regret_check=a.regret_check, solver_cls=a.solver, base_name=binding["name"])
    for r in rows:
        r["condition"] = a.condition
    for r in regret:
        r["condition"] = a.condition
    cpu = {"process_s": time.process_time() - cpu0, "children_s": _children_cpu() - ch0,
           "wall_s": time.perf_counter() - wall0}
    n = len(seeds)
    summary = {str(c): _policy_summary([r for r in rows if r["delegation_cost"] == c], net=True) for c in controllers}
    unit = {"core_s_per_world_all_policies_all_costs": (cpu["process_s"] + cpu["children_s"]) / n,
            **{f"core_s_per_episode_{k}": v / n for k, v in stats["cpu_s"].items()},
            "solver_children_core_s_per_world": cpu["children_s"] / n}
    ctrl = {str(c): {"path": str(paths[c]), "file_sha256": h.file_hash(paths[c]),
                     "controller_sha256": x.artifact["controller_sha256"], "delegation_cost": x.delegation_cost,
                     "family": x.family["name"] if isinstance(x.family, dict) else x.family, "margin": x.margin,
                     "margin_override": a.margin_override, "n_parameters": x.n_parameters,
                     "controller_units_per_evaluation": stats["controller_units"][str(c)]}
            for c, x in controllers.items()}
    meta = {"version": APIT_VERSION, "apic": "A-PI-C delegation cost (external utility per teacher-controlled step)",
            "feature_version": FEATURE_VERSION, "command": "eval", "base": binding["name"],
            "base_sha256": binding["sha256"], "condition": a.condition, "worlds": a.worlds,
            "protocol_worlds": protocol, "namespace": NAMESPACES[a.worlds], "chunk": a.chunk,
            "seeds": [seeds[0], seeds[-1] + 1], "device": "cpu", "threads": a.threads, "max_steps": train_cfg.max_steps,
            "neural_work_per_forward": train_cfg.neural_work_per_forward, "actor": info,
            "delegation_costs": [float(c) for c in controllers], "controllers": ctrl,
            "policies": ["d", "pit", "r1", "r2", "random"] + (["teacher"] if teacher else []),
            "cost_invariant_policies_run_once": ["d", "r1", "r2"] + (["teacher"] if teacher else []),
            "matched_random_rate": stats["matched_rate"],
            "regret": {"p_world": a.regret_p, "points": stats["regret_points"], "check": a.regret_check},
            "summary": summary, "solver_cache": stats["solver_cache"], "cpu": cpu, "cpu_by_policy_s": stats["cpu_s"],
            "unit_costs": unit, "sources": {**h.source_hashes(), "campaign05_apit(src)":
                                            h.file_hash(REPO / "src/tensegra/campaign05_apit.py"),
                                            "campaign05_apit(tool)": h.file_hash(Path(__file__))}}
    if regret and a.regret_check:
        meta["regret_check_all_equal"] = all(r["check_equal"] for r in regret)
    with gzip.open(stem.with_suffix(".json.gz"), "wt") as f:
        json.dump({"rows": rows, "regret": regret}, f, separators=(",", ":"))
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({"stem": str(stem), "summary": summary, "matched_rate": stats["matched_rate"],
                      "regret_points": stats["regret_points"],
                      **({"regret_check_all_equal": meta["regret_check_all_equal"]} if "regret_check_all_equal" in meta
                         else {}), "unit_costs": unit, "cpu": cpu}, indent=1))


def _read_eval(dirs, prefix="apit"):
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob(f"{prefix}-*.json.gz")):
            meta = json.loads(p.with_name(p.name[:-len(".json.gz")] + ".meta.json").read_text())
            with gzip.open(p, "rt") as f:
                out.append((meta, json.load(f)))
    return out


def _stats(rows):
    n = max(1, len(rows))
    return {"episodes": len(rows), "utility": sum(r["utility"] for r in rows) / n,
            "success": sum(r["success"] for r in rows) / n, "cost": sum(r["cost"] for r in rows) / n,
            "delegation_rate_per_eligible": (sum(r["delegations"] for r in rows) / max(1, sum(r["eligible"] for r in rows))),
            "delegations_per_episode": sum(r["delegations"] for r in rows) / n,
            "delegated_step_fraction": sum(r["teacher_steps"] for r in rows) / max(1, sum(r["decisions"] for r in rows)),
            "controller_evals_per_episode": sum(r.get("controller_evals", 0) for r in rows) / n,
            "consults_per_episode": sum(r.get("consults", 0) for r in rows) / n,
            "truncated": sum(bool(r["truncated"]) for r in rows), "unsupported": sum(bool(r["unsupported"]) for r in rows),
            **({"by_anchor": {an: {"eligible": sum(r.get("eligible_by_anchor", {}).get(an, 0) for r in rows),
                                   "delegations": sum(r.get("delegations_by_anchor", {}).get(an, 0) for r in rows)}
                              for an in ("call", "reuse_recompute", "commit_revise")},
                "delegation_ends": _sum_dicts(r.get("delegation_ends") for r in rows)}
               if any("eligible_by_anchor" in r for r in rows) else {})}


def _sum_dicts(ds):
    out = {}
    for d in ds:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + v
    return out


def _paired(rows_a, rows_b, key):
    """Mean of a - b over worlds present in both, with world-clustered bootstrap CIs."""
    b = {(r["condition"], r["seed"]): r for r in rows_b}
    diffs, worlds = [], []
    for r in rows_a:
        k = (r["condition"], r["seed"])
        if k in b:
            diffs.append(float(r[key]) - float(b[k][key]))
            worlds.append(f"{k[0]}/{k[1]}")
    return hr()._mean_ci(diffs, worlds)


def score(evals, *, protocol=True):
    by = {}
    teacher = {}
    regret = {}
    meta_seen = {}
    for meta, data in evals:
        if protocol and not meta["protocol_worlds"]:
            continue
        base = meta["base"]
        meta_seen.setdefault(base, set()).add((meta["condition"], meta["chunk"]))
        for r in data["rows"]:
            if r["policy"] == "teacher":
                teacher[(r["condition"], r["seed"])] = r
            else:
                by.setdefault(base, {}).setdefault(r["policy"], []).append(r)
        for r in data.get("regret") or []:
            regret.setdefault(base, []).append({**r, "condition": meta["condition"]})
    teacher_rows = list(teacher.values())
    rep = {"version": "e05-apit-score-v1", "registered": {
        "PI-T-1": "per lineage, IID group (iid_f0+iid_f2, 512 each): utility(pi_T) >= utility(D) + .015 AND "
                  "success(pi_T) >= success(D) - .01",
        "PI-T-2": "per lineage, IID group: utility(pi_T) >= max(always-teacher, R1, R2, random-matched) + .005 AND "
                  "delegated-step fraction(pi_T) < always-teacher's"},
        "notes": ["point estimates decide the registered criteria; CIs are world-clustered percentile bootstraps",
                  "always-teacher = dep_reuse reference evaluation (lineage-independent; from the x1-r0 jobs)",
                  "R-mask-scored metrics are not used; utility includes failures"],
        "lineages": {}}
    for base in sorted(by):
        pol = by[base]
        L = {"coverage": sorted(meta_seen[base]), "conditions": {}, "iid_group": {}}
        for group, conds in [("iid_group", IID)] + [(c, (c,)) for c in CONDITIONS]:
            sel = {p: [r for r in rs if r["condition"] in conds] for p, rs in pol.items()}
            worlds = {(r["condition"], r["seed"]) for r in sel.get("d", [])}   # paired: the lineage's worlds only
            sel["teacher"] = [r for r in teacher_rows if (r["condition"], r["seed"]) in worlds]
            if not sel.get("d"):
                continue
            g = {"policies": {p: _stats(rs) for p, rs in sel.items() if rs}}
            if sel.get("pit"):
                g["pit_minus"] = {p: {"utility": _paired(sel["pit"], rs, "utility"),
                                      "success": _paired(sel["pit"], rs, "success")}
                                  for p, rs in sel.items() if p != "pit" and rs}
            if group == "iid_group":
                L["iid_group"] = g
            else:
                L["conditions"][group] = g
        g = L["iid_group"]
        if g and "pit" in g["policies"]:
            ps = g["policies"]
            du, ds = ps["pit"]["utility"] - ps["d"]["utility"], ps["pit"]["success"] - ps["d"]["success"]
            L["PI-T-1"] = {"utility_diff": du, "success_diff": ds,
                           "utility_diff_ci": g["pit_minus"]["d"]["utility"],
                           "pass": du >= PI_T_1["utility_margin"] and ds >= PI_T_1["success_tolerance"]}
            comps = {p: ps[p]["utility"] for p in ("teacher", "r1", "r2", "random") if p in ps}
            if len(comps) == 4:
                best = max(comps, key=comps.get)
                L["PI-T-2"] = {"comparators_utility": comps, "best_comparator": best,
                               "utility_diff_vs_best": ps["pit"]["utility"] - comps[best],
                               "utility_diff_vs_best_ci": g["pit_minus"][best]["utility"],
                               "delegated_step_fraction": {"pit": ps["pit"]["delegated_step_fraction"],
                                                           "teacher": ps["teacher"]["delegated_step_fraction"]},
                               "pass": (ps["pit"]["utility"] >= comps[best] + PI_T_2["utility_margin"]
                                        and ps["pit"]["delegated_step_fraction"] < ps["teacher"]["delegated_step_fraction"])}
            else:
                L["PI-T-2"] = {"pass": None, "missing": sorted({"teacher", "r1", "r2", "random"} - set(comps))}
        if base in regret:
            L["delegation_regret"] = regret_summary(regret[base])
        rep["lineages"][base] = L
    ls = rep["lineages"].values()
    rep["summary"] = {"PI-T-1_pass_lineages": sum(bool(l.get("PI-T-1", {}).get("pass")) for l in ls),
                      "PI-T-2_pass_lineages": sum(bool(l.get("PI-T-2", {}).get("pass")) for l in ls),
                      "lineages": len(rep["lineages"])}
    return rep


def _cost_stats(rows):
    n = max(1, len(rows))
    return {**_stats(rows), "utility_gross": sum(r["utility_gross"] for r in rows) / n,
            "delegation_charge": sum(r["delegation_charge"] for r in rows) / n,
            "teacher_steps_per_episode": sum(r["teacher_steps"] for r in rows) / n}


def score_cost(evals, *, protocol=True):
    """A-PI-C: per delegation cost and lineage, PI-C-2 (primary) and PI-C-1 (reported) on the IID group, plus
    per-condition readings (net utility; world-clustered CIs; delegated-step fractions; charges)."""
    by, teacher, regret, seen = {}, {}, {}, {}
    for meta, data in evals:
        if protocol and not meta["protocol_worlds"]:
            continue
        base = meta["base"]
        seen.setdefault(base, set()).add((meta["condition"], meta["chunk"]))
        for r in data["rows"]:
            c = r["delegation_cost"]
            if r["policy"] == "teacher":
                teacher.setdefault(c, {})[(r["condition"], r["seed"])] = r
            else:
                by.setdefault(c, {}).setdefault(base, {}).setdefault(r["policy"], []).append(r)
        for r in data.get("regret") or []:
            regret.setdefault(r["delegation_cost"], {}).setdefault(base, []).append(r)
    rep = {"version": "e05-apic-score-v1", "registered": {
        "PI-C-2 (primary, per cost)": "per lineage, IID group (iid_f0+iid_f2, 512 each): utility(pi_T) >= "
                                      "max(D, always-teacher, R1, R2, random-matched) + .005 AND success(pi_T) >= "
                                      "success(D) - .01; the claim needs 3/3 lineages",
        "PI-C-1 (reported)": "per lineage, IID group: utility(pi_T) >= utility(D) + .015 AND success(pi_T) >= "
                             "success(D) - .01"},
        "notes": ["utility is NET of the external delegation cost c x teacher-controlled steps (D pays 0; always-"
                  "teacher pays every step); utility_gross and delegation_charge are reported",
                  "point estimates decide the registered criteria; CIs are world-clustered percentile bootstraps",
                  "always-teacher = dep_reuse reference evaluation (lineage-independent; from the x1-r0 / x1-r3 jobs)"],
        "costs": {}}
    for c in sorted(by):
        C = {"lineages": {}}
        for base in sorted(by[c]):
            pol = by[c][base]
            L = {"coverage": sorted(seen[base]), "conditions": {}, "iid_group": {}}
            for group, conds in [("iid_group", IID)] + [(k, (k,)) for k in CONDITIONS]:
                sel = {p: [r for r in rs if r["condition"] in conds] for p, rs in pol.items()}
                worlds = {(r["condition"], r["seed"]) for r in sel.get("d", [])}
                sel["teacher"] = [r for k, r in teacher.get(c, {}).items() if k in worlds]
                if not sel.get("d"):
                    continue
                g = {"policies": {p: _cost_stats(rs) for p, rs in sel.items() if rs}}
                if sel.get("pit"):
                    g["pit_minus"] = {p: {"utility": _paired(sel["pit"], rs, "utility"),
                                          "success": _paired(sel["pit"], rs, "success")}
                                      for p, rs in sel.items() if p != "pit" and rs}
                    comps = {p: g["policies"][p]["utility"] for p in ("d", "teacher", "r1", "r2", "random")
                             if p in g["policies"]}
                    best = max(comps, key=comps.get)
                    g["pit_minus_best"] = {"best_comparator": best,
                                           "utility_diff": g["policies"]["pit"]["utility"] - comps[best],
                                           "utility_diff_ci": g["pit_minus"][best]["utility"]}
                if group == "iid_group":
                    L["iid_group"] = g
                else:
                    L["conditions"][group] = g
            g = L["iid_group"]
            if g and "pit" in g["policies"]:
                ps = g["policies"]
                du, ds = ps["pit"]["utility"] - ps["d"]["utility"], ps["pit"]["success"] - ps["d"]["success"]
                L["PI-C-1"] = {"utility_diff": du, "success_diff": ds, "utility_diff_ci": g["pit_minus"]["d"]["utility"],
                               "pass": du >= PI_C_1["utility_margin"] and ds >= PI_C_1["success_tolerance"]}
                comps = {p: ps[p]["utility"] for p in ("d", "teacher", "r1", "r2", "random") if p in ps}
                if len(comps) == 5:
                    best = max(comps, key=comps.get)
                    L["PI-C-2"] = {"comparators_utility": comps, "best_comparator": best,
                                   "utility_diff_vs_best": ps["pit"]["utility"] - comps[best],
                                   "utility_diff_vs_best_ci": g["pit_minus"][best]["utility"],
                                   "success_diff_vs_d": ds,
                                   "delegated_step_fraction": {p: ps[p]["delegated_step_fraction"] for p in ps},
                                   "delegation_charge": {p: ps[p]["delegation_charge"] for p in ps},
                                   "pass": (ps["pit"]["utility"] >= comps[best] + PI_C_2["utility_margin"]
                                            and ds >= PI_C_2["success_tolerance"])}
                else:
                    L["PI-C-2"] = {"pass": None, "missing": sorted({"d", "teacher", "r1", "r2", "random"} - set(comps))}
            if base in regret.get(c, {}):
                L["delegation_regret"] = regret_summary(regret[c][base])
            C["lineages"][base] = L
        ls = list(C["lineages"].values())
        n2 = sum(bool(l.get("PI-C-2", {}).get("pass")) for l in ls)
        C["summary"] = {"PI-C-1_pass_lineages": sum(bool(l.get("PI-C-1", {}).get("pass")) for l in ls),
                        "PI-C-2_pass_lineages": n2, "lineages": len(ls),
                        "PI-C-2_registered_pass": len(ls) == 3 and n2 == 3}
        rep["costs"][str(c)] = C
    rep["summary"] = {k: v["summary"] for k, v in rep["costs"].items()}
    return rep


def regret_summary(rows, thr=0.005):
    import numpy as np
    if not rows:
        return {}
    w = [f"{r['condition']}/{r['seed']}" for r in rows]
    h = hr()
    fired = [r for r in rows if r["fired"]]
    unfired = [r for r in rows if not r["fired"]]
    out = {"label": "branch-evaluated delegation decisions of pi_T (1/16 world subsample; Q = utility-to-go under "
                    "pi_T continuation; hidden-state rollouts, evaluation only)",
           "decisions": len(rows), "fired": len(fired),
           "regret_per_decision": h._mean_ci([r["regret"] for r in rows], w),
           "adv_delegate_where_fired": h._mean_ci([r["adv_delegate"] for r in fired],
                                                  [f"{r['condition']}/{r['seed']}" for r in fired]),
           "adv_delegate_where_not_fired": h._mean_ci([r["adv_delegate"] for r in unfired],
                                                      [f"{r['condition']}/{r['seed']}" for r in unfired]),
           "false_positive_rate": (sum(r["adv_delegate"] < -thr for r in fired) / len(fired)) if fired else None,
           "missed_rate": (sum(r["adv_delegate"] > thr for r in unfired) / len(unfired)) if unfired else None,
           "by_anchor": {}}
    for an in sorted(set(r["anchor"] for r in rows)):
        sub = [r for r in rows if r["anchor"] == an]
        f = [r["adv_delegate"] for r in sub if r["fired"]]
        out["by_anchor"][an] = {"decisions": len(sub), "fired": len(f),
                                "mean_adv_delegate": float(np.mean([r["adv_delegate"] for r in sub])),
                                "mean_adv_where_fired": float(np.mean(f)) if f else None,
                                "mean_regret": float(np.mean([r["regret"] for r in sub]))}
    if all("check_equal" in r for r in rows):
        out["determinism_check_all_equal"] = all(r["check_equal"] for r in rows)
    return out


def cmd_score(a):
    if a.cost:
        rep = score_cost(_read_eval(a.eval, "apic"), protocol=not a.dev)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(rep, indent=2, default=str))
        short = {c: {b: {k: l.get(k) for k in ("PI-C-1", "PI-C-2")} for b, l in C["lineages"].items()}
                 for c, C in rep["costs"].items()}
        print(json.dumps({"summary": rep["summary"], "costs": short}, indent=1, default=str)[:8000])
        return
    rep = score(_read_eval(a.eval), protocol=not a.dev)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rep, indent=2, default=str))
    short = {b: {k: l.get(k) for k in ("PI-T-1", "PI-T-2")} for b, l in rep["lineages"].items()}
    print(json.dumps({"summary": rep["summary"], "lineages": short}, indent=1, default=str)[:6000])


# --- CLI ---------------------------------------------------------------------------------------

def main(argv=None, *, binding=None, load_actor_fn=None, solver=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("train")
    s.add_argument("--branch", type=Path, nargs="+", required=True, help="A-HR2 branch dirs (--option-class multi)")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dev", action="store_true", help="tests/smokes: accept dev-world / non-screening-base files")
    s.add_argument("--delegation-cost", type=float, default=None,
                   help="A-PI-C: train on labels adjusted by -c x teacher steps (registered c: 0.001, 0.003)")
    s = sub.add_parser("eval")
    s.add_argument("--base", required=True, help="x1-r0 ... x1-r2")
    s.add_argument("--condition", required=True, choices=CONDITIONS)
    s.add_argument("--chunk", type=int, nargs="+", required=True)
    s.add_argument("--chunk-size", type=int, default=CHUNK)
    s.add_argument("--episodes", type=int, default=None)
    s.add_argument("--worlds", choices=tuple(WORLD_BASES), default="apit")
    s.add_argument("--controller", type=Path, default=None, help="A-PI-T controller (apit/acf/dev worlds)")
    s.add_argument("--cost-controllers", nargs="+", default=None, metavar="COST=PATH",
                   help="A-PI-C: one frozen cost controller per registered cost (apic/acfc worlds; dev smokes)")
    s.add_argument("--margin-override", type=float, default=None, help="dev/tests only (e.g. inf)")
    s.add_argument("--policies", nargs="+", default=["d", "pit", "r1", "r2", "random"],
                   choices=("d", "pit", "r1", "r2", "random", "always"))
    s.add_argument("--teacher", choices=("auto", "yes", "no"), default="auto")
    s.add_argument("--regret-p", type=float, default=DEFAULT_REGRET_P)
    s.add_argument("--regret-check", action="store_true", help="also branch the chosen action (determinism)")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--no-verify-hash", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--eval", type=Path, nargs="+", required=True)
    s.add_argument("--dev", action="store_true", help="score dev (non-protocol) files")
    s.add_argument("--cost", action="store_true", help="A-PI-C: score apic-* files (PI-C-1/PI-C-2 per cost)")
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == "train":
        return cmd_train(a)
    if a.command == "score":
        return cmd_score(a)
    if a.worlds in PROTOCOL_KINDS and a.margin_override is not None:
        raise SystemExit("--margin-override is for dev worlds only")
    if binding is None and a.base not in hr().bases():
        raise SystemExit(f"unknown base {a.base!r}")
    a.binding, a.solver = binding, solver
    loader, memo = (load_actor_fn or hr().load_actor), {}

    def cached(b, verify=True):
        if b["name"] not in memo:
            memo[b["name"]] = loader(b, verify=verify)
        return memo[b["name"]]
    for chunk in a.chunk:
        b = argparse.Namespace(**vars(a))
        b.chunk, b.load_actor = chunk, cached
        cmd_eval(b)


if __name__ == "__main__":
    main()
