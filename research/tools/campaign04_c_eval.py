"""Track C evaluation of every arm on fresh worlds for one frozen base (CPU; trackc.md §6-7).

    CUDA_VISIBLE_DEVICES= python research/tools/campaign04_c_eval.py --base p1-boot-x1-r0 \
        --models results/c-train-p1-boot-x1-r0 --seeds 0 1 2 --conditions iid_f0 iid_f2 \
        --arms fixed core --output results/c-eval-p1-boot-x1-r0-iid

Arms (tokens; groups expand):
  fixed            fixed:greedy fixed:sampled fixed:r_mask (= D0 = default) fixed:r_sample
  core             appraisal_only learned random_matched threshold          (per meta seed)
  causal           clamp:sample clamp:mask_top clamp:stop zero:<group> x8 shuffle_telemetry
  retrained        model:shuffled model:drop-<group> x8 (learned rule with the variant's own m)
  appraisal_only   predictions logged, behaviour = D0 (also the "control pathway disabled" arm)
  learned          u = argmax_u Q(u) if best gain over default > registered m, else default
  random_matched   interventions at the learned arm's per-condition decision rate and u mix
                   (requires `learned` for the same seed/condition in the same job)
  threshold        sample-step when predicted P(success) < registered tau
  clamp:<u>        the learned rule decides WHEN; u is replaced by <u>
  zero:<group>     learned rule with a telemetry group zeroed at evaluation (reliance)
  shuffle_telemetry learned rule fed another active episode's telemetry (batch roll)

The metacontroller forward is charged in the episode (neural work units = actor units x
meta params / actor params); rows also carry utility without that charge. Branch
evaluation (--branch-eval N): on the first N worlds of the arms in --branch-eval-arms,
up to 2 decisions per episode (Bernoulli .1) are cloned and every intervention is
branched under D0 (calibration of Q-hat on controller-induced states; offline ledger).
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
import gzip
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign04_c_configs as C  # noqa: E402

FIXED = ("fixed:greedy", "fixed:sampled", "fixed:r_mask", "fixed:r_sample")
CORE = ("appraisal_only", "learned", "random_matched", "threshold")
CAUSAL = ("clamp:sample", "clamp:mask_top", "clamp:stop") + tuple(f"zero:{g}" for g in C.TELEMETRY_GROUPS) + (
    "shuffle_telemetry",)
RETRAINED = ("model:shuffled",) + tuple(f"model:drop-{g}" for g in C.TELEMETRY_GROUPS)
GROUPS = {"fixed": FIXED, "core": CORE, "causal": CAUSAL, "retrained": RETRAINED}


def expand(tokens):
    out = []
    for t in tokens:
        for x in GROUPS.get(t, (t,)):
            if x not in out:
                out.append(x)
    ok = set(FIXED) | set(CORE) | set(CAUSAL) | set(RETRAINED)
    bad = [x for x in out if x not in ok]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    # learned must precede random_matched
    if "random_matched" in out and "learned" in out:
        out.remove("random_matched")
        out.insert(out.index("learned") + 1, "random_matched")
    return out


def ece(pred, outcome, bins=10, min_count=20):
    from tensegra.campaign04_meta_train import ece as _ece
    return _ece(pred, outcome, bins, min_count)


def summarize(rows, arm):
    n = len(rows)
    dec = sum(r["decisions"] for r in rows)
    u = Counter()
    for r in rows:
        u.update(r["u_counts"])
    inter = sum(v for k, v in u.items() if k != "default")
    out = {"examples": n, "success": sum(r["success"] for r in rows) / n,
           "utility": sum(r["utility"] for r in rows) / n,
           "utility_no_meta_charge": sum(r["utility_no_meta"] for r in rows) / n,
           "cost": sum(r["cost"] for r in rows) / n, "steps": sum(r["steps"] for r in rows) / n,
           "no_progress_steps": sum(r["no_progress"] for r in rows) / n,
           "no_progress_episode_rate": sum(r["no_progress"] > 0 for r in rows) / n,
           "decisions": dec, "intervention_rate": inter / dec if dec else 0.0,
           "u_counts": dict(u), "meta_forwards": sum(r["meta_forwards"] for r in rows),
           "rule_interventions_per_episode": sum(r["rule_interventions"] for r in rows) / n,
           "truncated": sum(r["truncated"] for r in rows)}
    preds = [(p, r["success"]) for r in rows for p in r.get("p", [])]
    if preds:
        err, used, table = ece([p for p, _ in preds], [float(s) for _, s in preds])
        # realized ΔU under the label definition (no metacontrol charge after t)
        q = [abs(q0 - (r["utility"] - U + (r["decisions"] - t) * r["meta_charge_per_forward"]))
             for r in rows for t, (q0, U) in enumerate(zip(r["q0"], r["U"]))]
        out["calibration_steps"] = {
            "valid": arm == "appraisal_only",
            "scope": "P(success | D0) and Q(default) vs the realized outcome; valid only when behaviour after t is D0 "
                     "(appraisal_only); other arms: descriptive",
            "ece": err, "bins_used": used, "reliability": table, "q0_mae": sum(q) / len(q)}
        first = [(r["p"][0], float(r["success"])) for r in rows if r.get("p")]
        out["calibration_steps"]["ece_first_decision"] = ece([a for a, _ in first], [b for _, b in first])[0]
    points = [bp for r in rows for bp in r.get("branch_points", [])]
    if points:
        out["branch_eval"] = branch_metrics(points)
    return out


def branch_metrics(points):
    from tensegra.campaign04_meta import ADV_US
    res = {"points": len(points)}
    for k, u in enumerate(ADV_US):
        pairs = []
        for bp in points:
            actual = bp["actual_gain"].get(u)
            if actual is not None:
                pairs.append((bp["pred"]["adv"][k], actual))
        if pairs:
            res[f"adv_mae_{u}"] = sum(abs(a - b) for a, b in pairs) / len(pairs)
            res[f"adv_sign_agreement_{u}"] = sum((a > 0) == (b > 0) for a, b in pairs) / len(pairs)
            res[f"mean_pred_gain_{u}"] = sum(a for a, _ in pairs) / len(pairs)
            res[f"mean_actual_gain_{u}"] = sum(b for _, b in pairs) / len(pairs)
    q0 = [(bp["pred"]["q0"], bp["actual"]["default"]) for bp in points if bp["actual"]["default"] is not None]
    res["q0_mae"] = sum(abs(a - b) for a, b in q0) / len(q0) if q0 else None
    ps = [(bp["pred"]["p"], float(bp["success_default"])) for bp in points if bp["success_default"] is not None]
    res["ece_success_default"] = ece([a for a, _ in ps], [b for _, b in ps], min_count=10)[0] if ps else None
    chosen = [bp for bp in points if bp["u"] != "default"]
    if chosen:
        res["at_interventions"] = {"n": len(chosen),
                                   "pred_gain": sum(bp["pred_gain_chosen"] for bp in chosen) / len(chosen),
                                   "actual_gain": sum(bp["actual_gain"].get(bp["u"]) or 0.0 for bp in chosen)
                                   / len(chosen)}
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--models", type=Path, help="campaign04_c_train output directory")
    p.add_argument("--seeds", type=int, nargs="+", default=list(C.META_SEEDS))
    p.add_argument("--conditions", nargs="+", default=list(C.CONDITIONS))
    p.add_argument("--arms", nargs="+", default=["fixed", "core"])
    p.add_argument("--examples", type=int, default=C.EVAL_EXAMPLES)
    p.add_argument("--offset", type=int, default=0, help="first world index within each condition")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--branch-eval", type=int, default=0, help="worlds per arm/condition with branch evaluation")
    p.add_argument("--single-seed-arms", nargs="*", default=[],
                   help="arms run only for the first meta seed (e.g. appraisal_only, causal arms)")
    p.add_argument("--branch-eval-arms", nargs="+", default=["appraisal_only", "learned"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-verify-hash", action="store_true")
    p.add_argument("--dev-worlds", action="store_true", help="non-protocol smoke on Track C development seeds")
    a = p.parse_args(argv)
    import torch
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign02_training import sampling_rng_seed
    from tensegra.campaign03_depworld import depworld_executor
    from tensegra.campaign04_branch import SolverCache
    from tensegra.campaign04_meta import (ADV_US, CONTINUATION, FixedAgent, MetaAgent, digest_seed,
                                          meta_units_per_forward, new_episode, run, run_branches, snapshot)
    from campaign04_c_train import load_model
    torch.set_num_threads(a.threads)
    arms = expand(a.arms)
    C.check_fresh()
    if a.offset + a.examples > C.EVAL_STRIDE:
        raise SystemExit("outside the registered evaluation range")
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit("refusing to overwrite a nonempty output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    binding = C.base(a.base)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    actor, train_cfg, info = C.load_actor(binding, verify=not a.no_verify_hash)
    nwpf, max_steps = train_cfg.neural_work_per_forward, train_cfg.max_steps
    needs_models = [x for x in arms if not x.startswith("fixed:")]
    models = {}
    if needs_models:
        if a.models is None:
            raise SystemExit("--models is required for metacontrol arms")
        for seed in a.seeds:
            models[("main", seed)] = load_model(a.models / f"main-s{seed}.pt")
            for arm in needs_models:
                if arm.startswith("model:") and (seed == a.seeds[0] or arm not in expand(a.single_seed_arms)):
                    models[(arm[6:], seed)] = load_model(a.models / f"{arm[6:]}-s{seed}.pt")
    summary = {"base": binding["name"], "base_sha256": binding["sha256"], "base_type": binding["base_type"],
               "f10_stratum": binding["f10_stratum"], "actor": info, "continuation": CONTINUATION,
               "arms": arms, "seeds": a.seeds, "conditions": a.conditions, "examples": a.examples, "offset": a.offset,
               "namespace": C.NAMESPACE_EVAL, "sources": C.source_hashes(),
               "protocol_worlds": not a.dev_worlds,
               "models": {f"{k[0]}-s{k[1]}": {"file_sha256": C.file_hash(a.models / f"{k[0]}-s{k[1]}.pt"),
                                              "margin": v[1]["registration"]["margin"],
                                              "tau": v[1]["registration"]["tau"], "parameters": v[1]["parameters"]}
                          for k, v in models.items()},
               "charging": "actor forward: neural_work_per_forward per decision; metacontroller forward: "
                           "neural_work_per_forward x meta params / actor params per meta decision (charged in the "
                           "episode); diagnostic/telemetry: not charged (protocol-A1), CPU reported",
               "results": []}
    with BoundedSolver() as solver:
        for ci, cond in enumerate(a.conditions):
            if cond not in C.CONDITIONS:
                raise SystemExit(f"unknown condition {cond}")
            first = (C.DEV_EVAL_BASE if a.dev_worlds else C.EVAL_SEED_BASE) + C.EVAL_STRIDE * C.CONDITIONS.index(
                cond) + a.offset
            seeds = list(range(first, first + a.examples))
            cache = SolverCache(partial(depworld_executor, execute_call=solver.execute))
            folder = a.output / cond
            folder.mkdir()
            learned_mix = {}
            for arm in arms:
                arm_seeds = [None] if arm.startswith("fixed:") else (
                    a.seeds[:1] if arm in expand(a.single_seed_arms) else a.seeds)
                for meta_seed in arm_seeds:
                    t_wall, t_cpu = time.perf_counter(), time.process_time()
                    agent, model, record, meta_units = None, None, None, 0.0
                    if arm.startswith("fixed:"):
                        agent = FixedAgent(arm[6:])
                    else:
                        variant = arm[6:] if arm.startswith("model:") else "main"
                        model, record = models[(variant, meta_seed)]
                        reg = record["registration"]
                        meta_units = meta_units_per_forward(model, actor, nwpf)
                        if arm in ("appraisal_only", "threshold", "learned") or arm.startswith("model:"):
                            mode = "learned" if arm.startswith("model:") else arm
                            agent = MetaAgent(model, mode, margin=reg["margin"], tau=reg["tau"])
                        elif arm == "random_matched":
                            if ("learned", meta_seed) not in learned_mix:
                                raise SystemExit("random_matched needs the learned arm in the same job")
                            rate, probs = learned_mix[("learned", meta_seed)]
                            agent = MetaAgent(None, "random", rate=rate, u_probs=probs or {"sample": 1.0})
                        elif arm.startswith("clamp:"):
                            agent = MetaAgent(model, arm, margin=reg["margin"])
                        elif arm.startswith("zero:"):
                            agent = MetaAgent(model, "learned", margin=reg["margin"], zero_groups=(arm[5:],))
                        elif arm == "shuffle_telemetry":
                            agent = MetaAgent(model, "learned", margin=reg["margin"], shuffle=True)
                    sampling = C.SAMPLING_SEED if arm.startswith("fixed:") else C.META_SAMPLING_SEED
                    rows, timing = [], {}
                    branch_cpu = 0.0
                    for start in range(0, len(seeds), a.batch):
                        chunk = seeds[start:start + a.batch]
                        eps = [new_episode(C.world(s, cond, cache, C.NAMESPACE_EVAL), cap=max_steps,
                                           rng=random.Random(sampling_rng_seed(s, sampling)),
                                           aux_rng=random.Random(sampling_rng_seed(s, C.RANDOM_ARM_SEED)),
                                           telemetry=agent.telemetry,
                                           info={"seed": s, "idx": start + j, "U": [], "pts": []})
                               for j, s in enumerate(chunk)]
                        price = eps[0].env._spec.compute_price
                        branch_on = arm in a.branch_eval_arms and a.branch_eval > 0
                        points = []

                        def hook(ep, d, u, points=points, branch_on=branch_on, start=start):
                            if d.prediction is not None:
                                ep.info["U"].append(ep.env.current_utility())
                            if not branch_on or d.prediction is None:
                                return
                            if ep.info["idx"] >= a.branch_eval or len(ep.info["pts"]) >= 2:
                                return
                            rng = ep.info.setdefault("brng", random.Random(
                                digest_seed("c-eval-branch", ep.info["seed"], arm, meta_seed)))
                            if rng.random() < 0.1:
                                bp = snapshot(ep, d, stratum="eval", p_include=0.1, k_draws=4, draw_rng=rng)
                                bp.u_taken = u
                                ep.info["pts"].append(len(points))
                                points.append(bp)

                        run(actor, eps, agent, neural_work_per_forward=nwpf, meta_units=meta_units, charge_meta=True,
                            hook=hook, timing=timing)
                        branch_results = []
                        if points:
                            b0 = time.process_time()
                            branch_results = run_branches(actor, points, neural_work_per_forward=nwpf)
                            branch_cpu += time.process_time() - b0
                        for ep in eps:
                            out = ep.env.evaluate()
                            prog = ep.tracker.summary()
                            u_counts = Counter(x["u"] for x in ep.log)
                            meta_charge = ep.meta_forwards * meta_units * ep.env._spec.compute_price
                            row = {"seed": ep.info["seed"], "arm": arm, "meta_seed": meta_seed,
                                   "success": bool(out["verified_success"]), "utility": out["utility"],
                                   "utility_no_meta": out["utility"] + meta_charge, "cost": out["cost"],
                                   "meta_charge_per_forward": meta_units * ep.env._spec.compute_price,
                                   "steps": out["steps"], "compute_units": out["compute_units"],
                                   "meta_forwards": ep.meta_forwards, "no_progress": prog["no_progress"],
                                   "classes": prog["classes"], "decisions": len(ep.log), "u_counts": dict(u_counts),
                                   "rule_interventions": len(prog["interventions"]),
                                   "truncated": not ep.env.observe().done}
                            if ep.log and "pred" in ep.log[0]:
                                row["p"] = [round(x["pred"]["p"], 6) for x in ep.log]
                                row["q0"] = [round(x["pred"]["q0"], 6) for x in ep.log]
                                row["U"] = ep.info["U"]
                            bps = []
                            for k in ep.info["pts"]:
                                bp, r = points[k], branch_results[k]
                                actual = {"default": r["default"], "mask_top": r["mask_top"], "stop": r["stop"],
                                          "sample": sum(r["sample"]) / len(r["sample"])}
                                gain = {u: (None if actual[u] is None else actual[u] - actual["default"])
                                        for u in ADV_US}
                                pg = dict(zip(ADV_US, bp.prediction["adv"]))
                                bps.append({"step": bp.step, "u": bp.u_taken, "pred": bp.prediction,
                                            "actual": actual, "actual_gain": gain, "draws": r["sample"],
                                            "success_default": r["success_default"],
                                            "pred_gain_chosen": pg.get(bp.u_taken, 0.0)})
                            if bps:
                                row["branch_points"] = bps
                            rows.append(row)
                    name = arm.replace(":", "-") + ("" if meta_seed is None else f"-s{meta_seed}")
                    with gzip.open(folder / f"{name}.jsonl.gz", "wt") as f:
                        for r in rows:
                            f.write(json.dumps(r, sort_keys=True) + "\n")
                    res = {"condition": cond, "arm": arm, "meta_seed": meta_seed, **summarize(rows, arm),
                           "meta_units_per_forward": meta_units, "timing": timing,
                           "branch_eval_cpu_seconds": branch_cpu,
                           "wall_seconds": time.perf_counter() - t_wall, "process_cpu_seconds": time.process_time() - t_cpu,
                           "artifact": f"{cond}/{name}.jsonl.gz"}
                    if timing.get("decisions"):
                        per = timing["actor_s"] / timing["decisions"]
                        rate = nwpf / per if per > 0 else 0.0   # work units per actor-second
                        side = (timing["diagnostic_s"] + timing["telemetry_s"]) / timing["decisions"]
                        res["utility_with_diagnostic_cpu_charged"] = res["utility"] - (
                            res["decisions"] / res["examples"]) * side * rate * price
                        res["diagnostic_charge_rule"] = ("diagnostic + telemetry seconds per decision converted at the "
                                                         "actor's measured units-per-second (protocol-A1 same rate)")
                    if arm == "learned":
                        inter = {k: v for k, v in res["u_counts"].items() if k != "default"}
                        tot = sum(inter.values())
                        learned_mix[("learned", meta_seed)] = (res["intervention_rate"],
                                                               {k: v / tot for k, v in inter.items()} if tot else {})
                    if arm == "random_matched":
                        res["matched_to"] = {"rate": agent.rate, "u_probs": agent.u_probs}
                    summary["results"].append(res)
                    print(json.dumps({k: res.get(k) for k in ("condition", "arm", "meta_seed", "success", "utility",
                                                               "cost", "no_progress_steps", "intervention_rate",
                                                               "wall_seconds")}), flush=True)
                    (a.output / "summary.partial.json").write_text(json.dumps(summary, indent=1, default=str))
            summary.setdefault("solver_cache", {})[cond] = cache.stats()
    summary["resources"] = {"wall_seconds": time.perf_counter() - wall0,
                            "parent_cpu_seconds": time.process_time() - cpu0}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (a.output / "summary.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
