"""Validate progress diagnostic v1 against the P2a registered no-progress operationalization.

Reads a campaign02_evaluate.py output directory (the extended-03 P2a screening,
read-only) and, for every episode row:

1. regenerates the world from its seed and condition (spec hash checked against
   worlds.jsonl.gz), replays the recorded actions in a fresh ``DepWorkshop``
   (same address seed; in-process frozen solver behind the exact ``SolverCache``)
   and checks that every replayed feedback equals the recorded one (so the
   replayed public observations are the ones the actor saw);
2. runs ``ProgressTracker`` online over the replayed observations;
3. recomputes the registered P2a flags from the recorded history with the
   registered code (``campaign03_p2a_analysis.no_progress_flags``) and checks
   them against the registered per-cell numbers (``p2a-analysis.json``);
4. compares, step by step, P2a no-progress (idempotent or short cycle) with v1
   no-progress (idempotent or short_cycle) and attributes every disagreement to a
   documented difference category.

It also runs the supplied references dep_reuse and dep_recompute on the same
worlds with the tracker (they must score 0 no-progress).

    python research/tools/campaign04_progress_validate.py <p2a-screening-dir> \\
        --registered research/results/campaign-03/p2a-analysis/p2a-analysis.json --output <dir> [--limit N]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import partial
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
VALIDATION_VERSION = "progress-validate-v1"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def world_kwargs(world):
    world = dict(world)
    if "call_budgets" in world:
        world["call_budgets"] = tuple(world["call_budgets"])
    return world


def _json(x):
    return json.loads(json.dumps(x))


def disagreement_category(event, cls, p2a_idem, p2a_cycle, prev_kind):
    """Documented reason for a per-step disagreement (v1 vs P2a)."""
    kind = event["action"]["kind"]
    status = event["feedback"].get("status")
    p2a = p2a_idem or p2a_cycle
    mine = cls in ("idempotent", "short_cycle")
    if mine and not p2a:
        if kind == "call":
            return "v1-only: repeated call with unchanged inputs (records content-keyed; P2a: calls break cycles)"
        if kind == "retrieve":
            return "v1-only: retrieval of an already retrieved record content"
        if status != "success":
            return "v1-only: executed call with non-success status repeated"
        return f"v1-only: cycle through a repeated call/content-equal record ({kind})"
    if p2a and not mine:
        if cls == "triggered_revision":
            return "P2a-only: triggered revision exempt from cycles (v1 precedence)"
        if cls == "solver_work":
            return "P2a-only: solver work (P2a idempotent repeat of a call)"
        if cls == "rejected":
            return "P2a-only: rejected"
        if cls == "other_progress":
            if kind == "call":
                return "P2a-only: call creating a new record (v1 records in state)"
            return f"P2a-only: v1 state differs (records/plan context) ({kind})"
        return f"P2a-only: {cls} ({kind})"
    return None


def replay_row(row, spec, address_seed, executor, P2A, window):
    from tensegra.campaign02_world import Action
    from tensegra.campaign03_depworld import DepWorkshop
    from tensegra.campaign04_progress import ProgressTracker
    env = DepWorkshop(spec, executor=executor, address_seed=address_seed)
    tracker = ProgressTracker(env.observe(), window=window)
    history = row["outcome"]["history"]
    mismatches = 0
    classes = []
    for event in history:
        action = Action(event["action"]["kind"], event["action"].get("arguments") or {})
        o = env.step(action)
        if _json(o.feedback) != event["feedback"]:
            mismatches += 1
        classes.append(tracker.update(o, action).cls)
    spec_dict = _json(asdict(spec))
    flags = P2A.no_progress_flags(history, spec_dict.get("start"), P2A.initial_drafts(spec_dict))
    return tracker, classes, flags, mismatches


def run_reference(spec, address_seed, executor, mode, window):
    from tensegra.campaign03_depworld import DepReference, DepWorkshop
    from tensegra.campaign04_progress import ProgressTracker
    env = DepWorkshop(spec, executor=executor, address_seed=address_seed)
    ref, o = DepReference(mode), env.observe()
    tracker = ProgressTracker(o, window=window)
    while not o.done:
        a = ref.choose(o)
        o = env.step(a)
        tracker.update(o, a)
    return tracker


def validate(eval_dir: Path, registered=None, limit=None, references=("reuse", "recompute"), log=print):
    from tensegra.campaign02_protocol import execute
    from tensegra.campaign03_depworld import depworld_executor, generate_depworld
    from tensegra.campaign04_branch import SolverCache
    from tensegra.campaign04_progress import CODES, NO_PROGRESS, WINDOW
    P2A = _load("campaign03_p2a_analysis")
    online_checked = online_mismatch = 0
    summary = json.loads((eval_dir / "summary.json").read_text())
    conditions = {c["name"]: c for c in summary["config"]["conditions"]}
    worlds = {}
    cells = {}
    categories = Counter()
    examples = defaultdict(list)
    total_mismatch = 0
    start = time.process_time()
    cache = SolverCache(partial(depworld_executor, execute_call=execute))
    specs = {}
    for result in summary["results"]:
        c = result["condition"]
        role, (base, mode) = P2A.result_role(result), P2A.split_condition(c, result)
        if c not in worlds:
            worlds[c] = {w["seed"]: w for w in read_rows(eval_dir / c / "worlds.jsonl.gz")}
        rows = read_rows(eval_dir / result["artifact"])
        cap = (result.get("training_config") or {}).get("max_steps")
        cell = cells.setdefault(f"{role}/{mode}/{base}", Counter())
        # registered P2a metric on ALL rows (history only, no replay): checks the registered numbers
        for row in rows:
            m = P2A.episode_no_progress(row, worlds[c][row["seed"]]["spec"], cap)
            cell["all_episodes"] += 1
            cell["all_decisions"] += m["decisions"]
            cell["all_p2a_no_progress"] += m["no_progress"]
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            w = worlds[c][row["seed"]]
            key = (c, row["seed"])
            if key not in specs:
                spec = generate_depworld(row["seed"], **world_kwargs(conditions[c].get("world", {})))
                if canonical_hash(asdict(spec)) != w["spec_hash"]:
                    raise ValueError(f"regenerated world differs: {key}")
                specs[key] = spec
            spec = specs[key]
            tracker, classes, flags, mismatches = replay_row(row, spec, w["address_seed"], cache, P2A, WINDOW)
            total_mismatch += mismatches
            cell["replay_feedback_mismatches"] += mismatches
            if "progress" in row:  # rows written with the online diagnostic: offline must reproduce it
                online_checked += 1
                online_mismatch += row["progress"]["classes"] != "".join(CODES[cl] for cl in classes)
            m = P2A.episode_no_progress(row, w["spec"], cap)
            cell["episodes"] += 1
            cell["decisions"] += m["decisions"]
            cell["p2a_no_progress"] += m["no_progress"]
            cell["p2a_no_progress_episode"] += m["no_progress_episode"]
            mine_np = sum(cl in NO_PROGRESS for cl in classes)
            cell["v1_no_progress"] += mine_np
            cell["v1_no_progress_episode"] += int(mine_np > 0)
            cell["v1_latent_stalls"] += tracker.latent_stalls
            for cl in classes:
                cell[f"v1_class_{cl}"] += 1
            prev = None
            for t, (event, cl, (idem, cyc)) in enumerate(zip(row["outcome"]["history"], classes, flags), start=1):
                p2a, mine = idem or cyc, cl in NO_PROGRESS
                cell["steps_both"] += p2a and mine
                cell["steps_p2a_only"] += p2a and not mine
                cell["steps_v1_only"] += mine and not p2a
                cell["steps_agree"] += p2a == mine
                cat = disagreement_category(event, cl, idem, cyc, prev)
                if cat:
                    categories[cat] += 1
                    cell["cat:" + cat] += 1
                    if len(examples[cat]) < 3:
                        examples[cat].append({"cell": f"{role}/{mode}/{base}", "seed": row["seed"], "step": t,
                                              "action": event["action"], "status": event["feedback"].get("status"),
                                              "v1_class": cl, "p2a": [bool(idem), bool(cyc)]})
                prev = event["action"]["kind"]
        log(f"{role}/{mode}/{base}: {dict(cell)} cpu={time.process_time() - start:.0f}s")
    ref_cells = {}
    for c, ws in worlds.items():
        base = c[:-len("-sampled")] if c.endswith("-sampled") else c
        if c != base:
            continue
        for mode in references:
            cell = ref_cells.setdefault(f"dep_{mode}/{base}", Counter())
            for seed, w in list(ws.items())[:limit]:
                spec = specs.get((c, seed)) or generate_depworld(seed, **world_kwargs(conditions[c].get("world", {})))
                tracker = run_reference(spec, w["address_seed"], cache, mode, WINDOW)
                s = tracker.summary()
                cell["episodes"] += 1
                cell["decisions"] += len(tracker.classes)
                cell["v1_no_progress"] += s["no_progress"]
                cell["v1_no_progress_episode"] += int(s["no_progress"] > 0)
    table = {}
    for name, cell in cells.items():
        d = dict(cell)
        n, dec = d.get("episodes", 0), d.get("decisions", 0)
        d["p2a_no_progress_rate"] = d.get("p2a_no_progress", 0) / dec if dec else None
        d["v1_no_progress_rate"] = d.get("v1_no_progress", 0) / dec if dec else None
        d["p2a_no_progress_episode_rate"] = d.get("p2a_no_progress_episode", 0) / n if n else None
        d["v1_no_progress_episode_rate"] = d.get("v1_no_progress_episode", 0) / n if n else None
        d["step_agreement"] = d.get("steps_agree", 0) / dec if dec else None
        d["all_p2a_no_progress_rate"] = d["all_p2a_no_progress"] / d["all_decisions"] if d.get("all_decisions") else None
        reg = ((registered or {}).get("table") or {}).get(name)
        if reg is not None:
            d["registered_no_progress_rate"] = reg.get("no_progress_rate")
            d["registered_matches_recomputed"] = (reg.get("no_progress_rate") is not None and d["all_p2a_no_progress_rate"]
                                                  is not None and abs(reg["no_progress_rate"] - d["all_p2a_no_progress_rate"]) < 1e-12)
        table[name] = d
    decisions = sum(c["decisions"] for c in cells.values())
    agree = sum(c["steps_agree"] for c in cells.values())
    return {"validation_version": VALIDATION_VERSION, "evaluation_dir": str(eval_dir), "limit": limit,
            "window": WINDOW, "decisions": decisions, "step_agreement": agree / decisions if decisions else None,
            "steps_both": sum(c["steps_both"] for c in cells.values()),
            "steps_p2a_only": sum(c["steps_p2a_only"] for c in cells.values()),
            "steps_v1_only": sum(c["steps_v1_only"] for c in cells.values()),
            "registered_cells_compared": sum("registered_matches_recomputed" in d for d in table.values()),
            "registered_cells_matched": sum(bool(d.get("registered_matches_recomputed")) for d in table.values()),
            "replay_feedback_mismatches": total_mismatch, "online_rows_checked": online_checked,
            "online_replay_class_mismatches": online_mismatch, "disagreement_categories": dict(categories),
            "disagreement_examples": dict(examples), "cells": table,
            "references": {k: {**dict(v), "v1_no_progress_rate": v["v1_no_progress"] / v["decisions"] if v["decisions"] else None}
                           for k, v in ref_cells.items()},
            "solver_cache": cache.stats(), "process_cpu_seconds": time.process_time() - start}


def markdown(r):
    f = lambda x: "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    lines = [f"# Progress diagnostic v1 vs P2a registered no-progress ({r['validation_version']})", "",
             f"Evaluation: `{r['evaluation_dir']}`; limit {r['limit']}; window {r['window']}.", "",
             f"- decisions: {r['decisions']}; per-step agreement {f(r['step_agreement'])}; both {r['steps_both']}, "
             f"P2a-only {r['steps_p2a_only']}, v1-only {r['steps_v1_only']}",
             f"- replay feedback mismatches: {r['replay_feedback_mismatches']}", "",
             f"- registered P2a rates reproduced (all rows, registered code): {r.get('registered_cells_matched')}/"
             f"{r.get('registered_cells_compared')} cells", "",
             "Columns: registered = p2a-analysis.json; all-rows = registered code on every row; the remaining columns "
             "are on the replayed subset (limit).", "",
             "| cell | replayed | P2a registered | P2a all-rows | P2a subset | v1 subset | P2a eps | v1 eps | agreement |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, d in sorted(r["cells"].items()):
        lines.append(f"| {name} | {d.get('episodes')} | {f(d.get('registered_no_progress_rate'))} | "
                     f"{f(d.get('all_p2a_no_progress_rate'))} | {f(d['p2a_no_progress_rate'])} | {f(d['v1_no_progress_rate'])} | "
                     f"{f(d['p2a_no_progress_episode_rate'])} | {f(d['v1_no_progress_episode_rate'])} | "
                     f"{f(d['step_agreement'])} |")
    lines += ["", "## Disagreement categories (steps)", ""]
    for cat, n in sorted(r["disagreement_categories"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {n}: {cat}")
    lines += ["", "## References (v1 no-progress on the same worlds)", "", "| reference/condition | episodes | decisions | no-progress |",
              "|---|---|---|---|"]
    for name, d in sorted(r["references"].items()):
        lines.append(f"| {name} | {d['episodes']} | {d['decisions']} | {d['v1_no_progress']} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evaluation", type=Path)
    p.add_argument("--registered", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, help="episodes per cell (subset); default all")
    a = p.parse_args()
    registered = json.loads(a.registered.read_text()) if a.registered else None
    result = validate(a.evaluation, registered, a.limit)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "progress-validation.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (a.output / "progress-validation.md").write_text(markdown(result))
    print(json.dumps({k: result[k] for k in ("decisions", "step_agreement", "steps_both", "steps_p2a_only",
                                              "steps_v1_only", "replay_feedback_mismatches", "registered_cells_matched", "registered_cells_compared", "process_cpu_seconds")}))


if __name__ == "__main__":
    main()
