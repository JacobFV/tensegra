"""extended-04 A1 deployment-matrix analysis (exploratory; design §2 A1).

Reads one or more campaign02_evaluate.py output directories built from
``campaign04_a1_configs.py`` configs (the combined config or the split per-policy
configs; the same worlds either way) and recomputes every metric from raw rows.
Standard library only; deterministic.

    python research/tools/campaign04_a1_analysis.py <eval-dir> [<eval-dir> ...] --output <dir>

Per policy x mode x condition (a "cell"):
- success, utility, total cost (means over worlds); work per success = total work
  units / verified successes; steps; steps-to-cap rate;
- no-progress from the progress diagnostic v1 (``row["progress"]``, computed online):
  per-decision rate, per-episode count, fraction of episodes with any, idempotent /
  short-cycle rates, class counts, latent stalls, the cycle-length distribution;
- the P2a registered no-progress rate recomputed from the exact history
  (campaign03_p2a_analysis) for continuity with extended-03;
- intervention counts by kind (mask / sample / fallback / msample), the fraction that
  changed the greedy choice, interventions per episode;
- per-world variance summaries: SD of success, utility and cost across worlds; for
  every non-greedy mode the world-paired difference to the same policy's greedy cell
  (mean, SD, SE) and the worlds per cell needed to detect a utility / success
  difference of .01/.02/.05 (two-sided alpha .05, power .8, paired normal
  approximation): the inputs for the Track C power calculation (design v2 rev. 8).
Aggregates: IID group = equal-weighted mean over iid_f0 and iid_f2 (P1 convention),
per lineage and per family (bootstrap, p1_rl_final, p2a_c1_final; mean, min, max
over lineages); references are reported alongside as supplied schedules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ANALYSIS_VERSION = "a1-analysis-v1"
HERE = Path(__file__).resolve().parent
IID = ("iid_f0", "iid_f2")
CONDITIONS = ("iid_f0", "iid_f2", "events_train_kinds_p1", "foreign4")
Z_ALPHA, Z_BETA = 1.959963984540054, 0.8416212335729143
EFFECTS = (0.01, 0.02, 0.05)
NP_CLASSES = ("idempotent", "short_cycle")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def worlds_needed(sd_diff, effect):
    """Paired-design worlds for a two-sided alpha=.05, power=.8 normal test."""
    if sd_diff is None:
        return None
    if sd_diff == 0:
        return 1
    return math.ceil(((Z_ALPHA + Z_BETA) * sd_diff / effect) ** 2)


def episode(row, P2A=None, spec=None, cap=None):
    o = row["outcome"]
    p = row.get("progress") or {}
    counts = p.get("counts") or {}
    history = o.get("history", [])
    step_cap = (spec or {}).get("step_limit", 96)
    if cap is not None:
        step_cap = min(step_cap, cap)
    e = {"seed": row["seed"], "success": int(bool(o["verified_success"])), "utility": float(o["utility"]),
         "cost": float(o["cost"]), "work_units": int(o.get("work_units", 0)), "steps": int(o.get("steps", len(history))),
         "decisions": len(history), "steps_to_cap": int(not o["verified_success"] and len(history) >= step_cap),
         "has_progress": int(bool(p)), "no_progress": int(p.get("no_progress", 0)),
         "idempotent": int(counts.get("idempotent", 0)), "short_cycle": int(counts.get("short_cycle", 0)),
         "latent_stalls": int(p.get("latent_stalls", 0)), "repeat_rejections": int(p.get("repeat_rejections", 0)),
         "classes": counts, "cycle_lengths": list(p.get("cycle_lengths", [])),
         "interventions": list(p.get("interventions", [])),
         "diagnostic_cpu_seconds": float((row.get("deployment") or {}).get("diagnostic_cpu_seconds", 0.0))}
    if P2A is not None:
        e["p2a_no_progress"] = P2A.episode_no_progress(row, spec, cap)["no_progress"]
    return e


def aggregate(episodes):
    n = len(episodes)
    if not n:
        return {"episodes": 0}
    successes = sum(e["success"] for e in episodes)
    decisions = sum(e["decisions"] for e in episodes)
    classes = Counter()
    lengths = Counter()
    kinds, changed = Counter(), Counter()
    for e in episodes:
        classes.update(e["classes"])
        lengths.update(e["cycle_lengths"])
        for _, kind, _, ch in e["interventions"]:
            kinds[kind] += 1
            changed[kind] += ch
    ratio = lambda a, b: a / b if b else None
    out = {"episodes": n, "success": successes / n, "utility": mean([e["utility"] for e in episodes]),
           "cost": mean([e["cost"] for e in episodes]),
           "work_per_success": ratio(sum(e["work_units"] for e in episodes), successes),
           "work_per_episode": mean([e["work_units"] for e in episodes]),
           "steps": mean([e["steps"] for e in episodes]), "steps_to_cap_rate": mean([e["steps_to_cap"] for e in episodes]),
           "diagnostic_coverage": mean([e["has_progress"] for e in episodes]),
           "no_progress_rate": ratio(sum(e["no_progress"] for e in episodes), decisions),
           "idempotent_rate": ratio(sum(e["idempotent"] for e in episodes), decisions),
           "short_cycle_rate": ratio(sum(e["short_cycle"] for e in episodes), decisions),
           "no_progress_per_episode": mean([e["no_progress"] for e in episodes]),
           "no_progress_episode_rate": mean([int(e["no_progress"] > 0) for e in episodes]),
           "latent_stalls_per_episode": mean([e["latent_stalls"] for e in episodes]),
           "repeat_rejections_per_episode": mean([e["repeat_rejections"] for e in episodes]),
           "class_counts": dict(classes), "cycle_length_distribution": {str(k): v for k, v in sorted(lengths.items())},
           "interventions": {k: {"count": v, "changed_choice": changed[k], "per_episode": v / n} for k, v in kinds.items()},
           "interventions_per_episode": sum(kinds.values()) / n,
           "diagnostic_cpu_seconds_per_episode": mean([e["diagnostic_cpu_seconds"] for e in episodes]),
           "variance": {"success_sd": sd([e["success"] for e in episodes]), "utility_sd": sd([e["utility"] for e in episodes]),
                        "cost_sd": sd([e["cost"] for e in episodes])}}
    if all("p2a_no_progress" in e for e in episodes):
        out["p2a_no_progress_rate"] = ratio(sum(e["p2a_no_progress"] for e in episodes), decisions)
        out["p2a_no_progress_episode_rate"] = mean([int(e["p2a_no_progress"] > 0) for e in episodes])
    return out


def paired(a, b):
    """World-paired difference b - a (same seeds), for utility and success."""
    A = {e["seed"]: e for e in a}
    B = {e["seed"]: e for e in b}
    seeds = sorted(set(A) & set(B))
    out = {"worlds": len(seeds)}
    for metric in ("utility", "success", "cost"):
        d = [B[s][metric] - A[s][metric] for s in seeds]
        s = sd(d)
        out[metric] = {"mean_difference": mean(d), "sd_difference": s,
                       "se_difference": None if s is None else s / math.sqrt(len(d)),
                       "worlds_needed": {str(x): worlds_needed(s, x) for x in EFFECTS}}
    return out


def load(eval_dirs, verify=True, p2a=True):
    P2A = _load("campaign03_p2a_analysis") if p2a else None
    cells = {}
    identities = {}
    configs = []
    for eval_dir in eval_dirs:
        eval_dir = Path(eval_dir)
        summary = json.loads((eval_dir / "summary.json").read_text())
        configs.append({"dir": str(eval_dir), "config_sha256": summary.get("config_sha256"),
                        "sources": summary.get("sources")})
        conds = {c["name"]: c for c in summary["config"]["conditions"]}
        specs = {}
        for result in summary["results"]:
            name = result["condition"]
            cond = conds[name]
            base = cond.get("base_condition", name)
            mode = cond.get("mode", result.get("policy_mode", "greedy"))
            if name not in specs:
                specs[name] = {w["seed"]: w["spec"] for w in read_rows(eval_dir / name / "worlds.jsonl.gz")}
            artifact = eval_dir / result["artifact"]
            if verify and sha256(artifact) != result["artifact_sha256"]:
                raise ValueError(f"artifact hash mismatch: {artifact}")
            if result["kind"] == "learned":
                b = result.get("checkpoint_binding") or {}
                policy = result["arm"]
                identities[policy] = {"family": b.get("family", "other"), "lineage": b.get("lineage"),
                                      "sha256": b.get("sha256"), "kind": "learned"}
            else:
                policy = result["arm"].removeprefix("reference-")
                identities[policy] = {"family": "reference", "lineage": None, "kind": "supplied_schedule"}
                mode = "reference"
            cap = (result.get("training_config") or {}).get("max_steps")
            key = (policy, mode, base)
            if key in cells:
                raise ValueError(f"duplicate cell {key}")
            cells[key] = [episode(row, P2A, specs[name][row["seed"]], cap) for row in read_rows(artifact)]
    return cells, identities, configs


def analyze(eval_dirs, verify=True):
    cells, identities, configs = load(eval_dirs, verify)
    table = {k: aggregate(v) for k, v in cells.items()}
    # world-paired differences vs the same policy's greedy cell
    for (policy, mode, base), eps in cells.items():
        g = cells.get((policy, "greedy", base))
        if mode not in ("greedy", "reference") and g is not None:
            table[(policy, mode, base)]["paired_vs_greedy"] = paired(g, eps)
    modes = sorted({m for _, m, _ in table})
    groups = {}
    for policy in sorted(identities):
        for mode in modes:
            vals = {c: table.get((policy, mode, c)) for c in IID}
            if any(v is None for v in vals.values()):
                continue
            g = {}
            for m in ("success", "utility", "cost", "work_per_success", "no_progress_rate", "no_progress_episode_rate",
                      "steps_to_cap_rate", "interventions_per_episode", "p2a_no_progress_rate"):
                xs = [vals[c].get(m) for c in IID]
                g[m] = None if any(x is None for x in xs) else sum(xs) / len(xs)
            groups[f"{policy}/{mode}"] = g
    families = defaultdict(list)
    for policy, ident in identities.items():
        families[ident["family"]].append(policy)
    family_table = {}
    for fam, members in sorted(families.items()):
        for mode in modes:
            for cond in CONDITIONS + ("iid_group",):
                rows = []
                for p in sorted(members):
                    v = groups.get(f"{p}/{mode}") if cond == "iid_group" else table.get((p, mode, cond))
                    if v is not None:
                        rows.append((p, v))
                if not rows:
                    continue
                entry = {"lineages": {p: {m: v.get(m) for m in ("success", "utility", "cost", "work_per_success",
                                                                "no_progress_rate", "no_progress_episode_rate")}
                                      for p, v in rows}}
                for m in ("success", "utility", "cost", "work_per_success", "no_progress_rate", "no_progress_episode_rate"):
                    xs = [v.get(m) for _, v in rows if v.get(m) is not None]
                    entry[m] = {"mean": mean(xs), "min": min(xs) if xs else None, "max": max(xs) if xs else None}
                family_table[f"{fam}/{mode}/{cond}"] = entry
    return {"analysis_version": ANALYSIS_VERSION, "evaluations": configs, "iid_group": list(IID),
            "scope": "A1 is exploratory evaluation of frozen policies (no training, no selection); readings are not claims.",
            "policies": identities, "table": {f"{p}/{m}/{c}": v for (p, m, c), v in sorted(table.items())},
            "iid_groups": groups, "families": family_table}


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def markdown(r):
    lines = [f"# A1 deployment matrix ({r['analysis_version']})", "", r["scope"], "",
             "## IID group (iid_f0, iid_f2)", "",
             "| policy/mode | success | utility | cost | work/success | no-progress | NP episodes | to cap | interventions/ep |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, g in r["iid_groups"].items():
        lines.append(f"| {name} | {_f(g['success'])} | {_f(g['utility'])} | {_f(g['cost'])} | {_f(g['work_per_success'], 1)} | "
                     f"{_f(g['no_progress_rate'])} | {_f(g['no_progress_episode_rate'])} | {_f(g['steps_to_cap_rate'])} | "
                     f"{_f(g['interventions_per_episode'])} |")
    lines += ["", "## Per cell", "",
              "| policy/mode/condition | n | success | utility | cost | work/success | NP rate | NP eps | P2a NP | cycles (L:n) | "
              "utility SD | paired dU vs greedy (SD) | worlds for dU=.02 |", "|---" * 13 + "|"]
    for name, v in r["table"].items():
        pv = v.get("paired_vs_greedy", {}).get("utility", {})
        cyc = " ".join(f"{k}:{n}" for k, n in v.get("cycle_length_distribution", {}).items())
        lines.append(f"| {name} | {v['episodes']} | {_f(v['success'])} | {_f(v['utility'])} | {_f(v['cost'])} | "
                     f"{_f(v['work_per_success'], 1)} | {_f(v['no_progress_rate'])} | {_f(v['no_progress_episode_rate'])} | "
                     f"{_f(v.get('p2a_no_progress_rate'))} | {cyc} | {_f(v['variance']['utility_sd'])} | "
                     f"{_f(pv.get('mean_difference'))} ({_f(pv.get('sd_difference'))}) | "
                     f"{(pv.get('worlds_needed') or {}).get('0.02', 'n/a')} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evaluations", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-verify", action="store_true", help="skip artifact hash checks")
    a = p.parse_args()
    result = analyze(a.evaluations, verify=not a.no_verify)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "a1-analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (a.output / "a1-analysis.md").write_text(markdown(result))
    print(json.dumps({k: {m: v.get(m) for m in ("success", "utility", "no_progress_rate")}
                      for k, v in result["iid_groups"].items()}))


if __name__ == "__main__":
    main()
