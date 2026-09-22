"""Artifact-only Stage 2 analysis.

This module deliberately imports only the Python standard library.  It accepts
the runner's JSONL rows and writes a machine-readable summary, a Markdown
report, and standalone SVG figures.  The small public functions are also used
to pin down the study's statistical rules independently of the training code.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def complete_normalized_auc(points: Sequence[Sequence[float]], budget: int | float) -> float | None:
    """Trapezoidal curve area divided by budget, only for a complete [0,budget] curve."""
    if not _finite(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    ordered = sorted((float(step), float(value)) for step, value in points)
    if not ordered or ordered[0][0] != 0 or ordered[-1][0] != float(budget):
        return None
    if any(not (_finite(step) and _finite(value)) for step, value in ordered):
        return None
    if any(right[0] <= left[0] for left, right in zip(ordered, ordered[1:])):
        return None
    area = sum((b_step - a_step) * (a_value + b_value) / 2
               for (a_step, a_value), (b_step, b_value) in zip(ordered, ordered[1:]))
    return area / float(budget)


def efficiency_thresholds(
    curves: Sequence[Mapping[str, Any]], oracle: float, zero: float
) -> dict[str, dict[str, Any]]:
    """Return first observed count/checkpoint reaching predeclared oracle-gap thresholds."""
    if not (_finite(oracle) and _finite(zero)) or zero < oracle:
        raise ValueError("oracle and zero must be finite with zero >= oracle")
    if zero == oracle:
        return {label: {"threshold": None, "n_epsilon": None, "s_epsilon": None, "censored": True}
                for label in ("gap_10", "gap_25", "gap_50")}
    output: dict[str, dict[str, Any]] = {}
    for label, remaining_gap in (("gap_10", 0.10), ("gap_25", 0.25), ("gap_50", 0.50)):
        threshold = float(oracle) + remaining_gap * (float(zero) - float(oracle))
        n_epsilon = None
        s_epsilon = None
        for curve in sorted(curves, key=lambda item: item["trajectory_count"]):
            crossings = [float(step) for step, value in curve["points"]
                         if _finite(step) and _finite(value) and float(value) <= threshold]
            if crossings:
                n_epsilon = curve["trajectory_count"]
                s_epsilon = min(crossings)
                if s_epsilon.is_integer():
                    s_epsilon = int(s_epsilon)
                break
        output[label] = {"threshold": threshold, "n_epsilon": n_epsilon,
                         "s_epsilon": s_epsilon, "censored": n_epsilon is None}
    return output


def paired_effects(
    rows: Sequence[Mapping[str, Any]], baseline: str, treatment: str,
    *, mode_key: str = "mode", value_key: str = "value",
) -> dict[str, Any]:
    """Summarize baseline-minus-treatment effects using complete matched seeds only."""
    values: dict[Any, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    incomplete: set[Any] = set()
    relevant_seeds: set[Any] = set()
    for row in rows:
        mode = row.get(mode_key)
        if mode not in (baseline, treatment):
            continue
        seed = row.get("seed")
        relevant_seeds.add(seed)
        if row.get("status", "complete") != "complete" or not _finite(row.get(value_key)):
            incomplete.add(seed)
            continue
        values[seed][str(mode)].append(float(row[value_key]))
    effects = [{"seed": seed,
                "effect": statistics.fmean(values[seed][baseline]) - statistics.fmean(values[seed][treatment])}
               for seed in sorted(relevant_seeds, key=str)
               if seed not in incomplete and set(values[seed]) >= {baseline, treatment}]
    paired_seeds = {item["seed"] for item in effects}
    unmatched = sorted(relevant_seeds - incomplete - paired_seeds, key=str)
    effect_values = [item["effect"] for item in effects]
    return {
        "effects": effects,
        "mean": statistics.fmean(effect_values) if effect_values else None,
        "population_sd": statistics.pstdev(effect_values) if effect_values else None,
        "n_pairs": len(effects),
        "excluded": {"incomplete": sorted(incomplete, key=str), "unmatched": unmatched},
    }


def select_soft_strength(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, Any, Any], float]:
    """Select strength by across-seed validation mean within suite/domain/train config.

    Strengths are compared on their common validation seed intersection.  Test rows
    are ignored. Ties choose the numerically smaller strength deterministically.
    """
    groups: dict[tuple[Any, Any, Any], dict[float, dict[Any, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("split") not in ("validation", "val") or not _finite(row.get("value")):
            continue
        if row.get("status", "complete") != "complete" or not _finite(row.get("strength")):
            continue
        key = (row.get("suite"), row.get("domain"), row.get("train_config"))
        groups[key][float(row["strength"])][row.get("seed")].append(float(row["value"]))
    selected: dict[tuple[Any, Any, Any], float] = {}
    for key, by_strength in groups.items():
        common = set.intersection(*(set(by_seed) for by_seed in by_strength.values())) if by_strength else set()
        if not common:
            continue
        scores = {strength: statistics.fmean(
            statistics.fmean(by_strength[strength][seed]) for seed in common)
                  for strength in by_strength}
        selected[key] = min(scores, key=lambda strength: (scores[strength], strength))
    return selected


def _evaluation_rows(runs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    flat: list[dict[str, Any]] = []
    exclusions = {"incomplete_runs": 0, "nonfinite_evaluations": 0}
    for run in runs:
        if run.get("status") != "complete":
            exclusions["incomplete_runs"] += 1
            continue
        base = {key: run.get(key) for key in (
            "suite", "domain", "seed", "mode", "variant", "strength", "train_config",
            "trajectory_count", "count", "train_count", "train_sizes", "mechanism",
            "corruption", "fraction", "train_size", "eval_size")}
        base["mode"] = base["mode"] or base["variant"]
        base["trajectory_count"] = base["trajectory_count"] or base["count"] or base["train_count"]
        base["train_config"] = base["train_config"] or _train_config(run)
        base["train_corruption"] = run.get("corruption")
        base["train_fraction"] = run.get("fraction")
        for point in run.get("validation_curve", []):
            row = {**base, **point, "status": "complete", "split": "validation"}
            metric = _metric(row)
            if metric is not None:
                row["value"] = metric
                row["step"] = row.get("step", 0)
                flat.append(row)
            else:
                exclusions["nonfinite_evaluations"] += 1
        for evaluation in run.get("evaluations", []):
            row = {**base, **evaluation}
            row["status"] = "complete"
            metric = _metric(row)
            if metric is None:
                exclusions["nonfinite_evaluations"] += 1
                continue
            row["value"] = metric
            row["split"] = row.get("split", row.get("partition", "validation"))
            row["step"] = row.get("step", row.get("checkpoint", 0))
            row["eval_size"] = row.get("eval_size") or row.get("nodes")
            flat.append(row)
    return flat, exclusions


def _train_config(row: Mapping[str, Any]) -> str:
    sizes = row.get("train_sizes")
    size_label = "fixed" if not sizes else "sizes=" + ",".join(str(x) for x in sizes)
    return (f"{row.get('mechanism', 'uniform')}:{size_label}:n={row.get('train_count')}:"
            f"{row.get('corruption', 'clean')}@{row.get('fraction', 0)}")


def _metric(row: Mapping[str, Any]) -> float | None:
    for key in ("normalized_mse", "deterministic_rollout_mse", "stochastic_rollout_mse", "mse", "value"):
        if _finite(row.get(key)):
            return float(row[key])
    return None


def _mean_sd(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if _finite(value)]
    return {"mean": statistics.fmean(finite) if finite else None,
            "population_sd": statistics.pstdev(finite) if finite else None,
            "n": len(finite)}


def analyze_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique_runs, duplicate_runs = _deduplicate_runs(runs)
    evaluations, exclusions = _evaluation_rows(unique_runs)
    exclusions["duplicate_runs"] = duplicate_runs
    complete_runs = [row for row in unique_runs if row.get("status") == "complete"]
    suites: dict[str, Any] = {}
    for suite in sorted({str(row.get("suite", "unknown")) for row in complete_runs}):
        suite_rows = [row for row in evaluations if str(row.get("suite", "unknown")) == suite]
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in suite_rows:
            key = (row.get("domain"), row.get("mode"), row.get("train_config"),
                   row.get("trajectory_count"), row.get("corruption"), row.get("fraction"),
                   row.get("train_size"), row.get("eval_size"), row.get("split"), row.get("step"))
            grouped[key].append(row)
        aggregates = []
        for key, items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
            # Multiple graphs/evaluations from one seed are averaged before cross-seed SD.
            per_seed = {seed: statistics.fmean(r["value"] for r in items if r.get("seed") == seed)
                        for seed in {r.get("seed") for r in items}}
            precisions = [r.get("graph_quality", {}).get("precision") for r in items]
            recalls = [r.get("graph_quality", {}).get("recall") for r in items]
            aggregates.append({
                **dict(zip(("domain", "mode", "train_config", "trajectory_count", "corruption",
                            "fraction", "train_size", "eval_size", "split", "step"), key)),
                **_mean_sd(per_seed.values()), "seed_values": per_seed,
                "graph_precision": _mean_sd(value for value in precisions if _finite(value)),
                "graph_recall": _mean_sd(value for value in recalls if _finite(value)),
            })
        suites[suite] = {"aggregates": aggregates}

    selection_rows = []
    for run in complete_runs:
        strength = _strength(run)
        curve = run.get("validation_curve", [])
        if strength is None or not str(run.get("mode", "")).startswith("soft") or not curve:
            continue
        final = max(curve, key=lambda point: point.get("step", -1))
        metric = _metric(final)
        if metric is not None:
            selection_rows.append({"suite": run.get("suite"), "domain": run.get("domain"),
                                   "train_config": _train_config(run), "seed": run.get("seed"),
                                   "strength": strength, "split": "validation", "value": metric})
    selected = select_soft_strength(selection_rows)
    paired = _paired_contrasts(evaluations)
    efficiency = _efficiency_summary(complete_runs)
    learned = _learned_summary(complete_runs)
    return {
        "schema_version": 1,
        "input_run_count": len(runs),
        "complete_run_count": len(complete_runs),
        "exclusions": exclusions,
        "soft_strength_selection": [
            {"suite": key[0], "domain": key[1], "train_config": key[2], "strength": value}
            for key, value in sorted(selected.items(), key=lambda item: tuple(str(x) for x in item[0]))],
        "paired_contrasts": paired,
        "corruption_clean_none_contrasts": _corruption_reference_contrasts(evaluations),
        "efficiency": efficiency,
        "learned_coefficients": learned,
        "resources": _resource_summary(complete_runs),
        "suites": suites,
    }


def _deduplicate_runs(runs: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    """Keep the first exact run identity and report repeated artifact rows."""
    seen = set()
    unique = []
    duplicates = 0
    for row in runs:
        identity = (row.get("suite"), row.get("domain"), row.get("seed"), row.get("mode"),
                    row.get("mechanism"), tuple(row.get("train_sizes") or ()), row.get("train_count"),
                    row.get("corruption"), row.get("fraction"), row.get("node_identity"))
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        unique.append(row)
    return unique, duplicates


def _paired_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute exact-config contrasts, including strength-matched permuted controls."""
    results = []
    modes = {str(row.get("mode")) for row in rows}
    pairs = [("none", mode) for mode in sorted(modes - {"none"})]
    pairs += [(f"permuted{strength}", f"soft{strength}") for strength in ("1", "4")
              if {f"permuted{strength}", f"soft{strength}"} <= modes]
    strata: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = ("suite", "domain", "train_config", "trajectory_count", "corruption",
            "fraction", "eval_size", "split", "step")
    for row in rows:
        strata[tuple(row.get(key) for key in keys)].append(row)
    for stratum, items in sorted(strata.items(), key=lambda item: tuple(str(x) for x in item[0])):
        for baseline, treatment in pairs:
            result = paired_effects(items, baseline, treatment)
            if result["n_pairs"] or result["excluded"]["incomplete"] or result["excluded"]["unmatched"]:
                results.append({**dict(zip(keys, stratum)), "baseline": baseline,
                                "treatment": treatment, **result})
    return results


def _corruption_reference_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare retrained corruption models with clean-trained none on identical test graphs."""
    relevant = [row for row in rows if row.get("suite") == "corruption"
                and row.get("split") in ("test", "runtime_corruption")]
    references = {(row.get("domain"), row.get("seed"), row.get("graph_hash"),
                   row.get("corruption"), row.get("fraction")): row["value"]
                  for row in relevant if row.get("mode") == "none"
                  and row.get("train_corruption") == "clean"}
    grouped: dict[tuple[Any, ...], dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    unmatched: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for row in relevant:
        if row.get("train_corruption") in (None, "clean") or row.get("mode") == "none":
            continue
        group = (row.get("domain"), row.get("mode"), row.get("train_corruption"),
                 row.get("train_fraction"), row.get("corruption"), row.get("fraction"), row.get("eval_size"))
        ref_key = (row.get("domain"), row.get("seed"), row.get("graph_hash"),
                   row.get("corruption"), row.get("fraction"))
        if ref_key not in references:
            unmatched[group].add(row.get("seed"))
        else:
            grouped[group][row.get("seed")].append(references[ref_key] - row["value"])
    keys = ("domain", "mode", "train_corruption", "train_fraction", "eval_corruption",
            "eval_fraction", "eval_size")
    output = []
    for group in sorted(set(grouped) | set(unmatched), key=lambda item: tuple(str(x) for x in item)):
        effects = [{"seed": seed, "effect": statistics.fmean(values)}
                   for seed, values in sorted(grouped[group].items(), key=lambda item: str(item[0]))]
        stats = _mean_sd(item["effect"] for item in effects)
        output.append({**dict(zip(keys, group)), "baseline": "clean_none",
                       "effects": effects, **stats,
                       "unmatched_seeds": sorted(unmatched[group], key=str)})
    return output


def _efficiency_summary(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        if str(run.get("suite", "")).startswith("efficiency"):
            grouped[(run.get("suite"), run.get("domain"), run.get("mode"), run.get("seed"))].append(run)
    output = []
    for (suite, domain, mode, seed), items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        curves = []
        threshold_by_count = {}
        auc_by_count = {}
        for run in sorted(items, key=lambda item: item.get("train_count", 0)):
            count = run.get("train_count")
            points = [(point["step"], point["normalized_mse"])
                      for point in run.get("validation_curve", [])
                      if _finite(point.get("step")) and _finite(point.get("normalized_mse"))]
            curves.append({"trajectory_count": count, "points": points})
            budget = max((point[0] for point in points), default=0)
            auc_by_count[str(count)] = complete_normalized_auc(points, budget) if budget > 0 else None
            threshold_by_count[str(count)] = {
                key: {"threshold": run.get("thresholds", {}).get(key),
                      "s_epsilon": run.get("threshold_steps", {}).get(key),
                      "censored": run.get("threshold_steps", {}).get(key) is None}
                for key in ("10", "25", "50")}
        first = min(items, key=lambda item: item.get("train_count", 0)).get("validation_curve", [{}])[0]
        derived = efficiency_thresholds(curves, first.get("oracle_normalized_mse", 0),
                                        first.get("zero_normalized_mse", 0))
        output.append({"suite": suite, "domain": domain, "mode": mode, "seed": seed,
                       "normalization_trajectories_per_graph": min(
                           item.get("normalization_trajectories_per_graph", math.inf) for item in items),
                       "auc_by_train_count": auc_by_count,
                       "thresholds_by_train_count": threshold_by_count,
                       "minimum_count_thresholds": derived})
    return output


def _learned_summary(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        if run.get("mode") not in ("learned", "typed"):
            continue
        for point in run.get("validation_curve", []):
            coefficients = point.get("coefficients", [])
            flat = []
            for layer, layer_values in enumerate(coefficients):
                for head, head_values in enumerate(layer_values if isinstance(layer_values, list) else [layer_values]):
                    values = head_values if isinstance(head_values, list) else [head_values]
                    for channel, value in enumerate(values):
                        if _finite(value):
                            item = {"layer": layer, "head": head, "value": float(value)}
                            if isinstance(head_values, list):
                                item["channel"] = channel
                            flat.append(item)
            output.append({"suite": run.get("suite"), "domain": run.get("domain"), "mode": run.get("mode"),
                           "seed": run.get("seed"), "step": point.get("step"), "values": flat})
    return output


def _resource_summary(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row.get("suite"), row.get("mode"))].append(row)
    return [{"suite": key[0], "mode": key[1],
             "parameters": _mean_sd(row.get("parameters") for row in rows),
             "optimizer_examples": _mean_sd(row.get("optimizer_examples") for row in rows),
             "training_seconds": _mean_sd(row.get("training_seconds") for row in rows),
             "peak_rss_kib": max((row.get("process_peak_rss_kib", 0) for row in rows), default=0)}
            for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0]))]


def _strength(row: Mapping[str, Any]) -> float | None:
    if _finite(row.get("strength")):
        return float(row["strength"])
    mode = str(row.get("mode", ""))
    if mode.startswith("soft"):
        try:
            return float(mode.removeprefix("soft"))
        except ValueError:
            pass
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Stage 2 artifact analysis", "",
             f"Complete runs: {summary['complete_run_count']} / {summary['input_run_count']}.", "",
             "Incomplete runs and non-finite evaluations are excluded rather than counted as observations. "
             "Standard deviations are population SDs across seeds; transfer observations are averaged within seed first.", ""]
    exclusions = summary["exclusions"]
    lines += ["## Exclusions", "", "| reason | count |", "|---|---:|"]
    lines += [f"| {key.replace('_', ' ')} | {value} |" for key, value in exclusions.items()]
    for suite, content in summary["suites"].items():
        lines += ["", f"## {suite}", "",
                  "| domain | mode | config | count | corruption | fraction | train→eval | split | step | mean | pop. SD | seeds |",
                  "|---|---|---|---:|---|---:|---|---|---:|---:|---:|---:|"]
        for row in content["aggregates"]:
            size = f"{_fmt(row['train_size'])}→{_fmt(row['eval_size'])}"
            lines.append("| " + " | ".join(_fmt(row[key]) for key in (
                "domain", "mode", "train_config", "trajectory_count", "corruption", "fraction"))
                + f" | {size} | " + " | ".join(_fmt(row[key]) for key in (
                    "split", "step", "mean", "population_sd", "n")) + " |")
    if summary["soft_strength_selection"]:
        lines += ["", "## Validation-only soft-strength selection", "",
                  "| suite | domain | train config | selected strength |", "|---|---|---|---:|"]
        for row in summary["soft_strength_selection"]:
            lines.append("| " + " | ".join(_fmt(row[key]) for key in
                                                ("suite", "domain", "train_config", "strength")) + " |")
    lines += ["", "## Resource accounting", "",
              "| suite | mode | parameters | optimizer examples | training seconds | peak RSS KiB |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary["resources"]:
        lines.append(f"| {_fmt(row['suite'])} | {_fmt(row['mode'])} | "
                     f"{_fmt(row['parameters']['mean'])} | {_fmt(row['optimizer_examples']['mean'])} | "
                     f"{_fmt(row['training_seconds']['mean'])} | {_fmt(row['peak_rss_kib'])} |")
    lines += ["", "Thresholds and model choices use validation metrics only. Unreached efficiency thresholds are represented by null values with censoring, never by fabricated counts or speedups.", ""]
    return "\n".join(lines)


def write_plots(summary: Mapping[str, Any], output: Path) -> None:
    """Write standard scientific SVG plots; Matplotlib is an optional analysis dependency."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("--plots requires matplotlib; core analysis does not") from error

    efficiency = summary.get("efficiency", [])
    if efficiency:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in efficiency:
            by_mode[f"{row.get('suite')}:{row['mode']}"].append(row)
        for mode, mode_rows in sorted(by_mode.items()):
            counts = sorted({int(count) for row in mode_rows for count in row["auc_by_train_count"]})
            means, spreads = [], []
            for count in counts:
                values = [row["auc_by_train_count"].get(str(count)) for row in mode_rows]
                stats = _mean_sd(value for value in values if value is not None)
                means.append(stats["mean"]); spreads.append(stats["population_sd"])
            axes[0].errorbar(counts, means, yerr=spreads, marker="o", capsize=3, label=mode)
            for row in mode_rows:
                values = [row["auc_by_train_count"].get(str(count)) for count in counts]
                axes[0].plot(counts, values, alpha=.15, linewidth=.8)
        axes[0].set(xscale="log", xlabel="training trajectories", ylabel="normalized validation AUC",
                    title="Efficiency curves (mean ± population SD)")
        axes[0].legend(fontsize=8)
        threshold_labels = ("gap_10", "gap_25", "gap_50")
        x = range(len(threshold_labels))
        for mode, mode_rows in sorted(by_mode.items()):
            values = []
            for label in threshold_labels:
                reached = [row["minimum_count_thresholds"][label]["n_epsilon"] for row in mode_rows]
                reached = [value for value in reached if value is not None]
                values.append(statistics.fmean(reached) if reached else math.nan)
            axes[1].plot(x, values, marker="o", label=mode)
        axes[1].set(xticks=list(x), xticklabels=("10% gap", "25% gap", "50% gap"),
                    ylabel="minimum tested trajectories", title="Threshold sample efficiency")
        axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output / "efficiency.svg"); plt.close(fig)

    for suite in ("corruption", "transfer", "heterogeneous", "learned"):
        rows = summary.get("suites", {}).get(suite, {}).get("aggregates", [])
        terminal = [row for row in rows if row.get("split") in ("test", "runtime_corruption")
                    and _finite(row.get("mean"))]
        if not terminal:
            continue
        labels = [f"{row.get('mode')}\n{row.get('corruption') or ''} {row.get('eval_size') or ''}" for row in terminal]
        fig_width = max(8, min(20, len(terminal) * .34))
        fig, axis = plt.subplots(figsize=(fig_width, 5))
        positions = list(range(len(terminal)))
        axis.errorbar(positions, [row["mean"] for row in terminal],
                      yerr=[row["population_sd"] for row in terminal], fmt="o", capsize=3)
        axis.set(xticks=positions, xticklabels=labels, ylabel="normalized error",
                 title=f"{suite.capitalize()} test results (mean ± population SD)")
        axis.tick_params(axis="x", labelrotation=70)
        fig.tight_layout(); fig.savefig(output / f"{suite}.svg"); plt.close(fig)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(value)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="runner metrics JSONL")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--plots", action="store_true", help="write SVG figures (optional matplotlib)")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    runs = load_jsonl(args.input)
    summary = analyze_runs(runs)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "report.md").write_text(render_markdown(summary))
    if args.plots:
        write_plots(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
