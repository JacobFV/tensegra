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
    exclusions = {"incomplete_runs": 0, "nonfinite_evaluations": 0, "duplicate_evaluations": 0,
                  "nonfinite_coefficients": 0}
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
                row["metric_name"] = "normalized_mse"
                row["step"] = row.get("step", 0)
                flat.append(row)
            else:
                exclusions["nonfinite_evaluations"] += 1
            coefficients = _coefficient_values(point.get("coefficients", []))
            if any(not _finite(value) for value in coefficients):
                exclusions["nonfinite_coefficients"] += 1
        evaluation_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for evaluation in run.get("evaluations", []):
            identity = (evaluation.get("split"), evaluation.get("graph_hash"),
                        evaluation.get("trajectory_seed"), evaluation.get("corruption"),
                        evaluation.get("fraction"), evaluation.get("nodes"))
            evaluation_groups[identity].append(evaluation)
        for duplicate_rows in evaluation_groups.values():
            if len(duplicate_rows) > 1:
                exclusions["duplicate_evaluations"] += len(duplicate_rows)
                continue
            evaluation = duplicate_rows[0]
            row = {**base, **evaluation}
            row["status"] = "complete"
            if any(name in row and not _finite(row[name]) for name in _evaluation_metric_names()):
                exclusions["nonfinite_evaluations"] += 1
                continue
            metrics = _evaluation_metrics(row)
            if not metrics:
                exclusions["nonfinite_evaluations"] += 1
                continue
            row["split"] = row.get("split", row.get("partition", "validation"))
            row["step"] = row.get("step", row.get("checkpoint", 0))
            row["eval_size"] = row.get("eval_size") or row.get("nodes")
            for metric_name, metric in metrics.items():
                flat.append({**row, "metric_name": metric_name, "value": metric})
    return flat, exclusions


def _train_config(row: Mapping[str, Any]) -> str:
    sizes = row.get("train_sizes")
    size_label = "fixed" if not sizes else "sizes=" + ",".join(str(x) for x in sizes)
    return (f"{row.get('mechanism', 'uniform')}:{size_label}:n={row.get('train_count')}:"
            f"{row.get('corruption', 'clean')}@{row.get('fraction', 0)}")


def _metric(row: Mapping[str, Any]) -> float | None:
    for key in ("normalized_mse", "value"):
        if _finite(row.get(key)):
            return float(row[key])
    return None


def _evaluation_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in _evaluation_metric_names() if _finite(row.get(name))}


def _evaluation_metric_names() -> tuple[str, ...]:
    return tuple(f"{prefix}{outcome}{suffix}" for prefix in ("", "oracle_", "zero_")
                 for outcome in ("one_step", "deterministic_rollout", "stochastic_rollout")
                 for suffix in ("_mse", "_normalized_mse"))


def _coefficient_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for child in value for item in _coefficient_values(child)]
    return [value]


def _mean_sd(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if _finite(value)]
    return {"mean": statistics.fmean(finite) if finite else None,
            "population_sd": statistics.pstdev(finite) if finite else None,
            "n": len(finite)}


def analyze_runs(runs: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not config.get("checkpoints") or not config.get("counts"):
        raise ValueError("analysis config requires checkpoints and counts")
    budget = max(config["checkpoints"])
    unique_runs, duplicate_counts = _deduplicate_runs(runs)
    evaluations, exclusions = _evaluation_rows(unique_runs)
    exclusions.update(duplicate_counts)
    complete_runs = [row for row in unique_runs if row.get("status") == "complete"]
    suites: dict[str, Any] = {}
    for suite in sorted({str(row.get("suite", "unknown")) for row in complete_runs}):
        suite_rows = [row for row in evaluations if str(row.get("suite", "unknown")) == suite]
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in suite_rows:
            key = (row.get("domain"), row.get("mode"), row.get("train_config"),
                   row.get("trajectory_count"), row.get("corruption"), row.get("fraction"),
                   row.get("train_size"), row.get("eval_size"), row.get("split"), row.get("step"),
                   row.get("metric_name"))
            grouped[key].append(row)
        aggregates = []
        for key, items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
            # Multiple graphs/evaluations from one seed are averaged before cross-seed SD.
            per_seed = {seed: statistics.fmean(r["value"] for r in items if r.get("seed") == seed)
                        for seed in {r.get("seed") for r in items}}
            seeds = {r.get("seed") for r in items}
            precision_by_seed = [statistics.fmean(r["graph_quality"]["precision"] for r in items
                                                  if r.get("seed") == seed and _finite(r.get("graph_quality", {}).get("precision")))
                                 for seed in seeds if any(r.get("seed") == seed and _finite(r.get("graph_quality", {}).get("precision")) for r in items)]
            recall_by_seed = [statistics.fmean(r["graph_quality"]["recall"] for r in items
                                               if r.get("seed") == seed and _finite(r.get("graph_quality", {}).get("recall")))
                              for seed in seeds if any(r.get("seed") == seed and _finite(r.get("graph_quality", {}).get("recall")) for r in items)]
            aggregates.append({
                "suite": suite,
                **dict(zip(("domain", "mode", "train_config", "trajectory_count", "corruption",
                            "fraction", "train_size", "eval_size", "split", "step", "metric_name"), key)),
                **_mean_sd(per_seed.values()),
                "seed_values": [{"seed": seed, "value": value,
                                 "n_graphs": sum(r.get("seed") == seed for r in items)}
                                for seed, value in sorted(per_seed.items(), key=lambda item: str(item[0]))],
                "n_graphs": len(items), "graph_precision": _mean_sd(precision_by_seed),
                "graph_recall": _mean_sd(recall_by_seed),
            })
        suites[suite] = {"aggregates": aggregates}

    selection_rows = []
    for run in complete_runs:
        strength = _strength(run)
        curve = run.get("validation_curve", [])
        if strength is None or not str(run.get("mode", "")).startswith("soft") or not curve:
            continue
        final = next((point for point in curve if point.get("step") == budget), None)
        if final is None:
            continue
        metric = _metric(final)
        if metric is not None:
            selection_rows.append({"suite": run.get("suite"), "domain": run.get("domain"),
                                   "train_config": _train_config(run), "seed": run.get("seed"),
                                   "strength": strength, "split": "validation", "value": metric})
    selected = select_soft_strength(selection_rows)
    selection = select_soft_strength(selection_rows)
    for content in suites.values():
        for row in content["aggregates"]:
            strength = _strength(row)
            key = (row.get("suite"), row.get("domain"), row.get("train_config"))
            row["selected_by_validation"] = (strength is not None and selection.get(key) == strength)
    paired = _paired_contrasts(evaluations, selection)
    efficiency, efficiency_exclusions = _efficiency_summary(complete_runs, config)
    exclusions.update(efficiency_exclusions)
    learned = _learned_summary(complete_runs)
    return {
        "schema_version": 1,
        "input_run_count": len(runs),
        "complete_run_count": len(complete_runs),
        "exclusions": exclusions,
        "analysis_config": {"budget": budget, "counts": list(config["counts"]),
                            "checkpoints": list(config["checkpoints"])},
        "soft_strength_selection": [
            {"suite": key[0], "domain": key[1], "train_config": key[2], "strength": value}
            for key, value in sorted(selected.items(), key=lambda item: tuple(str(x) for x in item[0]))],
        "paired_contrasts": paired,
        "corruption_clean_none_contrasts": _corruption_reference_contrasts(evaluations, selection),
        "efficiency": efficiency,
        "learned_coefficients": learned,
        "resources": _resource_summary(complete_runs),
        "suites": suites,
    }


def _deduplicate_runs(runs: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Collapse identical run rows and exclude every conflicting run identity."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        identity = (row.get("suite"), row.get("domain"), row.get("seed"), row.get("mode"),
                    row.get("mechanism"), tuple(row.get("train_sizes") or ()), row.get("train_count"),
                    row.get("corruption"), row.get("fraction"), row.get("node_identity"))
        grouped[identity].append(row)
    unique = []
    identical = conflicting = 0
    for rows in grouped.values():
        serialized = {json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows}
        if len(serialized) > 1:
            conflicting += len(rows)
            continue
        unique.append(rows[0])
        identical += len(rows) - 1
    return unique, {"duplicate_runs_identical": identical, "duplicate_runs_conflicting": conflicting}


def _paired_contrasts(rows: Sequence[Mapping[str, Any]], selection: Mapping[tuple[Any, Any, Any], float]) -> list[dict[str, Any]]:
    """Compute exact-config contrasts, including strength-matched permuted controls."""
    results = []
    modes = {str(row.get("mode")) for row in rows}
    pairs = [("none", mode) for mode in sorted(modes - {"none"})]
    pairs += [(f"permuted{strength}", f"soft{strength}") for strength in ("1", "4")
              if {f"permuted{strength}", f"soft{strength}"} <= modes]
    strata: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = ("suite", "domain", "train_config", "trajectory_count", "corruption",
            "fraction", "eval_size", "split", "step", "metric_name")
    for row in rows:
        strata[tuple(row.get(key) for key in keys)].append(row)
    for stratum, items in sorted(strata.items(), key=lambda item: tuple(str(x) for x in item[0])):
        for baseline, treatment in pairs:
            result = _matched_pair_summary(items, baseline, treatment)
            if result["n_pairs"] or result["excluded"]["incomplete"] or result["excluded"]["unmatched"]:
                results.append({**dict(zip(keys, stratum)), "baseline": baseline,
                                "treatment": treatment, **result,
                                "selected_by_validation": _mode_selected(
                                    treatment, stratum[0], stratum[1], stratum[2], selection)})
    return results


def _mode_selected(mode: str, suite: Any, domain: Any, train_config: Any,
                   selection: Mapping[tuple[Any, Any, Any], float]) -> bool:
    strength = _strength({"mode": mode})
    return strength is not None and selection.get((suite, domain, train_config)) == strength


def _matched_pair_summary(items: Sequence[Mapping[str, Any]], baseline: str,
                          treatment: str) -> dict[str, Any]:
    by_seed: dict[Any, dict[str, dict[tuple[Any, ...], list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    incomplete: set[Any] = set()
    for row in items:
        mode = row.get("mode")
        if mode not in (baseline, treatment):
            continue
        seed = row.get("seed")
        if row.get("status", "complete") != "complete" or not _finite(row.get("value")):
            incomplete.add(seed); continue
        identity = (("validation", row.get("step")) if row.get("split") == "validation" else
                    (row.get("graph_hash"), row.get("trajectory_seed")))
        by_seed[seed][str(mode)][identity].append(float(row["value"]))
    effects, unmatched, duplicates = [], [], []
    for seed in sorted(by_seed, key=str):
        modes = by_seed[seed]
        duplicate_keys = [key for mode in (baseline, treatment) for key, values in modes[mode].items()
                          if len(values) != 1]
        if duplicate_keys:
            duplicates.append({"seed": seed, "keys": [list(key) for key in duplicate_keys]}); continue
        baseline_keys, treatment_keys = set(modes[baseline]), set(modes[treatment])
        if baseline_keys != treatment_keys or not baseline_keys:
            unmatched.append({"seed": seed,
                              "baseline_only": [list(key) for key in sorted(baseline_keys - treatment_keys, key=str)],
                              "treatment_only": [list(key) for key in sorted(treatment_keys - baseline_keys, key=str)]})
            continue
        graph_effects = [modes[baseline][key][0] - modes[treatment][key][0]
                         for key in sorted(baseline_keys, key=str)]
        effects.append({"seed": seed, "effect": statistics.fmean(graph_effects),
                        "n_references": len(graph_effects)})
    values = [row["effect"] for row in effects]
    return {"effects": effects, "mean": statistics.fmean(values) if values else None,
            "population_sd": statistics.pstdev(values) if values else None, "n_pairs": len(effects),
            "excluded": {"incomplete": sorted(incomplete, key=str), "unmatched": unmatched,
                         "duplicate_references": duplicates}}


def _corruption_reference_contrasts(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[tuple[Any, Any, Any], float]
) -> list[dict[str, Any]]:
    """Report retraining and runtime-corruption estimands with exact references."""
    relevant = [row for row in rows if row.get("suite") == "corruption"
                and row.get("split") in ("test", "runtime_corruption")]
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for target in relevant:
        mode = target.get("mode")
        if mode == "none":
            continue
        train_corruption = target.get("train_corruption")
        if train_corruption != "clean" and target.get("split") == "test":
            estimand = "retrained_vs_clean_none"
        elif train_corruption == "clean" and target.get("split") == "runtime_corruption":
            estimand = "runtime_corruption_clean_trained"
        else:
            continue
        group = (estimand, target.get("domain"), mode, train_corruption,
                 target.get("train_fraction"), target.get("corruption"), target.get("fraction"),
                 target.get("eval_size"), target.get("metric_name"), target.get("train_config"))
        groups[group].append({**target, "mode": "treatment"})
        for reference in relevant:
            if (reference.get("mode") == "none" and reference.get("train_corruption") == "clean"
                    and reference.get("corruption") == target.get("corruption")
                    and reference.get("fraction") == target.get("fraction")
                    and reference.get("metric_name") == target.get("metric_name")
                    and reference.get("domain") == target.get("domain")
                    and reference.get("seed") == target.get("seed")
                    and reference.get("graph_hash") == target.get("graph_hash")
                    and reference.get("trajectory_seed") == target.get("trajectory_seed")):
                groups[group].append({**reference, "mode": "reference"})
    keys = ("estimand", "domain", "mode", "train_corruption", "train_fraction",
            "eval_corruption", "eval_fraction", "eval_size", "metric_name", "train_config")
    output = []
    for group, items in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        result = _matched_pair_summary(items, "reference", "treatment")
        output.append({**dict(zip(keys, group)), "baseline": "clean_none", **result,
                       "selected_by_validation": _mode_selected(
                           str(group[2]), "corruption", group[1], group[9], selection)})
    return output


def _efficiency_summary(runs: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[Any, Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        if str(run.get("suite", "")).startswith("efficiency"):
            grouped[(run.get("suite"), run.get("domain"), run.get("mode"), run.get("seed"))].append(run)
    output = []
    exclusions = {"invalid_efficiency_curves": 0, "efficiency_reference_mismatch": 0,
                  "efficiency_normalization_mismatch": 0}
    budget = max(config["checkpoints"])
    expected_counts = set(config["counts"])
    for (suite, domain, mode, seed), items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        counts_present = [item.get("train_count") for item in items]
        if len(counts_present) != len(set(counts_present)) or set(counts_present) != expected_counts:
            exclusions["invalid_efficiency_curves"] += len(items)
            continue
        normalizations = {json.dumps(item.get("normalization"), sort_keys=True) for item in items}
        provenance = {item.get("normalization_trajectories_per_graph") for item in items}
        if len(normalizations) != 1 or len(provenance) != 1 or provenance != {min(expected_counts)}:
            exclusions["efficiency_normalization_mismatch"] += len(items)
            continue
        curves = []
        threshold_by_count = {}
        auc_by_count = {}
        for run in sorted(items, key=lambda item: item.get("train_count", 0)):
            count = run.get("train_count")
            points = [(point["step"], point["normalized_mse"])
                      for point in run.get("validation_curve", [])
                      if _finite(point.get("step")) and _finite(point.get("normalized_mse"))]
            steps = [point[0] for point in points]
            valid_curve = (len(steps) == len(set(steps)) and steps == sorted(steps)
                           and steps and steps[0] == 0 and steps[-1] == budget)
            if not valid_curve:
                exclusions["invalid_efficiency_curves"] += 1
            else:
                curves.append({"trajectory_count": count, "points": points})
            auc_by_count[str(count)] = complete_normalized_auc(points, budget) if valid_curve else None
            threshold_by_count[str(count)] = {
                key: {"threshold": run.get("thresholds", {}).get(key),
                      "s_epsilon": run.get("threshold_steps", {}).get(key),
                      "censored": run.get("threshold_steps", {}).get(key) is None}
                for key in ("10", "25", "50")}
        starts = [item.get("validation_curve", [{}])[0] for item in items if item.get("validation_curve")]
        references = {(point.get("oracle_normalized_mse"), point.get("zero_normalized_mse")) for point in starts}
        if len(references) != 1:
            exclusions["efficiency_reference_mismatch"] += len(items)
            continue
        oracle, zero = next(iter(references))
        if not (_finite(oracle) and _finite(zero)) or zero <= oracle:
            derived = efficiency_thresholds([], 0.0, 0.0)
        else:
            derived = efficiency_thresholds(curves, float(oracle), float(zero))
            for run in items:
                for percent, label in ((10, "gap_10"), (25, "gap_25"), (50, "gap_50")):
                    stored = run.get("thresholds", {}).get(str(percent))
                    if not _finite(stored) or not math.isclose(float(stored), derived[label]["threshold"]):
                        exclusions["efficiency_reference_mismatch"] += 1
        output.append({"suite": suite, "domain": domain, "mode": mode, "seed": seed,
                       "normalization_trajectories_per_graph": min(
                           item.get("normalization_trajectories_per_graph", math.inf) for item in items),
                       "auc_by_train_count": auc_by_count,
                       "budget": budget,
                       "thresholds_by_train_count": threshold_by_count,
                       "minimum_count_thresholds": derived})
    return output, exclusions


def _learned_summary(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        if run.get("mode") not in ("learned", "typed"):
            continue
        for point in run.get("validation_curve", []):
            coefficients = point.get("coefficients", [])
            if any(not _finite(value) for value in _coefficient_values(coefficients)):
                continue
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
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row.get("suite"), row.get("domain"), row.get("mode"), _train_config(row),
                 row.get("train_count"), row.get("mechanism"), row.get("node_identity"))].append(row)
    return [{"suite": key[0], "domain": key[1], "mode": key[2], "train_config": key[3],
             "train_count": key[4], "mechanism": key[5], "node_identity": key[6],
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
                  "| domain | mode | selected | config | count | corruption | fraction | train→eval | split | metric | mean | pop. SD | seeds | graphs | precision | recall |",
                  "|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
        primary_metrics = {"one_step_normalized_mse", "deterministic_rollout_normalized_mse",
                           "stochastic_rollout_normalized_mse"}
        terminal = [row for row in content["aggregates"] if row.get("split") in ("test", "runtime_corruption")
                    and row.get("metric_name") in primary_metrics]
        for row in terminal:
            size = f"{_fmt(row['train_size'])}→{_fmt(row['eval_size'])}"
            lines.append("| " + " | ".join(_fmt(value) for value in (
                row["domain"], row["mode"], row["selected_by_validation"], row["train_config"],
                row["trajectory_count"], row["corruption"], row["fraction"]))
                + f" | {size} | " + " | ".join(_fmt(row[key]) for key in (
                    "split", "metric_name", "mean", "population_sd", "n", "n_graphs"))
                + f" | {_fmt(row['graph_precision']['mean'])} | {_fmt(row['graph_recall']['mean'])} |")
    if summary["efficiency"]:
        lines += ["", "## Efficiency thresholds and AUC by seed", "",
                  "| suite | domain | mode | seed | train count | AUC | s10 | s25 | s50 | n10 | n25 | n50 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary["efficiency"]:
            minimum = row["minimum_count_thresholds"]
            for count, auc in sorted(row["auc_by_train_count"].items(), key=lambda item: int(item[0])):
                steps = row["thresholds_by_train_count"][count]
                lines.append("| " + " | ".join(_fmt(value) for value in (
                    row["suite"], row["domain"], row["mode"], row["seed"], count, auc,
                    steps["10"]["s_epsilon"], steps["25"]["s_epsilon"], steps["50"]["s_epsilon"],
                    minimum["gap_10"]["n_epsilon"], minimum["gap_25"]["n_epsilon"],
                    minimum["gap_50"]["n_epsilon"])) + " |")
    if summary["learned_coefficients"]:
        lines += ["", "## Learned signed coefficients", "",
                  "| suite | domain | mode | seed | step | coefficients | min | mean | max |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in summary["learned_coefficients"]:
            values = [item["value"] for item in row["values"]]
            lines.append("| " + " | ".join(_fmt(value) for value in (
                row["suite"], row["domain"], row["mode"], row["seed"], row["step"], len(values),
                min(values) if values else None, statistics.fmean(values) if values else None,
                max(values) if values else None)) + " |")
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
    """Write standard scientific SVG and PNG plots; Matplotlib is optional."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("--plots requires matplotlib; core analysis does not") from error

    efficiency_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary.get("efficiency", []):
        efficiency_groups[(str(row.get("suite")), str(row.get("domain")))].append(row)
    for (suite, domain), efficiency in efficiency_groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in efficiency:
            by_mode[str(row["mode"])].append(row)
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
        max_count = max(int(count) for row in efficiency for count in row["auc_by_train_count"])
        offsets = {mode: (index - (len(by_mode) - 1) / 2) * .04
                   for index, mode in enumerate(sorted(by_mode))}
        colors = {mode: plt.cm.tab10(index % 10) for index, mode in enumerate(sorted(by_mode))}
        for mode, mode_rows in sorted(by_mode.items()):
            labeled = False
            for row in mode_rows:
                for index, label in enumerate(threshold_labels):
                    value = row["minimum_count_thresholds"][label]["n_epsilon"]
                    if value is None:
                        axes[1].scatter(index + offsets[mode], max_count, marker="x", alpha=.7,
                                        color=colors[mode], label=mode if not labeled else None)
                    else:
                        axes[1].scatter(index + offsets[mode], value, marker="o", alpha=.7,
                                        color=colors[mode], label=mode if not labeled else None)
                    labeled = True
        axes[1].set(xticks=list(x), xticklabels=("10% gap", "25% gap", "50% gap"),
                    ylabel="minimum tested trajectories",
                    title=f"Thresholds (× = censored at {max_count})")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{suite}: {domain}")
        fig.tight_layout()
        stem = output / f"{suite}-{domain}-efficiency"
        fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".png"), dpi=160)
        plt.close(fig)

    for suite in sorted(summary.get("suites", {})):
        if str(suite).startswith("efficiency"):
            continue
        rows = summary.get("suites", {}).get(suite, {}).get("aggregates", [])
        terminal = [row for row in rows if row.get("split") in ("test", "runtime_corruption")
                    and row.get("metric_name") in {"one_step_normalized_mse",
                                                   "deterministic_rollout_normalized_mse",
                                                   "stochastic_rollout_normalized_mse"}
                    and _finite(row.get("mean"))]
        facets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in terminal:
            facets[(str(row.get("domain")), str(row.get("metric_name")))].append(row)
        for (domain, metric_name), facet_rows in facets.items():
            regimes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in facet_rows:
                regimes[str(row.get("train_config"))].append(row)
            fig, axes = plt.subplots(len(regimes), 1, squeeze=False,
                                     figsize=(12, max(4, 3.2 * len(regimes))))
            for axis, (regime, regime_rows) in zip(axes[:, 0], sorted(regimes.items())):
                labels = [f"{row.get('mode')}\n{row.get('corruption') or ''} {row.get('eval_size') or ''}"
                          for row in regime_rows]
                positions = list(range(len(regime_rows)))
                axis.errorbar(positions, [row["mean"] for row in regime_rows],
                              yerr=[row["population_sd"] for row in regime_rows], fmt="o", capsize=3)
                axis.set(xticks=positions, xticklabels=labels, ylabel="normalized MSE", title=regime)
                axis.tick_params(axis="x", labelrotation=45)
            fig.suptitle(f"{suite}: {domain} — {metric_name}")
            fig.tight_layout()
            safe_metric = metric_name.replace("_normalized_mse", "")
            stem = output / f"{suite}-{domain}-{safe_metric}"
            fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".png"), dpi=160)
            plt.close(fig)


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
    parser.add_argument("--config", type=Path,
                        help="study config JSON; default is input sibling summary.json")
    parser.add_argument("--plots", action="store_true", help="write SVG figures (optional matplotlib)")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    runs = load_jsonl(args.input)
    config_path = args.config or args.input.with_name("summary.json")
    if not config_path.exists():
        raise FileNotFoundError(f"analysis requires study config: {config_path}")
    config_artifact = json.loads(config_path.read_text())
    config = config_artifact.get("config", config_artifact)
    summary = analyze_runs(runs, config)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "report.md").write_text(render_markdown(summary))
    if args.plots:
        write_plots(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
