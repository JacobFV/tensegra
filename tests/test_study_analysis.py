import json

import pytest

from tensegra.study_analysis import (
    analyze_runs,
    complete_normalized_auc,
    efficiency_thresholds,
    paired_effects,
    select_soft_strength,
    write_plots,
)


def test_efficiency_thresholds_keep_unreached_counts_censored():
    curves = [
        {"trajectory_count": 8, "points": [(0, 1.0), (25, 0.8), (50, 0.7)]},
        {"trajectory_count": 16, "points": [(0, 1.0), (25, 0.6), (50, 0.41)]},
    ]

    result = efficiency_thresholds(curves, oracle=0.2, zero=1.0)

    assert result["gap_50"]["threshold"] == pytest.approx(0.6)
    assert {key: result["gap_50"][key] for key in ("n_epsilon", "s_epsilon", "censored")} == {
        "n_epsilon": 16, "s_epsilon": 25, "censored": False}
    assert result["gap_25"]["threshold"] == pytest.approx(0.4)
    assert {key: result["gap_25"][key] for key in ("n_epsilon", "s_epsilon", "censored")} == {
        "n_epsilon": None, "s_epsilon": None, "censored": True}
    assert result["gap_10"]["threshold"] == pytest.approx(0.28)
    assert result["gap_10"]["censored"] is True


def test_auc_requires_zero_and_complete_budget_and_uses_trapezoids():
    assert complete_normalized_auc([(0, 1.0), (50, 0.5), (100, 0.0)], 100) == pytest.approx(0.5)
    assert complete_normalized_auc([(50, 0.5), (100, 0.0)], 100) is None
    assert complete_normalized_auc([(0, 1.0), (50, 0.5)], 100) is None


def test_paired_effects_exclude_incomplete_and_mismatched_seed_pairs():
    rows = [
        {"seed": 0, "mode": "none", "value": 5.0, "status": "complete"},
        {"seed": 0, "mode": "soft4", "value": 3.0, "status": "complete"},
        {"seed": 1, "mode": "none", "value": 6.0, "status": "complete"},
        {"seed": 1, "mode": "soft4", "value": 2.0, "status": "partial"},
        {"seed": 2, "mode": "soft4", "value": 1.0, "status": "complete"},
    ]

    result = paired_effects(rows, baseline="none", treatment="soft4")

    assert result["effects"] == [{"seed": 0, "effect": 2.0}]
    assert result["mean"] == 2.0
    assert result["population_sd"] == 0.0
    assert result["excluded"] == {"incomplete": [1], "unmatched": [2]}


def test_paired_effects_average_repeated_graphs_within_seed_before_effect():
    rows = [
        {"seed": 0, "mode": "none", "value": 5.0},
        {"seed": 0, "mode": "none", "value": 7.0},
        {"seed": 0, "mode": "soft4", "value": 1.0},
        {"seed": 0, "mode": "soft4", "value": 3.0},
    ]
    assert paired_effects(rows, "none", "soft4")["effects"] == [{"seed": 0, "effect": 4.0}]


def test_soft_strength_selection_uses_validation_across_seeds_only():
    rows = [
        {"suite": "corruption", "domain": "sparse", "train_config": "drop25", "seed": 0,
         "strength": 1.0, "split": "validation", "value": 4.0},
        {"suite": "corruption", "domain": "sparse", "train_config": "drop25", "seed": 1,
         "strength": 1.0, "split": "validation", "value": 2.0},
        {"suite": "corruption", "domain": "sparse", "train_config": "drop25", "seed": 0,
         "strength": 4.0, "split": "validation", "value": 2.0},
        {"suite": "corruption", "domain": "sparse", "train_config": "drop25", "seed": 1,
         "strength": 4.0, "split": "validation", "value": 2.0},
        # Test favors strength 1, and must not influence the choice.
        {"suite": "corruption", "domain": "sparse", "train_config": "drop25", "seed": 0,
         "strength": 1.0, "split": "test", "value": 0.0},
        {"suite": "corruption", "domain": "sparse", "train_config": "drop25", "seed": 0,
         "strength": 4.0, "split": "test", "value": 10.0},
    ]

    selected = select_soft_strength(rows)

    assert selected == {("corruption", "sparse", "drop25"): 4.0}


def test_soft_strength_requires_matched_validation_seed_sets():
    rows = [
        {"suite": "efficiency", "domain": "sparse", "train_config": "n8", "seed": 0,
         "strength": 1.0, "split": "validation", "value": 1.0},
        {"suite": "efficiency", "domain": "sparse", "train_config": "n8", "seed": 0,
         "strength": 4.0, "split": "validation", "value": 2.0},
        {"suite": "efficiency", "domain": "sparse", "train_config": "n8", "seed": 1,
         "strength": 4.0, "split": "validation", "value": 0.0},
    ]

    assert select_soft_strength(rows) == {("efficiency", "sparse", "n8"): 1.0}


def test_analysis_preserves_signed_coefficients_per_layer_head_and_channel():
    run = {
        "suite": "heterogeneous", "domain": "sparse", "seed": 0, "mode": "typed",
        "mechanism": "signed", "train_count": 8, "corruption": "clean", "fraction": 0,
        "status": "complete", "validation_curve": [
            {"step": 0, "normalized_mse": 1.0, "coefficients": [[[-0.2, 0.3], [0.4, -0.5]]]}
        ], "evaluations": [],
    }

    values = analyze_runs([run], {"checkpoints": [0], "counts": [8]})["learned_coefficients"][0]["values"]

    assert values == [
        {"layer": 0, "head": 0, "channel": 0, "value": -0.2},
        {"layer": 0, "head": 0, "channel": 1, "value": 0.3},
        {"layer": 0, "head": 1, "channel": 0, "value": 0.4},
        {"layer": 0, "head": 1, "channel": 1, "value": -0.5},
    ]


def _evaluation(graph, trajectory, offset=0.0):
    row = {"split": "test", "nodes": 12, "graph_hash": graph,
           "trajectory_seed": trajectory, "corruption": "clean", "fraction": 0,
           "graph_quality": {"precision": 1.0, "recall": 1.0}}
    value = 1.0 + offset
    for prefix in ("", "oracle_", "zero_"):
        for outcome in ("one_step", "deterministic_rollout", "stochastic_rollout"):
            row[f"{prefix}{outcome}_mse"] = value
            value += 1
            row[f"{prefix}{outcome}_normalized_mse"] = value
            value += 1
    return row


def _run(mode, evaluations, curve=None):
    return {"suite": "transfer", "domain": "sparse", "seed": 0, "mode": mode,
            "mechanism": "uniform", "train_sizes": [12], "train_count": 8,
            "corruption": "clean", "fraction": 0, "status": "complete",
            "schedule_hash": "schedule-0", "shared_initialization_hash": "initial-0",
            "completed_steps": 0,
            "normalization": {"mean": 0.0, "std": 1.0},
            "normalization_trajectories_per_graph": 8,
            "validation_curve": curve or [{"step": 0, "normalized_mse": 2.0,
                                            "oracle_normalized_mse": .1,
                                            "zero_normalized_mse": 3.0,
                                            "coefficients": []}],
            "evaluations": evaluations, "parameters": 10, "optimizer_examples": 0,
            "training_seconds": 1.0, "process_peak_rss_kib": 100}


def test_runner_metrics_keep_exact_names_and_units():
    summary = analyze_runs([_run("none", [_evaluation("g0", 10)])],
                           {"checkpoints": [0], "counts": [8]})
    rows = summary["suites"]["transfer"]["aggregates"]
    metrics = {row["metric_name"]: row["mean"] for row in rows if row["split"] == "test"}
    assert len(metrics) == 18
    assert metrics["one_step_mse"] == 1.0
    assert metrics["one_step_normalized_mse"] == 2.0
    assert metrics["zero_stochastic_rollout_normalized_mse"] == 18.0


def test_pairing_rejects_missing_graph_and_trajectory_mismatch():
    baseline = _run("none", [_evaluation("g0", 10), _evaluation("g1", 11)])
    treatment = _run("soft4", [_evaluation("g0", 10, 1), _evaluation("g1", 99, 1)])
    summary = analyze_runs([baseline, treatment], {"checkpoints": [0], "counts": [8]})
    contrast = next(row for row in summary["paired_contrasts"]
                    if row["metric_name"] == "one_step_normalized_mse")
    assert contrast["n_pairs"] == 0
    assert contrast["excluded"]["unmatched"]


def test_pairing_matches_reordered_graphs_and_averages_effects_within_seed():
    baseline = _run("none", [_evaluation("g0", 10, 4), _evaluation("g1", 11, 6)])
    treatment = _run("soft4", [_evaluation("g1", 11, 2), _evaluation("g0", 10, 2)])
    summary = analyze_runs([baseline, treatment], {"checkpoints": [0], "counts": [8]})
    contrast = next(row for row in summary["paired_contrasts"]
                    if row["metric_name"] == "one_step_normalized_mse")
    assert contrast["effects"] == [{"seed": 0, "effect": 3.0, "n_references": 2}]


@pytest.mark.parametrize("field", ["schedule_hash", "shared_initialization_hash"])
def test_pairing_rejects_mismatched_or_missing_provenance(field):
    baseline = _run("none", [_evaluation("g0", 10)])
    treatment = _run("soft4", [_evaluation("g0", 10, 1)])
    treatment[field] = "different"
    summary = analyze_runs([baseline, treatment], {"checkpoints": [0], "counts": [8]})
    contrast = next(row for row in summary["paired_contrasts"]
                    if row["metric_name"] == "one_step_normalized_mse")
    assert contrast["n_pairs"] == 0
    assert contrast["excluded"]["pairing_provenance_mismatch"]

    treatment.pop(field)
    summary = analyze_runs([baseline, treatment], {"checkpoints": [0], "counts": [8]})
    contrast = next(row for row in summary["paired_contrasts"]
                    if row["metric_name"] == "one_step_normalized_mse")
    assert contrast["n_pairs"] == 0
    assert contrast["excluded"]["pairing_provenance_mismatch"]


def test_efficiency_auc_uses_configured_budget_not_observed_maximum():
    run = _run("none", [], curve=[
        {"step": 0, "normalized_mse": 1.0, "oracle_normalized_mse": .1,
         "zero_normalized_mse": 2.0, "coefficients": []},
        {"step": 300, "normalized_mse": .5, "oracle_normalized_mse": .1,
         "zero_normalized_mse": 2.0, "coefficients": []},
    ])
    run.update(suite="efficiency", train_sizes=None,
               thresholds={"10": .29, "25": .575, "50": 1.05},
               threshold_steps={"10": None, "25": 300, "50": 0})
    summary = analyze_runs([run], {"checkpoints": [0, 300, 600], "counts": [8]})
    assert summary["efficiency"][0]["auc_by_train_count"]["8"] is None
    assert summary["exclusions"]["invalid_efficiency_curves"] == 1


def test_duplicate_evaluation_identity_is_excluded():
    duplicate = _evaluation("g0", 10)
    summary = analyze_runs([_run("none", [duplicate, dict(duplicate)])],
                           {"checkpoints": [0], "counts": [8]})
    assert summary["exclusions"]["duplicate_evaluations"] == 2
    assert not [row for row in summary["suites"]["transfer"]["aggregates"]
                if row["split"] == "test"]


def test_downstream_rows_mark_validation_selected_strength_despite_test_preference():
    soft1 = _run("soft1", [_evaluation("g0", 10, 10)], curve=[
        {"step": 0, "normalized_mse": 1.0, "oracle_normalized_mse": .1,
         "zero_normalized_mse": 2.0, "coefficients": []}])
    soft4 = _run("soft4", [_evaluation("g0", 10, 0)], curve=[
        {"step": 0, "normalized_mse": 2.0, "oracle_normalized_mse": .1,
         "zero_normalized_mse": 2.0, "coefficients": []}])
    summary = analyze_runs([soft1, soft4], {"checkpoints": [0], "counts": [8]})
    rows = [row for row in summary["suites"]["transfer"]["aggregates"]
            if row["metric_name"] == "one_step_normalized_mse"]
    assert {row["mode"]: row["selected_by_validation"] for row in rows} == {
        "soft1": True, "soft4": False}


def test_test_metrics_are_labeled_with_completed_training_step():
    run = _run("none", [_evaluation("g0", 10)])
    run["completed_steps"] = 600
    summary = analyze_runs([run], {"checkpoints": [0, 600], "counts": [8]})
    test_rows = [row for row in summary["suites"]["transfer"]["aggregates"]
                 if row["split"] == "test"]
    assert test_rows and {row["step"] for row in test_rows} == {600}


def test_transfer_plot_writes_svg_and_png_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    evaluations = []
    for index, nodes in enumerate((12, 32, 64, 128)):
        evaluation = _evaluation(f"g{nodes}", 10 + index, offset=index)
        evaluation["nodes"] = nodes
        evaluations.append(evaluation)
    summary = analyze_runs([_run("none", evaluations)], {"checkpoints": [0], "counts": [8]})

    write_plots(summary, tmp_path)

    assert (tmp_path / "transfer-sparse-one_step.svg").stat().st_size > 1000
    assert (tmp_path / "transfer-sparse-one_step.png").stat().st_size > 1000
