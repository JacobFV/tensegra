import json

import pytest

from topoformer.study_analysis import (
    analyze_runs,
    complete_normalized_auc,
    efficiency_thresholds,
    paired_effects,
    select_soft_strength,
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

    values = analyze_runs([run])["learned_coefficients"][0]["values"]

    assert values == [
        {"layer": 0, "head": 0, "channel": 0, "value": -0.2},
        {"layer": 0, "head": 0, "channel": 1, "value": 0.3},
        {"layer": 0, "head": 1, "channel": 0, "value": 0.4},
        {"layer": 0, "head": 1, "channel": 1, "value": -0.5},
    ]
