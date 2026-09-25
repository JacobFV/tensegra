import json

import pytest
import torch

import tensegra.experiment as experiment
from tensegra.experiment import run, validate_config
from tensegra.evaluation import rollout
from tensegra.training import train_model


def tiny_config():
    return {"kinds": ["sparse"], "nodes": 5, "history": 2, "width": 8,
            "heads": 2, "layers": 1, "batch_size": 4, "train_count": 4,
            "validation_count": 2, "test_count": 2, "steps": 7,
            "optimizer_steps": 2, "seeds": [0], "learning_rate": 0.001,
            "noise": 0.01, "validation_interval": 1, "rollout_horizon": 2,
            "device": "cpu", "max_wall_seconds": 60,
            "modes": [{"name": "none", "mode": "none", "strength": 0.0},
                      {"name": "soft1", "mode": "soft", "strength": 1.0},
                      {"name": "hard", "mode": "hard", "strength": 0.0}]}


@pytest.mark.parametrize("change", [
    {"bogus": 1}, {"nodes": 0}, {"noise": -1}, {"width": 7},
    {"modes": [{"name": "x", "mode": "bad", "strength": 0}]},
])
def test_invalid_config_is_rejected(change):
    config = tiny_config(); config.update(change)
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


def test_tiny_run_is_deterministic_paired_and_writes_artifacts(tmp_path):
    first = run(tiny_config(), tmp_path / "a")
    second = run(tiny_config(), tmp_path / "b")
    def scientific(rows):
        return [(r["model"], r["mode_name"], r["test_raw_mse"], r["test_normalized_mse"],
                 r.get("validation_normalized_mse")) for r in rows]
    assert scientific(first["runs"]) == scientific(second["runs"])
    runs = first["runs"]
    assert {r["mode_name"] for r in runs if r["model"] == "graph"} == {"none", "soft1", "hard"}
    graph_runs = [r for r in runs if r["model"] == "graph"]
    assert len({r["initial_state_hash"] for r in graph_runs}) == 1
    assert len({(r["batch_schedule_seed"], r["batch_schedule_hash"]) for r in graph_runs}) == 1
    assert len({tuple(sorted(r["split_seeds"].items())) for r in graph_runs}) == 1
    for row in runs:
        assert row["test_raw_mse"] >= 0 and row["test_normalized_mse"] >= 0
    assert (tmp_path / "a" / "metrics.jsonl").exists()
    summary = json.loads((tmp_path / "a" / "summary.json").read_text())
    assert "paired_differences" in summary and "privileged_oracle" in {r["model"] for r in runs}


def test_refuses_nonempty_output_directory(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        run(tiny_config(), output)
    assert sentinel.read_text() == "keep"


def test_rows_use_consistent_resource_and_training_schema(tmp_path):
    summary = run(tiny_config(), tmp_path / "schema")
    common = {"parameters", "optimizer_steps_completed", "curve", "wall_seconds",
              "process_peak_rss_kib", "cuda_peak_bytes"}
    for row in summary["runs"]:
        assert common <= row.keys()
        if row["model"] in {"zero", "privileged_oracle", "persistence"}:
            assert row["parameters"] == 0
            assert row["optimizer_steps_completed"] == 0
            assert row["curve"] == []


def test_selection_and_pairing_are_reported_per_kind(tmp_path):
    config = tiny_config()
    config.update(kinds=["sparse", "robot"], nodes=9)
    summary = run(config, tmp_path / "per-kind")
    assert set(summary["selected_soft_mode"]) == {"sparse", "robot"}
    assert set(summary["paired_differences"]) == {"sparse", "robot"}
    for comparisons in summary["paired_differences"].values():
        for comparison in comparisons.values():
            assert {"raw", "normalized"} <= comparison.keys()
            assert {"values", "mean", "std"} <= comparison["raw"].keys()
            assert {"values", "mean", "std"} <= comparison["normalized"].keys()


def test_soft_only_run_writes_summary_without_paired_claim(tmp_path):
    config = tiny_config()
    config["modes"] = [{"name": "soft1", "mode": "soft", "strength": 1.0}]
    summary = run(config, tmp_path / "soft-only")
    assert summary["complete"]
    assert summary["selected_soft_mode"] == {"sparse": "soft1"}
    assert summary["paired_differences"] == {}
    persisted = json.loads((tmp_path / "soft-only" / "summary.json").read_text())
    assert persisted["paired_differences"] == {}


def test_incomplete_or_later_unbiased_mode_is_not_paired():
    rows = [
        {"kind": "sparse", "seed": 0, "model": "graph", "mode_name": "soft1",
         "mode": "soft", "graph_source": "true", "run_complete": True,
         "validation_normalized_mse": 0.4, "test_raw_mse": 0.2,
         "test_normalized_mse": 0.5},
        {"kind": "sparse", "seed": 0, "model": "graph", "mode_name": "none",
         "mode": "none", "graph_source": "true", "run_complete": False,
         "validation_normalized_mse": 0.6, "test_raw_mse": 0.3,
         "test_normalized_mse": 0.7},
    ]
    selected, paired = experiment._aggregate(rows, ["sparse"])
    assert selected == {"sparse": "soft1"}
    assert paired == {}


def test_deadline_curve_records_last_completed_step():
    model = torch.nn.Linear(2, 1)
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    curve = train_model(model, x, y, (x, y), steps=5, batch_size=2,
                        validation_interval=5, batch_indices=torch.tensor([[0, 1]] * 5),
                        deadline=0)
    assert curve[-1]["step"] == 1


def test_tiny_deadline_writes_coherent_partial_row(tmp_path):
    config = tiny_config()
    config.update(optimizer_steps=10_000, validation_interval=10_000,
                  max_wall_seconds=0.05)
    summary = run(config, tmp_path / "partial")
    assert not summary["complete"]
    assert len(summary["runs"]) == 1
    row = summary["runs"][0]
    assert 0 < row["optimizer_steps_completed"] < config["optimizer_steps"]
    assert row["curve"][-1]["step"] == row["optimizer_steps_completed"]
    assert json.loads((tmp_path / "partial" / "summary.json").read_text())["complete"] is False


def test_expired_boundary_does_not_start_second_graph_mode(tmp_path, monkeypatch):
    config = tiny_config()
    config["max_wall_seconds"] = 1
    clock = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(experiment.time, "monotonic", lambda: next(clock))

    def completed_mode(config, case, kind, seed, item, initial, initial_hash, device, deadline):
        return {"kind": kind, "seed": seed, "model": "graph", "mode_name": item["name"],
                "mode": item["mode"], "graph_source": "true", "run_complete": True}

    monkeypatch.setattr(experiment, "_train_graph_mode", completed_mode)
    summary = run(config, tmp_path / "boundary")
    assert not summary["complete"]
    assert [row["mode_name"] for row in summary["runs"]] == ["none"]


def test_expired_reference_boundary_does_not_start_mlp():
    config = tiny_config()
    device = torch.device("cpu")
    case = experiment._prepare_case(config, "sparse", 0, device)
    rows, complete = experiment._reference_rows(
        config, case, "sparse", 0, device, deadline=0
    )
    assert not complete
    assert "token_mlp" not in {row["model"] for row in rows}


def test_rollout_calls_predictor_once_per_horizon_step():
    class CountingPersistence(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x[..., -1]

    model = CountingPersistence()
    series = torch.randn(2, 8, 3)
    rollout(model, series, history=2, horizon=4, mean=torch.tensor(0.0), std=torch.tensor(1.0))
    assert model.calls == 4
