import json

import pytest
import torch

from topoformer.experiment import run, validate_config


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
    assert len({tuple(r["batch_indices"]) for r in graph_runs}) == 1
    assert len({tuple(sorted(r["split_seeds"].items())) for r in graph_runs}) == 1
    for row in runs:
        assert row["test_raw_mse"] >= 0 and row["test_normalized_mse"] >= 0
    assert (tmp_path / "a" / "metrics.jsonl").exists()
    summary = json.loads((tmp_path / "a" / "summary.json").read_text())
    assert "paired_differences" in summary and "privileged_oracle" in {r["model"] for r in runs}
