import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time

import torch

from .data import make_dynamics, trajectories, windows
from .evaluation import diagnostics, one_step, rollout
from .graphs import corrupt_graph
from .model import GraphPredictor, TokenMLP
from .training import train_model


KEYS = {"kinds", "nodes", "history", "width", "heads", "layers", "batch_size",
        "train_count", "validation_count", "test_count", "steps", "optimizer_steps",
        "seeds", "learning_rate", "noise", "validation_interval", "rollout_horizon",
        "device", "max_wall_seconds", "modes"}
POSITIVE = {"nodes", "history", "width", "heads", "layers", "batch_size", "train_count",
            "validation_count", "test_count", "steps", "optimizer_steps", "validation_interval",
            "rollout_horizon", "max_wall_seconds"}
INTEGER_SIZES = POSITIVE - {"max_wall_seconds"}


def validate_config(config):
    unknown = set(config) - KEYS
    missing = KEYS - set(config)
    if unknown or missing:
        raise ValueError(f"configuration keys invalid; unknown={sorted(unknown)}, missing={sorted(missing)}")
    for key in POSITIVE:
        if (isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                or not math.isfinite(config[key]) or config[key] <= 0):
            raise ValueError(f"{key} must be positive")
        if key in INTEGER_SIZES and not isinstance(config[key], int):
            raise TypeError(f"{key} must be an integer")
    if config["steps"] <= config["history"]:
        raise ValueError("steps must exceed history")
    if config["width"] % config["heads"]:
        raise ValueError("width must be divisible by heads")
    if (not math.isfinite(config["noise"]) or config["noise"] < 0
            or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0):
        raise ValueError("noise must be nonnegative and learning_rate positive")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not config["kinds"] or any(kind not in {"sparse", "robot"} for kind in config["kinds"]):
        raise ValueError("unsupported kind")
    if not config["seeds"] or not config["modes"]:
        raise ValueError("seeds and modes must be nonempty")
    names = set()
    for item in config["modes"]:
        if set(item) - {"name", "mode", "strength", "graph"} or set(item) < {"name", "mode", "strength"}:
            raise ValueError("invalid mode entry")
        if (item["mode"] not in {"none", "soft", "hard"}
                or not isinstance(item["strength"], (int, float))
                or not math.isfinite(item["strength"])
                or item["strength"] < 0 or item["name"] in names):
            raise ValueError("unsupported or duplicate mode")
        if item.get("graph", "true") not in {"true", "permuted"}:
            raise ValueError("unsupported graph source")
        names.add(item["name"])
    return config


def _hash_state(state):
    digest = hashlib.sha256()
    for value in state.values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _normalize(series, mean, std):
    return (series - mean) / std


def _graph_for(item, graph, seed):
    return corrupt_graph(graph, mode="permuted", seed=seed) if item.get("graph") == "permuted" else graph


def run(config: dict, output_dir: str) -> dict:
    config = validate_config(dict(config))
    torch.set_num_threads(2)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(config["device"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    started = time.monotonic()
    rows = []
    complete = True
    deadline = started + config["max_wall_seconds"]
    with metrics_path.open("w") as metrics_file:
      for kind in config["kinds"]:
       for seed in config["seeds"]:
        if time.monotonic() - started >= config["max_wall_seconds"]:
            complete = False
            break
        system = make_dynamics(kind, config["nodes"], seed)
        split_seeds = {"train": seed * 100 + 11, "validation": seed * 100 + 12,
                       "test": seed * 100 + 13, "shift": seed * 100 + 14}
        sets = {name: trajectories(system, count=config[f"{name}_count"], steps=config["steps"],
                                   seed=split_seed, noise=config["noise"])
                for name, split_seed in split_seeds.items() if name != "shift"}
        shifted = trajectories(system, count=config["test_count"], steps=config["steps"],
                               seed=split_seeds["shift"], noise=config["noise"], initial_scale=2, burn_in=0)
        mean = sets["train"].mean()
        std = sets["train"].std().clamp_min(1e-8)
        prepared = {name: windows(_normalize(value, mean, std), config["history"]) for name, value in sets.items()}
        original_test = windows(sets["test"], config["history"])
        generator = torch.Generator().manual_seed(seed + 1000)
        schedule = torch.randint(len(prepared["train"][0]),
                                 (config["optimizer_steps"], config["batch_size"]), generator=generator)
        torch.manual_seed(seed + 2000)
        prototype = GraphPredictor(config["history"], config["width"], config["heads"], config["layers"])
        initial = {key: value.clone() for key, value in prototype.state_dict().items()}
        initial_hash = _hash_state(initial)

        for item in config["modes"]:
            if time.monotonic() - started >= config["max_wall_seconds"]:
                complete = False
                break
            graph = _graph_for(item, system.read_graph, seed + 3000).to(device)
            model = GraphPredictor(config["history"], config["width"], config["heads"], config["layers"]).to(device)
            model.load_state_dict(initial)
            tx, ty = (value.to(device) for value in prepared["train"])
            vx, vy = (value.to(device) for value in prepared["validation"])
            mode_started = time.monotonic()
            def progress(point):
                print(f"domain={kind} seed={seed} mode={item['name']} step={point['step']}/{config['optimizer_steps']} train={point['train_loss']:.6g} validation={point['validation_loss']:.6g} elapsed={point['elapsed_seconds']:.2f}s", flush=True)
            curve = train_model(model, tx, ty, (vx, vy), graph=graph, mode=item["mode"], strength=item["strength"],
                                steps=config["optimizer_steps"], batch_size=config["batch_size"],
                                learning_rate=config["learning_rate"], validation_interval=config["validation_interval"],
                                batch_indices=schedule.to(device), progress=progress, deadline=deadline)
            test_x, test_y = (v.to(device) for v in prepared["test"])
            raw = one_step(model, test_x, test_y, graph=graph, mode=item["mode"], strength=item["strength"]) * std.item() ** 2
            row = {"kind": kind, "seed": seed, "model": "graph", "mode_name": item["name"],
                   "mode": item["mode"], "strength": item["strength"], "graph_source": item.get("graph", "true"), "split_seeds": split_seeds,
                   "initial_state_hash": initial_hash, "batch_indices": schedule.flatten().tolist(),
                   "validation_normalized_mse": curve[-1]["validation_loss"],
                   "validation_raw_mse": curve[-1]["validation_loss"] * std.item() ** 2,
                   "test_raw_mse": raw, "test_normalized_mse": raw / std.item() ** 2,
                   "rollout_raw_mse": rollout(model, sets["test"].to(device), config["history"], config["rollout_horizon"], mean.to(device), std.to(device), graph=graph, mode=item["mode"], strength=item["strength"]),
                   "transient_initial_state_shift_raw_mse": rollout(model, shifted.to(device), config["history"], config["rollout_horizon"], mean.to(device), std.to(device), graph=graph, mode=item["mode"], strength=item["strength"]),
                   "parameters": sum(p.numel() for p in model.parameters()), "wall_seconds": time.monotonic() - mode_started,
                   "optimizer_steps_completed": curve[-1]["step"],
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                   "curve": curve}
            row["rollout_normalized_mse"] = row["rollout_raw_mse"] / std.item() ** 2
            row["transient_initial_state_shift_normalized_mse"] = row["transient_initial_state_shift_raw_mse"] / std.item() ** 2
            rows.append(row)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            if curve[-1]["step"] < config["optimizer_steps"]:
                complete = False
                break

        # Calibration baselines use the same normalized windows.
        if not complete:
            break
        ox, oy = original_test
        refs = diagnostics(system, ox, oy)
        for name, value in refs.items():
            row = {"kind": kind, "seed": seed, "model": name, "mode_name": name,
                   "split_seeds": split_seeds, "test_raw_mse": value,
                   "test_normalized_mse": value / std.item() ** 2}
            rows.append(row)
            metrics_file.write(json.dumps(row) + "\n")
        class Persistence(torch.nn.Module):
            def forward(self, x):
                return x[..., -1]
        persistence_model = Persistence().to(device)
        persistence = torch.mean((ox[..., -1] - oy) ** 2).item()
        row = {"kind": kind, "seed": seed, "model": "persistence", "mode_name": "persistence",
               "split_seeds": split_seeds, "test_raw_mse": persistence,
               "test_normalized_mse": persistence / std.item() ** 2, "parameters": 0,
               "rollout_raw_mse": rollout(persistence_model, sets["test"].to(device), config["history"], config["rollout_horizon"], mean.to(device), std.to(device)),
               "transient_initial_state_shift_raw_mse": rollout(persistence_model, shifted.to(device), config["history"], config["rollout_horizon"], mean.to(device), std.to(device))}
        row["rollout_normalized_mse"] = row["rollout_raw_mse"] / std.item() ** 2
        row["transient_initial_state_shift_normalized_mse"] = row["transient_initial_state_shift_raw_mse"] / std.item() ** 2
        rows.append(row)
        metrics_file.write(json.dumps(row) + "\n")
        torch.manual_seed(seed + 4000)
        mlp = TokenMLP(config["history"], config["width"]).to(device)
        curve = train_model(mlp, tx, ty, (vx, vy), steps=config["optimizer_steps"], batch_size=config["batch_size"],
                            learning_rate=config["learning_rate"], validation_interval=config["validation_interval"],
                            batch_indices=schedule.to(device), deadline=deadline)
        mlp_raw = one_step(mlp, test_x, test_y) * std.item() ** 2
        row = {"kind": kind, "seed": seed, "model": "token_mlp", "mode_name": "token_mlp",
               "split_seeds": split_seeds, "test_raw_mse": mlp_raw,
               "test_normalized_mse": mlp_raw / std.item() ** 2,
               "validation_normalized_mse": curve[-1]["validation_loss"],
               "validation_raw_mse": curve[-1]["validation_loss"] * std.item() ** 2,
               "rollout_raw_mse": rollout(mlp, sets["test"].to(device), config["history"], config["rollout_horizon"], mean.to(device), std.to(device)),
               "transient_initial_state_shift_raw_mse": rollout(mlp, shifted.to(device), config["history"], config["rollout_horizon"], mean.to(device), std.to(device)),
               "parameters": sum(p.numel() for p in mlp.parameters()),
               "optimizer_steps_completed": curve[-1]["step"]}
        row["rollout_normalized_mse"] = row["rollout_raw_mse"] / std.item() ** 2
        row["transient_initial_state_shift_normalized_mse"] = row["transient_initial_state_shift_raw_mse"] / std.item() ** 2
        rows.append(row)
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        if curve[-1]["step"] < config["optimizer_steps"]:
            complete = False
       if not complete:
           break

    graph_rows = [r for r in rows if r["model"] == "graph"]
    soft_names = sorted({r["mode_name"] for r in graph_rows if r["mode"] == "soft" and r["graph_source"] == "true"})
    selected = min(soft_names, key=lambda name: sum(r["validation_normalized_mse"] for r in graph_rows if r["mode_name"] == name) / len([r for r in graph_rows if r["mode_name"] == name])) if soft_names else None
    paired = {}
    hard_names = sorted({r["mode_name"] for r in graph_rows if r["mode"] == "hard"})
    for name in ([selected] if selected else []) + hard_names:
        differences = []
        for r in graph_rows:
            if r["mode_name"] == name:
                base = next((b for b in graph_rows if b["kind"] == r["kind"] and b["seed"] == r["seed"] and b["mode"] == "none"), None)
                if base:
                    differences.append({"kind": r["kind"], "seed": r["seed"],
                                        "test_raw_mse_difference": r["test_raw_mse"] - base["test_raw_mse"]})
        if differences:
            values = [item["test_raw_mse_difference"] for item in differences]
            paired[name] = {"values": differences, "mean": sum(values) / len(values),
                            "std": torch.tensor(values).std(unbiased=False).item()}
    summary = {"complete": complete, "selected_soft_mode": selected, "paired_differences": paired,
               "runs": rows, "config": config, "git_revision": _git_revision(),
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)},
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"artifact={output} complete={complete}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)


if __name__ == "__main__":
    main()
