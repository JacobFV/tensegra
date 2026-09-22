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
    unknown, missing = set(config) - KEYS, KEYS - set(config)
    if unknown or missing:
        raise ValueError(f"configuration keys invalid; unknown={sorted(unknown)}, missing={sorted(missing)}")
    for key in POSITIVE:
        value = config[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be positive")
        if key in INTEGER_SIZES and not isinstance(value, int):
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
        required = {"name", "mode", "strength"}
        if set(item) - (required | {"graph"}) or not required <= set(item):
            raise ValueError("invalid mode entry")
        strength = item["strength"]
        if (item["mode"] not in {"none", "soft", "hard"}
                or not isinstance(strength, (int, float)) or not math.isfinite(strength)
                or strength < 0 or item["name"] in names):
            raise ValueError("unsupported or duplicate mode")
        if item.get("graph", "true") not in {"true", "permuted"}:
            raise ValueError("unsupported graph source")
        names.add(item["name"])
    return config


def _tensor_hash(tensors):
    digest = hashlib.sha256()
    values = tensors.values() if isinstance(tensors, dict) else (tensors,)
    for value in values:
        raw = value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
        digest.update(bytes(raw))
    return digest.hexdigest()


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _prepare_case(config, kind, seed, device):
    system = make_dynamics(kind, config["nodes"], seed)
    split_seeds = {"train": seed * 100 + 11, "validation": seed * 100 + 12,
                   "test": seed * 100 + 13, "shift": seed * 100 + 14}
    series = {
        name: trajectories(system, count=config[f"{name}_count"], steps=config["steps"],
                           seed=split_seeds[name], noise=config["noise"])
        for name in ("train", "validation", "test")
    }
    shifted = trajectories(
        system, count=config["test_count"], steps=config["steps"],
        seed=split_seeds["shift"], noise=config["noise"], initial_scale=2, burn_in=0,
    )
    mean, std = series["train"].mean(), series["train"].std().clamp_min(1e-8)
    prepared = {
        name: tuple(value.to(device) for value in windows((values - mean) / std, config["history"]))
        for name, values in series.items()
    }
    schedule_seed = seed + 1000
    generator = torch.Generator().manual_seed(schedule_seed)
    schedule = torch.randint(len(prepared["train"][0]),
                             (config["optimizer_steps"], config["batch_size"]),
                             generator=generator)
    return {"system": system, "split_seeds": split_seeds, "series": series,
            "shifted": shifted, "mean": mean, "std": std, "prepared": prepared,
            "schedule": schedule, "schedule_seed": schedule_seed,
            "schedule_hash": _tensor_hash(schedule)}


def _resource_fields(model, curve, started, device):
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_steps_completed": curve[-1]["step"] if curve else 0,
        "curve": curve,
        "wall_seconds": time.monotonic() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def _metric_fields(model, case, config, device, graph=None, mode="none", strength=0.0):
    test_x, test_y = case["prepared"]["test"]
    normalized = one_step(model, test_x, test_y, graph=graph, mode=mode, strength=strength)
    variance = case["std"].item() ** 2
    rollout_raw = rollout(model, case["series"]["test"].to(device), config["history"],
                          config["rollout_horizon"], case["mean"].to(device),
                          case["std"].to(device), graph=graph, mode=mode, strength=strength)
    shift_raw = rollout(model, case["shifted"].to(device), config["history"],
                        config["rollout_horizon"], case["mean"].to(device),
                        case["std"].to(device), graph=graph, mode=mode, strength=strength)
    return {"test_raw_mse": normalized * variance, "test_normalized_mse": normalized,
            "rollout_raw_mse": rollout_raw, "rollout_normalized_mse": rollout_raw / variance,
            "transient_initial_state_shift_raw_mse": shift_raw,
            "transient_initial_state_shift_normalized_mse": shift_raw / variance}


def _train_graph_mode(config, case, kind, seed, item, initial, initial_hash, device, deadline):
    graph = case["system"].read_graph
    if item.get("graph") == "permuted":
        graph = corrupt_graph(graph, mode="permuted", seed=seed + 3000)
    graph = graph.to(device)
    model = GraphPredictor(config["history"], config["width"], config["heads"],
                           config["layers"]).to(device)
    model.load_state_dict(initial)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    def progress(point):
        print(f"domain={kind} seed={seed} mode={item['name']} "
              f"step={point['step']}/{config['optimizer_steps']} "
              f"train={point['train_loss']:.6g} validation={point['validation_loss']:.6g} "
              f"elapsed={point['elapsed_seconds']:.2f}s", flush=True)

    curve = train_model(
        model, *case["prepared"]["train"], case["prepared"]["validation"],
        graph=graph, mode=item["mode"], strength=item["strength"],
        steps=config["optimizer_steps"], batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], validation_interval=config["validation_interval"],
        batch_indices=case["schedule"].to(device), progress=progress, deadline=deadline,
    )
    variance = case["std"].item() ** 2
    row = {"kind": kind, "seed": seed, "model": "graph", "mode_name": item["name"],
           "mode": item["mode"], "strength": item["strength"],
           "graph_source": item.get("graph", "true"), "split_seeds": case["split_seeds"],
           "initial_state_hash": initial_hash, "batch_schedule_seed": case["schedule_seed"],
           "batch_schedule_hash": case["schedule_hash"],
           "validation_normalized_mse": curve[-1]["validation_loss"],
           "validation_raw_mse": curve[-1]["validation_loss"] * variance}
    row.update(_metric_fields(model, case, config, device, graph, item["mode"], item["strength"]))
    row.update(_resource_fields(model, curve, started, device))
    row["run_complete"] = row["optimizer_steps_completed"] == config["optimizer_steps"]
    return row


def _reference_rows(config, case, kind, seed, device, deadline):
    variance = case["std"].item() ** 2
    original_x, original_y = windows(case["series"]["test"], config["history"])
    common = {"kind": kind, "seed": seed, "split_seeds": case["split_seeds"]}
    rows, started = [], time.monotonic()
    for name, value in diagnostics(case["system"], original_x, original_y).items():
        rows.append({**common, "model": name, "mode_name": name,
                     "test_raw_mse": value, "test_normalized_mse": value / variance,
                     "validation_raw_mse": None, "validation_normalized_mse": None,
                     "rollout_raw_mse": None, "rollout_normalized_mse": None,
                     "transient_initial_state_shift_raw_mse": None,
                     "transient_initial_state_shift_normalized_mse": None,
                     "parameters": 0, "optimizer_steps_completed": 0, "curve": [],
                     "wall_seconds": time.monotonic() - started,
                     "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "cuda_peak_bytes": None, "run_complete": True})

    class Persistence(torch.nn.Module):
        def forward(self, x):
            return x[..., -1]

    started = time.monotonic()
    persistence = Persistence().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    row = {**common, "model": "persistence", "mode_name": "persistence",
           "validation_raw_mse": None, "validation_normalized_mse": None}
    row.update(_metric_fields(persistence, case, config, device))
    row.update(_resource_fields(persistence, [], started, device))
    row["run_complete"] = True
    rows.append(row)

    if time.monotonic() >= deadline:
        return rows, False

    torch.manual_seed(seed + 4000)
    mlp = TokenMLP(config["history"], config["width"]).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    curve = train_model(
        mlp, *case["prepared"]["train"], case["prepared"]["validation"],
        steps=config["optimizer_steps"], batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], validation_interval=config["validation_interval"],
        batch_indices=case["schedule"].to(device), deadline=deadline,
    )
    row = {**common, "model": "token_mlp", "mode_name": "token_mlp",
           "validation_normalized_mse": curve[-1]["validation_loss"],
           "validation_raw_mse": curve[-1]["validation_loss"] * variance}
    row.update(_metric_fields(mlp, case, config, device))
    row.update(_resource_fields(mlp, curve, started, device))
    row["run_complete"] = row["optimizer_steps_completed"] == config["optimizer_steps"]
    rows.append(row)
    return rows, row["run_complete"]


def _statistics(values):
    tensor = torch.tensor([item["difference"] for item in values])
    return {"values": values, "mean": tensor.mean().item(),
            "std": tensor.std(unbiased=False).item()}


def _aggregate(rows, kinds):
    selected, paired = {}, {}
    graph_rows = [row for row in rows if row["model"] == "graph" and row["run_complete"]]
    for kind in kinds:
        domain = [row for row in graph_rows if row["kind"] == kind]
        soft_names = {row["mode_name"] for row in domain
                      if row["mode"] == "soft" and row["graph_source"] == "true"}
        if not soft_names:
            continue
        selected[kind] = min(soft_names, key=lambda name: sum(
            row["validation_normalized_mse"] for row in domain if row["mode_name"] == name
        ) / len([row for row in domain if row["mode_name"] == name]))
        comparisons = [selected[kind]] + sorted({row["mode_name"] for row in domain
                                                 if row["mode"] == "hard"})
        paired[kind] = {}
        for name in comparisons:
            raw, normalized = [], []
            for row in domain:
                if row["mode_name"] != name:
                    continue
                base = next(candidate for candidate in domain
                            if candidate["seed"] == row["seed"] and candidate["mode"] == "none")
                raw.append({"seed": row["seed"],
                            "difference": row["test_raw_mse"] - base["test_raw_mse"]})
                normalized.append({"seed": row["seed"],
                                   "difference": (row["test_normalized_mse"]
                                                  - base["test_normalized_mse"])})
            paired[kind][name] = {"raw": _statistics(raw),
                                  "normalized": _statistics(normalized)}
    return selected, paired


def run(config: dict, output_dir: str) -> dict:
    config = validate_config(dict(config))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(config["device"])
    deadline = time.monotonic() + config["max_wall_seconds"]
    rows, complete = [], True

    with (output / "metrics.jsonl").open("w") as metrics_file:
        for kind in config["kinds"]:
            for seed in config["seeds"]:
                if time.monotonic() >= deadline:
                    complete = False
                    break
                case = _prepare_case(config, kind, seed, device)
                torch.manual_seed(seed + 2000)
                prototype = GraphPredictor(config["history"], config["width"],
                                           config["heads"], config["layers"])
                initial = {key: value.clone() for key, value in prototype.state_dict().items()}
                initial_hash = _tensor_hash(initial)
                for item in config["modes"]:
                    if time.monotonic() >= deadline:
                        complete = False
                        break
                    row = _train_graph_mode(config, case, kind, seed, item, initial,
                                            initial_hash, device, deadline)
                    rows.append(row)
                    metrics_file.write(json.dumps(row) + "\n")
                    metrics_file.flush()
                    if not row["run_complete"]:
                        complete = False
                        break
                if not complete:
                    break
                if time.monotonic() >= deadline:
                    complete = False
                    break
                reference_rows, references_complete = _reference_rows(
                    config, case, kind, seed, device, deadline
                )
                for row in reference_rows:
                    rows.append(row)
                    metrics_file.write(json.dumps(row) + "\n")
                    metrics_file.flush()
                if not references_complete:
                    complete = False
                    break
                if not complete:
                    break
            if not complete:
                break

    selected, paired = _aggregate(rows, config["kinds"])
    summary = {"complete": complete, "selected_soft_mode": selected,
               "soft_selection_metric": "mean_validation_normalized_mse_per_kind",
               "paired_differences": paired, "runs": rows, "config": config,
               "git_revision": _git_revision(),
               "environment": {"python": platform.python_version(),
                               "torch": torch.__version__, "device": str(device)},
               "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "cuda_peak_bytes": (torch.cuda.max_memory_allocated(device)
                                   if device.type == "cuda" else None)}
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
