"""Development-only memory-interface acquisition on paired public snapshots.

Two narrowly constructed teacher decisions vary an omitted conflict or downstream
route cost. This is supervised local action selection, not autonomous planning.
Training/evaluation base-parameter parities are disjoint; opaque names and inventory order
are regenerated. No solver is called and no test result selects a checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import torch
from torch import nn

from topoformer.campaign02_memory import ROW_DIM
from topoformer.campaign02_memory_policy import MemoryCandidatePolicy, MemoryPolicyConfig
from topoformer.campaign02_references import make_reference
from topoformer.campaign02_training import prepare_frame, policy_batch, score_batch, digest
from topoformer.campaign02_world import Action, Item, WorldSpec, Workshop, action_catalog, encode_action, encode_observation


def snapshots(seed, *, evaluation=False):
    rng = random.Random(seed)
    names = [f"item_{value}" for value in rng.sample(range(10**6, 10**9), 4)]
    # Held-out parameter values; tests do not merely rename a stored training row.
    base = rng.randint(2, 20)*2+int(evaluation)
    price = rng.randint(2, 20)*2+int(evaluation)
    items = [Item(names[0], 0, base, price), Item(names[1], 0, base+1, price+1),
             Item(names[2], 1, base, price), Item(names[3], 1, base*3+10, price*3+10)]
    rng.shuffle(items)
    conflict = WorldSpec(tuple(items), (0, 1), base*2+2, price*2+2, (), ((0, 1, 1), (1, 2, 1)), 0, 2)
    distance = rng.randint(2, 20)*2+int(evaluation)
    route = WorldSpec((Item(names[0], 0, base, price),), (0,), base+1, price+1, (),
        ((0, 1, distance), (1, 2, 1), (0, 2, distance+3)), 0, 2, travel_price=.002, travel_limit=256)
    result = []
    for family, specs in (("conflict", (conflict, replace(conflict, incompatible=((names[0], names[2]),)))),
                          ("route", (route, replace(route, edges=((0, 1, distance), (1, 2, 5), (0, 2, distance+3)))))):
        for side, spec in enumerate(specs):
            env = Workshop(spec, address_seed=seed+917)
            for item in spec.items:
                env.step(Action("inspect", {"target": item.handle}))
            if family == "route":
                env.step(Action("choose_item", {"item": names[0]}))
                env.step(Action("commit_pending"))
                env.step(Action("inspect", {"target": "map"}))
            result.append((family, side, env.observe()))
    return result


def frames_for(seed, model, *, evaluation=False):
    teacher = make_reference("cheap_first_fallback_v2")
    frames, rows = [], []
    for family, side, observation in snapshots(seed, evaluation=evaluation):
        actions, frame = prepare_frame(observation, model)
        frame.target = actions.index(teacher.choose(observation, actions))
        frames.append(frame)
        rows.append({"seed": seed, "family": family, "side": side, "target": frame.target,
                     "action": asdict(actions[frame.target]), "observation": observation.to_dict(),
                     "memory_rows": len(frame.rows), "public_hash": digest(observation.to_dict())})
    assert rows[0]["action"]["kind"] != rows[1]["action"]["kind"]
    assert rows[2]["action"]["kind"] != rows[3]["action"]["kind"]
    return frames, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite outcomes")
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(cfg.get("threads", 1))
    torch.manual_seed(cfg["seed"])
    sample = snapshots(cfg["seed"])[0][2]
    model_cfg = MemoryPolicyConfig(len(encode_observation(sample)), len(encode_action(sample, action_catalog(sample)[0])),
        width=cfg.get("width", 1024), family=cfg.get("family", "lightweight"), memory_dim=ROW_DIM,
        **cfg.get("memory", {}))
    device = cfg.get("device", "cuda")
    model = MemoryCandidatePolicy(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 3e-4))
    args.output.mkdir(parents=True, exist_ok=True)
    curves, seen = [], set()
    started, cpu = time.perf_counter(), time.process_time()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    for update in range(cfg["updates"]):
        frames = []
        for j in range(cfg.get("batch_pairs", 2)):
            seed = cfg["training_seed_start"]+update*cfg.get("batch_pairs", 2)+j
            new_frames, rows = frames_for(seed, model)
            frames.extend(new_frames)
            seen.update(row["public_hash"] for row in rows)
        optimizer.zero_grad(set_to_none=True)
        loss, count = model.backward_supervised([[frame] for frame in frames], device, 1)
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        curves.append({"update": update+1, "loss": float(loss), "decisions": count,
                       "unique_public_snapshots_seen": len(seen)})
    model.eval()
    if cfg["training_seed_start"] < cfg["evaluation_seed_start"]+cfg.get("evaluation_pairs", 128) and cfg["evaluation_seed_start"] < cfg["training_seed_start"]+cfg["updates"]*cfg.get("batch_pairs", 2):
        raise ValueError("Training/evaluation seed intervals overlap")
    rows_out, per_family = [], {}
    with torch.no_grad():
        for j in range(cfg.get("evaluation_pairs", 128)):
            frames, rows = frames_for(cfg["evaluation_seed_start"]+j, model, evaluation=True)
            batch = policy_batch(model, frames, device)
            logits, _, _ = score_batch(model, batch)
            predictions = logits.argmax(-1).cpu().tolist()
            probabilities = logits.softmax(-1).cpu().tolist()
            for row, prediction, probs in zip(rows, predictions, probabilities):
                correct = prediction == row["target"]
                rows_out.append({**row, "prediction": prediction, "probabilities": probs, "correct": correct})
                cell = f"{row['family']}/{row['side']}"
                counts = per_family.setdefault(cell, {"correct": 0, "examples": 0})
                counts["correct"] += int(correct)
                counts["examples"] += 1
    with gzip.open(args.output/"predictions.jsonl.gz", "wt") as stream:
        for row in rows_out:
            stream.write(json.dumps(row, sort_keys=True)+"\n")
    torch.save({"model": model.state_dict(), "policy_config": asdict(model_cfg), "config": cfg}, args.output/"checkpoint.pt")
    summary = {"config": cfg, "policy_config": asdict(model_cfg), "parameter_count": sum(p.numel() for p in model.parameters()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "last update, no confirmation", "interpretation": "local supervised public snapshot distinction; no autonomous composition",
        "accuracy": sum(r["correct"] for r in rows_out)/len(rows_out), "cells": per_family,
        "examples": len(rows_out), "independent_counterfactual_pairs": len(rows_out)//2,
        "training_base_parameter_parity": "even", "evaluation_base_parameter_parity": "odd",
        "numerical_scope": "base parameters held out; derived integers can overlap, no general numerical-range claim",
        "curves": curves, "encoding": model.encoding_profile, "collation": model.collate_profile,
        "wall_seconds": time.perf_counter()-started, "parent_cpu_seconds": time.process_time()-cpu,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(device)) if torch.device(device).type == "cuda" else None}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
