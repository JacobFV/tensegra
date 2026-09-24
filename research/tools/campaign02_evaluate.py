"""Frozen public-controller evaluation on paired fresh workshop worlds.

No training, teacher forcing, checkpoint selection, or hidden-state actor input.
The configuration lists checkpoint hashes and explicit world-seed conditions.
Reference schedules are labeled supplied controls, never learned policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import torch

from topoformer.campaign02_policy import CandidatePolicy, PolicyConfig
from topoformer.campaign02_protocol import BoundedSolver
from topoformer.campaign02_references import ReferencePolicy, run_episode
from topoformer.campaign02_training import TrainConfig, batched_episodes, independent_address_seed
from topoformer.campaign02_world import Workshop, generate_world, protocol_executor


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_rows(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+"\n")


def episode_counts(outcome):
    """Reconstruct observable return-address contracts, not gold optimal choices."""
    counts = Counter()
    problem_types, returns, retrieved = {}, {}, {}
    for event in outcome["history"]:
        action, feedback = event["action"], event["feedback"]
        kind, args = action["kind"], action["arguments"]
        counts[f"action/{kind}"] += 1
        counts[f"status/{feedback.get('status', 'missing')}"] += 1
        if kind in {"start_subset", "build_route"} and feedback.get("status") == "success":
            problem_types[args["handle"]] = "constrained_subset" if kind == "start_subset" else "shortest_path"
        if kind == "call" and "return" in feedback:
            returns[feedback["return"]] = {"primitive": problem_types.get(args["problem"]),
                                          "version": event["state_version"]}
        if kind == "retrieve" and "record" in feedback:
            record = feedback["record"]
            retrieved[args["handle"]] = record
        if kind == "use_return":
            counts["return_use_attempts"] += 1
            multiple = len(returns) >= 2
            counts["multiple_return_use_attempts"] += int(multiple)
            record = retrieved.get(args["handle"], {})
            expected = {"subset": "constrained_subset", "route": "shortest_path"}.get(args.get("as"))
            # Route execution itself may discover an obstacle and advance version.
            stale = feedback.get("reason") == "stale_result"
            valid = (bool(record) and record.get("primitive") == expected and not stale
                     and record.get("certificate_valid", False)
                     and record.get("status") in {"success", "timeout"})
            counts["return_address_contract_valid"] += int(valid)
            counts["multiple_return_address_contract_valid"] += int(valid and multiple)
            counts["return_application_success"] += int(feedback.get("status") == "success")
            counts["stale_return_use"] += int(stale)
    reductions = outcome.get("reductions", [])
    counts["reductions"] = len(reductions)
    counts["correct_reductions"] = sum(bool(row["correct"]) for row in reductions)
    counts["certificate_valid_reductions"] = sum(bool(row["certificate_valid"]) for row in reductions)
    return dict(counts)


def summarize(rows):
    count = len(rows)
    successes = sum(bool(row["outcome"]["verified_success"]) for row in rows)
    totals = Counter()
    for row in rows:
        totals.update(row["counts"])
    cost = sum(row["outcome"]["cost"] for row in rows)
    fields = ("utility", "cost", "steps", "observations", "work_units", "travel_distance",
              "compute_units", "modeled_compute_cost", "solver_cpu_seconds")
    return {"examples": count, "success_count": successes, "success": successes/count,
            "means": {field: sum(row["outcome"].get(field, 0) for row in rows)/count for field in fields},
            "cost_per_success_including_failures": cost/successes if successes else None,
            "counts": dict(totals), "truncated": sum(bool(row.get("truncated", False)) for row in rows),
            "neural_forward_wall_seconds": sum(sum(step.get("neural_forward_wall_seconds_allocated", 0)
                                                       for step in row.get("trace", [])) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--limit", type=int, help="Explicit profile-only episode cap; not full protocol")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Profile cap must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(args.threads)
    sources = {"evaluator": file_hash(__file__)}
    import topoformer.campaign02_training as training
    for name in ("training", "policy", "protocol", "world", "references"):
        sources[name] = file_hash(Path(training.__file__).with_name(f"campaign02_{name}.py"))
    bindings = []
    for binding in cfg["checkpoints"]:
        actual = file_hash(binding["path"])
        if actual != binding["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {binding['name']}")
        bindings.append({**binding, "sha256": actual})
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    summary = {"config": cfg, "config_sha256": file_hash(args.config), "sources": sources,
               "checkpoints": bindings, "profile_limit": args.limit, "results": [], "paired": [],
               "policy_input_scope": "public observation/candidate encodings only; no teacher at evaluation",
               "address_metric_scope": "role/status/retrieval/provenance validity among extant returned records, not optimal-return oracle",
               "selection_rule": "frozen supplied checkpoint, no selection", "no_training": True}
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    with BoundedSolver() as solver:
        executor = partial(protocol_executor, execute_call=solver.execute)
        for condition in cfg["conditions"]:
            name = condition["name"]
            n = min(condition.get("examples", 512), args.limit or condition.get("examples", 512))
            if n < 1 or Path(name).name != name:
                raise ValueError("Invalid condition name or support")
            seeds = list(range(condition["seed_start"], condition["seed_start"]+n))
            specs = [generate_world(seed, **condition.get("world", {})) for seed in seeds]
            namespace = condition.get("address_namespace", cfg.get("address_namespace", "extended-02-frozen-eval-v1"))
            def factory(index):
                return Workshop(specs[index], executor=executor,
                    address_seed=independent_address_seed(seeds[index], namespace))
            folder = args.output/name
            folder.mkdir()
            world_rows = [{"seed": seed, "spec": asdict(spec), "spec_hash": canonical_hash(asdict(spec)),
                           "address_seed": independent_address_seed(seed, namespace)} for seed, spec in zip(seeds, specs)]
            write_rows(folder/"worlds.jsonl.gz", world_rows)
            by_arm = {}
            for binding in bindings:
                checkpoint = torch.load(binding["path"], map_location="cpu", weights_only=False)
                policy_cfg = PolicyConfig(**checkpoint["policy_config"])
                train_cfg = TrainConfig(**checkpoint["config"])
                policy = CandidatePolicy(policy_cfg).to(args.device)
                policy.load_state_dict(checkpoint["model"], strict=True)
                policy.eval()
                rows = []
                batch_size = cfg.get("evaluation_batch", 32)
                if batch_size < 1:
                    raise ValueError("Positive evaluation batch required")
                with torch.no_grad():
                    for offset in range(0, n, batch_size):
                        indices = list(range(offset, min(n, offset+batch_size)))
                        outputs = batched_episodes(policy, [factory(i) for i in indices], device=args.device,
                            max_steps=train_cfg.max_steps, neural_work_per_forward=train_cfg.neural_work_per_forward)
                        for i, result in zip(indices, outputs):
                            rows.append({"seed": seeds[i], "spec_hash": world_rows[i]["spec_hash"], **result,
                                         "counts": episode_counts(result["outcome"])})
                artifact = folder/f"{binding['name']}.jsonl.gz"
                write_rows(artifact, rows)
                summary["results"].append({"condition": name, "arm": binding["name"], "kind": "learned",
                    **summarize(rows), "artifact": str(artifact.relative_to(args.output)), "artifact_sha256": file_hash(artifact),
                    "policy_config": asdict(policy_cfg), "training_config": asdict(train_cfg),
                    "actual_parameter_count": sum(p.numel() for p in policy.parameters()),
                    "checkpoint_updates": checkpoint["updates"], "checkpoint_presentations": checkpoint["presentations"]})
                by_arm[binding["name"]] = rows
                del policy, checkpoint
            for mode in cfg.get("references", ["cheap", "always_tool", "cheap_first"]):
                rows = []
                for i, seed in enumerate(seeds):
                    outcome = run_episode(factory(i), ReferencePolicy(mode), cfg.get("reference_compute_tariff", 0.0))
                    outcome.pop("trace", None)  # Same information retained once in exact history.
                    rows.append({"seed": seed, "spec_hash": world_rows[i]["spec_hash"], "outcome": outcome,
                                 "counts": episode_counts(outcome), "truncated": False})
                artifact = folder/f"reference-{mode}.jsonl.gz"
                write_rows(artifact, rows)
                summary["results"].append({"condition": name, "arm": f"reference-{mode}", "kind": "supplied_schedule",
                    **summarize(rows), "artifact": str(artifact.relative_to(args.output)), "artifact_sha256": file_hash(artifact),
                    "reference_compute_tariff": cfg.get("reference_compute_tariff", 0.0),
                    "controller_cpu_seconds": sum(r["outcome"]["controller_cpu_seconds"] for r in rows)})
                by_arm[f"reference-{mode}"] = rows
            for binding in bindings:
                a = by_arm[binding["name"]]
                for other_name, b in by_arm.items():
                    if other_name == binding["name"]:
                        continue
                    transitions = Counter(f"{int(x['outcome']['verified_success'])}->{int(y['outcome']['verified_success'])}" for x,y in zip(b,a))
                    summary["paired"].append({"condition": name, "learned": binding["name"], "comparator": other_name,
                        "success_transitions_comparator_to_learned": dict(transitions), "support": n,
                        "mean_utility_difference": sum(x["outcome"]["utility"]-y["outcome"]["utility"] for x,y in zip(a,b))/n})
            (args.output/"summary.partial.json").write_text(json.dumps(summary, indent=2))
        summary["solver"] = {"executor_version": solver.executor_version,
            "startup_wall_seconds": solver.startup_wall_seconds, "startup_child_cpu_seconds": solver.startup_child_cpu_seconds,
            "call_wall_seconds": solver.call_wall_seconds, "restarts": solver.restarts}
    summary["resources"] = {"wall_seconds": time.perf_counter()-start_wall, "parent_cpu_seconds": time.process_time()-start_cpu,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
        "cuda_device_capacity_bytes": torch.cuda.get_device_properties(torch.device(args.device)).total_memory if torch.device(args.device).type == "cuda" else None,
        "accounting_scope": "parent CPU excludes solver process; authoritative inclusive workload ledger is external"}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    (args.output/"summary.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
