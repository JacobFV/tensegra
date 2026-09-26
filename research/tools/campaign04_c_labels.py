"""Track C counterfactual labels for one frozen base, one chunk (CPU; extended-04 trackc.md §3).

    CUDA_VISIBLE_DEVICES= python research/tools/campaign04_c_labels.py --base p1-boot-x1-r0 \
        --chunk 0 --output results/c-labels-p1-boot-x1-r0

Chunk c covers label worlds LABEL_SEED_BASE + c*CHUNK ... + CHUNK - 1 (the same worlds for
every base); world i uses condition iid_f0 (even i) or iid_f2 (odd i), the actors'
training mix. Writes labels-c<chunk>.pt (campaign04_meta_train.pack) and
labels-c<chunk>.json (stats, CPU, hashes, privilege disclosure). Refuses to overwrite.

The environment copy (clone_branch) and the solver-result cache are the label
generator's **oracle-simulation privilege**; their CPU is charged to the offline
label ledger (this job's receipt) and never to episode utility.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign04_c_configs as C  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--chunk", type=int, nargs="+", required=True, help="one or more chunks, run sequentially")
    p.add_argument("--chunk-size", type=int, default=C.CHUNK)
    p.add_argument("--episodes", type=int, default=None, help="smoke only: fewer worlds in this chunk")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--branch-batch", type=int, default=128)
    p.add_argument("--no-verify-hash", action="store_true")
    p.add_argument("--dev-worlds", action="store_true", help="non-protocol smoke on Track C development seeds")
    a = p.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    C.check_fresh()
    binding = C.base(a.base)
    actor, train_cfg, info = C.load_actor(binding, verify=not a.no_verify_hash)
    stems = []
    for chunk in a.chunk:
        stem = a.output / f"labels-c{chunk:03d}"
        if stem.with_suffix(".pt").exists():
            raise SystemExit(f"refusing to overwrite {stem}.pt")
        stems.append(stem)
    for chunk in a.chunk:
        run_chunk(a, chunk, binding, actor, train_cfg, info)


def run_chunk(a, chunk, binding, actor, train_cfg, info):
    import torch
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign03_depworld import depworld_executor
    from tensegra.campaign04_branch import SolverCache
    from tensegra.campaign04_meta import CONTINUATION, LABEL_VERSION, LabelConfig, generate_labels
    from tensegra.campaign04_meta_train import pack
    seed_base = C.DEV_LABEL_BASE if a.dev_worlds else C.LABEL_SEED_BASE
    first = seed_base + chunk * a.chunk_size
    n = a.chunk_size if a.episodes is None else min(a.episodes, a.chunk_size)
    if first + n > seed_base + (C.DEV_SPAN if a.dev_worlds else C.LABEL_SPAN):
        raise SystemExit("chunk outside the registered label range")
    a.output.mkdir(parents=True, exist_ok=True)
    stem = a.output / f"labels-c{chunk:03d}"
    if stem.with_suffix(".pt").exists():
        raise SystemExit(f"refusing to overwrite {stem}.pt")
    wall0, cpu0 = time.perf_counter(), time.process_time()
    cfg = LabelConfig(max_steps=train_cfg.max_steps)
    seeds = list(range(first, first + n))
    indices = [s - seed_base for s in seeds]
    with BoundedSolver() as solver:
        cache = SolverCache(partial(depworld_executor, execute_call=solver.execute))
        envs = [C.world(s, C.label_condition(i), cache, C.NAMESPACE_LABELS) for s, i in zip(seeds, indices)]
        episodes, points, stats = generate_labels(actor, envs, seeds, cfg,
                                                  neural_work_per_forward=train_cfg.neural_work_per_forward,
                                                  branch_batch=a.branch_batch)
        cache_stats = cache.stats()
    meta = {"base": binding["name"], "base_sha256": binding["sha256"], "base_type": binding["base_type"],
            "chunk": chunk, "seeds": [seeds[0], seeds[-1] + 1], "protocol_worlds": not a.dev_worlds, "conditions": list(C.LABEL_CONDITIONS),
            "label_version": LABEL_VERSION, "continuation": CONTINUATION, "label_config": cfg.__dict__,
            "neural_work_per_forward": train_cfg.neural_work_per_forward, "max_steps": train_cfg.max_steps,
            "actor": info, "namespace": C.NAMESPACE_LABELS, "sources": C.source_hashes(),
            "privilege": "environment clone + exact solver cache: oracle simulation for label generation only; "
                         "charged offline (label ledger), never in episode utility, never available at evaluation",
            "state_distribution": f"per-world base kind by seed hash {dict(cfg.base_mix)} (d0 = greedy+R-mask; eps = d0 "
                                  f"with p={cfg.eps} random sample/mask_top; sampled = T=1 from pi); points by online "
                                  f"Bernoulli per public stratum (flagged {cfg.p_flagged}, low_margin {cfg.p_low_margin} "
                                  f"[top1-top2 < {cfg.low_margin}], other {cfg.p_other}), <= {cfg.max_points} per episode"}
    torch.save(pack(episodes, points, meta), stem.with_suffix(".pt"))
    summary = {**meta, "stats": {k: v for k, v in stats.items()}, "solver_cache": cache_stats,
               "episodes_summary": {"success": sum(e["final"]["success"] for e in episodes) / max(1, len(episodes)),
                                    "by_kind": {k: sum(e["base_kind"] == k for e in episodes)
                                                for k in ("d0", "eps", "sampled")},
                                    "decisions": sum(len(e["u"]) for e in episodes),
                                    "capped_points": sum(e["capped_points"] for e in episodes)},
               "points_by_stratum": {s: sum(p["stratum"] == s for p in points) for s in ("flagged", "low_margin", "other")},
               "process_cpu_seconds": time.process_time() - cpu0, "wall_seconds": time.perf_counter() - wall0,
               "cpu_scope": "parent process; solver worker CPU is in the job receipt (process tree)"}
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("stats", "episodes_summary", "points_by_stratum", "process_cpu_seconds",
                                               "wall_seconds")}, default=str))



if __name__ == "__main__":
    main()
