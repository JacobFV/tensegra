"""extended-05 A-HR headroom tooling (design §1 + v2 revisions 1-4, 8; registry A-HR). CPU only; chunked.

    CUDA_VISIBLE_DEVICES= python research/tools/campaign05_hr.py verify-d --base x1-r0 --condition iid_f0 \
        --chunk 0 --output results/e05-hr/verify          # pre-flight: D == deploy_episodes r_mask (dev worlds)
    CUDA_VISIBLE_DEVICES= python research/tools/campaign05_hr.py states --base x1-r0 --condition iid_f0 \
        --chunk 0 --output results/e05-hr/states          # D rollouts + the public sampling record (no branching)
    CUDA_VISIBLE_DEVICES= python research/tools/campaign05_hr.py branch --base x1-r0 --condition iid_f0 \
        --chunk 0 --output results/e05-hr/branch          # anchored O(I) branch labels
    CUDA_VISIBLE_DEVICES= python research/tools/campaign05_hr.py branch --base x1-r0 --condition iid_f0 \
        --chunk 0 --option-set full --no-all-states --episodes 12 --output results/e05-hr/full
    python research/tools/campaign05_hr.py analyze --branch results/e05-hr/branch --full results/e05-hr/full \
        --states results/e05-hr/states --output results/e05-hr/analysis.json   # numpy only

A-HR2 (registry A-HR2; multi-step option class, same worlds/points/main line as A-HR):
    CUDA_VISIBLE_DEVICES= python research/tools/campaign05_hr.py branch --option-class multi --base x1-r0 \
        --condition iid_f0 --chunk 0 1 2 3 --output results/e05-hr2/branch   # D + delegate (option T) labels
    python research/tools/campaign05_hr.py analyze --multi results/e05-hr2/branch --branch results/e05-hr/branch \
        --output results/e05-hr2/analysis.json   # G1a-multi / G1b-multi (T alone and T combined with O(I))
Option B (next-call budget override) is identical to existing A-HR options (budget / call_now) and is
not re-branched; ``analyze`` reports its hindsight value from the A-HR labels.

Worlds (research/campaigns/extended-05/seed-ranges.json; disjointness asserted at start):
- ``--worlds hr`` (default): 220,000,000 + 100,000*i + n (i = condition index: iid_f0 0, iid_f2 1);
  chunk c covers n = c*chunk_size ... c*chunk_size + chunk_size - 1; the same worlds for every base.
- ``--worlds dev``: 2,250,000,000 + 100,000*i + n (non-protocol smokes/tests; default for verify-d).
Conditions: the IID group (iid_f0, iid_f2; campaign03_p1_configs.SEALED definitions).

Bases (frozen X1 bootstraps; sha256 verified at load): x1-r0..r2 (extended-03 p1-boot-x1-r*) and
x1-r3..r5 (extended-04 f-boot-x1-r*), round-0-slot-5-attempt-5.pt. Checkpoint paths come from the
hashed configs through a path-remap layer (tensegra-campaign0X -> structured-latent-dynamics-campaign0X;
configs are never edited). Lineage roles (design v2 revision 4): r0-r2 for A-HR and screening;
r3-r5 are reserved for confirmation, so ``states``/``branch`` refuse them (verify-d is allowed).
Device: CPU, 1 thread (registered D device for branching/labels).

Outputs never overwrite (refuse). Each chunk writes <stem>.json.gz (rows) and <stem>.meta.json
(stats, CPU incl. solver-worker children, unit costs, hashes, sampling and privilege disclosure).
Branch labels are hidden-state (true-world clone) rollouts: an optimistic, non-deployable privilege
of offline analysis; their CPU is charged to the job receipt.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # analysis is single-threaded (metered)
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
from functools import partial
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import resource
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEED_RANGES = REPO / "research/campaigns/extended-05/seed-ranges.json"
CONDITIONS = ("iid_f0", "iid_f2")
WORLD_BASES = {"hr": 220_000_000, "dev": 2_250_000_000}
STRIDE = 100_000
NAMESPACES = {"hr": "e05hr", "dev": "e05hr-dev"}
CHUNK = 32
RESERVED_FOR_CONFIRMATION = ("x1-r3", "x1-r4", "x1-r5")
ROOT_RENAME = (("/home/brand/tensegra-campaign03/", "/home/brand/structured-latent-dynamics-campaign03/"),
               ("/home/brand/tensegra-campaign04/", "/home/brand/structured-latent-dynamics-campaign04/"))

def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- seed ranges ------------------------------------------------------------------------

def seed_ranges(path=SEED_RANGES):
    return json.loads(Path(path).read_text())["ranges"]


def check_seed_ranges(path=SEED_RANGES):
    """Assert pairwise disjointness of every registered range, and that this tool's world
    ranges lie inside their registered extended-05 entries."""
    rs = seed_ranges(path)
    for i, a in enumerate(rs):
        if not a["lo"] < a["hi"]:
            raise ValueError(f"empty range {a['name']}")
        for b in rs[i + 1:]:
            if a["lo"] < b["hi"] and b["lo"] < a["hi"]:
                raise ValueError(f"seed ranges overlap: {a['name']} / {b['name']}")
    for kind, base in WORLD_BASES.items():
        lo, hi = base, base + STRIDE * len(CONDITIONS)
        if not any(r["campaign"] == "extended-05" and r["lo"] <= lo and hi <= r["hi"] for r in rs):
            raise ValueError(f"{kind} worlds [{lo}, {hi}) are not inside a registered extended-05 range")
    return rs


def world_seeds(kind, condition, chunk, chunk_size, episodes=None):
    i = CONDITIONS.index(condition)
    first = WORLD_BASES[kind] + STRIDE * i + chunk * chunk_size
    n = chunk_size if episodes is None else min(episodes, chunk_size)
    if chunk < 0 or (chunk + 1) * chunk_size > STRIDE:
        raise SystemExit("chunk outside the registered range")
    return list(range(first, first + n))


# --- bases ------------------------------------------------------------------------------

def _rename(path):
    for old, new in ROOT_RENAME:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def bases(root=REPO):
    out = {}
    sealed = json.loads((root / "configs/campaign03/p1-sealed-checkpoints.json").read_text())
    for r in range(3):
        b = next(x for x in sealed if x["name"] == f"p1-boot-x1-r{r}")
        out[f"x1-r{r}"] = {"name": f"x1-r{r}", "source": b["name"], "path": _rename(b["path"]), "sha256": b["sha256"],
                           "lineage": r, "campaign": "extended-03"}
    for r in (3, 4, 5):
        cfg = json.loads((root / f"configs/campaign04/f1-f-boot-x1-r{r}.json").read_text())
        b = cfg["checkpoints"][0]
        assert b["name"] == f"f-boot-x1-r{r}" and b["family"] == "bootstrap"
        out[f"x1-r{r}"] = {"name": f"x1-r{r}", "source": b["name"], "path": _rename(b["path"]), "sha256": b["sha256"],
                           "lineage": r, "campaign": "extended-04"}
    for b in out.values():
        if not b["path"].endswith("/checkpoints/round-0-slot-5-attempt-5.pt"):
            raise ValueError(f"unexpected checkpoint {b['path']}")
    return out


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_actor(binding, verify=True):
    """(policy, TrainConfig, info); same loader as extended-04 Track C (campaign04_c_configs.load_actor)."""
    import torch
    from tensegra.campaign02_population import build_policy
    from tensegra.campaign02_training import TrainConfig
    if verify and file_hash(binding["path"]) != binding["sha256"]:
        raise SystemExit(f"checkpoint hash mismatch: {binding['name']}")
    checkpoint = torch.load(binding["path"], map_location="cpu", weights_only=False)
    train_cfg = TrainConfig(**checkpoint["config"])
    policy = build_policy(checkpoint["policy_config"])
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, train_cfg, {"sha256": binding["sha256"], "parameters": sum(p.numel() for p in policy.parameters()),
                               "family": policy.config.family, "width": policy.config.width,
                               "feature_version": policy.config.feature_version}


_SEALED = {}


def _sealed():
    """campaign03_p1_configs.SEALED condition definitions (the P1 / extended-04 IID conditions)."""
    if not _SEALED:
        _SEALED.update(dict(_load("campaign03_p1_configs").SEALED))
    return _SEALED


def world(seed, condition, executor, namespace):
    from tensegra.campaign02_training import independent_address_seed
    from tensegra.campaign03_depworld import DepWorkshop, generate_depworld
    kwargs = dict(_sealed()[condition])
    if "call_budgets" in kwargs:
        kwargs["call_budgets"] = tuple(kwargs["call_budgets"])
    return DepWorkshop(generate_depworld(seed, **kwargs), executor=executor,
                       address_seed=independent_address_seed(seed, namespace))


def source_hashes():
    names = ("campaign05_options", "campaign04_branch", "campaign04_progress", "campaign04_telemetry",
             "campaign04_deploy", "campaign03_depworld", "campaign02_training")
    out = {n: file_hash(REPO / f"src/tensegra/{n}.py") for n in names}
    out["campaign05_hr"] = file_hash(Path(__file__))
    out["seed-ranges.json"] = file_hash(SEED_RANGES)
    return out


# --- shared job plumbing ------------------------------------------------------------------

def _children_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _stem(a, prefix):
    extra = ""
    if getattr(a, "option_set", "anchored") == "full":
        extra = "-full"
    if getattr(a, "option_class", "single") == "multi":
        extra += "-multi"
    return a.output / f"{prefix}-{a.base}-{a.condition}-{a.worlds}{extra}-c{a.chunk:03d}"


def _write(stem, rows, meta):
    if stem.with_suffix(".json.gz").exists():
        raise SystemExit(f"refusing to overwrite {stem}.json.gz")
    with gzip.open(stem.with_suffix(".json.gz"), "wt") as f:
        json.dump(rows, f, separators=(",", ":"))
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _setup(a, default_worlds):
    import torch
    torch.set_num_threads(a.threads)
    check_seed_ranges()
    a.worlds = a.worlds or default_worlds
    binding = bases()[a.base] if a.binding is None else a.binding
    actor, train_cfg, info = (a.load_actor or load_actor)(binding, verify=not a.no_verify_hash)
    a.output.mkdir(parents=True, exist_ok=True)
    return binding, actor, train_cfg, info


def _meta(a, binding, info, train_cfg, seeds, cfg, **extra):
    from tensegra.campaign05_options import CONTINUATION, HR_VERSION, OPTION_SET_VERSION
    return {"version": HR_VERSION, "option_set_version": OPTION_SET_VERSION, "command": a.command,
            "base": binding["name"], "base_source": binding.get("source"), "base_sha256": binding["sha256"],
            "condition": a.condition, "worlds": a.worlds, "protocol_worlds": a.worlds != "dev",
            "namespace": NAMESPACES[a.worlds], "chunk": a.chunk, "seeds": [seeds[0], seeds[-1] + 1],
            "continuation": CONTINUATION, "device": "cpu", "max_steps": train_cfg.max_steps,
            "neural_work_per_forward": train_cfg.neural_work_per_forward, "actor": info, "sampling": cfg.describe(),
            "hr_config": cfg.__dict__, "sources": source_hashes(),
            "privilege": "environment clone of the TRUE world + exact solver cache: hidden-state rollout labels; "
                         "optimistic, non-deployable; offline only, charged to this job's receipt", **extra}




# --- commands ----------------------------------------------------------------------------

def _run_hr(a, branch):
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign03_depworld import depworld_executor
    from tensegra.campaign04_branch import SolverCache
    from tensegra.campaign05_options import HRConfig, hr_labels
    if a.base in RESERVED_FOR_CONFIRMATION and a.binding is None:
        raise SystemExit(f"{a.base} is reserved for confirmation (design v2 revision 4); A-HR uses x1-r0..r2")
    if getattr(a, "option_class", "single") == "multi" and getattr(a, "option_set", "anchored") != "anchored":
        raise SystemExit("--option-class multi uses its own option set (D + delegate); do not combine with --option-set")
    binding, actor, train_cfg, info = _setup(a, "hr")
    stem = _stem(a, "branch" if branch else "states")
    if stem.with_suffix(".json.gz").exists():
        raise SystemExit(f"refusing to overwrite {stem}.json.gz")
    wall0, cpu0, ch0 = time.perf_counter(), time.process_time(), _children_cpu()
    mode = "multi" if getattr(a, "option_class", "single") == "multi" else getattr(a, "option_set", "anchored")
    cfg = HRConfig(max_steps=train_cfg.max_steps, option_mode=mode, all_states=not a.no_all_states)
    seeds = world_seeds(a.worlds, a.condition, a.chunk, a.chunk_size, a.episodes)
    with (a.solver or BoundedSolver)() as solver:
        cache = SolverCache(partial(depworld_executor, execute_call=solver.execute))
        envs = [world(s, a.condition, cache, NAMESPACES[a.worlds]) for s in seeds]
        episodes, points, stats = hr_labels(actor, envs, seeds, cfg,
                                            neural_work_per_forward=train_cfg.neural_work_per_forward,
                                            branch_batch=a.branch_batch, branch=branch)
        cache_stats = cache.stats()
    cpu = {"process_s": time.process_time() - cpu0, "children_s": _children_cpu() - ch0,
           "wall_s": time.perf_counter() - wall0}
    return binding, info, train_cfg, stem, cfg, seeds, episodes, points, stats, cache_stats, cpu


def _episode_summary(episodes):
    n = max(1, len(episodes))
    anchors = {}
    for e in episodes:
        for d in e["decisions"]:
            anchors[d["anchor"]] = anchors.get(d["anchor"], 0) + 1
    decisions = max(1, sum(anchors.values()))
    return {"episodes": len(episodes), "success": sum(e["success"] for e in episodes) / n,
            "utility": sum(e["utility"] for e in episodes) / n, "mean_T": sum(e["T"] for e in episodes) / n,
            "sampled_points_per_episode": sum(len(e["sampled_steps"]) for e in episodes) / n,
            "all_states_episodes": sum(e["all_states"] for e in episodes),
            "anchor_fraction": {k: v / decisions for k, v in sorted(anchors.items())}}


def cmd_states(a):
    binding, info, train_cfg, stem, cfg, seeds, episodes, _, stats, cache_stats, cpu = _run_hr(a, branch=False)
    summary = _episode_summary(episodes)
    total = cpu["process_s"] + cpu["children_s"]
    _write(stem, {"episodes": episodes, "points": []},
           _meta(a, binding, info, train_cfg, seeds, cfg, summary=summary, solver_cache=cache_stats, cpu=cpu,
                 unit_costs={"core_s_per_episode": total / max(1, len(episodes))}))
    print(json.dumps({"stem": str(stem), **summary, "cpu": cpu}))


def cmd_branch(a):
    from tensegra.campaign05_options import summarize_states
    binding, info, train_cfg, stem, cfg, seeds, episodes, points, stats, cache_stats, cpu = _run_hr(a, branch=True)
    total = cpu["process_s"] + cpu["children_s"]
    bsteps = sorted(stats.pop("branch_steps"))
    q = (lambda f: bsteps[min(len(bsteps) - 1, int(f * len(bsteps)))] if bsteps else None)  # noqa: E731
    branch_cpu = stats["cpu"]["branch_s"] + cpu["children_s"]   # solver worker CPU is (almost) all branch calls
    unit = {"core_s_per_branch": branch_cpu / max(1, stats["branches"]),
            "core_s_per_branch_step": branch_cpu / max(1, sum(bsteps)),
            "core_s_per_state": total / max(1, stats["points"]),
            "core_s_per_episode": total / max(1, stats["episodes"]),
            "core_s_main_line_per_episode": stats["cpu"]["main_s"] / max(1, stats["episodes"]),
            "branch_steps_quantiles": {"p50": q(.5), "p90": q(.9), "p99": q(.99), "max": bsteps[-1] if bsteps else None,
                                       "mean": sum(bsteps) / max(1, len(bsteps))},
            "long_branch_share_of_steps": (sum(x for x in bsteps if x >= 48) / max(1, sum(bsteps)))}
    summary = {**_episode_summary(episodes), **summarize_states(points), "branched_points": stats["points"],
               "sampled_points": stats["sampled_points"], "branches": stats["branches"],
               "default_checks": [stats["default_check_matches"], stats["default_checks"]]}
    extra = {}
    if a.option_class == "multi":
        extra = _multi_meta()
        summary.update(_multi_summary(points))
        unit["core_s_per_delegate_branch"] = branch_cpu / max(1, stats["branches"])
    if stats["default_check_matches"] != stats["default_checks"]:
        summary["WARNING"] = "determinism check failed: a cloned D branch differs from the main line"
    _write(stem, {"episodes": episodes, "points": points},
           _meta(a, binding, info, train_cfg, seeds, cfg, option_set=a.option_set, **extra, summary=summary, stats=stats,
                 solver_cache=cache_stats, cpu=cpu, unit_costs=unit,
                 cpu_scope="process_s = this process; children_s = solver worker(s) (reaped)"))
    print(json.dumps({"stem": str(stem), **summary, "unit_costs": unit, "cpu": cpu}))


def _multi_meta():
    from tensegra.campaign05_options import B_BUDGETS, DELEGATE_MAX_STEPS, MULTI_VERSION, TEACHER
    return {"option_class": "multi", "multi_version": MULTI_VERSION,
            "options": {"delegate": f"option T: a fresh public {TEACHER} teacher (DepReference('reuse'), stateless, public "
                                    f"observation + catalog only; a supplied sub-policy, disclosed) from the point until "
                                    f"its next successful commit (selection/assignment, direct or via use_return), the "
                                    f"episode end, or {DELEGATE_MAX_STEPS} teacher steps; then D; tracker carried "
                                    f"through; one controller forward charged per decision (teacher steps included)",
                        "budget_override": f"option B (b in {list(B_BUDGETS)}): identical to existing A-HR options "
                                           f"(budget / call_now); not re-branched (b_options, b_subset_of_oi recorded)"}}


def _multi_summary(points):
    dl = [o for p in points for o in p["options"] if o["type"] == "delegate"]
    ends = {}
    for o in dl:
        ends[o["delegate_end"]] = ends.get(o["delegate_end"], 0) + 1
    n = max(1, len(dl))
    return {"delegate_options": len(dl), "delegate_unavailable_points": len(points) - len(dl),
            "delegate_mean_adv": sum(o["dU"] - p["q_d"] for p in points for o in p["options"]
                                     if o["type"] == "delegate") / n,
            "delegate_mean_teacher_steps": sum(o["teacher_steps"] for o in dl) / n,
            "delegate_teacher_agree_rate": sum(o["teacher_agree"] for o in dl) / max(1, sum(o["teacher_steps"] for o in dl)),
            "delegate_end": ends, "b_subset_of_oi_all": all(p.get("b_subset_of_oi", True) for p in points),
            "b_points": sum(bool(p.get("b_options")) for p in points)}


def cmd_verify_d(a):
    """D wrapper (run_policy + d_choice) vs campaign04_deploy.deploy_episodes(mode="r_mask") on real
    worlds and a real base: actions, utility, compute units and tracker summary must be identical."""
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign03_depworld import depworld_executor
    from tensegra.campaign04_branch import SolverCache
    from tensegra.campaign04_deploy import deploy_episodes
    from tensegra.campaign05_options import HRConfig, d_rollouts, outcome
    binding, actor, train_cfg, info = _setup(a, "dev")
    stem = _stem(a, "verify-d")
    cpu0, ch0 = time.process_time(), _children_cpu()
    cfg = HRConfig(max_steps=train_cfg.max_steps)
    seeds = world_seeds(a.worlds, a.condition, a.chunk, a.chunk_size, a.episodes)
    with (a.solver or BoundedSolver)() as solver:
        cache = SolverCache(partial(depworld_executor, execute_call=solver.execute))
        make = lambda: [world(s, a.condition, cache, NAMESPACES[a.worlds]) for s in seeds]  # noqa: E731
        ref = deploy_episodes(actor, make(), mode="r_mask", max_steps=train_cfg.max_steps,
                              neural_work_per_forward=train_cfg.neural_work_per_forward)
        eps = d_rollouts(actor, make(), seeds, cfg, neural_work_per_forward=train_cfg.neural_work_per_forward)
        cache_stats = cache.stats()
    rows = []
    for s, r, ep in zip(seeds, ref, eps):
        o = outcome(ep.env)
        rows.append({"seed": s, "steps": len(ep.actions_taken),
                     "actions": ep.actions_taken == [t["action_index"] for t in r["trace"]],
                     "utility": o["utility"] == r["outcome"]["utility"],
                     "compute_units": o["compute_units"] == r["outcome"]["compute_units"],
                     "progress": ep.tracker.summary() == r["progress"], "success": o["success"]})
    identical = all(all(v for k, v in r.items() if k in ("actions", "utility", "compute_units", "progress"))
                    for r in rows)
    cpu = {"process_s": time.process_time() - cpu0, "children_s": _children_cpu() - ch0}
    _write(stem, rows, _meta(a, binding, info, train_cfg, seeds, cfg, identical=identical, solver_cache=cache_stats,
                             cpu=cpu))
    print(json.dumps({"stem": str(stem), "identical": identical, "episodes": len(rows),
                      "success": sum(r["success"] for r in rows) / len(rows),
                      "mean_steps": sum(r["steps"] for r in rows) / len(rows), "cpu": cpu}))
    if not identical:
        raise SystemExit("D wrapper differs from deploy_episodes r_mask")




# --- analysis (numpy; no torch) --------------------------------------------------------------

M_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)
FEATURE_KEYS = ("prob", "logit_gap", "rank_frac", "budget_log", "budget_frac", "budget_vs_d_log", "problem_latest",
                "prev_calls", "prev_timeouts", "prev_max_budget_log", "rel_type_match", "rel_request_match",
                "rel_canonical_match", "rel_dependency_match", "rel_requirements_match", "rel_selection_match",
                "rel_usable", "rec_timeout", "rec_age", "rec_foreign")


DELEGATE_KEYS = ("teacher_first_is_d",) + tuple(f"teacher_first_{k}" for k in (
    "call", "reuse", "retrieve", "recompute", "commit", "revise", "abstain", "verify", "other")) + tuple(
    f"anchor_{k}" for k in ("call", "reuse_recompute", "commit_revise", "other"))


def _read(dirs, prefix):
    out = []
    for d in dirs or []:
        for p in sorted(Path(d).glob(f"{prefix}-*.json.gz")):
            meta = json.loads(p.with_name(p.name[:-len(".json.gz")] + ".meta.json").read_text())
            with gzip.open(p, "rt") as f:
                out.append((meta, json.load(f)))
    return out


def _mean_ci(values, clusters=None, reps=2000, seed=0):
    """Mean with percentile bootstrap 90% and 95% CIs, resampling clusters (worlds)."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "ci90": None, "ci95": None}
    clusters = list(range(v.size)) if clusters is None else list(clusters)
    keys = sorted(set(clusters))
    pos = {k: i for i, k in enumerate(keys)}
    sums, counts = np.zeros(len(keys)), np.zeros(len(keys))
    for x, c in zip(v, clusters):
        sums[pos[c]] += x
        counts[pos[c]] += 1
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), (reps, len(keys)))
    boots = sums[pick].sum(1) / counts[pick].sum(1)
    return {"n": int(v.size), "clusters": len(keys), "mean": float(v.mean()),
            "ci90": [float(x) for x in np.percentile(boots, [5, 95])],
            "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}


def _world(group_condition, seed):
    """World key: the same world across lineages (paired) -> one cluster / one fold."""
    return f"{group_condition}/{seed}"


def _collect(branch):
    """[(meta, episode, [points of that episode sorted by step])] for anchored (non-full) branch files."""
    out = []
    for meta, rows in branch:
        if meta.get("option_set") == "full" or meta.get("option_class") == "multi":
            continue
        by = {}
        for p in rows["points"]:
            by.setdefault(p["seed"], []).append(p)
        for e in rows["episodes"]:
            out.append((meta, e, sorted(by.get(e["seed"], []), key=lambda p: p["step"])))
    return out


def _adv(p, o):
    return o["dU"] - p["q_d"]


def hindsight_stats(eps, thr=0.01):
    """HINDSIGHT (hidden-state) headroom H: optimistic, non-deployable (G1a necessary condition)."""
    import numpy as np
    pts = [(m, e, p) for m, e, ps in eps for p in ps]
    worlds_s = [_world(m["condition"], p["seed"]) for m, e, p in pts]
    h = [p["headroom"] for _, _, p in pts]
    hx = [p["headroom_excl_abstain"] for _, _, p in pts]
    classes = {}
    for _, _, p in pts:
        best = {}
        for o in p["options"]:
            if not o["is_d"]:
                best[o["type"]] = max(best.get(o["type"], -math.inf), _adv(p, o))
        for t, g in best.items():
            classes.setdefault(t, []).append(g)
    by_anchor = {}
    for _, _, p in pts:
        by_anchor.setdefault(p["anchor"], []).append(p["headroom_excl_abstain"])
    by_outcome = {}
    for _, e, p in pts:
        by_outcome.setdefault("D_success" if e["success"] else "D_failure", []).append(p["headroom_excl_abstain"])
    # episode level, single-deviation class at the sampled points (the registered policy class)
    ep_x, ep_a, ep_w, ep_cls = [], [], [], {}
    for m, e, ps in eps:
        sp = [p for p in ps if p["sampled"]]
        ep_x.append(max([0.0] + [p["headroom_excl_abstain"] for p in sp]))
        ep_a.append(max([0.0] + [p["headroom"] for p in sp]))
        ep_w.append(_world(m["condition"], e["seed"]))
        for t in set(o["type"] for p in sp for o in p["options"] if not o["is_d"]) | set(classes):
            g = max([0.0] + [_adv(p, o) for p in sp for o in p["options"] if not o["is_d"] and o["type"] == t])
            ep_cls.setdefault(t, []).append(g)
    all_x = [max([0.0] + [p["headroom_excl_abstain"] for p in ps]) for m, e, ps in eps if e["all_states"]]
    all_a = [max([0.0] + [p["headroom"] for p in ps]) for m, e, ps in eps if e["all_states"]]
    q = (lambda v: {k: float(np.quantile(v, k)) for k in (.5, .9, .95, .99)} if v else {})  # noqa: E731
    return {
        "label": "HINDSIGHT hidden-state bound (max over realized true-world branches): optimistic, non-deployable; "
                 "a necessary condition only (G1a); abstain is its own class and never counts toward G1",
        "states": len(pts), "episodes": len(eps), "worlds": len(set(ep_w)),
        "state_H_excl_abstain": {**_mean_ci(hx, worlds_s), "quantiles": q(hx),
                                 "frac_ge_thr": float(np.mean(np.array(hx) >= thr)) if hx else None},
        "state_H_incl_abstain": {**_mean_ci(h, worlds_s), "quantiles": q(h),
                                 "frac_ge_thr": float(np.mean(np.array(h) >= thr)) if h else None},
        "state_H_excl_abstain_by_anchor": {k: {"states": len(v), "mean": float(np.mean(v)),
                                               "frac_ge_thr": float(np.mean(np.array(v) >= thr))}
                                           for k, v in sorted(by_anchor.items())},
        "state_H_excl_abstain_by_D_episode_outcome": {k: {"states": len(v), "mean": float(np.mean(v)),
                                                          "frac_ge_thr": float(np.mean(np.array(v) >= thr))}
                                                      for k, v in sorted(by_outcome.items())},
        "state_best_gain_by_option_class": {t: {"states_with_class": len(v), "mean": float(np.mean(v)),
                                                "mean_positive_part": float(np.mean(np.maximum(v, 0))),
                                                "frac_ge_thr": float(np.mean(np.array(v) >= thr))}
                                            for t, v in sorted(classes.items())},
        "episode_single_deviation_H_excl_abstain": _mean_ci(ep_x, ep_w),
        "episode_single_deviation_H_incl_abstain": _mean_ci(ep_a, ep_w),
        "episode_single_deviation_H_by_class": {t: _mean_ci(v, ep_w)["mean"] for t, v in sorted(ep_cls.items())
                                                if len(v) == len(ep_w)},
        "all_states_episodes_H_excl_abstain": _mean_ci(all_x), "all_states_episodes_H_incl_abstain": _mean_ci(all_a),
        "G1_hindsight_excl_abstain_ge_.02": (_mean_ci(ep_x, ep_w)["mean"] or 0.0) >= 0.02,
        "thr": thr}


def _x(p, o):
    """Public feature vector [telemetry d_t, option features, 1] (cached on the option dict)."""
    x = o.get("_x")
    if x is None:
        import numpy as np
        f = o.get("features") or {}
        keys = FEATURE_KEYS + DELEGATE_KEYS if o["type"] == "delegate" else FEATURE_KEYS
        x = o["_x"] = np.array(list(p["telemetry"]) + [float(f.get(k, 0.0)) for k in keys] + [1.0])
    return x


def _fit(points, ridge, allow_abstain):
    import numpy as np
    rows = {}
    for p in points:
        for o in p["options"]:
            if o["is_d"] or (o["type"] == "abstain" and not allow_abstain):
                continue
            rows.setdefault(o["type"], []).append((_x(p, o), _adv(p, o)))
    models, const = {}, {}
    for t, data in rows.items():
        X = np.stack([x for x, _ in data])
        y = np.array([v for _, v in data])
        reg = ridge * np.eye(X.shape[1])
        reg[-1, -1] = 0.0
        models[t] = np.linalg.solve(X.T @ X + reg, X.T @ y)
        const[t] = float(y.mean())
    return models, const


def _deviate(ps, predict, margin, allow_abstain):
    """Single-deviation policy on one episode: at the sampled points in step order, deviate once to the
    option with the largest predicted advantage if it exceeds the margin; else D. Returns (gain, type)."""
    return _deviate_point(ps, predict, margin, allow_abstain)[:2]


def _deviate_point(ps, predict, margin, allow_abstain):
    """``_deviate`` plus the firing point (None if D is kept throughout)."""
    for p in ps:
        if not p["sampled"]:
            continue
        best, bp = None, margin
        for o in p["options"]:
            if o["is_d"] or (o["type"] == "abstain" and not allow_abstain):
                continue
            v = predict(p, o)
            if v is not None and v > bp:
                best, bp = o, v
        if best is not None:
            return _adv(p, best), best["type"], p
    return 0.0, None, None


def _predictor(models):
    def predict(p, o):
        w = models.get(o["type"])
        return None if w is None else float(_x(p, o) @ w)
    return predict


def _folds(keys, k, seed):
    keys = sorted(set(keys))
    random.Random(seed).shuffle(keys)
    return {w: i % k for i, w in enumerate(keys)}


def single_deviation_estimate(eps, folds=5, inner=3, ridge=1.0, allow_abstain=False, seed=0, breakdown=False):
    """SAME-INFORMATION estimate of the single-deviation-from-D policy (G1b): per option type, a ridge
    regression Q-hat^D advantage on public features [telemetry d_t, option features, 1], cross-fitted by
    WORLD (outer folds; the margin m chosen from M_GRID by inner world-folds on the training worlds only).
    The held-out episode gain equals the TRUE branched advantage of the chosen option at the first firing
    sampled point (0 if the rule never fires)."""
    wkey = [_world(m["condition"], e["seed"]) for m, e, _ in eps]
    fold = _folds(wkey, folds, seed)
    gains, types, worlds, chosen_m, const_gains, fixed = [], [], [], [], [], {m: [] for m in (0.0, 0.01)}
    outcomes, fire_anchor = [], []
    for f in range(folds):
        train = [x for x, w in zip(eps, wkey) if fold[w] != f]
        test = [(x, w) for x, w in zip(eps, wkey) if fold[w] == f]
        if not train or not test:
            continue
        tw = [_world(m["condition"], e["seed"]) for m, e, _ in train]
        ifold = _folds(tw, inner, seed + 1 + f)
        score = {m: 0.0 for m in M_GRID}
        for g in range(inner):
            itrain = [p for (m, e, ps), w in zip(train, tw) if ifold[w] != g for p in ps]
            itest = [ps for (m, e, ps), w in zip(train, tw) if ifold[w] == g]
            if not itrain or not itest:
                continue
            pred = _predictor(_fit(itrain, ridge, allow_abstain)[0])
            for m in M_GRID:
                score[m] += sum(_deviate(ps, pred, m, allow_abstain)[0] for ps in itest)
        m_star = max(M_GRID, key=lambda m: (score[m], m))
        models, const = _fit([p for _, _, ps in train for p in ps], ridge, allow_abstain)
        pred = _predictor(models)
        cpred = (lambda p, o: const.get(o["type"]))  # noqa: E731
        for (m, e, ps), w in test:
            g, t, fp = _deviate_point(ps, pred, m_star, allow_abstain)
            outcomes.append("D_success" if e["success"] else "D_failure")
            fire_anchor.append(fp["anchor"] if fp is not None else "none")
            gains.append(g)
            types.append(t or "D")
            worlds.append(w)
            chosen_m.append(m_star)
            const_gains.append(_deviate(ps, cpred, 0.0, allow_abstain)[0])
            for mm in fixed:
                fixed[mm].append(_deviate(ps, pred, mm, allow_abstain)[0])
    est = _mean_ci(gains, worlds)
    extra = {}
    if breakdown:
        def split(keys):
            out = {}
            for k in sorted(set(keys)):
                idx = [i for i, x in enumerate(keys) if x == k]
                out[k] = {**_mean_ci([gains[i] for i in idx], [worlds[i] for i in idx]),
                          "share_of_total_gain": sum(gains[i] for i in idx) / max(1, len(gains)),
                          "deviations": sum(types[i] != "D" for i in idx)}
            return out
        extra = {"by_D_episode_outcome": split(outcomes), "by_firing_anchor": split(fire_anchor)}
    return {"label": "SAME-INFORMATION single-deviation-from-D estimate (public features; per-type ridge Q-hat^D; "
                     f"cross-fitted by world, {folds} outer / {inner} inner folds; margin from inner folds); "
                     "episode gain = true branched advantage of the chosen option",
            "allow_abstain": allow_abstain, "episodes": len(gains), "gain_per_episode": est,
            "G1b_ge_.01_and_ci90_excludes_0": bool(est["mean"] is not None and est["mean"] >= 0.01
                                                   and est["ci90"][0] > 0),
            "deviation_rate": sum(t != "D" for t in types) / max(1, len(types)),
            "deviation_types": {t: types.count(t) for t in sorted(set(types))},
            "harmful_rate": sum(g < -1e-12 for g in gains) / max(1, len(gains)),
            "chosen_margins": {str(m): chosen_m.count(m) for m in sorted(set(chosen_m))},
            "fixed_margin_gain": {str(m): _mean_ci(v, worlds)["mean"] for m, v in fixed.items()},
            "constant_per_type_baseline_gain": _mean_ci(const_gains, worlds)["mean"], "ridge": ridge, **extra}


def full_catalog_stats(full, branch, thr):
    import numpy as np
    oi = {(m["base"], m["condition"], m["worlds"], p["seed"], p["step"]): p for m, r in branch
          if m.get("option_set") != "full" for p in r["points"]}
    h_full, h_oi, hx_full, hx_oi, overlap, matched, best = [], [], [], [], 0, 0, {}
    for m, r in full:
        for p in r["points"]:
            in_oi = [o for o in p["options"] if o.get("in_oi") or o["is_d"]]
            h_full.append(p["headroom"])
            h_oi.append(max(o["dU"] for o in in_oi) - p["q_d"])
            hx_full.append(p["headroom_excl_abstain"])
            hx_oi.append(max(o["dU"] for o in in_oi if o["types"][-1] != "abstain" or o["is_d"]) - p["q_d"])
            k = (m["base"], m["condition"], m["worlds"], p["seed"], p["step"])
            if k in oi:
                overlap += 1
                matched += int(math.isclose(oi[k]["headroom"], h_oi[-1], abs_tol=1e-12))
            if p["headroom_excl_abstain"] > 0:
                cand = [o for o in p["options"] if o["types"][-1] != "abstain"]
                b = max(cand, key=lambda o: (o["dU"], o["is_d"], -o["index"]))
                key = b["types"][-1] + ("" if b.get("in_oi") else " (outside O(I))")
                best[key] = best.get(key, 0) + 1

    def ratio(a, b):
        return float(np.mean(a) / np.mean(b)) if a and np.mean(b) > 0 else None
    return {"label": "full-catalog one-step deviation vs anchored O(I) on the same states (hindsight; O(I) value = "
                     "full-catalog branches restricted to O(I))",
            "states": len(h_full), "mean_H_full": float(np.mean(h_full)) if h_full else None,
            "mean_H_oi": float(np.mean(h_oi)) if h_oi else None, "ratio_full_over_oi": ratio(h_full, h_oi),
            "mean_H_full_excl_abstain": float(np.mean(hx_full)) if hx_full else None,
            "mean_H_oi_excl_abstain": float(np.mean(hx_oi)) if hx_oi else None,
            "ratio_full_over_oi_excl_abstain": ratio(hx_full, hx_oi),
            "difference_full_minus_oi_excl_abstain": float(np.mean(hx_full) - np.mean(hx_oi)) if hx_full else None,
            "frac_states_full_exceeds_oi_by_thr_excl_abstain":
                float(np.mean(np.array(hx_full) - np.array(hx_oi) >= thr)) if hx_full else None,
            "oi_file_overlap_states": overlap, "oi_file_consistency_matches": matched,
            "best_non_abstain_option_when_positive": best}


def _merge_multi(multi_eps, hr_eps):
    """Pair A-HR2 episodes with the A-HR episodes on the same (base, condition, worlds, seed): the main line
    must be identical (actions, utility) and every multi point must match an A-HR point (U_t, q_d, D index).
    Returns (combined episodes: A-HR points + the delegate option, B indices per point, report)."""
    hr = {(m["base"], m["condition"], m["worlds"], e["seed"]): (m, e, ps) for m, e, ps in hr_eps}
    combined, b_of, rep = [], {}, {"multi_episodes": len(multi_eps), "paired_episodes": 0, "main_line_mismatch": 0,
                                   "point_mismatch": 0, "unpaired_points": 0}
    for m, e, ps in multi_eps:
        k = (m["base"], m["condition"], m["worlds"], e["seed"])
        if k not in hr:
            continue
        hm, he, hps = hr[k]
        if he["actions"] != e["actions"] or he["utility"] != e["utility"]:
            rep["main_line_mismatch"] += 1
            continue
        rep["paired_episodes"] += 1
        by_step = {p["step"]: p for p in hps}
        new_ps = []
        for p in ps:
            h = by_step.get(p["step"])
            if h is None:
                rep["unpaired_points"] += 1
                continue
            if (h["U_t"], h["q_d"], h["d_index"], h["sampled"]) != (p["U_t"], p["q_d"], p["d_index"], p["sampled"]):
                rep["point_mismatch"] += 1
                continue
            b_of[(k, p["step"])] = set(p.get("b_options") or [])
            new_ps.append({**h, "options": list(h["options"]) + [o for o in p["options"] if o["type"] == "delegate"]})
        combined.append((hm, he, new_ps))
    return combined, b_of, rep


def _ep_single_dev(eps, pick):
    """Per-episode hindsight single-deviation gain: max(0, best advantage over the picked options at the
    sampled points). ``pick(key, p, o)`` selects candidate (non-D) options."""
    gains, worlds, outcome = [], [], []
    for m, e, ps in eps:
        k = (m["base"], m["condition"], m["worlds"], e["seed"])
        g = 0.0
        for p in ps:
            if p["sampled"]:
                for o in p["options"]:
                    if not o["is_d"] and pick(k, p, o):
                        g = max(g, _adv(p, o))
        gains.append(g)
        worlds.append(_world(m["condition"], e["seed"]))
        outcome.append("D_success" if e["success"] else "D_failure")
    return gains, worlds, outcome


def _by(values, worlds, keys):
    out = {}
    for k in sorted(set(keys)):
        idx = [i for i, x in enumerate(keys) if x == k]
        out[k] = _mean_ci([values[i] for i in idx], [worlds[i] for i in idx])
    return out


def multi_stats(multi_eps, combined, b_of, thr, a):
    """A-HR2 statistics: G1a-multi (hindsight single deviation; T alone, T with O(I), B from the A-HR labels)
    and G1b-multi (cross-fitted same-information single deviation; T alone and T with O(I))."""
    import numpy as np
    out = {"label": "A-HR2 multi-step options. G1a-multi = HINDSIGHT single-deviation bound (true-world branches; "
                    "optimistic, non-deployable; necessary condition only). G1b-multi = SAME-INFORMATION cross-fitted "
                    "estimate (public features; per-option-type ridge; margin from inner folds). Abstain excluded.",
           "thr": thr}
    is_t = (lambda k, p, o: o["type"] == "delegate")  # noqa: E731
    g, w, oc = _ep_single_dev(multi_eps, is_t)
    out["G1a_multi_T"] = {**_mean_ci(g, w), "by_D_episode_outcome": _by(g, w, oc),
                          "by_base": _by(g, w, [m["base"] for m, _, _ in multi_eps]),
                          "gate_ge_.02": (_mean_ci(g, w)["mean"] or 0.0) >= 0.02}
    # per-state delegate advantage (sampled and all-states points)
    st = [(m, e, p, o) for m, e, ps in multi_eps for p in ps for o in p["options"] if o["type"] == "delegate"]
    adv = [_adv(p, o) for _, _, p, o in st]
    sw = [_world(m["condition"], p["seed"]) for m, _, p, _ in st]
    anchors = [p["anchor"] for _, _, p, _ in st]
    ends = [o["delegate_end"] for *_, o in st]
    out["state_delegate_advantage"] = {
        **_mean_ci(adv, sw), "positive_part_mean": float(np.mean(np.maximum(adv, 0))) if adv else None,
        "frac_ge_thr": float(np.mean(np.array(adv) >= thr)) if adv else None,
        "frac_le_minus_thr": float(np.mean(np.array(adv) <= -thr)) if adv else None,
        "by_anchor": {k: {**v, "frac_ge_thr": float(np.mean([adv[i] >= thr for i, x in enumerate(anchors) if x == k]))}
                      for k, v in _by(adv, sw, anchors).items()},
        "by_D_episode_outcome": _by(adv, sw, ["D_success" if e["success"] else "D_failure" for _, e, _, _ in st]),
        "by_delegate_end": _by(adv, sw, ends),
        "teacher_steps_mean": float(np.mean([o["teacher_steps"] for *_, o in st])) if st else None,
        "teacher_agree_rate": (sum(o["teacher_agree"] for *_, o in st) / max(1, sum(o["teacher_steps"] for *_, o in st))),
        "teacher_first_is_d_rate": float(np.mean([o["teacher_first_is_d"] for *_, o in st])) if st else None,
        "delegate_end_counts": {k: ends.count(k) for k in sorted(set(ends))},
        "points_without_delegate": sum(1 for _, _, ps in multi_eps for p in ps
                                       if not any(o["type"] == "delegate" for o in p["options"]))}
    if combined:
        oi_na = (lambda k, p, o: o["type"] not in ("abstain", "delegate"))  # noqa: E731
        comb = (lambda k, p, o: o["type"] != "abstain")  # noqa: E731
        b_only = (lambda k, p, o: o["type"] != "delegate" and o["index"] in b_of.get((k, p["step"]), ()))  # noqa: E731
        res = {}
        for name, pick in (("O(I)_excl_abstain", oi_na), ("T_plus_O(I)_excl_abstain", comb),
                           ("B_from_A-HR_labels", b_only), ("T_on_paired_episodes", is_t)):
            g, w, oc = _ep_single_dev(combined, pick)
            res[name] = {**_mean_ci(g, w), "by_D_episode_outcome": _by(g, w, oc),
                         "gate_ge_.02": (_mean_ci(g, w)["mean"] or 0.0) >= 0.02}
        out["G1a_multi_paired"] = res
    if multi_eps and all("telemetry" in p for _, _, ps in multi_eps for p in ps):
        out["G1b_multi_T"] = single_deviation_estimate(multi_eps, a.folds, a.inner, a.ridge, False, breakdown=True)
    if combined and all("telemetry" in p for _, _, ps in combined for p in ps):
        out["G1b_multi_T_plus_O(I)_excl_abstain"] = single_deviation_estimate(combined, a.folds, a.inner, a.ridge,
                                                                               False, breakdown=True)
    return out


def cmd_analyze(a):
    branch = _read(a.branch, "branch")
    full = _read(a.full, "branch")
    states = _read(a.states, "states")
    rep = {"version": "e05-hr-analysis-v2", "thr": a.thr, "groups": {}, "notes": [
        "HINDSIGHT H is a hidden-state rollout bound (max over realized true-world branches): optimistic and "
        "non-deployable by construction; G1a (H excl. abstain >= .02 per episode) is a necessary condition only.",
        "G1b is the SAME-INFORMATION estimate (public features only) of the single-deviation-from-D policy class; "
        "its episode gain equals the true branched advantage at the first firing sampled point.",
        "Abstain is its own class everywhere and never pooled into G1/G1b (reported separately).",
        "Per-state headroom is never summed along trajectories."]}
    eps = _collect(branch)
    groups = {}
    for x in eps:
        groups.setdefault(f"{x[0]['base']}/{x[0]['condition']}/{x[0]['worlds']}", []).append(x)
    for k, v in sorted(groups.items()):
        rep["groups"][k] = {"hindsight": hindsight_stats(v, a.thr)}
        if a.per_group_estimate:
            rep["groups"][k]["G1b"] = single_deviation_estimate(v, a.folds, a.inner, a.ridge)
    if eps:
        rep["pooled"] = {"hindsight": hindsight_stats(eps, a.thr)}
        if all("telemetry" in p for _, _, ps in eps for p in ps):
            rep["pooled"]["G1b_excl_abstain"] = single_deviation_estimate(eps, a.folds, a.inner, a.ridge, False)
            rep["pooled"]["G1b_abstain_class_included_(separate)"] = single_deviation_estimate(
                eps, a.folds, a.inner, a.ridge, True)
    if full:
        rep["full_catalog"] = full_catalog_stats(full, branch, a.thr)
    if getattr(a, "multi", None):
        multi = []
        for meta, rows in _read(a.multi, "branch"):
            if meta.get("option_class") != "multi":
                raise SystemExit("--multi expects --option-class multi branch files")
            by = {}
            for p in rows["points"]:
                by.setdefault(p["seed"], []).append(p)
            multi += [(meta, e, sorted(by.get(e["seed"], []), key=lambda p: p["step"])) for e in rows["episodes"]]
        combined, b_of, pairing = _merge_multi(multi, eps)
        rep["multi"] = {"pairing": pairing, **multi_stats(multi, combined, b_of, a.thr, a)}
    if states:
        rep["states"] = {}
        for meta, rows in states:
            k = f"{meta['base']}/{meta['condition']}/{meta['worlds']}"
            rep["states"].setdefault(k, []).extend(rows["episodes"])
        rep["states"] = {k: _episode_summary(v) for k, v in rep["states"].items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in rep if k in ("pooled", "full_catalog", "multi")}, indent=1,
                     default=str)[:8000])


# --- CLI -----------------------------------------------------------------------------------

def main(argv=None, *, binding=None, load_actor_fn=None, solver=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("states", "branch", "verify-d"):
        s = sub.add_parser(name)
        s.add_argument("--base", required=True, help="x1-r0 ... x1-r2 (x1-r3..r5: verify-d only)")
        s.add_argument("--condition", required=True, choices=CONDITIONS)
        s.add_argument("--chunk", type=int, nargs="+", required=True, help="one or more chunks, run sequentially")
        s.add_argument("--chunk-size", type=int, default=CHUNK)
        s.add_argument("--episodes", type=int, default=None, help="fewer worlds in this chunk (smokes/subsamples)")
        s.add_argument("--worlds", choices=tuple(WORLD_BASES), default=None)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--threads", type=int, default=1)
        s.add_argument("--branch-batch", type=int, default=128)
        s.add_argument("--no-verify-hash", action="store_true")
        s.add_argument("--no-all-states", action="store_true", help="sampled points only (full-catalog subsample)")
        if name == "branch":
            s.add_argument("--option-set", choices=("anchored", "full"), default="anchored")
            s.add_argument("--option-class", choices=("single", "multi"), default="single",
                           help="multi: A-HR2 options (D + delegate); write to a separate --output directory")
    s = sub.add_parser("analyze")
    s.add_argument("--branch", type=Path, nargs="*")
    s.add_argument("--full", type=Path, nargs="*")
    s.add_argument("--states", type=Path, nargs="*")
    s.add_argument("--thr", type=float, default=0.01)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--inner", type=int, default=3)
    s.add_argument("--ridge", type=float, default=1.0)
    s.add_argument("--per-group-estimate", action="store_true")
    s.add_argument("--multi", type=Path, nargs="*", help="A-HR2 branch dirs (--option-class multi)")
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    a.binding, a.load_actor, a.solver = binding, load_actor_fn, solver
    if a.command == "analyze":
        return cmd_analyze(a)
    if a.base not in bases() and binding is None:
        raise SystemExit(f"unknown base {a.base!r}; one of {sorted(bases())}")
    loader, memo = (a.load_actor or load_actor), {}

    def cached(b, verify=True):   # one actor load (and hash check) per process
        if b["name"] not in memo:
            memo[b["name"]] = loader(b, verify=verify)
        return memo[b["name"]]
    for chunk in a.chunk:
        b = argparse.Namespace(**vars(a))
        b.chunk, b.load_actor = chunk, cached
        {"states": cmd_states, "branch": cmd_branch, "verify-d": cmd_verify_d}[a.command](b)


if __name__ == "__main__":
    main()
