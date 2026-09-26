"""extended-04 deployment procedures (campaign04_deploy), their evaluator wiring, the A1
config generator / analysis and the P2a validation tool (end to end on CPU)."""
from dataclasses import asdict
from functools import partial
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_training import TrainConfig, batched_episodes, sampling_rng_seed
from tensegra.campaign03_depworld import KINDS, DepWorkshop, depworld_executor, generate_depworld
from tensegra.campaign04_deploy import MODES, TrackedEnvironment, deploy_episodes, masked_log_probs
from tensegra.campaign04_progress import DECODE

ROOT = Path(__file__).resolve().parents[1]
DEV = 2_150_500_000  # extended-04 development seeds
EXEC = partial(depworld_executor, execute_call=execute)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"research/tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A1C = _load("campaign04_a1_configs")
A1A = _load("campaign04_a1_analysis")
VAL = _load("campaign04_progress_validate")


def make(seed, **kw):
    return DepWorkshop(generate_depworld(seed, foreign_records=2, p_event=0.5, **kw), executor=EXEC, address_seed=seed)


def model(seed=0):
    torch.manual_seed(seed)
    return CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1")).eval()


WALL = ("neural_forward_wall_seconds_allocated",)


def _strip_row(row):
    row = {k: v for k, v in row.items() if k not in ("timing", "progress", "deployment")}
    row["trace"] = [{k: v for k, v in t.items() if k not in WALL} for t in row["trace"]]
    row["outcome"] = {k: v for k, v in row["outcome"].items() if k != "solver_cpu_seconds"}
    return row


def test_greedy_and_sampled_are_bit_identical_to_batched_episodes():
    m = model()
    seeds = [DEV + i for i in range(6)]
    with torch.no_grad():
        for mode in ("greedy", "sampled"):
            samplers = lambda: None if mode == "greedy" else [random.Random(sampling_rng_seed(s, 5)) for s in seeds]
            kw = {} if mode == "greedy" else {"samplers": samplers()}
            a = batched_episodes(m, [make(s) for s in seeds], max_steps=40, **kw)
            b = deploy_episodes(m, [make(s) for s in seeds], mode=mode, max_steps=40, samplers=samplers())
            assert [_strip_row(x) for x in a] == [_strip_row(x) for x in b]
            assert all(len(x["progress"]["classes"]) == len(x["trace"]) for x in b)
            assert all(x["deployment"]["mode"] == mode and x["progress"]["interventions"] == [] for x in b)


class ThinkLover(torch.nn.Module):
    """Deterministic stand-in actor: scores think >> inspect > everything else."""

    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"feature_version": "d1", "family": "lightweight"})()

    def score(self, obs, candidates, hidden=None, mask=None):
        logits = 10 * candidates[..., KINDS.index("think")] + 5 * candidates[..., KINDS.index("inspect")]
        logits = logits - 0.01 * torch.arange(candidates.shape[1])  # strict order among ties
        return logits.masked_fill(~mask, float("-inf")), torch.zeros(obs.shape[0]), None


def test_r_mask_escapes_an_idempotent_loop_and_releases_masks():
    seeds = [DEV + 50 + i for i in range(3)]
    greedy = deploy_episodes(ThinkLover(), [make(s) for s in seeds], mode="greedy", max_steps=30)
    assert all(set(t["action"]["kind"] for t in r["trace"]) == {"think"} for r in greedy)
    assert all(r["progress"]["classes"] == "D" * 30 for r in greedy)
    masked = deploy_episodes(ThinkLover(), [make(s) for s in seeds], mode="r_mask", max_steps=30)
    for r in masked:
        kinds = [t["action"]["kind"] for t in r["trace"]]
        classes = r["progress"]["classes"]
        assert kinds[0] == "think" and kinds[1] == "inspect"  # think flagged at s0, masked at s0
        # every think is followed by a different action: a flagged think is never repeated at its state
        assert all(not (a == b == "think") for a, b in zip(kinds, kinds[1:]))
        assert "I" in classes and r["progress"]["interventions"]
        assert all(kind == "mask" and n >= 1 for _, kind, n, _ in r["progress"]["interventions"])
        assert any(ch for *_, ch in r["progress"]["interventions"])
        # the mask is released on every relevant change: think is chosen again at each new state
        assert kinds.count("think") > 3
        assert r["outcome"]["utility"] != greedy[0]["outcome"]["utility"]


def test_r_sample_samples_only_at_flagged_states():
    seeds = [DEV + 60 + i for i in range(3)]
    samplers = [random.Random(sampling_rng_seed(s, 3)) for s in seeds]
    rows = deploy_episodes(ThinkLover(), [make(s) for s in seeds], mode="r_sample", max_steps=30, samplers=samplers)
    for r in rows:
        steps = [i for i, *_ in r["progress"]["interventions"]]
        classes = r["progress"]["classes"]
        assert steps and steps[0] == 1  # step 0 is a think (not yet flagged); step 1 is at a flagged state
        # here every flagged state is reached by an idempotent step (it stays at the flagged state):
        # a sample happens exactly at the decisions that follow one
        assert steps == [i for i in range(1, len(classes)) if classes[i - 1] == "D"]
        assert all(kind == "sample" for _, kind, _, _ in r["progress"]["interventions"])
        assert any(t["action"]["kind"] != "think" for t in r["trace"])


def test_masked_sampled_never_chooses_masked_actions_and_logs_masked_logprob():
    seeds = [DEV + 70 + i for i in range(3)]
    m = model(1)
    samplers = [random.Random(sampling_rng_seed(s, 4)) for s in seeds]
    with torch.no_grad():
        rows = deploy_episodes(m, [make(s) for s in seeds], mode="masked_sampled", max_steps=40, samplers=samplers)
    for r in rows:
        for t in r["trace"]:
            assert t["masked_log_probability"] <= 1e-12
            assert t["masked_log_probability"] >= math.log(t["probability"]) - 1e-5  # renormalized >= raw
    # log pi_M for training
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    masked = torch.tensor([[False, False, True], [True, False, False]])
    lp = masked_log_probs(logits, masked)
    assert torch.isinf(lp[0, 2]) and torch.isinf(lp[1, 0])
    assert torch.allclose(lp[0, :2].exp().sum(), torch.tensor(1.0)) and torch.allclose(lp[1, 1:].exp(), torch.tensor([.5, .5]))


def test_masked_sampled_logprob_matches_renormalized_distribution():
    seeds = [DEV + 80]
    rows = deploy_episodes(ThinkLover(), [make(s) for s in seeds], mode="masked_sampled", max_steps=12,
                           samplers=[random.Random(1)])
    trace = rows[0]["trace"]
    assert all(t["masked_log_probability"] <= 1e-12 for t in trace)
    kinds = [t["action"]["kind"] for t in trace]
    assert kinds[0] == "think" and kinds[1] != "think"  # think masked after its first no-progress step


def test_tracked_environment_is_transparent_for_references():
    from tensegra.campaign02_references import make_reference, run_episode
    for i in range(5):
        a = run_episode(make(DEV + 90 + i), make_reference("dep_reuse"))
        tracked = TrackedEnvironment(make(DEV + 90 + i))
        b = run_episode(tracked, make_reference("dep_reuse"))
        strip = lambda o: {k: v for k, v in o.items() if k not in ("solver_cpu_seconds", "controller_cpu_seconds",
                                                                    "episode_cpu_seconds", "episode_wall_seconds")}
        assert strip(a) == strip(b)
        assert tracked.tracker.summary()["no_progress"] == 0


# ---------------------------------------------------------------------------
# A1 configs
# ---------------------------------------------------------------------------

def test_a1_config_structure_seeds_and_policies():
    cps = A1C.policies()
    assert [c["name"] for c in cps] == ["p1-boot-x1-r0", "p1-boot-x1-r1", "p1-boot-x1-r2", "p1-rl-x1-r0",
                                        "p1-rl-x1-r1", "p1-rl-x1-r2", "p2a-c1-final-x1-r2"]
    banks = json.loads((ROOT / "configs/campaign03/p1-banks.json").read_text())
    assert [c["sha256"] for c in cps[:3]] == [banks[f"x1-r{r}"]["sha256"] for r in range(3)]
    assert cps[-1]["sha256"].startswith("c3bb1bb1") and cps[5]["sha256"].startswith("d320a7ca")
    cfg = A1C.config(cps)
    assert len(cfg["conditions"]) == 16 and cfg["solver_cache"] and cfg["progress_diagnostic"]
    sealed = dict(_load("campaign03_p1_configs").SEALED)
    starts = {}
    for c in cfg["conditions"]:
        assert c["name"] == f"{c['base_condition']}-{c['mode']}" and c["policy_mode"] == c["mode"]
        assert c["world"] == sealed[c["base_condition"]] and c["examples"] == 256
        starts.setdefault(c["base_condition"], set()).add(c["seed_start"])
        assert (c.get("sampling_seed") == A1C.SAMPLING_SEED) == (c["mode"] in ("sampled", "r_sample"))
        assert (c["references"] == ["dep_reuse", "dep_recompute", "dep_greedy"]) == (c["mode"] == "greedy")
    assert {k: v.pop() for k, v in starts.items()} == {"iid_f0": 130_000_000, "iid_f2": 130_100_000,
                                                       "events_train_kinds_p1": 130_200_000, "foreign4": 130_300_000}
    ranges = A1C.check_fresh()
    used = [(110_000_000 + 100_000 * i, 110_000_000 + 100_000 * i + 256) for i in range(8)]
    used += [(120_000_000 + 100_000 * i, 120_000_000 + 100_000 * i + 256) for i in range(4)]
    used += [(1_990_000_000, 1_990_200_000), (2_000_000_000, 2_100_000_000), (3_000_000_000, 4_000_000_000)]
    for lo, hi in ranges.values():
        assert all(not (lo < b and a < hi) for a, b in used)
    with pytest.raises(ValueError):
        A1C.check_fresh(base=120_000_000)
    split = A1C.split_configs(cps)
    assert len(split) == 8 and split["a1-references"]["checkpoints"] == []
    assert all(c["references"] == [] for n, s in split.items() if n != "a1-references" for c in s["conditions"])
    assert {c["seed_start"] for s in split.values() for c in s["conditions"]} == set(A1C.seed_ranges()[c][0]
                                                                                     for c in A1C.CONDITIONS)
    with pytest.raises(ValueError):
        A1C.config(cps + [cps[0]])


# ---------------------------------------------------------------------------
# End to end: evaluator wiring, analysis, replay validation
# ---------------------------------------------------------------------------

def _checkpoints(tmp_path, n=2):
    out = []
    for i in range(n):
        torch.manual_seed(10 + i)
        m = CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1"))
        ck = tmp_path / f"policy-{i}.pt"
        torch.save({"config": asdict(TrainConfig(width=8, max_steps=24)), "policy_config": asdict(m.config),
                    "model": m.state_dict(), "updates": 0, "presentations": 0}, ck)
        out.append({"name": f"policy-{i}", "path": str(ck), "sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
                    "family": "bootstrap", "lineage": i})
    return out


def _evaluate(cfg, path, out):
    path.write_text(json.dumps(cfg))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, str(ROOT / "research/tools/campaign02_evaluate.py"), str(path), "--output",
                    str(out), "--device", "cpu"], check=True, env=env, timeout=1200)
    return json.loads((out / "summary.json").read_text())


def _read(path):
    return [json.loads(line) for line in gzip.open(path, "rt")]


REF_TIMING = ("controller_cpu_seconds", "episode_cpu_seconds", "episode_wall_seconds", "solver_cpu_seconds")


def _comparable(row):
    row = {k: v for k, v in row.items() if k not in ("timing", "progress", "deployment")}
    row["outcome"] = {k: v for k, v in row["outcome"].items() if k not in REF_TIMING}
    row["trace"] = [{k: v for k, v in t.items() if k not in WALL} for t in row.get("trace", [])]
    return row


def test_a1_end_to_end_evaluate_analysis_and_replay_validation(tmp_path):
    cps = _checkpoints(tmp_path)
    cfg = A1C.config(cps, examples=2, base=DEV + 1000)
    cfg["conditions"] = [c for c in cfg["conditions"] if c["base_condition"] in ("iid_f0", "iid_f2")]
    summary = _evaluate(cfg, tmp_path / "a1.json", tmp_path / "a1")
    assert set(summary["solver_cache"]) == {c["name"] for c in cfg["conditions"]}
    assert "campaign04_progress" in summary["sources"]
    modes = {(r["condition"], r.get("policy_mode", "greedy")) for r in summary["results"] if r["kind"] == "learned"}
    assert {m for _, m in modes} == set(A1C.MODES)
    for result in summary["results"]:
        rows = _read(tmp_path / "a1" / result["artifact"])
        assert all(len(r["progress"]["classes"]) == len(r["outcome"]["history"]) for r in rows)
        if result["kind"] == "learned":
            assert all("observation" not in t for r in rows for t in r["trace"])  # trace_observations off
            assert all(r["deployment"]["mode"] == result.get("policy_mode", "greedy") for r in rows)
    # diagnostic off: greedy / sampled rows are identical to the diagnostic-on rows (it only observes);
    # without the solver cache the histories are identical too (deterministic calls)
    plain = dict(cfg, progress_diagnostic=False, solver_cache=False)
    plain["conditions"] = [dict(c, progress_diagnostic=False) for c in cfg["conditions"]
                           if c["mode"] in ("greedy", "sampled")]
    summary2 = _evaluate(plain, tmp_path / "plain.json", tmp_path / "plain")
    assert "campaign04_progress" not in summary2["sources"] and "solver_cache" not in summary2
    for result in summary2["results"]:
        a = [_comparable(r) for r in _read(tmp_path / "plain" / result["artifact"])]
        b = [_comparable(r) for r in _read(tmp_path / "a1" / result["artifact"])]
        assert a == b, result["artifact"]
        assert all("progress" not in r for r in _read(tmp_path / "plain" / result["artifact"]))
    # A1 analysis
    r = A1A.analyze([tmp_path / "a1"])
    assert "policy-0/greedy/iid_f0" in r["table"] and "dep_reuse/reference/iid_f2" in r["table"]
    assert r["table"]["dep_reuse/reference/iid_f0"]["no_progress_rate"] == 0
    assert r["table"]["dep_recompute/reference/iid_f0"]["no_progress_rate"] == 0
    cell = r["table"]["policy-1/r_mask/iid_f0"]
    assert cell["diagnostic_coverage"] == 1.0 and "paired_vs_greedy" in cell
    assert set(cell["paired_vs_greedy"]["utility"]["worlds_needed"]) == {"0.01", "0.02", "0.05"}
    assert "p2a_no_progress_rate" in cell and "cycle_length_distribution" in cell
    assert r["iid_groups"]["policy-0/sampled"]["success"] is not None
    assert "bootstrap/r_mask/iid_group" in r["families"]
    assert A1A.markdown(r).startswith("# A1 deployment matrix")
    # replay validation: the replayed feedback equals the recorded feedback, and the offline
    # tracker reproduces the online classes exactly
    v = VAL.validate(tmp_path / "a1", None, limit=None, log=lambda *_: None)
    assert v["replay_feedback_mismatches"] == 0
    assert v["online_replay_class_mismatches"] == 0 and v["online_rows_checked"] > 0
    assert all(d["v1_no_progress"] == 0 for d in v["references"].values())


def test_evaluator_rejects_bad_modes(tmp_path):
    cps = _checkpoints(tmp_path, 1)
    cfg = A1C.config(cps, examples=1, base=DEV + 2000)
    cfg["conditions"] = [dict(cfg["conditions"][1])]
    cfg["conditions"][0].pop("sampling_seed")
    with pytest.raises(subprocess.CalledProcessError):
        _evaluate(cfg, tmp_path / "bad.json", tmp_path / "bad")
    assert set(MODES) >= set(A1C.MODES)
    assert DECODE["C"] == "short_cycle"
