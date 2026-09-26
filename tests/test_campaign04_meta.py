"""extended-04 Track C: telemetry, the metacontrol runner, counterfactual labels and
appraisal training (campaign04_telemetry / campaign04_meta / campaign04_meta_train)."""
from dataclasses import replace
from functools import partial
import importlib.util
import math
from pathlib import Path
import random

import pytest
import torch

from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_training import sampling_rng_seed
from tensegra.campaign03_depworld import Action, DepWorkshop, depworld_executor, generate_depworld
from tensegra.campaign04_branch import SolverCache
from tensegra.campaign04_deploy import deploy_episodes
from tensegra.campaign04_meta import (ADV_US, Episode, FixedAgent, LabelConfig, MetaAgent, MetaSpec, generate_labels,
                                      make_model, meta_units_per_forward, new_episode, outcome, run)
from tensegra.campaign04_meta_train import (Tensors, dev_metrics, merge, pack, register, split, train)
from tensegra.campaign04_progress import ProgressTracker
from tensegra.campaign04_telemetry import (DIM, GROUPS, INDEX, LAYOUT, TelemetryRecorder, group_indices, group_mask,
                                           policy_stats)

ROOT = Path(__file__).resolve().parents[1]
DEV = 2_170_000_000  # Track C development/test seeds (disjoint from every registered range)
EXEC = partial(depworld_executor, execute_call=execute)


def make(seed, executor=EXEC, **kw):
    return DepWorkshop(generate_depworld(seed, foreign_records=2, p_event=0.5, **kw), executor=executor,
                       address_seed=seed)


def actor(seed=0):
    torch.manual_seed(seed)
    return CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1")).eval()


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"research/tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- telemetry -----------------------------------------------------------------

def test_layout_groups_and_masks():
    assert DIM == len(LAYOUT) == len(INDEX)
    assert set(GROUPS) == {"policy", "budget", "cost", "diagnostic", "feedback", "info_events", "stage", "control"}
    assert sorted(i for g in GROUPS for i in group_indices(g)) == list(range(DIM))
    m = group_mask(("policy",))
    assert all(m[i] == 0.0 for i in group_indices("policy")) and sum(m) == DIM - len(group_indices("policy"))
    with pytest.raises(ValueError):
        group_indices("hidden_spec")


def _features_along(env, actions, stats):
    rec = TelemetryRecorder(env.observe())
    tracker = ProgressTracker(env.observe())
    out = []
    o = env.observe()
    for a in actions:
        out.append((o.to_dict(), rec.features(o, tracker, stats).values))
        env.charge_compute(1.0)
        after = env.step(a)
        rec.update(after, a, tracker.update(after, a), "default", 1.0)
        o = after
    return out


def test_telemetry_is_public_only_and_identical_for_identical_visible_histories():
    spec = generate_depworld(DEV + 1, foreign_records=2, p_event=0.5)
    other = replace(spec, funds=spec.funds + 7, capacity=spec.capacity + 3)   # hidden until inspected
    assert other != spec
    item = spec.items[0].handle
    actions = [Action("think"), Action("inspect", {"target": item}), Action("inspect", {"target": "map"}),
               Action("think"), Action("verify")]
    stats = policy_stats([0.5, 0.3, 0.2], 0, 0, 0, False)
    a = _features_along(DepWorkshop(spec, executor=EXEC, address_seed=5), actions, stats)
    b = _features_along(DepWorkshop(other, executor=EXEC, address_seed=5), actions, stats)
    assert [x[0] for x in a] == [x[0] for x in b]          # the visible histories are identical ...
    assert [x[1] for x in a] == [x[1] for x in b]          # ... so the telemetry is identical
    # once the difference becomes visible, the observation differs (sanity of the construction)
    ea, eb = DepWorkshop(spec, executor=EXEC), DepWorkshop(other, executor=EXEC)
    assert ea.step(Action("inspect", {"target": "requirements"})).requirements != \
        eb.step(Action("inspect", {"target": "requirements"})).requirements
    # the recorder holds only public numbers (no environment, spec or executor reference)
    rec = TelemetryRecorder(DepWorkshop(spec, executor=EXEC).observe())
    assert all(isinstance(v, (int, float, bool, str, dict, type(None))) for v in vars(rec).values())


def test_public_cost_reconstructs_environment_cost():
    m = actor(1)
    eps = [new_episode(make(DEV + 10 + i), rng=random.Random(i), aux_rng=random.Random(9 + i)) for i in range(4)]
    seen = []

    def hook(ep, d, u):
        seen.append((ep.recorder.public_cost(d.observation), -ep.env.current_utility()))
    run(m, eps, MetaAgent(mode="random", rate=0.3, u_probs={"sample": .5, "mask_top": .5}), hook=hook)
    assert len(seen) > 20
    assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in seen)


# --- runner equivalence ------------------------------------------------------------

def _summary(row):
    return [t["action_index"] for t in row["trace"]], row["outcome"]["utility"], row["progress"]


@pytest.mark.parametrize("mode", ["greedy", "sampled", "r_mask", "r_sample"])
def test_fixed_modes_identical_to_deploy_episodes(mode):
    m = actor(2)
    seeds = [DEV + 20 + i for i in range(5)]
    samplers = lambda: [random.Random(sampling_rng_seed(s, 7)) for s in seeds]
    ref = deploy_episodes(m, [make(s) for s in seeds], mode=mode, max_steps=40,
                          samplers=samplers() if mode in ("sampled", "r_sample") else None)
    eps = [new_episode(make(s), cap=40, rng=r, telemetry=False) for s, r in zip(seeds, samplers())]
    with torch.no_grad():
        run(m, eps, FixedAgent(mode))
    for r, ep in zip(ref, eps):
        actions, utility, progress = _summary(r)
        assert [x["a"] for x in ep.log] == actions
        assert outcome(ep.env)["utility"] == utility
        assert ep.tracker.summary() == progress


def test_controller_default_path_is_greedy_rmask_exactly_and_meta_charge_is_applied():
    m = actor(3)
    seeds = [DEV + 30 + i for i in range(5)]
    ref = deploy_episodes(m, [make(s) for s in seeds], mode="r_mask", max_steps=40)
    meta = make_model(MetaSpec())
    units = meta_units_per_forward(meta, m)
    assert units > 0
    for mode, kw in (("appraisal_only", {}), ("learned", {"margin": math.inf})):
        for charge in (False, True):
            eps = [new_episode(make(s), cap=40, rng=random.Random(0)) for s in seeds]
            run(m, eps, MetaAgent(meta, mode, **kw), meta_units=units, charge_meta=charge)
            for r, ep in zip(ref, eps):
                actions, utility, progress = _summary(r)
                assert [x["a"] for x in ep.log] == actions and all(x["u"] == "default" for x in ep.log)
                assert ep.tracker.summary() == progress
                o = outcome(ep.env)
                assert ep.meta_forwards == len(actions)
                extra = units * ep.meta_forwards if charge else 0.0
                assert math.isclose(o["compute_units"], r["outcome"]["compute_units"] + extra, abs_tol=1e-9)
                assert math.isclose(o["utility"], utility - extra * ep.env._spec.compute_price, abs_tol=1e-12)
                assert all("pred" in x for x in ep.log)


def test_interventions_mask_top_and_stop():
    m = actor(4)
    eps = [new_episode(make(DEV + 40 + i), cap=40, rng=random.Random(i)) for i in range(3)]
    picks = []

    class Agent(MetaAgent):
        def decide(self, decisions, episodes):
            us = ["mask_top" if episodes[d.index].taken == 0 else "stop" for d in decisions]
            picks.extend((d.default, d.mask_top(), d.actions[d.stop()].kind) for d in decisions)
            return us
    run(m, eps, Agent(mode="random"))
    for ep in eps:
        assert [x["u"] for x in ep.log] == ["mask_top", "stop"]
        assert ep.env.observe().done and not outcome(ep.env)["success"]
    for default, top, stop in picks:
        assert top != default and stop == "abstain"


# --- counterfactual labels --------------------------------------------------------

def test_branch_default_reproduces_main_line_and_costs_are_charged_once():
    m = actor(5)
    seeds = [DEV + 50 + i for i in range(6)]
    cache = SolverCache(EXEC)
    cfg = LabelConfig(base_mix=(("d0", 1.0),), check_fraction=1.0, p_flagged=1.0, p_low_margin=1.0, p_other=0.5,
                      max_points=4, max_steps=40)
    eps, points, stats = generate_labels(m, [make(s, cache) for s in seeds], seeds, cfg)
    assert stats["default_checks"] == len(points) > 5
    assert stats["default_check_matches"] == stats["default_checks"]   # clone+D0 == main line, bit for bit
    for p in points:
        ep = eps[p["episode"]]
        assert p["dU"]["default"] == ep["final"]["utility"] - ep["U"][p["step"]]
        # stop: abstain ends the episode; ΔU is exactly the step's action price plus one actor forward
        spec = generate_depworld(ep["seed"], foreign_records=2, p_event=0.5)
        assert math.isclose(p["dU"]["stop"], -(spec.action_price + spec.compute_price), abs_tol=1e-12)
        assert len(p["dU_sample"]) == cfg.k_draws and all(x is not None for x in p["dU_sample"])
    # main-line accounting: exactly one actor forward charged per decision
    for e in eps:
        assert math.isclose(e["final"]["compute_units"], len(e["u"]), abs_tol=1e-9)


def test_labels_for_every_base_kind_and_packing():
    m = actor(6)
    seeds = [DEV + 60 + i for i in range(8)]
    cfg = LabelConfig(p_flagged=1.0, p_low_margin=0.5, p_other=0.2, max_steps=24)
    eps, points, stats = generate_labels(m, [make(s) for s in seeds], seeds, cfg)
    kinds = {e["base_kind"] for e in eps}
    assert kinds <= {"d0", "eps", "sampled"} and len(kinds) >= 2
    for p in points:
        assert p["dU"]["default"] is not None and p["success_default"] is not None
        assert all(x is not None for x in p["dU_sample"])
    data = pack(eps, points, {"chunk": 0})
    data2 = merge([data, pack(eps, points, {"chunk": 1})])
    assert len(data2["ep_seed"]) == 2 * len(eps) and int(data2["pt_episode"].max()) >= len(eps)
    assert data["feats"].shape == (sum(len(e["u"]) for e in eps), DIM)


def test_label_targets_are_never_inputs():
    m = actor(7)
    seeds = [DEV + 80 + i for i in range(5)]
    eps, points, _ = generate_labels(m, [make(s) for s in seeds], seeds, LabelConfig(max_steps=30, p_other=0.5))
    data = pack(eps, points)
    tensors = Tensors(data)
    model = make_model(MetaSpec())
    idx = list(range(len(eps)))
    before = model.sequence(tensors.x[torch.tensor(idx)])
    corrupted = dict(data)
    for key in ("ep_utility", "ep_cost", "U", "pt_dU", "pt_dU_sample", "pt_cost_d"):
        corrupted[key] = data[key] + 123.0
    corrupted["ep_success"] = ~data["ep_success"]
    t2 = Tensors(corrupted)
    assert torch.equal(t2.x, tensors.x)
    after = model.sequence(t2.x[torch.tensor(idx)])
    assert all(torch.equal(before[k], after[k]) for k in before)
    # the shuffled-target control permutes targets and leaves inputs untouched
    t3 = Tensors(data)
    t3.shuffle_targets(idx, random.Random(0))
    assert torch.equal(t3.x, tensors.x)


def test_training_registration_and_group_drop():
    m = actor(8)
    seeds = [DEV + 100 + i for i in range(10)]
    eps, points, _ = generate_labels(m, [make(s) for s in seeds], seeds, LabelConfig(max_steps=30, p_other=0.5))
    data = pack(eps, points)
    tensors = Tensors(data)
    tr, dev = split(data, 0.3)
    assert tr and dev and not set(tr) & set(dev)
    model, hist = train(tensors, tr, MetaSpec(drop_groups=("policy",)), seed=0, epochs=3)
    assert len(hist) == 3 and all(math.isfinite(h) for h in hist)
    assert float(model.mask[group_indices("policy")].sum()) == 0.0
    metrics, rows = dev_metrics(model, tensors, data, dev)
    reg = register(rows)
    assert reg["margin"] in [r["margin"] for r in reg["margin_table"]]
    assert 0.0 <= reg["tau"] <= 1.0
    n = sum(p.numel() for p in model.parameters())
    assert n < 40_000 and model.spec.hidden <= 64


# --- tools (labels -> train -> eval -> analysis) with a tiny stand-in actor --------------

def test_track_c_tools_end_to_end(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "research/tools"))
    import campaign04_c_configs as C
    import campaign04_c_labels as L
    import campaign04_c_train as T
    import campaign04_c_eval as V
    import campaign04_c_analysis as A
    assert C.check_fresh()  # registered ranges are disjoint from every earlier range
    tiny = actor(11)
    binding = {"name": "tiny", "sha256": "0" * 64, "base_type": "loop_prone", "f10_stratum": "loop_prone",
               "path": "unused"}
    monkeypatch.setattr(C, "base", lambda name: binding)
    monkeypatch.setattr(C, "load_actor", lambda b, verify=True: (
        tiny, SimpleNamespace(max_steps=30, neural_work_per_forward=1.0),
        {"parameters": sum(p.numel() for p in tiny.parameters()), "sha256": b["sha256"]}))
    monkeypatch.setattr(C, "LABEL_SEED_BASE", DEV + 500_000)
    monkeypatch.setattr(C, "EVAL_SEED_BASE", DEV + 600_000)
    monkeypatch.setattr(C, "check_fresh", lambda *a, **k: None)
    labels = tmp_path / "labels"
    L.main(["--base", "tiny", "--chunk", "0", "1", "--chunk-size", "6", "--output", str(labels)])
    with pytest.raises(SystemExit):   # never overwrites
        L.main(["--base", "tiny", "--chunk", "0", "--chunk-size", "6", "--output", str(labels)])
    info = json.loads((labels / "labels-c000.json").read_text())
    assert info["stats"]["episodes"] == 6 and "oracle" in info["privilege"]
    models = tmp_path / "models"
    T.main(["--labels", str(labels), "--base", "tiny", "--variants", "main", "shuffled", "drop:policy",
            "--seeds", "0", "--epochs", "2", "--dev-fraction", "0.34", "--output", str(models)])
    model, record = T.load_model(models / "drop-policy-s0.pt")
    assert tuple(record["spec"]["drop_groups"]) == ("policy",)
    out = tmp_path / "eval"
    V.main(["--base", "tiny", "--models", str(models), "--seeds", "0", "--conditions", "iid_f0", "iid_f2",
            "--arms", "fixed", "core", "clamp:stop", "zero:policy", "shuffle_telemetry", "model:shuffled",
            "model:drop-policy", "--examples", "4", "--batch", "3", "--branch-eval", "2",
            "--single-seed-arms", "appraisal_only", "causal", "retrained", "--output", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    by = {(r["condition"], r["arm"]): r for r in summary["results"]}
    assert ("iid_f0", "fixed:r_mask") in by and ("iid_f2", "random_matched") in by
    # the appraisal-only arm behaves exactly like D0 (= fixed r_mask) apart from the metacontrol charge
    for c in ("iid_f0", "iid_f2"):
        assert by[(c, "appraisal_only")]["success"] == by[(c, "fixed:r_mask")]["success"]
        assert math.isclose(by[(c, "appraisal_only")]["utility_no_meta_charge"], by[(c, "fixed:r_mask")]["utility"],
                            abs_tol=1e-12)
        assert by[(c, "appraisal_only")]["utility"] < by[(c, "fixed:r_mask")]["utility"]
        assert "calibration_steps" in by[(c, "appraisal_only")]
    report = tmp_path / "analysis.json"
    A.main(["--eval", str(out), "--train", str(models), "--labels", str(labels), "--seeds", "0",
            "--output", str(report)])
    rep = json.loads(report.read_text())
    assert rep["bases"]["tiny"]["criterion"] is not None
    assert rep["labels"]["tiny"]["chunks"] == 2
