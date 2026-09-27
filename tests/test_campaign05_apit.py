"""extended-05 A-PI-T delegation controller (campaign05_apit, research/tools/campaign05_apit.py)."""
from dataclasses import replace
from functools import partial
import gzip
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_references import make_reference, run_episode
from tensegra.campaign03_depworld import Action, DepWorkshop, depworld_executor, generate_depworld
from tensegra.campaign04_deploy import deploy_episodes
from tensegra.campaign05_apit import (DELEGATE_KEYS, FEATURE_KEYS, FEATURE_VERSION, TRIGGERS, Controller, Delegator,
                                      RegretHook, canonical_hash, decision_features, episode_row, feature_names,
                                      feature_vector, run_delegator, run_regret_branches)
from tensegra.campaign05_options import (HRConfig, d_choice, hr_labels, new_ep, outcome, run_policy, teacher_index,
                                         make_teacher)

ROOT = Path(__file__).resolve().parents[1]
DEV = 2_250_600_000  # A-PI-T tests: inside the extended-05 development entry (non-protocol)
EXEC = partial(depworld_executor, execute_call=execute)


def make(seed, **kw):
    return DepWorkshop(generate_depworld(seed, foreign_records=2, p_event=0.5, **kw), executor=EXEC, address_seed=seed)


def actor(seed=0):
    torch.manual_seed(seed)
    return CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1")).eval()


def _load(name):
    import sys
    sys.path.insert(0, str(ROOT / "research/tools"))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"research/tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact(weights=None, margin=0.0, seed=0):
    names = feature_names()
    rng = np.random.default_rng(seed)
    models = {a: {"kind": "ridge", "lam": 1.0,
                  "w": (weights[a] if weights else (rng.normal(0, .01, len(names))).tolist())} for a in TRIGGERS}
    c = {"version": "test", "feature_version": FEATURE_VERSION, "feature_names": names, "triggers": list(TRIGGERS),
         "family": {"name": "ridge_l1", "kind": "ridge", "lam": 1.0}, "margin": margin, "models": models}
    return {"controller": c, "controller_sha256": canonical_hash(c), "protocol": False}


class Recording:
    """Controller stand-in recording (anchor, x) and firing by rule(call index)."""

    def __init__(self, fire_at=(), margin=0.0):
        self.calls, self.fire_at, self.margin, self.n_parameters = [], set(fire_at), margin, 10

    def predict(self, anchor, x):
        self.calls.append((anchor, list(x)))
        return 1.0 if len(self.calls) - 1 in self.fire_at else -1.0


# --- feature definition -----------------------------------------------------------------------

def test_feature_keys_equal_the_a_hr2_g1b_estimator():
    hr = _load("campaign05_hr")
    assert FEATURE_KEYS == hr.FEATURE_KEYS and DELEGATE_KEYS == hr.DELEGATE_KEYS
    tele = [0.1 * i for i in range(89)]
    f = {"prob": .3, "rel_usable": 1.0, "teacher_first_call": 1.0, "anchor_call": 1.0, "is_d": 0.0, "junk": 5.0}
    p, o = {"telemetry": tele}, {"type": "delegate", "features": f}
    assert np.array_equal(np.array(feature_vector(tele, f)), hr._x(p, o))
    assert len(feature_names()) == len(feature_vector(tele, f))


def test_controller_features_equal_training_features_on_d_states():
    """pi_T's feature vector at a D state == the A-HR2 training feature vector at the same state."""
    hr = _load("campaign05_hr")
    m = actor(5)
    seeds = [DEV + i for i in range(5)]
    cfg = HRConfig(max_steps=40, option_mode="multi", p_all_states=1.0)
    _, points, _ = hr_labels(m, [make(s) for s in seeds], seeds, cfg, branch_batch=64)
    train = {}
    for p in points:
        o = next((o for o in p["options"] if o["type"] == "delegate"), None)
        if o is not None and p["anchor"] in TRIGGERS:
            train[(p["seed"], p["step"])] = hr._x(p, o)
    assert len(train) >= 10
    # the same batch as the labelled main line (batch composition changes actor floats in the last bits);
    # every active episode is at the same step in a run_policy round, so calls are ordered by (step, episode)
    rec = Recording()
    eps = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("pit", controller=rec, record_predictions=True),
                        cap=40)
    order = sorted((step, i) for i, ep in enumerate(eps) for step, *_ in ep.info["apit"]["preds"])
    assert len(order) == len(rec.calls) == len(train)
    for (step, i), (anchor, x) in zip(order, rec.calls):
        key = (seeds[i], step)
        assert np.array_equal(np.array(x), train[key])


def test_controller_uses_public_features_only():
    """(a) decision_features reads only public decision fields; (b) worlds with different hidden values but
    the same visible history give identical controller inputs."""
    m = actor(6)
    seed = DEV + 20
    spec = generate_depworld(seed, foreign_records=2, p_event=0.5)
    other = replace(spec, funds=spec.funds + 7, capacity=spec.capacity + 3)
    envs = [DepWorkshop(s, executor=EXEC, address_seed=seed) for s in (spec, other)]
    for a in [Action("think"), Action("inspect", {"target": spec.items[0].handle}), Action("inspect", {"target": "map"})]:
        for e in envs:
            e.step(a)
    assert envs[0].observe().to_dict() == envs[1].observe().to_dict()
    decs = []
    for e in envs:
        run_policy(m, [new_ep(e, 1, telemetry=True)], d_choice, hook=lambda ep, d, c: decs.append(d))
    allowed = {"observation", "actions", "logits", "probabilities", "default", "telemetry"}

    class Guard:
        def __init__(self, d):
            object.__setattr__(self, "_d", d)

        def __getattr__(self, name):
            if name not in allowed:
                raise AssertionError(f"controller read non-public decision field {name}")
            return getattr(self._d, name)
    xs = []
    for d in decs:
        ti = teacher_index(make_teacher(), d.observation, d.actions)
        xs.append(decision_features(Guard(d), ti))
    assert xs[0] == xs[1]
    c = Controller(artifact())
    assert c.artifact["controller"]["feature_names"] == feature_names()
    assert not any("hidden" in n or "label" in n for n in feature_names())


# --- deployed policy ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2])
def test_pit_with_infinite_margin_is_bit_identical_to_d(seed):
    m = actor(seed)
    seeds = [DEV + 100 * seed + i for i in range(6)]
    ref = deploy_episodes(m, [make(s) for s in seeds], mode="r_mask", max_steps=96)
    d_eps = [new_ep(make(s), 96) for s in seeds]
    run_policy(m, d_eps, d_choice)
    ctrl = Controller(artifact(margin=0.0), margin=math.inf)
    pit = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("pit", controller=ctrl), cap=96)
    for r, de, pe in zip(ref, d_eps, pit):
        assert pe.actions_taken == de.actions_taken == [t["action_index"] for t in r["trace"]]
        assert outcome(pe.env) == outcome(de.env)
        assert pe.env.evaluate()["compute_units"] == r["outcome"]["compute_units"]
        assert pe.tracker.summary() == de.tracker.summary() == r["progress"]
        row = episode_row(pe, "pit")
        assert row["delegations"] == 0 and row["teacher_steps"] == 0
    assert sum(e.info["apit"]["controller_evals"] for e in pit) > 0     # the controller was evaluated
    # charged: same actions, utility lower by exactly evals x units x compute_price
    units = 0.25
    pit2 = run_delegator(m, [make(s) for s in seeds], seeds,
                         Delegator("pit", controller=ctrl, controller_units=units), cap=96)
    for de, pe in zip(d_eps, pit2):
        n = pe.info["apit"]["controller_evals"]
        assert pe.actions_taken == de.actions_taken
        assert pe.env.evaluate()["compute_units"] == pytest.approx(de.env.evaluate()["compute_units"] + n * units)


def test_d_policy_equals_d():
    m = actor(3)
    seeds = [DEV + 300 + i for i in range(4)]
    d_eps = [new_ep(make(s), 96) for s in seeds]
    run_policy(m, d_eps, d_choice)
    eps = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("d"), cap=96)
    assert [e.actions_taken for e in eps] == [e.actions_taken for e in d_eps]
    assert [outcome(e.env) for e in eps] == [outcome(e.env) for e in d_eps]


def test_always_teacher_equals_dep_reuse_reference_evaluation():
    m = actor(4)
    seeds = [DEV + 400 + i for i in range(6)]
    eps = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("always"), cap=96)
    for s, ep in zip(seeds, eps):
        ref = run_episode(make(s), make_reference("dep_reuse"), model_compute_tariff=1.0)
        got = ep.env.evaluate()
        assert ep.info["apit"]["delegation_out_of_catalog"] == 0
        assert got["utility"] == ref["utility"] and got["verified_success"] == ref["verified_success"]
        assert got["compute_units"] == ref["compute_units"] == len(ref["trace"])
        assert got["steps"] == ref["steps"] and got["work_units"] == ref["work_units"] and got["cost"] == ref["cost"]
        row = episode_row(ep, "always")
        assert row["teacher_steps"] == row["decisions"] == len(ref["trace"])


def test_charges_applied_once_and_delegation_semantics():
    m = actor(7)
    seeds = [DEV + 500 + i for i in range(6)]
    units = 0.125
    ctrl = Controller(artifact(), margin=-math.inf)          # delegate at every eligible decision
    eps = run_delegator(m, [make(s) for s in seeds], seeds,
                        Delegator("pit", controller=ctrl, controller_units=units, record_predictions=True), cap=96)
    total_delegations = 0
    for ep in eps:
        st = ep.info["apit"]
        ev = ep.env.evaluate()
        assert ev["compute_units"] == pytest.approx(ep.taken * 1.0 + st["controller_evals"] * units)
        assert ev["steps"] == ep.taken
        assert st["controller_evals"] == st["eligible"] == st["starts"]
        row = episode_row(ep, "pit")
        assert sum(row["delegation_ends"].values()) == row["delegations"]
        assert set(row["delegation_ends"]) <= {"commit", "max_steps", "out_of_catalog", "episode_end", "cap"}
        assert row["teacher_steps"] <= 12 * row["delegations"]
        total_delegations += row["delegations"]
    assert total_delegations > 0


def test_single_delegation_equals_option_t_branch():
    """A controller firing once at an A-HR2 point reproduces that point's delegate label (option T semantics)."""
    m = actor(8)
    seeds = [DEV + 600 + i for i in range(4)]
    cfg = HRConfig(max_steps=40, option_mode="multi", p_all_states=1.0)
    _, points, _ = hr_labels(m, [make(s) for s in seeds], seeds, cfg, branch_batch=64)
    probe = Recording()
    base = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("pit", controller=probe,
                                                                         record_predictions=True), cap=40)
    checked = 0
    for ep in base:
        s = ep.info["seed"]
        eligible_steps = [x[0] for x in ep.info["apit"]["preds"]]
        for p in points:
            if p["seed"] != s or p["step"] not in eligible_steps:
                continue
            o = next(o for o in p["options"] if o["type"] == "delegate")
            k = eligible_steps.index(p["step"])
            one = Recording(fire_at=[k])
            e2 = run_delegator(m, [make(s)], [s], Delegator("pit", controller=one), cap=40)[0]
            assert e2.env.evaluate()["utility"] == pytest.approx(p["U_t"] + o["dU"], abs=1e-12)
            assert e2.info["apit"]["starts"] == 1
            checked += 1
            if checked >= 8:
                return
    assert checked > 0


def test_random_rule_is_seeded_and_rules_trigger_on_their_anchor():
    m = actor(9)
    seeds = [DEV + 700 + i for i in range(5)]
    a = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("random", rate=.3, rng_key="b"), cap=60)
    b = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("random", rate=.3, rng_key="b"), cap=60)
    assert [e.actions_taken for e in a] == [e.actions_taken for e in b]
    for pol, anchor in (("r1", "call"), ("r2", "commit_revise")):
        eps = run_delegator(m, [make(s) for s in seeds], seeds, Delegator(pol), cap=60)
        for e in eps:
            st = e.info["apit"]
            assert set(st["eligible_by_anchor"]) <= {anchor} and st["starts"] == st["eligible"]


def test_regret_branches_deterministic():
    m = actor(10)
    seeds = [DEV + 800 + i for i in range(4)]
    dl = Delegator("pit", controller=Controller(artifact(seed=3), margin=0.0), controller_units=0.01)
    hook = RegretHook(1.0, max_points=4)
    run_delegator(m, [make(s) for s in seeds], seeds, dl, cap=40, hook=hook)
    assert hook.points
    rows = run_regret_branches(m, dl, hook.points, check=True)
    assert all(r["check_equal"] for r in rows)
    assert all(r["regret"] >= 0 for r in rows)


# --- tool ---------------------------------------------------------------------------------------

def test_seed_ranges_disjoint_and_registered(tmp_path):
    T = _load("campaign05_apit")
    rs = T.check_seed_ranges()
    ap = next(r for r in rs if "A-PI-T" in r["name"])
    assert (ap["lo"], ap["hi"]) == (240_000_000, 240_400_000)
    tr = next(r for r in rs if r["lo"] == 220_000_000)
    assert ap["lo"] >= tr["hi"] or ap["hi"] <= tr["lo"]
    assert T.world_seeds("apit", "foreign4", 15, 32)[-1] == 240_300_511
    with pytest.raises(SystemExit):
        T.world_seeds("apit", "iid_f0", 16, 32)          # 512 per condition only
    bad = json.loads((ROOT / "research/campaigns/extended-05/seed-ranges.json").read_text())
    bad["ranges"].append({"name": "overlap", "campaign": "extended-05", "lo": 240_000_100, "hi": 240_000_200})
    p = tmp_path / "s.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        T.check_seed_ranges(p)


class _InProcessSolver:
    execute = staticmethod(execute)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_tool_end_to_end(tmp_path):
    H, T = _load("campaign05_hr"), _load("campaign05_apit")
    tiny = actor(11)
    binding = {"name": "tiny", "sha256": "0" * 64, "source": "test"}
    load = lambda b, verify=True: (tiny, SimpleNamespace(max_steps=30, neural_work_per_forward=1.0),  # noqa: E731
                                   {"parameters": 1000, "sha256": b["sha256"]})
    kw = dict(binding=binding, load_actor_fn=load, solver=_InProcessSolver)
    for cond in ("iid_f0", "iid_f2"):
        H.main(["branch", "--option-class", "multi", "--base", "tiny", "--condition", cond, "--worlds", "dev",
                "--chunk", "0", "1", "--chunk-size", "8", "--output", str(tmp_path / "multi")], **kw)
    ctrl = tmp_path / "controller.json"
    with pytest.raises(SystemExit):   # protocol training refuses dev files
        T.main(["train", "--branch", str(tmp_path / "multi"), "--output", str(ctrl)])
    T.main(["train", "--branch", str(tmp_path / "multi"), "--output", str(ctrl), "--dev", "--folds", "2"])
    art = json.loads(ctrl.read_text())
    assert art["controller_sha256"] == canonical_hash(art["controller"]) and not art["protocol"]
    assert art["selection"]["chosen"]["margin"] in T.M_GRID
    assert art["controller"]["family"]["name"] in [f["name"] for f in T.FAMILIES]
    with pytest.raises(SystemExit):   # never overwrites
        T.main(["train", "--branch", str(tmp_path / "multi"), "--output", str(ctrl), "--dev"])
    with pytest.raises(SystemExit):   # confirmation lineages refused
        T.main(["eval", "--base", "x1-r3", "--condition", "iid_f0", "--chunk", "0", "--controller", str(ctrl),
                "--output", str(tmp_path / "x")])
    with pytest.raises(SystemExit):   # protocol worlds need the protocol controller
        T.main(["eval", "--base", "tiny", "--condition", "iid_f0", "--chunk", "0", "--controller", str(ctrl),
                "--output", str(tmp_path / "x"), "--episodes", "2"], **kw)
    for cond in ("iid_f0", "iid_f2", "foreign4"):
        T.main(["eval", "--base", "tiny", "--condition", cond, "--worlds", "dev", "--chunk", "0", "--chunk-size", "6",
                "--controller", str(ctrl), "--teacher", "yes", "--regret-p", "0.5", "--regret-check",
                "--output", str(tmp_path / "eval")], **kw)
    meta = json.loads(next((tmp_path / "eval").glob("*iid_f0*.meta.json")).read_text())
    assert meta["seeds"] == [2_250_500_000, 2_250_500_006] and not meta["protocol_worlds"]
    assert set(meta["summary"]) == {"d", "pit", "r1", "r2", "random", "teacher"}
    assert meta["controller"]["controller_units_per_evaluation"] == pytest.approx(
        art["n_parameters"] / 1000)
    assert meta.get("regret_check_all_equal", True)
    data = json.loads(gzip.open(next((tmp_path / "eval").glob("*iid_f0*.json.gz")), "rt").read())
    t = [r for r in data["rows"] if r["policy"] == "teacher"]
    for r in t:
        ref = run_episode(H.world(r["seed"], "iid_f0", EXEC, "e05apit-dev"), make_reference("dep_reuse"), 1.0)
        assert r["utility"] == ref["utility"]
    out = tmp_path / "score.json"
    T.main(["score", "--eval", str(tmp_path / "eval"), "--dev", "--output", str(out)])
    rep = json.loads(out.read_text())
    L = rep["lineages"]["tiny"]
    assert "PI-T-1" in L and "PI-T-2" in L and L["PI-T-2"]["pass"] is not None
    assert set(L["conditions"]) == {"iid_f0", "iid_f2", "foreign4"}
    assert L["iid_group"]["policies"]["teacher"]["delegated_step_fraction"] == 1.0
    assert L["iid_group"]["policies"]["d"]["delegated_step_fraction"] == 0.0
