"""extended-05 A-HR headroom tooling (campaign05_options, research/tools/campaign05_hr.py)."""
from dataclasses import replace
from functools import partial
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign03_depworld import Action, DepWorkshop, action_catalog, depworld_executor, generate_depworld
from tensegra.campaign04_branch import SolverCache
from tensegra.campaign04_deploy import deploy_episodes
from tensegra.campaign04_meta import FixedAgent, new_episode, run
from tensegra.campaign05_options import (OPTION_TYPES, HRConfig, Sampler, anchor_of, d_choice, d_rollouts, hr_labels,
                                         new_ep, option_set, outcome, run_policy)

ROOT = Path(__file__).resolve().parents[1]
DEV = 2_250_000_000  # extended-05 A-HR development/test seeds (research/campaigns/extended-05/seed-ranges.json)
EXEC = partial(depworld_executor, execute_call=execute)


def make(seed, executor=EXEC, **kw):
    return DepWorkshop(generate_depworld(seed, foreign_records=2, p_event=0.5, **kw), executor=executor,
                       address_seed=seed)


def actor(seed=0):
    torch.manual_seed(seed)
    return CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1")).eval()


def _tool():
    import sys
    sys.path.insert(0, str(ROOT / "research/tools"))
    spec = importlib.util.spec_from_file_location("campaign05_hr", ROOT / "research/tools/campaign05_hr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- deployment contract D -------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2])
def test_d_wrapper_bit_identical_to_deploy_r_mask_and_meta_run(seed):
    m = actor(seed)
    seeds = [DEV + 10 * seed + i for i in range(6)]
    ref = deploy_episodes(m, [make(s) for s in seeds], mode="r_mask", max_steps=96)
    eps = [new_ep(make(s), 96) for s in seeds]
    run_policy(m, eps, d_choice)
    meta = [new_episode(make(s), cap=96, telemetry=False) for s in seeds]
    with torch.no_grad():
        run(m, meta, FixedAgent("r_mask"))
    for r, ep, me in zip(ref, eps, meta):
        assert ep.actions_taken == [t["action_index"] for t in r["trace"]] == [x["a"] for x in me.log]
        assert outcome(ep.env)["utility"] == r["outcome"]["utility"] == outcome(me.env)["utility"]
        assert ep.tracker.summary() == r["progress"] == me.tracker.summary()
        assert ep.env.evaluate()["compute_units"] == r["outcome"]["compute_units"]


def test_d_wrapper_with_telemetry_does_not_change_choices():
    m = actor(3)
    seeds = [DEV + 40 + i for i in range(4)]
    a = d_rollouts(m, [make(s) for s in seeds], seeds, telemetry=False)
    b = d_rollouts(m, [make(s) for s in seeds], seeds, telemetry=True)
    assert [e.actions_taken for e in a] == [e.actions_taken for e in b]



# --- option set O(I) ---------------------------------------------------------------------

def _walk_options(env, m, steps=40):
    """D rollout recording the anchored and full option sets at every decision."""
    out = []

    def hook(ep, d, choice):
        n = len(d.actions)
        scores = d.logits[:n].tolist()
        out.append((option_set(d.observation, d.actions, d.default, "anchored", scores, d.masked),
                    option_set(d.observation, d.actions, d.default, "full"), d))
    run_policy(m, [new_ep(env, steps)], d_choice, hook=hook)
    return out


def test_anchored_option_set_is_public_only_and_well_formed():
    m = actor(4)
    anchors = set()
    for seed in (DEV + 50, DEV + 51, DEV + 52, DEV + 53):
        spec = generate_depworld(seed, foreign_records=2, p_event=0.5)
        rows = _walk_options(DepWorkshop(spec, executor=EXEC, address_seed=seed), m)
        assert rows
        for oi, full, d in rows:
            idx = [o["index"] for o in oi]
            assert idx == sorted(set(idx)) and sum(o["is_d"] for o in oi) == 1
            assert next(o for o in oi if o["is_d"])["index"] == d.default
            assert d.actions == action_catalog(d.observation)          # options index the public catalog
            assert [o["index"] for o in full] == list(range(len(d.actions)))
            da = d.actions[d.default]
            anchor, _ = anchor_of(d.observation, da)
            anchors.add(anchor)
            assert all(o["anchor"] == anchor for o in oi)
            types = {o["type"] for o in oi if not o["is_d"]}
            assert types <= set(OPTION_TYPES)
            assert "abstain" in types or da.kind == "abstain"
            allowed = {"call": {"budget", "call_other", "not_call"},
                       "reuse_recompute": {"reuse", "reuse_retrieve", "recompute"},
                       "commit_revise": {"revise_assign", "revise_select"}, "other": {"call_now"}}[anchor]
            assert types - {"abstain"} <= allowed
            for o in oi:
                a = d.actions[o["index"]]
                if o["type"] == "budget":
                    assert a.kind == "call" and a.arguments["problem"] == da.arguments["problem"]
                    assert a.arguments["budget"] != da.arguments["budget"]
                if o["type"] == "not_call":
                    assert a.kind not in ("call", "abstain")
                if o["type"] == "abstain":
                    assert a.kind == "abstain"
                if o["type"] == "revise_select":
                    assert a.kind == "uncommit" and d.observation.selection_id is not None
    assert len(anchors) >= 2
    o0 = make(DEV + 55).observe()
    acts0 = action_catalog(o0)
    think = next(i for i, a in enumerate(acts0) if a.kind == "think")
    oi0 = option_set(o0, acts0, think)
    assert {o["anchor"] for o in oi0} == {"other"} and {o["type"] for o in oi0} == {"D", "abstain", "call_now"}
    assert all(acts0[o["index"]].kind == "call" for o in oi0 if o["type"] == "call_now")   # foreign drafts are open
    # a function of the public observation and catalog only: worlds with different hidden values but the
    # same visible history give the same option set for every possible D choice
    seed = DEV + 54
    spec = generate_depworld(seed, foreign_records=2, p_event=0.5)
    other = replace(spec, funds=spec.funds + 7, capacity=spec.capacity + 3)
    acts = [Action("think"), Action("inspect", {"target": spec.items[0].handle}), Action("inspect", {"target": "map"})]
    envs = [DepWorkshop(s, executor=EXEC, address_seed=seed) for s in (spec, other)]
    for a in acts:
        obs = [e.observe() for e in envs]
        assert obs[0].to_dict() == obs[1].to_dict()
        for d_index in range(len(action_catalog(obs[0]))):
            sets = [option_set(o, action_catalog(o), d_index) for o in obs]
            assert sets[0] == sets[1]
        for e in envs:
            e.step(a)


def test_public_sampler_registered_rule():
    cfg = HRConfig()
    n_all = 0
    for s in range(DEV, DEV + 400):
        a, b = Sampler(s, cfg), Sampler(s, cfg)
        seq = [a("other" if t % 3 else "call") for t in range(40)]
        assert seq == [b("other" if t % 3 else "call") for t in range(40)]
        assert sum(x[0] for x in seq) <= cfg.max_points
        assert all(x[1] == (cfg.p_other if t % 3 else cfg.p_call) for t, x in enumerate(seq))
        assert all(x[2] == (x[0] or a.all_states) for x in seq)
        n_all += a.all_states
    assert 5 < n_all < 60
    assert not Sampler(DEV, replace(cfg, all_states=False)).all_states


# --- branch labels -------------------------------------------------------------------------

def _labels(m, seeds, teacher_factory=None, **cfg_kw):
    cache = SolverCache(EXEC)
    cfg = HRConfig(**{"check_fraction": 1.0, "max_steps": 40, "p_other": 0.3, **cfg_kw})
    kw = {} if teacher_factory is None else {"teacher_factory": teacher_factory}
    return hr_labels(m, [make(s, cache) for s in seeds], seeds, cfg, **kw), cfg


def test_branch_with_option_d_reproduces_main_line():
    m = actor(5)
    seeds = [DEV + 60 + i for i in range(6)]
    (eps, points, stats), cfg = _labels(m, seeds)
    assert stats["default_checks"] == len(points) > 10
    assert stats["default_check_matches"] == stats["default_checks"]   # clone + D == main line, bit for bit
    ref = d_rollouts(m, [make(s) for s in seeds], seeds, cfg)       # the labels' main line is plain D
    assert [e["actions"] for e in eps] == [r.actions_taken for r in ref]
    for p in points:
        assert p["headroom"] >= p["headroom_excl_abstain"] >= 0.0
        d = next(o for o in p["options"] if o["is_d"])
        assert d["dU"] == p["q_d"]
        assert max(o["dU"] for o in p["options"]) - p["q_d"] == p["headroom"]
    for e in eps:
        assert len(e["decisions"]) == e["T"]
        assert e["sampled_steps"] == [x["step"] for x in e["decisions"] if x["sampled"]]
        branched = {p["step"] for p in points if p["seed"] == e["seed"]}
        assert branched == (set(range(e["T"])) if e["all_states"] else set(e["sampled_steps"]))


def test_labels_use_the_evaluation_utility():
    m = actor(6)
    seeds = [DEV + 70 + i for i in range(4)]
    (eps, points, stats), cfg = _labels(m, seeds, check_fraction=0.0, p_other=0.5)
    info = {e["seed"]: e for e in eps}
    assert points
    checked = 0
    for p in points[:16]:
        e = info[p["seed"]]
        spec = generate_depworld(p["seed"], foreign_records=2, p_event=0.5)
        assert p["q_d"] == e["utility"] - p["U_t"]       # D: realized evaluate()["utility"] of the main line - U_t
        for o in p["options"]:
            if o["type"] == "abstain":
                assert math.isclose(o["dU"], -(spec.action_price + spec.compute_price), abs_tol=1e-12)
        # independent re-simulation of one non-D option: fresh world, replay D's first t actions,
        # take the option, then D; label = evaluate()["utility"] - U_t
        opt = next((o for o in p["options"] if not o["is_d"] and o["type"] != "abstain"), None)
        if opt is None:
            continue
        env = make(p["seed"])
        prefix = e["actions"][:p["step"]]
        state = {"u_t": None}

        def chooser(decisions, episodes, prefix=prefix, opt=opt):
            ep = episodes[0]
            if ep.taken < len(prefix):
                return [prefix[ep.taken]]
            if ep.taken == len(prefix):
                state["u_t"] = ep.env.current_utility()
                assert state["u_t"] == ep.env.evaluate()["utility"]
                return [opt["index"]]
            return [decisions[0].default]
        run_policy(m, [new_ep(env, 40)], chooser)
        assert state["u_t"] == p["U_t"]
        assert env.evaluate()["utility"] - p["U_t"] == opt["dU"]
        checked += 1
    assert checked >= 3


def test_full_catalog_mode_marks_oi_options():
    m = actor(7)
    seeds = [DEV + 80, DEV + 81]
    (eps, points, stats), cfg = _labels(m, seeds, option_mode="full", all_states=False, check_fraction=0.0,
                                        max_points=1, max_steps=16, p_other=0.5, p_call=0.5, p_reuse_recompute=0.5)
    assert points
    for p in points:
        assert len(p["options"]) == p["candidates"]
        assert any(o["in_oi"] for o in p["options"]) and not all(o["in_oi"] for o in p["options"])



# --- seed ranges, bases and the tool ------------------------------------------------------------

def test_seed_ranges_disjoint_and_registered():
    T = _tool()
    rs = T.check_seed_ranges()
    names = [r["name"] for r in rs]
    assert any("220M" in n for n in names) and any("230M" in n for n in names)
    for a in rs:
        if a["campaign"] == "extended-05":
            continue
        for kind, base in T.WORLD_BASES.items():
            lo, hi = base, base + T.STRIDE * len(T.CONDITIONS)
            assert not (lo < a["hi"] and a["lo"] < hi), (kind, a["name"])
    assert DEV == T.WORLD_BASES["dev"]
    for lo, hi in ((130_000_000, 131_000_000), (161_000_000, 162_000_000), (200_000_000, 201_000_000)):
        assert any(r["lo"] <= lo and hi <= r["hi"] for r in rs)


def test_seed_range_overlap_is_rejected(tmp_path):
    T = _tool()
    data = json.loads(T.SEED_RANGES.read_text())
    data["ranges"].append({"name": "bad", "campaign": "x", "lo": 220_000_050, "hi": 220_000_060})
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        T.check_seed_ranges(path)


def test_bases_six_frozen_bootstraps_with_renamed_roots():
    T = _tool()
    b = T.bases()
    assert sorted(b) == [f"x1-r{r}" for r in range(6)]
    assert len({x["sha256"] for x in b.values()}) == 6
    for r in range(3):
        assert b[f"x1-r{r}"]["path"] == (f"/home/brand/structured-latent-dynamics-campaign03/results/p1-boot-x1-r{r}"
                                         "/checkpoints/round-0-slot-5-attempt-5.pt")
    for r in (3, 4, 5):
        assert b[f"x1-r{r}"]["path"] == (f"/home/brand/structured-latent-dynamics-campaign04/results/f-boot-x1-r{r}"
                                         "/checkpoints/round-0-slot-5-attempt-5.pt")


class _InProcessSolver:
    execute = staticmethod(execute)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False



def test_tool_end_to_end(tmp_path):
    from types import SimpleNamespace
    T = _tool()
    tiny = actor(9)
    binding = {"name": "tiny", "sha256": "0" * 64, "source": "test"}
    load = lambda b, verify=True: (tiny, SimpleNamespace(max_steps=30, neural_work_per_forward=1.0),  # noqa: E731
                                   {"parameters": 1, "sha256": b["sha256"]})
    kw = dict(binding=binding, load_actor_fn=load, solver=_InProcessSolver)

    def args(cmd, cond="iid_f0", base="tiny", out="branch", *extra):
        return [cmd, "--base", base, "--condition", cond, "--worlds", "dev", "--chunk", "0", "--chunk-size", "10",
                "--output", str(tmp_path / out), *extra]
    with pytest.raises(SystemExit):   # r3-r5 are reserved for confirmation
        T.main(args("branch", base="x1-r3", out="x"))
    T.main(args("verify-d", out="verify"), **kw)
    T.main(args("states", out="states")[:8] + ["0", "1"] + args("states", out="states")[9:], **kw)   # two chunks
    for cond in ("iid_f0", "iid_f2"):
        T.main(args("branch", cond), **kw)
    with pytest.raises(SystemExit):   # never overwrites
        T.main(args("branch"), **kw)
    T.main(args("branch", "iid_f0", "tiny", "full", "--option-set", "full", "--no-all-states", "--episodes", "2"),
           **kw)
    meta = json.loads(next((tmp_path / "branch").glob("*iid_f0*.meta.json")).read_text())
    assert meta["summary"]["default_checks"][0] == meta["summary"]["default_checks"][1]
    assert "hidden-state" in meta["privilege"] and not meta["protocol_worlds"] and meta["device"] == "cpu"
    assert meta["seeds"] == [DEV, DEV + 10]
    out = tmp_path / "analysis.json"
    T.main(["analyze", "--branch", str(tmp_path / "branch"), "--full", str(tmp_path / "full"), "--states",
            str(tmp_path / "states"), "--folds", "2", "--inner", "2", "--per-group-estimate", "--output", str(out)])
    rep = json.loads(out.read_text())
    h = rep["pooled"]["hindsight"]
    assert h["episodes"] == 20 and "HINDSIGHT" in h["label"]
    assert h["episode_single_deviation_H_incl_abstain"]["mean"] >= h["episode_single_deviation_H_excl_abstain"]["mean"]
    g = rep["pooled"]["G1b_excl_abstain"]
    assert "SAME-INFORMATION" in g["label"] and g["episodes"] == 20 and "abstain" not in g["deviation_types"]
    fc = rep["full_catalog"]
    assert fc["states"] > 0 and fc["oi_file_consistency_matches"] == fc["oi_file_overlap_states"]
    assert fc["mean_H_full"] >= fc["mean_H_oi"]
    assert rep["states"]["tiny/iid_f0/dev"]["episodes"] == 20
    assert len(list((tmp_path / "states").glob("states-*.json.gz"))) == 2


def test_single_deviation_gain_equals_branched_advantage():
    T = _tool()

    def pt(step, sampled, advs, q_d=0.5):
        opts = [{"index": 0, "type": "D", "types": ["D", "x"], "is_d": True, "dU": q_d}]
        opts += [{"index": i + 1, "type": t, "types": [t], "is_d": False, "dU": q_d + v}
                 for i, (t, v) in enumerate(advs)]
        return {"step": step, "sampled": sampled, "q_d": q_d, "options": opts}
    ps = [pt(0, False, [("call_now", 0.3)]), pt(1, True, [("call_now", -0.1), ("abstain", 0.4)]),
          pt(2, True, [("call_now", 0.25)])]
    oracle = lambda p, o: o["dU"] - p["q_d"]  # noqa: E731
    assert T._deviate(ps, oracle, 0.0, False) == (0.25, "call_now")    # unsampled step 0 is never used
    assert T._deviate(ps, oracle, 0.0, True) == (pytest.approx(0.4), "abstain")
    assert T._deviate(ps, oracle, 0.5, False) == (0.0, None)


# --- A-HR2: multi-step options (delegate T; budget override B) ------------------------------------

from tensegra.campaign02_training import digest  # noqa: E402
from tensegra.campaign03_depworld import DepReference  # noqa: E402
from tensegra.campaign05_options import (DELEGATE_MAX_STEPS, budget_override_options, committed,  # noqa: E402
                                         make_teacher, multi_option_set)


def _labels_digest(m, seeds, **cfg_kw):
    """Digest of hr_labels' episodes/points (CPU timings excluded): pins the default A-HR outputs."""
    (eps, points, stats), _ = _labels(m, seeds, **cfg_kw)
    keep = {k: v for k, v in stats.items() if k not in ("cpu", "timing_branch")}
    return digest({"episodes": eps, "points": points, "stats": keep})


# Produced by the A-HR code at e29f06ba (before A-HR2) on the registered CPU environment.
GOLDEN_AHR = {"anchored": "6afe5ca141dee2e88b4d727828a8c28809338e05952c21ec0428b5be8430da78",
              "full": "2eb69cd2a933fa65964571569185892082d645e01e3e8bb77ce9c89e81076967"}


def test_default_a_hr_outputs_bit_identical_to_pre_hr2():
    m = actor(5)
    seeds = [DEV + 60 + i for i in range(4)]
    assert _labels_digest(m, seeds) == GOLDEN_AHR["anchored"]
    assert _labels_digest(m, seeds[:2], option_mode="full", all_states=False, max_points=2, max_steps=16,
                          p_other=0.5) == GOLDEN_AHR["full"]


def test_teacher_is_fresh_stateless_and_public_only():
    t = make_teacher()
    assert isinstance(t, DepReference) and t.mode == "reuse" and t.reference_name == "dep_reuse"
    assert set(vars(t)) == {"mode", "initial_budget", "reference_name"}     # no episode state, no environment
    # the teacher's choice (and the multi option set) is a function of the public observation: two worlds
    # with different hidden values and the same visible history give the same proposal; choose never
    # mutates the observation it is given
    seed = DEV + 54
    spec = generate_depworld(seed, foreign_records=2, p_event=0.5)
    other = replace(spec, funds=spec.funds + 7, capacity=spec.capacity + 3)
    envs = [DepWorkshop(s, executor=EXEC, address_seed=seed) for s in (spec, other)]
    for a in (Action("think"), Action("inspect", {"target": spec.items[0].handle}), Action("inspect", {"target": "map"}),
              None):
        obs = [e.observe() for e in envs]
        assert obs[0].to_dict() == obs[1].to_dict()     # same visible history, different hidden requirements
        before = obs[0].to_dict()
        acts = [make_teacher().choose(o, action_catalog(o)) for o in obs]
        assert acts[0] == acts[1] and obs[0].to_dict() == before
        cat = action_catalog(obs[0])
        for d_index in range(len(cat)):
            assert multi_option_set(obs[0], cat, d_index) == multi_option_set(obs[1], action_catalog(obs[1]), d_index)
        if a is not None:
            for e in envs:
                e.step(a)
    # along a whole teacher trajectory the proposal never reads the observation's identity/mutable state:
    # a deep copy gives the same action and the observation is left untouched
    from copy import deepcopy
    env, t = make(DEV + 91), make_teacher()
    o = env.observe()
    while not o.done and o.step < 40:
        before = o.to_dict()
        a = t.choose(o, action_catalog(o))
        assert a == make_teacher().choose(deepcopy(o), action_catalog(o)) and o.to_dict() == before
        o = env.step(a)


class _ReplayTeacher:
    """A 'teacher' that proposes D's main-line action for every main-line observation."""

    def __init__(self, table):
        self.table = table

    def choose(self, o, actions):
        return self.table[digest(o.to_dict())]


def test_delegate_that_picks_d_actions_reproduces_the_main_line():
    m = actor(5)
    seeds = [DEV + 60 + i for i in range(6)]
    table = {}

    def hook(ep, d, choice):
        table[digest(d.observation.to_dict())] = d.actions[d.default]
    d_rollouts(m, [make(s) for s in seeds], seeds, HRConfig(max_steps=40), hook=hook)
    (eps, points, stats), _ = _labels(m, seeds, option_mode="multi", teacher_factory=lambda: _ReplayTeacher(table))
    assert points and stats["default_check_matches"] == stats["default_checks"]
    for p in points:
        assert [o["type"] for o in p["options"]] == ["D", "delegate"]
        d, t = p["options"]
        assert t["index"] == d["index"] and t["teacher_first_is_d"]
        # T then D == the main line, bit for bit: realized utility, success, cost, length
        assert (t["dU"], t["success"], t["cost_rem"], t["dsteps"]) == (d["dU"], d["success"], d["cost_rem"],
                                                                        d["dsteps"])
        assert t["teacher_agree"] == t["teacher_steps"] <= DELEGATE_MAX_STEPS
        assert t["delegate_end"] in ("commit", "max_steps", "episode_end", "cap")


def test_delegate_label_matches_independent_resimulation():
    m = actor(6)
    seeds = [DEV + 70 + i for i in range(6)]
    (eps, points, stats), cfg = _labels(m, seeds, option_mode="multi", check_fraction=0.0, p_other=0.5)
    info = {e["seed"]: e for e in eps}
    ends, checked = set(), 0
    for p in points[:40]:
        opt = next((o for o in p["options"] if o["type"] == "delegate"), None)
        if opt is None:
            continue
        e = info[p["seed"]]
        prefix = e["actions"][:p["step"]]
        env = make(p["seed"])
        st = {"teacher": None, "n": 0, "on": False, "agree": 0}

        def chooser(decisions, episodes, prefix=prefix, st=st):
            d, ep = decisions[0], episodes[0]
            if ep.taken < len(prefix):
                return [prefix[ep.taken]]
            if ep.taken == len(prefix):
                st["u_t"] = ep.env.current_utility()
                st["teacher"], st["on"] = DepReference("reuse"), True     # fresh teacher at the point
            elif st["on"]:
                fb = d.observation.feedback
                if (fb.get("status") == "success" and ("selection_id" in fb or "assignment_id" in fb)) \
                        or st["n"] >= DELEGATE_MAX_STEPS:
                    st["on"] = False
            if st["on"]:
                a = d.actions.index(st["teacher"].choose(d.observation, d.actions))
                st["n"] += 1
                st["agree"] += a == d.default
                return [a]
            return [d.default]
        run_policy(m, [new_ep(env, cfg.max_steps)], chooser)
        assert st["u_t"] == p["U_t"]
        assert env.evaluate()["utility"] - p["U_t"] == opt["dU"]
        assert (st["n"], st["agree"]) == (opt["teacher_steps"], opt["teacher_agree"])
        ends.add(opt["delegate_end"])
        checked += 1
    assert checked >= 8
    assert "commit" in ends


def test_committed_uses_the_environment_completion_rule():
    env = make(DEV + 90)
    t = make_teacher()
    o = env.observe()
    n_commits = 0
    while not o.done and o.step < 60:
        before = env._completions
        o = env.step(t.choose(o, action_catalog(o)))
        assert committed(o) == (env._completions == before + 1)
        n_commits += committed(o)
    assert n_commits >= 1


def test_budget_override_is_a_subset_of_existing_options():
    """Option B (next-call budget override, b in {16, 1024, remaining}) is identical to existing A-HR
    options (budget at D-call points, call_now at other points), so it is not re-branched."""
    m = actor(4)
    kinds = set()
    for seed in (DEV + 50, DEV + 51, DEV + 52, DEV + 53):
        for oi, full, d in _walk_options(make(seed), m):
            b = budget_override_options(d.observation, d.actions, d.default)
            by = {o["index"]: o["type"] for o in oi}
            assert set(b) <= set(by)
            for i in b:
                a = d.actions[i]
                assert a.kind == "call" and a.arguments["budget"] in (16, 1024, d.observation.remaining_work)
                assert by[i] in ("budget", "call_now")
                kinds.add(by[i])
            da = d.actions[d.default]
            if da.kind == "call":
                assert {d.actions[i].arguments["budget"] for i in b} == {
                    x for x in (16, 1024, d.observation.remaining_work)
                    if x != da.arguments["budget"] and any(
                        a.kind == "call" and a.arguments == {**da.arguments, "budget": x} for a in d.actions)}
    assert "budget" in kinds
    o0 = make(DEV + 55).observe()        # 'other' anchor (D = think) with open foreign drafts: B = call_now options
    acts0 = action_catalog(o0)
    think = next(i for i, a in enumerate(acts0) if a.kind == "think")
    b0 = budget_override_options(o0, acts0, think)
    by0 = {o["index"]: o["type"] for o in option_set(o0, acts0, think)}
    assert b0 and all(by0.get(i) == "call_now" for i in b0)


def test_tool_multi_end_to_end(tmp_path):
    from types import SimpleNamespace
    T = _tool()
    tiny = actor(9)
    binding = {"name": "tiny", "sha256": "0" * 64, "source": "test"}
    load = lambda b, verify=True: (tiny, SimpleNamespace(max_steps=30, neural_work_per_forward=1.0),  # noqa: E731
                                   {"parameters": 1, "sha256": b["sha256"]})
    kw = dict(binding=binding, load_actor_fn=load, solver=_InProcessSolver)

    def args(out, cond, *extra):
        return ["branch", "--base", "tiny", "--condition", cond, "--worlds", "dev", "--chunk", "0", "1",
                "--chunk-size", "8", "--output", str(tmp_path / out), *extra]
    with pytest.raises(SystemExit):
        T.main(["branch", "--base", "x1-r4", "--condition", "iid_f0", "--chunk", "0", "--option-class", "multi",
                "--output", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        T.main(args("m", "iid_f0", "--option-class", "multi", "--option-set", "full"), **kw)
    for cond in ("iid_f0", "iid_f2"):
        T.main(args("branch", cond), **kw)
        T.main(args("multi", cond, "--option-class", "multi"), **kw)
    stems = sorted(p.name for p in (tmp_path / "multi").glob("*.json.gz"))
    assert stems[0] == "branch-tiny-iid_f0-dev-multi-c000.json.gz" and len(stems) == 4
    meta = json.loads((tmp_path / "multi" / "branch-tiny-iid_f0-dev-multi-c000.meta.json").read_text())
    assert meta["option_class"] == "multi" and "dep_reuse" in meta["options"]["delegate"]
    assert meta["summary"]["b_subset_of_oi_all"]
    assert meta["summary"]["default_checks"][0] == meta["summary"]["default_checks"][1]
    out = tmp_path / "a.json"
    T.main(["analyze", "--branch", str(tmp_path / "branch"), "--multi", str(tmp_path / "multi"), "--folds", "2",
            "--inner", "2", "--output", str(out)])
    rep = json.loads(out.read_text())
    mu = rep["multi"]
    assert mu["pairing"] == {"multi_episodes": 32, "paired_episodes": 32, "main_line_mismatch": 0,
                             "point_mismatch": 0, "unpaired_points": 0}
    pa = mu["G1a_multi_paired"]
    assert pa["O(I)_excl_abstain"]["mean"] == pytest.approx(
        rep["pooled"]["hindsight"]["episode_single_deviation_H_excl_abstain"]["mean"])
    assert pa["T_on_paired_episodes"]["mean"] == pytest.approx(mu["G1a_multi_T"]["mean"])
    assert pa["T_plus_O(I)_excl_abstain"]["mean"] >= max(pa["O(I)_excl_abstain"]["mean"], mu["G1a_multi_T"]["mean"])
    assert pa["O(I)_excl_abstain"]["mean"] >= pa["B_from_A-HR_labels"]["mean"] >= 0
    for k in ("G1b_multi_T", "G1b_multi_T_plus_O(I)_excl_abstain"):
        g = mu[k]
        assert g["episodes"] == 32 and "by_D_episode_outcome" in g and "by_firing_anchor" in g
        assert "abstain" not in g["deviation_types"]
    assert set(mu["G1b_multi_T"]["deviation_types"]) <= {"D", "delegate"}
    # the default analysis (no --multi) is unchanged
    out2 = tmp_path / "b.json"
    T.main(["analyze", "--branch", str(tmp_path / "branch"), "--folds", "2", "--inner", "2", "--output", str(out2)])
    rep2 = json.loads(out2.read_text())
    assert "multi" not in rep2 and rep2["pooled"] == rep["pooled"]
