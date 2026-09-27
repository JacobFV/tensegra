"""extended-05 A-PI-C: delegation cost on top of the A-PI-T controller machinery (campaign05_apit)."""
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensegra.campaign02_references import make_reference, run_episode
from tensegra.campaign05_options import outcome
from tensegra.campaign05_apit import (DELEGATION_COSTS, Controller, Delegator, RegretHook, apply_delegation_cost,
                                      canonical_hash, cost_adjusted_label, episode_row, feature_names,
                                      run_delegator, run_regret_branches)

from test_campaign05_apit import EXEC, _InProcessSolver, _load, actor, artifact, make

ROOT = Path(__file__).resolve().parents[1]
DEV = 2_250_700_000  # A-PI-C tests: inside the extended-05 development entry (non-protocol)


def cost_artifact(c, margin=0.0, seed=0, protocol=False):
    a = artifact(margin=margin, seed=seed)
    ctrl = dict(a["controller"], delegation_cost=c)
    return {"controller": ctrl, "controller_sha256": canonical_hash(ctrl), "protocol": protocol}


def _strip(row):
    return {k: v for k, v in row.items() if k not in ("utility_gross", "delegation_cost", "delegation_charge")}


# --- cost accounting ------------------------------------------------------------------------------

@pytest.mark.parametrize("pol", ["d", "r1", "r2", "random", "always", "pit"])
def test_zero_cost_reproduces_a_pi_t_rows_exactly(pol):
    m = actor(21)
    seeds = [DEV + i for i in range(5)]
    ctrl = Controller(artifact(seed=2), margin=0.0)
    kw = {"pit": {"controller": ctrl, "controller_units": .01}, "random": {"rate": .4, "rng_key": "k"}}.get(pol, {})
    ref = run_delegator(m, [make(s) for s in seeds], seeds, Delegator(pol, **kw), cap=60)
    got = run_delegator(m, [make(s) for s in seeds], seeds, Delegator(pol, delegation_cost=0.0, **kw), cap=60)
    for a, b in zip(ref, got):
        assert a.actions_taken == b.actions_taken
        assert outcome(a.env) == outcome(b.env)
        ra, rb = episode_row(a, pol), episode_row(b, pol)
        assert "delegation_cost" not in ra                        # default rows unchanged (A-PI-T/A-CF-T)
        assert _strip(rb) == ra                                    # bit-identical incl. utility
        assert rb["delegation_charge"] == 0.0 and rb["utility_gross"] == ra["utility"]


def test_cost_charged_once_per_teacher_step_and_never_to_the_environment():
    m = actor(22)
    seeds = [DEV + 100 + i for i in range(6)]
    c = 0.003
    ctrl = Controller(artifact(seed=4), margin=-math.inf)          # delegate at every eligible decision
    free = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("pit", controller=ctrl), cap=80)
    paid = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("pit", controller=ctrl, delegation_cost=c),
                         cap=80)
    charged = 0
    for a, b in zip(free, paid):
        assert a.actions_taken == b.actions_taken and outcome(a.env) == outcome(b.env)   # external: no env charge
        rb = episode_row(b, "pit")
        assert rb["delegation_charge"] == c * rb["teacher_steps"]
        assert rb["utility"] == rb["utility_gross"] - c * rb["teacher_steps"]
        assert rb["utility_gross"] == a.env.evaluate()["utility"]
        charged += rb["teacher_steps"]
        with pytest.raises(ValueError):
            apply_delegation_cost(rb, c)                            # charging twice refused
        assert episode_row(b, "pit") == rb                          # rebuilding the row charges once again, not twice
    assert charged > 0
    # D pays nothing; always-teacher pays every step of the episode
    d = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("d", delegation_cost=c), cap=80)
    assert all(episode_row(e, "d")["delegation_charge"] == 0 for e in d)
    al = run_delegator(m, [make(s) for s in seeds], seeds, Delegator("always", delegation_cost=c), cap=96)
    for s, e in zip(seeds, al):
        r = episode_row(e, "always")
        ref = run_episode(make(s), make_reference("dep_reuse"), model_compute_tariff=1.0)
        assert r["teacher_steps"] == r["decisions"] == len(ref["trace"])
        assert r["utility"] == ref["utility"] - c * len(ref["trace"])
    with pytest.raises(ValueError):
        Delegator("d", delegation_cost=-1e-3)


def test_cost_controller_refuses_a_different_cost():
    ctrl = Controller(cost_artifact(0.001))
    assert ctrl.delegation_cost == 0.001
    Delegator("pit", controller=ctrl, delegation_cost=0.001)
    for bad in (None, 0.003, 0.0):
        with pytest.raises(ValueError):
            Delegator("pit", controller=ctrl, delegation_cost=bad)
    assert Controller(artifact()).delegation_cost is None          # A-PI-T controller


def test_regret_branches_net_of_cost_are_deterministic_and_zero_cost_equal():
    m = actor(23)
    seeds = [DEV + 200 + i for i in range(4)]
    art = cost_artifact(0.003, seed=3)
    rows = {}
    for c in (None, 0.0, 0.003):
        a = art if c == 0.003 else artifact(seed=3)
        dl = Delegator("pit", controller=Controller(a, margin=0.0), controller_units=0.01, delegation_cost=c)
        hook = RegretHook(1.0, max_points=4)
        run_delegator(m, [make(s) for s in seeds], seeds, dl, cap=40, hook=hook)
        rows[c] = run_regret_branches(m, dl, hook.points, check=True)
        assert rows[c] and all(r["check_equal"] for r in rows[c]) and all(r["regret"] >= 0 for r in rows[c])
    for a, b in zip(rows[None], rows[0.0]):
        assert {k: v for k, v in b.items() if k != "delegation_cost"} == a
    # the net Q's differ from the free ones only through c x teacher steps (never positive for delegate lines)
    assert any(r["q_delegate"] != f["q_delegate"] for r, f in zip(rows[0.003], rows[None]))


# --- training -------------------------------------------------------------------------------------

def _branch(tmp_path, H, kw, seed=0):
    out = tmp_path / "multi"
    if not out.exists():
        for cond in ("iid_f0", "iid_f2"):
            H.main(["branch", "--option-class", "multi", "--base", "tiny", "--condition", cond, "--worlds", "dev",
                    "--chunk", "0", "1", "--chunk-size", "8", "--output", str(out)], **kw)
    return out


def _kw(seed=24):
    tiny = actor(seed)
    binding = {"name": "tiny", "sha256": "0" * 64, "source": "test"}
    load = lambda b, verify=True: (tiny, SimpleNamespace(max_steps=30, neural_work_per_forward=1.0),  # noqa: E731
                                   {"parameters": 1000, "sha256": b["sha256"]})
    return dict(binding=binding, load_actor_fn=load, solver=_InProcessSolver)


def test_label_adjustment_exact_and_zero_cost_training_reproduces_a_pi_t(tmp_path):
    H, T = _load("campaign05_hr"), _load("campaign05_apit")
    kw = _kw()
    br = _branch(tmp_path, H, kw)
    free, ep0, _ = T.training_points([br], dev=True)
    for c in (0.0, 0.001, 0.003):
        rows, ep, _ = T.training_points([br], dev=True, cost=c)
        assert ep == ep0 and len(rows) == len(free)
        for r, f in zip(rows, free):
            assert r["x"] == f["x"]                                    # features never depend on c
            assert 1 <= r["teacher_steps"] <= 12
            assert r["y"] == (r["dU"] - c * r["teacher_steps"]) - r["q_d"]
            if c == 0.0:
                assert r["y"] == f["y"]
    # teacher_steps is the recorded branch count (A-HR2 run_branches), read, not recomputed
    data = [d for _, d in H._read([br], "branch")]
    ts = [o["teacher_steps"] for d in data for p in d["points"] if p["anchor"] in ("call", "reuse_recompute",
                                                                                   "commit_revise")
          for o in p["options"] if o["type"] == "delegate"]
    assert sorted(ts) == sorted(r["teacher_steps"] for r in rows)
    with pytest.raises(ValueError):
        cost_adjusted_label(0.1, 0.0, None, 0.001)
    # c = 0 training == A-PI-T training (same models / margin / selection tables)
    a0, ac = tmp_path / "c0.json", tmp_path / "free.json"
    T.main(["train", "--branch", str(br), "--output", str(ac), "--dev", "--folds", "2"])
    T.main(["train", "--branch", str(br), "--output", str(a0), "--dev", "--folds", "2", "--delegation-cost", "0"])
    A0, AF = json.loads(a0.read_text()), json.loads(ac.read_text())
    assert "delegation_cost" not in AF["controller"] and "delegation_cost" not in AF
    assert A0["controller"]["models"] == AF["controller"]["models"]
    assert A0["controller"]["margin"] == AF["controller"]["margin"]
    assert A0["selection"]["objective_table"] == AF["selection"]["objective_table"]
    assert {k: v for k, v in A0["controller"].items() if k not in ("delegation_cost", "target")} == \
        {k: v for k, v in AF["controller"].items() if k != "target"}
    assert A0["delegation_cost"]["label_adjustment_exact"]
    with pytest.raises(SystemExit):   # unregistered cost refused on protocol training
        T.main(["train", "--branch", str(br), "--output", str(tmp_path / "x.json"), "--delegation-cost", "0.002"])


def test_missing_teacher_steps_refused(tmp_path):
    H, T = _load("campaign05_hr"), _load("campaign05_apit")
    br = _branch(tmp_path, H, _kw())
    bad = tmp_path / "bad"
    bad.mkdir()
    for p in br.glob("branch-*"):
        if p.name.endswith(".json.gz"):
            d = json.loads(gzip.open(p, "rt").read())
            for pt in d["points"]:
                for o in pt["options"]:
                    o.pop("teacher_steps", None)
            with gzip.open(bad / p.name, "wt") as f:
                json.dump(d, f)
        else:
            (bad / p.name).write_text(p.read_text())
    T.training_points([bad], dev=True)                            # A-PI-T labels do not need it
    with pytest.raises(SystemExit):
        T.training_points([bad], dev=True, cost=0.001)


# --- worlds and guards ----------------------------------------------------------------------------

def test_apic_worlds_registered_and_guarded(tmp_path):
    T = _load("campaign05_apit")
    rs = T.check_seed_ranges()
    for kind, name, lo in (("apic", "A-PI-C", 270_000_000), ("acfc", "A-CF-C", 280_000_000)):
        e = next(r for r in rs if name in r["name"])
        assert (e["lo"], e["hi"]) == (lo, lo + 400_000)
        assert T.world_seeds(kind, "iid_f0", 0, 32)[0] == lo
        assert T.world_seeds(kind, "foreign4", 15, 32)[-1] == lo + 300_511
        with pytest.raises(SystemExit):
            T.world_seeds(kind, "iid_f0", 16, 32)
        bad = json.loads((ROOT / "research/campaigns/extended-05/seed-ranges.json").read_text())
        for r in bad["ranges"]:
            if name in r["name"]:
                r["name"] = r["name"].replace(name, "other")
        p = tmp_path / f"s-{kind}.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            T.check_seed_ranges(p)
    assert len({T.NAMESPACES[k] for k in T.NAMESPACES}) == len(T.NAMESPACES)
    assert T.WORLD_BASES["apit"] == 240_000_000 and T.WORLD_BASES["acf"] == 260_000_000


def test_eval_guards(tmp_path):
    T = _load("campaign05_apit")
    kw = _kw()
    paths = {}
    for c in DELEGATION_COSTS:
        paths[c] = tmp_path / f"c{c}.json"
        paths[c].write_text(json.dumps(cost_artifact(c)))
    cc = [f"{c}={p}" for c, p in paths.items()]
    base = ["eval", "--condition", "iid_f0", "--chunk", "0", "--episodes", "1", "--output", str(tmp_path / "o")]
    refused = [
        ["--base", "x1-r3", "--worlds", "apic", "--cost-controllers", *cc],        # confirmation lineage on screen
        ["--base", "x1-r0", "--worlds", "acfc", "--cost-controllers", *cc],        # screen lineage on confirmation
        ["--base", "x1-r0", "--worlds", "apic"],                                   # cost worlds need controllers
        ["--base", "x1-r0", "--worlds", "apit", "--cost-controllers", *cc],        # A-PI-T worlds stay A-PI-T
        ["--base", "x1-r0", "--worlds", "acf", "--cost-controllers", *cc],
        ["--base", "x1-r0", "--worlds", "apic", "--cost-controllers", cc[0]],      # every registered cost
        ["--base", "x1-r0", "--worlds", "apic", "--cost-controllers", *cc],        # protocol needs protocol artifacts
        ["--base", "x1-r0", "--worlds", "apic", "--cost-controllers", f"0.003={paths[0.001]}", f"0.001={paths[0.003]}"],
        ["--base", "x1-r0", "--worlds", "apic", "--cost-controllers", *cc, "--margin-override", "1"],
    ]
    for extra in refused:
        with pytest.raises(SystemExit):
            T.main(base + extra, **{k: v for k, v in kw.items() if k != "binding"})
    with pytest.raises(SystemExit):   # key/cost mismatch also on dev worlds
        T.main(base + ["--base", "tiny", "--worlds", "dev", "--cost-controllers", f"0.003={paths[0.001]}"], **kw)
    assert not (tmp_path / "o").exists() or not list((tmp_path / "o").glob("*.json.gz"))


# --- tool end to end ------------------------------------------------------------------------------

def test_tool_end_to_end_cost(tmp_path):
    H, T = _load("campaign05_hr"), _load("campaign05_apit")
    kw = _kw()
    br = _branch(tmp_path, H, kw)
    paths = {}
    for c in DELEGATION_COSTS:
        paths[c] = tmp_path / f"controller-c{c}.json"
        T.main(["train", "--branch", str(br), "--output", str(paths[c]), "--dev", "--folds", "2",
                "--delegation-cost", str(c)])
        art = json.loads(paths[c].read_text())
        assert art["controller"]["delegation_cost"] == c and art["controller_sha256"] == canonical_hash(art["controller"])
        assert art["controller"]["feature_names"] == feature_names()      # public features only (same definition)
        assert art["selection"]["chosen"]["margin"] in T.M_GRID and not art["protocol"]
        assert art["delegation_cost"]["label_adjustment_exact"]
    cc = [f"{c}={p}" for c, p in paths.items()]
    for cond in ("iid_f0", "iid_f2", "foreign4"):
        T.main(["eval", "--base", "tiny", "--condition", cond, "--worlds", "dev", "--chunk", "0", "--chunk-size", "6",
                "--cost-controllers", *cc, "--teacher", "yes", "--regret-p", "0.5", "--regret-check",
                "--output", str(tmp_path / "eval")], **kw)
    # A-PI-T eval on the same dev worlds: cost-free comparators identical to the cost run's gross rows
    ctrl0 = tmp_path / "free.json"
    T.main(["train", "--branch", str(br), "--output", str(ctrl0), "--dev", "--folds", "2"])
    T.main(["eval", "--base", "tiny", "--condition", "iid_f0", "--worlds", "dev", "--chunk", "0", "--chunk-size", "6",
            "--controller", str(ctrl0), "--teacher", "yes", "--regret-p", "0", "--output", str(tmp_path / "evt")], **kw)
    free = json.loads(gzip.open(next((tmp_path / "evt").glob("apit-*iid_f0*.json.gz")), "rt").read())["rows"]
    meta = json.loads(next((tmp_path / "eval").glob("apic-*iid_f0*.meta.json")).read_text())
    assert meta["delegation_costs"] == list(DELEGATION_COSTS) and not meta["protocol_worlds"]
    assert meta.get("regret_check_all_equal", True)
    data = json.loads(gzip.open(next((tmp_path / "eval").glob("apic-*iid_f0*.json.gz")), "rt").read())
    for c in DELEGATION_COSTS:
        rows = [r for r in data["rows"] if r["delegation_cost"] == c]
        assert {r["policy"] for r in rows} == {"d", "pit", "r1", "r2", "random", "teacher"}
        for r in rows:
            assert r["delegation_charge"] == c * r["teacher_steps"] and r["utility"] == r["utility_gross"] - r["delegation_charge"]
            if r["policy"] == "d":
                assert r["delegation_charge"] == 0
            if r["policy"] == "teacher":
                assert r["teacher_steps"] == r["decisions"]
        for pol in ("d", "r1", "r2", "teacher"):
            a = sorted((_strip(r) | {"utility": r["utility_gross"]} for r in rows if r["policy"] == pol),
                       key=lambda r: r["seed"])
            b = sorted((r for r in free if r["policy"] == pol), key=lambda r: r["seed"])
            assert a == b
    out = tmp_path / "score.json"
    T.main(["score", "--cost", "--eval", str(tmp_path / "eval"), "--dev", "--output", str(out)])
    rep = json.loads(out.read_text())
    assert set(rep["costs"]) == {str(c) for c in DELEGATION_COSTS}
    for c in DELEGATION_COSTS:
        L = rep["costs"][str(c)]["lineages"]["tiny"]
        assert "PI-C-1" in L and L["PI-C-2"]["pass"] is not None
        assert set(L["PI-C-2"]["comparators_utility"]) == {"d", "teacher", "r1", "r2", "random"}
        assert set(L["conditions"]) == {"iid_f0", "iid_f2", "foreign4"}
        ps = L["iid_group"]["policies"]
        assert ps["teacher"]["delegated_step_fraction"] == 1.0 and ps["d"]["delegation_charge"] == 0
        assert ps["teacher"]["delegation_charge"] == pytest.approx(c * ps["teacher"]["teacher_steps_per_episode"])
    # the A-PI-T scorer never reads A-PI-C files
    T.main(["score", "--eval", str(tmp_path / "eval"), "--dev", "--output", str(tmp_path / "s2.json")])
    assert json.loads((tmp_path / "s2.json").read_text())["lineages"] == {}
    np.testing.assert_equal(len(T._read_eval([tmp_path / "eval"])), 0)
