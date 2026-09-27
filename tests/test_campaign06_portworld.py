"""Contracts of the extended-06 portfolio world (Track A)."""
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from tensegra import campaign06_portworld as pw

REPO = Path(__file__).resolve().parents[1]
DEV = 2_300_000_000   # ext06 dev/smoke range (tests only)


def small_instances(count=12, n_range=(6, 12), version=None, offset=0):
    out = []
    for s in range(DEV + offset, DEV + offset + count):
        kn = {"n_range": n_range, "possible_zero": 0.0, "hidden_cap_zero": 0.0}
        if version == "pw-v3":
            kn["n_max_by_class"] = {c: 99 for c in range(5)}
        insts, _ = pw.generate_episode(s, kn, version)
        out.append(insts[1])
    return out


def brute(inst=None, plan=None):
    """Brute-force optimum (value) of the TRUE instance or of a plan."""
    n = inst.n if inst is not None else plan.n
    best = 0
    for sel in range(1 << n):
        if inst is not None:
            ok, v = pw.check_solution(inst, sel)
        else:
            ok, v = plan.feasible(sel), plan.value(sel)
        if ok and v > best:
            best = v
    return best


# --- seed ranges -------------------------------------------------------------------

def test_seed_ranges_disjoint_and_registered():
    assert pw.check_seed_ranges()
    reg = json.loads((REPO / "research/campaigns/extended-06/seed-ranges.json").read_text())["ranges"]
    ours = [r for r in reg if r["name"].startswith("ext06 Track A portworld")]
    assert {(r["lo"], r["hi"]) for r in ours} == set(pw.SEED_RANGES.values())
    others = [r for r in reg if not r["name"].startswith("ext06 Track A") and r["name"] != "ext06 Track A portfolio-world instances (dev/train/select/confirm sub-ranges assigned in design)"]
    for a in ours:
        for b in others:
            assert a["hi"] <= b["lo"] or b["hi"] <= a["lo"], (a["name"], b["name"])
    with pytest.raises(AssertionError):
        pw.check_seed_ranges({"a": (300_000_000, 301_000_000), "b": (300_500_000, 302_000_000)})


# --- exact checker ---------------------------------------------------------------------

def _toy():
    # 4 items, 2 groups; item 3 requires item 2; (0,1) known conflict; (2,3) possible
    return pw.Instance(n=4, G=2, v=[5, 4, 3, 6], grp=[0, 0, 1, 1], wl=[3, 3, 2, 2], wg=[1, 1, 1, 1],
                       capl_true=[6, 3], cap_lo=[6, 2], cap_hi=[6, 5], hidden_cap=[False, True], capg=10,
                       conflicts=[(0, 1)], possible=[(0, 2)], possible_real=[True], parent=[-1, -1, -1, 2],
                       cache=0, prices={"c": 1e-6, "o": 0.01, "L": 1.0}, meta={"t": 0, "episode_seed": 0, "m": 0.0})


def test_checker():
    inst = _toy()
    assert pw.check_solution(inst, 0b0001) == (True, 5)
    assert pw.check_solution(inst, 0b0011)[0] is False           # known conflict
    assert pw.check_solution(inst, 0b0101)[0] is False           # realized possible conflict
    assert pw.check_solution(inst, 0b1000)[0] is False           # requires violated
    assert pw.check_solution(inst, 0b1100)[0] is False           # group-1 cap 3 < 4
    inst.capl_true[1] = 4
    assert pw.check_solution(inst, 0b1100) == (True, 9)
    inst.possible_real[0] = False
    assert pw.check_solution(inst, 0b0101) == (True, 8)
    inst.capg = 1
    assert pw.check_solution(inst, 0b1100)[0] is False           # global cap


def test_conservative_plan_is_sound():
    for inst in small_instances(8, (6, 10)):
        pv = pw.public_view(inst)
        P = pw.plan_from_public(pv, "cons")
        for sel in range(1 << inst.n):
            if P.feasible(sel):
                assert pw.check_solution(inst, sel)[0]


# --- optimality of B&B / PD / exact against brute force -----------------------------------

@pytest.mark.parametrize("version", ["pw-v1", "pw-v2", "pw-v3"])
def test_bnb_pd_exact_match_brute_force(version):
    for inst in small_instances(14, (6, 12), version) + small_instances(3, (15, 16), version, offset=50):
        opt = brute(inst)
        sel, val, _, cert, ub = pw.exact_optimum(inst)
        assert cert and ub == val == opt and pw.check_solution(inst, sel) == (True, opt)
        P = pw.true_plan(inst)
        bb = pw.BnB(P, 0)
        bb.run(10 ** 9)
        assert bb.finished and bb.inc_val == opt
        w = pw.Work()
        pd = pw.PD(P, pw.greedy(P, w), 0)
        pd.run(10 ** 9)
        s2, v2, fin, _, ub2 = pd.at_budget(10 ** 9)
        assert ub2 == v2
        assert fin and v2 == opt and P.feasible(s2)
        # conservative plan optimum
        C = pw.plan_from_public(pw.public_view(inst), "cons")
        bc = pw.BnB(C, 0); bc.run(10 ** 9)
        assert bc.inc_val == brute(plan=C)


def test_propagation_sound():
    for inst in small_instances(14, (6, 12)):
        P = pw.true_plan(inst)
        w = pw.Work()
        g = pw.greedy(P, w)
        fin, fout = pw.propagate(P, P.value(g), w)
        opt = brute(plan=P)
        if opt > P.value(g):   # every strictly better solution respects the fixings
            best_fix = max((P.value(s) for s in range(1 << P.n) if P.feasible(s) and s & fin == fin and not s & fout), default=0)
            assert best_fix == opt


# --- resumability and determinism --------------------------------------------------------

def _mid_instance(seed=DEV + 100):
    insts, _ = pw.generate_episode(seed, {"n_range": (30, 40), "corr_range": (0.8, 1.0)})
    return insts[1]


def test_bnb_resumable_and_at_budget():
    P = pw.true_plan(_mid_instance())
    w = pw.Work(); g = pw.greedy(P, w)
    a = pw.BnB(P, g); a.run(3000); a.run(20000); a.run(60000)
    b = pw.BnB(P, g); b.run(60000)
    assert (a.inc, a.inc_val, a.log, a.work.units, a.finished, a.stack, a.sel) == (b.inc, b.inc_val, b.log, b.work.units, b.finished, b.stack, b.sel)
    for bud in (500, 3000, 7777, 20000, 60000):
        c = pw.BnB(P, g); c.run(bud)
        v, s, fin, ch = b.at_budget(bud)
        assert (v, s) == (c.inc_val, c.inc)
        assert fin == c.finished
        assert ch == (c.work.units if c.finished else bud)


def test_repair_resumable():
    P = pw.true_plan(_mid_instance())
    w = pw.Work(); g = pw.greedy(P, w)
    a = pw.Repair(P, g)
    for b in (50, 200, 1000, 10 ** 6):
        a.run(b)
    b2 = pw.Repair(P, g); b2.run(10 ** 6)
    assert (a.sel, a.work.units, a.moves, a.done) == (b2.sel, b2.work.units, b2.moves, b2.done)
    assert P.feasible(a.sel) and P.value(a.sel) >= P.value(g)


def test_pd_at_budget_equals_fresh_run():
    P = pw.true_plan(_mid_instance(DEV + 101))
    w = pw.Work(); g = pw.greedy(P, w)
    big = pw.PD(P, g, 0); big.run(200000)
    for bud in (0, 1000, 10000, 100000):
        f = pw.PD(P, g, 0); f.run(bud)
        assert big.at_budget(bud)[:4] == f.at_budget(bud)[:4]


def test_evaluate_deterministic_and_feasible():
    r1 = pw.evaluate_episode(DEV + 7, version="pw-v3")
    r2 = pw.evaluate_episode(DEV + 7, version="pw-v3")
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    for r in r1:
        assert set(r["arms"]) == set(pw.arm_names())
        for a, x in r["arms"].items():
            assert (x["ok"] or a.startswith("opt:")) and 0 <= x["v"]
            if x["ok"]:
                assert x["v"] <= r["opt"]
            assert all(isinstance(u, int) and u >= 0 for u in x["st"].values())


# --- utility accounting ----------------------------------------------------------------------

def test_utility_accounting():
    rec = {"opt": 100, "prices": {"c": 1e-3, "o": 0.01, "L": 2.0}, "arms": {
        "cons:GR": {"v": 90, "ok": True, "st": {"cons:G": 10, "cons:R": 20}, "obs": 0},
        "cons:BB1000": {"v": 95, "ok": True, "st": {"cons:G": 10, "cons:R": 20, "cons:BB": 1000}, "obs": 0},
        "cons:BB10000": {"v": 99, "ok": True, "st": {"cons:G": 10, "cons:R": 20, "cons:BB": 4000}, "obs": 0},
        "insp:GR": {"v": 97, "ok": True, "st": {"insp:G": 10, "insp:R": 20}, "obs": 5},
        "insp:PD0": {"v": 98, "ok": True, "st": {"insp:G": 10, "insp:R": 20, "insp:PDprop": 70}, "obs": 5},
        "bad": {"v": 0, "ok": False, "st": {}, "obs": 0},
        "empty": {"v": 0, "ok": True, "st": {"cons:G": 1}, "obs": 0}}}
    assert pw.utility(rec, "cons:GR") == pytest.approx(0.90 - 1e-3 * 30)
    # resumed B&B: shared G/R and the B&B stage charged once at its max
    assert pw.utility(rec, ["cons:BB1000", "cons:BB10000"]) == pytest.approx(0.99 - 1e-3 * 4030)
    # switch modes: both computations charged, observations once, best output committed
    assert pw.utility(rec, ["cons:GR", "insp:GR", "insp:PD0"]) == pytest.approx(0.98 - 1e-3 * (30 + 100) - 0.01 * 5)
    assert pw.utility(rec, "bad") == pytest.approx(-2.0)
    assert pw.utility(rec, "empty") == pytest.approx(-2.0 - 1e-3)
    assert pw.total_work(rec, ["cons:BB1000", "cons:BB10000"]) == 4030


# --- public information ---------------------------------------------------------------------

def test_public_view_has_no_hidden_fields():
    names = {f.name for f in dataclasses.fields(pw.PublicView)}
    assert "capl_true" not in names and "possible_real" not in names


def test_public_features_ignore_hidden_truth():
    insts, _ = pw.generate_episode(DEV + 3, {"hidden_cap_zero": 0.0, "possible_zero": 0.0, "hidden_cap_range": (1.0, 1.0)})
    inst = insts[1]
    inst.cache = 0b1011
    assert inst.n_hidden() > 0
    twin = copy.deepcopy(inst)
    for g in range(twin.G):
        if twin.hidden_cap[g]:
            twin.capl_true[g] = twin.cap_hi[g] if twin.capl_true[g] != twin.cap_hi[g] else twin.cap_lo[g]
    twin.possible_real = [not r for r in twin.possible_real]
    assert pw.public_view(inst) == pw.public_view(twin)
    assert pw.public_features(pw.public_view(inst)) == pw.public_features(pw.public_view(twin))
    # the conservative plan (what every cons-mode method sees) is identical too
    a, b = pw.plan_from_public(pw.public_view(inst), "cons"), pw.plan_from_public(pw.public_view(twin), "cons")
    assert (a.capl, a.conf) == (b.capl, b.conf)
    # inspection is the only channel to the truth, and it is counted
    ins = pw.Inspector(inst)
    pw.plan_from_public(pw.public_view(inst), "insp", ins)
    assert ins.count == inst.n_hidden()


def test_selector_inputs_identical_for_identical_public_features():
    """The headroom tool's selectors read only rec['pub'] (+ prices, which are in pub) and,
    for sequential variants, the telemetry of calls they paid for."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("c6p", REPO / "research/tools/campaign06_portfolio.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    recs = pw.evaluate_episode(DEV + 11, version="pw-v3")
    r2 = copy.deepcopy(recs)
    for r in r2:   # scramble everything that is not public
        r["opt"] += 1; r["exact_work"] = 0
        for x in r["arms"].values():
            x["v"] = 0
    X1, n1 = mod.feature_matrix(recs)
    X2, n2 = mod.feature_matrix(r2)
    assert n1 == n2 and (X1 == X2).all()
    assert all(k in recs[0]["pub"] for k in ("log_c", "log_o"))


# --- oracle cross-checks, certified bounds, traces ----------------------------------------------

def test_surrogate_oracle_matches_plain_bnb_on_medium_instances():
    """Two bound implementations (method bound vs oracle's min(method, surrogate)) agree."""
    for s in range(DEV + 200, DEV + 206):
        insts, _ = pw.generate_episode(s, {"n_range": (24, 30)}, "pw-v3")
        inst = insts[1]
        _, val, _, cert, _ = pw.exact_optimum(inst)
        P = pw.true_plan(inst)
        bb = pw.BnB(P, 0); bb.run(10 ** 9)
        assert cert and bb.finished and bb.inc_val == val


def test_certified_upper_bounds_are_valid():
    for s in range(DEV + 300, DEV + 306):
        insts, _ = pw.generate_episode(s, {"n_range": (30, 40), "corr_classes": (2, 3)}, "pw-v3")
        inst = insts[1]
        _, opt, _, cert, _ = pw.exact_optimum(inst)
        assert cert
        P = pw.true_plan(inst)
        w = pw.Work(); g = pw.greedy(P, w)
        bb = pw.BnB(P, g); bb.run(pw.TRACE_MAX)
        for b, ub in bb.ub_at.items():
            assert ub >= opt and bb.at_budget(b)[0] <= opt
        pd = pw.PD(P, g, 0); pd.run(pw.TRACE_MAX)
        for b in (0,) + pw.TRACE_GRID:
            sel, v, fin, used, ub = pd.at_budget(b)
            assert v <= opt <= ub and P.feasible(sel)
            if fin:
                assert v == opt == ub
        # uncertified oracle runs are scored against a valid upper bound
        sel, v, _, cert2, ub = pw.exact_optimum(inst, cap_units=2000)
        assert ub >= opt >= v


def test_posterior_feasibility_is_exact():
    import itertools
    insts, _ = pw.generate_episode(DEV + 400, {"hidden_cap_zero": 0.0, "possible_zero": 0.0, "hidden_cap_range": (1.0, 1.0),
                                               "n_range": (9, 9), "possible_range": (0.2, 0.25), "groups_choices": (2,),
                                               "width_range": (0.1, 0.2)}, "pw-v3")
    inst = insts[1]
    pv = pw.public_view(inst)
    P = pw.plan_from_public(pv, "opt")
    w = pw.Work(); sel = pw.greedy(P, w)
    hid = [g for g in range(inst.G) if inst.hidden_cap[g]]
    tot, ok = 0, 0
    for caps in itertools.product(*[range(inst.cap_lo[g], inst.cap_hi[g] + 1) for g in hid]):
        for real in itertools.product([False, True], repeat=len(inst.possible)):
            t = copy.deepcopy(inst)
            for g, cval in zip(hid, caps):
                t.capl_true[g] = cval
            t.possible_real = list(real)
            tot += 1
            ok += pw.check_solution(t, sel)[0]
    assert 0 < pw.posterior_feasible_prob(pv, sel) < 1 or len(inst.possible) == 0
    assert abs(ok / tot - pw.posterior_feasible_prob(pv, sel)) < 1e-9


def test_trace_arm_matches_menu_arm():
    recs = pw.evaluate_episode(DEV + 12, version="pw-v3")
    for r in recs:
        for mode in pw.MODES:
            for b in pw.BUDGETS:
                ta = pw.trace_arm(r, mode, "PD", b)
                ma = r["arms"][f"{mode}:PD{b}"]
                assert (ta["v"], ta["ok"], ta["st"], ta["obs"]) == (ma["v"], ma["ok"], ma["st"], ma["obs"])


def test_opt_mode_can_fail_and_is_charged_loss():
    recs = []
    for s in range(DEV + 500, DEV + 512):
        recs += pw.evaluate_episode(s, {"hidden_cap_zero": 0.0, "hidden_cap_range": (1.0, 1.0), "width_range": (0.4, 0.5)}, version="pw-v3")
    fails = [(r, a) for r in recs for a in r["arms"] if a.startswith("opt:") and not r["arms"][a]["ok"]]
    assert fails, "optimistic plans should sometimes be infeasible"
    r, a = fails[0]
    assert pw.utility(r, a) <= -r["prices"]["L"]


def test_selector_charges_reduce_utility():
    recs = pw.evaluate_episode(DEV + 13, version="pw-v3")
    r = recs[0]
    assert r["feature_work"] > 0
    assert pw.utility(r, "cons:GR", extra_work=1000) == pytest.approx(pw.utility(r, "cons:GR") - 1000 * r["prices"]["c"])
