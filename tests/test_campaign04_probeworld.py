"""Probeworld (extended-04 Track B): exact DP labels, visible-history labels, semantics, splits, determinism.

Pure Python except the last (training smoke) test, which is skipped without torch."""
import itertools
import math
import random
from functools import lru_cache

import pytest

from tensegra import campaign04_probeworld as pw
from tensegra.campaign04_probeworld import (A_ABSTAIN, A_B1, A_B2, A_BUILD, A_COMMIT, A_COMMIT_INF, A_INSPECT,
                                           A_PROBE, A_PROP, A_USE, TF, TH, TM, TX, Config, ExactSolver)


def only(*acts):
    en = [False] * pw.N_ACTIONS
    for a in acts:
        en[a] = True
    return tuple(en)


def s0(cfg):
    return pw.initial_state(cfg)


# ------------------------------------------------------------------ worked example (21 vs 100)

def test_worked_example_21_vs_100():
    cfg = pw.worked_example_config()
    s = ExactSolver(cfg)
    q = s.q_values(s0(cfg))
    # expected cost: probe-first = 1 + 0.2 * 100 = 21 ; exact directly = 100 ; both succeed surely
    assert math.isclose(cfg.R - q[A_PROBE], 21.0, abs_tol=1e-9)
    assert math.isclose(cfg.R - q[A_B1], 100.0, abs_tol=1e-9)
    assert s.opt_set(s0(cfg)) == {A_PROBE}
    # the rational probe fails 20% of the time: that step is case (b), not (a)
    st = s0(cfg)
    nxt = pw.advance(cfg, st, A_PROBE, pw.O_FAILED, False, None)
    assert pw.case_type(s, st, A_PROBE, pw.O_FAILED, False, nxt) == "b"
    assert pw.case_type(s, st, A_B1, pw.O_SOLVED, False, pw.advance(cfg, st, A_B1, pw.O_SOLVED, False, None)) == "a"
    ok = pw.advance(cfg, st, A_PROBE, pw.O_SOLVED, False, None)
    assert pw.case_type(s, st, A_PROBE, pw.O_SOLVED, False, ok) == "d"


def test_worked_example_assumption_violations():
    rows = {r["variant"]: r for r in pw.worked_example_table()}
    assert rows["base (reliable detection)"]["opt_set"] == ["probe"]
    # unreliable detection with a large error loss: probing no longer pays (1 + .2*100 ... vs committing risk)
    assert rows["unreliable detection q=0.5, L=1000"]["opt_set"] == ["exact_b1"]
    # irreversible side effect: probe-first costs 1 + .2*(100 + D); flips at D = 395
    assert math.isclose(rows["probe side effect D=50"]["Q_probe_first"], 200 - (1 + .2 * (100 + 50)), abs_tol=1e-9)
    assert rows["probe side effect D=500"]["opt_set"] == ["exact_b1"]
    assert math.isclose(rows["heuristic success 30%"]["Q_probe_first"], 200 - (1 + .7 * 100), abs_tol=1e-9)


# ------------------------------------------------------------------ brute force over hidden worlds

def brute_force_value(cfg):
    """Independent expectimax over raw visible histories; beliefs by explicit enumeration of hidden worlds
    (z, theta_1..theta_k).  Only the action-availability rule is shared with the module."""
    zs = [(0.5, 0), (0.5, 1)] if cfg.corr > 0 else [(1.0, None)]

    def theta_p(z):
        p = list(cfg.prior)
        if z is None:
            return p
        ph = min(max(p[TH] + (cfg.corr if z == 0 else -cfg.corr), 0.0), 1.0)
        return [ph] + [(1 - ph) * p[t] / (1 - p[TH]) for t in (TM, TF, TX)]

    worlds = []
    for pz, z in zs:
        tp = theta_p(z)
        for ths in itertools.product(range(4), repeat=cfg.k):
            w = pz * math.prod(tp[t] for t in ths)
            if w > 0:
                worlds.append((w, ths))

    def query_ctx(h):
        """query index, reduced flag, event-fired flag from the raw history (independent scan)."""
        i, red, ev = 0, False, False
        for a, o, e, rev in h:
            if a in pw.TERMINAL:
                i, red, ev = i + 1, False, False
                continue
            if a == A_PROP and o == pw.O_REDUCED:
                red = True
            if e:
                red, ev = False, True
        return i, red, ev

    def cand(h):
        c, valid = 0, False
        for a, o, e, rev in h:
            if a in pw.TERMINAL:
                c, valid = 0, False
                continue
            if a in (A_B1, A_B2, A_USE) and o == pw.O_SOLVED:
                c, valid = 2, True
            elif a == A_PROBE and o == pw.O_SOLVED and not (c == 2 and valid):
                c, valid = 1, True
            if e:
                valid = False
        return c, valid

    def lik(h, ths):
        """P(visible outcomes of h | hidden world), event draws excluded (same for every world)."""
        L = 1.0
        i, red = 0, False
        for a, o, e, rev in h:
            th = ths[i]
            if a in pw.TERMINAL:
                if rev != th:
                    return 0.0
                i, red = i + 1, False
                continue
            if a == A_PROBE:
                p = (1.0 if o == pw.O_SOLVED else 0.0) if th == TH else (cfg.q if o == pw.O_FAILED else 1 - cfg.q)
            elif a == A_B1:
                okk = th in (TH, TM) or (th == TF and red)
                p = float(o == (pw.O_SOLVED if okk else pw.O_TIMEOUT))
            elif a in (A_B2, A_USE):
                p = float(o == (pw.O_INFEASIBLE if th == TX else pw.O_SOLVED))
            elif a == A_INSPECT:
                easy = th in (TH, TM)
                p = (1 - cfg.eta) if (o == pw.O_EASY) == easy else cfg.eta
            elif a == A_PROP:
                p = (cfg.p_conflict if o == pw.O_CONFLICT else 1 - cfg.p_conflict) if th == TX else float(o == pw.O_REDUCED)
            else:
                p = 1.0
            L *= p
            if a == A_PROP and o == pw.O_REDUCED:
                red = True
            if e:
                red = False
            if L == 0:
                return 0.0
        return L

    costs = {A_PROBE: cfg.c_probe, A_B1: cfg.c_b1, A_B2: cfg.c_b2, A_INSPECT: cfg.c_inspect, A_PROP: cfg.c_prop,
             A_BUILD: cfg.C_build, A_USE: cfg.C_execute + cfg.C_return + cfg.C_verify}
    outs = {A_PROBE: (pw.O_SOLVED, pw.O_FAILED), A_B1: (pw.O_SOLVED, pw.O_TIMEOUT),
            A_B2: (pw.O_SOLVED, pw.O_INFEASIBLE), A_USE: (pw.O_SOLVED, pw.O_INFEASIBLE),
            A_INSPECT: (pw.O_EASY, pw.O_HARD), A_PROP: (pw.O_CONFLICT, pw.O_REDUCED), A_BUILD: (pw.O_BUILT,)}

    @lru_cache(maxsize=None)
    def V(h):
        i, red, ev = query_ctx(h)
        if i >= cfg.k:
            return 0.0
        post = [(w * lik(h, ths), ths) for w, ths in worlds]
        Z = sum(x for x, _ in post)
        post = [(x / Z, ths) for x, ths in post if x > 0]
        acts = pw.available(cfg, pw.public_state_from_history(cfg, h))
        best = -1e18
        for a in acts:
            qa = 0.0
            if a in pw.TERMINAL:
                c, valid = cand(h)
                for pwld, ths in post:
                    th = ths[i]
                    if a == A_ABSTAIN:
                        r, o = 0.0, pw.O_ABSTAINED
                    else:
                        ok = (th == TX) if a == A_COMMIT_INF else (valid and (c == 2 or th == TH))
                        r, o = (cfg.R, pw.O_CORRECT) if ok else (-cfg.L, pw.O_WRONG)
                    qa += pwld * (r + V(h + ((a, o, False, th),)))
            else:
                pe = cfg.p_event if (cfg.p_event > 0 and not ev) else 0.0
                for pwld, ths in post:
                    th = ths[i]
                    r = -costs[a] - (cfg.D_side if (a == A_PROBE and th != TH) else 0.0)
                    for o in outs[a]:
                        po = lik(h + ((a, o, False, None),), ths) / lik(h, ths)
                        if po <= 0:
                            continue
                        for e, pev in ((False, 1 - pe), (True, pe)):
                            if pev > 0:
                                qa += pwld * po * pev * (r + V(h + ((a, o, e, None),)))
            best = max(best, qa)
        return best

    return V(())


BF_CASES = [
    dict(prior=(0.5, 0.2, 0.2, 0.1), enable=only(A_PROBE, A_B1, A_B2, A_COMMIT, A_COMMIT_INF, A_ABSTAIN)),
    dict(prior=(0.4, 0.1, 0.3, 0.2), q=0.6, L=150.0, enable=only(A_PROBE, A_B1, A_B2, A_COMMIT, A_COMMIT_INF, A_ABSTAIN)),
    dict(prior=(0.3, 0.2, 0.3, 0.2), eta=0.25, p_conflict=0.6, c_b2=70.0,
         enable=only(A_INSPECT, A_PROP, A_B1, A_COMMIT, A_COMMIT_INF, A_ABSTAIN)),
    dict(prior=(0.6, 0.2, 0.1, 0.1), p_event=0.3, enable=only(A_PROBE, A_B1, A_COMMIT, A_COMMIT_INF, A_ABSTAIN)),
    dict(prior=(0.6, 0.2, 0.1, 0.1), D_side=30.0, enable=only(A_PROBE, A_B1, A_B2, A_COMMIT, A_COMMIT_INF, A_ABSTAIN)),
    dict(prior=(0.5, 0.3, 0.1, 0.1), k=2, C_build=30.0, c_b2=40.0,
         enable=only(A_PROBE, A_B2, A_BUILD, A_USE, A_COMMIT, A_COMMIT_INF, A_ABSTAIN)),
    dict(prior=(0.5, 0.3, 0.1, 0.1), k=3, corr=0.3, enable=only(A_PROBE, A_B1, A_COMMIT, A_ABSTAIN)),
]


@pytest.mark.parametrize("kw", BF_CASES)
def test_dp_matches_brute_force(kw):
    cfg = Config(**kw)
    dp = ExactSolver(cfg).value(s0(cfg))
    bf = brute_force_value(cfg)
    assert math.isclose(dp, bf, rel_tol=1e-9, abs_tol=1e-7), (dp, bf)


def test_monte_carlo_pi_star_matches_v_star():
    for idx in range(3):
        cfg = pw.split_config("dev", idx)
        s = ExactSolver(cfg)
        v = s.value(s0(cfg))
        us = [pw.run_policy(cfg, 10_000 + w, lambda st, av: s.pi_star(st)).utility for w in range(3000)]
        m = sum(us) / len(us)
        se = math.sqrt(sum((u - m) ** 2 for u in us) / (len(us) - 1) / len(us))
        assert abs(m - v) < 4.5 * se + 1e-6, (idx, m, v, se)


# ------------------------------------------------------------------ labels are a function of visible history

def random_policy(rng):
    return lambda st, av: rng.choice(av)


def test_labels_function_of_visible_history():
    seen = {}
    rng = random.Random(3)
    for split in ("train", "heldout_comp"):
        for idx in range(6):
            cfg = pw.split_config(split, idx)
            s = ExactSolver(cfg)
            for w in range(60):
                ep = pw.Episode(cfg, 900 + w)
                while not ep.done:
                    h = (split, idx, tuple(ep.history))
                    assert pw.public_state_from_history(cfg, ep.history) == ep.state
                    lab = s.labels(ep.state)
                    key = (lab["V"], tuple(sorted(lab["Q"].items())), lab["opt_set"])
                    if h in seen:
                        assert seen[h] == key
                    seen[h] = key
                    ep.step(rng.choice(ep.available()))
    assert len(seen) > 500


def test_identical_visible_histories_different_hidden_types_same_labels():
    # unreliable detection: probe says "solved" on H (true) and on M (undetected failure)
    cfg = Config(prior=(0.5, 0.5, 0, 0), q=0.4, c_b1=30.0, enable=only(A_PROBE, A_B1, A_COMMIT, A_ABSTAIN))
    s = ExactSolver(cfg)
    found = {}
    for seed in range(400):
        ep = pw.Episode(cfg, seed)
        ep.step(A_PROBE)
        if ep.history[0][1] == pw.O_SOLVED:
            found.setdefault(ep.thetas[0], ep)
    assert set(found) == {TH, TM}
    a, b = found[TH], found[TM]
    assert a.history == b.history and a.state == b.state
    assert s.labels(a.state) == s.labels(b.state)
    # and the belief is the Bayes posterior: P(H | solved) = .5 / (.5 + .5 * .6)
    assert math.isclose(a.state[1][0][TH], 0.5 / 0.8, abs_tol=1e-9)


# ------------------------------------------------------------------ semantics

def test_timeout_is_unknown_not_infeasible():
    cfg = Config(prior=(0.25, 0.25, 0.25, 0.25), c_b1=10.0, c_b2=40.0, L=100.0,
                 enable=only(A_B1, A_B2, A_COMMIT, A_COMMIT_INF, A_ABSTAIN))
    s = ExactSolver(cfg)
    st = s0(cfg)
    to = pw.advance(cfg, st, A_B1, pw.O_TIMEOUT, False, None)
    b = to[1][0]
    assert b[TF] == pytest.approx(0.5) and b[TX] == pytest.approx(0.5)
    assert A_COMMIT not in pw.available(cfg, to)  # a timeout yields no candidate
    q_to = s.q_values(to)
    assert q_to[A_COMMIT_INF] == pytest.approx(0.5 * cfg.R - 0.5 * cfg.L)
    inf = pw.advance(cfg, to, A_B2, pw.O_INFEASIBLE, False, None)
    assert inf[1][0][TX] == pytest.approx(1.0)
    assert s.q_values(inf)[A_COMMIT_INF] == pytest.approx(cfg.R)
    assert s.value(to) != pytest.approx(s.value(inf))
    # simulator: X and F both time out at b1
    for th in (TF, TX):
        assert pw.outcome_dist(cfg, A_B1, th, False) == [(1.0, pw.O_TIMEOUT)]
    assert pw.outcome_dist(cfg, A_B1, TF, True) == [(1.0, pw.O_SOLVED)]  # propagation-reduced F


def test_commit_irreversible_and_wrong_answer_costs_L():
    cfg = Config(prior=(0.0, 0.0, 0.0, 1.0), k=2, L=77.0, enable=only(A_PROBE, A_COMMIT, A_COMMIT_INF, A_ABSTAIN),
                 q=0.0)  # q=0: every probe failure is undetected -> wrong candidate
    ep = pw.Episode(cfg, 1)
    ep.step(A_PROBE)
    assert ep.history[-1][1] == pw.O_SOLVED
    ep.step(A_COMMIT)
    assert ep.history[-1][1] == pw.O_WRONG and ep.history[-1][3] == TX
    assert ep.query == 1 and ep.state[1] == pw.local0(cfg)  # moved on; the query cannot be revisited
    assert A_COMMIT not in ep.available()  # no candidate carried over
    assert [x for x in ep.ledger if x[1] == "error_loss"] == [(1, "error_loss", 77.0)]
    ep.step(A_COMMIT_INF)
    assert ep.done and ep.successes == 1
    with pytest.raises(ValueError):
        ep.step(A_ABSTAIN)
    assert ep.utility == pytest.approx(cfg.R - 77.0 - cfg.c_probe)


def test_event_invalidates_declared_subset():
    cfg = Config(prior=(0.5, 0.5, 0, 0), p_event=0.5, enable=only(A_PROBE, A_B1, A_INSPECT, A_PROP, A_COMMIT, A_ABSTAIN))
    st = s0(cfg)
    st = pw.advance(cfg, st, A_INSPECT, pw.O_EASY, False, None)
    st = pw.advance(cfg, st, A_PROP, pw.O_REDUCED, False, None)
    st = pw.advance(cfg, st, A_B1, pw.O_SOLVED, True, None)  # event after the exact solve
    (_, _), (b, usage, reduced, cand, valid, ev) = st[0][:2], st[1]
    assert cand == pw.C_EXACT and not valid and not reduced and ev
    assert usage & pw.U_INSPECT and not usage & (pw.U_B1 | pw.U_PROP)  # inspect info stays, solvers re-runnable
    s = ExactSolver(cfg)
    assert s.q_values(st)[A_COMMIT] == pytest.approx(-cfg.L)  # stale candidate is wrong
    # an eps-optimal, non-failing step followed by an event is case (c)
    root = s0(cfg)
    a_star = s.pi_star(root)
    o = next(o for p, o, e, _, _, _ in s.transitions(root, a_star) if not e and (a_star, o) not in pw.FAILURE_OUTCOMES)
    nxt = pw.advance(cfg, root, a_star, o, True, None)
    assert pw.case_type(s, root, a_star, o, True, nxt) == "c"


def test_each_cost_charged_once():
    rng = random.Random(5)
    for idx in range(8):
        cfg = pw.split_config("train", idx)
        cfg = pw.with_(cfg, C_build=5.0, D_side=12.0)
        for w in range(40):
            ep = pw.Episode(cfg, w)
            while not ep.done:
                ep.step(rng.choice(ep.available()))
            per_step = {}
            for t, item, amt in ep.ledger:
                assert (t, item) not in per_step, "duplicate charge"
                per_step[(t, item)] = amt
            expect = 0.0
            for t, (a, o, e, rev) in enumerate(ep.history):
                q = sum(1 for tt, (aa, *_ ) in enumerate(ep.history[:t]) if aa in pw.TERMINAL)
                expect += pw.action_cost(cfg, a)
                if a == A_PROBE and ep.thetas[q] != TH:
                    expect += cfg.D_side
                if o == pw.O_WRONG:
                    expect += cfg.L
                if a == A_USE:
                    assert {k for (tt, k) in per_step if tt == t} == {"C_execute", "C_return", "C_verify"}
            assert ep.total_cost == pytest.approx(expect)
            assert ep.utility == pytest.approx(cfg.R * ep.successes - expect)
            assert sum(1 for x in ep.history if x[0] == A_BUILD) <= 1


# ------------------------------------------------------------------ splits and generator

def test_split_disjointness_by_generator_parameters():
    fam_params = {}
    for name, (cells, ks, combos, _) in pw.SPLITS.items():
        fam = "iid" if name in ("train", "dev", "test_iid") else name
        fam_params.setdefault(fam, set()).update(itertools.product(cells, ks, combos))
    fams = list(fam_params)
    for x, y in itertools.combinations(fams, 2):
        assert not (fam_params[x] & fam_params[y]), (x, y)
    assert 4 not in pw.SPLITS["train"][1] and pw.SPLITS["heldout_k"][1] == (4,)
    assert not set(pw.HELDOUT_COMBOS) & set(pw.SPLITS["train"][2])
    # held-out combos are compositions of flags each seen singly in training
    for combo in pw.HELDOUT_COMBOS:
        for j, f in enumerate(combo):
            if f:
                single = tuple(jj == j for jj in range(4))
                assert single in pw.SPLITS["train"][2]
    # generated configs land in their split's parameter set (recovered from public prices)
    seeds = {}
    for name in pw.SPLITS:
        for idx in range(80):
            cell, k, combo, seed = pw.generator_params(name, idx)
            cfg = pw.split_config(name, idx)
            assert pw.price_cell(cfg) == tuple(cell) and cfg.k == k and cfg.flags == tuple(combo)
            fam = "iid" if name in ("train", "dev", "test_iid") else name
            assert pw.split_of_params(cell, k, combo) == {fam}
            assert seed not in seeds
            seeds[seed] = name
    # world seeds of eval splits never collide with each other
    ws = [pw.world_seed(n, i, r) for n in pw.SPLITS for i in range(200) for r in (0, 500, 503)]
    assert len(ws) == len(set(ws))


def test_splits_never_defined_by_case_type():
    import inspect
    src = inspect.getsource(pw.generator_params) + inspect.getsource(pw.split_config) + inspect.getsource(pw.make_config)
    assert "case" not in src


def test_direct_success_in_every_split():
    for split in pw.SPLITS:
        n_direct = n = 0
        for idx in range(10):
            cfg = pw.split_config(split, idx)
            s = ExactSolver(cfg)
            for w in range(10):
                ep = pw.Episode(cfg, pw.world_seed(split, idx, w))
                cases = []
                while not ep.done:
                    st = ep.state
                    a = s.pi_star(st)
                    r = ep.step(a)
                    cases.append(pw.case_type(s, st, a, r[1], r[2], ep.state))
                n += 1
                n_direct += all(c == "d" for c in cases) and ep.successes == cfg.k
        assert n_direct > 0.1 * n, (split, n_direct, n)


def test_rho_k_and_k_change_optimal_strategy():
    base = Config(prior=FEATURE_PRIOR_MID, c_probe=3.0, c_b1=20.0, c_b2=45.0, C_execute=3.0, C_return=1.0,
                  C_verify=1.0, L=150.0)
    Cs = (10.0, 20.0, 60.0, 120.0, 240.0)
    pb = {}
    for k in (1, 2, 4, 8):
        for C_build in Cs:
            cfg = pw.with_(base, k=k, C_build=C_build)
            pb[(k, C_build)] = ExactSolver(cfg).pi_star_stats()["p_build"]
    # the optimal strategy changes with reuse horizon and build price
    assert pb[(8, 60.0)] == 1.0 and pb[(1, 60.0)] == 0.0
    assert pb[(1, 10.0)] > 0 and pb[(1, 20.0)] == 0.0 and pb[(8, 120.0)] == 0.0
    for k in (1, 2, 4, 8):
        seq = [pb[(k, c)] for c in Cs]
        assert seq == sorted(seq, reverse=True)  # decreasing in C_build (increasing in rho_k)
    for c in Cs:
        seq = [pb[(k, c)] for k in (1, 2, 4, 8)]
        assert seq == sorted(seq)  # increasing in k
    # the switch sits near effective rho = 1 (C_shortcut_eff = no-structure pi* cost per query)
    for (k, c), p in pb.items():
        r = pw.effective_rho(pw.with_(base, k=k, C_build=c))
        if r > 1.1:
            assert p > 0.5
        if r < 0.9:
            assert p < 0.5


FEATURE_PRIOR_MID = pw.FEATURE_PRIORS[(1, 1)]


# ------------------------------------------------------------------ determinism

def test_determinism():
    for split in ("train", "heldout_k"):
        assert pw.split_config(split, 3) == pw.split_config(split, 3)
    cfg = pw.split_config("train", 3)
    v1 = ExactSolver(cfg).value(s0(cfg))
    v2 = ExactSolver(cfg).value(s0(cfg))
    assert v1 == v2
    s = ExactSolver(cfg)
    h1 = pw.run_policy(cfg, 42, lambda st, av: s.pi_star(st)).history
    h2 = pw.run_policy(cfg, 42, lambda st, av: s.pi_star(st)).history
    assert h1 == h2


def test_training_smoke_if_torch(tmp_path):
    torch = pytest.importorskip("torch")
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "research/tools/campaign04_probeworld_train.py"
    spec = importlib.util.spec_from_file_location("pwtrain", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    lab = tmp_path / "labels"
    m.main(["labels", "--out", str(lab), "--n-train", "6", "--n-eval", "3"])
    for rung in ("L0", "L4"):
        run = tmp_path / rung
        m.main(["train", "--labels", str(lab), "--out", str(run), "--rung", rung, "--seed", "0", "--updates", "3",
                "--batch", "8", "--hidden", "32", "--log-every", "1"])
        m.main(["eval", "--labels", str(lab), "--run", str(run), "--worlds", "1", "--splits", "dev", "heldout_k"])
    # matched exposure: identical config/world stream and identical initial parameters across rungs
    import json
    a = json.loads((tmp_path / "L0" / "train_meta.json").read_text())
    b = json.loads((tmp_path / "L4" / "train_meta.json").read_text())
    assert a["params"] == b["params"] and a["episodes"] == b["episodes"]
    # model inputs contain no privileged label: identical visible histories -> identical inputs
    cfg = pw.split_config("dev", 0)
    x1 = m.encode(cfg.public_vector(), (A_PROBE, pw.O_SOLVED, False, None), (A_COMMIT, A_ABSTAIN), 0.0)
    x2 = m.encode(cfg.public_vector(), (A_PROBE, pw.O_SOLVED, False, None), (A_COMMIT, A_ABSTAIN), 0.0)
    assert x1 == x2 and len(x1) == m.IN_DIM
