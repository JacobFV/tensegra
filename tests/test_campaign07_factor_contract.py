"""Extended-07 P0a: factor contract + public-history reconstructibility of the 23 derived factor coordinates.

Pure Python (no torch).  New configurations / worlds only from the dev_smoke sub-range [6.8e9, 6.9e9)
(block 6.878e9 + i).  The trainer-level check (aux targets logged by run_batch == recomputation from the public
history) needs torch and lives in tests/test_campaign07_gradflow.py.

What is tested:
  1. research/campaigns/extended-07/factor-contract.json is in sync with the code (names, order, count 23, groups
     partition the coordinates, file == generator output).
  2. INDEPENDENT reconstruction: every coordinate is recomputed from (a) the actor's public config vector only
     (Config6.public_vector, R-free) and (b) the visible history (a, o, event, reveal) only -- with a brute-force
     Bayes posterior over (z, theta) written independently of pw.theta_prior / update_belief -- and compared with
     factor_features(cfg, episode.state) at every decision of random and pi* trajectories.
  3. Hidden-state invariance: decisions reached by the same visible history under different hidden draws (theta,
     z, event draws) have bit-identical factor vectors (non-vacuous: such groups are counted).
  4. The dependency identities of the contract hold; D/T are off in every b6/b6c family (f9 == 0, f22 == 1).
"""
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    from tensegra import campaign04_probeworld as pw
except ImportError:  # no torch locally
    import types
    for _k in [k for k in sys.modules if k == "tensegra" or k.startswith("tensegra.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("tensegra")
    _pkg.__path__ = [str(ROOT / "src" / "tensegra")]
    sys.modules["tensegra"] = _pkg
    from tensegra import campaign04_probeworld as pw
from tensegra import campaign06_probeworld as pw6

sys.path.insert(0, str(ROOT / "research" / "tools"))
import campaign07_factor_contract as fc  # noqa: E402

DEV7 = 6_878_000_000  # dev_smoke block for this test (configs DEV7 + i, worlds DEV7 + 500_000 + i)
assert pw6.SUBRANGES["dev_smoke"][0] <= DEV7 and DEV7 + 1_000_000 <= pw6.SUBRANGES["dev_smoke"][1]
COMBOS = tuple(pw6.TRAIN_TYPES) + ("SCE", "UCE", "UCE", "SCE", "T", "D", "UT", "SD", "CT", "ED")
TOL = 1e-9


# ------------------------------------------------------------------ independent public-only reconstruction

def public_view(v):
    """Everything factor_features needs, in R units, recovered from the public config vector only."""
    assert len(v) == pw6.PUBLIC_DIM6
    k = round(v[16] * 8)
    c_b2 = v[2]
    c_use = v[7] + v[8] + v[9]
    if v[6] < 1.0:  # C_build / R < 10: stored directly
        C_build = v[6] * 10.0
    else:  # clipped at 10 R in the vector: recover it from the rho entry (rho >= .09 > 1e-3 in every generator cell)
        rho = math.exp(3.0 * v[18])
        C_build = k * (c_b2 / rho - c_use)
    return {"c_probe": v[0], "c_b1": v[1], "c_b2": c_b2, "L": v[5], "C_build": C_build, "c_use": c_use,
            "D_side": v[10], "q": v[11], "corr": v[12], "p_event": v[13], "eta": v[14], "p_conflict": v[15],
            "k": k, "prior": tuple(v[23:27]), "deadline": round(v[27] * 2), "t_hard": v[28]}


def lik(P, a, o, theta, reduced):
    """P(visible outcome o | action a, type theta) -- written from the probeworld-v1 task text, not outcome_dist."""
    H, M, F, X = range(4)
    if a == pw.A_PROBE:
        if theta == H:
            return 1.0 if o == pw.O_SOLVED else 0.0
        return {pw.O_FAILED: P["q"], pw.O_SOLVED: 1.0 - P["q"]}.get(o, 0.0)
    if a == pw.A_B1:
        ok = theta in (H, M) or (theta == F and reduced)
        return 1.0 if o == (pw.O_SOLVED if ok else pw.O_TIMEOUT) else 0.0
    if a in (pw.A_B2, pw.A_USE):
        return 1.0 if o == (pw.O_INFEASIBLE if theta == X else pw.O_SOLVED) else 0.0
    if a == pw.A_INSPECT:
        easy = theta in (H, M)
        if o == pw.O_EASY:
            return 1.0 - P["eta"] if easy else P["eta"]
        return P["eta"] if easy else 1.0 - P["eta"]
    if a == pw.A_PROP:
        if theta == X:
            return {pw.O_CONFLICT: P["p_conflict"], pw.O_REDUCED: 1.0 - P["p_conflict"]}.get(o, 0.0)
        return 1.0 if o == pw.O_REDUCED else 0.0
    return 1.0  # build: no information


def factors_from_history(v, history):
    P = public_view(v)
    prior = P["prior"]
    last_term = max([n for n, r in enumerate(history) if r[0] in pw.TERMINAL], default=-1)
    reveals = [r[3] for r in history[:last_term + 1] if r[0] in pw.TERMINAL]
    cur = history[last_term + 1:]
    i = len(reveals)
    built = any(r[0] == pw.A_BUILD for r in history)

    def p_theta(ph):
        rest = 1.0 - prior[0]
        return [ph] + [(1.0 - ph) * prior[t] / rest if rest > 0 else 0.0 for t in (1, 2, 3)]
    if P["corr"] > 0:
        comps = []
        for sgn in (1.0, -1.0):
            ph = min(max(prior[0] + sgn * P["corr"], 0.0), 1.0)
            pt = p_theta(ph)
            w = 0.5
            for t in reveals:
                w *= pt[t]
            comps.append((w, pt))
    else:
        comps = [(1.0, list(prior))]
    post = [sum(w * pt[t] for w, pt in comps) for t in range(4)]
    reduced, ev, steps, cand, cval = False, False, 0, pw.C_NONE, False
    for a, o, e, _ in cur:
        post = [post[t] * lik(P, a, o, t, reduced) for t in range(4)]
        steps += 1
        if a in (pw.A_B1, pw.A_B2, pw.A_USE) and o == pw.O_SOLVED:
            cand, cval = pw.C_EXACT, True
        elif a == pw.A_PROBE and o == pw.O_SOLVED and not (cand == pw.C_EXACT and cval):
            cand, cval = pw.C_PROBE, True
        if a == pw.A_PROP and o == pw.O_REDUCED:
            reduced = True
        if e:
            cval, reduced, ev = False, False, True
    z = sum(post)
    bH, bM, bF, bX = [x / z for x in post]
    p_false = (1 - bH) * (1 - P["q"])
    p_solved = bH + p_false
    p_hs = bH / p_solved if p_solved > 0 else 0.0
    p_b1 = bH + bM + (bF if reduced else 0.0)
    hz = 0.0 if ev else P["p_event"]
    hard = P["t_hard"] * P["c_b2"] * (bF + bX)
    side = P["D_side"] * (1 - bH)
    b2 = (P["c_b2"] + hard) * (1 + hz)
    probe = P["c_probe"] + side + (1 - p_solved) * b2 + p_false * P["L"] + hz * P["c_probe"]
    b1 = P["c_b1"] + hard + (1 - p_b1) * b2
    rem = max(P["k"] - i, 1)
    use = (0.0 if built else P["C_build"] / rem) + P["c_use"]
    best_ns = min(b2, probe, b1)
    bv = 0.0 if built else rem * (best_ns - P["c_use"]) - P["C_build"]
    cl = lambda x: max(-5.0, min(5.0, x))
    ok = cand != pw.C_NONE and cval
    ex = ok and cand == pw.C_EXACT
    trust = 1.0 if ex else (p_hs if ok else 0.0)
    d = P["deadline"]
    return [bH, bM, bF, bX, bH, p_false, p_hs, p_b1, side, hard, hz, cl(b2), cl(probe), cl(b1), cl(use),
            cl(min(best_ns, use)), cl(bv), rem / 8.0, float(built), float(ok), float(ex), trust,
            (d - steps) / 2.0 if d > 0 else 1.0]


def _cfg(i):
    return pw6.stream_config(DEV7, i, pw.TRAIN_CELLS, pw6.V3_K, COMBOS)[0]


def _det_policy(history, av):
    """Deterministic function of the VISIBLE history (so equal prefixes take equal actions)."""
    h = 17
    for a, o, e, r in history:
        h = (h * 31 + a * 13 + o * 7 + int(e) * 3 + (r if r is not None else 5)) % 1_000_003
    return av[h % len(av)]


# ------------------------------------------------------------------ tests

def test_contract_json_in_sync():
    path = ROOT / "research" / "campaigns" / "extended-07" / "factor-contract.json"
    on_disk = json.loads(path.read_text())
    gen = fc.contract()
    assert on_disk == json.loads(json.dumps(gen)), "regenerate: campaign07_factor_contract.py contract --out ..."
    assert gen["n"] == pw6.N_FACTOR_FEATURES == 23
    assert [c["name"] for c in gen["coordinates"]] == list(pw6.FACTOR_FEATURES)
    idx = sorted(i for g in gen["groups"].values() for i in g)
    assert idx == list(range(23)) and len(gen["groups"]) == 4
    for c in gen["coordinates"]:  # every f-dependency points backwards or to a declared coordinate
        for d in c["depends_on"]:
            if d.startswith("f") and d[1:].isdigit():
                assert 0 <= int(d[1:]) < 23 and int(d[1:]) != c["index"]


def test_factors_reconstruct_from_public_vector_and_history():
    rng = random.Random(DEV7 + 900_000)
    n_dec, max_err, fams = 0, 0.0, set()
    for i in range(160):
        cfg = _cfg(i)
        fams.add(pw6.fam_key(pw6.flags_of(cfg)))
        solver = None
        if cfg.k <= 2 and i % 4 == 0:
            solver = pw6.ExactSolver(cfg)
            solver.value(pw.initial_state(cfg))
        for rep in range(2):
            ep = pw6.Episode(cfg, DEV7 + 500_000 + 10 * i + rep)
            v = cfg.public_vector()
            while not ep.done:
                f = pw6.factor_features(cfg, ep.state)
                g = factors_from_history(v, tuple(ep.history))
                err = max(abs(x - y) for x, y in zip(f, g))
                assert err < 1e-6, (i, rep, len(ep.history), [(n, x, y) for n, x, y in
                                                              zip(pw6.FACTOR_FEATURES, f, g) if abs(x - y) > 1e-6])
                max_err = max(max_err, err)
                assert ep.state == pw6.public_state_from_history(cfg, ep.history)
                n_dec += 1
                av = ep.available()
                ep.step(solver.pi_star(ep.state) if (solver is not None and rep == 0) else rng.choice(av))
    assert n_dec > 3000 and {"SCE", "UCE", "C", "E", "T", "D"} <= fams


def test_hidden_state_invariance():
    """Same visible history, different hidden draws -> identical factor vector (bitwise)."""
    groups_with_hidden_variation = 0
    for i in range(0, 60, 3):
        cfg = _cfg(i)
        seen = {}
        for rep in range(120):
            ep = pw6.Episode(cfg, DEV7 + 700_000 + 1000 * i + rep)
            while not ep.done:
                key = tuple(ep.history)
                hidden = (ep.thetas[ep.query], ep.z)
                f = tuple(pw6.factor_features(cfg, ep.state))
                prev = seen.get(key)
                if prev is None:
                    seen[key] = (f, {hidden})
                else:
                    assert prev[0] == f
                    if hidden not in prev[1]:
                        prev[1].add(hidden)
                        groups_with_hidden_variation += 1
                ep.step(_det_policy(ep.history, ep.available()))
    assert groups_with_hidden_variation > 200


def test_dependency_identities_and_b6_constants():
    rng = random.Random(DEV7 + 950_000)
    for i in range(80):
        cfg = _cfg(i)
        ep = pw6.Episode(cfg, DEV7 + 800_000 + i)
        while not ep.done:
            f = pw6.factor_features(cfg, ep.state)
            for name, fn in fc.ANALYTIC_DEPS.items():
                assert abs(fn(f, cfg, ep.state)) < TOL, name
            for j in (0, 1, 2, 3, 4, 5, 6, 7, 10, 21):
                assert -TOL <= f[j] <= 1 + TOL
            assert all(math.isfinite(x) for x in f) and all(-5 <= f[j] <= 5 for j in range(11, 17))
            ep.step(rng.choice(ep.available()))
    # D and T never occur in any b6 / b6c training, dev, test, hold, hist or octet family: f9 == 0 and f22 == 1
    fams = set(pw6.TRAIN_TYPES) | set(pw6.HOLD_FAMILIES) | set(pw6.HIST_FAMILIES)
    for spl in list(pw6.SPLITS6.values()) + list(pw6.SPLITS6C.values()):
        fams |= set(spl[0])
    assert not any(("D" in k) or ("T" in k) for k in fams)


def test_candidate_trust_double_counts_probe_evidence_under_U():
    """Pins the HISTORICAL semantics documented in factor-contract.md: after a probe reported 'solved', the belief
    already conditions on it, and candidate_trust applies the probe likelihood again (trust > bH = P(commit
    correct)).  Without U (q = 1) the two coincide.  A corrected coordinate must get a new name."""
    c = pw6.Config6(prior=(0.5, 0.2, 0.2, 0.1), q=0.5, c_probe=1.0)
    st = pw6.advance(c, pw.initial_state(c), pw.A_PROBE, pw.O_SOLVED, False, None)
    f = pw6.factor_features(c, st)
    bH = st[1][0][0]
    assert abs(bH - 0.5 / (0.5 + 0.5 * 0.5)) < 1e-12
    assert f[21] > bH + 0.1 and abs(f[21] - bH / (bH + (1 - bH) * 0.5)) < 1e-12
    c1 = pw6.Config6(prior=(0.5, 0.2, 0.2, 0.1), q=1.0)
    st1 = pw6.advance(c1, pw.initial_state(c1), pw.A_PROBE, pw.O_SOLVED, False, None)
    assert pw6.factor_features(c1, st1)[21] == st1[1][0][0] == 1.0
