"""Extended-06 Track B: probeworld-v3 (generator range revision + D/T extension), exact screening, split table v3
(design-v2 arms), balanced counterfactual octets, trainer b6 options and scorer.

Pure Python except the tests marked with ``importorskip("torch")``.  The extended-04/05 goldens
(tests/test_campaign05_trackb.py, tests/test_campaign05_bxc.py) pin every b1/b5/b5c path bit for bit."""
import importlib.util
import itertools
import json
import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    from tensegra import campaign04_probeworld as pw
except ImportError:  # no torch locally: the package __init__ imports torch; load the pure-Python modules directly
    import types
    for _k in [k for k in sys.modules if k == "tensegra" or k.startswith("tensegra.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("tensegra")
    _pkg.__path__ = [str(ROOT / "src" / "tensegra")]
    sys.modules["tensegra"] = _pkg
    from tensegra import campaign04_probeworld as pw
from tensegra import campaign05_probeworld as pw5
from tensegra import campaign06_probeworld as pw6

DEV = 6_800_000_000  # dev/smoke sub-range (never protocol)


def replace_(c, **kw):
    import dataclasses
    return dataclasses.replace(c, **kw)


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _solve(cfg):
    s = pw6.ExactSolver(cfg)
    s.value(pw.initial_state(cfg))
    return s


def _cfg(key, i, k=None):
    seed = DEV + 1000 * pw6.SCREEN_FAMILY_ORDER.index(pw6.canon(key)) + i if key != "0" else DEV + i
    cell, kk, _ = pw6.draw_params(seed, pw.TRAIN_CELLS, (1, 2) if k is None else (k,), (pw6.canon(key),))
    return pw6.make_config(cell, kk, pw6.flags_from_key(key), seed)


# ------------------------------------------------------------------ environment

def test_v3_with_extension_off_equals_v1():
    for i in range(10):
        c1 = pw5.split_config("b5_train", i)
        c6 = pw6.from_v1(c1)
        s1, s6 = pw.ExactSolver(c1), pw6.ExactSolver(c6)
        assert s1.value(pw.initial_state(c1)) == s6.value(pw.initial_state(c6))
        assert s1._Q == s6._Q
    assert pw6.env(pw.Config()) is pw and pw6.env(pw6.Config6()) is pw6.ENV6


def test_generator_ranges_and_independent_draws():
    for i in range(300):
        c = pw6.split6_config("b6_dev", i)
        f = pw6.flags_of(c)
        assert not f[4] and not f[5]  # D / T are not in any b6 split
        if f[2]:
            assert c.k >= 2 and 0.25 <= c.corr <= 0.45
        if f[3]:
            assert 0.3 <= c.p_event <= 0.6
    # re-generating without a factor = ablate, except the v1 conditional draw order (eta, p_conflict) for U/S/C/E;
    # the D / T / v3 corr / p_event draws never shift
    base = pw6.make_config((0, 1), 2, pw6.flags_from_key("SCET"), DEV + 7)
    for f in "SCET":
        other = pw6.make_config((0, 1), 2, pw6.flags_from_key("SCET".replace(f, "")), DEV + 7)
        abl = pw6.ablate(base, f)
        assert pw6.flags_of(other) == pw6.flags_of(abl)
        assert replace_(other, eta=0, p_conflict=0) == replace_(abl, eta=0, p_conflict=0)
        if f == "T":
            assert other == abl
    with pytest.raises(ValueError):
        pw6.make_config((0, 1), 1, pw6.flags_from_key("C"), DEV)


def test_deadline_and_hardness_semantics():
    c = _cfg("DT", 3, k=1)
    d, t = c.deadline, c.t_hard
    assert d in (1, 2) and 0.5 <= t <= 2.5
    st = pw.initial_state(c)
    for _ in range(d):
        a = pw.A_INSPECT if pw.A_INSPECT in pw6.available(c, st) else pw.A_PROP
        st = pw6.advance(c, st, a, pw.O_EASY if a == pw.A_INSPECT else pw.O_REDUCED, False, None)
    assert set(pw6.available(c, st)) <= pw.TERMINAL
    # T: exact calls on hard types charge the surcharge (ledger line), never on H / M, whatever the reduction state
    for theta, want in ((pw.TH, 0.0), (pw.TM, 0.0), (pw.TF, t * c.c_b2), (pw.TX, t * c.c_b2)):
        for red in (False, True):
            assert pw6.hard_surcharge(c, pw.A_B2, theta, red) == want
            assert pw6.hard_surcharge(c, pw.A_PROBE, theta, red) == 0.0
    ep = pw6.Episode(pw6.from_v1(pw.Config(prior=(0, 0, 1, 0), k=1), t_hard=1.0), 1)
    ep.step(pw.A_B2)
    assert ("exact_hardness" in [x[1] for x in ep.ledger])


def test_labels_are_functions_of_visible_history():
    rng = random.Random(3)
    seen = {}
    for key in ("SCE", "UCE", "DT", "UET"):
        for i in range(3):
            c = _cfg(key, i)
            s = _solve(c)
            for w in range(12):
                ep = pw6.Episode(c, DEV + 100 * i + w)
                while not ep.done:
                    st = pw6.public_state_from_history(c, ep.history)
                    assert st == ep.state
                    h = (c, tuple(ep.history))
                    lab = (tuple(sorted(s.q_values(st).items())), s.opt_set(st))
                    assert seen.setdefault(h, lab) == lab
                    ep.step(rng.choice(ep.available()))
    assert len(seen) > 300


def test_dp_value_matches_monte_carlo():
    for key in ("DT", "SCE"):
        c = _cfg(key, 1, k=2)
        s = _solve(c)
        us = []
        for w in range(2500):
            ep = pw6.Episode(c, DEV + 50_000 + w)
            while not ep.done:
                ep.step(s.pi_star(ep.state))
            us.append(ep.utility)
        m = sum(us) / len(us)
        se = (sum((u - m) ** 2 for u in us) / (len(us) - 1) / len(us)) ** 0.5
        assert abs(m - s.value(pw.initial_state(c))) < 4.5 * se + 1e-9


def test_mobius_and_foreign_policy():
    c = _cfg("SE", 2)
    full = ("S", "E")
    qs = {T: _solve(pw6.restrict(c, T)).q_values(pw.initial_state(c)) for m in range(3)
          for T in itertools.combinations(full, m)}
    add = pw6.mobius_additive(qs, full, 1)
    for a in add:
        assert add[a] == pytest.approx(qs[("S",)][a] + qs[("E",)][a] - qs[()][a])
    s = _solve(c)
    assert pw6.foreign_policy_value(s, s) == pytest.approx(s.value(pw.initial_state(c)))  # own policy: no regret
    ex, p = pw6.any_decision_change(s, s)
    assert not ex and p == 0.0


# ------------------------------------------------------------------ screening

def test_screen_record_and_rules(tmp_path):
    bs = _load("research/tools/campaign06_bscreen.py", "bscreen6")
    r = bs.screen_one(("SE", 900_000))
    assert set(r["factors"]) == {"S", "E"} and r["interaction"]["order1"]["resid_q"] >= 0
    for f, v in r["factors"].items():
        assert v["regret"] >= -1e-6 and len(v["ep_regret_grid"]) >= 1
        assert (v["regret"] > pw.EPS) <= v["any"]  # a regret above eps needs some changed decision
    # design-v2 rule on a synthetic summary: largest minimum 'any' relevance subject to GS closable <= .5
    fams = {
        "SCE": {"relevance": {x: {"any": v} for x, v in zip("SCE", (.8, .7, .6))}, "gs_regret": {"GS_regret": .2},
                "n_eligible": 10},
        "UCE": {"relevance": {x: {"any": v} for x, v in zip("UCE", (.9, .65, .8))}, "gs_regret": {"GS_regret": .1},
                "n_eligible": 10},
        "USE": {"relevance": {x: {"any": v} for x, v in zip("USE", (.9, .9, .9))}, "gs_regret": {"GS_regret": .7},
                "n_eligible": 10},
        "SET": {"relevance": {x: {"any": v} for x, v in zip("SET", (.9, .62, .9))}, "gs_regret": {"GS_regret": .1},
                "n_eligible": 10}}
    sel = bs.select_v2({"families": fams})
    assert sel["primary"] == "UCE" and sel["secondary"] == "SCE"  # USE fails GS; SET (extension) not clearer
    assert not sel["extension_adopted"] and sel["not_used_per_slot_reading"] == ["UCE", "SET"]
    fams["SET"]["relevance"]["E"]["any"] = .75  # extension clearly better (.75 > .65 + .05): both slots from it
    fams["CET"] = {"relevance": {x: {"any": .72} for x in "CET"}, "gs_regret": {"GS_regret": .1}, "n_eligible": 10}
    sel = bs.select_v2({"families": fams})
    assert sel["extension_adopted"] and (sel["primary"], sel["secondary"]) == ("SET", "CET")


# ------------------------------------------------------------------ split table v3 and arms

def test_split_table_v3_audit_and_arms():
    au = pw6.audit_split_table(384)
    assert au["pass"], au
    assert pw6.BASE_PAIR not in pw6.constituents(pw6.PRIMARY)
    for arm in pw6.ARM_SIZE:
        comp = pw6.arm_composition(arm, 384)
        assert not any(pw6.excluded_from_training(k) for k in comp)
        assert pw6.factor_frequency(comp) == pw6.factor_frequency(pw6.arm_composition("B0", 384))
    assert len(set(pw6.arm_pairs("B2"))) == 3 and len(pw6.arm_pairs("dose3")) == 4
    pools, rep = pw6.build_arm_pools(12, classify=pw6.make_classifier())
    b0, b1 = pools["B0"], pools["B1"]
    for key in pw6.TRAIN_TYPES:
        a0 = [x for x in b0 if x[0] == key]
        a1 = [x for x in b1 if x[0] == key]
        assert a1[:len(a0)] == a0  # B0 is B1's first half, type by type
    for arm in ("B2", "B3", "dose1", "dose2", "dose3"):
        r = rep[arm]
        if r["unmatched_first_class"] == 0:
            assert r["first_class_mix"] == r["first_class_mix_ref"]
        kept = set(pools[arm]) & set(pools[pw6.ARM_REPLACES[arm]])
        assert len(kept) == len(pools[arm]) - r["added"]


def test_seed_ranges_registry_extended06():
    reg = json.loads((ROOT / "research/campaigns/extended-06/seed-ranges.json").read_text())
    rs = reg["ranges"]
    by = {r["name"]: r for r in rs}

    def ancestors(r):  # transitive parents (B-FACT-C ranges nest inside a Track B v3 sub-range)
        out, p = set(), r.get("parent")
        while p in by and p not in out:
            out.add(p)
            p = by[p].get("parent")
        return out

    for r in rs:
        for p in ancestors(r):
            assert by[p]["lo"] <= r["lo"] and r["hi"] <= by[p]["hi"], (r["name"], p)
    for i, a in enumerate(rs):
        assert a["lo"] < a["hi"]
        for b in rs[i + 1:]:
            if a.get("contains") == b["name"] or b.get("contains") == a["name"] or \
                    a["name"] in b.get("parent", "") or b["name"] in a.get("parent", "") or \
                    a["name"] in ancestors(b) or b["name"] in ancestors(a):
                continue
            assert not (a["lo"] < b["hi"] and b["lo"] < a["hi"]), (a["name"], b["name"])
    sub = [r for r in rs if r.get("parent") == "ext06 Track B probeworld-v3 configurations"]
    assert {r["name"].split(":")[1].strip() for r in sub} >= set(pw6.SUBRANGES)
    assert pw6.check_subranges()


def test_counterfactual_octet_and_near_miss():
    old = pw6.V3_K
    pw6.V3_K = (1, 2)  # cheap (the octet construction reads V3_K at call time)
    try:
        _octets()
    finally:
        pw6.V3_K = old


def _octets():
    for i in range(6):
        cs = pw6.counterfactual_set("SCE", i, base=DEV + 500_000)
        assert set(cs["members"]) == {"0", "S", "C", "E", "SC", "SE", "CE", "SCE"}
        cfgs = {k: pw6.config_from_dict6(m["config"]) for k, m in cs["members"].items()}
        for key, c in cfgs.items():
            assert pw6.fam_key(pw6.flags_of(c)) == key
            assert c.c_probe == cfgs["SCE"].c_probe and c.k == cfgs["SCE"].k
        for h, t in cs["types"].items():
            full = cs["members"]["SCE"]["labels"][h]
            abl = [cs["members"][x]["labels"][h] for x in ("CE", "SE", "SC")]
            sing = [cs["members"][x]["labels"][h] for x in ("S", "C", "E")]
            assert t["flip_ablations"] == all(d["pi"] not in full["opt"] for d in abl)
            assert t["flip"] == all(d["pi"] not in full["opt"] for d in sing)
            nm = t["near_miss"]
            if nm is not None:
                c2 = pw6.config_from_dict6(nm["config"])
                assert pw6.flags_of(c2) == pw6.flags_of(cfgs["SCE"]) and nm["label"]["unique"]
                assert nm["label"]["pi"] not in full["opt"]
                assert pw6.inherited(_solve(c2), "SCE", h, 1)
    pairs = pw6.one_factor_pairs(cs)
    assert len(pairs) == 12 and all(pw6.canon(a + f if a != "0" else f) == b for a, b, f in pairs)


def test_factor_features_public():
    c = _cfg("UCE", 1)
    s = _solve(c)
    ep = pw6.Episode(c, DEV + 3)
    while not ep.done:
        ff = pw6.factor_features(c, ep.state)
        assert len(ff) == pw6.N_FACTOR_FEATURES and all(isinstance(x, float) for x in ff)
        assert ff == pw6.factor_features(c, pw6.public_state_from_history(c, ep.history))
        ep.step(s.pi_star(ep.state))


# ------------------------------------------------------------------ torch: b6 trainer paths end to end

TINY = ["--updates", "2", "--batch", "4", "--hidden", "16", "--log-every", "1"]


def test_b6_end_to_end_tiny(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    m = _load("research/tools/campaign04_probeworld_train.py", "pwtrain_b6")
    sc = _load("research/tools/campaign06_bscore.py", "bscore6")
    monkeypatch.setattr(pw6, "V3_K", (1, 2))  # tiny and cheap (k = 8 correlated+events DPs reach ~1.4e5 states)
    lab = tmp_path / "b6"
    m.main(["labels", "--out", str(lab), "--split-set", "b6", "--n-train", "12", "--n-eval", "2", "--n-hold", "3",
            "--n-hist", "1", "--n-cf", "2", "--b6-shard", "2", "--hold-relevance"])
    meta = json.loads((lab / "labels_meta.json").read_text())
    assert meta["split_set"]["audit"]["pass"]
    for arm in pw6.ARM_SIZE:
        assert (lab / f"b6_{arm}.pkl").exists()
    assert (lab / "cf_SCE.json").exists() and (lab / "b6_hold_SCE.shards.json").exists()
    runs = {}
    for name, extra in (("B0", ["--train-split", "b6_B0"]), ("B2", ["--train-split", "b6_B2"]),
                        ("SUP", ["--train-split", "b6_B0", "--inputs", "factors6", "--arch", "fuse"]),
                        ("LRN", ["--train-split", "b6_B0", "--arch", "fuse", "--factor-mode", "learned"]),
                        ("RAWF", ["--train-split", "b6_B0", "--arch", "fuse"])):
        d = tmp_path / name
        m.main(["train", "--labels", str(lab), "--out", str(d), "--rung", "L1", "--seed", "30", *extra, *TINY])
        tm = json.loads((d / "train_meta.json").read_text())
        assert tm["public_extra"] == 4 and tm["train_split"] == extra[1]
        if name == "LRN":
            assert tm["fuse"]["factor_mode"] == "learned" and tm["fuse"]["stop_gradient"]
            assert "aux" in json.loads((d / "train_log.json").read_text())[0]
        if name == "SUP":
            assert tm["fuse"]["factor_mode"] == "supplied"
        m.main(["eval", "--labels", str(lab), "--run", str(d), "--split-set", "b6", "--episode-rows", "--cf",
                "--worlds", "1", "--world-offset", "900"])
        cf = json.loads((d / "cf_eval.json").read_text())
        assert set(cf["families"]) == {"SCE", "UCE"}
        runs[name] = [str(d)]
    res = sc.score({"B0": runs["B0"], "B2": runs["B2"]}, ref="B0", n_boot=20)
    g = res["arms"]["B0"]["groups"]
    assert "b6_hold_SCE" in g and "cf:pooled" in g
    assert "pair0" in res["paired"]["B2-B0"]
