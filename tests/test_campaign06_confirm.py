"""A-CF-SMALL confirmation pipeline of research/tools/campaign06_portfolio.py (cf-fit / cf-score)
and bit-identity of the pre-existing headroom path."""
import argparse, gzip, json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign06_golden as golden   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEV = 2_300_000_000   # ext06 dev/smoke range (tests only)
# test populations (dev/smoke range; 5 episodes each)
TPOPS = {"train0": (DEV + 200, DEV + 205), "train1": (DEV + 210, DEV + 215), "train2": (DEV + 220, DEV + 225),
         "select": (DEV + 300, DEV + 305), "confirm": (DEV + 400, DEV + 405)}


@pytest.fixture(scope="module")
def mod():
    return golden.load_tool()


def _write(pw, path, lo, hi):
    with gzip.open(path, "wt") as f:
        for s in range(lo, hi):
            for r in pw.evaluate_episode(s, version="pw-v3"):
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def data(mod, tmp_path_factory):
    d = tmp_path_factory.mktemp("cfdata")
    files = {k: _write(mod.pw, d / f"{k}.jsonl.gz", *v) for k, v in TPOPS.items()}
    mats = {k: mod.cf_matrices(mod.load([f])[0]) for k, f in files.items()}
    return files, mats


# --- bit-identity of the existing path ------------------------------------------------------

def test_headroom_bit_identical_to_base_commit(mod, tmp_path):
    """headroom at the base commit (de28b9f7) wrote the golden fixture; the refactored tool
    (population_matrices) must reproduce every number exactly."""
    gold = json.loads((REPO / "tests/fixtures/campaign06_headroom_golden.json").read_text())
    p = tmp_path / "rec.jsonl.gz"
    golden.write_records(mod.pw, p)
    now = golden.headroom_small(mod, p, tmp_path)
    assert now["per_instance"] == gold["per_instance"]
    assert now["summary"] == gold["summary"]
    # the confirm pipeline's matrices are headroom's matrices
    P = mod.cf_matrices(mod.load([p])[0])
    for k in ("X", "UA", "UC", "USB", "UH", "c", "o", "L", "fwork", "Q", "FAILA", "Wk", "Ob"):
        assert np.array_equal(P[k], mod.G[k]), k
    for probe in mod.SEQ_PROBES:
        for k, v in P["SEQ"][probe].items():
            assert np.array_equal(v, mod.G["SEQ"][probe][k]), (probe, k)


# --- populations ------------------------------------------------------------------------------

def test_cf_populations_disjoint_and_in_role_ranges(mod):
    pw = mod.pw
    pops = mod.cf_populations()
    assert pops["train0"] == (310_000_000, 310_000_640)
    assert pops["train1"] == (311_000_000, 311_000_640) and pops["train2"] == (312_000_000, 312_000_640)
    assert pops["select"] == (320_000_000, 320_000_640) and pops["confirm"] == (330_000_000, 330_000_640)
    role = {"train0": "train", "train1": "train", "train2": "train", "select": "select", "confirm": "confirm"}
    for k, (lo, hi) in pops.items():
        r0, r1 = pw.SEED_RANGES[role[k]]
        assert r0 <= lo < hi <= r1
        for other in ("dev_builder", "dev_gate"):
            o0, o1 = pw.SEED_RANGES[other]
            assert hi <= o0 or o1 <= lo
    seeds = [set(range(*v)) for v in pops.values()]
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            assert not seeds[i] & seeds[j]
    sm = mod.cf_populations(60, smoke=True)
    for lo, hi in sm.values():
        assert pw.SEED_RANGES["dev_builder"][0] <= lo < hi <= pw.SEED_RANGES["dev_builder"][1]
    with pytest.raises(AssertionError):
        mod.cf_populations(1_000_001)          # lineages would collide
    with pytest.raises(AssertionError):
        mod.cf_populations(20_000, smoke=True)  # smoke populations would collide


# --- fit semantics ------------------------------------------------------------------------------

def test_cf_families_reproduce_headroom_families(mod, data):
    """Fit on train0, apply to select == headroom's family functions with tr = train rows,
    te = select rows (same menus, models and charges)."""
    files, mats = data
    recs = mod.load([files["train0"]])[0] + mod.load([files["select"]])[0]
    X, names = mod.feature_matrix(recs)
    P, _ = mod.population_matrices(recs, X, names)
    mod.G.clear(); mod.G.update(P)
    NT = len(mats["train0"]["c"])
    tr, te = np.arange(NT), np.arange(NT, len(recs))
    mod.G.update(X=X, names=names, eps=np.array([r["episode"] for r in recs]), minleaf=max(20, int(0.03 * NT)))
    arts, chosen, sel = mod.cf_fit(mats["train0"], mats["select"], log=lambda *_: None)
    PS = mats["select"]
    assert np.array_equal(mod.cf_apply(PS, arts["A0_single"]), mod.fam_single(tr, te))
    assert np.array_equal(mod.cf_apply(PS, arts["cascade_tuned"]), mod.fam_cascade(tr, te))
    assert np.array_equal(mod.cf_apply(PS, arts["strong_baseline"]), mod.fam_strong(tr, te))
    assert np.array_equal(mod.cf_apply(PS, arts["hand"]), mod.fam_hand(tr, te))
    for name, d in (("threshold_any", 1), ("tree_d2", 2), ("tree_d3", 3)):
        assert np.array_equal(mod.cf_apply(PS, arts[name]), mod.fam_tree(d)(tr, te)), name
    for cfg in mod.GBT_GRID:
        u, _ = mod.fam_learned_oneshot(tr, te, cfg)
        assert np.allclose(mod.cf_apply(PS, arts[f"learned_oneshot[cfg={cfg[0]}x{cfg[1]}]"]), u, rtol=0, atol=1e-12)
    cfg = tuple(arts[chosen["learned_oneshot"]]["cfg"])
    for p in mod.SEQ_PROBES:
        u, _ = mod.fam_learned_seq(p)(tr, te, cfg)
        assert np.allclose(mod.cf_apply(PS, arts[f"learned_seq[{p}]"]), u, rtol=0, atol=1e-12), p
    assert np.allclose(mod.cf_apply(PS, arts["learned_oneshot_direct"]), mod.fam_learned_direct(tr, te), rtol=0, atol=1e-12)
    # selector compute is charged
    for n in ("tree_d2", "logistic_tuned", "learned_oneshot", "learned_sequential"):
        ch = mod.cf_charge(PS, mod.cf_resolve(arts, chosen, n), arts)
        assert (ch > 0).all(), n


def test_selection_uses_select_population_only(mod, data):
    """Every fitted candidate is a function of the training population only; the select
    population enters only through the recorded select means, and every choice is the argmax
    of those means (reproducible from the frozen artifacts)."""
    files, mats = data
    a1, c1, s1 = mod.cf_fit(mats["train0"], mats["select"], log=lambda *_: None)
    a2, c2, s2 = mod.cf_fit(mats["train0"], mats["train1"], log=lambda *_: None)   # a different select population
    for n in a1:
        if n.startswith(("learned_seq[", "learned_sequential")):
            continue   # these use the chosen GBT config (compare only when it agrees)
        assert mod._sha(mod._pickle(a1[n])) == mod._sha(mod._pickle(a2[n])), n
    if c1["learned_oneshot"] == c2["learned_oneshot"]:
        for n in a1:
            assert mod._sha(mod._pickle(a1[n])) == mod._sha(mod._pickle(a2[n])), n
    for arts, ch, sel, PS in ((a1, c1, s1, mats["select"]), (a2, c2, s2, mats["train1"])):
        for n, art in arts.items():
            assert sel[n] == float(mod.cf_apply(PS, art, arts).mean())
        assert ch["best_simple"] == max(mod.CF_SIMPLE, key=lambda n: sel[n])
        assert ch["logistic_tuned"] == max((n for n in arts if n.startswith("logistic[")), key=lambda n: sel[n])
        assert ch["learned_oneshot"] == max((n for n in arts if n.startswith("learned_oneshot[")), key=lambda n: sel[n])


# --- CLI pipeline: roles, freezing, sealed scoring --------------------------------------------------

def _ns(**kw):
    base = dict(episodes=5, lineages=3, smoke=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_pipeline_roles_freeze_and_sealed_score(mod, data, tmp_path, monkeypatch):
    files, _ = data
    monkeypatch.setattr(mod, "cf_populations", lambda episodes=5, lineages=3, smoke=False: dict(TPOPS))
    loaded = []
    real_load = mod.load
    monkeypatch.setattr(mod, "load", lambda paths: (loaded.extend(paths), real_load(paths))[1])

    # role violations are refused before anything is fit
    with pytest.raises(SystemExit):   # select files from the confirm population
        mod.cmd_cf_fit(_ns(lineage=0, train=[files["train0"]], select=[files["confirm"]], out=str(tmp_path / "bad1")))
    with pytest.raises(SystemExit):   # lineage 1 given lineage 0's training population
        mod.cmd_cf_fit(_ns(lineage=1, train=[files["train0"]], select=[files["select"]], out=str(tmp_path / "bad2")))
    part = _write(mod.pw, tmp_path / "part.jsonl.gz", DEV + 300, DEV + 303)
    with pytest.raises(SystemExit):   # incomplete select population
        mod.cmd_cf_fit(_ns(lineage=0, train=[files["train0"]], select=[part], out=str(tmp_path / "bad3")))
    assert not any((tmp_path / b / "fit.pkl").exists() for b in ("bad1", "bad2", "bad3"))

    loaded.clear()
    fits = []
    for L in range(3):
        out = tmp_path / f"fit{L}"
        mod.cmd_cf_fit(_ns(lineage=L, train=[files[f"train{L}"]], select=[files["select"]], out=str(out)))
        fits.append(str(out))
        meta = json.loads((out / "fit.json").read_text())
        assert meta["confirm_touched"] is False and meta["lineage"] == L
        assert meta["fit_pkl_sha256"] == mod._sha((out / "fit.pkl").read_bytes())
    assert files["confirm"] not in loaded   # the confirm population is never read before scoring
    with pytest.raises(SystemExit):   # frozen once
        mod.cmd_cf_fit(_ns(lineage=0, train=[files["train0"]], select=[files["select"]], out=fits[0]))
    # determinism: refitting lineage 0 elsewhere gives the same frozen bytes
    mod.cmd_cf_fit(_ns(lineage=0, train=[files["train0"]], select=[files["select"]], out=str(tmp_path / "refit0")))
    assert (tmp_path / "refit0/fit.pkl").read_bytes() == (Path(fits[0]) / "fit.pkl").read_bytes()

    shas = [json.loads((Path(f) / "fit.json").read_text())["fit_pkl_sha256"] for f in fits]
    with pytest.raises(SystemExit):   # wrong expected hash
        mod.cmd_cf_score(_ns(fits=fits, confirm=[files["confirm"]], out=str(tmp_path / "s0"), boot=50,
                             expect_sha=["deadbeef"] + shas[1:]))
    with pytest.raises(SystemExit):   # "confirm" records from a training population
        mod.cmd_cf_score(_ns(fits=fits, confirm=[files["train2"]], out=str(tmp_path / "s1"), boot=50, expect_sha=shas))
    with pytest.raises(SystemExit):   # the same lineage twice
        mod.cmd_cf_score(_ns(fits=[fits[0], str(tmp_path / "refit0"), fits[2]], confirm=[files["confirm"]],
                             out=str(tmp_path / "s2"), boot=50, expect_sha=None))
    out = tmp_path / "score"
    mod.cmd_cf_score(_ns(fits=fits, confirm=[files["confirm"]], out=str(out), boot=50, expect_sha=[s[:16] for s in shas]))
    s = json.loads((out / "score.json").read_text())
    assert set(s["lineages"]) == {"0", "1", "2"}
    assert s["confirm"]["episodes"] == 5 and 0 <= s["oracle_certified_fraction"] <= 1
    z = np.load(out / "per_instance.npz")
    for L in range(3):
        r = s["lineages"][str(L)]
        d = z[f"L{L}__learned_oneshot"] - z[f"L{L}__best_simple"]
        assert r["estimates"]["learned_oneshot"]["primary_vs_best_simple_select"]["mean"] == pytest.approx(d.mean())
        assert r["utilities"]["best_simple"]["mean"] == pytest.approx(r["utilities"][r["chosen"]["best_simple"]]["mean"])
        assert (z[f"L{L}__learned_oneshot"] <= z["oracle"] + 1e-12).all()
    assert s["paired"]["learned_oneshot"]["lineages_passing"] == sum(
        s["lineages"][str(L)]["estimates"]["learned_oneshot"]["primary_vs_best_simple_select"]["pass"] for L in range(3))
    with pytest.raises(SystemExit):   # scored once
        mod.cmd_cf_score(_ns(fits=fits, confirm=[files["confirm"]], out=str(out), boot=50, expect_sha=shas))
    # a tampered artifact is refused
    b = bytearray((Path(fits[1]) / "fit.pkl").read_bytes()); b[-5] ^= 1
    (Path(fits[1]) / "fit.pkl").write_bytes(bytes(b))
    with pytest.raises(SystemExit):
        mod.cmd_cf_score(_ns(fits=fits, confirm=[files["confirm"]], out=str(tmp_path / "s3"), boot=50, expect_sha=None))
