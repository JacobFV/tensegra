"""Extended-07 Phase 1 diagnostic runner (research/tools/campaign07_diag.py) and scorer (campaign07_diagscore.py).

Synthetic tests use small random-initialised models and configurations from the dev_smoke seed sub-range (never
protocol).  The historical-reproduction tests read the extended-06 artifacts READ-ONLY and are skipped where they are
absent (they run on pro6000; override the root with E06_RESULTS)."""
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))
import campaign07_diag as D  # noqa: E402
import campaign07_diagscore as S  # noqa: E402

T, pw, pw6 = D.T, D.pw, D.pw6
E06 = Path(os.environ.get("E06_RESULTS", os.path.expanduser("~/structured-latent-dynamics-campaign06/results")))
DEV_SEED = 6_860_000_000  # dev_smoke sub-range


def _configs(n=6):
    out = []
    fams = ["SCE", "SC", "S", "E", "0", "CE"]
    for j in range(n):
        fam = fams[j % len(fams)]
        seed = DEV_SEED + j
        k = 2
        cfg = pw6.make_config(pw.TRAIN_CELLS[j % len(pw.TRAIN_CELLS)], k, pw6.flags_from_key(fam), seed)
        s = pw6.ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        out.append((j, cfg, s))
    return out


@pytest.fixture(scope="module")
def pool():
    return _configs()


def _model(mode, seed=0, hidden=32):
    torch.manual_seed(seed)
    if mode == "B0":
        m = T.ProbeNet(hidden, public_extra=4)
    elif mode == "SUP":
        m = T.ProbeNet(hidden, inputs="factors6", arch="fuse", public_extra=4)
    else:
        m = T.ProbeNet(hidden, arch="fuse", public_extra=4, factor_mode={"LRN": "learned", "RAWF": "none"}[mode])
    m.eval()
    return m


def _save_run(tmp, name, mode, seed=0):
    d = tmp / name
    d.mkdir()
    m = _model(mode, seed)
    torch.save(m.state_dict(), d / "model.pt")
    meta = {"rung": "L1", "seed": seed, "hidden": 32, "public_extra": 4}
    if mode != "B0":
        meta["arch"] = "fuse"
        meta["fuse"] = {"factor_mode": {"LRN": "learned", "RAWF": "none", "SUP": "supplied"}[mode]}
    if mode == "SUP":
        meta["inputs"] = "factors6"
    (d / "train_meta.json").write_text(json.dumps(meta))
    return d


def _labels(tmp, pool_rows, split="b6_dev"):
    d = tmp / "labels"
    d.mkdir()
    with open(d / f"{split}.shard000.pkl", "wb") as f:
        pickle.dump([(i, c, s._V, s._Q) for i, c, s in pool_rows], f)
    (d / f"{split}.shards.json").write_text(json.dumps({"shards": [f"{split}.shard000"]}))
    return d


@pytest.fixture
def dev_worlds(monkeypatch):
    monkeypatch.setattr(T, "split_world_seed", lambda split, idx, rep: 6_880_000_000 + 1000 * idx + rep)


def _items(pool_rows):
    return [(c, s, 6_880_000_000 + 1000 * i) for i, c, s in pool_rows]


def _recs(path):
    with gzip.open(path, "rt") as f:
        head = json.loads(f.readline())["_meta"]
        return head, [json.loads(line) for line in f]


# ------------------------------------------------------------------ forward / drive equal the trainer's paths

@pytest.mark.parametrize("mode", ["LRN", "RAWF", "SUP", "B0"])
def test_forward_is_model_step(mode, pool):
    m = _model(mode)
    cfg, s = pool[0][1], pool[0][2]
    st = pw.initial_state(cfg)
    av = pw6.available(cfg, st)
    x = torch.tensor([T.encode(cfg.public_vector(), None, av, 0.0) + T.supplied(cfg, st, getattr(m, "inputs", "public"))])
    h = torch.randn(1, 32)
    with torch.no_grad():
        h1, z1 = m.step(x, h)
        h2, zpre, pred, phi, z2 = D.forward(m, x, h)
    assert torch.equal(h1, h2) and torch.equal(z1, z2)
    assert (pred is None) == (mode in ("RAWF", "B0"))


@pytest.mark.parametrize("mode", ["LRN", "RAWF", "SUP", "B0"])
def test_free_run_equals_run_batch_greedy(mode, pool):
    """drive() in free mode takes exactly campaign04 run_batch's greedy actions (same episodes, same utilities)."""
    m = _model(mode, seed=3)
    items = _items(pool)
    with torch.no_grad():
        eps, _, _ = T.run_batch(m, items, "greedy", need_labels=False)
    tr = [D.Track(c, s, f"x:{j}", ws) for j, (c, s, ws) in enumerate(items)]
    D.drive(m, tr, {"protocol": "B", "model": mode})
    assert [t.ep.history for t in tr] == [e.history for e in eps]
    assert [t.ep.utility for t in tr] == [e.utility for e in eps]


def test_common_history_rebuild_equals_free_run_state(pool):
    """Replaying a model's own free-run history through the same model rebuilds the identical state (pre-fusion
    latent, predictions, probabilities, greedy action) at every decision; the history probability is the product of
    the model's own probabilities of its (greedy) actions."""
    m = _model("LRN", seed=5)
    items = _items(pool)
    free = [D.Track(c, s, f"x:{j}", ws) for j, (c, s, ws) in enumerate(items)]
    D.drive(m, free, {"protocol": "B", "model": "m"})
    forced = [D.Track(c, s, f"x:{j}", ws, history=t.prefix, src="own") for j, ((c, s, ws), t) in enumerate(zip(items, free))]
    D.drive(m, forced, {"protocol": "A", "model": "m"})
    for tf, tr in zip(free, forced):
        assert len(tf.recs) == len(tr.recs) > 0
        for a, b in zip(tf.recs, tr.recs):
            for key in ("z_pre", "pred", "probs", "a", "target", "Q", "opt", "ctx", "hist_id", "gap"):
                assert a[key] == b[key], key
            assert b["a_hist"] == a["a"] and b["p_hist"] == a["probs"][a["a"]]
        ep = D.episode_record_forced(tr, {})
        assert ep["n_diverge"] == 0


def test_replay_never_transplants_state(pool):
    """The same history replayed through two different models gives each model's own state (different latents)."""
    items = _items(pool)[:2]
    m1, m2 = _model("LRN", seed=1), _model("LRN", seed=2)
    free = [D.Track(c, s, "x", ws) for c, s, ws in items]
    D.drive(m1, free, {})
    t1 = [D.Track(c, s, "x", ws, history=t.prefix) for (c, s, ws), t in zip(items, free)]
    t2 = [D.Track(c, s, "x", ws, history=t.prefix) for (c, s, ws), t in zip(items, free)]
    D.drive(m1, t1, {})
    D.drive(m2, t2, {})
    assert all(a["z_pre"] != b["z_pre"] for x, y in zip(t1, t2) for a, b in zip(x.recs, y.recs))
    assert [r["target"] for r in t1[0].recs] == [r["target"] for r in t2[0].recs]


# ------------------------------------------------------------------ interventions

def test_pred_replacement_is_identity(pool):
    m = _model("LRN", seed=7)
    ivn = D.Intervener(["none", "pred", "scale_err:1"], "LRN")
    tr = [D.Track(c, s, "x", ws) for c, s, ws in _items(pool)]
    D.drive(m, tr, {}, ivn)
    n = 0
    for t in tr:
        for r in t.recs:
            assert r["iv"]["pred"]["a"] == r["a"] and r["iv"]["pred"]["probs"] == r["probs"]  # bit-identical
            assert r["iv"]["scale_err:1"]["a"] == r["a"]  # t + 1 * (p - t): identity up to float rounding
            n += 1
    assert n > 0
    # exact replacement with targets equal to the prediction is the identity at the tensor level
    st = pw.initial_state(pool[0][1])
    av = pw6.available(pool[0][1], st)
    x = torch.tensor([T.encode(pool[0][1].public_vector(), None, av, 0.0)])
    with torch.no_grad():
        _, z, pred, phi, zf = D.forward(m, x, torch.zeros(1, 32))
        p = pred[0].tolist()
        alt = ivn.phi("exact", p, list(p), "k") if "exact" in ivn.names else D.Intervener(["exact"], "LRN").phi("exact", p, list(p), "k")
        assert torch.equal(D.fuse_out(m, z, torch.tensor([alt])), zf)


def test_exact_replacement_changes_only_phi(pool):
    """Immediate interventions never alter the logged state or the acted trajectory."""
    m = _model("LRN", seed=8)
    items = _items(pool)
    a = [D.Track(c, s, "x", ws) for c, s, ws in items]
    b = [D.Track(c, s, "x", ws) for c, s, ws in items]
    D.drive(m, a, {})
    D.drive(m, b, {}, D.Intervener(["exact", "iso:prob", "dep:strat", "keep:build", "mirror"], "LRN"))
    assert [t.prefix for t in a] == [t.prefix for t in b]
    for ta, tb in zip(a, b):
        for ra, rb in zip(ta.recs, tb.recs):
            assert ra["z_pre"] == rb["z_pre"] and ra["probs"] == rb["probs"]
            assert set(rb["iv"]) == {"exact", "iso:prob", "dep:strat", "keep:build", "mirror"}


@pytest.mark.parametrize("kind", ["RAWF", "SUP", "B0"])
def test_non_lrn_refuses_exact_injection(kind):
    with pytest.raises(ValueError, match="refusing factor injection"):
        D.Intervener(["exact"], kind)
    with pytest.raises(ValueError, match="refusing"):
        D.Intervener(["none", "iso:prob"], kind)
    D.Intervener(["none"], kind)  # no injection: allowed


def test_group_definitions():
    groups, deps, src = D.load_groups(Path("/nonexistent"))
    assert sorted(x for g in groups.values() for x in g) == sorted(D.FEATS) and src.startswith("provisional")
    iv = D.Intervener(["dep:strat"], "LRN")
    cl = iv.coords("dep:strat")
    assert set(groups["strat"]) <= set(cl) and "belief_H" in cl and "exp_side_cost_rel" in cl
    assert iv.coords("iso:strat") == list(groups["strat"])
    assert set(iv.coords("keep:prob")) == set(D.FEATS) - set(groups["prob"])
    with pytest.raises(ValueError):
        D.Intervener(["iso:nope"], "LRN")
    with pytest.raises(ValueError):
        D.Intervener(["gauss:1"], "LRN")  # needs a support reference


def test_context_labels():
    st = lambda q: ((q, False, 0, 0), None)  # noqa: E731
    assert D.context([], st(0)) == "initial"
    assert D.context([(pw.A_B2, pw.O_SOLVED, False, None), (pw.A_COMMIT, pw.O_CORRECT, False, pw.TH)], st(1)) == "query2_after_H"
    assert D.context([(pw.A_COMMIT, pw.O_CORRECT, False, pw.TM)], st(1)) == "query2_after_notH"
    assert D.context([(pw.A_COMMIT, pw.O_CORRECT, False, pw.TH)], st(2)) == "later_query_start"
    assert D.context([(pw.A_PROBE, pw.O_FAILED, False, None)], st(0)) == "after_probe_failed"
    assert D.context([(pw.A_B1, pw.O_TIMEOUT, False, None)], st(0)) == "after_b1_timeout"
    assert D.context([(pw.A_PROP, pw.O_REDUCED, False, None)], st(0)) == "other"


# ------------------------------------------------------------------ end-to-end run, no-overwrite, scorer

def test_run_end_to_end_and_no_overwrite(tmp_path, pool, dev_worlds):
    lab = _labels(tmp_path, pool)
    lrn = _save_run(tmp_path, "lrn", "LRN", 1)
    raw = _save_run(tmp_path, "rawf", "RAWF", 1)
    out = tmp_path / "out"
    argv = ["run", "--labels", str(lab), "--pool", "b6_dev", "--model", f"LRN-s1={lrn}", "--model", f"RAWF-s1={raw}",
            "--ivs", "none", "exact", "iso:prob", "dep:strat", "--rollout", "exact", "--out", str(out), "--tag", "t"]
    D.main(argv)
    files = sorted(p.name for p in out.iterdir())
    for proto in ("B", "A-pistar", "A-own_LRN-s1", "A-own_RAWF-s1", "R-exact"):
        assert f"diag-v1-{proto}-LRN-s1.jsonl.gz" in files
    assert "diag-v1-R-exact-RAWF-s1.jsonl.gz" not in files  # rollout interventions: LRN only
    _, b = _recs(out / "diag-v1-B-LRN-s1.jsonl.gz")
    _, own = _recs(out / "diag-v1-A-own_LRN-s1-LRN-s1.jsonl.gz")
    bd = [r for r in b if r["kind"] == "decision"]
    od = [r for r in own if r["kind"] == "decision"]
    assert [(r["z_pre"], r["probs"], r["a"]) for r in bd] == [(r["z_pre"], r["probs"], r["a"]) for r in od]
    assert all("iv" in r for r in bd) and all(r["iv"]["exact"]["ok"] in (True, False) for r in bd)
    _, rb = _recs(out / "diag-v1-B-RAWF-s1.jsonl.gz")
    assert all("iv" not in r and r["pred"] is None for r in rb if r["kind"] == "decision")  # never injected into RAWF
    eps = [r for r in own if r["kind"] == "episode"]
    assert eps and all(e["support_flag"] in ("supported", "low", "unsupported") for e in eps)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        D.main(argv)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        D.Writer(out / "diag-v1-B-LRN-s1.jsonl.gz", {})
    # outputs inside an input (run dir / labels dir / historical root) are refused
    with pytest.raises(SystemExit, match="inside read-only"):
        D.main(argv[:-4] + ["--out", str(lrn / "diag"), "--tag", "t2"])
    with pytest.raises(SystemExit, match="inside read-only"):
        D.check_out_dir(os.path.join(D.READ_ONLY_ROOTS[0], "results", "x"))
    # scorer: reads, scores, refuses to overwrite
    sc = tmp_path / "score.json"
    S.main(["--files", *[str(out / f) for f in files if f.endswith(".gz")], "--out", str(sc), "--n-boot", "200",
            "--contrast", "LRN", "RAWF"])
    res = json.loads(sc.read_text())
    assert "B|LRN" in res["cells"] and "prediction" in res["cells"]["B|LRN"]
    assert "interventions" in res["cells"]["A-pistar|LRN"]
    assert "B|LRN-RAWF" in res["contrasts"] and "A-own_LRN|LRN-RAWF" in res["contrasts"]
    with pytest.raises(SystemExit):
        S.main(["--files", str(out / files[0]), "--out", str(sc)])


def test_cf_path(tmp_path, pool):
    """Counterfactual-only path: labels' eps-optimal sets reproduced; members x decision types logged."""
    base = 6_830_000_000
    idx = next(i for i in range(100) if pw6.draw_params(base + i, pw.TRAIN_CELLS, pw6.V3_K, ("SCE",))[1] == 2)
    cs = pw6.counterfactual_set("SCE", idx, base=base)
    (tmp_path / "cflabels").mkdir()
    p = tmp_path / "cflabels" / "cf_SCE.json"
    p.write_text(json.dumps([cs]))
    lrn = _save_run(tmp_path, "lrn", "LRN", 2)
    out = tmp_path / "o"
    D.main(["run", "--cf", str(p), "--model", f"LRN-s2={lrn}", "--ivs", "exact", "--out", str(out)])
    _, recs = _recs(out / "diag-v1-cf-SCE-LRN-s2.jsonl.gz")
    n_exp = sum(1 for h in cs["types"] for m in cs["members"].values() if m["labels"][h] is not None) + \
        sum(t["near_miss"] is not None for t in cs["types"].values())
    assert len(recs) == n_exp
    for r in recs:
        # the choice equals campaign04 model_choice (the historical cf_eval path)
        key = r["cf"]["member"]
        cfg = (pw6.config_from_dict6(cs["types"][r["cf"]["type"]]["near_miss"]["config"]) if key == "near_miss"
               else pw6.config_from_dict6(cs["members"][key]["config"]))
        m, _ = D.load_model(lrn)
        assert r["a"] == T.model_choice(m, cfg, pw6.DECISION_TYPES[r["cf"]["type"]])
        assert r["ctx"] in {"initial", "after_probe_solved", "after_probe_failed", "after_b1_timeout", "query2_after_H",
                            "query2_after_notH"}


def test_two_level_bootstrap_basic():
    rng = np.random.default_rng(0)
    cells_a, cells_b = {}, {}
    for s in range(5):
        for c in range(50):
            cells_a[(s, f"c{c}")] = np.array([float(rng.random() < 0.7), 1.0])
            cells_b[(s, f"c{c}")] = np.array([float(rng.random() < 0.5), 1.0])
    ea = S.boot_endpoint(cells_a, 2000, 1)
    assert ea["ci"][0] < ea["mean"] < ea["ci"][1]
    assert ea == S.boot_endpoint(cells_a, 2000, 1)  # fixed seed: reproducible
    c = S.boot_contrast(cells_a, cells_b, 2000, 1)
    assert c["ci"][0] > 0 and c["signs"]["positive"] == 5 and "half1" in c["mc_check"]


# ------------------------------------------------------------------ historical reproduction (pro6000 only)

HIST_RUN = E06 / "e06-tb-lrn-s35" / "run"
B6C = E06 / "e06-tbc-labels" / "labels"
need_hist = pytest.mark.skipif(not (HIST_RUN / "eval_b6c.json").exists() or not B6C.exists(),
                               reason="extended-06 artifacts not present")


@need_hist
def test_free_run_reproduces_historical_eval_b6c(tmp_path):
    out = tmp_path / "hist"
    D.main(["run", "--labels", str(B6C), "--pool", "b6c_hold_SCE", "--model", f"LRN-s35={HIST_RUN}",
            "--protocols", "B", "--no-latent", "--check-historical", "--out", str(out), "--tag", "hist"])
    rep = json.loads((out / "diag-v1-summary-hist.json").read_text())
    assert rep["historical_check"]["LRN-s35"]["status"] == "identical", rep["historical_check"]
    _, recs = _recs(out / "diag-v1-B-LRN-s35.jsonl.gz")
    mine = {}
    for r in recs:
        if r["kind"] == "decision":
            mine.setdefault(r["cfg_id"], []).append(r["a"])
    with gzip.open(HIST_RUN / "eval_b6c_episodes.jsonl.gz", "rt") as f:
        f.readline()
        hist = {f"{r['split']}:{r['cfg_idx']}": [d["a"] for d in r["decisions"]] for r in map(json.loads, f)}
    assert mine == hist and len(mine) == 400


@need_hist
def test_cf_reproduces_historical_cf_eval_b6c(tmp_path):
    out = tmp_path / "cf"
    D.main(["run", "--cf", str(B6C / "cf_SCE.json"), "--n-octets", "10", "--model", f"LRN-s35={HIST_RUN}",
            "--no-latent", "--out", str(out)])
    _, recs = _recs(out / "diag-v1-cf-SCE-LRN-s35.jsonl.gz")
    hist = json.loads((HIST_RUN / "cf_eval_b6c.json").read_text())["families"]["SCE"][:10]
    n = 0
    for r in recs:
        c = r["cf"]
        t = hist[c["octet"]]["types"][c["type"]]
        h = t["near_miss"] if c["member"] == "near_miss" else t["members"][c["member"]]
        assert r["a"] == h["a"] and r["ok"] == h["ok"], (c, r["a"], h)
        n += 1
    assert n > 100
