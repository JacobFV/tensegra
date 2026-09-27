"""Extended-05 Track B tooling: split table v2 (B-SPLIT), supplied public state (BX2/BO), modular arm (BX3),
B-LOC localization, B-X scorer, and bit-identity of the extended-04 B1/B2/F2 trainer paths under default options.

Pure Python except the tests marked with ``importorskip("torch")``."""
import hashlib
import importlib.util
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


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _train_module():
    return _load("research/tools/campaign04_probeworld_train.py", "pwtrain_e05")


# ------------------------------------------------------------------ bit-identity goldens (extended-04 paths)

TINY = ["--updates", "4", "--batch", "6", "--hidden", "16", "--log-every", "1"]


def _digest_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _sd_digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def b1_tiny_digests(m, tmp):
    """Digests of the extended-04 code paths on tiny runs: labels (B1 split set), B1 train L1/L4, B1 eval (offset
    500), F2 eval (offset 700), B2 own-value on, references.  Used both to pin goldens with the UNMODIFIED tool
    (extended-05 base d3e1d279's trainer = 7851b490) and to test that the extended-05 additions leave these paths
    bit-identical under default options."""
    import torch
    tmp = pathlib.Path(tmp)
    lab = tmp / "labels"
    m.main(["labels", "--out", str(lab), "--n-train", "6", "--n-eval", "3"])
    out = {"labels": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(lab.glob("*.pkl"))}}
    meta = json.loads((lab / "labels_meta.json").read_text())
    out["labels_meta"] = _digest_obj({k: ({kk: vv for kk, vv in v.items() if kk != "cpu_s"} if isinstance(v, dict) else v)
                                      for k, v in meta.items()})
    strip_log = lambda rows: [{k: v for k, v in r.items() if not k.startswith("cpu_s")} for r in rows]
    for name, rung, extra, offset in (("L1", "L1", [], None), ("L4", "L4", [], None), ("L4f2", "L4", [], "700"),
                                      ("L4b2", "L4", ["--own-value", "--own-episodes", "5"], None)):
        run = tmp / name
        m.main(["train", "--labels", str(lab), "--out", str(run), "--rung", rung, "--seed", "0", *TINY, *extra])
        ev = ["eval", "--labels", str(lab), "--run", str(run), "--worlds", "1"]
        if offset:
            ev += ["--world-offset", offset]
        m.main(ev)
        tm = json.loads((run / "train_meta.json").read_text())
        e = json.loads((run / "eval.json").read_text())
        e.pop("cpu_s")
        rec = {"sd": _sd_digest(torch.load(run / "model.pt")),
               "log": _digest_obj(strip_log(json.loads((run / "train_log.json").read_text()))),
               "meta": _digest_obj({k: v for k, v in tm.items() if not k.startswith("cpu_s")
                                    and k != "own_value"}),
               "eval": _digest_obj(e), "files": sorted(p.name for p in run.iterdir())}
        if (run / "failure_records.json").exists():
            rec["failure_records"] = _digest_obj(json.loads((run / "failure_records.json").read_text()))
        out[name] = rec
    refs = tmp / "refs.json"
    m.main(["references", "--labels", str(lab), "--out", str(refs), "--worlds", "1"])
    out["refs"] = _digest_obj(json.loads(refs.read_text()))
    return out


# captured on the pro6000 with the UNMODIFIED extended-04 trainer (metered: trackb-golden-20260927T013415)
B1_TOOL_GOLDEN = {
    "torch": "2.14.0+cu130",
    "labels": {"dev.pkl": "cbf00030d9693d97", "heldout_comp.pkl": "10a66d5d02212208",
               "heldout_k.pkl": "20fa95ae873cc3c3", "heldout_price.pkl": "916f42867a349c4a",
               "test_iid.pkl": "530052a6752f221b", "train.pkl": "4c365ee2102d4f5a"},
    "labels_meta": "0d03b02b40d28869",
    "L1": {"sd": "75edde6a146bd480", "log": "f0af51878556d5a9", "meta": "e61f3c2f6877a8fd", "eval": "26a6c395b867e767",
           "files": ["eval.json", "model.pt", "train_log.json", "train_meta.json"]},
    "L4": {"sd": "37cc0f2d2ef73e01", "log": "835a704736872bdd", "meta": "f536b56af95a70c4", "eval": "eddfa6f178ec528a",
           "files": ["eval.json", "model.pt", "train_log.json", "train_meta.json"]},
    "L4f2": {"sd": "37cc0f2d2ef73e01", "log": "835a704736872bdd", "meta": "f536b56af95a70c4",
             "eval": "1c9cba2891c2424d", "files": ["eval.json", "model.pt", "train_log.json", "train_meta.json"]},
    "L4b2": {"sd": "8a56e019481ae1d0", "log": "835a704736872bdd", "meta": "4e1e97476400b79d",
             "eval": "b99c4582d461afbf", "files": ["eval.json", "failure_records.json", "model.pt",
                                                   "own_value_log.json", "train_log.json", "train_meta.json"],
             "failure_records": "397b93edd475a7ee"},
    "refs": "e5233499e4b42cf8",
}
# extended-04 split table: first 5 configurations and generator parameters of every split (unmodified module)
PW_SPLITS_GOLDEN = "33472494f731e5bb"


def test_b1_b2_f2_paths_bit_identical_under_defaults(tmp_path):
    torch = pytest.importorskip("torch")
    got = b1_tiny_digests(_train_module(), tmp_path)
    # structure is build-independent: no extended-05 key/file appears under defaults
    for name in ("L1", "L4", "L4f2", "L4b2"):
        assert got[name]["files"] == B1_TOOL_GOLDEN[name]["files"]
    assert got["L4"]["sd"] == got["L4f2"]["sd"] and got["L4"]["log"] == got["L4f2"]["log"]
    if torch.__version__ != B1_TOOL_GOLDEN["torch"]:
        pytest.skip("golden digests pinned on the pro6000 torch build only")
    assert got == {k: v for k, v in B1_TOOL_GOLDEN.items() if k != "torch"}


# ------------------------------------------------------------------ split table v2 (B-SPLIT), pure Python

def test_extended04_split_table_unchanged():
    h = hashlib.sha256()
    for s in pw.SPLITS:
        for i in range(5):
            h.update(repr(pw.split_config(s, i)).encode())
            h.update(repr(pw.generator_params(s, i)).encode())
    assert h.hexdigest()[:16] == PW_SPLITS_GOLDEN
    # the old table keeps U+C and S+E in training (historical reproduction); v2 removes them
    assert set(pw5.NEW_HOLD_COMBOS) <= set(pw.TRAIN_COMBOS)
    for s in pw.SPLITS:
        assert pw5.split_config(s, 7) == pw.split_config(s, 7)
        assert pw5.world_seed(s, 7, 503) == pw.world_seed(s, 7, 503)


def test_split_table_v2_holds_and_support():
    names = {pw5.combo_name(c) for c in pw5.NEW_HOLD_COMBOS}
    assert names == {"unreliable+correlated", "side_effect+events"}
    assert {pw5.combo_name(c) for c in pw5.HIST_COMBOS} == {"unreliable+events", "side_effect+correlated"}
    for split, (cells, ks, combos, base) in pw5.SPLITS5.items():
        assert base not in {v[3] for v in pw.SPLITS.values()}
        if split not in pw5.NEW_HOLD_SPLITS:
            assert not set(pw5.NEW_HOLD_COMBOS) & set(combos), split  # holds never in training/dev/selection
        if split != "b5x_train" and split not in pw5.NEW_HOLD_SPLITS:
            assert not set(pw5.HIST_COMBOS) & set(combos), split  # historical pairs only in the exposure pool
    assert set(pw5.HIST_COMBOS) <= set(pw5.SPLITS5["b5x_train"][2])
    assert len(pw5.B5_TRAIN_COMBOS) == 7 and len(pw5.B5X_TRAIN_COMBOS) == 9
    for c in pw5.NEW_HOLD_COMBOS:  # every flag of a hold is seen singly in training
        for j, f in enumerate(c):
            if f:
                assert tuple(jj == j for jj in range(4)) in pw5.SPLITS5["b5_train"][2]
    seen = set()
    for split in pw5.SPLITS5:
        for idx in range(40):
            cell, k, cmb, seed = pw5.generator_params(split, idx)
            cfg = pw5.split_config(split, idx)
            assert cfg.k == k and cfg.flags == tuple(cmb) and seed not in seen
            seen.add(seed)
            fams = pw5.split_of_params(cell, k, cmb)
            if split in pw5.B5_TRAINING_SPLITS:
                assert not fams & set(pw5.NEW_HOLD_SPLITS)
            if split in pw5.NEW_HOLD_SPLITS:
                assert fams == {split}


def test_hold_pool_size_rule_and_flag_sensitivity():
    recs = [{"probe_unique": x} for x in (0, 1, 0, 1, 1, 0)]
    assert pw5.hold_pool_size(recs, 2) == 4 and pw5.hold_pool_size(recs, 3) == 5 and pw5.hold_pool_size(recs, 4) is None
    recs2 = [{"probe_unique": 1, "flag_sensitive": f} for f in (False, False, True, False, True)]
    assert pw5.hold_pool_size(recs2, 2) == 2 and pw5.hold_pool_size(recs2, 2, 2) == 5
    assert pw5.hold_pool_size(recs2, 2, 3) is None
    # correlated is inert at k = 1 unless the clipped P(H | z) makes the mixture differ from the prior
    cfg = pw.Config(prior=(0.5, 0.3, 0.1, 0.1), q=0.6, corr=0.2, k=1, C_build=1e9)
    V = pw.ExactSolver(cfg).value(pw.initial_state(cfg))
    fs = pw5.flag_sensitivity(cfg, V, (2,))
    assert fs["checked_flags"] == ["correlated"] and fs["flag_sensitive"] is False
    cfg2 = pw.with_(cfg, k=3)
    fs2 = pw5.flag_sensitivity(cfg2, pw.ExactSolver(cfg2).value(pw.initial_state(cfg2)), (2,))
    assert fs2["flag_sensitive"] is True


def test_foreign_policy_evaluation_and_ablation():
    cfg = pw.Config(prior=(0.6, 0.2, 0.1, 0.1), q=0.6, p_event=0.2, k=1, C_build=1e9)
    s = pw.ExactSolver(cfg)
    V = s.value(pw.initial_state(cfg))
    assert pw5.evaluate_foreign_policy(s, pw.ExactSolver(cfg)) == pytest.approx(V, abs=1e-9)
    for j in (0, 3):
        abl = pw5.ablate(cfg, j)
        assert not abl.flags[j] and sum(abl.flags) == 1
        assert pw5.evaluate_foreign_policy(s, pw.ExactSolver(abl)) <= V + 1e-9
    ci = pw5.composition_interaction(cfg)
    assert set(ci["naive_regret"]) == {"without_unreliable", "without_events"}
    assert all(v >= -1e-9 for v in ci["naive_regret"].values())


def test_supplied_features_public_and_history_function():
    assert pw5.supplied_dim("public") == 0 and pw5.supplied_dim("belief") == 4
    assert pw5.supplied_dim("bx2") == len(pw5.BX2_FEATURES)
    rng = random.Random(3)
    for split in ("b5_hold_uc", "b5_hold_se", "b5_train"):
        for idx in range(4):
            cfg = pw5.split_config(split, idx)
            seen = {}
            for w in range(15):
                ep = pw.Episode(cfg, 77 + w)
                while not ep.done:
                    st = ep.state
                    f = pw5.supplied_features(cfg, st, "bx2")
                    assert len(f) == len(pw5.BX2_FEATURES) and all(x == x for x in f)
                    assert f[:4] == list(st[1][0]) == pw5.supplied_features(cfg, st, "belief")
                    rebuilt = pw.public_state_from_history(cfg, ep.history)
                    assert pw5.supplied_features(cfg, rebuilt, "bx2") == f
                    key = tuple(ep.history)
                    assert seen.setdefault(key, f) == f
                    assert pw5.supplied_features(cfg, st, "public") == []
                    ep.step(rng.choice(ep.available()))


def test_decision_record_and_rank_pairs():
    cfg = pw.worked_example_config()
    s = pw.ExactSolver(cfg)
    st = pw.initial_state(cfg)
    d = pw5.decision_record(s, st, pw.A_B1)
    assert d["opt"] == [pw.A_PROBE] and d["delta"] == pytest.approx(79.0)
    assert d["probe_margin"] == pytest.approx(79.0)
    assert pw5.q_rank_pairs([0, 1, 2], [10.0, 5.0, 5.2], [3.0, 1.0, 9.0]) == (2, 1)  # (0,1) right, (0,2) wrong


def test_bsplit_audit_small():
    au = _load("research/tools/campaign05_bsplit_audit.py", "bsplit_audit_t")
    import types
    res = au.audit(types.SimpleNamespace(n_pair=2, n_hist=1))
    bad = [c for c in res["checks"] if not c["pass"] and not c["check"].startswith("A2")]  # A2 needs a real sample
    assert not bad, bad
    assert set(res["pairs"]) == {pw5.combo_name(c) for c in pw5.ALL_PAIRS}


# ------------------------------------------------------------------ B-X scorer (synthetic rows)

def _row(split, idx, pu, fp, ok=True, k=1, fs=None):
    d0 = {"a": 0 if fp else 1, "opt": [0] if pu else [1], "delta": 0.0 if (fp == pu) else 3.0, "Q": [1.0, 2.0],
          "qhat": [1.0, 2.0], "avail": [0, 1], "step_in_query": 0, "probe_margin": 1.0 if pu else -1.0}
    r = {"split": split, "cfg_idx": idx, "rep": 0, "k": k, "combo": "x", "U": 10.0 - d0["delta"], "V_star": 10.0,
         "success": 1.0, "pi_star_success": 1.0, "rho_eff": 1.0, "first_probe": int(fp), "probe_eps_opt": int(pu),
         "probe_unique_opt": int(pu), "decisions": [d0]}
    if fs is not None:
        r["flag_sensitive"] = fs
    return r


def test_bx_scorer_metrics_and_bhr_gate():
    pytest.importorskip("numpy")
    sc = _load("research/tools/campaign05_bx_score.py", "bxscore_t")
    rows = [_row("b5_hold_se", i, pu=True, fp=i < 15) for i in range(25)] + \
           [_row("b5_hold_se", 100 + i, pu=False, fp=False) for i in range(10)]
    cl = sc.clusters_of(rows, False, sc.GROUPS["b5_hold_se"])
    m = sc.metrics(sc._total(cl), False, sc._n_cfg_pu(cl))
    assert m["probe_unique"] == pytest.approx(15 / 25) and m["n_probe_unique_configs"] == 25
    assert m["probe_not_opt"] == 0.0 and m["q_rank_acc"] == "n/a" and "probe_unique" not in m["insufficient"]
    assert m["by_depth"]["s0"]["n"] == 35
    (ci,), n = sc.bootstrap([cl], [False], n_boot=200)
    assert n == 35 and ci["probe_unique"][0] <= 0.6 <= ci["probe_unique"][1]
    groups = {"b5_hold_uc": [{"probe_unique": 0.9, "n_probe_unique": 30, "n_probe_unique_configs": 30}] * 3,
              "b5_hold_se": [{"probe_unique": 0.5, "n_probe_unique": 30, "n_probe_unique_configs": 30}] * 3,
              "uc_flag_sensitive": [{"probe_unique": 0.5, "n_probe_unique": 5, "n_probe_unique_configs": 5}] * 3}
    g = sc.bhr_gate(groups)
    assert g["b5_hold_uc"]["verdict"] == "no_failure" and g["b5_hold_se"]["verdict"] == "evaluable"
    assert g["uc_flag_sensitive"]["verdict"] == "insufficient"


# ------------------------------------------------------------------ torch: b5 end to end, BX3, replay, B-LOC

def test_bx3_modular_param_match_and_gating():
    torch = pytest.importorskip("torch")
    m = _train_module()
    h = m.matched_hidden("modular")
    net = m.ProbeNet(h, arch="modular")
    assert abs(m.n_params(net) - m.B1_PARAMS) / m.B1_PARAMS < 0.01
    assert m.n_params(m.ProbeNet(128)) == m.B1_PARAMS
    # the modules are gated by the public flag bits: all flags off -> identical to the flat pre-activation
    torch.manual_seed(0)
    net = m.ProbeNet(32, arch="modular")
    cfg = pw.Config()
    x = torch.tensor([m.encode(cfg.public_vector(), None, (pw.A_PROBE, pw.A_B1), 0.0)])
    h0 = torch.zeros(1, 32)
    hm, _ = net.step(x, h0)
    assert torch.allclose(hm, net.gru(torch.tanh(net.inp(x)), h0))
    cfg2 = pw5.split_config("b5_hold_uc", 0)
    x2 = torch.tensor([m.encode(cfg2.public_vector(), None, (pw.A_PROBE, pw.A_B1), 0.0)])
    assert not torch.allclose(net.step(x2, h0)[0], net.gru(torch.tanh(net.inp(x2)), h0))
    # matched_hidden never perturbs the caller's torch RNG
    torch.manual_seed(5)
    a = torch.rand(3)
    torch.manual_seed(5)
    m.matched_hidden("modular")
    assert torch.equal(a, torch.rand(3))


def test_b5_end_to_end_tiny(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    m = _train_module()
    lab = tmp_path / "b5"
    m.main(["labels", "--out", str(lab), "--split-set", "b5", "--n-train", "6", "--n-train-x", "8", "--n-eval", "3",
            "--hold-target", "2", "--hold-shard", "3", "--hold-max", "60", "--hold-target-sensitive", "0"])
    meta = json.loads((lab / "labels_meta.json").read_text())
    assert meta["split_set"]["split_table"] == pw5.SPLIT_TABLE_VERSION
    for hold in pw5.NEW_HOLD_SPLITS:
        info = json.loads((lab / f"{hold}.shards.json").read_text())
        s0 = json.loads((lab / f"{hold}_s0.json").read_text())
        assert info["reached_target"] and info["eligible"] == 2 and len(s0) == info["n_configs"]
        assert s0[-1]["probe_unique"] and pw5.hold_pool_size(s0, 2) == info["n_configs"]
        assert len(info["shards"]) == -(-info["n_configs"] // 3)
        pool = m.load_pool(lab, hold)
        assert [i for i, _, _ in pool] == list(range(info["n_configs"]))
        if hold == "b5_hold_uc":
            assert all("flag_sensitive" in r for r in s0)
    tr = json.loads((lab / "labels_meta.json").read_text())
    assert "unreliable+correlated" not in tr["b5_train"]["combo_counts"]
    tiny = ["--updates", "2", "--batch", "4", "--hidden", "16", "--log-every", "1"]
    arms = {"B0": [], "BX1": ["--train-split", "b5x_train"], "BX2": ["--inputs", "bx2"],
            "BX3": ["--arch", "modular"]}
    runs = {}
    for arm, extra in arms.items():
        d = tmp_path / arm
        base = [] if "--train-split" in extra else ["--train-split", "b5_train"]
        m.main(["train", "--labels", str(lab), "--out", str(d), "--rung", "L1", "--seed", "0", *base, *extra,
                *(tiny if arm != "BX3" else [x for x in tiny if x not in ("--hidden", "16")]), ])
        m.main(["eval", "--labels", str(lab), "--run", str(d), "--worlds", "1", "--split-set", "b5", "--episode-rows",
                "--split-worlds", "b5_hold_uc=1", "b5_hold_se=1"])
        runs[arm] = d
        tm = json.loads((d / "train_meta.json").read_text())
        assert tm["train_split"] in ("b5_train", "b5x_train")
        e = json.loads((d / "eval.json").read_text())
        assert set(e["splits"]) == set(pw5.B5_EVAL_SPLITS)
        assert "sharded" in e["splits"]["b5_hold_uc"]["free_running_greedy"]
    assert json.loads((runs["BX3"] / "train_meta.json").read_text())["arch"] == "modular"
    assert json.loads((runs["BX2"] / "train_meta.json").read_text())["in_dim"] == m.IN_DIM + len(pw5.BX2_FEATURES)
    head, rows = _load("research/tools/campaign05_bx_score.py", "bxs_e2e").read_rows(runs["B0"])
    assert head["q_head_trained"] is False and {r["split"] for r in rows} == set(pw5.B5_EVAL_SPLITS)
    for r in rows:  # exact labels in the records
        for d in r["decisions"]:
            assert d["delta"] >= -1e-9 and d["a"] in d["avail"] and set(d["opt"]) <= set(d["avail"])
        assert r["gap_regret"] == pytest.approx(sum(d["delta"] for d in r["decisions"]))
    uc = [r for r in rows if r["split"] == "b5_hold_uc"]
    assert all("flag_sensitive" in r for r in uc) and len({r["cfg_idx"] for r in uc}) == len(uc)
    m.main(["references", "--labels", str(lab), "--out", str(tmp_path / "refs.json"), "--worlds", "1",
            "--split-set", "b5", "--split-worlds", "b5_hold_uc=1", "b5_hold_se=1"])
    sc = _load("research/tools/campaign05_bx_score.py", "bxs_e2e2")
    res = sc.score({a: [str(runs[a])] for a in runs}, n_boot=20)
    assert set(res["arms"]) == set(runs) and "B_HR" in res and set(res["BX_primary"]) == {"BX1", "BX2", "BX3"}


def test_replay_history_matches_greedy_rollout():
    torch = pytest.importorskip("torch")
    m = _train_module()
    torch.manual_seed(1)
    for inputs, arch in (("public", "flat"), ("belief", "flat"), ("bx2", "flat"), ("public", "modular")):
        net = m.ProbeNet(24, inputs=inputs, arch=arch)
        items = []
        for idx in range(3):
            cfg = pw5.split_config("b5_hold_se", idx)
            items.append((cfg, pw.ExactSolver(cfg), 1000 + idx))
        with torch.no_grad():
            eps, steps, ep_steps = m.run_batch(net, items, "greedy", need_labels=False)
            for i, info_list in enumerate(ep_steps):
                hist = [info["rec"] for info in info_list]
                rep = m.replay_history(net, items[i][0], hist)
                assert len(rep) == len(info_list)
                for (av, logits, qh), info in zip(rep, info_list):
                    assert int(logits.argmax()) == info["a"]
                    rec = steps[info["step_row"]]
                    ref = net.pi(rec["z"]).masked_fill(~rec["mask"], -1e9)[info["row"]]
                    assert torch.allclose(logits, ref, atol=1e-5)


def test_bloc_classification_tiny(tmp_path):
    torch = pytest.importorskip("torch")
    m = _train_module()
    lab = tmp_path / "labels"
    m.main(["labels", "--out", str(lab), "--n-train", "6", "--n-eval", "4"])
    tiny = ["--updates", "2", "--batch", "4", "--hidden", "16", "--log-every", "1"]
    for name, extra in (("b-train-L4-s0", ["--rung", "L4"]), ("f-btrain-L1-s3", ["--rung", "L1"]),
                        ("bo-s0", ["--rung", "L1", "--inputs", "belief"])):
        m.main(["train", "--labels", str(lab), "--out", str(tmp_path / name / "run"), "--seed", "0", *extra, *tiny])
    bl = _load("research/tools/campaign05_bloc.py", "bloc_t")
    out = tmp_path / "bloc.json"
    bl.main(["--labels", str(lab), "--runs", str(tmp_path / "b-train-L4-s0/run"), str(tmp_path / "f-btrain-L1-s3/run"),
             "--out", str(out), "--records", str(tmp_path / "rec.jsonl.gz"), "--bo-runs", str(tmp_path / "bo-s0/run"),
             "--worlds", "2"])
    res = json.loads(out.read_text())
    l4, l1 = res["models"]["b-train-L4-s0"], res["models"]["f-btrain-L1-s3"]
    assert l4["world_offset"] == 500 and l1["world_offset"] == 700
    assert l4["all"]["q_head_trained"] is True and l1["all"]["q_head_trained"] is False
    for ent in (l4, l1):
        a = ent["all"]
        assert sum(a["first_error_class_counts"].values()) == a["episodes_with_consequential_error"]
        assert set(ent["by_pair"]) <= {"unreliable+events", "side_effect+correlated"}
        assert a["bo_available"] == (a["episodes_with_consequential_error"] > 0)
    # L1 has no trained Q head: never ranking / value_estimate
    assert l1["all"]["first_error_class_counts"]["ranking"] == 0
    assert l1["all"]["first_error_class_counts"]["value_estimate"] == 0
    # classification rules on hand-made records
    d = {"a": 1, "opt": [0], "avail": [0, 1, 2], "qhat": [5.0, 1.0, 0.0], "tie_opt": False}
    assert bl.classify(d, True, None)[0] == "ranking"
    assert bl.classify({**d, "qhat": [0.0, 5.0, 1.0]}, True, None)[0] == "value_estimate"
    assert bl.classify({**d, "qhat": [0.0, 5.0, 1.0]}, True, {"argmax": [0], "eps_opt": True})[0] == "belief_dependent"
    assert bl.classify({**d, "tie_opt": True}, True, None)[0] == "deployment"
    assert bl.classify(d, False, None)[0] == "unlocalized"


if __name__ == "__main__":  # golden capture: python tests/test_campaign05_trackb.py OUTDIR
    import torch
    print(json.dumps({"torch": torch.__version__, **b1_tiny_digests(_train_module(), sys.argv[1])}, indent=1))
