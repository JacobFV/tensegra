"""Extended-05 B-XC tooling: split b5c_hold_uc (fresh U+C draws, seed base 5.9e9), trainer --split-set b5c, scorer
groups and the registered B-XC primary, seed-range registry, and bit-identity of every b5 path (split table v2, b5
labels / train / eval / references / scorer) against goldens captured with the UNMODIFIED tools (base 0923cff2).

Pure Python except the tests marked with ``importorskip("torch")``."""
import gzip
import hashlib
import importlib.util
import json
import pathlib
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


def _digest_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ split table v2 unchanged (pure Python)

# split table v2 (SPLITS5: definition, first 5 configurations / generator params / world seeds / family of every split,
# label/eval split lists, sizing constants), computed with the UNMODIFIED module (base 0923cff2)
B5_TABLE_GOLDEN = "f939fec80ed4607d"


def b5_table_digest():
    h = hashlib.sha256()
    for s in pw5.SPLITS5:
        h.update(repr((s, pw5.SPLITS5[s])).encode())
        for i in range(5):
            h.update(repr((pw5.split_config(s, i), pw5.generator_params(s, i), pw5.world_seed(s, i, 903),
                           pw5.family(s))).encode())
    h.update(repr((pw5.B5_LABEL_SPLITS, pw5.B5_EVAL_SPLITS, pw5.SENSITIVITY_FLAGS["b5_hold_uc"],
                   pw5.HOLD_TARGET_SENSITIVE["b5_hold_uc"], pw5.HOLD_TARGET_ELIGIBLE, pw5.B5_TRAINING_SPLITS,
                   pw5.NEW_HOLD_SPLITS, pw5.VERSION, pw5.SPLIT_TABLE_VERSION)).encode())
    return h.hexdigest()[:16]


def test_b5_split_table_unchanged():
    assert b5_table_digest() == B5_TABLE_GOLDEN
    assert "b5c_hold_uc" not in pw5.SPLITS5 and "b5c_hold_uc" not in pw.SPLITS
    assert pw5.SPLIT_SETS["b5"] == {"label_splits": pw5.B5_LABEL_SPLITS, "eval_splits": pw5.B5_EVAL_SPLITS}


def test_b5c_split_definition():
    cells, ks, combos, base = pw5.SPLITS5C["b5c_hold_uc"]
    assert (cells, ks, combos) == pw5.SPLITS5["b5_hold_uc"][:3] and base == 5_900_000_000
    assert [pw5.combo_name(c) for c in combos] == ["unreliable+correlated"]
    assert pw5.SPLIT_SETS["b5c"] == {"label_splits": (("b5c_hold_uc", "sized"),), "eval_splits": ("b5c_hold_uc",)}
    assert pw5.SENSITIVITY_FLAGS["b5c_hold_uc"] == (2,) and pw5.HOLD_TARGET_SENSITIVE["b5c_hold_uc"] == 20
    assert pw5.split_params("b5c_hold_uc") == pw5.SPLITS5C["b5c_hold_uc"]
    old = {pw5.split_config("b5_hold_uc", i) for i in range(300)}
    for i in range(50):
        cell, k, cmb, seed = pw5.generator_params("b5c_hold_uc", i)
        cfg = pw5.split_config("b5c_hold_uc", i)
        assert seed == base + i and cfg.k == k and cfg.flags == cmb and cfg not in old
        assert pw5.world_seed("b5c_hold_uc", i, 900) == base + pw.WORLD_SEED_OFFSET + 1000 * i + 900
        assert pw5.split_of_params(cell, k, cmb) == {"b5_hold_uc"}  # same family, never a training family
    assert pw5.family("b5c_hold_uc") == "b5c_hold_uc"


def test_seed_ranges_registry_b5c_disjoint():
    au = _load("research/tools/campaign05_bsplit_audit.py", "bsplit_audit_bxc")
    ranges = json.loads(au.SEED_RANGES.read_text())["ranges"]
    assert au.seed_range_overlaps(ranges) == []
    b5c = [r for r in ranges if r["lo"] == 5_900_000_000]
    assert len(b5c) == 1 and b5c[0]["hi"] == 6_000_000_000 and b5c[0]["campaign"] == "extended-05"
    # disjoint from every b5 / extended-04 probeworld range (configs and eval worlds of every split)
    for name, (_, _, _, base) in list(pw5.SPLITS5.items()) + list(pw.SPLITS.items()):
        lo, hi = base, base + pw.WORLD_SEED_OFFSET + 1000 * pw5.HOLD_MAX_CONFIGS + 1000
        assert not (lo < 6_000_000_000 and 5_900_000_000 < hi), name
        assert any(r["lo"] <= lo and hi <= r["hi"] for r in ranges), name  # every earlier probeworld range registered
    # B-XC training worlds (seeds 20-22) registered and disjoint (overlap check above)
    assert any(r["lo"] <= 8_000_000_000 + 20 * 100_000_000 and 8_000_000_000 + 23 * 100_000_000 <= r["hi"]
               for r in ranges)
    bad = ranges + [{"name": "bad", "campaign": "x", "lo": 5_950_000_000, "hi": 5_950_000_001}]
    assert au.seed_range_overlaps(bad)


def test_b5c_split_support_audit_small():
    import types
    au = _load("research/tools/campaign05_bsplit_audit.py", "bsplit_audit_bxc2")
    res = au.audit_b5c(types.SimpleNamespace(n_b5c=300, n_hist=2))
    assert res["all_pass"], [c for c in res["checks"] if not c["pass"]]
    assert {c["check"][:2] for c in res["checks"]} == {"E1", "E2", "E3", "E4", "F1", "F2", "F3", "F4", "F5", "G1", "G2"}


# ------------------------------------------------------------------ scorer: b5c groups and the B-XC primary

def _row(split, idx, pu, fp, gap=0.0, k=2, fs=None, not_opt_probe=False):
    """One episode (1 world per configuration).  pu: probe uniquely optimal at s0; fp: model probes first."""
    opt = [0] if pu else [1]
    a0 = 0 if fp else 1
    d0 = {"a": a0, "opt": opt, "delta": gap, "Q": [1.0, 2.0], "qhat": [1.0, 2.0], "avail": [0, 1],
          "step_in_query": 0, "probe_margin": 1.0 if pu else -1.0}
    r = {"split": split, "cfg_idx": idx, "rep": 0, "k": k, "combo": "unreliable+correlated", "U": 10.0 - gap,
         "V_star": 10.0, "success": 1.0, "pi_star_success": 1.0, "rho_eff": 1.0, "first_probe": int(fp),
         "probe_eps_opt": int(pu), "probe_unique_opt": int(pu), "decisions": [d0]}
    if fs is not None:
        r["flag_sensitive"] = fs
    return r


def _rows(n_probe, gap, split="b5c_hold_uc"):
    """40 eligible configurations (first 20 flag-sensitive), n_probe of them probed; 20 non-eligible, not probed."""
    rows = [_row(split, i, True, i < n_probe, gap=gap if i >= n_probe else 0.0, k=1 if i % 2 else 2, fs=i < 20)
            for i in range(40)]
    rows += [_row(split, 100 + i, False, False, k=8, fs=False) for i in range(20)]
    return rows


def _write_run(d, seed, train_split, rows, offset=900, name="eval_episodes.jsonl.gz"):
    d.mkdir(parents=True, exist_ok=True)
    head = {"_meta": {"rung": "L1", "seed": seed, "inputs": "public", "arch": "flat", "params": 129189,
                      "train_split": train_split, "q_head_trained": False, "world_offset": offset, "eps": 0.5}}
    with gzip.open(d / name, "wt") as f:
        f.write(json.dumps(head) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(d)


def test_scorer_b5c_groups_same_metrics_as_b5_hold_uc():
    pytest.importorskip("numpy")
    sc = _load("research/tools/campaign05_bx_score.py", "bxscore_bxc")
    rows_c = _rows(25, 3.0)
    rows_b = _rows(25, 3.0, split="b5_hold_uc")
    for g_c, g_b in (("b5c_hold_uc", "b5_hold_uc"), ("b5c_uc_flag_sensitive", "uc_flag_sensitive"),
                     ("b5c_uc_k1", "uc_k1"), ("b5c_uc_k_gt1", "uc_k_gt1")):
        cc = sc.clusters_of(rows_c, False, sc.GROUPS[g_c])
        cb = sc.clusters_of(rows_b, False, sc.GROUPS[g_b])
        assert cc and sc.metrics(sc._total(cc), False, sc._n_cfg_pu(cc)) == sc.metrics(sc._total(cb), False,
                                                                                       sc._n_cfg_pu(cb))
        assert not sc.clusters_of(rows_b, False, sc.GROUPS[g_c])  # b5c groups never select b5 rows
    m = sc.metrics(sc._total(sc.clusters_of(rows_c, False, sc.GROUPS["b5c_hold_uc"])), False)
    assert m["probe_unique"] == pytest.approx(25 / 40) and m["gap_regret"] == pytest.approx(15 * 3.0 / 60)
    assert "b5c_hold_uc" in sc.CI_GROUPS and "b5c_uc_flag_sensitive" in sc.CI_GROUPS


def _bxc(sc, tmp, b0, bx1, **kw):
    arms = {"B0": [_write_run(tmp / f"b0-s{s}", s, "b5_train", r, **kw) for s, r in zip((20, 21, 22), b0)],
            "BX1": [_write_run(tmp / f"bx1-s{s}", s, "b5x_train", r, **kw) for s, r in zip((20, 21, 22), bx1)]}
    return arms, sc.bxc_score(arms, n_boot=50)


def test_bxc_primary_rule(tmp_path):
    pytest.importorskip("numpy")
    sc = _load("research/tools/campaign05_bx_score.py", "bxscore_bxc2")
    b0 = [_rows(20, 3.0), _rows(22, 3.0), _rows(24, 3.0)]  # .50 .55 .60
    # confirmed: gains +.15 +.10 +.075 -> mean .108 >= .10, all > 0; gap regret lower 3/3
    arms, res = _bxc(sc, tmp_path / "a", b0, [_rows(26, 3.0), _rows(26, 3.0), _rows(27, 3.0)])
    assert res["valid"] and res["n_configurations"] == 60
    p = res["primary"]
    assert p["mean_probe_unique_gain"] == pytest.approx((6 + 4 + 3) / 40 / 3)
    assert p["all_3_pairs_positive"] and p["gap_regret_lower_3_of_3"] and p["pass"] and p["verdict"] == "confirmed"
    g = res["groups"]["b5c_hold_uc"]
    assert g["mean_diff_ci95"]["probe_unique"][0] <= p["mean_probe_unique_gain"] <= g["mean_diff_ci95"]["probe_unique"][1]
    assert set(res["groups"]) == set(sc.BXC_GROUPS) and res["secondary"]["BX1_not_opt_le_10_all_seeds"]
    assert res["secondary"]["flag_sensitive"]["probe_unique_gain"]["per_pair"]
    # mean gain below +.10 (+.10 +.10 +.05 -> .083)
    _, res = _bxc(sc, tmp_path / "b", b0, [_rows(24, 3.0), _rows(26, 3.0), _rows(26, 3.0)])
    assert not res["primary"]["mean_gain_ge_lift"] and res["primary"]["verdict"] == "not_confirmed"
    # a zero pair (big mean but one pair not > 0)
    _, res = _bxc(sc, tmp_path / "c", b0, [_rows(34, 3.0), _rows(34, 3.0), _rows(24, 3.0)])
    assert res["primary"]["mean_gain_ge_lift"] and not res["primary"]["all_3_pairs_positive"]
    assert not res["primary"]["pass"]
    # gap regret not lower in one pair (BX1 misses cost more per miss)
    _, res = _bxc(sc, tmp_path / "d", b0, [_rows(26, 3.0), _rows(26, 3.0), _rows(27, 9.0)])
    assert res["primary"]["all_3_pairs_positive"] and res["primary"]["mean_gain_ge_lift"]
    assert res["groups"]["b5c_hold_uc"]["gap_regret_lower"]["n_lower"] == 2 and not res["primary"]["pass"]
    # validity: wrong world offset -> invalid even when the rule passes
    _, res = _bxc(sc, tmp_path / "e", b0, [_rows(26, 3.0), _rows(26, 3.0), _rows(27, 3.0)], offset=500)
    assert res["primary"]["pass"] and not res["valid"] and res["primary"]["verdict"] == "invalid"
    # CLI
    out = tmp_path / "bxc.json"
    sc.main(["--bxc", "--arm", "B0", *arms["B0"], "--arm", "BX1", *arms["BX1"], "--out", str(out), "--n-boot", "20"])
    j = json.loads(out.read_text())
    assert j["B_XC"]["primary"]["verdict"] == "confirmed" and set(j["arms"]["B0"]["groups"]) == set(sc.BXC_GROUPS)


def test_scorer_b5_outputs_unaffected_by_b5c_groups(tmp_path):
    """On b5 rows the new groups select nothing: score() output == the unmodified scorer's (base 0923cff2) output."""
    pytest.importorskip("numpy")
    sc = _load("research/tools/campaign05_bx_score.py", "bxscore_bxc3")
    rows = _rows(25, 3.0, split="b5_hold_uc") + _rows(30, 2.0, split="b5_hold_se")
    rows2 = _rows(31, 3.0, split="b5_hold_uc") + _rows(33, 2.0, split="b5_hold_se")
    arms = {"B0": [_write_run(tmp_path / f"b0-{s}", s, "b5_train", rows) for s in (10, 11, 12)],
            "BX1": [_write_run(tmp_path / f"x-{s}", s, "b5x_train", rows2) for s in (10, 11, 12)]}
    res = sc.score(arms, n_boot=30)
    for arm in res["arms"].values():
        assert not any(g.startswith("b5c") for g in arm["groups"])
    assert _digest_obj(res) == SCORE_SYNTH_GOLDEN


# score() on the synthetic runs above, computed with the UNMODIFIED scorer (base 0923cff2; numpy default_rng)
SCORE_SYNTH_GOLDEN = "27e990696be4eeec"


# ------------------------------------------------------------------ torch: b5 bit-identity goldens and b5c end to end

TINY = ["--updates", "2", "--batch", "4", "--hidden", "16", "--log-every", "1"]
B5_LABEL_ARGS = ["--split-set", "b5", "--n-train", "6", "--n-train-x", "8", "--n-eval", "3", "--hold-target", "2",
                 "--hold-shard", "3", "--hold-max", "60", "--hold-target-sensitive", "0"]
B5_EVAL_ARGS = ["--worlds", "1", "--split-set", "b5", "--episode-rows", "--world-offset", "900",
                "--split-worlds", "b5_hold_uc=1", "b5_hold_se=1"]


def _sd_digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _strip_cpu(obj):
    if isinstance(obj, dict):
        return {k: _strip_cpu(v) for k, v in obj.items() if not k.startswith("cpu_s")}
    if isinstance(obj, list):
        return [_strip_cpu(v) for v in obj]
    return obj


def b5_tiny_digests(m, sc, tmp):
    """Digests of the b5 paths on tiny runs: labels --split-set b5 (pools, shards, s0 facts, meta), train B0 / BX1,
    eval --split-set b5 --episode-rows (offset 900), references --split-set b5, score() and --bhr-only.  Captured
    with the UNMODIFIED tools (base 0923cff2) and compared with the B-XC tools."""
    import torch
    tmp = pathlib.Path(tmp)
    lab = tmp / "b5"
    m.main(["labels", "--out", str(lab), *B5_LABEL_ARGS])
    out = {"labels": {p.name: (hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.suffix == ".pkl" else
                               _digest_obj(_strip_cpu(json.loads(p.read_text()))))
                      for p in sorted(lab.iterdir())}}
    runs = {}
    for arm, split in (("B0", "b5_train"), ("BX1", "b5x_train")):
        d = tmp / arm
        m.main(["train", "--labels", str(lab), "--out", str(d), "--rung", "L1", "--seed", "0", "--train-split", split,
                *TINY])
        m.main(["eval", "--labels", str(lab), "--run", str(d), *B5_EVAL_ARGS])
        with gzip.open(d / "eval_episodes.jsonl.gz", "rt") as f:
            ep = f.read()
        out[arm] = {"sd": _sd_digest(torch.load(d / "model.pt")),
                    "meta": _digest_obj(_strip_cpu(json.loads((d / "train_meta.json").read_text()))),
                    "log": _digest_obj(_strip_cpu(json.loads((d / "train_log.json").read_text()))),
                    "eval": _digest_obj(_strip_cpu(json.loads((d / "eval.json").read_text()))),
                    "episodes": hashlib.sha256(ep.encode()).hexdigest()[:16],
                    "files": sorted(p.name for p in d.iterdir())}
        runs[arm] = str(d)
    refs = tmp / "refs.json"
    m.main(["references", "--labels", str(lab), "--out", str(refs), "--worlds", "1", "--split-set", "b5",
            "--world-offset", "900", "--split-worlds", "b5_hold_uc=1", "b5_hold_se=1"])
    out["refs"] = _digest_obj(json.loads(refs.read_text()))
    out["score"] = _digest_obj(sc.score({"B0": [runs["B0"]], "BX1": [runs["BX1"]]}, n_boot=20))
    bhr = tmp / "bhr.json"
    sc.main(["--bhr-only", "--arm", "B0", runs["B0"], "--out", str(bhr), "--n-boot", "20"])
    out["bhr"] = _digest_obj(json.loads(bhr.read_text()))
    return out


# captured on the pro6000 with the UNMODIFIED tools of base 0923cff2 (metered: b5c-golden-20260927T031610)
_RUN_FILES = ["eval.json", "eval_episodes.jsonl.gz", "model.pt", "train_log.json", "train_meta.json"]
B5_TOOL_GOLDEN = {
    "torch": "2.14.0+cu130",
    "labels": {
        "b5_dev.pkl": "2255a5a5d7fe47aa", "b5_heldout_k.pkl": "51eb91477ab3e3b8",
        "b5_heldout_price.pkl": "e71ddb944e8b240b", "b5_hold_se.shard000.pkl": "000cf850a814b315",
        "b5_hold_se.shard001.pkl": "ffe92b61d2922337", "b5_hold_se.shards.json": "3d0636ab3faf2b3f",
        "b5_hold_se_s0.json": "426df800b6411b56", "b5_hold_uc.shard000.pkl": "f61c159e46c9a18b",
        "b5_hold_uc.shard001.pkl": "c8dbc5a7ed4ee211", "b5_hold_uc.shard002.pkl": "692a6e9b72d4620b",
        "b5_hold_uc.shard003.pkl": "618072491e93fd2e", "b5_hold_uc.shard004.pkl": "aedada332b727879",
        "b5_hold_uc.shard005.pkl": "7c7d6891bb6d1a07", "b5_hold_uc.shard006.pkl": "ccff49650ec79350",
        "b5_hold_uc.shard007.pkl": "3907f1e685efb822", "b5_hold_uc.shard008.pkl": "e3b3c3635e28a0d1",
        "b5_hold_uc.shard009.pkl": "665e24ab8a9fb8b5", "b5_hold_uc.shards.json": "eedbb821ea14fe38",
        "b5_hold_uc_s0.json": "596d7bce7cdd2f2e", "b5_test_iid.pkl": "815672460ac2a572",
        "b5_train.pkl": "cbef7bbfea69a92c", "b5x_train.pkl": "f65089abfc809dc6",
        "heldout_comp.pkl": "10a66d5d02212208", "labels_meta.json": "a69a3b1a1051081e"},
    "B0": {"sd": "ebe8e317cb892461", "meta": "80c5271eeefd20bd", "log": "92c22f256a3b15b5", "eval": "70148cc2271d6106",
           "episodes": "f4feff61f35f87cf", "files": _RUN_FILES},
    "BX1": {"sd": "5b8ea43cd2d52608", "meta": "13f8a343d21cd236", "log": "670285a4ca090d7b",
            "eval": "2ad2cb7d10ce6806", "episodes": "1033cfaed7559227", "files": _RUN_FILES},
    "refs": "731a9886ef8ffe33",
    "score": "b5689eb337e81b58",
    "bhr": "a72166664f59080a",
}


def test_b5_paths_bit_identical(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    m = _load("research/tools/campaign04_probeworld_train.py", "pwtrain_bxc")
    sc = _load("research/tools/campaign05_bx_score.py", "bxscore_bxc4")
    got = b5_tiny_digests(m, sc, tmp_path)
    assert got["B0"]["files"] == B5_TOOL_GOLDEN["B0"]["files"]
    if torch.__version__ != B5_TOOL_GOLDEN["torch"]:
        pytest.skip("golden digests pinned on the pro6000 torch build only")
    assert got == {k: v for k, v in B5_TOOL_GOLDEN.items() if k != "torch"}


def test_b5c_end_to_end_tiny(tmp_path):
    """labels --split-set b5c builds ONLY b5c_hold_uc; models trained on a b5 labels dir are evaluated against the
    b5c labels dir (eval / references --split-set b5c, --world-offset 900, --split-worlds, --episode-rows); --bxc."""
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    m = _load("research/tools/campaign04_probeworld_train.py", "pwtrain_bxc2")
    sc = _load("research/tools/campaign05_bx_score.py", "bxscore_bxc5")
    lab5 = tmp_path / "b5"
    m.main(["labels", "--out", str(lab5), *B5_LABEL_ARGS])
    labc = tmp_path / "b5c"
    m.main(["labels", "--out", str(labc), "--split-set", "b5c", "--hold-target", "3", "--hold-shard", "2",
            "--hold-max", "200", "--hold-target-sensitive", "1"])
    files = sorted(p.name for p in labc.iterdir())
    info = json.loads((labc / "b5c_hold_uc.shards.json").read_text())
    assert files == sorted(["labels_meta.json", "b5c_hold_uc.shards.json", "b5c_hold_uc_s0.json"] +
                           [f"{s}.pkl" for s in info["shards"]])
    meta = json.loads((labc / "labels_meta.json").read_text())
    assert set(meta) == {"b5c_hold_uc", "version", "split_set", "eps", "continuation"}
    assert meta["split_set"]["name"] == "b5c" and meta["split_set"]["splits"]["b5c_hold_uc"]["config_seed_base"] == 5_900_000_000
    s0 = json.loads((labc / "b5c_hold_uc_s0.json").read_text())
    assert info["reached_target"] and pw5.hold_pool_size(s0, 3, 1) == info["n_configs"] == len(s0)
    assert info["target_eligible_flag_sensitive"] == 1 and all("flag_sensitive" in r for r in s0)
    assert all(r["combo"] == "unreliable+correlated" for r in s0)
    runs = {"B0": [], "BX1": []}
    for arm, split in (("B0", "b5_train"), ("BX1", "b5x_train")):
        for seed in (20, 21, 22):
            d = tmp_path / f"{arm}-s{seed}"
            m.main(["train", "--labels", str(lab5), "--out", str(d), "--rung", "L1", "--seed", str(seed),
                    "--train-split", split, *TINY])
            m.main(["eval", "--labels", str(labc), "--run", str(d), "--split-set", "b5c", "--episode-rows",
                    "--world-offset", "900", "--split-worlds", "b5c_hold_uc=1"])
            e = json.loads((d / "eval.json").read_text())
            assert list(e["splits"]) == ["b5c_hold_uc"] and "sharded" in e["splits"]["b5c_hold_uc"]["free_running_greedy"]
            runs[arm].append(str(d))
    head, rows = sc.read_rows(runs["B0"][0])
    assert head["world_offset"] == 900 and head["train_split"] == "b5_train"
    assert {r["split"] for r in rows} == {"b5c_hold_uc"} and len(rows) == info["n_configs"]
    assert sorted(r["cfg_idx"] for r in rows) == list(range(info["n_configs"])) and {r["rep"] for r in rows} == {0}
    s0map = {r["idx"]: r for r in s0}
    for r in rows:
        assert r["flag_sensitive"] == s0map[r["cfg_idx"]]["flag_sensitive"]
        assert r["probe_unique_opt"] == s0map[r["cfg_idx"]]["probe_unique"]
        # the world is the registered b5c world (offset 900, rep 0): pi* outcome reproduced from the seed
        assert r["gap_regret"] == pytest.approx(sum(d["delta"] for d in r["decisions"]))
    refs = tmp_path / "refs.json"
    m.main(["references", "--labels", str(labc), "--out", str(refs), "--split-set", "b5c", "--world-offset", "900",
            "--split-worlds", "b5c_hold_uc=1"])
    assert list(json.loads(refs.read_text())) == ["b5c_hold_uc"]
    out = tmp_path / "bxc.json"
    sc.main(["--bxc", "--arm", "B0", *runs["B0"], "--arm", "BX1", *runs["BX1"], "--out", str(out), "--n-boot", "20"])
    res = json.loads(out.read_text())["B_XC"]
    assert res["valid"], res["validity"]
    assert res["primary"]["verdict"] in ("confirmed", "not_confirmed")
    assert len(res["groups"]["b5c_hold_uc"]["pairs"]) == 3


if __name__ == "__main__":
    # golden capture (run from a checkout of the UNMODIFIED base with this file copied into its tests/):
    #   python tests/test_campaign05_bxc.py OUTDIR
    import tempfile
    import torch
    m = _load("research/tools/campaign04_probeworld_train.py", "pwtrain_cap")
    sc = _load("research/tools/campaign05_bx_score.py", "bxscore_cap")
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows(25, 3.0, split="b5_hold_uc") + _rows(30, 2.0, split="b5_hold_se")
    rows2 = _rows(31, 3.0, split="b5_hold_uc") + _rows(33, 2.0, split="b5_hold_se")
    arms = {"B0": [_write_run(out / "synth" / f"b0-{s}", s, "b5_train", rows) for s in (10, 11, 12)],
            "BX1": [_write_run(out / "synth" / f"x-{s}", s, "b5x_train", rows2) for s in (10, 11, 12)]}
    print(json.dumps({"score_synth": _digest_obj(sc.score(arms, n_boot=30)), "b5_table": b5_table_digest(),
                      "b5_tool": {"torch": torch.__version__, **b5_tiny_digests(m, sc, out / "tiny")}}, indent=1))
