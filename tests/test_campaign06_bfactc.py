"""Extended-06 B-FACT-C: the fresh confirmation split (split set b6c) and the two-level pooled primary.

Everything is additive: the b6 split table (campaign06_probeworld.split_table_digest_input), its audit, and the default
campaign06_bprimary output are pinned by hashes taken before the b6c code existed.  No test generates a configuration
from the registered b6c bases: the tool tests redirect SPLITS6C / CF_BASE_C to the dev_smoke sub-range."""
import gzip
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
    from tensegra import campaign04_probeworld as pw  # noqa: F401
from tensegra import campaign06_probeworld as pw6

sys.path.insert(0, str(ROOT / "research" / "tools"))

# pinned before any b6c code (base commit 07fe43bf)
B6_SPLIT_TABLE_SHA = "625566129bace0cd602c710202f19775b91b4b10f904408d591087e49b7c477f"
B6_AUDIT_SHA = "3e198bd3f5dbec51688375d85782c54990f5b70e35b8764819a9365348d5792d"
BPRIMARY_DEFAULT_JSON_SHA = "51730cb09708060eabf0276eb33cde35a7a4de1e7199eb50f8dfa36c7f856039"
BPRIMARY_DEFAULT_MD_SHA = "886437a7ba754fc36960ea74610dbad9cb35c16c6d261197748c2151cce87695"

DEV_HOLD = 6_840_000_000  # dev_smoke sub-range (never protocol); worlds up to 6.894e9 stay inside it
DEV_CF = 6_830_000_000


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ split table: b6 unchanged, b6c disjoint

def test_b6_split_table_bit_identical():
    assert hashlib.sha256(repr(pw6.split_table_digest_input()).encode()).hexdigest() == B6_SPLIT_TABLE_SHA
    assert hashlib.sha256(json.dumps(pw6.audit_split_table(384), sort_keys=True).encode()).hexdigest() == B6_AUDIT_SHA
    assert pw6.SPLIT_TABLE_VERSION == "probeworld-split-v3"
    assert "b6c_hold_SCE" not in pw6.SPLITS6 and pw6.B6_EVAL_SPLITS == tuple(
        s for s, v in pw6.SPLITS6.items() if v[2] in ("test", "hold", "hist"))


def test_b6c_bases_registered_and_disjoint():
    au = pw6.audit_b6c()
    assert au["pass"] and au["protocol_bases"], au
    assert pw6.SPLITS6C["b6c_hold_SCE"] == (("SCE",), 6_420_000_000, "hold")
    assert pw6.CF_BASE_C == {"SCE": 6_650_000_000}
    lo, hi, _ = pw6.SUBRANGES["hold"]
    assert lo <= 6_420_000_000 and 6_420_000_000 + pw6.WORLD_SEED_OFFSET + pw6.MAX_POOL * 1000 <= hi
    lo, hi, _ = pw6.SUBRANGES["cf"]
    assert lo <= 6_650_000_000 and 6_650_000_000 + pw6.MAX_POOL <= hi
    assert 6_650_000_000 not in pw6.CF_BASE.values()
    # the originally proposed hold base 6.45e9 is the eval-world block of b6_hold_SCE: the audit must reject it
    old = dict(pw6.SPLITS6C)
    try:
        pw6.SPLITS6C["b6c_hold_SCE"] = (("SCE",), 6_450_000_000, "hold")
        assert not pw6.audit_b6c()["disjoint"]
    finally:
        pw6.SPLITS6C.clear()
        pw6.SPLITS6C.update(old)
    reg = json.loads((ROOT / "research/campaigns/extended-06/seed-ranges.json").read_text())["ranges"]
    names = [r for r in reg if r["name"].startswith("ext06 B-FACT-C")]
    assert len(names) == 3
    spans = sorted((r["lo"], r["hi"], r["parent"]) for r in names)
    hb, cb = 6_420_000_000, 6_650_000_000
    assert spans == [(hb, hb + pw6.MAX_POOL, "ext06 Track B v3: hold"),
                     (hb + pw6.WORLD_SEED_OFFSET, hb + pw6.WORLD_SEED_OFFSET + pw6.MAX_POOL * 1000,
                      "ext06 Track B v3: hold"),
                     (cb, cb + pw6.MAX_POOL, "ext06 Track B v3: cf")]
    # registered blocks are exactly the audited b6c blocks
    assert sorted((lo, hi) for lo, hi, _ in pw6._pool_blocks(pw6.SPLITS6C, pw6.CF_BASE_C, "")) == \
        [(lo, hi) for lo, hi, _ in spans]


def test_b6c_config_same_generator():
    """b6c configurations use the b6 generator path with only the base changed (checked at a dev_smoke base)."""
    old = dict(pw6.SPLITS6C)
    try:
        pw6.SPLITS6C["b6c_hold_SCE"] = (("SCE",), DEV_HOLD, "hold")
        for i in range(3):
            c = pw6.split6_config("b6c_hold_SCE", i)
            cell, k, key = pw6.draw_params(DEV_HOLD + i, pw.TRAIN_CELLS, pw6.V3_K, ("SCE",))
            assert c == pw6.make_config(cell, k, pw6.flags_from_key(key), DEV_HOLD + i)
            assert pw6.world_seed6("b6c_hold_SCE", i, 0) == DEV_HOLD + pw6.WORLD_SEED_OFFSET + 1000 * i
    finally:
        pw6.SPLITS6C.clear()
        pw6.SPLITS6C.update(old)
    assert pw6.world_seed6("b6_hold_SCE", 0, 0) == 6_450_000_000  # b6 unchanged (and why 6.45e9 was rejected)


# ------------------------------------------------------------------ bprimary: default unchanged, two-level rule

DTYPES = ["first", "after_probe_solved", "q2_after_H"]
MEMBERS = ["0", "S", "C", "E", "SC", "SE", "CE", "SCE"]


def make_run(d, seed, bias, group="b6_hold_SCE", n_cfg=30, n_oct=12, eval_name="eval_episodes.jsonl.gz",
             cf_name="cf_eval.json"):
    """Synthetic eval run (episode rows + cf records) in the format campaign06_bscore reads; deterministic."""
    rng = random.Random(seed * 7919 + int(bias * 1000))
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(d / eval_name, "wb", mtime=0) as g:
        lines = [json.dumps({"_meta": {"seed": seed, "rung": "L1"}})]
        for split in (group, "b6_test_base"):
            for i in range(n_cfg):
                ds = []
                for t in range(rng.randint(1, 4)):
                    ds.append({"opt": [rng.randrange(4)] if rng.random() < .6 else [0, 1], "a": rng.randrange(4),
                               "delta": 0.0 if rng.random() < .5 + bias else rng.random() * 3,
                               "step_in_query": t % 2, "rec": [rng.randrange(3), rng.randrange(3)]})
                lines.append(json.dumps({"split": split, "cfg_idx": i, "rep": 0, "decisions": ds,
                                         "gap_regret": round(rng.random() * 10 - 20 * bias, 4), "V_star": 50.0,
                                         "U": round(40 + rng.random() * 10, 4), "success": rng.random() < .7,
                                         "pi_star_success": True, "eligible": rng.random() < .9}))
        g.write(("\n".join(lines) + "\n").encode())
    sets = []
    for i in range(n_oct):
        types = {}
        for h in DTYPES:
            mem = {k: {"a": rng.randrange(3), "ok": rng.random() < .5 + bias, "opt": [rng.randrange(2)],
                       "unique": rng.random() < .8} for k in MEMBERS}
            nm = {"a": 0, "ok": rng.random() < .5 + bias, "opt": [0]} if rng.random() < .7 else None
            types[h] = {"flip": rng.random() < .6, "unique": True, "members": mem, "near_miss": nm}
        sets.append({"index": i, "k": 2, "types": types})
    (d / cf_name).write_text(json.dumps({"_meta": {"eps": .5, "decision_types": DTYPES}, "families": {"SCE": sets}}))
    return str(d)


def _fixture(tmp, seeds=(30, 31, 32), **kw):
    ref = [make_run(tmp / f"ref{s}", s, 0.0, **kw) for s in seeds]
    arm = [make_run(tmp / f"arm{s}", s, 0.1, **kw) for s in seeds]
    return ref, arm


def test_bprimary_default_bit_identical(tmp_path):
    bp = _load("research/tools/campaign06_bprimary.py", "bprimary_default")
    ref, arm = _fixture(tmp_path)
    bp.main(["--ref", "B0", *ref, "--arm", "B2", *arm, "--out", str(tmp_path / "p.json"), "--md",
             str(tmp_path / "p.md"), "--n-boot", "50"])
    assert _sha(tmp_path / "p.json") == BPRIMARY_DEFAULT_JSON_SHA
    assert _sha(tmp_path / "p.md") == BPRIMARY_DEFAULT_MD_SHA


def test_bprimary_two_level_rule(tmp_path):
    bp = _load("research/tools/campaign06_bprimary.py", "bprimary_two")
    seeds = (35, 36, 37, 38, 39)
    kw = dict(group="b6c_hold_SCE", eval_name="eval_b6c_episodes.jsonl.gz", cf_name="cf_eval_b6c.json")
    ref, arm = _fixture(tmp_path, seeds, **kw)
    out = tmp_path / "q.json"
    bp.main(["--group", "b6c_hold_SCE", "--two-level", "--ref", "RAWF", *ref, "--arm", "LRN", *arm,
             "--out", str(out), "--md", str(tmp_path / "q.md"), "--n-boot", "60"])
    res = json.loads(out.read_text())
    assert res["group"] == "b6c_hold_SCE" and res["ref"] == "RAWF" and res["min_sign_pairs"] == 4
    assert res["files"] == ["eval_b6c_episodes.jsonl.gz", "cf_eval_b6c.json"]
    c = res["contrasts"]["LRN-RAWF"]
    assert c["n_pairs"] == 5 and c["seeds"] == [[s, s] for s in seeds] and c["bootstrap"].startswith("two-level")
    for name, e in c["endpoints"].items():
        per = e["per_pair"]
        assert e["signs"] == {"negative": sum(x < 0 for x in per), "positive": sum(x > 0 for x in per),
                              "zero": sum(x == 0 for x in per)}
        assert abs(e["mean"] - sum(per) / 5) < 1e-12
        assert e["ci"][0] <= e["ci"][1]
    # same per-pair differences as the configuration-only contrast
    A = [bp.load(r, "b6c_hold_SCE") for r in arm]
    R = [bp.load(r, "b6c_hold_SCE") for r in ref]
    one = bp.contrast(A, R, 60, seed=1000)
    for name in c["endpoints"]:
        assert one["endpoints"][name]["per_pair"] == c["endpoints"][name]["per_pair"]
    assert bp.contrast_two_level(A, R, 60, seed=1000)["endpoints"] == c["endpoints"]
    v, e = c["verdict"], c["endpoints"]
    assert v["Q1"] == (e["gap_regret"]["mean"] < 0 and e["gap_regret"]["ci"][1] < 0
                       and e["gap_regret"]["signs"]["negative"] >= 4)
    assert v["Q2"] == (e["later_acc"]["mean"] > 0 and e["later_acc"]["ci"][0] > 0
                       and e["later_acc"]["signs"]["positive"] >= 4)
    assert v["Q3"] == (e["cf_near_miss_acc"]["mean"] >= -.02 and e["iv_false_change"]["mean"] <= .02)
    assert v["SUPPORTED"] == (v["Q1"] and v["Q2"] and v["Q3"])
    # the sign rule binds: 3/5 negative fails Q1 even with a negative mean and CI; 4/5 passes
    e2 = json.loads(json.dumps(c))
    e2["endpoints"]["gap_regret"].update(mean=-1.0, ci=[-2.0, -0.5], signs={"negative": 3, "positive": 2, "zero": 0})
    assert not bp.verdict_two_level(e2)["Q1"]
    e2["endpoints"]["gap_regret"]["signs"] = {"negative": 4, "positive": 1, "zero": 0}
    assert bp.verdict_two_level(e2)["Q1"]
    # a hand-rolled two-level bootstrap (pairs, then configurations; one draw shared by the pairs) reproduces the CI
    rng = random.Random(1000)
    keys = sorted(set.intersection(*[set(x["S"]) for x in A + R]))
    ckeys = sorted(set.intersection(*[set(x["C"]) for x in A + R]))
    boots = []
    for _ in range(60):
        pix = [rng.randrange(5) for _ in range(5)]
        ix = [rng.randrange(len(keys)) for _ in keys]
        for _ in ckeys:
            rng.randrange(len(ckeys))
        boots.append(sum(bp.gap([A[q]["S"][keys[i]] for i in ix]) - bp.gap([R[q]["S"][keys[i]] for i in ix])
                         for q in pix) / 5)
    boots.sort()
    assert c["endpoints"]["gap_regret"]["ci"] == [boots[int(0.025 * 60)], boots[int(0.975 * 60) - 1]]


def test_bprimary_group_flag_without_two_level(tmp_path):
    """--group b6_hold_SCE (the default value) alone is the B-ARMS tool; --group b6c_* reads the b6c file names."""
    bp = _load("research/tools/campaign06_bprimary.py", "bprimary_group")
    ref, arm = _fixture(tmp_path)
    bp.main(["--group", "b6_hold_SCE", "--ref", "B0", *ref, "--arm", "B2", *arm, "--out", str(tmp_path / "p.json"),
             "--md", str(tmp_path / "p.md"), "--n-boot", "50"])
    assert _sha(tmp_path / "p.json") == BPRIMARY_DEFAULT_JSON_SHA
    with pytest.raises(FileNotFoundError):
        bp.main(["--group", "b6c_hold_SCE", "--ref", "B0", *ref, "--arm", "B2", *arm, "--out",
                 str(tmp_path / "x.json"), "--n-boot", "5"])


def test_bscore_b6c_file_names(tmp_path):
    sc = _load("research/tools/campaign06_bscore.py", "bscore_b6c")
    kw = dict(group="b6c_hold_SCE", eval_name="eval_b6c_episodes.jsonl.gz", cf_name="cf_eval_b6c.json")
    ref, arm = _fixture(tmp_path, (35,), **kw)
    sc.main(["--arm", "RAWF", *ref, "--arm", "LRN", *arm, "--ref", "RAWF", "--b6c", "--out", str(tmp_path / "s.json"),
             "--n-boot", "10"])
    res = json.loads((tmp_path / "s.json").read_text())
    assert "b6c_hold_SCE" in res["arms"]["LRN"]["groups"] and "cf:pooled" in res["arms"]["LRN"]["groups"]


# ------------------------------------------------------------------ torch: b6c labels + eval end to end (dev_smoke)

TINY = ["--updates", "2", "--batch", "4", "--hidden", "16", "--log-every", "1"]


def test_b6c_end_to_end_dev_smoke(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    m = _load("research/tools/campaign04_probeworld_train.py", "pwtrain_b6c")
    bp = _load("research/tools/campaign06_bprimary.py", "bprimary_e2e")
    monkeypatch.setattr(pw6, "V3_K", (1, 2))  # tiny and cheap
    # TEST HOOK: the b6c bases are redirected to dev_smoke; no configuration is generated from 6.42e9 / 6.65e9
    monkeypatch.setitem(pw6.SPLITS6C, "b6c_hold_SCE", (("SCE",), DEV_HOLD, "hold"))
    monkeypatch.setitem(pw6.CF_BASE_C, "SCE", DEV_CF)
    au = pw6.audit_b6c()
    assert au["pass"] and au["dev_smoke_bases"] and not au["protocol_bases"]
    lab = tmp_path / "b6"
    m.main(["labels", "--out", str(lab), "--split-set", "b6", "--parts", "train", "--n-train", "12", "--arms", "B0"])
    labc = tmp_path / "b6c"
    with pytest.raises(SystemExit):
        m.main(["labels", "--out", str(labc), "--split-set", "b6c", "--parts", "train"])
    m.main(["labels", "--out", str(labc), "--split-set", "b6c", "--parts", "eval", "--n-hold", "3", "--b6-shard", "2"])
    m.main(["labels", "--out", str(labc), "--split-set", "b6c", "--parts", "cf", "--n-cf", "2"])
    assert sorted(p.name for p in labc.glob("labels_meta*")) == ["labels_meta.b6c.cf.json", "labels_meta.b6c.eval.json"]
    assert sorted(p.name for p in labc.glob("cf_*.json")) == ["cf_SCE.json"]
    assert not list(labc.glob("b6_*"))  # only the b6c pool
    info = json.loads((labc / "b6c_hold_SCE.shards.json").read_text())
    assert info["config_seed_base"] == DEV_HOLD and info["n_configs"] == 3 and len(info["shards"]) == 2
    s0 = json.loads((labc / "b6c_hold_SCE_s0.json").read_text())
    assert [r["family"] for r in s0] == ["SCE"] * 3
    sets = json.loads((labc / "cf_SCE.json").read_text())
    assert [s["seed"] for s in sets] == [DEV_CF, DEV_CF + 1]
    # the octets are exactly counterfactual_set at the b6c base (same construction / near-miss logic)
    assert json.loads(json.dumps(pw6.counterfactual_set("SCE", 1, base=DEV_CF))) == sets[1]
    runs = {}
    for name, extra in (("SUP", ["--inputs", "factors6", "--arch", "fuse"]),
                        ("LRN", ["--arch", "fuse", "--factor-mode", "learned"]),
                        ("RAWF", ["--arch", "fuse"])):
        d = tmp_path / name
        m.main(["train", "--labels", str(lab), "--out", str(d), "--rung", "L1", "--seed", "35",
                "--train-split", "b6_B0", *extra, *TINY])
        (d / "eval.json").write_text("sentinel")  # a run's b6 eval files must never be touched by b6c
        m.main(["eval", "--labels", str(labc), "--run", str(d), "--split-set", "b6c", "--episode-rows", "--cf",
                "--world-offset", "0", "--worlds", "1"])
        assert (d / "eval.json").read_text() == "sentinel" and not (d / "cf_eval.json").exists()
        ev = json.loads((d / "eval_b6c.json").read_text())
        assert list(ev["splits"]) == ["b6c_hold_SCE"] and ev["split_set"] == "b6c"
        with gzip.open(d / "eval_b6c_episodes.jsonl.gz", "rt") as f:
            head = json.loads(f.readline())["_meta"]
            rows = [json.loads(x) for x in f if x.strip()]
        assert head["split_set"] == "b6c" and head["world_offset"] == 0
        assert {r["split"] for r in rows} == {"b6c_hold_SCE"} and len(rows) == 3
        cf = json.loads((d / "cf_eval_b6c.json").read_text())
        assert set(cf["families"]) == {"SCE"} and len(cf["families"]["SCE"]) == 2
        runs[name] = str(d)
    assert [pw6.world_seed6("b6c_hold_SCE", i, 0) for i in range(3)] == \
        [DEV_HOLD + pw6.WORLD_SEED_OFFSET + 1000 * i for i in range(3)]
    # b6c eval refuses a b6 labels dir
    with pytest.raises(AssertionError):
        m.main(["eval", "--labels", str(lab), "--run", runs["RAWF"], "--split-set", "b6c", "--worlds", "1"])
    out = tmp_path / "bfactc.json"
    bp.main(["--group", "b6c_hold_SCE", "--two-level", "--min-sign", "1", "--ref", "RAWF", runs["RAWF"],
             "--arm", "LRN", runs["LRN"], "--arm", "SUP", runs["SUP"], "--out", str(out), "--n-boot", "20"])
    res = json.loads(out.read_text())
    assert set(res["contrasts"]) == {"LRN-RAWF", "SUP-RAWF"}
    assert res["contrasts"]["LRN-RAWF"]["support"]["configs"] == 3
