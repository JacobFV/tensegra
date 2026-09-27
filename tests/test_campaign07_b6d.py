"""Extended-07 P2-CONFIRM: the fresh confirmation split (split set b6d).

Everything is additive: the b6 split table / audit (hashes pinned before the b6c code), the b6c table / audit and a
sample of b6 generator outputs are pinned by hashes taken at base c7d22aff (before any b6d code).  No test generates a
configuration or world from the registered b6d bases (6.43e9 / 6.48e9 / 6.66e9): the tool tests redirect SPLITS6D /
CF_BASE_D to the dev_smoke sub-range (registered in extended-07/seed-ranges.json as 'ext07 dev_smoke: b6d tooling
tests ...'): hold configs 6.846e9, eval worlds 6.896e9, octets 6.835e9, bank worlds 6.837e9; training worlds of the
dev seed 97."""
import gzip
import hashlib
import json
import pathlib
import shutil
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

# pinned before any b6c code (base 07fe43bf; tests/test_campaign06_bfactc.py)
B6_SPLIT_TABLE_SHA = "625566129bace0cd602c710202f19775b91b4b10f904408d591087e49b7c477f"
B6_AUDIT_SHA = "3e198bd3f5dbec51688375d85782c54990f5b70e35b8764819a9365348d5792d"
# pinned before any b6d code (base c7d22aff)
B6C_AUDIT_SHA = "bbe578a896a9e0fb354f349f8544460409a9761da728b303bd6dc42382062aec"
B6C_TABLE_SHA = "c7536f18946697b91241f1b7cf07bbe6b70d0ef5b9dbb0d08b4d241755944737"
B6_GEN_SHA = "2a531fa8234fd4208ef7c3b2a24d732dd377ed822dea936237f7f4edd468b521"

DEV_HOLD_D = 6_846_000_000  # dev_smoke; eval worlds 6.896e9 .. 6.9e9 (MAX_POOL block) stay inside it
DEV_CF_D = 6_835_000_000
DEV_BANK_W = 6_837_000_000
DEV_SEED = 97
HB, WB, CB = 6_430_000_000, 6_480_000_000, 6_660_000_000  # registered protocol bases (never generated from here)


def _sha(obj):
    return hashlib.sha256(obj.encode()).hexdigest()


# ------------------------------------------------------------------ b6 and b6c bit-identical

def test_b6_and_b6c_bit_identical():
    assert _sha(repr(pw6.split_table_digest_input())) == B6_SPLIT_TABLE_SHA
    assert _sha(json.dumps(pw6.audit_split_table(384), sort_keys=True)) == B6_AUDIT_SHA
    assert _sha(json.dumps(pw6.audit_b6c(), sort_keys=True)) == B6C_AUDIT_SHA
    t = (pw6.B6C_VERSION, tuple(sorted(pw6.SPLITS6C.items())), tuple(sorted(pw6.CF_BASE_C.items())),
         pw6.B6C_EVAL_SPLITS)
    assert _sha(repr(t)) == B6C_TABLE_SHA
    h = hashlib.sha256()
    for s in ("b6_hold_SCE", "b6_test_base", "b6_hist_" + pw6.HIST_FAMILIES[0]):
        for i in range(3):
            h.update(repr(pw6.split6_config(s, i)).encode())
            h.update(str(pw6.world_seed6(s, i, 1)).encode())
    h.update(str([pw6.world_seed6("b6c_hold_SCE", i, r) for i in (0, 5, 399) for r in (0, 1)]).encode())
    assert h.hexdigest() == B6_GEN_SHA
    assert not set(pw6.SPLITS6D) & (set(pw6.SPLITS6) | set(pw6.SPLITS6C))
    assert pw6.B6_EVAL_SPLITS == tuple(s for s, v in pw6.SPLITS6.items() if v[2] in ("test", "hold", "hist"))
    assert pw6.B6C_EVAL_SPLITS == ("b6c_hold_SCE",)


# ------------------------------------------------------------------ b6d bases, registry, disjointness

def test_b6d_bases_registered_and_disjoint():
    au = pw6.audit_b6d()
    assert au["pass"] and au["protocol_bases"] and not au["dev_smoke_bases"], au
    assert pw6.SPLITS6D == {"b6d_hold_SCE": (("SCE",), HB, "hold")} and pw6.CF_BASE_D == {"SCE": CB}
    assert HB + pw6.WORLD_SEED_OFFSET == WB and pw6.B6D_EVAL_SPLITS == ("b6d_hold_SCE",)
    lo, hi, _ = pw6.SUBRANGES["hold"]
    assert lo <= HB and WB + pw6.MAX_POOL * 1000 <= hi
    lo, hi, _ = pw6.SUBRANGES["cf"]
    assert lo <= CB and CB + pw6.MAX_POOL <= hi
    # registered P2-CONFIRM blocks (extended-07/seed-ranges.json) lie inside the audited b6d blocks
    reg = json.loads((ROOT / "research/campaigns/extended-07/seed-ranges.json").read_text())["ranges"]
    ent = {r["name"]: (r["lo"], r["hi"]) for r in reg if r["name"].startswith("ext07 P2 CONFIRM fresh S+C+E")}
    assert sorted(ent.values()) == [(HB, HB + pw6.MAX_POOL), (WB, WB + 400 * 1000), (CB, CB + pw6.MAX_POOL)]
    blocks = pw6._pool_blocks(pw6.SPLITS6D, pw6.CF_BASE_D, "")
    assert all(any(b[0] <= lo and hi <= b[1] for b in blocks) for lo, hi in ent.values())
    # the eval worlds of the 400 registered configurations (any rep < 1000) are inside the registered world block
    assert pw6.world_seed6("b6d_hold_SCE", 0, 0) == WB and pw6.world_seed6("b6d_hold_SCE", 399, 999) < WB + 400_000
    # every b6 / b6c block (pools, eval worlds, training streams, octets) is disjoint from every b6d block
    others = pw6._pool_blocks(pw6.SPLITS6, pw6.CF_BASE, "") + pw6._pool_blocks(pw6.SPLITS6C, pw6.CF_BASE_C, "") + [
        (pw6.type_seed(k, 0), pw6.type_seed(k, 0) + pw6.TYPE_STREAM_STRIDE, "") for k in pw6.TRAIN_TYPES]
    assert au["n_blocks_checked"] == len(others)
    assert all(b[1] <= o[0] or o[1] <= b[0] for b in blocks for o in others)


@pytest.mark.parametrize("hold,cf", [(6_420_000_000, CB),   # b6c hold configs
                                     (6_400_000_000, CB),   # b6_hold_SCE configs
                                     (6_450_000_000, CB),   # b6_hold_SCE eval worlds
                                     (6_470_000_000, CB),   # b6c eval worlds
                                     (6_200_000_000, CB),   # training streams
                                     (HB, 6_650_000_000),   # b6c octets
                                     (HB, 6_600_000_000)])  # b6 SCE octets
def test_audit_rejects_overlaps(monkeypatch, hold, cf):
    """Audit only (no configuration is generated): a b6d base on any b6 / b6c block fails disjointness."""
    monkeypatch.setitem(pw6.SPLITS6D, "b6d_hold_SCE", (("SCE",), hold, "hold"))
    monkeypatch.setitem(pw6.CF_BASE_D, "SCE", cf)
    au = pw6.audit_b6d()
    assert not au["disjoint"] and not au["pass"]


def test_audit_rejects_outside_subranges(monkeypatch):
    monkeypatch.setitem(pw6.SPLITS6D, "b6d_hold_SCE", (("SCE",), 6_585_000_000, "hold"))  # 'hist' / 'cf'
    au = pw6.audit_b6d()
    assert au["disjoint"] and not au["inside_subranges"] and not au["pass"]
    monkeypatch.setitem(pw6.SPLITS6D, "b6d_hold_SCE", (("UC",), HB, "hold"))  # not a held-out family
    assert not pw6.audit_b6d()["families_are_holds"]


def test_b6d_config_same_generator(monkeypatch):
    """b6d configurations use the b6 generator path with only the base changed (at a dev_smoke base)."""
    monkeypatch.setitem(pw6.SPLITS6D, "b6d_hold_SCE", (("SCE",), DEV_HOLD_D, "hold"))
    monkeypatch.setitem(pw6.CF_BASE_D, "SCE", DEV_CF_D)
    au = pw6.audit_b6d()
    assert au["pass"] and au["dev_smoke_bases"] and not au["protocol_bases"]
    assert pw6.is_split6("b6d_hold_SCE") and pw6.split6_entry("b6d_hold_SCE")[1] == DEV_HOLD_D
    for i in range(2):
        c = pw6.split6_config("b6d_hold_SCE", i)
        cell, k, key = pw6.draw_params(DEV_HOLD_D + i, pw.TRAIN_CELLS, pw6.V3_K, ("SCE",))
        assert c == pw6.make_config(cell, k, pw6.flags_from_key(key), DEV_HOLD_D + i)
        assert pw6.world_seed6("b6d_hold_SCE", i, 0) == DEV_HOLD_D + pw6.WORLD_SEED_OFFSET + 1000 * i
    with pytest.raises(KeyError):
        pw6.split6_entry("b6x_hold_SCE")


# ------------------------------------------------------------------ job commands

def test_b6d_jobs_validate():
    pytest.importorskip("torch")
    import campaign07_p2 as P
    jobs = P.b6d_jobs("abcdef12")
    names = [n for _, n, _ in jobs]
    assert names[:2] == ["e07-p2c-labels-eval", "e07-p2c-labels-cf"] and len(names) == 2 + len(P.CONFIRM_SEEDS)
    assert P.CONFIRM_SEEDS == (50, 51, 52, 53, 54)
    ev, cf = jobs[0][2], jobs[1][2]
    lab = ["research/tools/campaign04_probeworld_train.py", "labels", "--split-set", "b6d", "--parts"]
    assert ev[ev.index("--") + 1:] == P.ENV + lab + ["eval", "--n-hold", "400", "--out", P.B6D]
    assert cf[cf.index("--") + 1:] == P.ENV + lab + ["cf", "--n-cf", "320", "--out", P.B6D]
    assert P.B6D == "/home/brand/structured-latent-dynamics-campaign07/results/e07-p2c-labels/labels"
    for argv in (ev, cf):
        assert argv[argv.index("--max-concurrent") + 1] == "6" and argv[argv.index("--cpu-cap") + 1] == "8000"
        assert argv[argv.index("--wall-cap") + 1] == "14400"
    for _, name, argv in jobs[2:]:
        cmd = argv[argv.index("--") + 1:]
        assert cmd[cmd.index("--pool") + 1] == "b6d_hold_SCE" and cmd[cmd.index("--labels") + 1] == P.B6D
        assert [cmd[i + 1] for i, x in enumerate(cmd) if x == "--cf"] == [f"{P.B6D}/cf_SCE.json"]
        specs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--model"]
        assert sum("::exact" in s for s in specs) == 3 and sum("::pred=" in s for s in specs) == 3
        assert len(specs) == 4 + 6 and "b6c" not in " ".join(cmd) and "UCE" not in " ".join(cmd)


# ------------------------------------------------------------------ torch: b6d labels + eval + diag (dev_smoke)

TINY = ["--updates", "2", "--batch", "4", "--hidden", "16", "--log-every", "1"]


def _files(d):
    return {p.name: p.stat().st_mtime_ns for p in pathlib.Path(d).iterdir()}


def test_b6d_end_to_end_dev_smoke(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import campaign04_probeworld_train as T
    import campaign07_diag as D
    import campaign07_diagscore as S
    import campaign07_p2 as P
    monkeypatch.setattr(pw6, "V3_K", (1, 2))  # tiny and cheap
    # TEST HOOK: the b6d bases are redirected to dev_smoke; nothing is generated from 6.43e9 / 6.48e9 / 6.66e9
    monkeypatch.setitem(pw6.SPLITS6D, "b6d_hold_SCE", (("SCE",), DEV_HOLD_D, "hold"))
    monkeypatch.setitem(pw6.CF_BASE_D, "SCE", DEV_CF_D)
    au = pw6.audit_b6d()
    assert au["pass"] and au["dev_smoke_bases"] and not au["protocol_bases"]
    lab = tmp_path / "b6"
    T.main(["labels", "--out", str(lab), "--split-set", "b6", "--parts", "train", "--n-train", "12", "--arms", "B0"])
    labd = tmp_path / "b6d"
    with pytest.raises(SystemExit):
        T.main(["labels", "--out", str(labd), "--split-set", "b6d", "--parts", "train"])
    T.main(["labels", "--out", str(labd), "--split-set", "b6d", "--parts", "eval", "--n-hold", "3", "--b6-shard", "2"])
    T.main(["labels", "--out", str(labd), "--split-set", "b6d", "--parts", "cf", "--n-cf", "2"])
    assert sorted(p.name for p in labd.glob("labels_meta*")) == ["labels_meta.b6d.cf.json", "labels_meta.b6d.eval.json"]
    assert sorted(p.name for p in labd.glob("cf_*.json")) == ["cf_SCE.json"]
    assert not list(labd.glob("b6_*")) and not list(labd.glob("b6c_*"))  # only the b6d pool
    meta = json.loads((labd / "labels_meta.b6d.eval.json").read_text())["split_set"]
    assert meta["name"] == "b6d" and meta["registry"] == "P2-CONFIRM" and meta["audit"]["dev_smoke_bases"]
    info = json.loads((labd / "b6d_hold_SCE.shards.json").read_text())
    assert info["config_seed_base"] == DEV_HOLD_D and info["n_configs"] == 3 and len(info["shards"]) == 2
    assert [r["family"] for r in json.loads((labd / "b6d_hold_SCE_s0.json").read_text())] == ["SCE"] * 3
    sets = json.loads((labd / "cf_SCE.json").read_text())
    assert [s["seed"] for s in sets] == [DEV_CF_D, DEV_CF_D + 1]
    assert json.loads(json.dumps(pw6.counterfactual_set("SCE", 1, base=DEV_CF_D))) == sets[1]

    # models: historical-style arms, the S x R 2x2 arms, a bank predictor and an exact-contract consumer
    common = ["train", "--labels", str(lab), "--train-split", "b6_B0", "--rung", "L1", "--seed", str(DEV_SEED)]
    runs = {}
    for name, extra in (("RAWF", ["--arch", "fuse"]), ("LRN", ["--arch", "fuse", "--factor-mode", "learned"]),
                        ("S0R1", ["--arch", "fuse", "--shape", "0", "--read", "1"]),
                        ("S1R0", ["--arch", "fuse", "--shape", "1", "--read", "0"])):
        runs[name] = tmp_path / name
        T.main([*common, "--out", str(runs[name]), *extra, *TINY])
    bank = tmp_path / "bank" / "bank.pkl"
    P.main(["bank", "--labels", str(lab), "--pool", "b6_B0", "--n-configs", "6", "--world-base", str(DEV_BANK_W),
            "--source", "pistar:1", "--out", str(bank)])
    bk = ["--bank", str(bank), "--bank-updates", "2", "--updates", "0", "--batch", "4", "--hidden", "16",
          "--log-every", "1"]
    runs["PRED"] = tmp_path / "PRED"
    T.main([*common, "--out", str(runs["PRED"]), "--arch", "fuse", "--shape", "1", "--read", "0", "--predictor-only", *bk])
    runs["CONS"] = tmp_path / "CONS"
    T.main([*common, "--out", str(runs["CONS"]), "--inputs", "factors6", "--arch", "fuse", "--phi-contract", "exact",
            *bk])

    # trainer eval --split-set b6d: only the three b6d files are written
    for name in ("RAWF", "LRN", "S0R1"):
        d = runs[name]
        (d / "eval.json").write_text("sentinel")
        (d / "eval_b6c.json").write_text("sentinel")
        before = _files(d)
        with pytest.raises(SystemExit):  # fixed output names
            T.main(["eval", "--labels", str(labd), "--run", str(d), "--split-set", "b6d", "--out-name", "x.json",
                    "--world-offset", "0", "--worlds", "1"])
        T.main(["eval", "--labels", str(labd), "--run", str(d), "--split-set", "b6d", "--episode-rows", "--cf",
                "--world-offset", "0", "--worlds", "1"])
        after = _files(d)
        assert {k: v for k, v in after.items() if k in before} == before
        assert sorted(set(after) - set(before)) == ["cf_eval_b6d.json", "eval_b6d.json", "eval_b6d_episodes.jsonl.gz"]
        ev = json.loads((d / "eval_b6d.json").read_text())
        assert list(ev["splits"]) == ["b6d_hold_SCE"] and ev["split_set"] == "b6d"
        with gzip.open(d / "eval_b6d_episodes.jsonl.gz", "rt") as f:
            head = json.loads(f.readline())["_meta"]
            rows = [json.loads(x) for x in f if x.strip()]
        assert head["split_set"] == "b6d" and head["world_offset"] == 0
        assert {r["split"] for r in rows} == {"b6d_hold_SCE"} and len(rows) == 3
        cf = json.loads((d / "cf_eval_b6d.json").read_text())
        assert set(cf["families"]) == {"SCE"} and len(cf["families"]["SCE"]) == 2
    assert [pw6.world_seed6("b6d_hold_SCE", i, 0) for i in range(3)] == \
        [DEV_HOLD_D + pw6.WORLD_SEED_OFFSET + 1000 * i for i in range(3)]
    # b6d eval refuses a b6 labels dir and a b6d dir built from other bases
    with pytest.raises(AssertionError):
        T.main(["eval", "--labels", str(lab), "--run", str(runs["RAWF"]), "--split-set", "b6d", "--worlds", "1"])
    with monkeypatch.context() as m2:
        m2.setitem(pw6.SPLITS6D, "b6d_hold_SCE", (("SCE",), DEV_HOLD_D + 10_000, "hold"))
        with pytest.raises(AssertionError):
            T.check_b6d_labels(labd, ["b6d_hold_SCE"], False)
    with monkeypatch.context() as m2:
        m2.setitem(pw6.CF_BASE_D, "SCE", DEV_CF_D + 10_000)
        with pytest.raises(AssertionError):
            T.check_b6d_labels(labd, [], True)

    # diag on the b6d pool + b6d octets: 2x2 arms, LRN / RAWF, consumers as '::exact' and '::pred=PREDRUN'
    out = tmp_path / "diag"
    models = sum((["--model", f"{n}={runs[n]}"] for n in ("RAWF", "LRN", "S0R1", "S1R0")), [])
    models += ["--model", f"CX={runs['CONS']}::exact", "--model", f"CP={runs['CONS']}::pred={runs['PRED']}"]
    D.main(["run", "--labels", str(labd), "--pool", "b6d_hold_SCE", "--cf", str(labd / "cf_SCE.json"), *models,
            "--protocols", "B", "A-pistar", "--ivs", "none", "exact", "--no-latent", "--check-historical",
            "--worlds", "1", "--world-offset", "0", "--tag", "t", "--out", str(out)])
    summ = json.loads((out / "diag-v1-summary-t.json").read_text())
    assert summ["header"]["population"] == "b6d"
    kinds = {n: v["kind"] for n, v in summ["header"]["models"].items()}
    assert kinds == {"RAWF": "RAWF", "LRN": "LRN", "S0R1": "S0R1", "S1R0": "S1R0", "CX": "CONS-exact",
                     "CP": "CONS-pred"}
    assert sorted(summ["header"]["ivs_applied_to"]) == ["CP", "LRN", "S0R1"]
    hc = summ["historical_check"]
    assert all(hc[n]["status"] == "identical" and hc[n]["file"].endswith("eval_b6d.json")
               for n in ("RAWF", "LRN", "S0R1")), hc
    assert hc["S1R0"]["status"] == "missing"
    for name in kinds:
        with gzip.open(out / f"diag-v1-B-{name}.jsonl.gz", "rt") as f:
            assert json.loads(f.readline())["_meta"]["population"] == "b6d"
            recs = [json.loads(x) for x in f if '"decision"' in x]
        assert recs and {r["cfg_id"].split(":")[0] for r in recs} == {"b6d_hold_SCE"}
        with gzip.open(out / f"diag-v1-cf-SCE-{name}.jsonl.gz", "rt") as f:
            f.readline()
            crec = [json.loads(x) for x in f if x.strip()]
        assert {r["cf"]["seed"] for r in crec} == {DEV_CF_D, DEV_CF_D + 1}
    # the free-running b6d diag equals the trainer's b6d eval episodes (same worlds, same actions)
    with gzip.open(runs["S0R1"] / "eval_b6d_episodes.jsonl.gz", "rt") as f:
        f.readline()
        ev_rows = [json.loads(x) for x in f if x.strip()]
    with gzip.open(out / "diag-v1-B-S0R1.jsonl.gz", "rt") as f:
        f.readline()
        dg = [json.loads(x) for x in f if '"episode"' in x]
    assert len(dg) == len(ev_rows) == 3
    assert [r["gap_regret"] for r in dg] == [r["gap_regret"] for r in ev_rows]

    # population guards: never mix b6d with other inputs
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(labd / "cf_SCE.json", other / "cf_SCE.json")
    for argv in (["--labels", str(labd), "--pool", "b6d_hold_SCE", "--cf", str(other / "cf_SCE.json")],
                 ["--labels", str(lab), "--pool", "b6_B0", "--cf", str(labd / "cf_SCE.json")],
                 ["--cf", str(labd / "cf_SCE.json"), "--cf", str(other / "cf_SCE.json")],
                 ["--labels", str(lab), "--pool", "b6d_hold_SCE"]):
        with pytest.raises((SystemExit, AssertionError)):
            D.main(["run", *argv, "--model", f"RAWF={runs['RAWF']}", "--tag", "x", "--out", str(tmp_path / "dx")])
    assert not (tmp_path / "dx").exists() or not list((tmp_path / "dx").iterdir())
    out2 = tmp_path / "diag-other"
    D.main(["run", "--cf", str(other / "cf_SCE.json"), "--model", f"RAWF={runs['RAWF']}", "--n-octets", "1",
            "--no-latent", "--tag", "o", "--out", str(out2)])
    assert "population" not in json.loads((out2 / "diag-v1-summary-o.json").read_text())["header"]

    # scorer: b6d files score (population recorded); b6d + non-b6d files are refused
    files = sorted(str(p) for p in out.glob("diag-v1-*.jsonl.gz") if not p.name.startswith("diag-v1-hist-"))
    S.main(["--files", *files, "--out", str(tmp_path / "s.json"), "--n-boot", "20", "--contrast", "S0R1", "S1R0"])
    res = json.loads((tmp_path / "s.json").read_text())
    assert res["population"] == "b6d" and "B|CONS-pred" in res["cells"] and "B|S0R1-S1R0" in res["contrasts"]
    S.main(["--dirs", str(out2), "--out", str(tmp_path / "s2.json"), "--n-boot", "5"])
    assert "population" not in json.loads((tmp_path / "s2.json").read_text())
    with pytest.raises(SystemExit):
        S.main(["--files", files[0], str(next(out2.glob("diag-v1-cf-*.jsonl.gz"))), "--out", str(tmp_path / "s3.json"),
                "--n-boot", "5"])
    assert not (tmp_path / "s3.json").exists()
