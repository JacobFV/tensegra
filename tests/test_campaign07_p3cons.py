"""Extended-07 P3 consumer comparison (brief 15 B / 16): ordinary stronger consumer (mlp) vs explicit interaction
consumer (bil: factor-group-pair bilinear terms + factor gates on the trunk) vs gates only (gate), all against the
plain SUP-architecture consumer (CONS, 1-layer fusion).

  * the plain CONS path (trainer + diag) is bit-identical to the unmodified code: goldens captured on the pro6000 with
    the trainer of base 98122257 (``python tests/test_campaign07_p3cons.py --golden`` in a checkout of that base);
  * every variant starts from the plain CONS tensors of its seed, copied explicitly (hash recorded and equal), and
    computes exactly the plain CONS function at initialization (zero-initialized added output layers);
  * the raw residual path W_f [z; phi] is present and live in every variant;
  * variants train on bank mode under the exact / mix contracts; the first bank batch and its loss are identical across
    variants (common data stream, common function at init);
  * the diag runner loads them (::exact and ::pred=) under --phis own / exact / zero / mean, its forward equals
    model.step, and a free-running episode equals the trainer's greedy rollout;
  * the P3 job matrix (screen 40-44 and --confirm-seeds 50-54) and the parameter / work table.
CPU only, 1 thread; dev configurations 6.890-6.891e9 (P2 dev block), test seed 97.
"""
import hashlib
import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_campaign07_p2 as B  # noqa: E402  (pool, bank, predictors, helpers)

M, P, D = B.M, B.P, B.D
pw, pw6 = B.pw, B.pw6
torch.set_num_threads(1)
ARCHS = ("mlp", "bil", "gate")


@pytest.fixture(autouse=True)
def _pool_patch(monkeypatch):
    monkeypatch.setattr(M, "load_pool", lambda labels, split: B.POOL)


def _oof(tmp, bank):
    out = pathlib.Path(tmp) / "oof.pt"
    if not out.exists():
        runs = B.train_predictors(tmp, bank)
        P.main(["oof", "--bank", str(bank), "--fold-run", f"0={runs[0]}", "--fold-run", f"1={runs[1]}",
                "--full-run", str(runs["full"]), "--out", str(out)])
    return out


def cons_flags(bank, c, oof=None, arch=None, n=3):
    return (["--inputs", "factors6", "--arch", "fuse", "--bank", str(bank), "--bank-updates", str(n),
             "--phi-contract", c] + ([] if c == "exact" else ["--oof", str(oof)])
            + ([] if arch in (None, "lin") else ["--cons-arch", arch]))


def cons_digests(tmp):
    """Plain CONS (exact, mix) after 3 bank updates: model + log digests (the golden recipe)."""
    bank = B.make_bank(tmp)
    oof = _oof(tmp, bank)
    out = {}
    for c in ("exact", "mix"):
        d = B.train(M, tmp, f"gold-cons-{c}", cons_flags(bank, c, oof), updates=0)
        out[c] = {"model": B.sd_digest(torch.load(d / "model.pt")), "log": B.log_digest(d)}
    return out


# captured on the pro6000 with the UNMODIFIED trainer of base 98122257 (receipt CONS_GOLDEN_RECEIPT)
CONS_GOLDEN = {
    "exact": {"model": "84230509c415f27b", "log": "7f604473c725bdbf"},
    "mix": {"model": "a415c946d82b0672", "log": "901ed0634df29fb5"},
}
CONS_GOLDEN_RECEIPT = "results/dev/e07-dev-p3-golden-20260927T200235-1210326-process (2.9 core-s)"


# ------------------------------------------------------------------ plain CONS unchanged

def test_plain_cons_bit_identical_to_unmodified_trainer(tmp_path):
    assert cons_digests(tmp_path) == CONS_GOLDEN


def test_plain_cons_has_no_variant_and_diag_forward_unchanged(tmp_path):
    bank = B.make_bank(tmp_path)
    d = B.train(M, tmp_path, "c", cons_flags(bank, "exact"), updates=0)
    meta = json.loads((d / "train_meta.json").read_text())
    assert "cons" not in meta["fuse"] and "cons_arch" not in meta["p2"]["consumer"]
    m, _ = D.load_model(f"{d}::exact")
    assert getattr(m, "xfuse", None) is None
    x, h = _inputs()
    with torch.no_grad():
        hn, z, pred, phi, zf = D.forward(m, x, h)
        hs, zs = m.step(x, h)
        ref = torch.tanh(m.fuse(torch.cat([z, x[:, m.base_dim:]], 1)))  # the historical op
    assert torch.equal(hn, hs) and torch.equal(zf, zs) and torch.equal(zf, ref)


# ------------------------------------------------------------------ variants: init, residual path, training

def _inputs(B_=4):
    rows = []
    for _, cfg, _ in B.POOL[:B_]:
        st = pw.initial_state(cfg)
        av = pw6.available(cfg, st)
        rows.append(M.encode(cfg.public_vector(), None, av, 0.0) + M.supplied(cfg, st, "factors6"))
    g = torch.Generator().manual_seed(5)
    return torch.tensor(rows), torch.randn(len(rows), M.HIDDEN, generator=g)


def _variant_runs(tmp, c="exact", n=0, archs=("lin",) + ARCHS):
    bank = B.make_bank(tmp)
    oof = _oof(tmp, bank) if c != "exact" else None
    return {a: B.train(M, tmp, f"v-{a}-{c}-{n}", cons_flags(bank, c, oof, a, n), updates=0) for a in archs}


def test_variants_share_the_plain_consumer_init(tmp_path):
    runs = _variant_runs(tmp_path)
    lin = torch.load(runs["lin"] / "model.pt")
    lin_meta = json.loads((runs["lin"] / "train_meta.json").read_text())
    x, h = _inputs()
    with torch.no_grad():
        ref = B.load_run(runs["lin"])[0].step(x, h)
    for a in ARCHS:
        sd = torch.load(runs[a] / "model.pt")
        meta = json.loads((runs[a] / "train_meta.json").read_text())
        common = {k: v for k, v in sd.items() if not k.startswith("xfuse.")}
        assert set(common) == set(lin) and all(torch.equal(common[k], lin[k]) for k in lin), a
        cons = meta["p2"]["consumer"]
        assert cons["cons_arch"] == a and cons["cons_common_sha256"] == lin_meta["p2"]["init_sha256"]
        assert cons["cons_common_sha256"] == M.state_sha(lin)
        assert meta["fuse"]["cons"]["groups"] == M.contract_groups()
        m, _ = B.load_run(runs[a])
        with torch.no_grad():
            out = m.step(x, h)
        # zero-initialized added output layers: the variant computes the plain CONS function at initialization
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]), a
        pc = cons["cons_params"]
        assert pc["active"] - pc["variant"] == lin_meta["params"] - sum(
            v.numel() for k, v in lin.items() if k.split(".")[0] in ("v", "q", "stage", "dep", "switch", "case"))
        assert pc["total"] == meta["params"] == lin_meta["params"] + pc["variant"]
    # the variants' added tensors are this seed's construction draw after the CONS tensors; the explicit copy leaves
    # every common tensor at the CONS value even if construction order changed
    with pytest.raises(SystemExit):
        B.train(M, tmp_path, "bad", cons_flags(B.make_bank(tmp_path), "exact", arch="bil") + ["--init-from", "x"],
                updates=0)
    with pytest.raises(SystemExit):  # a fusion variant is a consumer
        B.train(M, tmp_path, "bad2", ["--arch", "fuse", "--cons-arch", "mlp"], updates=0)


def test_mlp_width_matched_to_bil():
    g = M.contract_groups()
    counts = {}
    for a in ARCHS:
        torch.manual_seed(0)
        m = M.ProbeNet(M.HIDDEN, inputs="factors6", arch="fuse", public_extra=pw6.PUBLIC_EXTRA6,
                       cons=M.cons_spec(a))
        counts[a] = M.cons_param_counts(m)["variant"]
    w = M.matched_cons_mlp_width()
    step = M.HIDDEN + pw6.N_FACTOR_FEATURES + 1 + M.HIDDEN  # added parameters per MLP unit
    assert abs(counts["mlp"] - counts["bil"]) <= step / 2
    nf, H, r = pw6.N_FACTOR_FEATURES, M.HIDDEN, 8
    sizes = [len(v) for v in g.values()]
    pairs = [(i, j) for i in range(4) for j in range(i, 4)]
    assert counts["bil"] == (nf * H + H) + sum((sizes[i] + 1) * r + (sizes[j] + 1) * r for i, j in pairs) \
        + (len(pairs) * r * H + H)
    assert counts["gate"] == nf * H + H and counts["mlp"] == (H + nf) * w + w + w * H + H
    tab = P.cons_cost_table()
    assert tab["rows"]["mlp"]["variant"] == counts["mlp"] and tab["rows"]["lin"]["variant"] == 0
    assert tab["rows"]["bil"]["decision_flops"] > tab["rows"]["lin"]["decision_flops"]
    assert len({r["train_episodes"] for r in tab["rows"].values()}) == 1  # equal training work (updates x batch)


def test_raw_residual_path_present_and_live():
    x, h = _inputs()
    for a in ARCHS:
        torch.manual_seed(3)
        m = M.ProbeNet(M.HIDDEN, inputs="factors6", arch="fuse", public_extra=pw6.PUBLIC_EXTRA6, cons=M.cons_spec(a))
        with torch.no_grad():  # make every added term non-trivial
            for p in m.xfuse.parameters():
                p.add_(0.05 * torch.randn_like(p))
        hn = m.gru(torch.tanh(m.inp(x[:, :m.base_dim])), h)
        z = m.trunk(hn).detach().requires_grad_(True)
        phi = x[:, m.base_dim:]
        out = m.xfuse(m.fuse, z, phi)
        assert torch.equal(out, m.fuse_step(x, z))
        out.sum().backward()
        assert m.fuse.weight.grad is not None and m.fuse.weight.grad[:, :M.HIDDEN].abs().sum() > 0  # z columns
        assert m.fuse.weight.grad[:, M.HIDDEN:].abs().sum() > 0  # phi columns
        with torch.no_grad():  # the residual path alone: the added terms switched off
            for n_, p in m.xfuse.named_parameters():
                if n_.split(".")[0] in ("l2", "gate", "out"):
                    p.zero_()
            assert torch.equal(m.xfuse(m.fuse, z, phi), torch.tanh(m.fuse(torch.cat([z, phi], 1))))
            m.fuse.weight.mul_(0.5)
            assert not torch.equal(m.xfuse(m.fuse, z, phi), out)


@pytest.mark.parametrize("c", ["exact", "mix"])
def test_variants_train_on_bank(tmp_path, c):
    runs = _variant_runs(tmp_path, c, n=3)
    inits = _variant_runs(tmp_path, c, n=0, archs=ARCHS)
    logs = {a: json.loads((d / "train_log.json").read_text()) for a, d in runs.items()}
    first = {a: lg[0]["imit"] for a, lg in logs.items()}
    assert len(set(first.values())) == 1, first  # same first batch, same function at init
    for a in ARCHS:
        assert len(logs[a]) == 3 and all(r["stage"] == "bank" for r in logs[a])
        sd, sd0 = torch.load(runs[a] / "model.pt"), torch.load(inits[a] / "model.pt")
        for k in sd:
            head = k.split(".")[0]
            changed = not torch.equal(sd[k], sd0[k])
            if head in ("v", "q", "stage", "dep", "switch", "case"):
                assert not changed, k  # heads unused by the bank imitation loss
            else:
                assert changed, k  # the whole decision path trains, including every added tensor
        meta = json.loads((runs[a] / "train_meta.json").read_text())
        assert meta["p2"]["consumer"]["contract"] == c and meta["p2"]["bank"]["updates"] == 3
        assert meta["p2"]["bank"]["decisions_seen"] == json.loads(
            (runs["lin"] / "train_meta.json").read_text())["p2"]["bank"]["decisions_seen"]


# ------------------------------------------------------------------ diag runner

def test_diag_loads_variants_with_phi_conditions(tmp_path, tmp_path_factory, monkeypatch):
    import gzip

    import test_campaign07_diag as TD
    monkeypatch.setattr(M, "split_world_seed", lambda split, idx, rep: B.DEVP2 + 950_000 + 1000 * idx + rep)
    bank = B.make_bank(tmp_path)
    oof = _oof(tmp_path, bank)
    runs = _variant_runs(tmp_path, "mix", n=2)
    pred = tmp_path / "pred-full"
    lab = B._labels(tmp_path)
    cf = TD._cf_file(tmp_path)
    models = []
    for a, d in runs.items():
        models += ["--model", f"{a.upper()}X={d}::exact", "--model", f"{a.upper()}P={d}::pred={pred}"]
    out = tmp_path_factory.mktemp("p3diag") / "diag"  # not inside the bank's directory (read-only input)
    D.main(["run", "--labels", str(lab), "--pool", "b6_dev", "--cf", str(cf), *models, "--protocols", "B", "A-pistar",
            "--ivs", "none", "exact", "--phis", "own", "exact", "zero", "mean", "--phi-mean-bank", str(bank),
            "--tag", "t", "--out", str(out)])
    summ = json.loads((out / "diag-v1-summary-t.json").read_text())
    hm = summ["header"]["models"]
    for a in ("lin",) + ARCHS:
        assert hm[f"{a.upper()}X"]["kind"] == "CONS-exact" and hm[f"{a.upper()}P"]["kind"] == "CONS-pred"
        assert hm[f"{a.upper()}X"].get("cons_arch") == (None if a == "lin" else a)
    names = {f"{a.upper()}{k}" for a in ("lin",) + ARCHS for k in "XP"}
    assert sorted(summ["header"]["phi_conditions"]["runs"]) == sorted(
        D.cond_name(n, c) for n in names for c in ("exact", "zero", "mean"))
    for n in names:
        for c in ("exact", "zero", "mean"):
            for proto in ("B", "cf-SCE"):
                with gzip.open(out / f"diag-v1-{proto}-{n}@phi={c}.jsonl.gz", "rt") as f:
                    f.readline()
                    recs = [json.loads(line) for line in f]
                assert recs and all(r["phi"] == c for r in recs if r["kind"] == "decision")
    # the variant's exact-input condition reproduces its own ::exact decisions
    def key(path):
        with gzip.open(path, "rt") as f:
            f.readline()
            return [(r["cfg_id"], r["t"], r["a"], r["probs"]) for r in map(json.loads, f) if r["kind"] == "decision"]
    assert key(out / "diag-v1-B-BILX.jsonl.gz") == key(out / "diag-v1-B-BILX@phi=exact.jsonl.gz")
    # forward == model.step; free-running B == the trainer's greedy rollout
    for a in ARCHS:
        m, _ = D.load_model(f"{runs[a]}::exact")
        x, h = _inputs()
        with torch.no_grad():
            hn, z, _, _, zf = D.forward(m, x, h)
            hs, zs = m.step(x, h)
        assert torch.equal(hn, hs) and torch.equal(zf, zs)
        items = [(cfg, s, B.DEVP2 + 950_000 + 1000 * idx) for idx, cfg, s in B.POOL[:6]]
        with torch.no_grad():
            eps, _, _ = M.run_batch(m, items, "greedy", need_labels=False)
        tr = [D.Track(cfg, s, "x", ws) for cfg, s, ws in items]
        D.drive(m, tr, {})
        assert [t.ep.history for t in tr] == [e.history for e in eps]
        # ::pred= : the variant reading the separate predictor's output at its input tail
        mp, _ = D.load_model(f"{runs[a]}::pred={pred}")
        c, _ = B.load_run(runs[a])
        p, _ = B.load_run(pred)
        h2 = torch.randn(x.shape[0], 2 * M.HIDDEN)
        with torch.no_grad():
            hn, z, pr, _, zf = D.forward(mp, x, h2)
            hp, _ = p.step(x[:, :p.base_dim], h2[:, M.HIDDEN:])
            hc, zc = c.step(torch.cat([x[:, :c.base_dim], p.last_aux], 1), h2[:, :M.HIDDEN])
        assert torch.equal(pr, p.last_aux) and torch.equal(zf, zc) and torch.equal(hn, torch.cat([hc, hp], 1))
    assert oof.exists()


# ------------------------------------------------------------------ job matrix

def _cmd(argv):
    return argv[argv.index("--") + 1:]


def test_p3_job_matrix_screen():
    jobs = P.p3_jobs("abcdef12")
    st = {}
    for s, n, argv in jobs:
        st.setdefault(s, []).append((n, argv))
    assert set(st) == {"P3-cons", "P3-eval", "P3-score"}
    assert len(st["P3-cons"]) == 3 * 2 * len(P.P2_SEEDS) and len(st["P3-eval"]) == 2 * len(P.P2_SEEDS)
    names = [n for _, n, _ in jobs]
    assert len(names) == len(set(names)) and all(n.startswith("e07-p3-") for n in names)
    for n, argv in st["P3-cons"]:
        cmd = _cmd(argv)
        arch, c, s = n.split("-")[3], n.split("-")[4], int(n.split("-s")[-1])
        assert cmd[cmd.index("--cons-arch") + 1] == arch and cmd[cmd.index("--phi-contract") + 1] == c
        assert cmd[cmd.index("--bank") + 1] == f"{P.R7}/e07-p2-bank/bank.pkl" and "--init-from" not in cmd
        assert cmd[cmd.index("--bank-updates") + 1] == "4000" and cmd[cmd.index("--updates") + 1] == "0"
        if c == "mix":
            assert cmd[cmd.index("--oof") + 1] == f"{P.R7}/e07-p2-oof-s{s}/oof.pt"
    for n, argv in st["P3-eval"]:
        cmd = _cmd(argv)
        c, s = n.split("-")[3], int(n.split("-s")[-1])
        i = cmd.index("--phis")
        assert cmd[i + 1:i + 5] == ["own", "exact", "zero", "mean"]
        assert f"{P.TBC}/cf_SCE.json" in cmd and P.UCE in cmd and cmd[cmd.index("--pool") + 1] == "b6c_hold_SCE"
        ms = [cmd[j + 1] for j, x in enumerate(cmd) if x == "--model"]
        assert len(ms) == 8
        assert f"CONS3-lin-{c}-exact-s{s}={P.R7}/e07-p2-cons-{c}-s{s}/run::exact" in ms
        assert f"CONS3-bil-{c}-pred-s{s}={P.R7}/e07-p3-cons-bil-{c}-s{s}/run::pred={P.R7}/e07-p2-pred-ffull-s{s}/run" in ms
    cmd = _cmd(st["P3-score"][0][1])
    assert cmd[cmd.index("--boot-seed") + 1] == "7" and cmd[cmd.index("--n-boot") + 1] == "20000"
    assert ["--contrast", "CONS3-bil-mix-pred@own", "CONS3-mlp-mix-pred@own"] == cmd[
        cmd.index("CONS3-bil-mix-pred@own") - 1:cmd.index("CONS3-bil-mix-pred@own") + 2]
    assert sum(x == "--contrast" for x in cmd) == 2 * 2 * len(P.P3_PAIRS) * 4
    # the P3 stages are reachable from the jobs command; the P2 matrix is unchanged
    assert [j[0] for j in P.job_matrix("abcdef12")].count("P3-2x2-bank") == 4 * len(P.P2_SEEDS)


def test_p3_job_matrix_confirm():
    jobs = P.p3_jobs("abcdef12", confirm=True)
    st = {}
    for s, n, argv in jobs:
        st.setdefault(s, []).append((n, argv))
    S = P.CONFIRM_SEEDS
    assert len(st["P3-pred"]) == 6 * len(S) and len(st["P3-oof"]) == len(S)
    assert len(st["P3-cons"]) == 4 * 2 * len(S) and len(st["P3-eval"]) == 2 * len(S)
    stages = [s for s, _, _ in jobs]
    assert stages.index("P3-oof") < stages.index("P3-cons") < stages.index("P3-eval") < stages.index("P3-score")
    lin = [a for n, a in st["P3-cons"] if "-lin-" in n]
    assert len(lin) == 2 * len(S) and all("--cons-arch" not in _cmd(a) for a in lin)
    for n, argv in st["P3-eval"]:
        cmd = _cmd(argv)
        assert n.startswith("e07-p3c-eval-") and cmd[cmd.index("--pool") + 1] == "b6d_hold_SCE"
        assert P.UCE not in cmd and [cmd[j + 1] for j, x in enumerate(cmd) if x == "--cf"] == [f"{P.B6D}/cf_SCE.json"]
        s = int(n.split("-s")[-1])
        assert any(x.endswith(f"::pred={P.R7}/e07-p3-pred-ffull-s{s}/run") for x in cmd)
    cmd = _cmd(st["P3-score"][0][1])
    assert cmd[cmd.index("--boot-seed") + 1] == "20260928"


if __name__ == "__main__" and "--golden" in sys.argv:
    import tempfile
    M.load_pool = lambda labels, split: B.POOL
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(cons_digests(pathlib.Path(d)), indent=1))
