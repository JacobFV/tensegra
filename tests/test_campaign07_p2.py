"""Extended-07 Phase 2 infrastructure: genuine S x R 2x2 arms, common valid-history bank, controlled consumers.

Mechanical tests (gradient-flow.md 6.5 and the Phase-2 commission):
  (a) S0 isolation: S0R0 trained with aux weight 0 / 1 / 1000 -> bitwise-identical actor parameters, actor Adam state
      and trajectories after N updates (split clip / split optimizer); S0R1: with the aux parameters held fixed, one
      actor step does not depend on the aux weight;
  (b) R0: the actor loss puts exactly zero gradient on the predictor, and policy outputs are invariant to it;
  (c) all four arms start from bitwise-identical tensors (constructed once per seed; --init-from copies them);
  (d) S1 vs S0 differ only through the trunk gradient path of the auxiliary loss;
  (e) historical arms (B0 / RAWF / LRN / SUP) train bit-identically to the UNMODIFIED trainer (goldens captured on the
      pro6000 with base c3974f5c before any change), and S1R1 with --clip-mode global --action-rng stream reproduces
      historical LRN bit for bit;
  plus the counter action RNG, the history bank (labels exact, replay == on-policy forward), bank training, K-fold
  out-of-fold predictions, consumer contracts and the diag runner on the new checkpoints.

Configurations: dev_smoke block 6.890-6.891e9 (P2 dev; never protocol).  Training worlds: test seed 97
(TRAIN_WORLD_BASE + 1e8 * 97 = 1.77e10 band; dev only).  CPU only, 1 thread.
``python tests/test_campaign07_p2.py --golden`` prints the historical-arm digests.
"""
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign06_probeworld as pw6  # noqa: E402

DEVP2 = 6_890_000_000
assert pw6.SUBRANGES["dev_smoke"][0] <= DEVP2 and DEVP2 + 1_000_000 <= pw6.SUBRANGES["dev_smoke"][1]
HSEED = 97
HIST_ARMS = {
    "B0": [],
    "RAWF": ["--arch", "fuse"],
    "LRN": ["--arch", "fuse", "--factor-mode", "learned", "--aux-weight", "1"],
    "SUP": ["--inputs", "factors6", "--arch", "fuse"],
}
torch.set_num_threads(1)


def _load(path=ROOT / "research/tools/campaign04_probeworld_train.py", name="pwtrain_e07p2"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sys.path.insert(0, str(ROOT / "research" / "tools"))
import campaign07_p2 as P  # noqa: E402
import campaign07_diag as D  # noqa: E402

M = P.T  # the trainer module shared by campaign07_p2 / campaign07_diag (one CounterRNG class)


def _pool(n=12):
    out = []
    for i in range(n):
        cfg = pw6.stream_config(DEVP2, i, pw.TRAIN_CELLS, (1, 2), pw6.TRAIN_TYPES)[0]
        s = pw6.ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        out.append((i, cfg, s))
    return out


POOL = _pool()


def sd_digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def log_digest(d):
    rows = json.loads((pathlib.Path(d) / "train_log.json").read_text())
    rows = [{k: v for k, v in r.items() if not k.startswith("cpu_s")} for r in rows]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def train(mod, tmp, name, flags, updates=3, batch=8, seed=HSEED, pool=POOL):
    orig = mod.load_pool
    mod.load_pool = lambda labels, split: pool
    try:
        d = pathlib.Path(tmp) / name
        mod.main(["train", "--labels", "unused", "--out", str(d), "--rung", "L1", "--seed", str(seed),
                  "--updates", str(updates), "--batch", str(batch), "--log-every", "1", *flags])
    finally:
        mod.load_pool = orig
    return d


def hist_digests(mod, tmp):
    out = {}
    for arm, flags in HIST_ARMS.items():
        d = train(mod, tmp, f"hist-{arm}", flags)
        out[arm] = {"model": sd_digest(torch.load(d / "model.pt")), "log": log_digest(d)}
    return out


# captured on the pro6000 with the UNMODIFIED trainer of base c3974f5c (see GOLDEN_RECEIPT)
HIST_GOLDEN = {
    "B0": {"model": "689fc198159c9fcb", "log": "b53aedfe0bda3211"},
    "RAWF": {"model": "7d96483ef95ee74c", "log": "2e1ed5d4b4eda90e"},
    "LRN": {"model": "6021e9eca1594592", "log": "24a124aa8ea1779e"},
    "SUP": {"model": "b2e4e5be2771a79f", "log": "70e95a6a798d20fc"},
}
GOLDEN_RECEIPT = "results/dev/e07-dev-p2-golden-20260927T171056-1146990-process (3.0 core-s)"


@pytest.fixture(autouse=True)
def _pool_patch(monkeypatch):
    monkeypatch.setattr(M, "load_pool", lambda labels, split: POOL)


# ------------------------------------------------------------------ helpers

SR = {"S0R0": (0, 0), "S1R0": (1, 0), "S0R1": (0, 1), "S1R1": (1, 1)}


def sr_flags(arm, *extra):
    sh, rd = SR[arm]
    return ["--arch", "fuse", "--shape", str(sh), "--read", str(rd), *extra]


def load_run(d):
    meta = json.loads((pathlib.Path(d) / "train_meta.json").read_text())
    m = M.build_model_from_meta(meta)
    m.load_state_dict(torch.load(pathlib.Path(d) / "model.pt"))
    return m, meta


def actor_sd(sd):
    return {k: v for k, v in sd.items() if not k.startswith("aux.")}


def sr_model(arm, seed=HSEED):
    torch.manual_seed(1000 + seed)
    sh, rd = SR[arm]
    return M.ProbeNet(M.HIDDEN, arch="fuse", public_extra=pw6.PUBLIC_EXTRA6, factor_mode="sr", shape=sh, read=rd)


def sample_batch(model, seed=0, B=8):
    items = [(cfg, s, DEVP2 + 500_000 + 100 * seed + j) for j, (_, cfg, s) in enumerate(POOL[:B])]
    eps, steps, ep_steps = M.run_batch(model, items, "sample", M.CounterRNG(seed))
    M.attach_case_labels(items, ep_steps, steps)
    return items, eps, steps, ep_steps


def actor_loss(model, batch):
    items, eps, steps, ep_steps = batch
    total, parts = M.losses(model, items, eps, steps, ep_steps, dict(M.RUNG_WEIGHTS["L1"], aux=0.0))
    return total, parts


# ------------------------------------------------------------------ (e) historical arms unchanged

def test_historical_arms_bit_identical_to_unmodified_trainer(tmp_path):
    assert hist_digests(M, tmp_path) == HIST_GOLDEN


def test_historical_vs_reference_copy(tmp_path):
    """Direct comparison with a copy of the unmodified trainer when it is available (pro6000 dev sync: _ref/)."""
    ref = ROOT / "_ref" / "ref_trainer_c3974f5c.py"
    if not ref.exists():
        pytest.skip("reference copy absent (goldens cover this)")
    R = _load(ref, "pwtrain_ref")
    got = hist_digests(M, tmp_path / "new")
    old = {}
    for arm, flags in HIST_ARMS.items():
        d = train(R, tmp_path / "old", f"hist-{arm}", flags)
        old[arm] = {"model": sd_digest(torch.load(d / "model.pt")), "log": log_digest(d)}
    assert got == old


def test_s1r1_historical_coupling_reproduces_lrn(tmp_path):
    """S1R1 with --clip-mode global --action-rng stream is historical LRN bit for bit (same init: aux before fuse;
    same forward; one Adam + one global clip; the shared action stream)."""
    d = train(M, tmp_path, "s1r1-hist", sr_flags("S1R1", "--clip-mode", "global", "--action-rng", "stream"))
    assert sd_digest(torch.load(d / "model.pt")) == HIST_GOLDEN["LRN"]["model"]
    lrn = train(M, tmp_path, "lrn", HIST_ARMS["LRN"])
    keys = ("update", "U", "regret", "rl", "imit", "aux", "value_mse", "entropy")
    a = json.loads((d / "train_log.json").read_text())
    b = json.loads((lrn / "train_log.json").read_text())
    assert [{k: r[k] for k in keys} for r in a] == [{k: r[k] for k in keys} for r in b]


def test_historical_eval_loader_unchanged(tmp_path):
    """cmd_eval / diag construct historical runs with the identical constructor call (state_dict keys and values)."""
    for arm, flags in HIST_ARMS.items():
        d = train(M, tmp_path, f"h0-{arm}", flags, updates=0)
        m, meta = load_run(d)
        dm, _ = D.load_model(d)
        sd = torch.load(d / "model.pt")
        assert sd_digest(m.state_dict()) == sd_digest(sd) == sd_digest(dm.state_dict())
        assert D.model_kind(dm) == arm


# ------------------------------------------------------------------ (c) common initialization

def test_four_arms_start_bitwise_identical(tmp_path):
    sds, shas = {}, set()
    for arm in SR:
        d = train(M, tmp_path, f"init-{arm}", sr_flags(arm), updates=0)
        sds[arm] = torch.load(d / "model.pt")
        meta = json.loads((d / "train_meta.json").read_text())
        shas.add(meta["p2"]["init_sha256"])
        assert meta["fuse"]["arm"] == arm and meta["fuse"]["phi_const"] == [0.0] * pw6.N_FACTOR_FEATURES
    assert len(shas) == 1
    ref = sds["S1R1"]
    assert {k for k in ref if k.startswith(("aux.", "fuse."))} == {
        "aux.0.weight", "aux.0.bias", "aux.2.weight", "aux.2.bias", "fuse.weight", "fuse.bias"}
    for arm, sd in sds.items():
        assert sorted(sd) == sorted(ref) and all(torch.equal(sd[k], ref[k]) for k in ref), arm
    # the explicit common init (campaign07_p2 init) is the same tensor set; --init-from copies it (strict)
    P.main(["init", "--seed", str(HSEED), "--out", str(tmp_path / "init")])
    info = json.loads((tmp_path / "init" / "init.json").read_text())
    assert info["sha256"] == shas.pop()
    other = train(M, tmp_path, "from-init", sr_flags("S0R1", "--init-from", str(tmp_path / "init" / "init.pt"),
                                                    "--init-sha", info["sha256"]), updates=0, seed=HSEED + 1)
    sd = torch.load(other / "model.pt")
    assert all(torch.equal(sd[k], ref[k]) for k in ref)  # another seed's construction, overwritten by the copy
    meta = json.loads((other / "train_meta.json").read_text())
    assert meta["p2"]["init_sha256"] == info["sha256"] != meta["p2"]["init_sha256_constructed"]
    with pytest.raises(SystemExit):
        train(M, tmp_path, "bad-sha", sr_flags("S0R1", "--init-from", str(tmp_path / "init" / "init.pt"),
                                               "--init-sha", "0" * 64), updates=0)
    with pytest.raises(SystemExit):  # an LRN-shaped state dict cannot be copied into a RAWF-shaped model
        train(M, tmp_path, "bad-keys", ["--arch", "fuse", "--init-from", str(tmp_path / "init" / "init.pt")],
              updates=0)


# ------------------------------------------------------------------ (a) S0 isolation

def test_s0r0_aux_weight_cannot_affect_actor(tmp_path):
    """S0R0 trained with aux weight 0 / 1 / 1000 for N updates: bitwise-identical actor parameters, identical
    trajectories (U, regret, actor losses, actor gradient norm = clip input); only the predictor differs."""
    runs = {w: train(M, tmp_path, f"s0r0-w{w}", sr_flags("S0R0", "--aux-weight", str(w)), updates=4) for w in (0, 1, 1000)}
    sds = {w: torch.load(d / "model.pt") for w, d in runs.items()}
    logs = {w: json.loads((d / "train_log.json").read_text()) for w, d in runs.items()}
    for w in (1, 1000):
        a0, aw = actor_sd(sds[0]), actor_sd(sds[w])
        assert all(torch.equal(a0[k], aw[k]) for k in a0), w
        keys = ("U", "regret", "rl", "imit", "value_mse", "entropy", "gnorm_A")
        assert [{k: r[k] for k in keys} for r in logs[0]] == [{k: r[k] for k in keys} for r in logs[w]]
    assert not torch.equal(sds[0]["aux.2.weight"], sds[1]["aux.2.weight"])  # the probe itself does train


def test_s0r1_one_actor_step_independent_of_aux_weight(tmp_path):
    """S0R1: the policy reads sg(pred) by design, so after the first step the predictor (and hence phi) differs with
    w; the actor STEP itself (clip, Adam) never sees the aux loss: after one update the actor parameters are
    bitwise identical for w = 0 / 1 / 1000."""
    sds = {w: actor_sd(torch.load(train(M, tmp_path, f"s0r1-w{w}", sr_flags("S0R1", "--aux-weight", str(w)),
                                         updates=1) / "model.pt")) for w in (0, 1, 1000)}
    assert all(torch.equal(sds[0][k], sds[w][k]) for w in (1, 1000) for k in sds[0])
    two = {w: actor_sd(torch.load(train(M, tmp_path, f"s0r1-2-w{w}", sr_flags("S0R1", "--aux-weight", str(w)),
                                        updates=2) / "model.pt")) for w in (0, 1)}
    assert not all(torch.equal(two[0][k], two[1][k]) for k in two[0])  # R1 reads the (differently) trained probe


def test_split_clip_excludes_aux_gradients():
    """Split mode: A's clip norm is the norm of A's gradients only; global mode clips A and X together."""
    model = sr_model("S1R1")
    batch = sample_batch(model)
    total, _ = M.losses(model, *batch, dict(M.RUNG_WEIGHTS["L1"], aux=1.0))
    A = [p for n, p in model.named_parameters() if not n.startswith("aux.")]
    X = [p for n, p in model.named_parameters() if n.startswith("aux.")]
    groups = [(torch.optim.Adam(A, lr=1e-3), A, "A"), (torch.optim.Adam(X, lr=1e-3), X, "X")]
    for o, _, _ in groups:
        o.zero_grad()
    total.backward(retain_graph=True)
    nA = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in A)))
    nX = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in X)))
    model.zero_grad()
    gn = M.p2_step(groups, total)
    assert abs(gn["gnorm_A"] - nA) < 1e-5 * nA and abs(gn["gnorm_X"] - nX) < 1e-5 * max(nX, 1e-9)
    assert nX > 0


# ------------------------------------------------------------------ (b) R0: the policy never depends on the predictor

@pytest.mark.parametrize("arm", ["S0R0", "S1R0"])
def test_r0_actor_gradient_zero_and_policy_invariant(arm):
    model = sr_model(arm)
    batch = sample_batch(model)
    total, _ = actor_loss(model, batch)
    X = [p for n, p in model.named_parameters() if n.startswith("aux.")]
    g = torch.autograd.grad(total, X, allow_unused=True)
    assert all(x is None or torch.all(x == 0) for x in g)
    H = model.gru.hidden_size
    items = batch[0]
    with torch.no_grad():
        _, s1, _ = M.run_batch(model, items, "greedy", need_labels=False)
        for p in X:
            p.add_(torch.randn_like(p))
        _, s2, _ = M.run_batch(model, items, "greedy", need_labels=False)
    assert all(torch.equal(a["logp_all"], b["logp_all"]) and torch.equal(a["chosen"], b["chosen"]) for a, b in zip(s1, s2))
    assert not all(torch.equal(a["aux"], b["aux"]) for a, b in zip(s1, s2))
    # the fusion layer's phi columns receive no actor gradient in R0 (constant zero input)
    total, _ = actor_loss(model, sample_batch(model, seed=1))
    gf = torch.autograd.grad(total, [model.fuse.weight])[0]
    assert torch.all(gf[:, H:] == 0)


def test_r1_actor_gradient_zero_on_predictor_but_policy_reads_it():
    model = sr_model("S0R1")
    batch = sample_batch(model)
    total, _ = actor_loss(model, batch)
    X = [p for n, p in model.named_parameters() if n.startswith("aux.")]
    assert all(x is None or torch.all(x == 0) for x in torch.autograd.grad(total, X, allow_unused=True))  # sg(pred)
    H = model.gru.hidden_size
    assert torch.any(torch.autograd.grad(total, [model.fuse.weight])[0][:, H:] != 0)
    items = batch[0]
    with torch.no_grad():
        _, s1, _ = M.run_batch(model, items, "greedy", need_labels=False)
        for p in X:
            p.add_(torch.randn_like(p))
        _, s2, _ = M.run_batch(model, items, "greedy", need_labels=False)
    assert not all(torch.equal(a["logp_all"], b["logp_all"]) for a, b in zip(s1, s2))


def test_phi_const_train_mean_option(tmp_path):
    bank = make_bank(tmp_path)
    d = train(M, tmp_path, "r0-mean", sr_flags("S0R0", "--phi-const", "train_mean", "--bank", str(bank),
                                                "--bank-updates", "1"), updates=0)
    meta = json.loads((d / "train_meta.json").read_text())
    b = M.load_bank(bank)
    rows = torch.tensor([r for e in b["episodes"] for r in e["phi"]], dtype=torch.float64).mean(0).float()
    assert meta["fuse"]["phi_const"] == rows.tolist() and meta["fuse"]["phi_const_mode"] == "train_mean"
    m, _ = load_run(d)
    assert torch.equal(m.phi_const, rows)
    assert "phi_const" not in torch.load(d / "model.pt")  # never part of the (common) parameters


# ------------------------------------------------------------------ (d) S1 vs S0: only the trunk gradient path

def test_s1_vs_s0_differ_only_by_aux_trunk_gradient():
    m0, m1 = sr_model("S0R1"), sr_model("S1R1")
    assert sd_digest(m0.state_dict()) == sd_digest(m1.state_dict())
    b0, b1 = sample_batch(m0), sample_batch(m1)
    for r0, r1 in zip(b0[2], b1[2]):  # identical forward: logits, predictions, actions
        assert torch.equal(r0["logp_all"], r1["logp_all"]) and torch.equal(r0["aux"], r1["aux"])
        assert torch.equal(r0["chosen"], r1["chosen"])
    w = dict(M.RUNG_WEIGHTS["L1"], aux=1.0)
    names = [n for n, _ in m0.named_parameters()]
    t0, p0 = M.losses(m0, *b0, w)
    t1, p1 = M.losses(m1, *b1, w)
    assert float(t0) == float(t1)
    g0 = dict(zip(names, torch.autograd.grad(t0, list(m0.parameters()), allow_unused=True)))
    g1 = dict(zip(names, torch.autograd.grad(t1, list(m1.parameters()), retain_graph=True, allow_unused=True)))
    trunk = [n for n in names if n.split(".")[0] in ("inp", "gru", "trunk")]
    ga = dict(zip(trunk, torch.autograd.grad(p1["aux"], [dict(m1.named_parameters())[n] for n in trunk])))
    for n in names:
        a, b = g0[n], g1[n]
        if n in trunk:
            assert not torch.equal(a, b), n
            assert torch.allclose(b - a, ga[n], atol=1e-6, rtol=1e-4), n  # the difference IS the aux trunk gradient
        elif a is None or b is None:
            assert (a is None or torch.all(a == 0)) and (b is None or torch.all(b == 0)), n
        else:
            assert torch.equal(a, b), n  # pi, v, fuse, aux, unused heads: identical


# ------------------------------------------------------------------ RNG

def test_counter_rng_common_across_arms():
    """u(seed, episode world seed, step) is the same in every arm, whatever the episode lengths."""
    seen = {}

    class Rec(M.CounterRNG):
        def u(self, ws, t):
            x = super().u(ws, t)
            seen.setdefault(self.tag, {})[(ws, t)] = x
            return x
    for tag, seed in (("a", 1), ("b", 2)):
        model = sr_model("S1R1", seed=seed)
        r = Rec(HSEED)
        r.tag = tag
        items = [(cfg, s, DEVP2 + 700_000 + j) for j, (_, cfg, s) in enumerate(POOL)]
        M.run_batch(model, items, "sample", r, need_labels=False)
    common = set(seen["a"]) & set(seen["b"])
    assert common and all(seen["a"][k] == seen["b"][k] for k in common)
    # training consumes no torch RNG (only construction does)
    model = sr_model("S1R0")
    st = torch.get_rng_state()
    batch = sample_batch(model)
    total, _ = M.losses(model, *batch, dict(M.RUNG_WEIGHTS["L1"], aux=1.0))
    A = [p for n, p in model.named_parameters() if not n.startswith("aux.")]
    X = [p for n, p in model.named_parameters() if n.startswith("aux.")]
    M.p2_step([(torch.optim.Adam(A), A, "A"), (torch.optim.Adam(X), X, "X")], total)
    assert torch.equal(st, torch.get_rng_state())


def test_aux_group_weights():
    model = sr_model("S1R1")
    items, eps, steps, ep_steps = sample_batch(model)
    base = M.aux_mse(steps, {})
    cw = M.aux_coord_weights(["G1=1", "G2=1", "G3=1", "G4=1"])
    assert torch.equal(M.aux_mse(steps, {"aux_cw": torch.tensor(cw)}), base)
    groups = M.contract_groups()
    cw = M.aux_coord_weights(["G4=0"])
    assert all(cw[j] == 0.0 for j in groups["G4"]) and sum(cw) == 23 - len(groups["G4"])
    with pytest.raises(SystemExit):
        M.aux_coord_weights(["G9=1"])


# ------------------------------------------------------------------ history bank

def make_bank(tmp, name="bank.pkl", sources=("pistar:1", "eps:0.3:1"), model_run=None):
    p = pathlib.Path(tmp) / name
    if p.exists():
        return p
    src = list(sources) + ([f"model:M={model_run}:1"] if model_run else [])
    P.main(["bank", "--labels", "unused", "--pool", "b6_B0", "--world-base", str(DEVP2 + 900_000), "--source", *src,
            "--out", str(p)])
    return p


def test_bank_labels_are_exact_and_histories_valid(tmp_path):
    d = train(M, tmp_path, "src", HIST_ARMS["RAWF"], updates=1)
    bank = M.load_bank(make_bank(tmp_path, model_run=d))
    meta = json.loads(pathlib.Path(str(tmp_path / "bank.pkl") + ".json").read_text())
    assert meta["sha256"] == M._sha_path(tmp_path / "bank.pkl")
    by = {idx: (cfg, s) for idx, cfg, s in POOL}
    assert {e["src"] for e in bank["episodes"]} == {"pistar", "eps0.3", "M"}
    assert len(bank["episodes"]) == 3 * len(POOL)
    assert len({e["ws"] for e in bank["episodes"]}) == len(bank["episodes"])
    for e in bank["episodes"]:
        cfg, s = by[e["cfg_idx"]]
        env = M._env(cfg)
        st = pw.initial_state(cfg)
        for t, rec in enumerate(e["hist"]):
            assert sorted(env.available(cfg, st)) == e["avail"][t] and rec[0] in e["avail"][t]
            assert sorted(s.opt_set(st)) == e["opt"][t] and e["phi"][t] == pw6.factor_features(cfg, st)
            assert e["q"][t] == st[0][0] and e["V"][t] == s.value(st)
            st = env.advance(cfg, st, *rec)
        assert e["vec"] == list(cfg.public_vector())
        if e["src"] == "pistar":
            assert all(e["a_opt"])
        assert abs(e["G"][0] * 100.0 - e["U"]) < 1e-6


def test_bank_replay_equals_on_policy_forward(tmp_path):
    """Replaying stored histories rebuilds exactly the inputs / states / policy of the on-policy rollout."""
    model = sr_model("S1R1")
    items = [(cfg, s, DEVP2 + 800_000 + j) for j, (_, cfg, s) in enumerate(POOL)]
    with torch.no_grad():
        eps, steps, ep_steps = M.run_batch(model, items, "sample", M.CounterRNG(3), need_labels=False)
        bank = [P.label_episode(cfg, s, ep_steps[j], ws, j, "m", 0, eps[j]) for j, (cfg, s, ws) in enumerate(items)]
        rsteps = M.replay_bank_batch(model, bank)
    assert len(steps) == len(rsteps)
    for a, b in zip(steps, rsteps):
        assert torch.equal(a["idx"], b["idx"]) and torch.equal(a["logp_all"], b["logp_all"])
        assert torch.equal(a["z"], b["z"]) and torch.equal(a["aux"], b["aux"])
        assert torch.equal(a["aux_t"], b["aux_t"].float()) if "aux_t" in a else True


def test_bank_training_arms(tmp_path):
    bank = make_bank(tmp_path)
    runs = {}
    for arm in SR:
        runs[arm] = train(M, tmp_path, f"bank-{arm}", sr_flags(arm, "--bank", str(bank), "--bank-updates", "3"),
                          updates=0)
    logs = {arm: json.loads((d / "train_log.json").read_text()) for arm, d in runs.items()}
    assert all(r["stage"] == "bank" for lg in logs.values() for r in lg)
    # identical data and init: the first update's aux loss is identical in all four arms, the imitation loss is
    # identical within each R level (R0 reads the constant, R1 the prediction)
    assert len({lg[0]["aux"] for lg in logs.values()}) == 1
    assert logs["S0R0"][0]["imit"] == logs["S1R0"][0]["imit"] and logs["S0R1"][0]["imit"] == logs["S1R1"][0]["imit"]
    meta = json.loads((runs["S1R1"] / "train_meta.json").read_text())
    assert meta["p2"]["bank"]["sha256"] == M._sha_path(bank) and meta["p2"]["bank"]["updates"] == 3
    # S0 isolation holds in bank mode too
    a = {w: actor_sd(torch.load(train(M, tmp_path, f"bank-s0r0-{w}", sr_flags("S0R0", "--bank", str(bank),
                                      "--bank-updates", "3", "--aux-weight", str(w)), updates=0) / "model.pt"))
         for w in (0, 1000)}
    assert all(torch.equal(a[0][k], a[1000][k]) for k in a[0])
    # on-policy fine-tuning is separately flagged
    with pytest.raises(SystemExit):
        train(M, tmp_path, "bank-noflag", sr_flags("S1R1", "--bank", str(bank), "--bank-updates", "1"), updates=2)
    ft = train(M, tmp_path, "bank-ft", sr_flags("S1R1", "--bank", str(bank), "--bank-updates", "2", "--finetune"),
               updates=2)
    stages = [r["stage"] for r in json.loads((ft / "train_log.json").read_text())]
    assert stages == ["bank", "bank", "onpolicy", "onpolicy"]


# ------------------------------------------------------------------ predictors, out-of-fold, consumers

def test_folds_are_by_configuration_and_stratified(tmp_path):
    bank = M.load_bank(make_bank(tmp_path))
    f = M.bank_folds(bank, 2)
    assert set(f) == {idx for idx, _, _ in POOL} and set(f.values()) == {0, 1}
    for combo in {e["combo"] for e in bank["episodes"]}:
        cs = sorted({e["cfg_idx"] for e in bank["episodes"] if e["combo"] == combo})
        assert [f[c] for c in cs] == [j % 2 for j in range(len(cs))]


def train_predictors(tmp, bank, K=2, n=3):
    runs = {}
    for k in range(K):
        runs[k] = train(M, tmp, f"pred-f{k}", sr_flags("S1R0", "--bank", str(bank), "--bank-updates", str(n),
                                                       "--predictor-only", "--bank-folds", str(K),
                                                       "--bank-exclude-fold", str(k)), updates=0)
    runs["full"] = train(M, tmp, "pred-full", sr_flags("S1R0", "--bank", str(bank), "--bank-updates", str(n),
                                                       "--predictor-only"), updates=0)
    return runs


def test_predictor_only_trains_aux_and_trunk_only(tmp_path):
    bank = make_bank(tmp_path)
    runs = train_predictors(tmp_path, bank)
    init = torch.load(train(M, tmp_path, "pred-init", sr_flags("S1R0"), updates=0) / "model.pt")
    sd = torch.load(runs[0] / "model.pt")
    for k in sd:
        changed = not torch.equal(sd[k], init[k])
        assert changed == (k.split(".")[0] in ("aux", "inp", "gru", "trunk")), k
    meta = json.loads((runs[0] / "train_meta.json").read_text())
    b = M.load_bank(bank)
    folds = M.bank_folds(b, 2)
    assert meta["p2"]["bank"]["episodes_available"] == sum(folds[e["cfg_idx"]] != 0 for e in b["episodes"])


def test_oof_predictions(tmp_path):
    bank = make_bank(tmp_path)
    runs = train_predictors(tmp_path, bank)
    out = tmp_path / "oof.pt"
    P.main(["oof", "--bank", str(bank), "--fold-run", f"0={runs[0]}", "--fold-run", f"1={runs[1]}",
            "--full-run", str(runs["full"]), "--out", str(out)])
    res = torch.load(out)
    b = M.load_bank(bank)
    folds = M.bank_folds(b, 2)
    assert set(res["preds"]) == {e["id"] for e in b["episodes"]} and len(res["err_rms"]) == 23
    for k in (0, 1):
        m, meta = load_run(runs[k])
        eps = [e for e in b["episodes"] if folds[e["cfg_idx"]] == k]
        # each decision is predicted by the model that never saw its configuration
        assert set(res["fold_runs"][k]["configs"]) == {e["cfg_idx"] for e in eps}
        assert not set(res["fold_runs"][k]["configs"]) & {e["cfg_idx"] for e in b["episodes"] if folds[e["cfg_idx"]] != k}
        again = P.predict_bank(m, eps)
        assert all(again[e["id"]] == res["preds"][e["id"]] for e in eps)
    with pytest.raises(AssertionError):  # a fold run must have excluded exactly its fold
        P.main(["oof", "--bank", str(bank), "--fold-run", f"0={runs[1]}", "--fold-run", f"1={runs[0]}",
                "--out", str(tmp_path / "bad.pt")])


def test_consumer_contracts(tmp_path):
    build_consumers(tmp_path)


def build_consumers(tmp_path):
    bank = make_bank(tmp_path)
    runs = train_predictors(tmp_path, bank)
    oof = tmp_path / "oof.pt"
    P.main(["oof", "--bank", str(bank), "--fold-run", f"0={runs[0]}", "--fold-run", f"1={runs[1]}",
            "--out", str(oof)])
    sup = ["--inputs", "factors6", "--arch", "fuse", "--bank", str(bank), "--bank-updates", "2"]
    cons = {}
    for c in ("exact", "oof", "mix"):
        extra = [] if c == "exact" else ["--oof", str(oof)]
        cons[c] = train(M, tmp_path, f"cons-{c}", sup + ["--phi-contract", c] + extra, updates=0)
        meta = json.loads((cons[c] / "train_meta.json").read_text())
        assert meta["p2"]["consumer"]["contract"] == c
    # contract inputs
    b = M.load_bank(bank)
    res = torch.load(oof)
    p2 = {"phi_contract": "mix", "oof": res, "noise_seed": 0, "mix_p": 0.5}
    sel = b["episodes"][:6]
    m1, m2 = M.contract_phis(p2, sel, 1, 0), M.contract_phis(p2, sel, 1, 0)
    assert m1 == m2 and m1 != M.contract_phis(p2, sel, 1, 1)
    assert any(x == e["phi"] for x, e in zip(m1, sel)) or any(x != e["phi"] for x, e in zip(m1, sel))
    assert M.contract_phis(dict(p2, phi_contract="exact"), sel, 1, 0) == [e["phi"] for e in sel]
    assert M.contract_phis(dict(p2, phi_contract="oof"), sel, 1, 0) == [res["preds"][e["id"]] for e in sel]
    with pytest.raises(SystemExit):  # consumers are the SUP architecture
        train(M, tmp_path, "cons-bad", ["--arch", "fuse", "--bank", str(bank), "--bank-updates", "1",
                                         "--phi-contract", "exact"], updates=0)
    with pytest.raises(SystemExit):  # oof consumers are bank-only
        train(M, tmp_path, "cons-ft", sup + ["--phi-contract", "oof", "--oof", str(oof), "--finetune"], updates=1)
    return cons, runs


# ------------------------------------------------------------------ diag runner on the new checkpoints

def _labels(tmp, split="b6_dev"):
    import pickle
    d = pathlib.Path(tmp) / "labels"
    d.mkdir()
    with open(d / f"{split}.shard000.pkl", "wb") as f:
        pickle.dump([(i, c, s._V, s._Q) for i, c, s in POOL[:6]], f)
    (d / f"{split}.shards.json").write_text(json.dumps({"shards": [f"{split}.shard000"]}))
    return d


def test_predicted_consumer_forward(tmp_path):
    cons, runs = build_consumers(tmp_path)
    m, _ = D.load_model(f"{cons['oof']}::pred={runs['full']}")
    c, _ = load_run(cons["oof"])
    p, _ = load_run(runs["full"])
    assert D.model_kind(m) == "CONS-pred" and D.hidden_size(m) == 2 * M.HIDDEN
    cfg = POOL[0][1]
    st = pw.initial_state(cfg)
    av = pw6.available(cfg, st)
    x = torch.tensor([M.encode(cfg.public_vector(), None, av, 0.0) + M.supplied(cfg, st, "factors6")])
    h = torch.randn(1, 2 * M.HIDDEN)
    with torch.no_grad():
        hn, z, pred, phi, zf = D.forward(m, x, h)
        hp, zp = p.step(x[:, :p.base_dim], h[:, M.HIDDEN:])
        xc = torch.cat([x[:, :c.base_dim], p.last_aux], 1)  # the consumer reading the prediction at its input tail
        hc, zc = c.step(xc, h[:, :M.HIDDEN])
    assert torch.equal(pred, p.last_aux) and torch.equal(zf, zc) and torch.equal(hn, torch.cat([hc, hp], 1))
    with pytest.raises(SystemExit):
        D.load_model(str(cons["oof"]))  # a consumer needs an explicit input spec


def test_diag_runs_on_new_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "split_world_seed", lambda split, idx, rep: DEVP2 + 950_000 + 1000 * idx + rep)
    cons, runs = build_consumers(tmp_path)
    arms = {arm: train(M, tmp_path, f"d-{arm}", sr_flags(arm), updates=1) for arm in SR}
    lab = _labels(tmp_path)
    out = tmp_path / "diag"
    models = sum((["--model", f"{a}={d}"] for a, d in arms.items()), [])
    models += ["--model", f"CX={cons['exact']}::exact", "--model", f"CP={cons['oof']}::pred={runs['full']}"]
    D.main(["run", "--labels", str(lab), "--pool", "b6_dev", *models, "--protocols", "B", "A-pistar",
            "--ivs", "none", "exact", "--tag", "t", "--out", str(out)])
    summ = json.loads((out / "diag-v1-summary-t.json").read_text())
    kinds = {n: v["kind"] for n, v in summ["header"]["models"].items()}
    assert kinds == {"S0R0": "S0R0", "S1R0": "S1R0", "S0R1": "S0R1", "S1R1": "S1R1", "CX": "CONS-exact",
                     "CP": "CONS-pred"}
    assert sorted(summ["header"]["ivs_applied_to"]) == ["CP", "S0R1", "S1R1"]
    assert summ["header"]["models"]["CP"]["predictor"]["run"] == str(runs["full"])
    import gzip
    for name in kinds:
        with gzip.open(out / f"diag-v1-B-{name}.jsonl.gz", "rt") as f:
            f.readline()
            recs = [json.loads(line) for line in f if '"decision"' in line]
        assert recs and all(r["pred"] is not None for r in recs)
        assert all(("iv" in r) == (name in ("CP", "S0R1", "S1R1")) for r in recs)
        if name == "CX":
            assert all(abs(p - t) <= 1e-6 * max(1.0, abs(t)) + 1e-6 for r in recs for p, t in zip(r["pred"], r["target"]))
    # free-running B protocol equals the trainer's greedy rollout for an S x R arm
    m, _ = load_run(arms["S0R1"])
    items = [(cfg, s, DEVP2 + 950_000 + 1000 * idx) for idx, cfg, s in POOL[:6]]
    with torch.no_grad():
        eps, _, _ = M.run_batch(m, items, "greedy", need_labels=False)
    tr = [D.Track(c, s, "x", ws) for c, s, ws in items]
    D.drive(m, tr, {})
    assert [t.ep.history for t in tr] == [e.history for e in eps]
    # an intervention on an R0 arm is refused (untrained constant channel)
    with pytest.raises(ValueError):
        D.Intervener(["exact"], "S1R0")


def test_seed_ranges_disjoint():
    assert P.check_ranges(P.proposed_ranges()) == []


def test_job_matrix_validates():
    jobs = P.job_matrix("abcdef12", smoke=True)
    names = [n for _, n, _ in jobs]
    assert len(names) == len(set(names)) and all(n.startswith("e07-p2s-") for n in names)
    full = P.job_matrix("abcdef12")
    assert sum(1 for s, _, _ in full if s == "2x2-bank") == 4 * len(P.P2_SEEDS)
    assert sum(1 for s, _, _ in full if s == "consumer") == 3 * len(P.P2_SEEDS)


if __name__ == "__main__" and "--golden" in sys.argv:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(hist_digests(M, d), indent=1))
