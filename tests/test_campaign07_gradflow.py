"""Extended-07 P0b: gradient-flow audit of the extended-06 factorized arms (RAWF / LRN / SUP; B0 for reference),
built exactly as the trainer builds them (``campaign04_probeworld_train.py train`` with ``--updates 0`` for the
initial weights; ``run_batch`` + ``losses`` for the gradients).

Asserted:
  * which loss term (policy-gradient, entropy, set-valued imitation, value, auxiliary factor MSE, and the five
    zero-weight L1 heads) puts a NONZERO gradient on which parameter group (inp, gru, trunk, aux, fuse_z, fuse_phi,
    fuse_b, pi, v, q, stage, dep, switch, case);
  * the decomposition reproduces ``losses()``'s total exactly (so the table is the trainer's);
  * parameter counts (total / receiving gradient under L1);
  * initialization identity across arms under one seed (which tensors differ and why: constructor order);
  * coupling of the auxiliary loss into actor training beyond the trunk: the global-norm clip coefficient, the shared
    Adam optimizer's trunk moments, the torch RNG (untouched after init), the python action-sampling RNG (one draw
    per active decision: trajectories desynchronize the stream), and the unnormalized MSE;
  * the LRN auxiliary targets logged by run_batch equal an independent recomputation from the public history.

Configurations/worlds: dev_smoke block 6.879e9 (never protocol).  CPU only, 1 thread.
``python tests/test_campaign07_gradflow.py`` prints the JSON report used in gradient-flow.md.
"""
import importlib.util
import json
import pathlib
import random
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign06_probeworld as pw6  # noqa: E402
from test_campaign07_factor_contract import factors_from_history  # noqa: E402

DEVG = 6_879_000_000
assert pw6.SUBRANGES["dev_smoke"][0] <= DEVG and DEVG + 1_000_000 <= pw6.SUBRANGES["dev_smoke"][1]
SEED = 35
ARMS = {  # arm: trainer flags (on top of --rung L1 --seed 35), exactly as extended-06 launched them
    "B0": [],
    "RAWF": ["--arch", "fuse"],
    "LRN": ["--arch", "fuse", "--factor-mode", "learned", "--aux-weight", "1"],
    "SUP": ["--inputs", "factors6", "--arch", "fuse"],
}
GROUPS = ("inp", "gru", "trunk", "aux", "fuse_z", "fuse_phi", "fuse_b", "pi", "v", "q", "stage", "dep", "switch",
          "case")
TERMS = ("pg", "entropy", "imit", "value", "aux", "zero_weight_heads")
torch.set_num_threads(1)


def _load():
    spec = importlib.util.spec_from_file_location("pwtrain_e07", ROOT / "research/tools/campaign04_probeworld_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load()


def _pool(n=12):
    out = []
    for i in range(n):
        cfg = pw6.stream_config(DEVG, i, pw.TRAIN_CELLS, (1, 2), pw6.TRAIN_TYPES)[0]
        s = pw6.ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        out.append((i, cfg, s))
    return out


POOL = _pool()


def init_state(arm, tmp, seed=SEED, pool=POOL):
    """Initial weights exactly as cmd_train creates them (``train --updates 0`` saves the untouched model)."""
    orig = M.load_pool
    M.load_pool = lambda labels, split: pool
    try:
        d = pathlib.Path(tmp) / f"{arm}-{seed}"
        M.main(["train", "--labels", "unused", "--out", str(d), "--rung", "L1", "--seed", str(seed),
                "--updates", "0", *ARMS[arm]])
    finally:
        M.load_pool = orig
    meta = json.loads((d / "train_meta.json").read_text())
    return torch.load(d / "model.pt"), meta


def build(arm, tmp):
    sd, meta = init_state(arm, tmp)
    fuse = arm != "B0"
    fm = {"RAWF": None, "LRN": "learned", "SUP": None}.get(arm)
    model = M.ProbeNet(meta["hidden"], inputs=meta.get("inputs", "public"), arch="fuse" if fuse else "flat",
                       public_extra=pw6.PUBLIC_EXTRA6, factor_mode=fm)
    model.load_state_dict(sd)
    return model, meta


def batch(model, B=12, seed=0):
    items = [(cfg, s, DEVG + 500_000 + 100 * seed + j) for j, (_, cfg, s) in enumerate(POOL[:B])]
    eps, steps, ep_steps = M.run_batch(model, items, "sample", random.Random(DEVG + seed))
    M.attach_case_labels(items, ep_steps, steps)
    return items, eps, steps, ep_steps


def terms(model, items, eps, steps, ep_steps):
    """The trainer's L1 objective split into its terms (the value term includes its 0.5 weight, entropy its -0.01)."""
    R = 100.0
    G = {}
    for info_list in ep_steps:
        g = 0.0
        for info in reversed(info_list):
            g += info["reward"]
            G[(info["step_row"], info["row"])] = g / R
    pg, vl, ent, il = [], [], [], []
    for t, rec in enumerate(steps):
        n = len(rec["idx"])
        Gt = torch.tensor([G[(t, j)] for j in range(n)])
        v = model.v(rec["z"]).squeeze(-1)
        lp = rec["logp_all"].gather(1, rec["chosen"].unsqueeze(1)).squeeze(1)
        pg.append(-(lp * (Gt - v).detach()))
        vl.append((v - Gt) ** 2)
        p = rec["logp_all"].exp()
        ent.append(-(p * rec["logp_all"].masked_fill(~rec["mask"], 0.0)).sum(-1))
        il.append(-torch.logsumexp(rec["logp_all"].masked_fill(~rec["opt"], -1e9), -1))
    cat = lambda xs: torch.cat(xs).mean()
    out = {"pg": cat(pg), "value": 0.5 * cat(vl), "entropy": -0.01 * cat(ent), "imit": cat(il)}
    _, parts = M.losses(model, items, eps, steps, ep_steps, M.RUNG_WEIGHTS["L1"])
    zero = sum(0.0 * parts[k] for k in ("dep", "switch", "q"))  # stage/case enter through dep/switch
    out["zero_weight_heads"] = zero
    if "aux" in steps[0]:
        out["aux"] = parts["aux"]
    return out


def group_params(model):
    H = model.gru.hidden_size
    g = {}
    for name, p in model.named_parameters():
        g.setdefault(name.split(".")[0], []).append((name, p, None))
    if "fuse" in g:
        w = dict((n, p) for n, p, _ in g.pop("fuse"))
        g["fuse_z"] = [("fuse.weight", w["fuse.weight"], (slice(None), slice(0, H)))]
        g["fuse_phi"] = [("fuse.weight", w["fuse.weight"], (slice(None), slice(H, None)))]
        g["fuse_b"] = [("fuse.bias", w["fuse.bias"], None)]
    return g


def grad_table(model, loss_terms):
    groups = group_params(model)
    params = list(dict.fromkeys(p for lst in groups.values() for _, p, _ in lst))
    table = {}
    for term, L in loss_terms.items():
        if not torch.is_tensor(L) or not L.requires_grad:
            table[term] = {g: 0.0 for g in groups}
            continue
        grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        gmap = {id(p): gr for p, gr in zip(params, grads)}
        row = {}
        for g, lst in groups.items():
            s = 0.0
            for _, p, sl in lst:
                gr = gmap[id(p)]
                if gr is not None:
                    s += float((gr[sl] if sl is not None else gr).pow(2).sum())
            row[g] = s ** 0.5
        table[term] = row
    return table


EXPECTED_NONZERO = {
    # policy-side terms reach the policy head and everything below it; never the value head or the aux predictor
    "pg": {"pi", "fuse_z", "fuse_phi", "fuse_b", "trunk", "gru", "inp"},
    "entropy": {"pi", "fuse_z", "fuse_phi", "fuse_b", "trunk", "gru", "inp"},
    "imit": {"pi", "fuse_z", "fuse_phi", "fuse_b", "trunk", "gru", "inp"},
    "value": {"v", "fuse_z", "fuse_phi", "fuse_b", "trunk", "gru", "inp"},
    # the auxiliary factor loss: predictor + shared trunk/GRU/input layer; NOT the fusion layer or any head
    "aux": {"aux", "trunk", "gru", "inp"},
    "zero_weight_heads": set(),
}


def expected(arm, term):
    e = set(EXPECTED_NONZERO[term])
    if arm == "B0":
        e -= {"fuse_z", "fuse_phi", "fuse_b"}
    if arm == "RAWF":
        e -= {"fuse_phi"}  # phi == 0: the 23 factor columns of the fusion layer never receive a gradient
    return e


# ------------------------------------------------------------------ tests

@pytest.mark.parametrize("arm", list(ARMS))
def test_gradient_flow_table(arm, tmp_path):
    model, _ = build(arm, tmp_path)
    items, eps, steps, ep_steps = batch(model)
    lt = terms(model, items, eps, steps, ep_steps)
    total, _ = M.losses(model, items, eps, steps, ep_steps,
                        dict(M.RUNG_WEIGHTS["L1"], **({"aux": 1.0} if arm == "LRN" else {})))
    mine = sum(v for v in lt.values())
    assert abs(float(total) - float(mine)) < 1e-5 * max(1.0, abs(float(total)))  # decomposition == trainer
    assert ("aux" in lt) == (arm == "LRN")
    table = grad_table(model, lt)
    for term, row in table.items():
        nz = {g for g, v in row.items() if v > 0.0}
        assert nz == expected(arm, term), (arm, term, sorted(nz), sorted(expected(arm, term)))


def test_total_l1_gradient_zero_on_unused_heads(tmp_path):
    for arm in ARMS:
        model, _ = build(arm, tmp_path)
        items, eps, steps, ep_steps = batch(model)
        w = dict(M.RUNG_WEIGHTS["L1"], **({"aux": 1.0} if arm == "LRN" else {}))
        total, _ = M.losses(model, items, eps, steps, ep_steps, w)
        model.zero_grad()
        total.backward()
        for name, p in model.named_parameters():
            if name.split(".")[0] in ("q", "stage", "dep", "switch", "case"):
                assert p.grad is not None and torch.all(p.grad == 0), name  # exact zeros: Adam leaves them at init
        if arm == "RAWF":
            H = model.gru.hidden_size
            assert torch.all(model.fuse.weight.grad[:, H:] == 0)


def param_counts(tmp):
    out = {}
    for arm in ARMS:
        model, meta = build(arm, tmp)
        total = sum(p.numel() for p in model.parameters())
        dead = sum(p.numel() for n, p in model.named_parameters()
                   if n.split(".")[0] in ("q", "stage", "dep", "switch", "case"))
        H = model.gru.hidden_size
        if arm == "RAWF":
            dead += H * pw6.N_FACTOR_FEATURES  # fuse columns fed by phi == 0
        if arm == "SUP":
            dead += H  # the exp_hard_cost_rel column: the supplied input is exactly 0 on every b6 configuration
        out[arm] = {"total": total, "meta_params": meta["params"], "zero_gradient_under_L1_on_b6": dead,
                    "trained": total - dead}
    return out


def test_parameter_counts(tmp_path):
    pc = param_counts(tmp_path)
    assert {a: v["total"] for a, v in pc.items()} == {"B0": 129_701, "RAWF": 149_157, "LRN": 168_636,
                                                      "SUP": 149_157}
    assert all(v["total"] == v["meta_params"] for v in pc.values())
    assert pc["LRN"]["total"] - pc["RAWF"]["total"] == 128 * 128 + 128 + 128 * 23 + 23  # the aux MLP exactly


def init_diff(tmp):
    sds = {arm: init_state(arm, tmp)[0] for arm in ARMS}
    out = {}
    for a, b in (("RAWF", "SUP"), ("RAWF", "LRN"), ("B0", "RAWF"), ("B0", "LRN")):
        common = sorted(set(sds[a]) & set(sds[b]))
        out[f"{a}|{b}"] = {"common_equal": [k for k in common if torch.equal(sds[a][k], sds[b][k])],
                           "common_differ": [k for k in common if not torch.equal(sds[a][k], sds[b][k])],
                           f"only_{a}": sorted(set(sds[a]) - set(sds[b])),
                           f"only_{b}": sorted(set(sds[b]) - set(sds[a]))}
    return out


def test_initialization_differences(tmp_path):
    d = init_diff(tmp_path)
    assert d["RAWF|SUP"]["common_differ"] == [] and not d["RAWF|SUP"]["only_RAWF"] and not d["RAWF|SUP"]["only_SUP"]
    # aux is constructed BEFORE fuse, so the fusion layer's draw differs between LRN and RAWF/SUP
    assert d["RAWF|LRN"]["common_differ"] == ["fuse.bias", "fuse.weight"]
    assert d["RAWF|LRN"]["only_LRN"] == ["aux.0.bias", "aux.0.weight", "aux.2.bias", "aux.2.weight"]
    assert d["B0|RAWF"]["common_differ"] == [] and d["B0|LRN"]["common_differ"] == []


def test_aux_coupling_paths(tmp_path):
    """LRN: the aux loss never reaches pi / v / fuse before clipping, but changes their update through the global
    clip coefficient and changes the shared trunk's gradient and Adam moments."""
    model, _ = build("LRN", tmp_path)
    items, eps, steps, ep_steps = batch(model)
    lt = terms(model, items, eps, steps, ep_steps)
    actor = lt["pg"] + lt["value"] + lt["entropy"] + lt["imit"]
    params = [p for n, p in model.named_parameters()]
    names = [n for n, _ in model.named_parameters()]
    g_actor = torch.autograd.grad(actor, params, retain_graph=True, allow_unused=True)
    g_tot = torch.autograd.grad(actor + 100.0 * lt["aux"], params, retain_graph=True, allow_unused=True)
    for n, a, b in zip(names, g_actor, g_tot):
        top = n.split(".")[0]
        if top in ("pi", "v", "fuse"):
            assert torch.allclose(a, b, atol=1e-7, rtol=0), n  # pre-clip: identical
        if top in ("trunk", "gru", "inp"):
            assert not torch.allclose(a, b), n
        if top == "aux":
            assert a is None and b is not None

    def norm(gs):
        return float(torch.sqrt(sum((g.pow(2).sum() for g in gs if g is not None), torch.tensor(0.0))))
    n_a, n_t = norm(g_actor), norm(g_tot)
    coef_a, coef_t = min(1.0, 1.0 / (n_a + 1e-6)), min(1.0, 1.0 / (n_t + 1e-6))
    assert coef_t < coef_a  # clip couples the aux loss into every actor parameter's step
    # the torch RNG is not consumed by training (only by initialization)
    st = torch.get_rng_state()
    opt = torch.optim.Adam(params, lr=1e-3)
    items, eps, steps, ep_steps = batch(model, seed=1)
    total, _ = M.losses(model, items, eps, steps, ep_steps, dict(M.RUNG_WEIGHTS["L1"], aux=1.0))
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    assert torch.equal(st, torch.get_rng_state())


class CountingRandom(random.Random):
    n = 0

    def random(self):
        self.n += 1
        return super().random()


def test_action_rng_one_draw_per_decision(tmp_path):
    """run_batch draws one uniform per ACTIVE decision from the shared action RNG, so any difference in episode
    lengths between arms shifts every later draw (common random numbers are lost after the first divergence)."""
    model, _ = build("RAWF", tmp_path)
    items = [(cfg, s, DEVG + 600_000 + j) for j, (_, cfg, s) in enumerate(POOL)]
    rng = CountingRandom(1)
    eps, steps, ep_steps = M.run_batch(model, items, "sample", rng)
    assert rng.n == sum(len(x) for x in ep_steps)


def test_lrn_targets_are_public_history_functions(tmp_path):
    model, _ = build("LRN", tmp_path)
    items, eps, steps, ep_steps = batch(model, seed=2)
    n = 0
    for t, rec in enumerate(steps):
        for j, i in enumerate(rec["idx"].tolist()):
            hist = tuple(eps[i].history[:t])
            g = torch.tensor(factors_from_history(items[i][0].public_vector(), hist))
            assert torch.allclose(rec["aux_t"][j], g.float(), atol=1e-5), (i, t)
            n += 1
    assert n > 30


def test_aux_mse_is_unweighted_mean(tmp_path):
    model, _ = build("LRN", tmp_path)
    items, eps, steps, ep_steps = batch(model)
    _, parts = M.losses(model, items, eps, steps, ep_steps, dict(M.RUNG_WEIGHTS["L1"], aux=1.0))
    a = torch.cat([r["aux"] for r in steps])
    t = torch.cat([r["aux_t"] for r in steps])
    assert torch.allclose(parts["aux"], ((a - t) ** 2).mean(), atol=1e-7)


def report(tmp):
    rep = {"params": param_counts(tmp), "init": init_diff(tmp), "grad_norms": {}}
    for arm in ARMS:
        model, _ = build(arm, tmp)
        items, eps, steps, ep_steps = batch(model)
        rep["grad_norms"][arm] = grad_table(model, terms(model, items, eps, steps, ep_steps))
    return rep


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(report(d), indent=1))
