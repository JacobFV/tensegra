"""Extended-07 Phase 2 infrastructure: common initialization, the common valid-history bank, out-of-fold factor
predictions for controlled consumers, seed-range checks and the launch-command matrix.

Training itself is done by campaign04_probeworld_train.py train (extended-07 options; see p2-infra.md):
  S x R 2x2 arms   --arch fuse --shape {0,1} --read {0,1} --init-from INIT.pt [--bank BANK --bank-updates N]
  K-fold predictor --arch fuse --shape 1 --read 0 --bank BANK --bank-updates N --predictor-only --bank-exclude-fold k
  consumer         --inputs factors6 --arch fuse --bank BANK --bank-updates N --phi-contract exact|oof|mix [--oof F]

Subcommands
  init   --seed S --out DIR          the common initial state_dict of the four S x R arms of a seed (constructed once,
                                     exactly as the trainer constructs them: torch.manual_seed(1000 + S), identical
                                     module list inp..case, aux, fuse), saved as DIR/init.pt + DIR/init.json (sha256)
  bank   --labels L --pool b6_B0 --source SPEC ... --world-base W --out FILE
                                     record valid public histories of fixed behaviour sources on the TRAINING pool with
                                     exact per-decision labels (available set, eps-optimal set, Q*, V*, the 23 exact
                                     factor targets, reward, return-to-go).  SPEC:
                                       pistar:R            pi* (lowest-index representative), R worlds per config
                                       eps:E:R             pi* with probability-E uniform deviations (counter RNG)
                                       model:NAME=RUN:R    a frozen checkpoint sampling on-policy (softmax; counter RNG)
                                     world seed = W + 1e6 * source_index + 1000 * cfg_idx + r
  oof    --bank FILE --fold-run k=RUN ... [--full-run RUN] --out FILE
                                     out-of-fold predictions of K predictors (each trained with --bank-exclude-fold k)
                                     for every bank decision (a decision is predicted by the model that never saw its
                                     configuration), per-coordinate RMS / MAE (the mix contract's sigma), per-group
                                     nMAE, and (optional) the in-sample error of the full-data predictor
  seeds                              check the proposed P2 seed / world ranges against seed-ranges.json (disjointness)
  jobs   SHA [--smoke]               print the validated launch commands (campaign07_remote.py launch-cmd)
  jobs   SHA --confirm [--run-tag T] P2-CONFIRM: the b6d label jobs (e07-p2c-labels-eval / -cf) and the b6d diag
                                     evaluation of the confirmation lineages (seeds 50-54; e07-p2c-eval-s<seed>)
"""
from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign04_probeworld_train as T  # noqa: E402  (inserts src/ on sys.path)
from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign06_probeworld as pw6  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SEED_RANGES = REPO / "research" / "campaigns" / "extended-07" / "seed-ranges.json"


# ------------------------------------------------------------------------------------------------ init

def make_init(seed, hidden=T.HIDDEN):
    """The S x R model exactly as cmd_train constructs it for a b6 pool (public_extra 4)."""
    torch.manual_seed(1000 + seed)
    return T.ProbeNet(hidden, arch="fuse", public_extra=pw6.PUBLIC_EXTRA6, factor_mode="sr", shape=1, read=1)


def cmd_init(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "init.pt").exists():
        raise SystemExit(f"refusing to overwrite {out / 'init.pt'}")
    model = make_init(a.seed, a.hidden)
    sd = model.state_dict()
    torch.save(sd, out / "init.pt")
    info = {"seed": a.seed, "torch_seed": 1000 + a.seed, "hidden": a.hidden, "sha256": T.state_sha(sd),
            "params": T.n_params(model), "tensors": sorted(sd),
            "note": "common initial parameters of S0R0/S1R0/S0R1/S1R1 (phi_const is a non-persistent buffer)"}
    (out / "init.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("seed", "sha256", "params")}))


# ------------------------------------------------------------------------------------------------ bank

def parse_source(spec):
    kind = spec.split(":", 1)[0]
    if kind == "pistar":
        return {"kind": "pistar", "name": "pistar", "reps": int(spec.split(":")[1])}
    if kind == "eps":
        _, e, r = spec.split(":")
        return {"kind": "eps", "name": f"eps{e}", "eps": float(e), "reps": int(r)}
    if kind == "model":
        body, r = spec[len("model:"):].rsplit(":", 1)
        name, run = body.split("=", 1)
        return {"kind": "model", "name": name, "run": run, "reps": int(r)}
    raise SystemExit(f"unknown bank source {spec!r}")


def label_episode(cfg, s, infos, ws, cfg_idx, src_name, rep, ep):
    """Exact per-decision labels along a finished episode (states recorded before each decision)."""
    env = T._env(cfg)
    e = {"id": f"{src_name}:{cfg_idx}:{rep}", "src": src_name, "cfg_idx": cfg_idx,
         "combo": pw6.combo_name(cfg.flags), "k": cfg.k, "ws": ws, "vec": list(cfg.public_vector()),
         "hist": [], "avail": [], "q": [], "opt": [], "Q": [], "V": [], "phi": [], "r": [], "a_opt": []}
    for info in infos:
        st = info["state"]
        av = sorted(env.available(cfg, st))
        q = s.q_values(st)
        e["hist"].append(list(info["rec"]))
        e["avail"].append(av)
        e["q"].append(st[0][0])
        e["opt"].append(sorted(s.opt_set(st)))
        e["Q"].append([q[b] for b in av])
        e["V"].append(s.value(st))
        e["phi"].append(pw6.factor_features(cfg, st))
        e["r"].append(info["reward"])
        e["a_opt"].append(info["a"] in e["opt"][-1])
    g, G = 0.0, []
    for r in reversed(e["r"]):
        g += r
        G.append(g / 100.0)
    e["G"] = G[::-1]
    e["U"] = ep.utility
    e["V0"] = s.value(pw.initial_state(cfg))
    return e


def run_rule(cfg, s, ws, choose):
    ep = T._env(cfg).Episode(cfg, ws)
    infos = []
    while not ep.done:
        st = ep.state
        a = choose(st, ep.available(), len(infos))
        n0, s0 = len(ep.ledger), ep.successes
        r = ep.step(a)
        infos.append({"state": st, "a": a, "rec": r, "reward": T.pw_reward(ep, n0, s0)})
    return ep, infos


def cmd_bank(a):
    torch.set_num_threads(1)
    t0 = time.process_time()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pool = T.load_pool(a.labels, a.pool)
    if a.n_configs:  # evenly spaced over the pool (b6 pools are stored in combination blocks: keeps every combination)
        n = len(pool)
        pool = [pool[(i * n) // a.n_configs] for i in range(min(a.n_configs, n))]
    sources = [parse_source(x) for x in a.source]
    assert len({x["name"] for x in sources}) == len(sources), "duplicate source names"
    assert len(pool) <= 1000 and all(x["reps"] <= 1000 for x in sources) and len(sources) <= 10
    episodes, src_meta = [], {}
    for j, src in enumerate(sources):
        ts = time.process_time()
        ws_of = lambda idx, r: a.world_base + 1_000_000 * j + 1000 * idx + r  # noqa: E731
        todo = [(idx, cfg, s, r) for idx, cfg, s in pool for r in range(src["reps"])]
        if src["kind"] in ("pistar", "eps"):
            for idx, cfg, s, r in todo:
                ws = ws_of(idx, r)
                if src["kind"] == "pistar":
                    choose = lambda st, av, t, s=s: s.pi_star(st)  # noqa: E731
                else:
                    def choose(st, av, t, s=s, ws=ws, e=src["eps"]):
                        rng = random.Random(f"e07-bank:{a.world_base}:{j}:{ws}:{t}")
                        if rng.random() < e:
                            return sorted(av)[rng.randrange(len(av))]
                        return s.pi_star(st)
                ep, infos = run_rule(cfg, s, ws, choose)
                episodes.append(label_episode(cfg, s, infos, ws, idx, src["name"], r, ep))
        else:
            import campaign07_diag as D
            model, meta = D.load_model(src["run"])
            src["model_sha256"] = D._sha_file(Path(src["run"]) / "model.pt")
            src["model_kind"] = D.model_kind(model)
            rng = T.CounterRNG(f"bank:{a.world_base}:{j}")
            for c0 in range(0, len(todo), 256):
                part = todo[c0:c0 + 256]
                items = [(cfg, s, ws_of(idx, r)) for idx, cfg, s, r in part]
                with torch.no_grad():
                    eps_, steps, ep_steps = T.run_batch(model, items, "sample", rng, need_labels=False)
                for (idx, cfg, s, r), ep, infos in zip(part, eps_, ep_steps):
                    episodes.append(label_episode(cfg, s, infos, ws_of(idx, r), idx, src["name"], r, ep))
        mine = [e for e in episodes if e["src"] == src["name"]]
        nd = sum(len(e["hist"]) for e in mine)
        src_meta[src["name"]] = {**src, "index": j, "episodes": len(mine), "decisions": nd,
                                 "world_seeds": [a.world_base + 1_000_000 * j, a.world_base + 1_000_000 * j + 1000 * len(pool)],
                                 "behaviour_eps_optimal_rate": sum(sum(e["a_opt"]) for e in mine) / max(nd, 1),
                                 "mean_regret": sum(e["V0"] - e["U"] for e in mine) / max(len(mine), 1),
                                 "cpu_s": round(time.process_time() - ts, 2)}
        print(json.dumps({k: v for k, v in src_meta[src["name"]].items() if k != "world_seeds"}), flush=True)
    combos = {}
    for e in episodes:
        combos[e["combo"]] = combos.get(e["combo"], 0) + 1
    meta = {"format": T.BANK_FORMAT, "labels": a.labels, "pool": a.pool, "n_configs": len(pool),
            "world_base": a.world_base, "sources": src_meta, "episodes": len(episodes),
            "decisions": sum(len(e["hist"]) for e in episodes), "combo_episodes": combos,
            "labels_note": "per decision: avail (sorted), q = queries done, opt = solver opt_set (trainer imitation "
                           "target), Q* over avail, V*, phi = campaign06 factor_features (23 exact targets), reward, "
                           "G = return-to-go / 100; the behaviour action is hist[t][0]",
            "folds_rule": "campaign04 train bank_folds: per combination, configurations in index order -> j % K",
            "cpu_s": round(time.process_time() - t0, 2)}
    import pickle
    with open(out, "xb") as f:
        pickle.dump({"meta": meta, "episodes": episodes}, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta["sha256"] = T._sha_path(out)
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("episodes", "decisions", "combo_episodes", "cpu_s", "sha256")}))


# ------------------------------------------------------------------------------------------------ out-of-fold

@torch.no_grad()
def predict_bank(model, eps, chunk=256):
    """{episode id: [[23] per decision]} of a predictor's aux output along the stored histories."""
    out = {}
    for c0 in range(0, len(eps), chunk):
        part = eps[c0:c0 + chunk]
        steps = T.replay_bank_batch(model, part)
        for t, rec in enumerate(steps):
            rows = rec["aux"].tolist()
            for j, i in enumerate(rec["idx"].tolist()):
                out.setdefault(part[i]["id"], []).append(rows[j])
    return out


def err_stats(eps, preds):
    import numpy as np
    P = np.array([row for e in eps for row in preds[e["id"]]], dtype=float)
    Y = np.array([row for e in eps for row in e["phi"]], dtype=float)
    d = P - Y
    groups = T.contract_groups()
    sd = Y.std(0)
    nmae = {g: float(np.mean([np.abs(d[:, j]).mean() / sd[j] for j in idx if sd[j] > 0])) for g, idx in groups.items()}
    return {"n": int(len(P)), "err_rms": np.sqrt((d ** 2).mean(0)).tolist(), "err_mae": np.abs(d).mean(0).tolist(),
            "target_std": sd.tolist(), "group_nmae": nmae, "mse": float((d ** 2).mean())}


def cmd_oof(a):
    import campaign07_diag as D
    torch.set_num_threads(1)
    t0 = time.process_time()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    bank = T.load_bank(a.bank)
    bsha = T._sha_path(a.bank)
    runs = {int(k): v for k, v in (x.split("=", 1) for x in a.fold_run)}
    K = len(runs)
    assert sorted(runs) == list(range(K)), "need fold runs 0..K-1"
    folds = T.bank_folds(bank, K)
    preds, fold_info = {}, {}
    for k, run in sorted(runs.items()):
        model, meta = D.load_model(run)
        b = meta.get("p2", {}).get("bank", {})
        assert b.get("sha256") == bsha and b.get("exclude_fold") == k and b.get("folds") == K, \
            f"fold run {run} was not trained on this bank with --bank-exclude-fold {k} --bank-folds {K}"
        eps = [e for e in bank["episodes"] if folds[e["cfg_idx"]] == k]
        preds.update(predict_bank(model, eps))
        fold_info[k] = {"run": run, "model_sha256": D._sha_file(Path(run) / "model.pt"), "episodes": len(eps),
                        "configs": sorted({e["cfg_idx"] for e in eps}), "seed": meta["seed"]}
    assert set(preds) == {e["id"] for e in bank["episodes"]}
    res = {"bank_sha256": bsha, "bank": a.bank, "folds": K, "fold_runs": fold_info, "preds": preds,
           **err_stats(bank["episodes"], preds)}
    if a.full_run:
        model, meta = D.load_model(a.full_run)
        full = predict_bank(model, bank["episodes"])
        res["full_run"] = {"run": a.full_run, "model_sha256": D._sha_file(Path(a.full_run) / "model.pt"),
                           "in_sample": {k: v for k, v in err_stats(bank["episodes"], full).items()}}
    res["cpu_s"] = round(time.process_time() - t0, 2)
    torch.save(res, out)
    summ = {k: v for k, v in res.items() if k != "preds"}
    Path(str(out) + ".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"oof_group_nmae": res["group_nmae"], "mse": res["mse"],
                      "full_in_sample_nmae": res.get("full_run", {}).get("in_sample", {}).get("group_nmae"),
                      "cpu_s": res["cpu_s"]}))


# ------------------------------------------------------------------------------------------------ seed ranges

P2_SEEDS = (40, 41, 42, 43, 44)
DEV_SEEDS = (97,)       # tests (tests/test_campaign07_p2.py)
SMOKE_SEEDS = (47,)     # P2 smokes on pro6000
BANK_WORLD_BASE = 12_900_000_000        # protocol bank (sources j < 10: [base, base + 1e7))
BANK_DEV_WORLD_BASE = 12_950_000_000    # smoke bank


def proposed_ranges():
    out = []
    for s in P2_SEEDS:
        out.append({"name": f"ext07 P2 training worlds seed {s} (8e9 + 1e8*seed; on-policy / fine-tuning; + support "
                            f"reference offset 7e7)", "lo": T.TRAIN_WORLD_BASE + s * 100_000_000,
                    "hi": T.TRAIN_WORLD_BASE + (s + 1) * 100_000_000, "status": "proposed"})
    for s in SMOKE_SEEDS + DEV_SEEDS:
        out.append({"name": f"ext07 P2 dev/smoke training worlds seed {s}", "lo": T.TRAIN_WORLD_BASE + s * 100_000_000,
                    "hi": T.TRAIN_WORLD_BASE + (s + 1) * 100_000_000, "status": "development (proposed)"})
    out.append({"name": "ext07 P2 history bank worlds (protocol; W + 1e6*source + 1000*cfg_idx + r)",
                "lo": BANK_WORLD_BASE, "hi": BANK_WORLD_BASE + 10_000_000, "status": "proposed"})
    out.append({"name": "ext07 P2 history bank worlds (smoke/dev)", "lo": BANK_DEV_WORLD_BASE,
                "hi": BANK_DEV_WORLD_BASE + 10_000_000, "status": "development (proposed)"})
    out.append({"name": "ext07 P2 tests configs (dev_smoke 6.890-6.891e9)", "lo": 6_890_000_000,
                "hi": 6_891_000_000, "status": "development (proposed; inside the ext06 dev_smoke parent)"})
    return out


def check_ranges(proposed, path=SEED_RANGES):
    reg = json.loads(Path(path).read_text())["ranges"]
    # development ranges may nest inside the ext06 dev_smoke sub-range (and its parent), as the registered ext07
    # dev blocks 6.870-6.890e9 do; they may not overlap any other entry (including those ext07 dev blocks)
    smoke = [r for r in reg if r["name"].endswith("dev_smoke")]
    containers = {r["name"] for r in smoke} | {r.get("parent") for r in smoke if r.get("parent")}
    clashes = []
    for p in proposed:
        for r in reg:
            if p["lo"] < r["hi"] and r["lo"] < p["hi"]:
                if "dev" in p["status"] and r["name"] in containers and r["lo"] <= p["lo"] and p["hi"] <= r["hi"]:
                    continue
                clashes.append((p["name"], r["name"]))
    for i, p in enumerate(proposed):
        for q in proposed[i + 1:]:
            if p["lo"] < q["hi"] and q["lo"] < p["hi"]:
                clashes.append((p["name"], q["name"]))
    return clashes


def cmd_seeds(a):
    prop = proposed_ranges()
    clashes = check_ranges(prop)
    print(json.dumps({"proposed": prop, "clashes": clashes}, indent=1))
    if clashes:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ jobs

R6 = "/home/brand/structured-latent-dynamics-campaign06/results"
R7 = "/home/brand/structured-latent-dynamics-campaign07/results"
PYR = "/home/brand/structured-latent-dynamics-campaign03/env/bin/python"
TB = f"{R6}/e06-tb-labels/labels"
TBC = f"{R6}/e06-tbc-labels/labels"
UCE = f"{R6}/e06-tb-labels-uce/labels/cf_UCE.json"
TRAIN = "research/tools/campaign04_probeworld_train.py"
P2 = "research/tools/campaign07_p2.py"
DIAG = "research/tools/campaign07_diag.py"
ENV = ["env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "PYTHONPATH=src", PYR]
ARMS = {"s0r0": (0, 0), "s1r0": (1, 0), "s0r1": (0, 1), "s1r1": (1, 1)}
CONTRACTS = ("exact", "oof", "mix")


def job_matrix(sha, smoke=False, seeds=P2_SEEDS, bank_updates=4000, finetune=0, k=5, onpolicy=0):
    """[(stage, job name, caps, needs, mkdir, argv)] in dependency order.  Smoke: one seed, tiny budgets."""
    import campaign07_remote as rem
    tag = "e07-p2s" if smoke else "e07-p2"
    wb = BANK_DEV_WORLD_BASE if smoke else BANK_WORLD_BASE
    bank_dir = f"{R7}/{tag}-bank"
    bank = f"{bank_dir}/bank.pkl"
    srcs = (["pistar:1", "eps:0.2:1", f"model:RAWF-s30={R6}/e06-tb-rawf-s30/run:1"] if smoke else
            ["pistar:2", "eps:0.1:2", "eps:0.3:2", f"model:RAWF-s30={R6}/e06-tb-rawf-s30/run:4",
             f"model:B0-s30={R6}/e06-tb-b0-s30/run:4"])
    n_cfg = ["--n-configs", "24"] if smoke else []
    bu = 20 if smoke else bank_updates
    jobs = []

    def add(stage, name, caps, needs, mkdir, cmd):
        argv = [name, sha, "--wall-cap", str(caps[0]), "--cpu-cap", str(caps[1])]
        for n in needs:
            argv += ["--needs", n]
        for m in mkdir:
            argv += ["--mkdir", m]
        argv += ["--"] + ENV + cmd
        pa = rem.parse_launch(argv)
        assert pa.job.startswith("e07-")
        rem.check_paths_in_cmd(pa.cmd)
        for d in pa.mkdir:
            assert d.startswith(f"{rem.ROOT}/results/")
        jobs.append((stage, name, argv))

    models = [f"{R6}/e06-tb-rawf-s30/run/model.pt"] + ([] if smoke else [f"{R6}/e06-tb-b0-s30/run/model.pt"])
    add("bank", f"{tag}-bank", (3600, 3600), [f"{TB}/b6_B0.pkl"] + models, [bank_dir],
        [P2, "bank", "--labels", TB, "--pool", "b6_B0", *n_cfg, "--world-base", str(wb), "--source", *srcs,
         "--out", bank])
    seeds = SMOKE_SEEDS if smoke else seeds
    for s in seeds:
        init = f"{R7}/{tag}-init-s{s}"
        add("init", f"{tag}-init-s{s}", (600, 300), [], [init], [P2, "init", "--seed", str(s), "--out", init])
    common = ["--labels", TB, "--train-split", "b6_B0", "--train-n", "384", "--rung", "L1", "--batch",
              "8" if smoke else "64"]
    for s in seeds:
        init = f"{R7}/{tag}-init-s{s}/init.pt"
        for arm, (sh, rd) in ARMS.items():
            o = f"{R7}/{tag}-{arm}-bank-s{s}"
            ft = ["--finetune", "--updates", str(finetune)] if finetune else ["--updates", "0"]
            add("2x2-bank", f"{tag}-{arm}-bank-s{s}", (7200, 7200), [bank, init, f"{TB}/b6_B0.pkl"], [o],
                [TRAIN, "train", *common, "--seed", str(s), "--arch", "fuse", "--shape", str(sh), "--read", str(rd),
                 "--init-from", init, "--bank", bank, "--bank-updates", str(bu), *ft, "--log-every",
                 "5" if smoke else "100", "--out", f"{o}/run"])
            if smoke or onpolicy:  # the historical-style on-policy 2x2 (no bank; RL + imitation + aux)
                o = f"{R7}/{tag}-{arm}-onp-s{s}"
                add("2x2-onpolicy", f"{tag}-{arm}-onp-s{s}", (1800, 1800) if smoke else (7200, 7200),
                    [init, f"{TB}/b6_B0.pkl"], [o],
                    [TRAIN, "train", *common, "--seed", str(s), "--arch", "fuse", "--shape", str(sh), "--read",
                     str(rd), "--init-from", init, "--updates", "10" if smoke else str(onpolicy), "--log-every",
                     "5" if smoke else "100", "--out", f"{o}/run"])
    kk = 2 if smoke else k
    pbu = 20 if smoke else bank_updates
    for s in seeds:
        for f in list(range(kk)) + ["full"]:
            o = f"{R7}/{tag}-pred-f{f}-s{s}"
            fold = [] if f == "full" else ["--bank-folds", str(kk), "--bank-exclude-fold", str(f)]
            add("predictor", f"{tag}-pred-f{f}-s{s}", (7200, 7200), [bank, f"{TB}/b6_B0.pkl"], [o],
                [TRAIN, "train", *common, "--seed", str(s), "--arch", "fuse", "--shape", "1", "--read", "0",
                 "--bank", bank, "--bank-updates", str(pbu), "--predictor-only", *fold, "--updates", "0",
                 "--log-every", "5" if smoke else "100", "--out", f"{o}/run"])
        oof_dir = f"{R7}/{tag}-oof-s{s}"
        add("oof", f"{tag}-oof-s{s}", (1800, 1800),
            [bank] + [f"{R7}/{tag}-pred-f{f}-s{s}/run/model.pt" for f in list(range(kk)) + ["full"]], [oof_dir],
            [P2, "oof", "--bank", bank, *sum((["--fold-run", f"{f}={R7}/{tag}-pred-f{f}-s{s}/run"]
                                              for f in range(kk)), []),
             "--full-run", f"{R7}/{tag}-pred-ffull-s{s}/run", "--out", f"{oof_dir}/oof.pt"])
        for c in CONTRACTS:
            o = f"{R7}/{tag}-cons-{c}-s{s}"
            extra = [] if c == "exact" else ["--oof", f"{oof_dir}/oof.pt"]
            add("consumer", f"{tag}-cons-{c}-s{s}", (7200, 7200),
                [bank, f"{TB}/b6_B0.pkl"] + ([f"{oof_dir}/oof.pt"] if c != "exact" else []), [o],
                [TRAIN, "train", *common, "--seed", str(s), "--inputs", "factors6", "--arch", "fuse", "--bank", bank,
                 "--bank-updates", str(pbu), "--phi-contract", c, *extra, "--updates", "0",
                 "--log-every", "5" if smoke else "100", "--out", f"{o}/run"])
    for s in seeds:  # evaluation: diag runner, b6c_hold_SCE + SCE octets (+ UCE), free-running + pi* histories
        o = f"{R7}/{tag}-eval-s{s}"
        m, needs = [], []
        for arm in ARMS:
            for kind in ("bank",) + (("onp",) if onpolicy else ()):
                m += ["--model", f"{arm.upper()}-{kind}-s{s}={R7}/{tag}-{arm}-{kind}-s{s}/run"]
                needs.append(f"{R7}/{tag}-{arm}-{kind}-s{s}/run/model.pt")
        pred = f"{R7}/{tag}-pred-ffull-s{s}/run"
        for c in CONTRACTS:
            run = f"{R7}/{tag}-cons-{c}-s{s}/run"
            m += ["--model", f"CONS-{c}-exact-s{s}={run}::exact", "--model", f"CONS-{c}-pred-s{s}={run}::pred={pred}"]
            needs.append(f"{run}/model.pt")
        needs.append(f"{pred}/model.pt")
        sub = ["--n-configs", "24", "--n-octets", "8"] if smoke else []
        add("eval", f"{tag}-eval-s{s}", (7200, 7200), needs + [f"{TBC}/b6c_hold_SCE.shards.json", f"{TBC}/cf_SCE.json"],
            [o], [DIAG, "run", "--labels", TBC, "--pool", "b6c_hold_SCE", "--cf", f"{TBC}/cf_SCE.json",
                  *([] if smoke else ["--cf", UCE]), *m, "--protocols", "B", "A-pistar",
                  "--ivs", "none", "exact", "--no-latent", *sub, "--tag", f"{tag}-s{s}", "--out", o])
    return jobs


# ------------------------------------------------------------------------------------------------ P2-CONFIRM (b6d)

CONFIRM_SEEDS = (50, 51, 52, 53, 54)  # registry P2-CONFIRM (training worlds 1.30e10-1.35e10, registered)
B6D = f"{R7}/e07-p2c-labels/labels"   # the b6d labels dir (both parts write into it; built before any evaluation)
B6D_N_HOLD, B6D_N_CF = 400, 320       # registry P2-CONFIRM: 400 configurations, 320 octets


def _launch_argv(sha, name, caps, needs, mkdir, cmd, max_concurrent=None):
    import campaign07_remote as rem
    argv = [name, sha] + (["--max-concurrent", str(max_concurrent)] if max_concurrent else []) + [
        "--cpu-cap", str(caps[1]), "--wall-cap", str(caps[0])]
    for n in needs:
        argv += ["--needs", n]
    for m in mkdir:
        argv += ["--mkdir", m]
    argv += ["--"] + ENV + cmd
    pa = rem.parse_launch(argv)
    assert pa.job.startswith("e07-") and (max_concurrent or 0) <= rem.MAX_CONCURRENT_CEILING
    rem.check_paths_in_cmd(pa.cmd)
    assert all(d.startswith(f"{rem.ROOT}/results/") for d in pa.mkdir)
    return argv


def b6d_jobs(sha, seeds=CONFIRM_SEEDS, run_tag="e07-p2", eval_tag="e07-p2c", onpolicy=False, labels=B6D,
             max_concurrent=6):
    """P2-CONFIRM launch commands: the two b6d label parts (eval: b6d_hold_SCE, 400 configurations; cf: 320 SCE
    octets; both into one labels dir, built before any confirmation evaluation) and, per confirmation seed, the diag
    evaluation of the frozen lineage (the four S x R arms + the consumers as '::exact' and '::pred=PREDRUN' specs; the
    same model set / protocols / interventions as the screen's eval stage) on the b6d pool and the b6d SCE octets
    ONLY (no UCE; campaign07_diag refuses mixed populations).  run_tag: the tag the lineages were trained under
    (job_matrix(sha, seeds=CONFIRM_SEEDS) names them e07-p2-<arm>-bank-s<seed>)."""
    lab_dir = str(Path(labels).parent)
    base = ["research/tools/campaign04_probeworld_train.py", "labels", "--split-set", "b6d"]
    jobs = [("b6d-labels", "e07-p2c-labels-eval",
             _launch_argv(sha, "e07-p2c-labels-eval", (14400, 8000), [PYR], [labels],
                          base + ["--parts", "eval", "--n-hold", str(B6D_N_HOLD), "--out", labels], max_concurrent)),
            ("b6d-labels", "e07-p2c-labels-cf",
             _launch_argv(sha, "e07-p2c-labels-cf", (14400, 8000), [PYR], [labels],
                          base + ["--parts", "cf", "--n-cf", str(B6D_N_CF), "--out", labels], max_concurrent))]
    assert lab_dir.startswith(f"{R7}/")
    for s in seeds:
        o = f"{R7}/{eval_tag}-eval-s{s}"
        m, needs = [], []
        for arm in ARMS:
            for kind in ("bank",) + (("onp",) if onpolicy else ()):
                m += ["--model", f"{arm.upper()}-{kind}-s{s}={R7}/{run_tag}-{arm}-{kind}-s{s}/run"]
                needs.append(f"{R7}/{run_tag}-{arm}-{kind}-s{s}/run/model.pt")
        pred = f"{R7}/{run_tag}-pred-ffull-s{s}/run"
        for c in CONTRACTS:
            run = f"{R7}/{run_tag}-cons-{c}-s{s}/run"
            m += ["--model", f"CONS-{c}-exact-s{s}={run}::exact", "--model", f"CONS-{c}-pred-s{s}={run}::pred={pred}"]
            needs.append(f"{run}/model.pt")
        needs.append(f"{pred}/model.pt")
        jobs.append(("b6d-eval", f"{eval_tag}-eval-s{s}", _launch_argv(
            sha, f"{eval_tag}-eval-s{s}", (7200, 7200),
            needs + [f"{labels}/b6d_hold_SCE.shards.json", f"{labels}/cf_SCE.json",
                     f"{labels}/labels_meta.b6d.eval.json", f"{labels}/labels_meta.b6d.cf.json"], [o],
            [DIAG, "run", "--labels", labels, "--pool", "b6d_hold_SCE", "--cf", f"{labels}/cf_SCE.json", *m,
             "--protocols", "B", "A-pistar", "--ivs", "none", "exact", "--no-latent", "--tag", f"{eval_tag}-s{s}",
             "--out", o], max_concurrent)))
    return jobs


def cmd_jobs(a):
    if a.confirm:
        for stage, name, argv in b6d_jobs(a.sha, run_tag=a.run_tag, onpolicy=bool(a.onpolicy)):
            print(f"# [{stage}] {name}")
            print("python research/tools/campaign07_remote.py launch-cmd " + shlex.join(argv))
        return
    jobs = job_matrix(a.sha, smoke=a.smoke, finetune=a.finetune, bank_updates=a.bank_updates, onpolicy=a.onpolicy)
    for stage, name, argv in jobs:
        print(f"# [{stage}] {name}")
        print("python research/tools/campaign07_remote.py launch-cmd " + shlex.join(argv))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--hidden", type=int, default=T.HIDDEN)
    s.add_argument("--out", required=True)
    s = sub.add_parser("bank")
    s.add_argument("--labels", required=True)
    s.add_argument("--pool", default="b6_B0")
    s.add_argument("--n-configs", type=int, default=None)
    s.add_argument("--source", nargs="+", required=True)
    s.add_argument("--world-base", type=int, required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("oof")
    s.add_argument("--bank", required=True)
    s.add_argument("--fold-run", action="append", required=True, metavar="K=RUNDIR")
    s.add_argument("--full-run", default=None)
    s.add_argument("--out", required=True)
    sub.add_parser("seeds")
    s = sub.add_parser("jobs")
    s.add_argument("sha")
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--finetune", type=int, default=0)
    s.add_argument("--bank-updates", type=int, default=4000)
    s.add_argument("--onpolicy", type=int, default=0, help="also the historical-style on-policy 2x2 (U updates)")
    s.add_argument("--confirm", action="store_true",
                   help="P2-CONFIRM: only the b6d label jobs and the b6d diag evaluation of seeds 50-54")
    s.add_argument("--run-tag", default="e07-p2", help="--confirm: tag the confirmation lineages were trained under")
    a = p.parse_args(argv)
    {"init": cmd_init, "bank": cmd_bank, "oof": cmd_oof, "seeds": cmd_seeds, "jobs": cmd_jobs}[a.cmd](a)


if __name__ == "__main__":
    main()
