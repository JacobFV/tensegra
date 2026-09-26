"""Extended-04 Track B: probeworld factorization ladder (L0-L4), CPU only.

Subcommands (all deterministic given their arguments; see research/campaigns/extended-04/probeworld.md):

  labels  --out DIR [--n-train N] [--n-eval N]
      Exact DP label tables for every split pool (oracle label compute; charged as offline CPU).
  train   --labels DIR --out DIR --rung L0..L4 --seed S [--updates U] [--batch B]
      One ladder run.  Every rung: identical model (all heads present), identical config/world stream,
      identical optimizer and number of updates/episodes; rungs differ only in which loss weights are > 0.
      --own-value (protocol-B2, default off): add a separate head v_own on the stop-gradient trunk features,
      trained on the realized return-to-go of the model's OWN free-running greedy rollouts at the current
      parameters (continuation OWN_VALUE_CONTINUATION).  Off -> B1 behaviour bit-identical.
  eval    --labels DIR --run DIR [--worlds W]
      Free-running greedy and teacher-forced (pi* histories) evaluation on dev/test_iid/heldout_* pools.
  summarize --runs DIR... --out FILE
      Aggregate eval JSONs across seeds/rungs (+ reference policies).

Privileged labels (Q*, A*, stage, dependency, switch, case) enter ONLY training losses; model inputs are the
public config vector plus the visible step record and the public available-action mask.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tensegra import campaign04_probeworld as pw  # noqa: E402

RUNGS = ("L0", "L1", "L2", "L3", "L4")
# loss weights per rung (cumulative ladder); every rung keeps the actor-critic RL loss.
RUNG_WEIGHTS = {
    "L0": dict(rl=1.0, imit=0.0, dep=0.0, switch=0.0, q=0.0),
    "L1": dict(rl=1.0, imit=1.0, dep=0.0, switch=0.0, q=0.0),
    "L2": dict(rl=1.0, imit=1.0, dep=0.5, switch=0.0, q=0.0),
    "L3": dict(rl=1.0, imit=1.0, dep=0.5, switch=0.5, q=0.0),
    "L4": dict(rl=1.0, imit=1.0, dep=0.5, switch=0.5, q=1.0),
}
EVAL_SPLITS = ("dev", "test_iid", "heldout_price", "heldout_k", "heldout_comp")
N_STAGES = len(pw.STAGES)
N_DEP = len(pw.DEP_BITS)
STEP_DIM = (pw.N_ACTIONS + 1) + (pw.N_OUTCOMES + 1) + 1 + (4 + 1)  # prev action, outcome, event, reveal
IN_DIM = pw.PUBLIC_DIM + STEP_DIM + pw.N_ACTIONS + 1  # + available mask + queries_done/k
HIDDEN = 128
TRAIN_WORLD_BASE = 8_000_000_000
# protocol-B2 own-greedy rollouts: world seeds TRAIN_WORLD_BASE + 1e8*seed + OWN_WORLD_OFFSET + n.  Inside the
# seed's training band (1e8 wide) but disjoint from the sampled-training worlds (< 256k used per run).
OWN_WORLD_OFFSET = 60_000_000
OWN_VALUE_CONTINUATION = "own_greedy_policy_current_params_mc_v1"


# ------------------------------------------------------------------------------------------------ labels

def build_pool(split: str, n: int):
    pool = []
    for idx in range(n):
        cfg = pw.split_config(split, idx)
        s = pw.ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        pool.append((idx, cfg, s._V, s._Q))
    return pool


def cmd_labels(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split, n in [("train", a.n_train)] + [(s, a.n_eval) for s in EVAL_SPLITS]:
        t0 = time.process_time()
        pool = build_pool(split, n)
        dt = time.process_time() - t0
        with open(out / f"{split}.pkl", "wb") as f:
            pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
        nst = [len(V) for _, _, V, _ in pool]
        meta[split] = {"n_configs": n, "cpu_s": round(dt, 2), "states_total": sum(nst), "states_max": max(nst),
                       "k_counts": {k: sum(1 for _, c, _, _ in pool if c.k == k) for k in pw.K_VALUES},
                       "combo_counts": _combo_counts(pool)}
        print(split, meta[split], flush=True)
    meta["version"] = pw.VERSION
    meta["eps"] = pw.EPS
    meta["continuation"] = pw.CONTINUATION
    (out / "labels_meta.json").write_text(json.dumps(meta, indent=1))


def _combo_counts(pool):
    d = {}
    for _, c, _, _ in pool:
        key = "+".join(n for n, f in zip(pw.FLAG_NAMES, c.flags) if f) or "none"
        d[key] = d.get(key, 0) + 1
    return d


def load_pool(labels_dir, split):
    with open(Path(labels_dir) / f"{split}.pkl", "rb") as f:
        pool = pickle.load(f)
    out = []
    for idx, cfg, V, Q in pool:
        s = pw.ExactSolver(cfg)
        s._V, s._Q = V, Q
        out.append((idx, cfg, s))
    return out


# ------------------------------------------------------------------------------------------------ model

class ProbeNet(nn.Module):
    """GRU over visible-history tokens + public prices.  All heads exist in every rung (matched capacity)."""

    def __init__(self, hidden=HIDDEN, own_value=False):
        super().__init__()
        self.inp = nn.Linear(IN_DIM, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, pw.N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, pw.N_ACTIONS)
        self.stage = nn.Linear(hidden, N_STAGES)
        self.dep = nn.Linear(hidden, N_DEP)
        self.switch = nn.Linear(hidden, 1)
        self.case = nn.Linear(hidden, 4)
        if own_value:  # protocol-B2; created LAST so every B1 parameter gets the identical initialization
            self.v_own = nn.Linear(hidden, 1)

    def step(self, x, h):
        h = self.gru(torch.tanh(self.inp(x)), h)
        z = self.trunk(h)
        return h, z


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def encode(cfg_vec, prev, avail, qfrac):
    """cfg_vec: public list; prev: (a, o, event, reveal) or None; avail: tuple of action ids."""
    x = list(cfg_vec)
    a1 = [0.0] * (pw.N_ACTIONS + 1)
    o1 = [0.0] * (pw.N_OUTCOMES + 1)
    r1 = [0.0] * 5
    ev = 0.0
    if prev is None:
        a1[-1] = o1[-1] = 1.0
        r1[-1] = 1.0
    else:
        a, o, e, rev = prev
        a1[a] = 1.0
        o1[o] = 1.0
        ev = float(e)
        r1[rev if rev is not None else 4] = 1.0
    m = [0.0] * pw.N_ACTIONS
    for a in avail:
        m[a] = 1.0
    return x + a1 + o1 + [ev] + r1 + m + [qfrac]


CASE_IDX = {"a": 0, "b": 1, "c": 2, "d": 3}


def run_batch(model, items, mode, rng=None, need_labels=True):
    """Roll out a batch.  items: list of (cfg, solver, world_seed).  mode: 'sample' | 'greedy' | 'teacher'.
    Returns per-step tensors and per-episode records."""
    B = len(items)
    eps = [pw.Episode(cfg, ws) for cfg, _, ws in items]
    vecs = [cfg.public_vector() for cfg, _, _ in items]
    h = torch.zeros(B, model.gru.hidden_size)
    prev = [None] * B
    prev_state = [None] * B
    prev_act = [None] * B
    steps = []  # per step: dict of tensors over active episodes
    ep_steps = [[] for _ in range(B)]  # per-episode step info (python)
    while True:
        act = [i for i in range(B) if not eps[i].done]
        if not act:
            break
        avails = [eps[i].available() for i in act]
        x = torch.tensor([encode(vecs[i], prev[i], av, eps[i].query / eps[i].cfg.k) for i, av in zip(act, avails)])
        idx = torch.tensor(act)
        hn, z = model.step(x, h[idx])
        h = h.index_copy(0, idx, hn)
        mask = torch.zeros(len(act), pw.N_ACTIONS, dtype=torch.bool)
        for j, av in enumerate(avails):
            mask[j, list(av)] = True
        logits = model.pi(z).masked_fill(~mask, -1e9)
        logp_all = Fn.log_softmax(logits, -1)
        if mode == "sample":
            with torch.no_grad():
                probs = logp_all.exp()
                u = torch.tensor([rng.random() for _ in act]).unsqueeze(1)
                choice = (probs.cumsum(-1) < u).sum(-1).clamp(max=pw.N_ACTIONS - 1)
            chosen = []
            for j, c in enumerate(choice.tolist()):
                if not mask[j, c]:  # numerical edge: fall back to last available
                    c = max(avails[j])
                chosen.append(c)
        elif mode == "greedy":
            chosen = logits.argmax(-1).tolist()
        else:  # teacher: deterministic representative of pi*
            chosen = [items[i][1].pi_star(eps[i].state) for i in act]
        rec = {"idx": idx, "mask": mask, "logp_all": logp_all, "z": z, "chosen": torch.tensor(chosen)}
        if need_labels:
            Qt = torch.zeros(len(act), pw.N_ACTIONS)
            opt = torch.zeros(len(act), pw.N_ACTIONS, dtype=torch.bool)
            stg = torch.zeros(len(act), N_STAGES, dtype=torch.bool)
            dep = torch.zeros(len(act), N_DEP)
            sw = torch.zeros(len(act))
            swm = torch.zeros(len(act))
            vstar = torch.zeros(len(act))
            for j, i in enumerate(act):
                s = items[i][1]
                st = eps[i].state
                q = s.q_values(st)
                for a_, v_ in q.items():
                    Qt[j, a_] = v_
                for a_ in s.opt_set(st):
                    opt[j, a_] = True
                for g in pw.stage_set(s, st):
                    stg[j, g] = True
                dep[j] = torch.tensor(pw.dependency_bits(st))
                vstar[j] = s.value(st)
                if prev_state[i] is not None and prev_act[i] not in pw.TERMINAL:
                    swm[j] = 1.0
                    sw[j] = float(pw.switch_needed(s, prev_state[i], prev_act[i], st))
            rec.update(Q=Qt, opt=opt, stage=stg, dep=dep, sw=sw, swm=swm, vstar=vstar)
        for j, i in enumerate(act):
            ep = eps[i]
            st = ep.state
            a_ = chosen[j]
            nledger = len(ep.ledger)
            succ0 = ep.successes
            r = ep.step(a_)
            reward = pw_reward(ep, nledger, succ0)
            info = {"state": st, "a": a_, "rec": r, "next": ep.state, "reward": reward, "step_row": len(steps),
                    "row": j}
            ep_steps[i].append(info)
            prev[i] = r
            prev_state[i], prev_act[i] = st, a_
        steps.append(rec)
    return eps, steps, ep_steps


def pw_reward(ep, nledger, succ0):
    return ep.cfg.R * (ep.successes - succ0) - sum(x for _, _, x in ep.ledger[nledger:])


def attach_case_labels(items, ep_steps, steps):
    """Case type of each step (the agent's own action) and prev-case targets for the next step."""
    for i, info_list in enumerate(ep_steps):
        s = items[i][1]
        for n, info in enumerate(info_list):
            a, o, e, _ = info["rec"]
            info["case"] = pw.case_type(s, info["state"], a, o, e, info["next"])
    for rec in steps:
        rec["prev_case"] = torch.full((len(rec["idx"]),), -1, dtype=torch.long)
    for i, info_list in enumerate(ep_steps):
        for n in range(1, len(info_list)):
            prev_info, info = info_list[n - 1], info_list[n]
            if prev_info["a"] in pw.TERMINAL:
                continue
            steps[info["step_row"]]["prev_case"][info["row"]] = CASE_IDX[prev_info["case"]]


def losses(model, items, eps, steps, ep_steps, w):
    R = 100.0
    # returns-to-go (scaled by 1/R)
    G = {}
    for i, info_list in enumerate(ep_steps):
        g = 0.0
        for info in reversed(info_list):
            g += info["reward"]
            G[(info["step_row"], info["row"])] = g / R
    out = {}
    pl, vl, il, dl, sl, ql, ent = [], [], [], [], [], [], []
    for t, rec in enumerate(steps):
        n = len(rec["idx"])
        Gt = torch.tensor([G[(t, j)] for j in range(n)])
        z = rec["z"]
        v = model.v(z).squeeze(-1)
        logp_all = rec["logp_all"]
        lp = logp_all.gather(1, rec["chosen"].unsqueeze(1)).squeeze(1)
        adv = (Gt - v).detach()
        pl.append(-(lp * adv))
        vl.append((v - Gt) ** 2)
        p = logp_all.exp()
        ent.append(-(p * logp_all.masked_fill(~rec["mask"], 0.0)).sum(-1))
        # L1: set-valued CE  -log sum_{a in A*} pi(a)
        il.append(-torch.logsumexp(logp_all.masked_fill(~rec["opt"], -1e9), -1))
        # L2: stage set-CE + dependency BCE
        slog = Fn.log_softmax(model.stage(z), -1)
        stage_ce = -torch.logsumexp(slog.masked_fill(~rec["stage"], -1e9), -1)
        dep_bce = Fn.binary_cross_entropy_with_logits(model.dep(z), rec["dep"], reduction="none").mean(-1)
        dl.append(stage_ce + dep_bce)
        # L3: switch BCE (masked) + prev-case CE (masked)
        swl = Fn.binary_cross_entropy_with_logits(model.switch(z).squeeze(-1), rec["sw"], reduction="none") * rec["swm"]
        pc = rec["prev_case"]
        cm = (pc >= 0).float()
        cl = Fn.cross_entropy(model.case(z), pc.clamp(min=0), reduction="none") * cm
        sl.append(swl + cl)
        # L4: Q* regression over available actions (scaled 1/R), Huber
        qd = Fn.smooth_l1_loss(model.q(z), rec["Q"] / R, reduction="none") * rec["mask"].float()
        ql.append(qd.sum(-1) / rec["mask"].float().sum(-1))
    cat = lambda xs: torch.cat(xs).mean()
    out = {"rl": cat(pl) + 0.5 * cat(vl) - 0.01 * cat(ent), "imit": cat(il), "dep": cat(dl), "switch": cat(sl),
           "q": cat(ql), "value_mse": cat(vl).detach(), "entropy": cat(ent).detach()}
    total = sum(w[k] * out[k] for k in ("rl", "imit", "dep", "switch", "q"))
    return total, out


def cmd_train(a):
    torch.set_num_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.process_time()
    pool = load_pool(a.labels, "train")
    t_load = time.process_time() - t_start
    w = RUNG_WEIGHTS[a.rung]
    torch.manual_seed(1000 + a.seed)  # same init across rungs for a given seed
    model = ProbeNet(a.hidden, own_value=a.own_value)
    # B1 parameters (everything except v_own): optimizer and gradient clipping see exactly these, so v_own can
    # change neither the Adam state nor the clip coefficient of the policy/trunk.
    main_params = [p for n, p in model.named_parameters() if not n.startswith("v_own.")]
    opt = torch.optim.Adam(main_params, lr=a.lr)
    own = None
    if a.own_value:
        own = {"opt": torch.optim.Adam(model.v_own.parameters(), lr=a.lr), "rng": random.Random(11_000 + a.seed),
               "worlds": 0, "steps": 0, "cpu_s": 0.0, "log": []}
    data_rng = random.Random(7_000 + a.seed)  # same config/world stream across rungs for a given seed
    act_rng = random.Random(9_000 + a.seed)
    log = []
    world_counter = 0
    for upd in range(a.updates):
        items = []
        for _ in range(a.batch):
            idx, cfg, s = pool[data_rng.randrange(len(pool))]
            # training worlds: a seed range disjoint from every eval split's world seeds (< 5e9)
            items.append((cfg, s, TRAIN_WORLD_BASE + a.seed * 100_000_000 + world_counter))
            world_counter += 1
        eps, steps, ep_steps = run_batch(model, items, "sample", act_rng)
        attach_case_labels(items, ep_steps, steps)
        total, parts = losses(model, items, eps, steps, ep_steps, w)
        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(main_params, 1.0)
        opt.step()
        if own is not None and (upd + 1) % a.own_every == 0:
            own_value_update(model, pool, a, own, upd)
        if upd % a.log_every == 0 or upd == a.updates - 1:
            U = sum(e.utility for e in eps) / len(eps)
            Vs = sum(s.value(pw.initial_state(c)) for c, s, _ in items) / len(items)
            row = {"update": upd, "U": U, "V_star": Vs, "regret": Vs - U,
                   **{k: float(v.detach()) for k, v in parts.items()}, "cpu_s": time.process_time() - t_start}
            log.append(row)
            print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {"rung": a.rung, "seed": a.seed, "weights": w, "updates": a.updates, "batch": a.batch,
            "episodes": a.updates * a.batch, "lr": a.lr, "hidden": a.hidden, "params": n_params(model),
            "in_dim": IN_DIM, "cpu_s_total": time.process_time() - t_start, "cpu_s_label_load": t_load,
            "version": pw.VERSION, "continuation": pw.CONTINUATION, "eps": pw.EPS}
    if own is not None:  # key absent when off (B1 train_meta unchanged)
        meta["own_value"] = {"continuation": OWN_VALUE_CONTINUATION, "every": a.own_every,
                             "episodes_per_collection": a.own_episodes, "collections": own["steps"],
                             "greedy_episodes_total": own["worlds"], "stop_gradient": True,
                             "loss": "MSE(v_own(z.detach()), G_own/R), one Adam step per collection", "lr": a.lr,
                             "world_seed_base": TRAIN_WORLD_BASE + a.seed * 100_000_000 + OWN_WORLD_OFFSET,
                             "config_rng_seed": 11_000 + a.seed, "pool": "train", "cpu_s": own["cpu_s"]}
        (out / "own_value_log.json").write_text(json.dumps(own["log"]))
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "train_log.json").write_text(json.dumps(log))


def own_returns(ep_steps, R=100.0):
    """{(step_row, row): realized return-to-go / R} of the rollout's own continuation."""
    G = {}
    for info_list in ep_steps:
        g = 0.0
        for info in reversed(info_list):
            g += info["reward"]
            G[(info["step_row"], info["row"])] = g / R
    return G


def own_value_update(model, pool, a, own, upd):
    """protocol-B2: collect a.own_episodes free-running GREEDY episodes of the current model on train-pool configs
    (own config RNG and world-seed range; the B1 data/action RNGs are untouched) and take one Adam step of v_own
    toward their realized return-to-go.  Stop-gradient: v_own sees z.detach(), so the trunk/policy never receive
    its gradient.  Labels are not computed (need_labels=False): the target is the model's own return only."""
    t0 = time.process_time()
    items = []
    for _ in range(a.own_episodes):
        idx, cfg, s = pool[own["rng"].randrange(len(pool))]
        items.append((cfg, s, TRAIN_WORLD_BASE + a.seed * 100_000_000 + OWN_WORLD_OFFSET + own["worlds"]))
        own["worlds"] += 1
    with torch.no_grad():
        eps, steps, ep_steps = run_batch(model, items, "greedy", need_labels=False)
    G = own_returns(ep_steps)
    preds, targets = [], []
    for t, rec in enumerate(steps):
        preds.append(model.v_own(rec["z"].detach()).squeeze(-1))
        targets.append(torch.tensor([G[(t, j)] for j in range(len(rec["idx"]))]))
    loss = ((torch.cat(preds) - torch.cat(targets)) ** 2).mean()
    own["opt"].zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.v_own.parameters(), 1.0)
    own["opt"].step()
    own["steps"] += 1
    own["cpu_s"] += time.process_time() - t0
    if upd % a.log_every < a.own_every or upd == a.updates - 1:
        U = sum(e.utility for e in eps) / len(eps)
        Vs = sum(s.value(pw.initial_state(c)) for c, s, _ in items) / len(items)
        row = {"update": upd, "own_greedy_U": U, "own_greedy_regret": Vs - U, "v_own_mse": float(loss.detach()),
               "cpu_s_own": own["cpu_s"]}
        own["log"].append(row)
        print(json.dumps(row), flush=True)


# ------------------------------------------------------------------------------------------------ eval

def episode_metrics(items, eps, ep_steps, steps=None, model=None):
    rows = []
    for i, ep in enumerate(eps):
        cfg, s, _ = items[i]
        s0 = pw.initial_state(cfg)
        gap = 0.0
        cases = {"a": 0, "b": 0, "c": 0, "d": 0}
        sw = {f"{j}_{c}": 0 for j in ("just", "unjust") for c in "abcd"}
        built = 0
        prev = None
        first = {"first_probe": None, "probe_eps_opt": None, "probe_unique_opt": None}
        for info in ep_steps[i]:
            st, a = info["state"], info["a"]
            q = s.q_values(st)
            if first["first_probe"] is None:  # B-H4 (protocol-B1): first decision of the episode
                v0 = s.value(st)
                opt = {b for b, qb in q.items() if qb >= v0 - pw.EPS}
                first = {"first_probe": int(a == pw.A_PROBE), "probe_eps_opt": int(pw.A_PROBE in opt),
                         "probe_unique_opt": int(opt == {pw.A_PROBE})}
            gap += s.value(st) - q[a]
            case = info.get("case")
            if case is None:
                ao = info["rec"]
                case = pw.case_type(s, st, a, ao[1], ao[2], info["next"])
                info["case"] = case
            cases[case] += 1
            if a == pw.A_BUILD:
                built = 1
            if prev is not None and prev["a"] not in pw.TERMINAL:
                if pw.ACTION_STAGE[a] != pw.ACTION_STAGE[prev["a"]]:
                    j = "just" if q[a] >= s.value(st) - pw.EPS else "unjust"
                    sw[f"{j}_{prev['case']}"] += 1
            prev = info
        rows.append({"cfg_k": cfg.k, "rho": cfg.rho(), "rho_eff": _rho_eff(cfg), "flags": list(cfg.flags), "U": ep.utility,
                     "V_star": s.value(s0), "gap_regret": gap, "success": ep.successes / cfg.k,
                     "cost": ep.total_cost, "wrong": ep.wrong, "built": built, "steps": len(ep_steps[i]),
                     "cases": cases, "switches": sw, **first})
    return rows


_RHO_EFF: dict = {}


def _rho_eff(cfg):
    r = _RHO_EFF.get(cfg)
    if r is None:
        r = _RHO_EFF[cfg] = pw.effective_rho(cfg)
    return r


def value_calibration(model_rows):
    """(pred, target) pairs -> binned reliability (10 quantile bins) and MAE, in units of R."""
    if not model_rows:
        return {}
    pairs = sorted(model_rows)
    nb = 10
    bins = []
    for b in range(nb):
        chunk = pairs[b * len(pairs) // nb:(b + 1) * len(pairs) // nb]
        if not chunk:
            continue
        mp = sum(p for p, _ in chunk) / len(chunk)
        mt = sum(t for _, t in chunk) / len(chunk)
        bins.append({"n": len(chunk), "pred": mp, "target": mt})
    ece = sum(b["n"] * abs(b["pred"] - b["target"]) for b in bins) / len(pairs)
    mae = sum(abs(p - t) for p, t in pairs) / len(pairs)
    return {"reliability_error": ece, "mae": mae, "bins": bins}


def summarize_rows(rows):
    n = len(rows)
    if n == 0:
        return {}
    mean = lambda f: sum(f(r) for r in rows) / n
    def se(f):
        m = mean(f)
        return math.sqrt(sum((f(r) - m) ** 2 for r in rows) / max(n - 1, 1) / n)
    out = {"n": n, "U": mean(lambda r: r["U"]), "V_star": mean(lambda r: r["V_star"]),
           "regret": mean(lambda r: r["V_star"] - r["U"]), "regret_se": se(lambda r: r["V_star"] - r["U"]),
           "gap_regret": mean(lambda r: r["gap_regret"]), "gap_regret_se": se(lambda r: r["gap_regret"]),
           "success": mean(lambda r: r["success"]), "cost": mean(lambda r: r["cost"]),
           "wrong_per_ep": mean(lambda r: r["wrong"]), "build_rate": mean(lambda r: r["built"]),
           "steps": mean(lambda r: r["steps"])}
    def cond_rate(sel):
        sub = [r for r in rows if r.get("first_probe") is not None and sel(r)]
        return (sum(r["first_probe"] for r in sub) / len(sub)) if sub else None, len(sub)
    out["first_probe_rate_when_not_opt"], out["n_probe_not_opt"] = cond_rate(lambda r: not r["probe_eps_opt"])
    out["first_probe_rate_when_unique_opt"], out["n_probe_unique_opt"] = cond_rate(lambda r: r["probe_unique_opt"])
    for c in "abcd":
        out[f"case_{c}"] = mean(lambda r: r["cases"][c])
    for key in rows[0]["switches"]:
        out[f"switch_{key}"] = mean(lambda r: r["switches"][key])
    return out


RHO_BINS = (0.0, 0.5, 1.0, 2.0, 4.0, 1e9)


def rho_curve(rows, ref_rows):
    """Structure-build rate vs rho_k bin (declared and effective) and k (agent vs pi* on the same worlds)."""
    out = {}
    for key in ("rho", "rho_eff"):
        curve = []
        for lo, hi in zip(RHO_BINS[:-1], RHO_BINS[1:]):
            sel = [j for j, r in enumerate(rows) if lo <= r[key] < hi]
            if sel:
                curve.append({"lo": lo, "hi": hi, "n": len(sel),
                              "build_rate": sum(rows[j]["built"] for j in sel) / len(sel),
                              "build_rate_pi_star": sum(ref_rows[j]["built"] for j in sel) / len(sel),
                              "gap_regret": sum(rows[j]["gap_regret"] for j in sel) / len(sel)})
        out["by_" + key] = curve
    by_k = []
    for k in pw.K_VALUES:
        sel = [j for j, r in enumerate(rows) if r["cfg_k"] == k]
        if sel:
            by_k.append({"k": k, "n": len(sel), "build_rate": sum(rows[j]["built"] for j in sel) / len(sel),
                         "build_rate_pi_star": sum(ref_rows[j]["built"] for j in sel) / len(sel)})
    out["by_k"] = by_k
    return out


def eval_items(pool, split, worlds):
    return [(cfg, s, pw.world_seed(split, idx, 500 + r)) for idx, cfg, s in pool for r in range(worlds)]


def reference_policy_eval(items, name):
    eps, ep_steps = [], []
    for cfg, s, ws in items:
        ep = pw.Episode(cfg, ws)
        infos = []
        while not ep.done:
            st = ep.state
            a = REFERENCE_POLICIES[name](s, st, ep.available())
            n0, s0 = len(ep.ledger), ep.successes
            r = ep.step(a)
            infos.append({"state": st, "a": a, "rec": r, "next": ep.state, "reward": pw_reward(ep, n0, s0)})
        eps.append(ep)
        ep_steps.append(infos)
    return eps, ep_steps


def _ref_exact(s, st, av):
    """Fixed rule: exact b2 then commit (or commit_infeasible)."""
    if pw.A_COMMIT in av and st[1][4]:
        return pw.A_COMMIT
    if pw.A_B2 in av:
        return pw.A_B2
    return pw.A_COMMIT_INF if st[1][0][pw.TX] > 0.5 else pw.A_ABSTAIN


def _ref_probe_first(s, st, av):
    """Fixed rule: probe first; if it reports solved commit, else exact b2 then commit."""
    if pw.A_COMMIT in av and st[1][4]:
        return pw.A_COMMIT
    if pw.A_PROBE in av and not st[1][5]:
        return pw.A_PROBE
    return _ref_exact(s, st, av)


REFERENCE_POLICIES = {"pi_star": lambda s, st, av: s.pi_star(st), "fixed_exact_b2": _ref_exact,
                      "fixed_probe_first": _ref_probe_first}


@torch.no_grad()
def model_eval(model, items, mode, chunk=256):
    eps_all, steps_rows, ep_steps_all, calib_v, calib_vstar, calib_q = [], [], [], [], [], []
    has_own = hasattr(model, "v_own")
    b2 = {"v_own_own": [], "v_own_vstar": [], "vstar_own": [], "commit": [], "episode": []}
    tf = {"n": 0, "argmax_in_opt": 0, "q_gap": 0.0}
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        eps, steps, ep_steps = run_batch(model, part, mode, need_labels=True)
        # value head: predicted return-to-go vs realized (own policy) and vs V* (pi*)
        heads = []
        for rec in steps:
            logits = model.pi(rec["z"]).masked_fill(~rec["mask"], -1e9)
            heads.append((model.v(rec["z"]).squeeze(-1), model.q(rec["z"]), logits.argmax(-1)))
        vown = [model.v_own(rec["z"]).squeeze(-1) for rec in steps] if has_own else None
        for i, info_list in enumerate(ep_steps):
            if has_own:
                b2_episode_records(b2, info_list, steps, heads, vown)
            g = 0.0
            for info in reversed(info_list):
                g += info["reward"]
                t, j = info["step_row"], info["row"]
                rec = steps[t]
                vhat, qhat, am = heads[t]
                calib_v.append((float(vhat[j]), g / 100.0))
                calib_vstar.append((float(vhat[j]), float(rec["vstar"][j]) / 100.0))
                a = info["a"]
                calib_q.append((float(qhat[j, a]), float(rec["Q"][j, a]) / 100.0))
                if has_own:
                    b2["v_own_own"].append((float(vown[t][j]), g / 100.0))
                    b2["v_own_vstar"].append((float(vown[t][j]), float(rec["vstar"][j]) / 100.0))
                    b2["vstar_own"].append((float(rec["vstar"][j]) / 100.0, g / 100.0))
                if mode == "teacher":
                    amj = int(am[j])
                    tf["n"] += 1
                    tf["argmax_in_opt"] += int(bool(rec["opt"][j, amj]))
                    tf["q_gap"] += float(rec["vstar"][j] - rec["Q"][j, amj])
        eps_all += eps
        ep_steps_all += ep_steps
    rows = episode_metrics(items, eps_all, ep_steps_all)
    res = {"summary": summarize_rows(rows),
           "value_calibration_own_return": value_calibration(calib_v),
           "value_vs_vstar": value_calibration(calib_vstar),
           "q_head_vs_qstar_taken": value_calibration(calib_q)}
    if has_own:  # protocol-B2 keys; absent for B1 models (eval.json unchanged)
        res["v_own_calibration_own_return"] = value_calibration(b2["v_own_own"])
        res["v_own_vs_vstar"] = value_calibration(b2["v_own_vstar"])
        res["vstar_as_predictor_own_return"] = value_calibration(b2["vstar_own"])  # noise-floor reference
        res["failure_prediction"] = failure_prediction(b2)
    if mode == "teacher":
        res["teacher_forced"] = {"n_steps": tf["n"], "argmax_in_opt_rate": tf["argmax_in_opt"] / max(tf["n"], 1),
                                 "mean_q_gap": tf["q_gap"] / max(tf["n"], 1)}
    return res, rows


B2_PREDICTORS = ("v_own", "v", "q_head_taken", "qstar_taken", "vstar")


def b2_episode_records(b2, info_list, steps, heads, vown):
    """Failure-prediction records (protocol-B2).  Commit level: every commit / commit_infeasible the model takes;
    positive = wrong (outcome O_WRONG); predictors read at the pre-commit state.  Episode level: predictors at the
    episode's first step; positive = the episode has >= 1 wrong commit."""
    def preds(info):
        t, j, a = info["step_row"], info["row"], info["a"]
        rec, (vhat, qhat, _) = steps[t], heads[t]
        return {"v_own": float(vown[t][j]), "v": float(vhat[j]), "q_head_taken": float(qhat[j, a]),
                "qstar_taken": float(rec["Q"][j, a]) / 100.0, "vstar": float(rec["vstar"][j]) / 100.0}
    wrong_any = 0
    for info in info_list:
        if info["a"] in (pw.A_COMMIT, pw.A_COMMIT_INF):
            wrong = int(info["rec"][1] == pw.O_WRONG)
            wrong_any |= wrong
            b2["commit"].append({"wrong": wrong, **preds(info)})
    if info_list:
        b2["episode"].append({"wrong": wrong_any, **preds(info_list[0])})


def auroc(pos_scores, neg_scores):
    """P(score_pos > score_neg) + .5 P(tie); None if a class is empty."""
    if not pos_scores or not neg_scores:
        return None
    neg = sorted(neg_scores)
    tot = 0.0
    for x in pos_scores:
        lo, hi = bisect.bisect_left(neg, x), bisect.bisect_right(neg, x)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos_scores) * len(neg))


def failure_prediction(b2):
    """AUROC of LOW predicted value for wrong commits (score = -prediction).  Raw records are returned under
    '_records' (cmd_eval moves them to failure_records.json for the scorer's pooled statistics)."""
    out = {"_records": {}}
    for level in ("commit", "episode"):
        recs = b2[level]
        pos = [r for r in recs if r["wrong"]]
        neg = [r for r in recs if not r["wrong"]]
        out[level] = {"n_pos": len(pos), "n_neg": len(neg),
                      "auroc": {k: auroc([-r[k] for r in pos], [-r[k] for r in neg]) for k in B2_PREDICTORS}}
        out["_records"][level] = recs
    return out


def cmd_eval(a):
    torch.set_num_threads(1)
    run = Path(a.run)
    meta = json.loads((run / "train_meta.json").read_text())
    model = ProbeNet(meta["hidden"], own_value="own_value" in meta)
    model.load_state_dict(torch.load(run / "model.pt"))
    model.eval()
    t0 = time.process_time()
    result = {"rung": meta["rung"], "seed": meta["seed"], "splits": {}}
    failure_records = {}
    for split in a.splits:
        pool = load_pool(a.labels, split)
        items = eval_items(pool, split, a.worlds)
        ref_eps, ref_steps = reference_policy_eval(items, "pi_star")
        ref_rows = episode_metrics(items, ref_eps, ref_steps)
        free, rows = model_eval(model, items, "greedy")
        free["rho_curve"] = rho_curve(rows, ref_rows)
        teach, _ = model_eval(model, items, "teacher")
        by_combo = {}
        for r in rows:
            key = "+".join(n for n, f in zip(pw.FLAG_NAMES, r["flags"]) if f) or "none"
            by_combo.setdefault(key, []).append(r)
        free["by_condition"] = {k: summarize_rows(v) for k, v in by_combo.items()}
        for mode, res in (("free_running_greedy", free), ("teacher_forced", teach)):
            if "failure_prediction" in res:
                failure_records.setdefault(split, {})[mode] = res["failure_prediction"].pop("_records")
        result["splits"][split] = {"free_running_greedy": free, "teacher_forced": teach}
        print(split, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                 for k, v in free["summary"].items()}), flush=True)
    result["cpu_s"] = time.process_time() - t0
    (run / "eval.json").write_text(json.dumps(result, indent=1))
    if failure_records:  # protocol-B2 only
        (run / "failure_records.json").write_text(json.dumps(failure_records))


def cmd_references(a):
    """Reference policies (pi*, fixed rules) on the same eval worlds."""
    res = {}
    for split in a.splits:
        pool = load_pool(a.labels, split)
        items = eval_items(pool, split, a.worlds)
        res[split] = {}
        for name in REFERENCE_POLICIES:
            eps, steps = reference_policy_eval(items, name)
            rows = episode_metrics(items, eps, steps)
            res[split][name] = summarize_rows(rows)
            if name == "pi_star":
                res[split][name]["rho_curve"] = rho_curve(rows, rows)
        print(split, {n: round(v["regret"], 2) for n, v in res[split].items()}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_summarize(a):
    rows = {}
    for d in a.runs:
        p = Path(d) / "eval.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text())
        tm = json.loads((Path(d) / "train_meta.json").read_text())
        rows.setdefault(e["rung"], []).append((e, tm))
    table = {}
    keys = ("regret", "gap_regret", "success", "cost", "U", "V_star", "build_rate", "case_a", "case_b", "case_c",
            "switch_just_b", "switch_unjust_b", "switch_just_c", "switch_unjust_c", "switch_unjust_d")
    for rung, lst in sorted(rows.items()):
        table[rung] = {"seeds": [tm["seed"] for _, tm in lst], "params": lst[0][1]["params"],
                       "train_cpu_s": [tm["cpu_s_total"] for _, tm in lst], "splits": {}}
        for split in lst[0][0]["splits"]:
            per = [e["splits"][split] for e, _ in lst]
            ent = {}
            for key in keys:
                vals = [p["free_running_greedy"]["summary"][key] for p in per]
                ent[key] = {"mean": sum(vals) / len(vals), "per_seed": vals}
            tfv = [p["teacher_forced"]["teacher_forced"]["argmax_in_opt_rate"] for p in per]
            ent["tf_argmax_in_opt"] = {"mean": sum(tfv) / len(tfv), "per_seed": tfv}
            cal = [p["free_running_greedy"]["value_calibration_own_return"]["reliability_error"] for p in per]
            ent["value_reliability_error"] = {"mean": sum(cal) / len(cal), "per_seed": cal}
            curves = [p["free_running_greedy"]["rho_curve"]["by_rho_eff"] for p in per]
            ent["rho_eff_curve"] = [{"lo": c0["lo"], "hi": c0["hi"], "n": c0["n"],
                                     "build_rate": sum(c[j]["build_rate"] for c in curves) / len(curves),
                                     "build_rate_pi_star": c0["build_rate_pi_star"]}
                                    for j, c0 in enumerate(curves[0]) if all(len(c) == len(curves[0]) for c in curves)]
            table[rung]["splits"][split] = ent
    Path(a.out).write_text(json.dumps(table, indent=1))
    for rung, t in table.items():
        print(rung, {s: round(v["gap_regret"]["mean"], 2) for s, v in t["splits"].items()})


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("labels")
    s.add_argument("--out", required=True)
    s.add_argument("--n-train", type=int, default=384)
    s.add_argument("--n-eval", type=int, default=128)
    s = sub.add_parser("train")
    s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rung", choices=RUNGS, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--updates", type=int, default=4000)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--hidden", type=int, default=HIDDEN)
    s.add_argument("--log-every", type=int, default=100)
    s.add_argument("--own-value", action="store_true", help="protocol-B2 v_own head (default off: B1 identical)")
    s.add_argument("--own-every", type=int, default=1, help="collect own greedy rollouts every N updates")
    s.add_argument("--own-episodes", type=int, default=64, help="greedy episodes per collection")
    s = sub.add_parser("eval")
    s.add_argument("--labels", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--worlds", type=int, default=4)
    s.add_argument("--splits", nargs="+", default=list(EVAL_SPLITS))
    s = sub.add_parser("references")
    s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--worlds", type=int, default=4)
    s.add_argument("--splits", nargs="+", default=list(EVAL_SPLITS))
    s = sub.add_parser("summarize")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    {"labels": cmd_labels, "train": cmd_train, "eval": cmd_eval, "references": cmd_references,
     "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
