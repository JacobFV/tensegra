#!/usr/bin/env python3
"""P2a independent auditor: torch-side checks (CPU only, read-only on results).

Written by the independent P2a auditor WITHOUT reading campaign03_p2a_collapse_audit.py.
It reuses the training code's public encoding (public_frame/collate), world generator,
executor and CandidatePolicy; the rollout loops, KL, advantage and gradient
decompositions below are the auditor's own.

Subcommands (run from a source snapshot with PYTHONPATH=src):
  weights   compare model/optimizer tensors of C0 vs P1 X1-rl-r2 (and C1) per attempt
  rollouts  greedy + sampled rollouts on Part D probe worlds for a list of checkpoints,
            KL(boot || ck) and argmax disagreement on boot-greedy and own-greedy states
  grads     actor / critic / entropy / KL gradient decomposition on one sampled batch,
            plus the AdamW update direction with the checkpoint's saved optimizer state
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from functools import partial

import torch

try:
    from tensegra import campaign02_training as T
    from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld, action_key
except ImportError:  # pre-rename snapshot
    from topoformer import campaign02_training as T
    from topoformer.campaign02_policy import CandidatePolicy, PolicyConfig
    from topoformer.campaign02_protocol import BoundedSolver
    from topoformer.campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld, action_key

R = "/home/brand/tensegra-campaign03/results/"
WORLD = {"categories": 3, "choices": 3, "locations": 7, "slots": 6, "compute_price": 0.0001,
         "event_trigger": "progress", "p_event": 0.5, "event_kinds": ["edge_closed", "capacity_reduced", "slot_closed"]}
NAMESPACE = "e03p2a-partd-probe"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_path(run, attempt):
    r, s = divmod(attempt, 6)
    return f"{R}{run}/checkpoints/round-{r}-slot-{s}-attempt-{attempt}.pt"


BOOT = R + "p1-boot-x1-r2/checkpoints/round-0-slot-5-attempt-5.pt"


def load(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = T.normalized_policy_config(saved["policy_config"])
    m = CandidatePolicy(PolicyConfig(**cfg))
    m.load_state_dict(saved["model"], strict=True)
    m.eval()
    return m, saved


def probe_worlds(n_per_cond):
    out = []
    for i in range(n_per_cond):
        for base, foreign in ((1990000000, 0), (1990100000, 2)):
            out.append((base + i, foreign))
    return out


def make_env(seed, foreign, solver):
    spec = generate_depworld(seed, **{**WORLD, "foreign_records": foreign})
    return DepWorkshop(spec, executor=partial(depworld_executor, execute_call=solver.execute),
                       address_seed=T.independent_address_seed(seed, NAMESPACE))


def decision_state(d):
    return json.dumps({k: d.get(k) for k in ("problems", "pending", "pending_assignment", "selected", "selection_id",
                                             "assignment", "assignment_id", "finish_time", "verified", "position")}
                      | {"retrieved": sorted((d.get("retrieved") or {}).keys())}, sort_keys=True)


def rollout(model, worlds, solver, *, gen=None, max_steps=96, batch=32):
    """Own batched rollout loop. gen=None -> argmax; else torch.Generator sampling.
    Returns per-episode dicts with outcome summary and the visited public observations."""
    eps = []
    for b0 in range(0, len(worlds), batch):
        envs = [make_env(s, f, solver) for s, f in worlds[b0:b0 + batch]]
        obs = [e.observe() for e in envs]
        rec = [{"seed": worlds[b0 + i][0], "states": [], "keys": [], "status": [], "pre": []} for i in range(len(envs))]
        for _ in range(max_steps):
            act = [i for i, o in enumerate(obs) if not o.done]
            if not act:
                break
            frames = []
            for i in act:
                actions, o, feats = T.public_frame(obs[i], model.config.feature_version)
                frames.append((actions, T.Frame(o, feats, 0)))
            bt = T.collate([f for _, f in frames])
            with torch.no_grad():
                logits, _, _ = model.score(bt[0], bt[1], None, bt[2])
            if gen is None:
                ch = logits.argmax(-1).tolist()
            else:
                ch = torch.multinomial(logits.softmax(-1), 1, generator=gen).squeeze(1).tolist()
            for r, i in enumerate(act):
                a = frames[r][0][ch[r]]
                rec[i]["states"].append(obs[i])
                rec[i]["keys"].append(action_key(a))
                rec[i]["pre"].append(decision_state(obs[i].to_dict()))
                envs[i].charge_compute(1.0)
                obs[i] = envs[i].step(a)
                rec[i]["status"].append(obs[i].feedback.get("status"))
                rec[i]["kind"] = rec[i].get("kind", []) + [a.kind]
        for i, e in enumerate(envs):
            o = e.evaluate()
            rec[i].update(success=bool(o["verified_success"]), utility=o["utility"], steps=o["steps"],
                          work=o["work_units"], post_last=decision_state(obs[i].to_dict()))
        eps.extend(rec)
    return eps


def no_progress(ep, window=6):
    """Idempotent repeats and short cycles (same reading as the screening reconstruction)."""
    pre, keys, st, kinds = ep["pre"], ep["keys"], ep["status"], ep["kind"]
    n = len(pre)
    post = pre[1:] + [ep["post_last"]]
    seen_inspect, last_break, last_key, npc, idem = set(), -1, None, 0, 0
    nev = [len(s.to_dict().get("events") or []) for s in ep["states"]]
    for k in range(n):
        brk = kinds[k] == "call"
        if kinds[k] == "inspect":
            if keys[k] not in seen_inspect:
                brk = True
                seen_inspect.add(keys[k])
        if k + 1 < n and nev[k + 1] > nev[k]:
            brk = True
        if brk:
            last_break = k
        if st[k] == "success":
            is_idem = last_key == keys[k] and post[k] == pre[k]
            is_cyc = (not brk) and any(post[k] == pre[m] for m in range(max(k - window + 1, last_break + 1, 0), k + 1))
            npc += is_idem or is_cyc
            idem += is_idem
            last_key = keys[k]
    return npc, idem, n


def kl_on_states(ref, model, states, fv):
    """Mean KL(ref || model) over given public observations, plus argmax disagreement."""
    kls, dis = [], 0
    for b0 in range(0, len(states), 64):
        chunk = states[b0:b0 + 64]
        frames = [T.Frame(*T.public_frame(s, fv)[1:], 0) for s in chunk]
        bt = T.collate(frames)
        with torch.no_grad():
            lr, _, _ = ref.score(bt[0], bt[1], None, bt[2])
            lm, _, _ = model.score(bt[0], bt[1], None, bt[2])
        v = bt[2]
        pr = lr.log_softmax(-1).masked_fill(~v, 0.0)
        pm = lm.log_softmax(-1).masked_fill(~v, 0.0)
        kls += (pr.exp() * (pr - pm)).sum(-1).tolist()
        dis += int((lr.argmax(-1) != lm.argmax(-1)).sum())
    return {"mean": sum(kls) / len(kls), "argmax_disagreement": dis / len(kls), "states": len(kls)}


def summarize(eps):
    n = len(eps)
    np_, idem, dec = 0, 0, 0
    kinds = {}
    for e in eps:
        a, b, c = no_progress(e)
        np_ += a; idem += b; dec += c
        for k in e["kind"]:
            kinds[k] = kinds.get(k, 0) + 1
    return {"episodes": n, "success": sum(e["success"] for e in eps) / n, "utility": sum(e["utility"] for e in eps) / n,
            "steps_to_cap": sum((e["steps"] >= 96) and not e["success"] for e in eps) / n,
            "no_progress_rate": np_ / dec, "idempotent_rate": idem / dec, "decisions_per_episode": dec / n,
            "work_per_success": sum(e["work"] for e in eps) / max(1, sum(e["success"] for e in eps)),
            "action_mix": {k: v / dec for k, v in sorted(kinds.items())},
            "per_world_success": {str(e["seed"]): e["success"] for e in eps}}


def cmd_weights(a):
    out = {}
    for att in a.attempts:
        pa, pb, pc = ckpt_path("p2a-c0-x1-r2", att), ckpt_path("p1-rl-x1-r2", att), ckpt_path("p2a-c1-x1-r2", att)
        A = torch.load(pa, map_location="cpu", weights_only=False)
        B = torch.load(pb, map_location="cpu", weights_only=False)
        C = torch.load(pc, map_location="cpu", weights_only=False)
        same = all(torch.equal(A["model"][k], B["model"][k]) for k in A["model"]) and A["model"].keys() == B["model"].keys()
        maxdiff_c1 = max(float((A["model"][k].float() - C["model"][k].float()).abs().max()) for k in A["model"])

        def opt_equal(x, y):
            sx, sy = x["optimizer"]["state"], y["optimizer"]["state"]
            if sx.keys() != sy.keys():
                return False
            return all(torch.equal(torch.as_tensor(sx[i][k]), torch.as_tensor(sy[i][k])) for i in sx for k in sx[i])
        def deq(x, y):
            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                return isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape and torch.equal(x, y)
            if isinstance(x, dict) and isinstance(y, dict):
                return x.keys() == y.keys() and all(deq(x[k], y[k]) for k in x)
            if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                return len(x) == len(y) and all(deq(u, v) for u, v in zip(x, y))
            return x == y
        other = {k: deq(A[k], B[k]) for k in A if k not in ("model", "optimizer") and k in B}
        other["_keys_equal"] = A.keys() == B.keys()
        out[att] = {"c0_sha": sha(pa)[:16], "p1_sha": sha(pb)[:16], "model_tensors_identical": same,
                    "optimizer_state_identical": opt_equal(A, B), "n_tensors": len(A["model"]),
                    "c0_vs_c1_max_abs_weight_diff": maxdiff_c1,
                    "top_level_keys": sorted(A.keys()),
                    "differing_non_tensor_fields": [k for k, v in other.items() if v is not True]}
        print(att, out[att], flush=True)
    json.dump(out, open(a.output, "w"), indent=1)


def cmd_rollouts(a):
    torch.set_num_threads(1)
    worlds = probe_worlds(a.worlds_per_cond)
    boot, _ = load(BOOT)
    res = {"worlds": worlds, "subjects": {}}
    with BoundedSolver() as solver:
        t0 = time.process_time()
        boot_eps = rollout(boot, worlds, solver)
        boot_states = [s for e in boot_eps for s in e["states"]]
        res["boot_greedy"] = summarize(boot_eps)
        print("boot", {k: v for k, v in res["boot_greedy"].items() if k != "per_world_success"}, time.process_time() - t0, flush=True)
        for spec in a.subjects:
            name, run, att = spec.split(":")
            path = BOOT if att == "boot" else ckpt_path(run, int(att))
            m, _ = load(path)
            t0 = time.process_time()
            g = rollout(m, worlds, solver)
            rec = {"path": path, "sha256": sha(path), "greedy": summarize(g)}
            if a.samples:
                samp = []
                for s in range(a.samples):
                    gen = torch.Generator().manual_seed(777000 + s)
                    samp += rollout(m, worlds, solver, gen=gen)
                rec["sampled"] = summarize(samp)
            own = [s for e in g for s in e["states"]]
            rec["kl_boot_to_ck"] = {"boot_greedy_states": kl_on_states(boot, m, boot_states, m.config.feature_version),
                                    "own_greedy_states": kl_on_states(boot, m, own, m.config.feature_version)}
            rec["cpu"] = time.process_time() - t0
            res["subjects"][name] = rec
            print(name, {k: (v if k in ("kl_boot_to_ck", "cpu") else {kk: vv for kk, vv in v.items() if kk not in ("per_world_success", "action_mix")})
                         for k, v in rec.items() if k not in ("path", "sha256")}, flush=True)
    json.dump(res, open(a.output, "w"), indent=1, default=str)


def cmd_grads(a):
    torch.set_num_threads(1)
    worlds = probe_worlds(4)  # 8 worlds, interleaved f0/f2
    out = {}
    with BoundedSolver() as solver:
        for spec in a.subjects:
            name, path, refpath = spec.split("|")
            torch.manual_seed(20260926)
            m, saved = load(path)
            m.train()
            ref, _ = load(refpath)
            envs = [make_env(s, f, solver) for s, f in worlds]
            results, terms, kls = T.batched_on_policy(m, envs, max_steps=96, sample=True, reference=ref)
            # own decomposition of the registered objective
            flat_adv, per = [], []
            for ep in terms:
                g, part = 0.0, []
                for logp, value, ent, r in reversed(ep):
                    g += r
                    part.append((logp, value, ent, g))
                per.append(list(reversed(part)))
            raw = [g - float(v.detach()) for ep in per for (_, v, _, g) in ep]
            mu = sum(raw) / len(raw)
            sd = math.sqrt(sum((x - mu) ** 2 for x in raw) / len(raw)) + 1e-6
            logps = torch.stack([lp for ep in per for (lp, _, _, _) in ep])
            vals = torch.stack([v for ep in per for (_, v, _, _) in ep])
            ents = torch.stack([e for ep in per for (_, _, e, _) in ep])
            rets = torch.tensor([g for ep in per for (_, _, _, g) in ep])
            adv = torch.tensor([(x - mu) / sd for x in raw])
            kl = torch.stack([k for ep in kls for k in ep]).mean()
            parts = {"actor": -(logps * adv).mean(), "critic": 0.5 * (rets - vals).square().mean(),
                     "entropy": -0.003 * ents.mean(), "kl": 0.3 * kl}
            groups = {"shared": [p for n, p in m.named_parameters() if n.startswith(("observation", "context"))],
                      "actor_head": [p for n, p in m.named_parameters() if n.startswith(("candidate", "scorer"))],
                      "critic_head": [p for n, p in m.named_parameters() if n.startswith("value")]}
            params = [p for g in groups.values() for p in g]
            assert len(params) == len(list(m.parameters()))
            grads = {}
            for k, loss in parts.items():
                gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                grads[k] = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gs)]
            ns = len(groups["shared"])
            na = len(groups["actor_head"])
            sl = {"shared": slice(0, ns), "actor_head": slice(ns, ns + na), "critic_head": slice(ns + na, len(params))}

            def vec(gl, s):
                return torch.cat([g.reshape(-1) for g in gl[s]])
            norms = {k: {s: float(vec(g, sl[s]).norm()) for s in sl} | {"all": float(torch.cat([x.reshape(-1) for x in g]).norm())}
                     for k, g in grads.items()}
            total = [sum(gs) for gs in zip(*grads.values())]
            tot_norm = float(torch.cat([x.reshape(-1) for x in total]).norm())
            cos = {}
            keys = list(grads)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    x, y = vec(grads[keys[i]], sl["shared"]), vec(grads[keys[j]], sl["shared"])
                    cos[f"{keys[i]}|{keys[j]}"] = float(x @ y / (x.norm() * y.norm() + 1e-30))
            # AdamW direction with the saved optimizer state: first-step update for each gradient
            st = saved["optimizer"]["state"]
            grp = saved["optimizer"]["param_groups"][0]
            b1, b2 = grp["betas"]
            eps = grp["eps"]
            order = [n for n, _ in m.named_parameters()]
            name_of = {id(p): n for n, p in m.named_parameters()}
            idx = {n: i for i, n in enumerate(order)}

            def adam_dir(gl, clip=True):
                scale = 1.0
                if clip:
                    nrm = float(torch.cat([x.reshape(-1) for x in gl]).norm())
                    scale = min(1.0, 1.0 / (nrm + 1e-6))
                ds = []
                for p, g in zip(params, gl):
                    s = st.get(idx[name_of[id(p)]])
                    g = g * scale
                    if s is None:
                        ds.append(g.sign()); continue
                    step = float(s["step"]) + 1
                    mm = b1 * s["exp_avg"] + (1 - b1) * g
                    vv = b2 * s["exp_avg_sq"] + (1 - b2) * g * g
                    ds.append((mm / (1 - b1 ** step)) / ((vv / (1 - b2 ** step)).sqrt() + eps))
                return ds
            # fresh-gradient contribution only (momentum removed): compare with zero-gradient step
            base = adam_dir([torch.zeros_like(p) for p in params], clip=False)
            d_tot = [x - y for x, y in zip(adam_dir(total), base)]
            d_act = [x - y for x, y in zip(adam_dir(grads["actor"]), base)]
            d_cri = [x - y for x, y in zip(adam_dir(grads["critic"]), base)]
            full_tot = adam_dir(total)

            def c(u, v, s):
                x, y = vec(u, sl[s]), vec(v, sl[s])
                return float(x @ y / (x.norm() * y.norm() + 1e-30))
            out[name] = {"path": path, "sha256": sha(path), "reference": refpath, "decisions": len(raw),
                         "parts": {k: float(v) for k, v in parts.items()}, "norms": norms, "total_norm": tot_norm,
                         "would_clip": tot_norm > 1.0, "shared_cosine": cos,
                         "shared_critic_over_actor": norms["critic"]["shared"] / max(norms["actor"]["shared"], 1e-30),
                         "adam": {"has_state": bool(st), "state_step": float(next(iter(st.values()))["step"]) if st else None,
                                  "cos_shared_fresh_total_vs_actor_only": c(d_tot, d_act, "shared"),
                                  "cos_shared_fresh_total_vs_critic_only": c(d_tot, d_cri, "shared"),
                                  "cos_shared_full_step_vs_momentum_only": c(full_tot, base, "shared"),
                                  "fresh_norm_shared": float(vec(d_tot, sl["shared"]).norm()),
                                  "momentum_norm_shared": float(vec(base, sl["shared"]).norm())}}
            print(name, json.dumps(out[name], indent=None)[:1500], flush=True)
    json.dump(out, open(a.output, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("weights"); w.add_argument("--attempts", type=int, nargs="+"); w.add_argument("--output", required=True)
    r = sp.add_parser("rollouts"); r.add_argument("--subjects", nargs="+"); r.add_argument("--worlds-per-cond", type=int, default=32)
    r.add_argument("--samples", type=int, default=1); r.add_argument("--output", required=True)
    g = sp.add_parser("grads"); g.add_argument("--subjects", nargs="+"); g.add_argument("--output", required=True)
    a = ap.parse_args()
    {"weights": cmd_weights, "rollouts": cmd_rollouts, "grads": cmd_grads}[a.cmd](a)


if __name__ == "__main__":
    main()
