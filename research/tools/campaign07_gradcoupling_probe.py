"""Extended-07 P0b: how strongly does the LRN auxiliary loss couple into actor training through the global-norm clip
and the shared trunk?  Read-only on extended-06 artifacts (labels pool, LRN checkpoint); writes one JSON.

Per batch (the trainer's own run_batch / losses, L1 + aux 1.0, clip 1.0 over all parameters):
  n_actor   ||grad of the actor terms (pg + value + entropy + imit)|| over all parameters
  n_aux     ||grad of the aux term|| over all parameters (aux head + trunk/GRU/inp)
  n_total   ||grad of the trainer's total||;  clip coefficient min(1, 1/n_total) vs min(1, 1/n_actor)
  trunk_aux_share  ||aux grad on inp/gru/trunk||^2 / (||actor||^2 + ||aux||^2 on inp/gru/trunk)
  cos_trunk        cosine between the actor and aux gradients on inp/gru/trunk
Modes: 'init' trains a fresh seed-35 LRN (the trainer's init) for --updates updates, recording the statistics before
each step (Adam, lr 1e-3, clip as the trainer); 'final' evaluates --batches batches on a frozen checkpoint (no step).
Data: --labels/b6_B0.pkl (read-only); worlds 6.879e9 + 700,000 + n (dev_smoke; never protocol).
"""
import argparse
import importlib.util
import json
import pathlib
import random
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("pwtrain", ROOT / "research/tools/campaign04_probeworld_train.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from tensegra import campaign06_probeworld as pw6  # noqa: E402

WORLD = 6_879_700_000


def actor_aux_terms(model, items, eps, steps, ep_steps):
    total, parts = M.losses(model, items, eps, steps, ep_steps, dict(M.RUNG_WEIGHTS["L1"], aux=1.0))
    aux = parts["aux"]
    return total - aux, aux, total


def stats(model, actor, aux, total):
    params = list(model.parameters())
    names = [n for n, _ in model.named_parameters()]
    ga = torch.autograd.grad(actor, params, retain_graph=True, allow_unused=True)
    gx = torch.autograd.grad(aux, params, retain_graph=True, allow_unused=True)
    z = lambda g, p: g if g is not None else torch.zeros_like(p)
    ga = [z(g, p) for g, p in zip(ga, params)]
    gx = [z(g, p) for g, p in zip(gx, params)]
    sq = lambda gs: float(sum(g.pow(2).sum() for g in gs))
    shared = [i for i, n in enumerate(names) if n.split(".")[0] in ("inp", "gru", "trunk")]
    a_sh = torch.cat([ga[i].flatten() for i in shared])
    x_sh = torch.cat([gx[i].flatten() for i in shared])
    n_a, n_x = sq(ga) ** .5, sq(gx) ** .5
    n_t = sq([a + b for a, b in zip(ga, gx)]) ** .5
    return {"n_actor": n_a, "n_aux": n_x, "n_total": n_t, "clip_total": min(1.0, 1.0 / (n_t + 1e-6)),
            "clip_actor_only": min(1.0, 1.0 / (n_a + 1e-6)),
            "trunk_aux_share": float(x_sh.pow(2).sum() / (a_sh.pow(2).sum() + x_sh.pow(2).sum())),
            "cos_trunk": float(torch.nn.functional.cosine_similarity(a_sh, x_sh, dim=0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--mode", choices=("init", "final"), required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=35)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    t0 = time.process_time()
    pool = M.load_pool(a.labels, "b6_B0")
    torch.manual_seed(1000 + a.seed)
    model = M.ProbeNet(128, inputs="public", arch="fuse", public_extra=pw6.PUBLIC_EXTRA6, factor_mode="learned")
    if a.mode == "final":
        model.load_state_dict(torch.load(a.ckpt))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_rng, act_rng = random.Random(7_000 + a.seed), random.Random(9_000 + a.seed)
    rows, w = [], 0
    n = a.updates if a.mode == "init" else a.batches
    for upd in range(n):
        items = []
        for _ in range(a.batch):
            _, cfg, s = pool[data_rng.randrange(len(pool))]
            items.append((cfg, s, WORLD + w))
            w += 1
        eps, steps, ep_steps = M.run_batch(model, items, "sample", act_rng)
        M.attach_case_labels(items, ep_steps, steps)
        actor, aux, total = actor_aux_terms(model, items, eps, steps, ep_steps)
        r = stats(model, actor, aux, total)
        r["update"] = upd
        r["aux_mse"] = float(aux)
        rows.append(r)
        if a.mode == "init":
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt.step()
    def summ(rs):
        k = len(rs)
        return {key: sum(r[key] for r in rs) / k for key in rs[0] if key != "update"} | {
            "frac_clipped": sum(r["n_total"] > 1.0 for r in rs) / k,
            "frac_clipped_actor_only": sum(r["n_actor"] > 1.0 for r in rs) / k,
            "mean_clip_ratio_total_over_actor": sum(r["clip_total"] / r["clip_actor_only"] for r in rs) / k}
    out = {"mode": a.mode, "ckpt": a.ckpt, "n": n, "batch": a.batch,
           "summary_all": summ(rows), "summary_last_quarter": summ(rows[3 * n // 4:]), "rows": rows,
           "cpu_s": time.process_time() - t0}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
